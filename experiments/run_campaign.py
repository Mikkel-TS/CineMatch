"""Recorded CineMatch campaign. Execute from repository root; existing runs are immutable.

Examples: .venv/bin/python -B -u experiments/run_campaign.py --stage screen
          .venv/bin/python -B -u experiments/run_campaign.py --config path/to/proposal.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# In a recorded worker this points to its immutable source snapshot.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import torch
from cinematch.data import digest
from cinematch.training import load_checkpoint, checkpoint_parameters
from campaign_common import (SAMPLE, FULL, CAMPAIGN, write_json, data_path, benchmark_identity,
    load_training, validation_profiles, evaluate_score_matrix, comparisons, blend_scores)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def location(root, run_id):
    return root / "artifacts/experiments" / run_id


def reference_model(root, name):
    checkpoint = load_checkpoint(root / ("artifacts/mf16/model.pt" if name == SAMPLE
                             else f"artifacts/experiments/{FULL}/model/model.pt"))
    mean, bias, factors = checkpoint_parameters(checkpoint)
    return {"mean": np.array(mean), "bias": bias, "factors": factors,
            "movie_ids": checkpoint["movie_ids"].numpy(), "item_counts": checkpoint["item_counts"].numpy(),
            "baseline_bias": checkpoint["baseline_bias"].numpy()}, {
            "fold_in_regularization": checkpoint["fold_in_regularization"],
            "bias_regularization": checkpoint["bias_regularization"]}


def predictor(root, run_id, seen=None):
    from campaign_mf import scorer
    seen = set() if seen is None else set(seen)
    if run_id in seen:
        raise ValueError("Cyclic model dependency")
    seen.add(run_id)
    if run_id in (SAMPLE, FULL):
        model, config = reference_model(root, run_id)
        return lambda i,r: scorer(model,i,r,config), model["movie_ids"], model["item_counts"], True
    record = read_json(root / f"experiments/records/{run_id}.json")
    config = record["configuration"]
    model_path = location(root, run_id) / "model"
    kind = config["kind"]
    if kind == "hybrid":
        base, movie_ids, counts, _ = predictor(root, config["base_run"], seen)
        return lambda i,r: blend_scores(base(i,r), counts, i,r,config), movie_ids, counts, False
    if kind == "neighbors":
        from campaign_neighbors import load_neighbors, score_neighbors
        model = load_neighbors(model_path / "model.npz")
        return lambda i,r: score_neighbors(model,i,r,config), model["movie_ids"], model["counts"], False
    if kind == "content":
        from campaign_neighbors import load_content, score_content
        model = load_content(model_path / "model.npz")
        return lambda i,r: score_content(model,i,r,config), model["movie_ids"], model["counts"], False
    with np.load(model_path / "model.npz", allow_pickle=False) as archive:
        model = {k:archive[k] for k in archive.files}
    if kind == "mf":
        # Trainer may select a fold-in penalty; retain its recorded selection.
        chosen = read_json(model_path / "selected_config.json") if (model_path / "selected_config.json").exists() else config
        return lambda i,r: scorer(model,i,r,chosen), model["movie_ids"], model["item_counts"], True
    if kind == "ranking":
        from campaign_ranking import score_ranking
        return lambda i,r: score_ranking(model,i,r,config), model["movie_ids"], model["item_counts"], False
    raise ValueError(f"Unknown model kind {kind}")


def scoring_matrix(score, profiles):
    # Inference receives support only. Target labels never reach the score function.
    return np.asarray([score(i,r) for i,r in zip(profiles["support_items"], profiles["support_ratings"])])


def reference_cache(root):
    return root / f"artifacts/experiments/{CAMPAIGN}/references"


def prepare_references(root):
    path = reference_cache(root)
    if path.exists():
        manifest = read_json(path / "manifest.json")
        for file, expected in manifest["checksums"].items():
            if digest(root / file) != expected:
                raise ValueError(f"Reference input changed: {file}")
        return
    path.mkdir(parents=True)
    profiles = validation_profiles(root)
    metrics, arrays = {}, {"user_ids":profiles["user_ids"]}
    for label, run_id in (("original_sample",SAMPLE),("full_user",FULL)):
        score, ids, counts, rated = predictor(root,run_id)
        metrics[label], arrays[label], top = evaluate_score_matrix(scoring_matrix(score,profiles),profiles,rated)
        pop = "established_popularity" if run_id == SAMPLE else "full_popularity"
        metrics[pop], arrays[pop], _ = evaluate_score_matrix(np.broadcast_to(counts,(len(profiles["user_ids"]),len(ids))),profiles)
    write_json(path / "metrics.json", metrics)
    np.savez_compressed(path / "per_user.npz", **arrays)
    inputs = ["artifacts/mf16/model.pt",f"artifacts/experiments/{FULL}/model/model.pt",
              "artifacts/data/validation.npz","artifacts/data/catalog.csv","src/cinematch/evaluation.py"]
    write_json(path / "manifest.json", {"benchmark_id":benchmark_identity(root),
        "checksums":{p:digest(root/p) for p in inputs},"created_at":now(),"test_evaluated":False})


def get_references(root, scale, parent):
    path = reference_cache(root)
    metrics = read_json(path / "metrics.json")
    with np.load(path / "per_user.npz",allow_pickle=False) as a:
        rows = {k:a[k] for k in a.files}
    refs = {k:(metrics[k],rows[k]) for k in metrics}
    match = "established_popularity" if scale == "sample" else "full_popularity"
    refs["training_matched_popularity"] = refs[match]
    if parent in (SAMPLE,FULL):
        refs["immediate_parent"] = refs["original_sample" if parent == SAMPLE else "full_user"]
    elif parent:
        p = read_json(root / f"experiments/records/{parent}.json")
        if p["benchmark_id"] != benchmark_identity(root) or p["status"] != "complete":
            raise ValueError("Parent incomplete or benchmark differs")
        with np.load(location(root,parent) / "validation_per_user.npz",allow_pickle=False) as a:
            if not np.array_equal(rows["user_ids"], a["user_ids"]):
                raise ValueError("Parent evaluation user order differs")
            refs["immediate_parent"] = (p["metrics"],a["rows"])
    return refs


def diagnostics(root, profiles, scores, tops, counts, rows, references):
    catalog = pd.read_csv(root / "artifacts/data/catalog.csv")
    n = len(profiles["user_ids"])
    pop_top = np.array([__import__("cinematch.evaluation",fromlist=["top_items"]).top_items(counts,i)
                       for i in profiles["support_items"]])
    overlap = np.mean([len(set(a)&set(b))/5 for a,b in zip(tops,pop_top)])
    groups = {}
    positive_supports = (profiles["support_ratings"] >=4).sum(axis=1)
    for name,mask in (("zero_or_one_positive_seed",positive_supports<=1),
                      ("two_or_three_positive_seeds",(positive_supports>=2)&(positive_supports<=3)),
                      ("four_or_five_positive_seeds",positive_supports>=4)):
        eligible = mask & np.isfinite(rows[:,1])
        groups[name] = {"users":int(eligible.sum()),"ndcg_at_5":float(rows[eligible,1].mean()) if eligible.any() else None,
            "popularity_ndcg_at_5":float(references["training_matched_popularity"][1][eligible,1].mean()) if eligible.any() else None}
    examples=[]
    # Fixed user positions selected before inspecting scores, not cherry-picked wins.
    for j in [0,1,2,3,4]:
        if j>=n: break
        lo,hi=profiles["target_offsets"][j:j+2]
        positives=set(profiles["target_items"][lo:hi][profiles["target_ratings"][lo:hi]>=4])
        examples.append({"validation_user_id":int(profiles["user_ids"][j]),
            "onboarding":[{"movie_id":int(catalog.iloc[i].movieId),"title":str(catalog.iloc[i].title),"rating":float(r)}
                for i,r in zip(profiles["support_items"][j],profiles["support_ratings"][j])],
            "recommendations":[{"movie_id":int(catalog.iloc[i].movieId),"title":str(catalog.iloc[i].title),
                "training_ratings":int(counts[i]),"observed_positive":bool(i in positives),"score":float(scores[j,i])} for i in tops[j]]})
    return {"mean_top5_overlap_with_matched_popularity":float(overlap),
            "identical_ordered_lists_to_popularity_fraction":float(np.mean(np.all(tops==pop_top,axis=1))),
            "recommended_training_support_median":float(np.median(counts[tops])),
            "recommended_slots_with_under_100_training_ratings":float(np.mean(counts[tops]<100)),
            "support_groups":groups,"examples":examples,
            "limits":"Exploratory subgroups; missing ratings are unknown, not dislikes."}


def refresh_outputs(root):
    records=[]
    for p in sorted((root / "experiments/records").glob("*.json")):
        r=read_json(p)
        if r.get("campaign_id")==CAMPAIGN:
            records.append(r)
    leaders=[]
    for r in records:
        leaders.append({"run_id":r["run_id"],"family":r["family"],"parent_run":r["parent_run"],
            "status":r["status"],"evidence_stage":r.get("evidence_stage","exploratory_validation"),
            "result_status":"exploratory_validation",
            "training_scale":r["configuration"]["scale"],"record":f"records/{r['run_id']}.json",
            "benchmark_id":r["benchmark_id"],"metrics":r.get("metrics"),"comparisons":r.get("comparisons"),
            "runtime_seconds":r.get("elapsed_seconds"),"decision":r.get("interpretation",{}).get("decision"),
            "error":r.get("error")})
    leaders.sort(key=lambda r:-(r.get("metrics") or {}).get("ndcg_at_5",-1))
    final_file=root/f"experiments/final-results/{CAMPAIGN}.json"
    final_result=read_json(final_file) if final_file.exists() else None
    write_json(root / "experiments/leaderboard.json", {"campaign_id":CAMPAIGN,"updated_at":now(),
        "reference_metrics":read_json(reference_cache(root)/"metrics.json") if (reference_cache(root)/"metrics.json").exists() else {},
        "limits":"Validation search results are exploratory, including full-data replications. Confidence intervals do not adjust for selection across runs.",
        "attempts":leaders,
        "separate_final_holdout":None if final_result is None else {
            "result_file":f"final-results/{CAMPAIGN}.json","status":final_result["status"],
            "benchmark_id":final_result.get("final_benchmark_id"),"candidate":final_result["candidate_run"],
            "evidence_stage":"reserved_final_holdout","metrics":final_result.get("metrics"),
            "comparisons":final_result.get("comparisons"),"do_not_mix_with_validation_rows":True}})
    lines=["# CineMatch systematic experiment journal", "", "This extends the existing experiment records. Original checkpoints, datasets and records are preserved. App adoption is separate.","",
           "Every row links to a pre-execution record, exact executed source snapshot, artifacts and interpretation. Full-data validation is still exploratory; it is not final holdout confirmation.","",
           "| Run | Family | Data | Status | NDCG@5 | Δ original | Δ full-user | Δ popularity | Decision |",
           "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for r in records:
        m=r.get("metrics",{})
        def delta(k):
            v=r.get("comparisons",{}).get(k,{}).get("ndcg_at_5_difference")
            return f"{v:+.4f}" if v is not None else "—"
        score=f"{m['ndcg_at_5']:.4f}" if m else "—"
        lines.append(f"| [{r['run_id']}](records/{r['run_id']}.json) | {r['family']} | {r['configuration']['scale']} | {r['status']} | {score} | {delta('original_sample')} | {delta('full_user')} | {delta('established_popularity')} | {r.get('interpretation',{}).get('decision','pending')} |")
    for r in records:
        lines.extend(["",f"## {r['run_id']}","",r["question"],"",f"Hypothesis: {r['hypothesis']}","",
            f"Change: {r['changed_variable']}","",f"Status: **{r['status']}**. Parent: `{r['parent_run']}`. Reproduce: `{r['reproduction_command']}`."])
        if r.get("interpretation"):
            lines.extend(["",r["interpretation"]["conclusion"],"",r["interpretation"]["limitations"],"",f"Next decision: {r['interpretation']['decision']}."])
        if r.get("error"): lines.extend(["",f"Failure: {r['error']}"])
    if final_result and final_result["status"]=="complete":
        lines.extend(["","## Reserved final holdout — separate benchmark","",
            f"One frozen candidate, `{final_result['candidate_run']}`, was evaluated once against all fixed references. No tuning followed this evaluation. These numbers are not directly comparable to the validation table above.","",
            "| Method | NDCG@5 | Hit rate@5 | Recall@5 | Distinct movies |",
            "| --- | ---: | ---: | ---: | ---: |"])
        for label,m in final_result["metrics"].items():
            lines.append(f"| {label} | {m['ndcg_at_5']:.6f} | {100*m['hit_rate_at_5']:.2f}% | {100*m['recall_at_5']:.2f}% | {m['distinct_recommended_movies']} |")
        delta=final_result["comparisons"]["established_popularity"]
        lines.extend(["",f"Candidate minus popularity NDCG: {delta['ndcg_at_5_difference']:+.6f}; paired95% interval {delta['bootstrap_95_percent']}. App deployment remains unchanged.","",
            f"[Frozen final result](final-results/{CAMPAIGN}.json). The one-shot sentinel and exact frozen evaluation source are in the linked artifact directory."])
    (root/"experiments/JOURNAL.md").write_text("\n".join(lines)+"\n")


def start_record(root, proposal):
    run_id=proposal["run_id"]
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_id):
        raise ValueError("Invalid run ID")
    run=location(root,run_id)
    record=root/f"experiments/records/{run_id}.json"
    if run.exists() or record.exists(): raise ValueError(f"Run already exists: {run_id}")
    run.mkdir(parents=True)
    source=run/"source"
    if proposal.get("replay_source"):
        origin=root/proposal["replay_source"]
        shutil.copytree(origin,source)
    else:
        shutil.copytree(root/"src",source/"src",ignore=shutil.ignore_patterns("__pycache__","*.egg-info"))
        (source/"experiments").mkdir()
        for p in (root/"experiments").glob("*.py"): shutil.copy2(p,source/"experiments"/p.name)
        for name in ("requirements.lock","pyproject.toml"): shutil.copy2(root/name,source/name)
        (source/"environment.lock").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True,stderr=subprocess.DEVNULL))
        (source/"git.diff").write_text(subprocess.check_output(["git","diff","HEAD"],cwd=root,text=True))
    config=proposal["configuration"]
    data=data_path(root,config["scale"])
    manifest=read_json(data/"manifest.json")
    full=read_json(data_path(root,"full")/"manifest.json")
    plan={**proposal,"campaign_id":CAMPAIGN,"status":"planned","created_at":now(),
        "benchmark_id":benchmark_identity(root),"training_data":{"scale":config["scale"],"path":str(data.relative_to(root)),
            "manifest_sha256":digest(data/"manifest.json"),"training_arrays_sha256":manifest["sha256"]["train.npz"],
            "users":manifest["training_users"],"ratings":manifest["training_ratings"]},
        "holdout_contract":{"validation_sha256":manifest["sha256"]["validation.npz"],
            "reserved_fresh_test_sha256":full["sha256"]["fresh_test.npz"],"fresh_test_loaded":False,
            "all_reserved_users_excluded_from_training":True},
        "fixed_conditions":["500 frozen validation users, exact five onboarding ratings and target arrays", "9,021 catalogue IDs and order",
            "Exclude onboarding; positive heldout rating >=4; original top_items and evaluation metric arithmetic", "Original, full-user and both popularity controls retained", "No app deployment"],
        "source_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
        "source_sha256":{str(p.relative_to(source)):digest(p) for p in sorted(source.rglob("*")) if p.is_file()},
        "environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"torch":str(torch.__version__),"pandas":pd.__version__,
            "dependencies":"source/environment.lock","cpu_threads":2},
        "artifact_directory":str(run.relative_to(root)),"model_artifacts":str((run/"model").relative_to(root)),
        "evidence_stage":proposal.get("evidence_stage","exploratory_validation"),
        "reproduction_command":f".venv/bin/python -B -u experiments/run_campaign.py --replay {run_id} --run-id new-unique-id",
        "reference_inputs":read_json(reference_cache(root)/"manifest.json"),
        "adoption_decision":"Not deployed; current app model unchanged"}
    write_json(run/"plan.json",plan)
    write_json(record,plan)
    refresh_outputs(root)
    return plan


def worker(root,run_id):
    torch.set_num_threads(2)
    run=location(root,run_id)
    record_path=root/f"experiments/records/{run_id}.json"
    record=read_json(run/"plan.json")
    config=record["configuration"]
    start=time.monotonic()
    record.update(status="running",started_at=now())
    write_json(record_path,record)
    refresh_outputs(root)
    try:
        if benchmark_identity(root)!=record["benchmark_id"]: raise ValueError("Benchmark changed after planning")
        planned_data=data_path(root,config["scale"])
        if digest(planned_data/"manifest.json")!=record["training_data"]["manifest_sha256"]:
            raise ValueError("Training manifest changed after planning")
        if digest(planned_data/"train.npz")!=record["training_data"]["training_arrays_sha256"]:
            raise ValueError("Training arrays changed after planning")
        prepare_references(root)
        for name,expected in record["source_sha256"].items():
            if digest(run/"source"/name)!=expected: raise ValueError(f"Executed snapshot changed: {name}")
        profiles=validation_profiles(root)
        output=run/"model"
        kind=config["kind"]
        if kind=="mf" and not config.get("train",False):
            output.mkdir()
            model,defaults=reference_model(root,config["source_run"])
            np.savez_compressed(output/"model.npz",**model)
            write_json(output/"selected_config.json",{**defaults,**config})
        elif kind=="hybrid":
            output.mkdir()
            write_json(output/"model.json",{"kind":"hybrid","base_run":config["base_run"],"configuration":config})
        elif config.get("reuse_model_run"):
            previous=read_json(root/f"experiments/records/{config['reuse_model_run']}.json")
            if previous["status"]!="complete" or previous["training_data"]!=record["training_data"]:
                raise ValueError("Reused model must be complete and use identical training data")
            output.mkdir()
            shutil.copy2(location(root,config["reuse_model_run"])/"model/model.npz",output/"model.npz")
            write_json(output/"derivation.json",{"source_run":config["reuse_model_run"],"configuration":config,
                "source_model_sha256":digest(output/"model.npz")})
        else:
            training=load_training(root,config["scale"])
            if kind=="mf":
                from campaign_mf import train_mf
                model=train_mf(training,config,output,profiles)
            elif kind=="neighbors":
                from campaign_neighbors import train_neighbors
                model=train_neighbors(training,config,output)
            elif kind=="content":
                from campaign_neighbors import train_content
                model=train_content(pd.read_csv(root/"artifacts/data/catalog.csv"),training,config,output)
            elif kind=="ranking":
                from campaign_ranking import train_ranking
                model=train_ranking(training,config,output,profiles)
            else: raise ValueError(f"Unknown kind: {kind}")
            del training,model
        score,ids,counts,rated=predictor(root,run_id)
        expected=pd.read_csv(root/"artifacts/data/catalog.csv").movieId.to_numpy()
        if not np.array_equal(ids,expected): raise ValueError("Scorer catalogue IDs/order differ")
        scoring_start=time.monotonic()
        scores=scoring_matrix(score,profiles)
        if scores.shape!=(len(profiles["user_ids"]),len(ids)):
            raise ValueError("Scorer returned a different candidate catalogue width")
        inference_seconds=time.monotonic()-scoring_start
        metrics,rows,tops=evaluate_score_matrix(scores,profiles,rated)
        refs=get_references(root,config["scale"],record["parent_run"])
        compared=comparisons(metrics,rows,refs)
        np.savez_compressed(run/"validation_per_user.npz",user_ids=profiles["user_ids"],rows=rows,top_items=tops)
        details=diagnostics(root,profiles,scores,tops,counts,rows,refs)
        write_json(run/"diagnostics.json",details)
        write_json(run/"comparison.json",compared)
        diff=compared["established_popularity"]["ndcg_at_5_difference"]
        interval=compared["established_popularity"]["bootstrap_95_percent"]
        evidence="interval above zero" if interval[0]>0 else "interval below zero" if interval[1]<0 else "interval includes zero"
        decision="investigate and confirm" if diff>0 else "retain as diagnostic; not an improvement over popularity"
        interpretation={"conclusion":f"Observed NDCG@5 {metrics['ndcg_at_5']:.6f}; difference from established popularity {diff:+.6f} ({evidence}).",
            "limitations":"Exploratory validation search. User-bootstrap uncertainty excludes training-seed variability and best-of-many selection. Coverage is not quality. Existing retrospective onboarding/exposure limitations remain.",
            "decision":decision,"personalization_check":details["mean_top5_overlap_with_matched_popularity"]}
        record.update(status="complete",finished_at=now(),elapsed_seconds=time.monotonic()-start,
            metrics=metrics,comparisons=compared,interpretation=interpretation,
            resource_use={"peak_process_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "rss_units":"bytes on macOS; process lifetime high-water mark","validation_scoring_seconds":inference_seconds,
                "milliseconds_per_user_cpu":inference_seconds/len(profiles["user_ids"])*1000,
                "model_artifact_bytes":sum(p.stat().st_size for p in output.rglob("*") if p.is_file())},
            artifact_checksums={str(p.relative_to(run)):digest(p) for p in sorted(run.rglob("*")) if p.is_file() and "source" not in p.parts and p.name not in ("run.log","result.json","plan.json")})
        print(f"COMPLETE {run_id}: NDCG={metrics['ndcg_at_5']:.6f}, delta popularity={diff:+.6f}",flush=True)
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",finished_at=now(),
            elapsed_seconds=time.monotonic()-start,error=f"{type(error).__name__}: {error}",traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        write_json(run/"result.json",record)
        write_json(record_path,record)
        refresh_outputs(root)
    return 0 if record["status"]=="complete" else 1


def proposal(slug,family,config,question,change,parent=SAMPLE,hypothesis=None,stage="exploratory_validation"):
    return {"run_id":"cm-20261119-"+slug,"family":family,"parent_run":parent,"configuration":{"scale":"sample","seed":42,**config},
        "question":question,"hypothesis":hypothesis or "This change may improve recovery of held-out likes; no improvement is assumed.",
        "changed_variable":change,"evidence_stage":stage}


def screening_plan():
    p=[]
    base={"kind":"mf","source_run":SAMPLE}
    for label,config,q in [
        ("fit-lambda01",{"fold_in_regularization":.1},"Does weaker new-user regularisation retain useful taste signal?"),
        ("fit-lambda001",{"fold_in_regularization":.01},"Does very weak new-user regularisation help or overfit five ratings?"),
        ("fit-no-intercept",{"fold_in_regularization":.1,"fit_user_bias":False},"Does removing the new-user intercept improve sparse fitting?"),
        ("fit-centered",{"fold_in_regularization":.1,"user_center":True},"Does centring onboarding ratings reduce rating-scale differences?")]:
        p.append(proposal(label,"profile_fitting",{**base,**config},q,json.dumps(config)))
    for label,config in [("bias-zero",{"movie_bias_scale":0}), ("bias-half",{"movie_bias_scale":.5}),
                         ("bias-shrink100",{"bias_shrinkage_tau":100}), ("bias-shrink1000",{"bias_shrinkage_tau":1000})]:
        p.append(proposal(label,"movie_bias",{**base,**config},"Does reducing unsupported movie-prior influence improve ranking?",json.dumps(config)))
    for label,config in [("blend25",{"alpha":.25}),("blend50",{"alpha":.5}),
                         ("blend-confidence",{"alpha":.5,"confidence":True}), ("rerank100",{"candidate_count":100})]:
        p.append(proposal(label,"popularity_hybrid",{"kind":"hybrid","base_run":SAMPLE,**config},
            "Can popularity stabilise sparse personalisation without erasing it?",json.dumps(config)))
    p.append(proposal("full-blend25","popularity_hybrid",{"kind":"hybrid","scale":"full","base_run":FULL,"alpha":.25},
        "Does the same blend improve the full-user model?","25% standardized MF + 75% standardized log-count",FULL))
    common={"kind":"mf","train":True,"epochs":12,"patience":3,"fold_in_grid":[1.],"threads":2}
    for label,config,q in [("mf8",{"factors":8},"Does a smaller latent profile generalise better from five ratings?"),
                         ("mf32",{"factors":32},"Does a larger latent space improve new-user ranking?"),
                         ("mf-biasreg10",{"factors":16,"bias_count_regularization":10.},"Does evidence-aware bias regularisation protect low-support movies?"),
                         ("mf-huber",{"factors":16,"loss":"huber","huber_delta":1.},"Does a robust rating loss improve ranking?")]:
        p.append(proposal(label,"mf_training",{**common,**config},q,json.dumps(config)+"; fixed fold-in lambda1, maximum12 epochs"))
    common={"kind":"neighbors","feedback":"positive","top_k":100,"shrink":10,"similarity":"cosine","aggregation":"positive"}
    for label,config in [("knn-cos100",{}),("knn-cos300",{"top_k":300,"shrink":50}),
                         ("knn-conditional",{"similarity":"conditional"}),("knn-observed",{"feedback":"observed"}),
                         ("knn-signed",{"aggregation":"signed"})]:
        p.append(proposal(label,"item_neighbors",{**common,**config},"Can item co-occurrence transfer five onboarding ratings into better rankings?",json.dumps({**common,**config})))
    common={"kind":"ranking","factors":16,"epochs":8,"patience":3,"pairs_per_epoch":300000,"use_item_bias":True,"aggregation":"positive","threads":2}
    for label,config in [("bpr16",{}),("bpr16-no-bias",{"use_item_bias":False}),("bpr32",{"factors":32})]:
        p.append(proposal(label,"pairwise_ranking",{**common,**config},"Can a pairwise item-context objective improve recommendations for unseen users?",json.dumps({**common,**config})))
    p.append(proposal("genre","metadata",{"kind":"content","aggregation":"signed"},"Do verified MovieLens genres help from sparse ratings?","Unit-normalized genre vectors and signed onboarding aggregation"))
    p.append(proposal("genre-blend25","metadata",{"kind":"hybrid","base_run":"cm-20261119-genre","alpha":.25},
        "Does genre taste improve a popularity prior?","25% standardized genre score + 75% standardized log-count",parent="cm-20261119-genre"))
    return p


def execute(root,proposals):
    prepare_references(root)
    outcomes=[]
    for p in proposals:
        record=root/f"experiments/records/{p['run_id']}.json"
        if record.exists():
            print(f"PRESERVED {p['run_id']}: {read_json(record)['status']}",flush=True)
            continue
        plan=start_record(root,p)
        run=location(root,p["run_id"])
        env={**os.environ,"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2","VECLIB_MAXIMUM_THREADS":"2","MKL_NUM_THREADS":"2","PYTHONDONTWRITEBYTECODE":"1"}
        command=[sys.executable,"-B","-u",str(run/"source/experiments/run_campaign.py"),"--worker",p["run_id"],"--root",str(root)]
        print(f"START {p['run_id']} ({p['family']}, {p['configuration']['scale']})",flush=True)
        with (run/"run.log").open("x") as log:
            process=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                for line in process.stdout:
                    log.write(line);log.flush();print(line,end="",flush=True)
                code=process.wait()
            except BaseException as error:
                process.terminate();process.wait()
                latest=read_json(record)
                latest.update(status="interrupted",finished_at=now(),error=f"Parent interrupted: {type(error).__name__}: {error}")
                write_json(record,latest);write_json(run/"result.json",latest);refresh_outputs(root)
                raise
        latest=read_json(record)
        if latest["status"] in ("planned","running"):
            latest.update(status="failed",finished_at=now(),error=f"Worker exited {code} before final status; inspect run.log")
            write_json(record,latest);write_json(run/"result.json",latest);refresh_outputs(root)
        outcomes.append({"run_id":p["run_id"],"exit_code":code})
    return outcomes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["screen"])
    parser.add_argument("--config",type=Path)
    parser.add_argument("--worker")
    parser.add_argument("--replay")
    parser.add_argument("--run-id")
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--refresh",action="store_true")
    args=parser.parse_args()
    root=args.root.resolve()
    if args.worker: return worker(root,args.worker)
    if args.refresh: refresh_outputs(root);return 0
    if args.replay:
        if not args.run_id: parser.error("--replay requires a new --run-id")
        old=read_json(location(root,args.replay)/"plan.json")
        p={k:old[k] for k in ("family","parent_run","configuration","question","hypothesis","changed_variable","evidence_stage")}
        p.update(run_id=args.run_id,retry_of=args.replay,replay_source=str((location(root,args.replay)/"source").relative_to(root)))
        proposals=[p]
    elif args.config:
        value=read_json(args.config);proposals=value if isinstance(value,list) else [value]
    elif args.stage=="screen":
        proposals=screening_plan()
        write_json(root/f"experiments/campaigns/{CAMPAIGN}.json",{"campaign_id":CAMPAIGN,"created_at":now(),
            "phase":"screening","planned_experiments":proposals,"fresh_holdout":"closed until candidate and evaluation are frozen",
            "breadth":"Seven families; sample screens then selected full-data confirmations", "concurrency":"one experiment worker, two numerical threads",
            "excluded_families":[],"existing_evidence":"Two recorded MF runs; prior chat diagnostics were unrecorded and are rerun here; automated tests are not model experiments."})
    else: parser.error("Choose --stage, --config, --replay or --refresh")
    execute(root,proposals)
    return 0


if __name__=="__main__":
    raise SystemExit(main())

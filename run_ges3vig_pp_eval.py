#!/usr/bin/env python3
"""
Ges3ViG post-paper (pp) evaluation suite.

Runs the full battery of inference-only evaluations for the ORIGINAL
(straight-arm / no-elbow) Ges3ViG model using a single fixed checkpoint, and
collects the IoU metrics for each experiment.

Experiments
-----------
  1. noelbow_random        : NoElbowImputeRefer val, RANDOM seed (the "original" style).
  2. noelbow_5seeds        : NoElbowImputeRefer val, 5 FIXED seeds -> mean +/- std of IoU.
  3. imputerefer           : ImputeRefer (bent-elbow, extra samples) val with the SAME
                             checkpoint (no retraining) -> what happens on the bigger set.
  4. noise_sweep           : Vary shoulder-rotation noise degrees for the straight-arm
                             config, for BOTH noelbowimputerefer AND imputerefer.
  5. realhumans            : ImputeReferRealHumans val (imputed real-human avatars).
  6. nohuman               : NoHumanImputeRefer val (no-human sensitivity analysis).
  7. pc_noise_sweep        : additive Gaussian noise over the WHOLE point cloud,
                             sweeping sigma, for noelbow/imputerefer/nohuman _noise configs.
  8. partial_sweep         : partial / incomplete capture (frustum fov sweep +
                             random dropout keep-ratio sweep) for the same configs.

Each run's full stdout is saved to  output/ges3vig-pp/<expt>.log  and a combined
summary (parsed IoU per experiment, plus mean/std for the 5-seed run) is written to
output/ges3vig-pp/SUMMARY.txt and SUMMARY.json.

Run it INSIDE the ges3vig-gpu conda env, from the original_Ges3ViG directory:

    conda activate ges3vig-gpu
    python run_ges3vig_pp_eval.py                       # full eval (slow, hours)
    python run_ges3vig_pp_eval.py --quick               # fast smoke (limit_test_batches=25)
    python run_ges3vig_pp_eval.py --gpu 1
    python run_ges3vig_pp_eval.py --only noelbow_5seeds noise_sweep
    python run_ges3vig_pp_eval.py --seeds 0 1 2 3 4 --degrees 0 3 6 9
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "output", "ges3vig-pp")
DEFAULT_CKPT = "/nas_hcfc/HRI_general/common/checkpoints/Ges3ViG_ImputeRefer.ckpt"

# Proxemic-zone (distance) evaluator. When --distance is passed we swap the data
# evaluator for this one; it prints the normal unique/multiple/overall IoU table
# AND an extra per-zone (intimate/personal/social/public) table that we parse.
# Zones = Hall's proxemics in metres: intimate<0.46, personal<1.22, social<3.70,
# public>=3.70 (human shoulder-centre -> target object centre).
DIST_EVAL_TARGET = "ges3vig.evaluation.scanrefer_distance_evaluator.ScanReferDistanceEvaluator"
DIST_ZONES = ["intimate", "personal", "social", "public"]

# Combined evaluator: standard + proxemic-distance + elbow-angle, and it dumps a
# per-query JSON (one row per sample: distance, arm_angle, eval_type, raw IoU) to
# output/ges3vig-pp/perquery/<run>.json so any graph/analysis can be redone offline.
FULL_EVAL_TARGET = "ges3vig.evaluation.scanrefer_full_evaluator.ScanReferFullEvaluator"
PERQUERY_DIR = os.path.join(RESULTS_DIR, "perquery")
ANGLE_BINS = ["20-50°", "50-80°", "80-110°", "110-150°"]

# All experiment keys (for --only validation / help)
ALL_EXPERIMENTS = [
    "noelbow_random",
    "noelbow_5seeds",
    "imputerefer",
    "noise_sweep",
    "realhumans",
    "nohuman",
    "pc_noise_sweep",     # Q7: Gaussian noise over the whole point cloud, sigma sweep
    "partial_sweep",      # Q8: partial / incomplete capture
]


# --------------------------------------------------------------------------- #
# Running test.py + capturing / parsing output
# --------------------------------------------------------------------------- #
def run_test(name, overrides, args, log_path):
    """Run test.py with the given hydra overrides, tee output to log_path, return the text."""
    cmd = [
        sys.executable, "test.py",
        f"ckpt_path={args.ckpt}",
        f"experiment_name=ges3vigpp_{name}",
        f"trainer.limit_test_batches={args.limit_test_batches}",
    ] + overrides

    # Optionally swap in a breakdown evaluator (keeps verbose + metric_name from
    # the data config; only the _target_ class changes). --full also dumps a
    # per-query JSON next to the logs.
    if getattr(args, "full", False):
        os.makedirs(PERQUERY_DIR, exist_ok=True)
        cmd.append(f"data.evaluator._target_={FULL_EVAL_TARGET}")
        cmd.append(f"+data.evaluator.results_save_path={os.path.join(PERQUERY_DIR, name + '.json')}")
    elif getattr(args, "distance", False):
        cmd.append(f"data.evaluator._target_={DIST_EVAL_TARGET}")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    header = (
        f"{'#' * 78}\n"
        f"# EXPERIMENT: {name}\n"
        f"# time: {datetime.now().isoformat(timespec='seconds')}\n"
        f"# cmd : {' '.join(cmd)}\n"
        f"# gpu : {args.gpu}\n"
        f"{'#' * 78}\n"
    )
    print(header)

    captured = [header]
    with open(log_path, "w") as logf:
        logf.write(header)
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            # collapse carriage-return progress bars for the saved log
            clean = line.rstrip("\n")
            logf.write(clean + "\n")
            logf.flush()
            captured.append(clean + "\n")
            # echo a trimmed view to console (skip tqdm spam)
            if "\r" not in line and "it/s]" not in line:
                sys.stdout.write(clean + "\n")
        proc.wait()
    text = "".join(captured)
    if proc.returncode != 0:
        print(f"  [WARN] {name} exited with code {proc.returncode} (see {log_path})")
    return text, proc.returncode


_NUM = r"([0-9]+\.[0-9]+|nan)"


def _row(text, label):
    """Parse a results-table row that starts with `label` (e.g. '0.25')."""
    m = re.search(rf"^{re.escape(label)}\s+{_NUM}\s+{_NUM}\s+{_NUM}", text, re.MULTILINE)
    if not m:
        return None
    def f(x):
        return float("nan") if x == "nan" else float(x)
    return {"unique": f(m.group(1)), "multiple": f(m.group(2)), "overall": f(m.group(3))}


def parse_iou(text):
    """Return {'iou25': {...}, 'iou50': {...}} parsed from the ScanRefer results table,
    or None if not found."""
    r25 = _row(text, "0.25")
    r50 = _row(text, "0.50")
    if r25 is None and r50 is None:
        return None
    return {"iou25": r25, "iou50": r50}


def fmt_iou(iou):
    if iou is None:
        return "  (no IoU table found - check the .log)"
    def line(tag, d):
        if d is None:
            return f"    IoU@{tag}: <missing>"
        return (f"    IoU@{tag}: overall={d['overall']:.1f}  "
                f"unique={d['unique']:.1f}  multiple={d['multiple']:.1f}")
    return line("0.25", iou["iou25"]) + "\n" + line("0.50", iou["iou50"])


def parse_distance(text):
    """Parse the per-zone table printed by ScanReferDistanceEvaluator:

        distance    count     IoU@0.25      IoU@0.50
        intimate    0         nan           nan
        personal    123       45.6          30.1
        ...

    Returns {zone: {"count": int, "iou25": float|nan, "iou50": float|nan}} or
    None if the table is absent (e.g. the run used the plain evaluator)."""
    out = {}
    for zone in DIST_ZONES:
        m = re.search(rf"^{zone}\s+([0-9]+)\s+{_NUM}\s+{_NUM}", text, re.MULTILINE)
        if not m:
            continue
        def f(x):
            return float("nan") if x == "nan" else float(x)
        out[zone] = {"count": int(m.group(1)), "iou25": f(m.group(2)), "iou50": f(m.group(3))}
    return out or None


def parse_angle(text):
    """Parse the elbow-angle table printed by ScanReferFullEvaluator.
    Returns {bin: {"count","iou25","iou50"}} or None."""
    out = {}
    for b in ANGLE_BINS:
        m = re.search(rf"^{re.escape(b)}\s+([0-9]+)\s+{_NUM}\s+{_NUM}", text, re.MULTILINE)
        if not m:
            continue
        def f(x):
            return float("nan") if x == "nan" else float(x)
        out[b] = {"count": int(m.group(1)), "iou25": f(m.group(2)), "iou50": f(m.group(3))}
    return out or None


def _dist(text, args):
    """Spread-helper: attach distance (and angle, under --full) breakdowns to a
    summary entry. Returns {} when no breakdown flag is set."""
    out = {}
    if getattr(args, "full", False) or getattr(args, "distance", False):
        d = parse_distance(text)
        if d:
            out["distance"] = d
    if getattr(args, "full", False):
        a = parse_angle(text)
        if a:
            out["angle"] = a
    return out


def _fmt_breakdown(table, order, header):
    if not table:
        return None
    lines = [f"    {header} (count | IoU@0.25 | IoU@0.50):"]
    for k in order:
        d = table.get(k)
        if d is None:
            continue
        i25 = "  nan" if d["iou25"] != d["iou25"] else f"{d['iou25']:5.1f}"
        i50 = "  nan" if d["iou50"] != d["iou50"] else f"{d['iou50']:5.1f}"
        lines.append(f"      {k:<10} n={d['count']:<6} {i25}   {i50}")
    return "\n".join(lines)


def fmt_distance(dist):
    return _fmt_breakdown(dist, DIST_ZONES, "by proxemic zone")


def fmt_angle(angle):
    return _fmt_breakdown(angle, ANGLE_BINS, "by elbow angle")


def mean_std(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.pstdev(vals)


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def expt_noelbow_random(args, summary):
    name = "noelbow_random"
    text, _ = run_test(
        name,
        ["data=noelbowimputerefer", "data.inference.split=val", "test_seed=random"],
        args, os.path.join(RESULTS_DIR, f"{name}.log"),
    )
    summary[name] = {"iou": parse_iou(text), "note": "NoElbowImputeRefer val, random seed",
                     **_dist(text, args)}


def expt_noelbow_5seeds(args, summary):
    per_seed = {}
    o25, o50, u25, u50, m25, m50 = ([] for _ in range(6))
    for seed in args.seeds:
        name = f"noelbow_seed{seed}"
        text, _ = run_test(
            name,
            ["data=noelbowimputerefer", "data.inference.split=val", f"test_seed={seed}"],
            args, os.path.join(RESULTS_DIR, f"{name}.log"),
        )
        iou = parse_iou(text)
        per_seed[seed] = {"iou": iou, **_dist(text, args)} if getattr(args, "distance", False) else iou
        if iou and iou["iou25"]:
            o25.append(iou["iou25"]["overall"]); u25.append(iou["iou25"]["unique"]); m25.append(iou["iou25"]["multiple"])
        if iou and iou["iou50"]:
            o50.append(iou["iou50"]["overall"]); u50.append(iou["iou50"]["unique"]); m50.append(iou["iou50"]["multiple"])

    agg = {
        "iou25": {k: mean_std(v) for k, v in (("overall", o25), ("unique", u25), ("multiple", m25))},
        "iou50": {k: mean_std(v) for k, v in (("overall", o50), ("unique", u50), ("multiple", m50))},
    }
    summary["noelbow_5seeds"] = {
        "seeds": list(args.seeds),
        "per_seed": per_seed,
        "aggregate_mean_std": agg,
        "note": "NoElbowImputeRefer val, fixed seeds -> mean +/- std",
    }


def expt_imputerefer(args, summary):
    name = "imputerefer"
    text, _ = run_test(
        name,
        ["data=imputerefer", "data.inference.split=val", f"test_seed={args.seeds[0]}"],
        args, os.path.join(RESULTS_DIR, f"{name}.log"),
    )
    summary[name] = {"iou": parse_iou(text),
                     "note": "ImputeRefer (bent-elbow, extra samples) val, SAME checkpoint, no retraining",
                     **_dist(text, args)}


def _select_datasets(args, datasets):
    """Filter a sweep's dataset tuple by --datasets (exact match). None => keep all."""
    if not args.datasets:
        return list(datasets)
    return [d for d in datasets if d in args.datasets]


def expt_noise_sweep(args, summary):
    entry = summary.setdefault("noise_sweep",
                               {"runs": {}, "note": "Shoulder-rotation noise sweep (straight-arm) for noelbow + imputerefer"})
    out = entry["runs"]
    for dataset in _select_datasets(args, ("noelbowimputerefer", "imputerefer")):
        for deg in args.degrees:
            perc = round(deg / 1.8, 4)  # config percentage; DEG = perc * 1.8
            name = f"noise_{dataset}_deg{deg}"
            text, _ = run_test(
                name,
                [f"data={dataset}", "data.inference.split=val",
                 f"test_seed={args.seeds[0]}",
                 f"data.human_perturbation_percentage={perc}"],
                args, os.path.join(RESULTS_DIR, f"{name}.log"),
            )
            out[name] = {"dataset": dataset, "degrees": deg,
                         "human_perturbation_percentage": perc, "iou": parse_iou(text),
                         **_dist(text, args)}


def expt_realhumans(args, summary):
    name = "realhumans"
    text, _ = run_test(
        name,
        ["data=imputerefer_realhumans", "data.inference.split=val", f"test_seed={args.seeds[0]}"],
        args, os.path.join(RESULTS_DIR, f"{name}.log"),
    )
    summary[name] = {"iou": parse_iou(text), "note": "ImputeReferRealHumans val (imputed real-human avatars)",
                     **_dist(text, args)}


def expt_nohuman(args, summary):
    name = "nohuman"
    text, _ = run_test(
        name,
        ["data=nohumanimputerefer", "data.inference.split=val", f"test_seed={args.seeds[0]}"],
        args, os.path.join(RESULTS_DIR, f"{name}.log"),
    )
    summary[name] = {"iou": parse_iou(text), "note": "NoHumanImputeRefer val (no-human sensitivity)",
                     **_dist(text, args)}


def expt_pc_noise_sweep(args, summary):
    """Q7: additive Gaussian noise over the ENTIRE point cloud, sweeping sigma (meters)."""
    entry = summary.setdefault("pc_noise_sweep",
                               {"runs": {}, "note": "Q7: whole-cloud Gaussian noise sweep (sigma in meters)"})
    out = entry["runs"]
    for dataset in _select_datasets(args, ("noelbowimputerefer_noise", "imputerefer_noise", "nohumanimputerefer_noise")):
        for sigma in args.noise_sigmas:
            name = f"pcnoise_{dataset}_sigma{sigma}"
            text, _ = run_test(
                name,
                [f"data={dataset}", "data.inference.split=val", f"test_seed={args.seeds[0]}",
                 "data.eval_corruption.enabled=true",
                 "data.eval_corruption.noise.enabled=true",
                 f"data.eval_corruption.noise.sigma={sigma}",
                 "data.eval_corruption.partial.enabled=false"],
                args, os.path.join(RESULTS_DIR, f"{name}.log"),
            )
            out[name] = {"dataset": dataset, "noise_sigma": sigma, "iou": parse_iou(text),
                         **_dist(text, args)}


def expt_partial_sweep(args, summary):
    """Q8: partial / incomplete capture. frustum (fov sweep) + dropout (keep_ratio sweep)."""
    entry = summary.setdefault("partial_sweep",
                               {"runs": {}, "note": "Q8: partial capture - frustum (fov) + dropout (keep_ratio)"})
    out = entry["runs"]
    for dataset in _select_datasets(args, ("noelbowimputerefer_noise", "imputerefer_noise", "nohumanimputerefer_noise")):
        # frustum capture: only points inside a random camera view survive
        for fov in args.frustum_fovs:
            name = f"partial_{dataset}_frustum_fov{fov}"
            text, _ = run_test(
                name,
                [f"data={dataset}", "data.inference.split=val", f"test_seed={args.seeds[0]}",
                 "data.eval_corruption.enabled=true",
                 "data.eval_corruption.noise.enabled=false",
                 "data.eval_corruption.partial.enabled=true",
                 "data.eval_corruption.partial.mode=frustum",
                 f"data.eval_corruption.partial.fov_deg={fov}"],
                args, os.path.join(RESULTS_DIR, f"{name}.log"),
            )
            out[name] = {"dataset": dataset, "mode": "frustum", "fov_deg": fov, "iou": parse_iou(text),
                         **_dist(text, args)}
        # random dropout: fraction of points kept
        for keep in args.keep_ratios:
            name = f"partial_{dataset}_dropout_keep{keep}"
            text, _ = run_test(
                name,
                [f"data={dataset}", "data.inference.split=val", f"test_seed={args.seeds[0]}",
                 "data.eval_corruption.enabled=true",
                 "data.eval_corruption.noise.enabled=false",
                 "data.eval_corruption.partial.enabled=true",
                 "data.eval_corruption.partial.mode=dropout",
                 f"data.eval_corruption.partial.keep_ratio={keep}"],
                args, os.path.join(RESULTS_DIR, f"{name}.log"),
            )
            out[name] = {"dataset": dataset, "mode": "dropout", "keep_ratio": keep, "iou": parse_iou(text),
                         **_dist(text, args)}


EXPERIMENT_FUNCS = {
    "noelbow_random": expt_noelbow_random,
    "noelbow_5seeds": expt_noelbow_5seeds,
    "imputerefer": expt_imputerefer,
    "noise_sweep": expt_noise_sweep,
    "realhumans": expt_realhumans,
    "nohuman": expt_nohuman,
    "pc_noise_sweep": expt_pc_noise_sweep,
    "partial_sweep": expt_partial_sweep,
}


# --------------------------------------------------------------------------- #
# Summary writing
# --------------------------------------------------------------------------- #
def write_summary(summary, args):
    json_path = os.path.join(RESULTS_DIR, "SUMMARY.json")
    txt_path = os.path.join(RESULTS_DIR, "SUMMARY.txt")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append("=" * 78)
    lines.append("Ges3ViG post-paper evaluation summary")
    lines.append(f"checkpoint : {args.ckpt}")
    lines.append(f"generated  : {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"limit_test_batches: {args.limit_test_batches}  (1.0 = full set)")
    lines.append("=" * 78)

    for key in ALL_EXPERIMENTS:
        if key not in summary:
            continue
        entry = summary[key]
        lines.append("")
        lines.append(f"### {key}  -  {entry.get('note', '')}")

        # helper: append distance + angle breakdown tables if the entry carries them
        def _emit_dist(entry_or_run):
            for block in (fmt_distance(entry_or_run.get("distance")),
                          fmt_angle(entry_or_run.get("angle"))):
                if block:
                    lines.append(block)

        if key == "noelbow_5seeds":
            agg = entry["aggregate_mean_std"]
            lines.append(f"    seeds: {entry['seeds']}")
            for tag in ("iou25", "iou50"):
                lab = "0.25" if tag == "iou25" else "0.50"
                o = agg[tag]["overall"]; u = agg[tag]["unique"]; m = agg[tag]["multiple"]
                lines.append(f"    IoU@{lab}:  overall={o[0]:.1f}+/-{o[1]:.1f}   "
                             f"unique={u[0]:.1f}+/-{u[1]:.1f}   multiple={m[0]:.1f}+/-{m[1]:.1f}")
        elif key == "noise_sweep":
            for rname, r in entry["runs"].items():
                lines.append(f"  [{r['dataset']}  {r['degrees']} deg "
                             f"(perc={r['human_perturbation_percentage']})]")
                lines.append(fmt_iou(r["iou"]))
                _emit_dist(r)
        elif key == "pc_noise_sweep":
            for rname, r in entry["runs"].items():
                lines.append(f"  [{r['dataset']}  sigma={r['noise_sigma']} m]")
                lines.append(fmt_iou(r["iou"]))
                _emit_dist(r)
        elif key == "partial_sweep":
            for rname, r in entry["runs"].items():
                detail = (f"fov={r['fov_deg']}" if r["mode"] == "frustum"
                          else f"keep_ratio={r['keep_ratio']}")
                lines.append(f"  [{r['dataset']}  {r['mode']}  {detail}]")
                lines.append(fmt_iou(r["iou"]))
                _emit_dist(r)
        else:
            lines.append(fmt_iou(entry.get("iou")))
            _emit_dist(entry)

    lines.append("")
    lines.append("=" * 78)
    report = "\n".join(lines)
    with open(txt_path, "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\n[written] {txt_path}\n[written] {json_path}")


# --------------------------------------------------------------------------- #
def main():
    global RESULTS_DIR, PERQUERY_DIR
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="Checkpoint to evaluate")
    p.add_argument("--gpu", default="1", help="CUDA_VISIBLE_DEVICES id (default 1)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="Fixed seeds for the 5-seed run (also seeds[0] for single-seed expts)")
    p.add_argument("--degrees", type=int, nargs="+", default=[0, 3, 6, 9, 12, 18, 24, 30, 36],
                   help="Shoulder-rotation noise degrees to sweep (default 0,3,6,9)")
    p.add_argument("--noise-sigmas", dest="noise_sigmas", type=float, nargs="+",
                   default=[0.0, 0.01, 0.02, 0.05, 0.1],
                   help="Q7 whole-cloud Gaussian noise sigma (meters) sweep")
    p.add_argument("--frustum-fovs", dest="frustum_fovs", type=float, nargs="+",
                   default=[120, 90, 60],
                   help="Q8 frustum-capture field-of-view (deg) sweep")
    p.add_argument("--keep-ratios", dest="keep_ratios", type=float, nargs="+",
                   default=[0.75, 0.5, 0.25],
                   help="Q8 random-dropout keep-ratio sweep")
    p.add_argument("--limit-test-batches", dest="limit_test_batches", default="1.0",
                   help="Pass-through to trainer.limit_test_batches (1.0=full, int=#batches)")
    p.add_argument("--quick", action="store_true",
                   help="Smoke mode: limit_test_batches=25 (fast subset validation)")
    p.add_argument("--distance", action="store_true",
                   help="Use the proxemic-zone distance evaluator: also report IoU broken "
                        "down by human<->target distance zone (intimate/personal/social/public). "
                        "Composes with any --only/--datasets selection.")
    p.add_argument("--full", action="store_true",
                   help="Use the combined evaluator: standard + proxemic-distance + elbow-angle "
                        "breakdowns in one log, plus a per-query JSON dump under "
                        "output/ges3vig-pp/perquery/<run>.json for offline analysis. "
                        "Overrides --distance.")
    p.add_argument("--only", nargs="+", choices=ALL_EXPERIMENTS, default=ALL_EXPERIMENTS,
                   help="Subset of experiments to run")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Restrict the sweep experiments (noise_sweep/pc_noise_sweep/partial_sweep) "
                        "to these exact dataset names, e.g. 'imputerefer imputerefer_noise'. "
                        "Default: all datasets in each sweep.")
    p.add_argument("--fresh", action="store_true",
                   help="Start SUMMARY from scratch. Default: merge into the existing SUMMARY.json "
                        "so reruns only overwrite the experiments/runs you actually re-ran.")
    p.add_argument("--results-dir", dest="results_dir", default=RESULTS_DIR,
                   help="Output directory for logs/SUMMARY/perquery (default output/ges3vig-pp). "
                        "Use e.g. output/ges3vig-pp_v2 to keep a run separate.")
    args = p.parse_args()

    # Redirect all outputs (logs, SUMMARY, per-query JSON) to the chosen dir.
    RESULTS_DIR = os.path.abspath(args.results_dir)
    PERQUERY_DIR = os.path.join(RESULTS_DIR, "perquery")

    if args.quick:
        args.limit_test_batches = "25"

    if not os.path.exists(args.ckpt):
        print(f"ERROR: checkpoint not found: {args.ckpt}")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results -> {RESULTS_DIR}")
    print(f"Experiments: {args.only}")

    # merge into the existing SUMMARY.json by default so a partial rerun (e.g. only the
    # imputerefer datasets) doesn't wipe out previously-successful experiments/runs.
    summary = {}
    existing_json = os.path.join(RESULTS_DIR, "SUMMARY.json")
    if not args.fresh and os.path.exists(existing_json):
        with open(existing_json) as f:
            summary = json.load(f)
        print(f"Merging into existing summary: {existing_json}")
    summary["_meta"] = {"ckpt": args.ckpt, "seeds": args.seeds, "degrees": args.degrees,
                        "limit_test_batches": args.limit_test_batches}
    if args.datasets:
        print(f"Sweep datasets restricted to: {args.datasets}")

    for key in ALL_EXPERIMENTS:
        if key in args.only:
            EXPERIMENT_FUNCS[key](args, summary)

    write_summary(summary, args)


if __name__ == "__main__":
    main()

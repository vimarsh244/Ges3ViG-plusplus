import json
import numpy as np
from tqdm import tqdm

from ges3vig.evaluation.scanrefer_evaluator import ScanReferEvaluator


class ScanReferAngleEvaluator(ScanReferEvaluator):
    """Evaluator that breaks down IoU accuracy by elbow joint angle bins.

    Human meshes are generated with elbow bent at angles from 20° to 150°
    (in steps of 10°). This evaluator groups predictions by the arm angle
    used during scene composition, so you can see how pointing accuracy
    degrades (or improves) as the elbow angle changes.

    Default bins: [20,50), [50,80), [80,110), [110,150]
    """

    def __init__(self, *args, angle_bins=None, angle_labels=None,
                 results_save_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Default: 4 bins spanning the 20-150 degree range
        self.angle_bins = angle_bins or [20.0, 50.0, 80.0, 110.0, 160.0]
        self.angle_labels = angle_labels or [
            "20-50°", "50-80°", "80-110°", "110-150°"
        ]
        if len(self.angle_bins) != len(self.angle_labels) + 1:
            raise ValueError(
                f"angle_bins (len {len(self.angle_bins)}) must be "
                f"len(angle_labels) + 1 = {len(self.angle_labels) + 1}"
            )
        self.results_save_path = results_save_path

    def _print_angle_results(self, iou_25_by_angle, iou_50_by_angle,
                             counts_by_angle):
        """Pretty-print a table of IoU grouped by angle bin."""
        print()
        print(f"{'=' * 70}")
        print("  Sensitivity Analysis: IoU vs Elbow Joint Angle")
        print(f"{'=' * 70}")
        print("{0:<16}{1:<10}{2:<14}{3:<14}".format(
            "angle range", "count", "IoU@0.25", "IoU@0.50"))
        print(f"{'-' * 70}")
        for label in self.angle_labels:
            count = counts_by_angle.get(label, 0)
            iou_25 = iou_25_by_angle.get(label, np.nan)
            iou_50 = iou_50_by_angle.get(label, np.nan)
            print("{0:<16}{1:<10}{2:<14.1f}{3:<14.1f}".format(
                label, count, iou_25 * 100, iou_50 * 100))

        # Also print per-10-degree breakdown
        print(f"\n{'-' * 70}")
        print("  Per-angle (10° steps):")
        print(f"{'-' * 70}")
        print("{0:<16}{1:<10}{2:<14}{3:<14}".format(
            "angle", "count", "IoU@0.25", "IoU@0.50"))
        print(f"{'-' * 70}")
        for angle in sorted(self._per_angle_25.keys()):
            cnt = self._per_angle_counts.get(angle, 0)
            acc25 = self._per_angle_25.get(angle, np.nan)
            acc50 = self._per_angle_50.get(angle, np.nan)
            print("{0:<16}{1:<10}{2:<14.1f}{3:<14.1f}".format(
                f"{angle}°", cnt, acc25 * 100, acc50 * 100))
        print(f"{'=' * 70}")

    def evaluate(self, predictions):
        """Run standard evaluation + angle-binned sensitivity analysis."""
        all_gt_info_len = len(self.ground_truths)
        eval_type_mask = np.empty(all_gt_info_len, dtype=bool)
        angle_vals = np.full(all_gt_info_len, np.nan, dtype=np.float32)
        iou_25_tps = np.zeros(all_gt_info_len, dtype=bool)
        iou_50_tps = np.zeros(all_gt_info_len, dtype=bool)

        iterator = enumerate(
            tqdm(predictions.items(), desc="Evaluating")
            if self.verbose else predictions.items()
        )

        for i, (key, value) in iterator:
            gt_info = self.ground_truths[key]
            eval_type_mask[i] = self.evaluation_types[gt_info["eval_type"]]
            angle = gt_info.get("arm_angle", None)
            if angle is not None:
                angle_vals[i] = float(angle)
            iou_25_tps[i], iou_50_tps[i] = self._evaluate_one_query(
                value, gt_info)

        # ── Standard unique/multiple breakdown ──────────────────────────
        iou_25_results = {}
        iou_50_results = {}
        for sub_group in self.evaluation_types.keys():
            sel = eval_type_mask == self.evaluation_types[sub_group]
            if np.any(sel):
                iou_25_results[sub_group] = (
                    np.count_nonzero(iou_25_tps[sel]) / np.count_nonzero(sel))
                iou_50_results[sub_group] = (
                    np.count_nonzero(iou_50_tps[sel]) / np.count_nonzero(sel))
            else:
                iou_25_results[sub_group] = np.nan
                iou_50_results[sub_group] = np.nan
        iou_25_results["overall"] = (
            np.count_nonzero(iou_25_tps) / iou_25_tps.shape[0])
        iou_50_results["overall"] = (
            np.count_nonzero(iou_50_tps) / iou_50_tps.shape[0])

        # ── Angle-binned breakdown ──────────────────────────────────────
        iou_25_by_angle = {}
        iou_50_by_angle = {}
        counts_by_angle = {}
        for idx, label in enumerate(self.angle_labels):
            lo = self.angle_bins[idx]
            hi = self.angle_bins[idx + 1]
            mask = (angle_vals >= lo) & (angle_vals < hi)
            count = int(np.count_nonzero(mask))
            counts_by_angle[label] = count
            if count > 0:
                iou_25_by_angle[label] = (
                    np.count_nonzero(iou_25_tps[mask]) / count)
                iou_50_by_angle[label] = (
                    np.count_nonzero(iou_50_tps[mask]) / count)
            else:
                iou_25_by_angle[label] = np.nan
                iou_50_by_angle[label] = np.nan

        # ── Per-10-degree fine-grained breakdown ────────────────────────
        self._per_angle_25 = {}
        self._per_angle_50 = {}
        self._per_angle_counts = {}
        for angle_deg in range(20, 160, 10):
            mask = (angle_vals >= angle_deg - 0.5) & (angle_vals < angle_deg + 9.5)
            count = int(np.count_nonzero(mask))
            self._per_angle_counts[angle_deg] = count
            if count > 0:
                self._per_angle_25[angle_deg] = (
                    np.count_nonzero(iou_25_tps[mask]) / count)
                self._per_angle_50[angle_deg] = (
                    np.count_nonzero(iou_50_tps[mask]) / count)
            else:
                self._per_angle_25[angle_deg] = np.nan
                self._per_angle_50[angle_deg] = np.nan

        # ── Print ───────────────────────────────────────────────────────
        if self.verbose:
            self._print_results(iou_25_results, iou_50_results)
            self._print_angle_results(
                iou_25_by_angle, iou_50_by_angle, counts_by_angle)

        # ── Optionally save to JSON ─────────────────────────────────────
        full_results = {
            f"{self.metric_name}@0.25": iou_25_results,
            f"{self.metric_name}@0.5": iou_50_results,
            f"{self.metric_name}@0.25_by_angle": iou_25_by_angle,
            f"{self.metric_name}@0.5_by_angle": iou_50_by_angle,
            "angle_counts": counts_by_angle,
            "per_angle_detail": {
                str(a): {
                    "count": self._per_angle_counts[a],
                    "IoU@0.25": float(self._per_angle_25[a])
                    if not np.isnan(self._per_angle_25.get(a, np.nan)) else None,
                    "IoU@0.50": float(self._per_angle_50[a])
                    if not np.isnan(self._per_angle_50.get(a, np.nan)) else None,
                }
                for a in sorted(self._per_angle_counts.keys())
            },
        }

        if self.results_save_path:
            # make serialisable (numpy floats → python floats)
            def _convert(obj):
                if isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, dict):
                    return {k: _convert(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_convert(v) for v in obj]
                return obj

            with open(self.results_save_path, "w") as f:
                json.dump(_convert(full_results), f, indent=2)
            print(f"\n==> Angle sensitivity results saved to: "
                  f"{self.results_save_path}")

        return full_results

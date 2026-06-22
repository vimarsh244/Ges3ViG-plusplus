"""Combined ScanRefer evaluator: standard + proxemic-distance + elbow-angle.

One evaluator that produces, in a single pass / single log:

  * the usual unique / multiple / overall IoU@0.25 & IoU@0.50 table,
  * a proxemic-zone breakdown (intimate / personal / social / public),
    by human shoulder-centre -> target object-centre distance (Hall's zones),
  * an elbow-angle breakdown (coarse bins + per-10-degree),
  * and, crucially, a PER-QUERY dump (scene/object/ann, eval_type, distance,
    arm_angle, raw IoU, tp@0.25, tp@0.50) written to `results_save_path` as JSON
    so all graphs / analysis can be re-derived offline "our way".

Both `distance` and `arm_angle` come from the GT dict built in
`ges3vig/model/ges3vig.py:_parse_gt` (no-ops / None when absent, e.g. NoHuman).
"""
import json
import numpy as np
import torch
from tqdm import tqdm

from ges3vig.util.utils import get_batch_aabb_pair_ious
from ges3vig.evaluation.scanrefer_evaluator import ScanReferEvaluator


class ScanReferFullEvaluator(ScanReferEvaluator):
    def __init__(self, *args,
                 distance_bins=None, distance_labels=None,
                 angle_bins=None, angle_labels=None,
                 results_save_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Hall's proxemics (metres): intimate<0.46, personal<1.22, social<3.70, public>=3.70
        self.distance_bins = distance_bins or [0.0, 0.46, 1.22, 3.70, 1e9]
        self.distance_labels = distance_labels or ["intimate", "personal", "social", "public"]
        if len(self.distance_bins) != len(self.distance_labels) + 1:
            raise ValueError("distance_bins must be len(distance_labels) + 1")
        # elbow angle bins (degrees); meshes span 20-150 in 10-degree steps
        self.angle_bins = angle_bins or [20.0, 50.0, 80.0, 110.0, 160.0]
        self.angle_labels = angle_labels or ["20-50°", "50-80°", "80-110°", "110-150°"]
        if len(self.angle_bins) != len(self.angle_labels) + 1:
            raise ValueError("angle_bins must be len(angle_labels) + 1")
        self.results_save_path = results_save_path

    # ------------------------------------------------------------------ #
    # per-query raw IoU (float), reusing the base box-validation logic
    # ------------------------------------------------------------------ #
    def _iou_one_query(self, pred_info, gt_info):
        def _to_aabb_tensor(aabb):
            t = torch.as_tensor(aabb)
            if t.numel() == 0:
                return None
            if t.ndim == 1:
                t = t.view(2, 3) if t.numel() == 6 else None
            if t is None:
                return None
            if t.ndim == 2 and t.shape == (2, 3):
                t = t.unsqueeze(0)
            if t.ndim != 3 or t.shape[-2:] != (2, 3):
                return None
            return t

        pred_aabb = _to_aabb_tensor(pred_info["aabb_bound"])
        gt_aabb = _to_aabb_tensor(gt_info["aabb_bound"])
        if pred_aabb is None or gt_aabb is None:
            return 0.0
        return get_batch_aabb_pair_ious(pred_aabb, gt_aabb)[0].item()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _binned(values, tp, bins, labels):
        """Return ({label: acc}, {label: count}) for tp masked into value bins."""
        accs, counts = {}, {}
        for idx, label in enumerate(labels):
            mask = (values >= bins[idx]) & (values < bins[idx + 1])
            n = int(np.count_nonzero(mask))
            counts[label] = n
            accs[label] = (np.count_nonzero(tp[mask]) / n) if n else np.nan
        return accs, counts

    def _print_breakdown(self, title, col0, accs25, accs50, counts, labels):
        print(f"\n{'=' * 66}")
        print(title)
        print(f"{'-' * 66}")
        print("{0:<16}{1:<10}{2:<14}{3:<14}".format(col0, "count", "IoU@0.25", "IoU@0.50"))
        print(f"{'-' * 66}")
        for label in labels:
            print("{0:<16}{1:<10}{2:<14.1f}{3:<14.1f}".format(
                label, counts.get(label, 0),
                accs25.get(label, np.nan) * 100, accs50.get(label, np.nan) * 100))
        print(f"{'=' * 66}")

    def evaluate(self, predictions):
        n = len(self.ground_truths)
        eval_type_mask = np.empty(n, dtype=bool)
        distance_vals = np.full(n, np.nan, dtype=np.float32)
        angle_vals = np.full(n, np.nan, dtype=np.float32)
        iou_vals = np.zeros(n, dtype=np.float32)
        iou_25_tps = np.zeros(n, dtype=bool)
        iou_50_tps = np.zeros(n, dtype=bool)
        per_query = []

        items = predictions.items()
        iterator = enumerate(tqdm(items, desc="Evaluating") if self.verbose else items)
        for i, (key, value) in iterator:
            gt_info = self.ground_truths[key]
            eval_type_mask[i] = self.evaluation_types[gt_info["eval_type"]]
            dist = gt_info.get("distance", None)
            angle = gt_info.get("arm_angle", None)
            if dist is not None:
                distance_vals[i] = dist
            if angle is not None:
                angle_vals[i] = float(angle)
            iou = self._iou_one_query(value, gt_info)
            iou_vals[i] = iou
            iou_25_tps[i] = iou >= 0.25
            iou_50_tps[i] = iou >= 0.50
            per_query.append({
                "scene_id": key[0], "object_id": int(key[1]), "ann_id": int(key[2]),
                "eval_type": gt_info["eval_type"],
                "distance": None if dist is None else float(dist),
                "arm_angle": None if angle is None else float(angle),
                "iou": float(iou),
                "tp@0.25": bool(iou >= 0.25), "tp@0.50": bool(iou >= 0.50),
            })

        # ── standard unique / multiple / overall ──────────────────────────
        iou_25_results, iou_50_results = {}, {}
        for sub_group in self.evaluation_types.keys():
            sel = eval_type_mask == self.evaluation_types[sub_group]
            if np.any(sel):
                iou_25_results[sub_group] = np.count_nonzero(iou_25_tps[sel]) / np.count_nonzero(sel)
                iou_50_results[sub_group] = np.count_nonzero(iou_50_tps[sel]) / np.count_nonzero(sel)
            else:
                iou_25_results[sub_group] = np.nan
                iou_50_results[sub_group] = np.nan
        iou_25_results["overall"] = np.count_nonzero(iou_25_tps) / n
        iou_50_results["overall"] = np.count_nonzero(iou_50_tps) / n

        # ── distance (proxemic) + angle breakdowns ────────────────────────
        d25, d_cnt = self._binned(distance_vals, iou_25_tps, self.distance_bins, self.distance_labels)
        d50, _ = self._binned(distance_vals, iou_50_tps, self.distance_bins, self.distance_labels)
        a25, a_cnt = self._binned(angle_vals, iou_25_tps, self.angle_bins, self.angle_labels)
        a50, _ = self._binned(angle_vals, iou_50_tps, self.angle_bins, self.angle_labels)

        # per-10-degree fine breakdown
        per_angle = {}
        for deg in range(20, 160, 10):
            mask = (angle_vals >= deg - 0.5) & (angle_vals < deg + 9.5)
            cnt = int(np.count_nonzero(mask))
            per_angle[deg] = {
                "count": cnt,
                "IoU@0.25": float(np.count_nonzero(iou_25_tps[mask]) / cnt) if cnt else None,
                "IoU@0.50": float(np.count_nonzero(iou_50_tps[mask]) / cnt) if cnt else None,
            }

        if self.verbose:
            self._print_results(iou_25_results, iou_50_results)
            self._print_breakdown(
                "  Proxemic-distance breakdown (human -> target, metres)",
                "distance", d25, d50, d_cnt, self.distance_labels)
            self._print_breakdown(
                "  Elbow-angle breakdown (degrees)",
                "angle range", a25, a50, a_cnt, self.angle_labels)

        results = {
            f"{self.metric_name}@0.25": iou_25_results,
            f"{self.metric_name}@0.5": iou_50_results,
            f"{self.metric_name}@0.25_distance": d25,
            f"{self.metric_name}@0.5_distance": d50,
            "distance_counts": d_cnt,
            f"{self.metric_name}@0.25_by_angle": a25,
            f"{self.metric_name}@0.5_by_angle": a50,
            "angle_counts": a_cnt,
        }

        if self.results_save_path:
            def _convert(o):
                if isinstance(o, (np.floating, np.float32, np.float64)):
                    return float(o)
                if isinstance(o, np.integer):
                    return int(o)
                if isinstance(o, np.bool_):
                    return bool(o)
                if isinstance(o, dict):
                    return {k: _convert(v) for k, v in o.items()}
                if isinstance(o, (list, tuple)):
                    return [_convert(v) for v in o]
                return o
            dump = {
                "summary": _convert(results),
                "per_angle_detail": {str(k): v for k, v in per_angle.items()},
                "per_query": per_query,
                "config": {
                    "distance_bins": self.distance_bins, "distance_labels": self.distance_labels,
                    "angle_bins": self.angle_bins, "angle_labels": self.angle_labels,
                },
            }
            with open(self.results_save_path, "w") as f:
                json.dump(dump, f, indent=2)
            print(f"\n==> Full per-query results saved to: {self.results_save_path}")

        return results

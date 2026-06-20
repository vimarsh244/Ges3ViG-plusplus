import numpy as np
from tqdm import tqdm

from ges3vig.evaluation.scanrefer_evaluator import ScanReferEvaluator


class ScanReferDistanceEvaluator(ScanReferEvaluator):
    def __init__(self, *args, distance_bins=None, distance_labels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.distance_bins = distance_bins or [0.0, 0.46, 1.22, 3.70, 1e9]
        self.distance_labels = distance_labels or ["intimate", "personal", "social", "public"]
        if len(self.distance_bins) != len(self.distance_labels) + 1:
            raise ValueError("distance_bins must be len(distance_labels) + 1")

    def _print_distance_results(self, iou_25_results, iou_50_results, counts_by_dist):
        print(f"{'=' * 66}")
        print("{0:<12}{1:<10}{2:<14}{3:<14}".format("distance", "count", "IoU@0.25", "IoU@0.50"))
        print(f"{'-' * 66}")
        for label in self.distance_labels:
            count = counts_by_dist.get(label, 0)
            iou_25 = iou_25_results.get(label, np.nan)
            iou_50 = iou_50_results.get(label, np.nan)
            print("{0:<12}{1:<10}{2:<14.1f}{3:<14.1f}".format(
                label, count, iou_25 * 100, iou_50 * 100
            ))
        print(f"{'=' * 66}")

    def evaluate(self, predictions):
        all_gt_info_len = len(self.ground_truths)
        eval_type_mask = np.empty(all_gt_info_len, dtype=bool)
        distance_vals = np.full(all_gt_info_len, np.nan, dtype=np.float32)
        iou_25_tps = np.zeros(all_gt_info_len, dtype=bool)
        iou_50_tps = np.zeros(all_gt_info_len, dtype=bool)
        iterator = enumerate(tqdm(predictions.items(), desc="Evaluating") if self.verbose else predictions.items())

        for i, (key, value) in iterator:
            gt_info = self.ground_truths[key]
            eval_type_mask[i] = self.evaluation_types[gt_info["eval_type"]]
            dist = gt_info.get("distance", None)
            if dist is not None:
                distance_vals[i] = dist
            iou_25_tps[i], iou_50_tps[i] = self._evaluate_one_query(value, gt_info)

        iou_25_results = {}
        iou_50_results = {}
        for sub_group in self.evaluation_types.keys():
            selected_indices = eval_type_mask == self.evaluation_types[sub_group]
            if np.any(selected_indices):
                iou_25_results[sub_group] = np.count_nonzero(iou_25_tps[selected_indices]) / np.count_nonzero(selected_indices)
                iou_50_results[sub_group] = np.count_nonzero(iou_50_tps[selected_indices]) / np.count_nonzero(selected_indices)
            else:
                iou_25_results[sub_group] = np.nan
                iou_50_results[sub_group] = np.nan
        iou_25_results["overall"] = np.count_nonzero(iou_25_tps) / iou_25_tps.shape[0]
        iou_50_results["overall"] = np.count_nonzero(iou_50_tps) / iou_50_tps.shape[0]

        iou_25_dist = {}
        iou_50_dist = {}
        dist_counts = {}
        for idx, label in enumerate(self.distance_labels):
            start = self.distance_bins[idx]
            end = self.distance_bins[idx + 1]
            mask = (distance_vals >= start) & (distance_vals < end)
            count = np.count_nonzero(mask)
            dist_counts[label] = int(count)
            if count > 0:
                iou_25_dist[label] = np.count_nonzero(iou_25_tps[mask]) / count
                iou_50_dist[label] = np.count_nonzero(iou_50_tps[mask]) / count
            else:
                iou_25_dist[label] = np.nan
                iou_50_dist[label] = np.nan

        if self.verbose:
            self._print_results(iou_25_results, iou_50_results)
            self._print_distance_results(iou_25_dist, iou_50_dist, dist_counts)

        return {
            f"{self.metric_name}@0.25": iou_25_results,
            f"{self.metric_name}@0.5": iou_50_results,
            f"{self.metric_name}@0.25_distance": iou_25_dist,
            f"{self.metric_name}@0.5_distance": iou_50_dist,
            "distance_counts": dist_counts,
        }

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from ges3vig.util.utils import get_batch_aabb_pair_ious
from ges3vig.evaluation.general_evaluator import GeneralEvaluator


class ScanReferEvaluator(GeneralEvaluator):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluation_types = {"unique": 0, "multiple": 1}

    def _print_results(self, iou_25_results, iou_50_results):
        print(f"{'=' * 43}")
        print("{0:<12}{1:<12}{2:<12}{3:<12}".format("IoU", "unique", "multiple", "overall"))
        print(f"{'-' * 43}")
        line_1_str = '{:<12}'.format("0.25")
        for sub_group_type, score in iou_25_results.items():
            line_1_str += '{:<12.1f}'.format(score * 100)
        print(line_1_str)
        line_2_str = '{:<12}'.format("0.50")
        for sub_group_type, score in iou_50_results.items():
            line_2_str += '{:<12.1f}'.format(score * 100)
        print(line_2_str)
        print(f"{'=' * 43}")

    def evaluate(self, predictions):
        all_gt_info_len = len(self.ground_truths)
        eval_type_mask = np.empty(all_gt_info_len, dtype=bool)
        iou_25_tps = np.zeros(all_gt_info_len, dtype=bool)
        iou_50_tps = np.zeros(all_gt_info_len, dtype=bool)
        iterator = enumerate(tqdm(predictions.items(), desc="Evaluating") if self.verbose else predictions.items())
        for i, (key, value) in iterator:
            eval_type_mask[i] = self.evaluation_types[self.ground_truths[key]["eval_type"]]
            iou_25_tps[i], iou_50_tps[i] = self._evaluate_one_query(value, self.ground_truths[key])
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

        if self.verbose:
            self._print_results(iou_25_results, iou_50_results)

        return {f"{self.metric_name}@0.25": iou_25_results, f"{self.metric_name}@0.5": iou_50_results}

    def _evaluate_one_query(self, pred_info, gt_info):
        # initialize true positives
        iou_25_tp = 0
        iou_50_tp = 0
        def _to_aabb_tensor(aabb):
            t = torch.as_tensor(aabb)
            if t.numel() == 0:
                return None
            if t.ndim == 1:
                if t.numel() == 6:
                    t = t.view(2, 3)
                else:
                    return None
            if t.ndim == 2 and t.shape == (2, 3):
                t = t.unsqueeze(0)
            if t.ndim != 3 or t.shape[-2:] != (2, 3):
                return None
            return t

        # TODO: convert to batch process
        pred_aabb = _to_aabb_tensor(pred_info["aabb_bound"])
        gt_aabb = _to_aabb_tensor(gt_info["aabb_bound"])
        # skip invalid boxes
        if pred_aabb is None or gt_aabb is None:
            return iou_25_tp, iou_50_tp
        iou = get_batch_aabb_pair_ious(pred_aabb, gt_aabb)[0].item()
        if iou >= 0.25:
            iou_25_tp += 1
        if iou >= 0.5:
            iou_50_tp += 1
        return iou_25_tp, iou_50_tp

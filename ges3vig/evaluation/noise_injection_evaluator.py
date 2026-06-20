"""
Evaluator for robustness / corruption runs.

Functionally identical to :class:`ScanReferEvaluator` (same Acc@IoU0.25 and
Acc@IoU0.50 over unique / multiple / overall), but it is a distinct, selectable
evaluator that also prints the active ``eval_corruption`` settings so that every
robustness run is self-documenting and the numbers can never be confused with a
clean baseline.

The actual point-cloud corruption (Q7 Gaussian noise, Q8 partial capture) is
performed in the data loader via ``ges3vig.data.dataset.corruption`` and is
controlled by ``data.eval_corruption``.  This evaluator only *reports* it: pass
the same block in as ``corruption_cfg`` (see the ``*_noise.yaml`` configs).
"""

from ges3vig.evaluation.scanrefer_evaluator import ScanReferEvaluator


class NoiseInjectionEvaluator(ScanReferEvaluator):

    def __init__(self, *args, corruption_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.corruption_cfg = corruption_cfg

    def _print_corruption(self):
        cfg = self.corruption_cfg
        print(f"{'=' * 43}")
        print("ROBUSTNESS EVALUATION - input point cloud corrupted")
        print(f"{'-' * 43}")
        if not cfg or not cfg.get("enabled", False):
            print("eval_corruption: DISABLED (clean baseline)")
            print(f"{'=' * 43}")
            return
        print(f"seed           : {cfg.get('seed', 42)}")
        noise = cfg.get("noise", None)
        if noise and noise.get("enabled", False):
            print(
                "Q7 noise       : mode={mode} sigma={sigma} mu={mu} "
                "rgb_sigma={rgb_sigma}".format(
                    mode=noise.get("mode", "isotropic"),
                    sigma=noise.get("sigma", 0.0),
                    mu=noise.get("mu", 0.0),
                    rgb_sigma=noise.get("rgb_sigma", 0.0),
                )
            )
        else:
            print("Q7 noise       : off")
        partial = cfg.get("partial", None)
        if partial and partial.get("enabled", False):
            print(
                "Q8 partial     : mode={mode} keep_ratio={kr} fov_deg={fov} "
                "max_range={mr} num_boxes={nb} box_frac={bf}".format(
                    mode=partial.get("mode", "dropout"),
                    kr=partial.get("keep_ratio", 1.0),
                    fov=partial.get("fov_deg", 90.0),
                    mr=partial.get("max_range", None),
                    nb=partial.get("num_boxes", 1),
                    bf=partial.get("box_frac", 0.2),
                )
            )
        else:
            print("Q8 partial     : off")
        print(f"{'=' * 43}")

    def evaluate(self, predictions):
        if self.verbose:
            self._print_corruption()
        return super().evaluate(predictions)

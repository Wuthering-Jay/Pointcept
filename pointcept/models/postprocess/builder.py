"""
Prediction Postprocess Builder
"""

import torch
import torch.nn.functional as F

from pointcept.utils.registry import Registry

POSTPROCESSORS = Registry("postprocessors")
PREDICTION_AGGREGATORS = Registry("prediction_aggregators")


class PostprocessCompose(object):
    def __init__(self, cfg=None, class_names=None, ignore_index=-1):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(
                POSTPROCESSORS.build(
                    cfg=t_cfg,
                    default_args=dict(
                        class_names=class_names,
                        ignore_index=ignore_index,
                    ),
                )
            )

    def __call__(self, pred_dict):
        for transform in self.transforms:
            pred_dict = transform(pred_dict)
        return pred_dict


def build_postprocess(cfg=None, class_names=None, ignore_index=-1):
    if cfg is None or not cfg.get("enable", False):
        return None
    return PostprocessCompose(
        cfg=cfg.get("transforms", []),
        class_names=class_names,
        ignore_index=ignore_index,
    )


def build_prediction_aggregator(cfg=None, class_names=None, ignore_index=-1):
    cfg = cfg if cfg is not None else dict(type="SoftmaxSumAggregator")
    return PREDICTION_AGGREGATORS.build(
        cfg=cfg,
        default_args=dict(
            class_names=class_names,
            ignore_index=ignore_index,
        ),
    )


def build_prediction_dict(seg_logits, data_dict=None):
    prob = F.softmax(seg_logits, dim=-1)
    pred = prob.max(dim=1)[1]
    pred_dict = dict(
        seg_logits=seg_logits,
        prob=prob,
        pred=pred,
    )
    if data_dict is not None:
        for key in [
            "coord",
            "offset",
            "segment",
            "origin_coord",
            "origin_offset",
            "origin_segment",
            "inverse",
            "name",
        ]:
            if key in data_dict:
                pred_dict[key] = data_dict[key]
    return pred_dict


class BasePredictionAggregator(object):
    def __init__(self, class_names=None, ignore_index=-1):
        self.class_names = list(class_names) if class_names is not None else None
        self.ignore_index = ignore_index
        self.score = None

    def reset(self, num_points, num_classes, device):
        self.score = torch.zeros((num_points, num_classes), device=device)

    def update(self, index, logits):
        raise NotImplementedError

    def finalize(self, data_dict=None):
        assert self.score is not None
        return build_prediction_dict(self.score, data_dict=data_dict)

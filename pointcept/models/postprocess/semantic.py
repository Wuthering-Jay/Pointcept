"""
Semantic Segmentation Postprocess
"""

import torch
import pointops

from .builder import POSTPROCESSORS, PREDICTION_AGGREGATORS, BasePredictionAggregator


def _resolve_class_index(class_index, class_name, class_names):
    if class_index is not None:
        return int(class_index)
    if class_name is None:
        raise ValueError("Either class_index or class_name must be provided")
    if class_names is None:
        raise ValueError("class_names is required when class_name is used")
    if class_name not in class_names:
        raise KeyError(f"{class_name} is not in class_names")
    return class_names.index(class_name)


def _resolve_class_list(class_indices, class_names_cfg, class_names):
    if class_indices is not None:
        return [int(idx) for idx in class_indices]
    if class_names_cfg is not None:
        if class_names is None:
            raise ValueError("class_names is required when candidate class names are used")
        return [_resolve_class_index(None, name, class_names) for name in class_names_cfg]
    return None


class _BaseClasswisePostprocess(object):
    def __init__(
        self,
        class_names=None,
        ignore_index=-1,
        class_index=None,
        class_name=None,
        candidate_class_indices=None,
        candidate_class_names=None,
        record_stats=True,
    ):
        self.class_names = list(class_names) if class_names is not None else None
        self.ignore_index = ignore_index
        self.class_index = _resolve_class_index(class_index, class_name, self.class_names)
        self.candidate_class_indices = _resolve_class_list(
            candidate_class_indices,
            candidate_class_names,
            self.class_names,
        )
        self.record_stats = record_stats

    def _append_trace(self, pred_dict, changed_mask):
        if not self.record_stats:
            return
        if "postprocess_trace" not in pred_dict:
            pred_dict["postprocess_trace"] = []
        pred_dict["postprocess_trace"].append(
            dict(
                type=self.__class__.__name__,
                class_index=self.class_index,
                class_name=self.class_names[self.class_index]
                if self.class_names is not None
                else str(self.class_index),
                changed_points=int(changed_mask.sum().item()),
            )
        )

    def _select_fallback(self, prob, topk_idx, target_mask, fallback):
        fallback_pred = (
            topk_idx[target_mask, 1].clone()
            if topk_idx.shape[1] > 1
            else topk_idx[target_mask, 0].clone()
        )
        if fallback == "top2":
            return fallback_pred
        if fallback == "best_candidate":
            if self.candidate_class_indices is None or len(self.candidate_class_indices) == 0:
                raise ValueError("candidate_class_indices/candidate_class_names is required")
            candidate_scores = prob[target_mask][:, self.candidate_class_indices]
            candidate_idx = candidate_scores.max(dim=1)[1]
            return prob.new_tensor(self.candidate_class_indices, dtype=torch.long)[candidate_idx]
        return None


@PREDICTION_AGGREGATORS.register_module()
class SoftmaxSumAggregator(BasePredictionAggregator):
    def update(self, index, logits):
        prob = torch.softmax(logits, dim=-1)
        self.score[index.long(), :] += prob

    def finalize(self, data_dict=None):
        score = self.score
        prob = score / score.sum(dim=1, keepdim=True).clamp(min=1e-10)
        pred = prob.max(dim=1)[1]
        pred_dict = dict(
            seg_logits=score,
            prob=prob,
            pred=pred,
        )
        if data_dict is not None:
            for key, value in data_dict.items():
                pred_dict[key] = value
        return pred_dict


@POSTPROCESSORS.register_module()
class ConfidenceFallback(_BaseClasswisePostprocess):
    def __init__(
        self,
        min_prob,
        fallback="top2",
        fallback_class_index=None,
        fallback_class_name=None,
        **kwargs,
    ):
        super(ConfidenceFallback, self).__init__(**kwargs)
        self.min_prob = float(min_prob)
        self.fallback = fallback
        self.fallback_class_index = None
        if fallback_class_index is not None or fallback_class_name is not None:
            self.fallback_class_index = _resolve_class_index(
                fallback_class_index, fallback_class_name, self.class_names
            )

    def __call__(self, pred_dict):
        prob = pred_dict["prob"]
        pred = pred_dict["pred"].clone()
        topk = min(2, prob.shape[1])
        topk_idx = prob.topk(topk, dim=1)[1]
        target_mask = pred == self.class_index
        if not target_mask.any():
            return pred_dict
        low_conf_mask = target_mask & (prob[:, self.class_index] < self.min_prob)
        if not low_conf_mask.any():
            return pred_dict

        if self.fallback_class_index is not None:
            pred[low_conf_mask] = self.fallback_class_index
        else:
            pred[low_conf_mask] = self._select_fallback(
                prob, topk_idx, low_conf_mask, self.fallback
            )
        pred_dict["pred"] = pred
        self._append_trace(pred_dict, low_conf_mask)
        return pred_dict


@POSTPROCESSORS.register_module()
class MarginFallback(_BaseClasswisePostprocess):
    def __init__(
        self,
        max_margin,
        fallback="top2",
        fallback_class_index=None,
        fallback_class_name=None,
        **kwargs,
    ):
        super(MarginFallback, self).__init__(**kwargs)
        self.max_margin = float(max_margin)
        self.fallback = fallback
        self.fallback_class_index = None
        if fallback_class_index is not None or fallback_class_name is not None:
            self.fallback_class_index = _resolve_class_index(
                fallback_class_index, fallback_class_name, self.class_names
            )

    def __call__(self, pred_dict):
        prob = pred_dict["prob"]
        pred = pred_dict["pred"].clone()
        topk = min(2, prob.shape[1])
        topk_prob, topk_idx = prob.topk(topk, dim=1)
        margin = topk_prob[:, 0] - (
            topk_prob[:, 1] if topk > 1 else torch.zeros_like(topk_prob[:, 0])
        )
        target_mask = pred == self.class_index
        if not target_mask.any():
            return pred_dict
        small_margin_mask = target_mask & (margin < self.max_margin)
        if not small_margin_mask.any():
            return pred_dict

        if self.fallback_class_index is not None:
            pred[small_margin_mask] = self.fallback_class_index
        else:
            pred[small_margin_mask] = self._select_fallback(
                prob, topk_idx, small_margin_mask, self.fallback
            )
        pred_dict["pred"] = pred
        self._append_trace(pred_dict, small_margin_mask)
        return pred_dict


@POSTPROCESSORS.register_module()
class ConnectivityRelabel(_BaseClasswisePostprocess):
    def __init__(
        self,
        min_component_size,
        k_neighbors=16,
        fallback="neighbor_major",
        fallback_class_index=None,
        fallback_class_name=None,
        **kwargs,
    ):
        super(ConnectivityRelabel, self).__init__(**kwargs)
        self.min_component_size = int(min_component_size)
        self.k_neighbors = int(k_neighbors)
        self.fallback = fallback
        self.fallback_class_index = None
        if fallback_class_index is not None or fallback_class_name is not None:
            self.fallback_class_index = _resolve_class_index(
                fallback_class_index, fallback_class_name, self.class_names
            )

    def __call__(self, pred_dict):
        if "coord" not in pred_dict:
            return pred_dict
        coord = pred_dict["coord"]
        pred = pred_dict["pred"].clone()
        offset = pred_dict.get("offset")
        if offset is None:
            offset = coord.new_tensor([coord.shape[0]], dtype=torch.int32)
        if not isinstance(offset, torch.Tensor):
            offset = torch.as_tensor(offset, device=coord.device, dtype=torch.int32)
        target_mask = pred == self.class_index
        if int(target_mask.sum().item()) == 0:
            return pred_dict

        idx, _ = pointops.knn_query(self.k_neighbors, coord.float(), offset.int())
        valid_knn_mask = idx >= 0
        neighbor_pred = pred[idx.long().clamp(min=0)]
        visited = torch.zeros(pred.shape[0], dtype=torch.bool, device=pred.device)
        changed_mask = torch.zeros_like(visited)
        target_indices = torch.where(target_mask)[0]

        for seed in target_indices.tolist():
            if visited[seed]:
                continue
            queue = [seed]
            component = []
            visited[seed] = True
            while queue:
                current = queue.pop()
                component.append(current)
                neighbors = idx[current][valid_knn_mask[current]].long()
                valid_neighbors = neighbors[pred[neighbors] == self.class_index]
                for neighbor in valid_neighbors.tolist():
                    if target_mask[neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            if len(component) >= self.min_component_size:
                continue
            component_tensor = pred.new_tensor(component, dtype=torch.long)
            if self.fallback_class_index is not None:
                new_class = self.fallback_class_index
            else:
                boundary_neighbors = idx[component_tensor].reshape(-1)
                boundary_neighbors = boundary_neighbors[boundary_neighbors >= 0].long()
                boundary_neighbors = boundary_neighbors[pred[boundary_neighbors] != self.class_index]
                if boundary_neighbors.numel() == 0:
                    continue
                neighbor_classes = pred[boundary_neighbors]
                counts = torch.bincount(neighbor_classes, minlength=pred_dict["prob"].shape[1])
                new_class = int(counts.max(dim=0)[1].item())
            pred[component_tensor] = new_class
            changed_mask[component_tensor] = True

        pred_dict["pred"] = pred
        self._append_trace(pred_dict, changed_mask)
        return pred_dict

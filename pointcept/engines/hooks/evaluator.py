"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import json
import os
import numpy as np
import torch
import torch.distributed as dist
import pointops
from uuid import uuid4

import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS


class _StreamingStats(object):
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = None
        self.max = None

    def update(self, tensor):
        if tensor.numel() == 0:
            return
        tensor = tensor.detach().float()
        self.count += int(tensor.numel())
        self.sum += float(tensor.sum().item())
        self.sum_sq += float((tensor * tensor).sum().item())
        tensor_min = float(tensor.min().item())
        tensor_max = float(tensor.max().item())
        self.min = tensor_min if self.min is None else min(self.min, tensor_min)
        self.max = tensor_max if self.max is None else max(self.max, tensor_max)

    def merge(self, other):
        self.count += int(other["count"])
        self.sum += float(other["sum"])
        self.sum_sq += float(other["sum_sq"])
        other_min = other["min"]
        other_max = other["max"]
        if other_min is not None:
            self.min = other_min if self.min is None else min(self.min, other_min)
        if other_max is not None:
            self.max = other_max if self.max is None else max(self.max, other_max)

    def to_dict(self):
        if self.count == 0:
            return dict(count=0, mean=None, std=None, min=None, max=None)
        mean = self.sum / self.count
        var = max(self.sum_sq / self.count - mean * mean, 0.0)
        return dict(
            count=self.count,
            mean=mean,
            std=var**0.5,
            min=self.min,
            max=self.max,
        )

    def state_dict(self):
        return dict(
            count=self.count,
            sum=self.sum,
            sum_sq=self.sum_sq,
            min=self.min,
            max=self.max,
        )


class _SemSegDiagnosticCollector(object):
    def __init__(
        self,
        num_classes,
        class_names,
        ignore_index,
        topk=(1, 2, 3),
        prob_num_bins=20,
        top_confusions=10,
        pair_topk=5,
    ):
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.ignore_index = ignore_index
        self.topk = sorted({int(k) for k in topk if int(k) > 0})
        self.max_k = min(max(self.topk), self.num_classes) if self.topk else 1
        self.prob_num_bins = int(prob_num_bins)
        self.top_confusions = int(top_confusions)
        self.pair_topk = int(pair_topk)

        self.total_points = 0
        self.correct_points = 0

        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.support = np.zeros(self.num_classes, dtype=np.int64)
        self.predicted = np.zeros(self.num_classes, dtype=np.int64)
        self.correct_per_class = np.zeros(self.num_classes, dtype=np.int64)

        self.topk_hits = {k: 0 for k in self.topk}
        self.topk_hits_per_class = {
            k: np.zeros(self.num_classes, dtype=np.int64) for k in self.topk
        }

        self.mis_pred_prob_sum = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.float64
        )
        self.mis_gt_prob_sum = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.float64
        )

        self.correct_max_prob_hist = np.zeros(self.prob_num_bins, dtype=np.int64)
        self.incorrect_max_prob_hist = np.zeros(self.prob_num_bins, dtype=np.int64)
        self.correct_margin_prob_hist = np.zeros(self.prob_num_bins, dtype=np.int64)
        self.incorrect_margin_prob_hist = np.zeros(self.prob_num_bins, dtype=np.int64)
        self.incorrect_gt_prob_hist = np.zeros(self.prob_num_bins, dtype=np.int64)

        self.correct_top1_logit_stats = _StreamingStats()
        self.incorrect_top1_logit_stats = _StreamingStats()
        self.incorrect_gt_logit_stats = _StreamingStats()

    def _histogram(self, values):
        if values.numel() == 0:
            return np.zeros(self.prob_num_bins, dtype=np.int64)
        values = values.detach().float().clamp_(0, 1)
        hist = torch.histc(values, bins=self.prob_num_bins, min=0, max=1)
        return hist.cpu().numpy().astype(np.int64)

    def update(self, logits, target):
        target = target.view(-1).long()
        logits = logits.reshape(-1, logits.shape[-1])
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        logits = logits[valid_mask]
        if target.numel() == 0:
            return

        probs = torch.softmax(logits, dim=1)
        topk_prob, topk_idx = probs.topk(self.max_k, dim=1)
        pred = topk_idx[:, 0]
        max_prob = topk_prob[:, 0]
        gt_prob = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        top1_logit, _ = logits.max(dim=1)
        gt_logit = logits.gather(1, target.unsqueeze(1)).squeeze(1)

        if self.num_classes > 1:
            top2_prob = topk_prob[:, 1]
        else:
            top2_prob = torch.zeros_like(max_prob)
        prob_margin = max_prob - top2_prob

        correct_mask = pred == target
        incorrect_mask = ~correct_mask

        pair_index = target * self.num_classes + pred
        confusion = torch.bincount(
            pair_index, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)
        confusion = confusion.cpu().numpy().astype(np.int64)

        self.confusion += confusion
        self.support += confusion.sum(axis=1)
        self.predicted += confusion.sum(axis=0)
        self.correct_per_class += np.diag(confusion)
        self.total_points += int(target.numel())
        self.correct_points += int(correct_mask.sum().item())

        for k in self.topk:
            hit_mask = (topk_idx[:, : min(k, self.max_k)] == target.unsqueeze(1)).any(dim=1)
            self.topk_hits[k] += int(hit_mask.sum().item())
            hit_per_class = torch.bincount(
                target, weights=hit_mask.float(), minlength=self.num_classes
            )
            self.topk_hits_per_class[k] += hit_per_class.cpu().numpy().astype(np.int64)

        self.correct_max_prob_hist += self._histogram(max_prob[correct_mask])
        self.incorrect_max_prob_hist += self._histogram(max_prob[incorrect_mask])
        self.correct_margin_prob_hist += self._histogram(prob_margin[correct_mask])
        self.incorrect_margin_prob_hist += self._histogram(prob_margin[incorrect_mask])
        self.incorrect_gt_prob_hist += self._histogram(gt_prob[incorrect_mask])

        self.correct_top1_logit_stats.update(top1_logit[correct_mask])
        self.incorrect_top1_logit_stats.update(top1_logit[incorrect_mask])
        self.incorrect_gt_logit_stats.update(gt_logit[incorrect_mask])

        if incorrect_mask.any():
            mis_pair_index = pair_index[incorrect_mask]
            mis_pred_prob_sum = torch.bincount(
                mis_pair_index,
                weights=max_prob[incorrect_mask],
                minlength=self.num_classes * self.num_classes,
            ).reshape(self.num_classes, self.num_classes)
            mis_gt_prob_sum = torch.bincount(
                mis_pair_index,
                weights=gt_prob[incorrect_mask],
                minlength=self.num_classes * self.num_classes,
            ).reshape(self.num_classes, self.num_classes)
            self.mis_pred_prob_sum += mis_pred_prob_sum.cpu().numpy()
            self.mis_gt_prob_sum += mis_gt_prob_sum.cpu().numpy()

    def state_dict(self):
        return dict(
            total_points=self.total_points,
            correct_points=self.correct_points,
            confusion=self.confusion,
            support=self.support,
            predicted=self.predicted,
            correct_per_class=self.correct_per_class,
            topk_hits=self.topk_hits,
            topk_hits_per_class=self.topk_hits_per_class,
            mis_pred_prob_sum=self.mis_pred_prob_sum,
            mis_gt_prob_sum=self.mis_gt_prob_sum,
            correct_max_prob_hist=self.correct_max_prob_hist,
            incorrect_max_prob_hist=self.incorrect_max_prob_hist,
            correct_margin_prob_hist=self.correct_margin_prob_hist,
            incorrect_margin_prob_hist=self.incorrect_margin_prob_hist,
            incorrect_gt_prob_hist=self.incorrect_gt_prob_hist,
            correct_top1_logit_stats=self.correct_top1_logit_stats.state_dict(),
            incorrect_top1_logit_stats=self.incorrect_top1_logit_stats.state_dict(),
            incorrect_gt_logit_stats=self.incorrect_gt_logit_stats.state_dict(),
        )

    def merge_state_dict(self, state_dict):
        self.total_points += int(state_dict["total_points"])
        self.correct_points += int(state_dict["correct_points"])
        self.confusion += state_dict["confusion"]
        self.support += state_dict["support"]
        self.predicted += state_dict["predicted"]
        self.correct_per_class += state_dict["correct_per_class"]
        for k in self.topk:
            self.topk_hits[k] += int(state_dict["topk_hits"][k])
            self.topk_hits_per_class[k] += state_dict["topk_hits_per_class"][k]
        self.mis_pred_prob_sum += state_dict["mis_pred_prob_sum"]
        self.mis_gt_prob_sum += state_dict["mis_gt_prob_sum"]
        self.correct_max_prob_hist += state_dict["correct_max_prob_hist"]
        self.incorrect_max_prob_hist += state_dict["incorrect_max_prob_hist"]
        self.correct_margin_prob_hist += state_dict["correct_margin_prob_hist"]
        self.incorrect_margin_prob_hist += state_dict["incorrect_margin_prob_hist"]
        self.incorrect_gt_prob_hist += state_dict["incorrect_gt_prob_hist"]
        self.correct_top1_logit_stats.merge(state_dict["correct_top1_logit_stats"])
        self.incorrect_top1_logit_stats.merge(state_dict["incorrect_top1_logit_stats"])
        self.incorrect_gt_logit_stats.merge(state_dict["incorrect_gt_logit_stats"])

    def _hist_to_dict(self, hist):
        bin_edges = np.linspace(0.0, 1.0, self.prob_num_bins + 1).tolist()
        return dict(bin_edges=bin_edges, counts=hist.tolist())

    def summarize(self):
        total_points = self.total_points
        error_points = total_points - self.correct_points
        overall_acc = self.correct_points / (total_points + 1e-10)
        topk_accuracy = {
            str(k): self.topk_hits[k] / (total_points + 1e-10) for k in self.topk
        }

        row_sum = self.confusion.sum(axis=1, keepdims=True)
        confusion_row_normalized = np.divide(
            self.confusion,
            np.maximum(row_sum, 1),
            out=np.zeros_like(self.confusion, dtype=np.float64),
            where=row_sum > 0,
        )

        class_summary = []
        top_misclassifications = []
        for class_idx in range(self.num_classes):
            support = int(self.support[class_idx])
            predicted = int(self.predicted[class_idx])
            correct = int(self.correct_per_class[class_idx])
            false_negative = support - correct
            false_positive = predicted - correct
            recall = correct / (support + 1e-10)
            precision = correct / (predicted + 1e-10)
            error_rate = false_negative / (support + 1e-10)

            topk_hit_rates = {}
            for k in self.topk:
                topk_hit_rates[f"top{k}"] = (
                    self.topk_hits_per_class[k][class_idx] / (support + 1e-10)
                    if support > 0
                    else None
                )

            wrong_row = self.confusion[class_idx].copy()
            wrong_row[class_idx] = 0
            mistaken = []
            if wrong_row.sum() > 0:
                mistake_indices = np.argsort(-wrong_row)[: self.pair_topk]
                for pred_idx in mistake_indices:
                    count = int(wrong_row[pred_idx])
                    if count <= 0:
                        continue
                    mistaken.append(
                        dict(
                            pred_index=int(pred_idx),
                            pred_name=self.class_names[pred_idx],
                            count=count,
                            rate_within_class=count / (support + 1e-10),
                            mean_pred_prob=self.mis_pred_prob_sum[class_idx, pred_idx]
                            / count,
                            mean_gt_prob=self.mis_gt_prob_sum[class_idx, pred_idx]
                            / count,
                        )
                    )

            class_summary.append(
                dict(
                    index=class_idx,
                    name=self.class_names[class_idx],
                    support=support,
                    predicted=predicted,
                    correct=correct,
                    false_negative=false_negative,
                    false_positive=false_positive,
                    recall=recall,
                    precision=precision,
                    error_rate=error_rate,
                    topk_hit_rates=topk_hit_rates,
                    top_mistaken_predictions=mistaken,
                )
            )

        confusion_no_diag = self.confusion.copy()
        np.fill_diagonal(confusion_no_diag, 0)
        pair_order = np.argsort(confusion_no_diag.reshape(-1))[::-1]
        for flat_idx in pair_order[: self.top_confusions]:
            count = int(confusion_no_diag.reshape(-1)[flat_idx])
            if count <= 0:
                continue
            gt_idx = flat_idx // self.num_classes
            pred_idx = flat_idx % self.num_classes
            gt_support = max(int(self.support[gt_idx]), 1)
            top_misclassifications.append(
                dict(
                    gt_index=int(gt_idx),
                    gt_name=self.class_names[gt_idx],
                    pred_index=int(pred_idx),
                    pred_name=self.class_names[pred_idx],
                    count=count,
                    rate_within_gt=count / gt_support,
                    mean_pred_prob=self.mis_pred_prob_sum[gt_idx, pred_idx] / count,
                    mean_gt_prob=self.mis_gt_prob_sum[gt_idx, pred_idx] / count,
                )
            )

        return dict(
            total_points=total_points,
            correct_points=self.correct_points,
            error_points=error_points,
            overall_accuracy=overall_acc,
            overall_error_rate=error_points / (total_points + 1e-10),
            topk_accuracy=topk_accuracy,
            class_summary=class_summary,
            top_misclassifications=top_misclassifications,
            confusion_matrix=self.confusion.tolist(),
            confusion_matrix_row_normalized=confusion_row_normalized.tolist(),
            distributions=dict(
                correct_max_prob=self._hist_to_dict(self.correct_max_prob_hist),
                incorrect_max_prob=self._hist_to_dict(self.incorrect_max_prob_hist),
                correct_margin_prob=self._hist_to_dict(self.correct_margin_prob_hist),
                incorrect_margin_prob=self._hist_to_dict(self.incorrect_margin_prob_hist),
                incorrect_gt_prob=self._hist_to_dict(self.incorrect_gt_prob_hist),
            ),
            logit_statistics=dict(
                correct_top1=self.correct_top1_logit_stats.to_dict(),
                incorrect_top1=self.incorrect_top1_logit_stats.to_dict(),
                incorrect_gt=self.incorrect_gt_logit_stats.to_dict(),
            ),
        )


@HOOKS.register_module()
class ClsEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["cls_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                label,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_acc  # save for saver
        self.trainer.comm_info["current_metric_name"] = "allAcc"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def __init__(self, write_cls_iou=False, diagnostic=None):
        # 是否写入每个类别的IoU
        self.write_cls_iou = write_cls_iou
        self.diagnostic = diagnostic if diagnostic is not None else dict(enable=False)

    def _build_diagnostic_collector(self):
        if not self.diagnostic or not self.diagnostic.get("enable", False):
            return None
        return _SemSegDiagnosticCollector(
            num_classes=self.trainer.cfg.data.num_classes,
            class_names=self.trainer.cfg.data.names,
            ignore_index=self.trainer.cfg.data.ignore_index,
            topk=self.diagnostic.get("topk", (1, 2, 3)),
            prob_num_bins=self.diagnostic.get("prob_num_bins", 20),
            top_confusions=self.diagnostic.get("top_confusions", 10),
            pair_topk=self.diagnostic.get("pair_topk", 5),
        )

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        diagnostic_collector = self._build_diagnostic_collector()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            diagnostic_logits = output
            segment = input_dict["segment"]
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                idx = idx.flatten().long()
                pred = pred[idx]
                diagnostic_logits = output[idx]
                segment = input_dict["origin_segment"]
            if diagnostic_collector is not None:
                diagnostic_collector.update(diagnostic_logits, segment)
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            accuracy = np.sum(intersection) / (np.sum(target + 1e-10))
            
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} Acc {acc:.4f}".format(
                    iter=i + 1,
                    max_iter=len(self.trainer.val_loader),
                    loss=loss.item(),
                    acc=accuracy
                )
            )
            if self.trainer.cfg.empty_cache:
                torch.cuda.empty_cache()
            if self.trainer.cfg.empty_cache_freq > 0:
                if (i + 1) % self.trainer.cfg.empty_cache_freq == 0:
                    torch.cuda.empty_cache()

        diagnostic_summary = None
        if diagnostic_collector is not None:
            comm.synchronize()
            gathered_states = comm.gather(diagnostic_collector.state_dict(), dst=0)
            if comm.is_main_process():
                merged_collector = self._build_diagnostic_collector()
                for state_dict in gathered_states:
                    merged_collector.merge_state_dict(state_dict)
                diagnostic_summary = merged_collector.summarize()

        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        # acc_class = intersection / (target + 1e-10)
        rec_class = intersection / (target + 1e-10)
        pre_class = intersection / (union+intersection-target + 1e-10)
        f1_class = 2 * (pre_class * rec_class) / (pre_class + rec_class + 1e-10)
        m_iou = np.mean(iou_class)
        # m_acc = np.mean(acc_class)
        m_rec = np.mean(rec_class)
        m_pre = np.mean(pre_class)
        m_f1 = np.mean(f1_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mPre/mRec/mF1/OA {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_pre, m_rec, m_f1, all_acc
            )
        )
        if diagnostic_summary is not None:
            topk_items = [
                "Top{} {:.4f}".format(k, diagnostic_summary["topk_accuracy"][str(k)])
                for k in sorted(map(int, diagnostic_summary["topk_accuracy"].keys()))
            ]
            self.trainer.logger.info(
                "Diagnostic: total/error/OA {}/{}/{:.4f}, {}.".format(
                    diagnostic_summary["total_points"],
                    diagnostic_summary["error_points"],
                    diagnostic_summary["overall_accuracy"],
                    ", ".join(topk_items),
                )
            )
            if len(diagnostic_summary["top_misclassifications"]) > 0:
                top_pairs = []
                for item in diagnostic_summary["top_misclassifications"][:3]:
                    top_pairs.append(
                        "{}->{} {} ({:.2%})".format(
                            item["gt_name"],
                            item["pred_name"],
                            item["count"],
                            item["rate_within_gt"],
                        )
                    )
                self.trainer.logger.info(
                    "Diagnostic Top Confusions: {}.".format(" | ".join(top_pairs))
                )
        # 计算最长的类别名称长度 & 最大的索引宽度
        max_name_length = max(len(name) for name in self.trainer.cfg.data.names)
        max_idx_width = len(str(self.trainer.cfg.data.num_classes - 1))
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx:<{idx_width}}-{name:<{name_width}} Result: iou/pre/rec/f1 {iou:.4f}/{pre:.4f}/{rec:.4f}/{f1:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    pre=pre_class[i],
                    rec=rec_class[i],
                    f1=f1_class[i],
                    idx_width=max_idx_width,
                    name_width=max_name_length
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mPre", m_pre, current_epoch)
            self.trainer.writer.add_scalar("val/mRec", m_rec, current_epoch)
            self.trainer.writer.add_scalar("val/mF1", m_f1, current_epoch)
            self.trainer.writer.add_scalar("val/OA", all_acc, current_epoch)
            if self.write_cls_iou:
                for i in range(self.trainer.cfg.data.num_classes):
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
                        iou_class[i],
                        current_epoch,
                    )
        if diagnostic_summary is not None and comm.is_main_process():
            diagnostic_summary["epoch"] = current_epoch
            diagnostic_summary["metrics"] = dict(
                loss=loss_avg,
                mIoU=m_iou,
                mPre=m_pre,
                mRec=m_rec,
                mF1=m_f1,
                OA=all_acc,
            )
            diagnostic_path = os.path.join(
                self.trainer.cfg.save_path,
                "diagnostic_epoch_{:04d}.json".format(current_epoch),
            )
            with open(diagnostic_path, "w", encoding="utf-8") as f:
                json.dump(diagnostic_summary, f, indent=2, ensure_ascii=False)
            self.trainer.logger.info(
                "Diagnostic json saved to {}".format(diagnostic_path)
            )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self, segment_ignore_index=(-1,), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.trainer.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.trainer.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.trainer.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.trainer.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = []
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert (
                len(input_dict["offset"]) == 1
            )  # currently only support bs 1 for each GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]

            segment = input_dict["segment"]
            instance = input_dict["instance"]
            # map to origin
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                idx = idx.cpu().flatten().long()
                output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes.append(dict(gt=gt_instances, pred=pred_instance))

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        self.trainer.logger.info(
            "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                all_ap, all_ap_50, all_ap_25
            )
        )
        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                    idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
            self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
            self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
        self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver

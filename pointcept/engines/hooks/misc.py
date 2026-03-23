"""
Misc Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
import glob
import os
import shutil
import time
import gc
import torch
import torch.utils.data
from collections import OrderedDict

if sys.version_info >= (3, 10):
    from collections.abc import Sequence
else:
    from collections import Sequence
from pointcept.utils.timer import Timer
from pointcept.utils.comm import is_main_process, synchronize
from pointcept.utils.cache import shared_dict
from pointcept.utils.scheduler import CosineScheduler
import pointcept.utils.comm as comm

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        self._start_time = time.perf_counter()
        _remain_epoch = self.trainer.max_epoch - self.trainer.start_epoch
        self._remain_iter = _remain_epoch * len(self.trainer.train_loader)

    def before_epoch(self):
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self, interval=1):
        self.interval = interval
        self.curr_iter = 0
        self.model_output_keys = []

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        self.curr_iter = self.trainer.start_epoch * len(self.trainer.train_loader)
        # if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
        #     wandb.define_metric("params/*", step_metric="Iter")
        #     wandb.define_metric("train_batch/*", step_metric="Iter")
        #     wandb.define_metric("train/*", step_metric="Epoch")

    def before_step(self):
        self.curr_iter += 1
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def _format_loss_str(self, key_vals):
        """Format loss keys: total loss + optional parenthetical of sub-losses.

        key_vals: list of (key, value) tuples from model_output_keys × storage.
        Returns a string like: ``loss: 2.3702 (ce: 1.2345 lovasz: 1.1357)``
        For non-loss keys, appended normally.
        """
        main_parts = []   # non-subloss entries
        sub_parts = []    # loss/{name} entries
        for key, val in key_vals:
            if key.startswith("loss/"):
                short = key[len("loss/"):]
                sub_parts.append(f"{short}: {val:.4f}")
            else:
                entry = f"{key}: {val:.4f}"
                if key == "loss" and sub_parts == [] and any(
                    k.startswith("loss/") for k, _ in key_vals
                ):
                    # Will append sub after
                    main_parts.append(("__loss__", entry))
                else:
                    main_parts.append((key, entry))

        result = ""
        for tag, entry in main_parts:
            result += entry + " "
            if tag == "__loss__" and sub_parts:
                result += "(" + "  ".join(sub_parts) + ") "
        if not any(t == "__loss__" for t, _ in main_parts) and sub_parts:
            result += "(" + "  ".join(sub_parts) + ") "
        return result

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            self.model_output_keys = model_output_dict.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, model_output_dict[key].item())

        # Only log every `interval` steps; always accumulate storage for correct avg
        if self.curr_iter % self.interval != 0:
            self.trainer.comm_info["iter_info"] = ""
            return

        key_vals = [
            (key, self.trainer.storage.history(key).val)
            for key in self.model_output_keys
        ]
        self.trainer.comm_info["iter_info"] += self._format_loss_str(key_vals)
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)
        self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/lr", lr, self.curr_iter)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.curr_iter,
                )
            # if self.trainer.cfg.enable_wandb:

            #     wandb.log(
            #         {"Iter": self.curr_iter, "params/lr": lr}, step=self.curr_iter
            #     )
            #     for key in self.model_output_keys:
            #         wandb.log(
            #             {
            #                 "Iter": self.curr_iter,
            #                 f"train_batch/{key}": self.trainer.storage.history(key).val,
            #             },
            #             step=wandb.run.step,
            #         )

    def after_epoch(self):
        epoch_info = "Train result: "
        key_vals = [
            (key, self.trainer.storage.history(key).avg)
            for key in self.model_output_keys
        ]
        epoch_info += self._format_loss_str(key_vals)
        self.trainer.logger.info(epoch_info)
        if self.trainer.writer is not None:
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )

            # if self.trainer.cfg.enable_wandb:

            #     for key in self.model_output_keys:
            #         wandb.log(
            #             {
            #                 "Epoch": self.trainer.epoch + 1,
            #                 f"train/{key}": self.trainer.storage.history(key).avg,
            #             },
            #             step=wandb.run.step,
            #         )


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None or int, None indicate only save model last

    def after_epoch(self):
        if is_main_process():
            is_best = False
            if self.trainer.cfg.evaluate:
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )

            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "scaler": (
                        self.trainer.scaler.state_dict()
                        if self.trainer.cfg.enable_amp
                        else None
                    ),
                    "best_metric_value": self.trainer.best_metric_value,
                },
                filename + ".tmp",
            )
            os.replace(filename + ".tmp", filename)
            if is_best:
                shutil.copyfile(
                    filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )
            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"epoch_{self.trainer.epoch + 1}.pth",
                    ),
                )


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement, 1)
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")


@HOOKS.register_module()
class PreciseEvaluator(HookBase):
    def __init__(self, test_last=False):
        self.test_last = test_last

    def after_train(self):
        from pointcept.engines.test import TESTERS
        
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Precise Evaluation >>>>>>>>>>>>>>>>"
        )
        torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        tester = TESTERS.build(
            dict(type=cfg.test.type, cfg=cfg, model=self.trainer.model)
        )
        if self.test_last:
            self.trainer.logger.info("=> Testing on model_last ...")
        else:
            self.trainer.logger.info("=> Testing on model_best ...")
            best_path = os.path.join(
                self.trainer.cfg.save_path, "model", "model_best.pth"
            )
            checkpoint = torch.load(best_path, weights_only=False)
            state_dict = checkpoint["state_dict"]
            tester.model.load_state_dict(state_dict, strict=True)
        tester.test()


@HOOKS.register_module()
class DataCacheOperator(HookBase):
    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        self.data_list = self.get_data_list()

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split))
        else:
            raise NotImplementedError
        return data_list

    def get_cache_name(self, data_path):
        data_name = data_path.replace(os.path.dirname(self.data_root), "")
        return "pointcept" + data_name.replace(os.path.sep, "-")

    def before_train(self):
        self.trainer.logger.info(
            f"=> Caching dataset: {self.data_root}, split: {self.split} ..."
        )
        if is_main_process():
            dataset = self.trainer.train_loader.dataset
            for i in range(len(dataset)):
                data_dict = dataset[i]
                name = data_dict["name"]
                shared_dict(f"Pointcept-{name}", data_dict)
        synchronize()


@HOOKS.register_module()
class RuntimeProfiler(HookBase):
    def __init__(
        self,
        forward=True,
        backward=True,
        interrupt=False,
        warm_up=2,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.forward = forward
        self.backward = backward
        self.interrupt = interrupt
        self.warm_up = warm_up
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import profile, record_function, ProfilerActivity

        for i, input_dict in enumerate(self.trainer.train_loader):
            if i == self.warm_up + 1:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            if self.forward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as forward_prof:
                    with record_function("model_inference"):
                        output_dict = self.trainer.model(input_dict)
            else:
                output_dict = self.trainer.model(input_dict)
            loss = output_dict["loss"]
            if self.backward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as backward_prof:
                    with record_function("model_inference"):
                        loss.backward()
            self.trainer.logger.info(f"Profile: [{i + 1}/{self.warm_up + 1}]")
        if self.forward:
            self.trainer.logger.info(
                "Forward profile: \n"
                + str(
                    forward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            forward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "forward_trace.json")
            )

        if self.backward:
            self.trainer.logger.info(
                "Backward profile: \n"
                + str(
                    backward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            backward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "backward_trace.json")
            )
        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class RuntimeProfilerV2(HookBase):
    def __init__(
        self,
        interrupt=False,
        wait=1,
        warmup=1,
        active=10,
        repeat=1,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.interrupt = interrupt
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import (
            profile,
            record_function,
            ProfilerActivity,
            schedule,
            tensorboard_trace_handler,
        )

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=tensorboard_trace_handler(self.trainer.cfg.save_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
        for i, input_dict in enumerate(self.trainer.train_loader):
            if i >= (self.wait + self.warmup + self.active) * self.repeat:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with record_function("model_forward"):
                output_dict = self.trainer.model(input_dict)
                loss = output_dict["loss"]
            with record_function("model_backward"):
                loss.backward()
            prof.step()
            self.trainer.logger.info(
                f"Profile: [{i + 1}/{(self.wait + self.warmup + self.active) * self.repeat}]"
            )
        self.trainer.logger.info(
            "Profile: \n"
            + str(
                prof.key_averages().table(
                    sort_by=self.sort_by, row_limit=self.row_limit
                )
            )
        )
        prof.stop()

        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class WeightDecaySchedular(HookBase):
    def __init__(
        self,
        base_value=0.04,
        final_value=0.2,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.scheduler = None

    def before_train(self):
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        self.scheduler = CosineScheduler(
            base_value=self.base_value,
            final_value=self.final_value,
            total_iters=self.trainer.cfg.scheduler.total_steps,
        )
        self.scheduler.iter = curr_step

    def before_step(self):
        wd = self.scheduler.step()
        for param_group in self.trainer.optimizer.param_groups:
            param_group["weight_decay"] = wd
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/wd", wd, self.scheduler.iter)


@HOOKS.register_module()
class GarbageHandler(HookBase):
    def __init__(self, interval=150, disable_auto=True, empty_cache=False):
        self.interval = interval
        self.disable_auto = disable_auto
        self.empty_cache = empty_cache
        self.iter = 1

    def before_train(self):
        if self.disable_auto:
            gc.disable()
            self.trainer.logger.info("Disable automatic garbage collection")

    def before_epoch(self):
        self.iter = 1

    def after_step(self):
        if self.iter % self.interval == 0:
            gc.collect()
            if self.empty_cache:
                torch.cuda.empty_cache()
            self.trainer.logger.info("Garbage collected")
        self.iter += 1

    def after_train(self):
        gc.collect()
        torch.cuda.empty_cache()


@HOOKS.register_module()
class CacheCleaner(HookBase):
    """GPU/CPU cache cleaner hook (interval-based + adaptive).

    缓存清理策略:
    - **固定清理点**: after_epoch (每个 epoch 结束)、after_train (训练→测试过渡)
      始终执行 ``gc.collect()`` + ``torch.cuda.empty_cache()`` 释放碎片。
    - **固定间隔清理**: 当 ``step_clean_interval`` 为正整数时，每隔 N 个 step
      执行一次缓存清理。设为 ``None`` 时不进行固定间隔清理。
    - **自适应清理**: 统计每个 step 的耗时，当某个 step 耗时超过
      ``mean × time_multiplier`` 或超过 ``abs_threshold_sec`` 绝对阈值时，
      执行清理以缓解显存震荡导致的卡顿。
    - **warmup**: 前 ``warmup_steps`` 步只收集统计，不触发自适应清理。

    每次清理都会在日志中输出原因和耗时信息。

    Args:
        warmup_steps (int): 热身步数，期间只统计不触发自适应清理。默认 10。
        time_multiplier (float): 相对阈值倍数。step 耗时 > mean × multiplier 时触发。默认 2.0。
        abs_threshold_sec (float): 绝对阈值秒数。step 耗时 > 该值时触发。
            设为 0 或 None 禁用。默认 None。
        window_size (int): 滑动窗口大小，用于计算均值。默认 50。
        step_clean_interval (int | None): 固定间隔清理步数。为正整数时每 N 步
            清理一次缓存；为 ``None`` 时不进行固定间隔清理。默认 None。
    """

    def __init__(
        self,
        warmup_steps=10,
        time_multiplier=2.0,
        abs_threshold_sec=None,
        window_size=50,
        step_clean_interval=None,
    ):
        self.warmup_steps = warmup_steps
        self.time_multiplier = time_multiplier
        self.abs_threshold_sec = abs_threshold_sec
        self.window_size = window_size
        self.step_clean_interval = step_clean_interval

        # runtime state
        self._step_times = []   # recent step durations (sliding window, train)
        self._step_count = 0
        self._step_start = None
        self._clean_count = 0   # total cleans
        # separate window for val/test iterations
        self._ext_times = []
        self._ext_count = 0
        # logger may come from trainer (hook mode) or be set externally (test mode)
        self._logger = None

    # ------ helpers ------

    @property
    def logger(self):
        """Return the logger — from trainer if attached, else the externally set one."""
        if self._logger is not None:
            return self._logger
        if hasattr(self, "trainer") and self.trainer is not None:
            return self.trainer.logger
        return None

    @logger.setter
    def logger(self, value):
        self._logger = value

    def _clean(self, reason: str):
        """Execute gc + cuda empty_cache and log the event."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._clean_count += 1
        log = self.logger
        if log is not None:
            log.info(
                f"[CacheCleaner] cache cleared (#{self._clean_count}) — {reason}"
            )

    def check_and_clean(self, elapsed: float, context: str = "val") -> bool:
        """Interval + adaptive clean check for external callers (val / test loops).

        Uses a separate sliding window from the train-step window so that
        val/test timing statistics are independent of training timing.

        Args:
            elapsed: Wall-clock seconds for this iteration.
            context: Label used in log messages (e.g. "val iter 42/200").

        Returns:
            True if a clean was triggered, False otherwise.
        """
        self._ext_count += 1
        self._ext_times.append(elapsed)
        if len(self._ext_times) > self.window_size:
            self._ext_times.pop(0)

        # Fixed interval cleaning
        if (
            self.step_clean_interval is not None
            and self._ext_count % self.step_clean_interval == 0
        ):
            self._clean(
                f"{context} — interval clean (every {self.step_clean_interval} iters)"
            )
            return True

        # Skip adaptive check during warmup
        if self._ext_count <= self.warmup_steps:
            return False

        mean_time = sum(self._ext_times) / len(self._ext_times)

        if elapsed > mean_time * self.time_multiplier:
            self._clean(
                f"{context} took {elapsed:.3f}s "
                f"(mean {mean_time:.3f}s × {self.time_multiplier} = "
                f"{mean_time * self.time_multiplier:.3f}s)"
            )
            return True

        if self.abs_threshold_sec and elapsed > self.abs_threshold_sec:
            self._clean(
                f"{context} took {elapsed:.3f}s "
                f"(abs threshold {self.abs_threshold_sec:.3f}s)"
            )
            return True

        return False

    def reset_ext_stats(self):
        """Reset external (val/test) timing statistics."""
        self._ext_times.clear()
        self._ext_count = 0

    # ------ fixed cleaning points ------

    def after_epoch(self):
        # Reset ext window each epoch so statistics stay fresh
        self.reset_ext_stats()
        self._clean(f"end of epoch {self.trainer.epoch}")

    def after_train(self):
        self._clean("training finished, preparing for test")

    # ------ step-level cleaning (train) ------

    def before_step(self):
        self._step_start = time.perf_counter()

    def after_step(self):
        if self._step_start is None:
            return

        elapsed = time.perf_counter() - self._step_start
        self._step_count += 1

        # update sliding window
        self._step_times.append(elapsed)
        if len(self._step_times) > self.window_size:
            self._step_times.pop(0)

        # Fixed interval cleaning
        if (
            self.step_clean_interval is not None
            and self._step_count % self.step_clean_interval == 0
        ):
            self._clean(
                f"step {self._step_count} — interval clean "
                f"(every {self.step_clean_interval} steps)"
            )
            return

        # skip warmup phase for adaptive cleaning
        if self._step_count <= self.warmup_steps:
            return

        mean_time = sum(self._step_times) / len(self._step_times)

        # check relative threshold
        if elapsed > mean_time * self.time_multiplier:
            self._clean(
                f"step {self._step_count} took {elapsed:.3f}s "
                f"(mean {mean_time:.3f}s × {self.time_multiplier} = "
                f"{mean_time * self.time_multiplier:.3f}s)"
            )
            return

        # check absolute threshold
        if self.abs_threshold_sec and elapsed > self.abs_threshold_sec:
            self._clean(
                f"step {self._step_count} took {elapsed:.3f}s "
                f"(abs threshold {self.abs_threshold_sec:.3f}s)"
            )

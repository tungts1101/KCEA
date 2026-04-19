import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import sys
import logging
import time
import copy
import gc
import itertools
from data_manager import DataManager
from config import DATA_TABLE, BASE_CONFIG, EXPERIMENT_CONFIGS, PARAM_SWEEP
from helper import (
    Model,
    compute_metrics,
    accuracy,
    set_random,
    merge,
    count_parameters,
    seed_worker,
)
from torch.distributions import MultivariateNormal

g = torch.Generator()
g.manual_seed(0)

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CHECKPOINT_ABLATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_ablation")
os.makedirs(CHECKPOINT_ABLATION_DIR, exist_ok=True)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_ABLATION_DIR = "logs_ablation"
os.makedirs(LOG_ABLATION_DIR, exist_ok=True)


class Learner:
    def __init__(self, config):
        self._config = config
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_task = -1
        self.mlp_matrix = []
        self._cls_to_task_idx = {}

        self.model = Model(config)
        self.model.cuda()
        self._total_written_bytes = 0  # cumulative bytes written to disk across all saves
        self._save(self.model.get_backbone_trainable_params(), self.backbone_checkpoint())
        self._mlp_faa = 0.0
        self._mlp_ffm = 0.0
        self._mlp_asa = 0.0
        self._pre_align_faa = None   # post-merge accuracy before alignment (None if alignment not used)
        self._pre_align_ffm = None
        self._pre_align_asa = None
        self._total_align_time = 0.0
        self._total_train_time = 0.0
        self._total_eval_time = 0.0
        self._total_merge_time = 0.0
        self._total_gauss_time = 0.0
        self._total_data_time = 0.0
        # per-task trainable param counts recorded just before SGD (peak per task)
        self._task_trainable_params = []
        # peak on-disk storage observed at any point (bytes)
        self._peak_storage_bytes = 0

    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks
        self._total_classnum = data_manager.get_total_classnum()
        self._t_learn_start = time.time()

        self.model.cuda()

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            self.train()  # SGD + merge only; alignment handled below

            # ── Pre-alignment eval (post-merge accuracy) ──────────────────────
            # Skipped for task 0 (single head — nothing to align across).
            if self._config.get("train_ca", False) and self._cur_task > 0:
                t_eval = time.time()
                _, faa_pre, ffm_pre, asa_pre, grouped_pre = self._eval_task_metrics()
                self._total_eval_time += time.time() - t_eval
                self._pre_align_faa = faa_pre
                self._pre_align_ffm = ffm_pre
                self._pre_align_asa = asa_pre
                # Log full matrix at final task so it can be parsed into result.json
                if self._total_classes == self._total_classnum:
                    pre_matrix = self.mlp_matrix + [grouped_pre]
                    logging.info("[Evaluation] Accuracy matrix (post-merge):")
                    for row in pre_matrix:
                        logging.info(f"  {[round(float(v), 2) for v in row]}")

            # ── Alignment ─────────────────────────────────────────────────────
            if self._config.get("train_ca", False):
                self.align_classifier()

            # ── Final eval (post-align, or post-merge/train if no alignment) ──
            t_eval = time.time()
            self.eval()
            self._total_eval_time += time.time() - t_eval

            # ── Side-by-side comparison (both lines together) ─────────────────
            if self._config.get("train_ca", False) and self._cur_task > 0:
                logging.info(
                    f"[Evaluation] Task {self._cur_task} [post-merge ]: "
                    f"FAA={faa_pre:.2f}, FFM={ffm_pre:.2f}, ASA={asa_pre:.2f}"
                )
                logging.info(
                    f"[Evaluation] Task {self._cur_task} [post-align ]: "
                    f"FAA={self._mlp_faa:.2f}, FFM={self._mlp_ffm:.2f}, ASA={self._mlp_asa:.2f}"
                )
                logging.info(
                    f"[Evaluation] Task {self._cur_task} [Δ alignment]: "
                    f"ΔFAA={self._mlp_faa - faa_pre:+.2f}, "
                    f"ΔFFM={self._mlp_ffm - ffm_pre:+.2f}, "
                    f"ΔASA={self._mlp_asa - asa_pre:+.2f}"
                )

            self.after_task()
            self._peak_storage_bytes = max(self._peak_storage_bytes, self._current_storage_bytes())

        torch.save(self.model.state_dict(), self.model_checkpoint())  # reproduce only, not counted in storage

        self._log_final_summary()

    def _save(self, obj, path):
        """torch.save wrapper that accumulates total bytes written to disk."""
        torch.save(obj, path)
        size = os.path.getsize(path)
        self._total_written_bytes += size
        logging.info(f"[Storage] Saved {os.path.basename(path)}  ({size / 1024**2:.2f} MB)  |  cumulative written: {self._total_written_bytes / 1024**2:.2f} MB")

    def _current_storage_bytes(self):
        """Sum bytes of all checkpoint files belonging to this run."""
        scan_dir = CHECKPOINT_ABLATION_DIR if self._config.get("train_ablation", False) else CHECKPOINT_DIR
        if not os.path.isdir(scan_dir):
            return 0
        pfx = self.prefix()
        return sum(
            os.path.getsize(os.path.join(scan_dir, f))
            for f in os.listdir(scan_dir)
            if f.startswith(pfx)
        )

    def _log_final_summary(self):
        wall_time = time.time() - self._t_learn_start

        # ── Time breakdown ────────────────────────────────────────────────────
        sgd_time   = self._total_train_time - self._total_merge_time  # SGD epochs only
        gauss_time = self._total_gauss_time                           # Gaussian stats (inside align)
        nes_time   = self._total_align_time - gauss_time              # NES loop only
        other_time = wall_time - self._total_train_time - self._total_align_time - self._total_eval_time - self._total_data_time
        active_time = wall_time - self._total_eval_time - self._total_data_time

        # ── Storage breakdown ─────────────────────────────────────────────────
        pfx = self.prefix()
        ckpt_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith(pfx)] if os.path.isdir(CHECKPOINT_DIR) else []

        def _bytes_matching(keyword):
            return sum(
                os.path.getsize(os.path.join(CHECKPOINT_DIR, f))
                for f in ckpt_files if keyword in f
            )

        backbone_bytes   = _bytes_matching("_backbone_")
        merged_bytes     = _bytes_matching("_merged_")
        head_bytes       = sum(
            os.path.getsize(os.path.join(CHECKPOINT_DIR, f))
            for f in ckpt_files
            if "_head_" in f and "_alignment" not in f
        )
        align_head_bytes = _bytes_matching("_alignment")

        # Minimum inference-time storage:
        #   incremental merge → only the last merged backbone is needed (1 file)
        #   non-incremental   → all per-task adapters + last merged backbone
        #   no merge at all   → all per-task adapters
        # In all cases, all head files are needed.
        num_tasks_run = self._cur_task + 1
        last_merged_path = self.merged_checkpoint(self._cur_task)
        last_merged_bytes = os.path.getsize(last_merged_path) if os.path.exists(last_merged_path) else 0
        if self._config.get("train_merge", False):
            if self._config.get("model_merge_incremental", False):
                inference_backbone_bytes = last_merged_bytes  # 1 backbone
            else:
                inference_backbone_bytes = backbone_bytes + last_merged_bytes
        else:
            inference_backbone_bytes = backbone_bytes  # per-task adapters only
        inference_bytes = inference_backbone_bytes + head_bytes + align_head_bytes

        # Gaussian statistics live in RAM (not on disk) — compute their size
        gauss_bytes = 0
        if hasattr(self, "_class_means"):
            gauss_bytes += self._class_means.numel() * 4
        if hasattr(self, "_class_covs"):
            gauss_bytes += self._class_covs.numel() * 4

        # ── Trainable parameter breakdown ─────────────────────────────────────
        peak_trainable     = max((t["total"] for t in self._task_trainable_params), default=0)
        peak_total_at_task = next((t["model_total"] for t in self._task_trainable_params if t["total"] == peak_trainable), 0)
        total_params       = count_parameters(self.model)

        def _mb(b): return b / 1024 ** 2

        logging.info("\n" + "=" * 80)
        logging.info("[Summary] ╔══ END-OF-RUN RESOURCE REPORT ══╗")
        logging.info(f"[Summary]   Method : {self._config.get('config_name', 'kcea')}  |  Dataset : {self._config['dataset_name']}  |  Seed : {self._config['seed']}")
        logging.info("[Summary]")

        # Time
        logging.info(f"[Summary] ── Wall-clock time: {wall_time:.1f}s ──────────────────────────")
        logging.info(f"[Summary]   Training (SGD, all tasks)              : {sgd_time:.1f}s  ({sgd_time/wall_time*100:.1f}%)")
        logging.info(f"[Summary]   Parameter merging (TIES)               : {self._total_merge_time:.1f}s  ({self._total_merge_time/wall_time*100:.1f}%)")
        logging.info(f"[Summary]   Gaussian statistics (per task)         : {gauss_time:.1f}s  ({gauss_time/wall_time*100:.1f}%)")
        logging.info(f"[Summary]   NES alignment (per task)               : {nes_time:.1f}s  ({nes_time/wall_time*100:.1f}%)")
        logging.info(f"[Summary]   Evaluation / test inference            : {self._total_eval_time:.1f}s  ({self._total_eval_time/wall_time*100:.1f}%)")
        logging.info(f"[Summary]   Data loading (train batches)           : {self._total_data_time:.1f}s  ({self._total_data_time/wall_time*100:.1f}%)")
        logging.info(f"[Summary]   Other (I/O, misc)                      : {max(other_time,0):.1f}s  ({max(other_time,0)/wall_time*100:.1f}%)")
        logging.info(f"[Summary]   ──────────────────────────────────────────────────────────────")
        logging.info(f"[Summary]   Active compute (wall − eval − data)    : {active_time:.1f}s  ({active_time/wall_time*100:.1f}%)")
        logging.info("[Summary]")

        # Storage
        # Breakdown: frozen pretrained backbone (shared, not owned) vs PEFT adapter vs heads vs stats
        frozen_bytes   = sum(p.numel() * p.element_size() for p in self.model.backbone.parameters() if not p.requires_grad)
        peft_bytes     = sum(p.numel() * p.element_size() for p in self.model.backbone.parameters() if p.requires_grad)
        head_ram_bytes = sum(p.numel() * p.element_size() for p in self.model.classifier.parameters())
        full_model_bytes = frozen_bytes + peft_bytes + head_ram_bytes   # comparable to baseline "model weights in RAM"
        owned_ram_bytes  = peft_bytes + head_ram_bytes + gauss_bytes    # what KCEA actually needs to store

        num_tasks_run    = self._cur_task + 1
        total_disk_bytes = backbone_bytes + merged_bytes + head_bytes + align_head_bytes
        logging.info(f"[Summary] ── Storage ─────────────────────────────────────────────────────")
        logging.info(f"[Summary]   Model weights in RAM (final task)      : {_mb(full_model_bytes):.1f} MB  [baseline-comparable]")
        logging.info(f"[Summary]     Frozen pretrained backbone              : {_mb(frozen_bytes):.1f} MB  (shared, not stored by KCEA)")
        logging.info(f"[Summary]     PEFT adapter params                     : {_mb(peft_bytes):.1f} MB")
        logging.info(f"[Summary]     Classifier heads (all tasks)            : {_mb(head_ram_bytes):.1f} MB")
        logging.info(f"[Summary]   Actual RAM owned by KCEA                : {_mb(owned_ram_bytes):.1f} MB  [PEFT + heads + statistics]")
        logging.info(f"[Summary]     Gaussian statistics (means + covs)      : {_mb(gauss_bytes):.1f} MB  ({self._total_classes} classes × {self.model.feature_dim}-dim)")
        logging.info(f"[Summary]   Total written to disk (training)       : {_mb(self._total_written_bytes):.1f} MB")
        logging.info(f"[Summary]   On-disk checkpoints (current)          : {_mb(total_disk_bytes):.1f} MB  ({len(ckpt_files)} files)")
        logging.info(f"[Summary]     Task adapter checkpoints (_backbone_T)  : {_mb(backbone_bytes):.1f} MB  ({num_tasks_run} files)")
        logging.info(f"[Summary]     Merged backbone checkpoints (_merged_T) : {_mb(merged_bytes):.1f} MB  ({num_tasks_run} files)")
        logging.info(f"[Summary]     Classifier heads (_head_T)              : {_mb(head_bytes):.1f} MB  ({num_tasks_run} files)")
        if align_head_bytes > 0:
            n_align = len([f for f in ckpt_files if "_alignment" in f])
            logging.info(f"[Summary]     Post-alignment heads (_head_T_align)  : {_mb(align_head_bytes):.1f} MB  ({n_align} files)")
        logging.info(f"[Summary]   Minimum inference storage              : {_mb(inference_bytes):.1f} MB  (1 backbone + {num_tasks_run} heads)")
        logging.info("[Summary]")

        # Parameters
        last_trainable = self._task_trainable_params[-1]["total"] if self._task_trainable_params else 0

        logging.info(f"[Summary] ── Trainable parameters ─────────────────────────────────────────")
        logging.info(f"[Summary]   Total model params (final task)        : {total_params:,}")
        logging.info(f"[Summary]   Peak trainable (any task)              : {peak_trainable:,}  ({peak_trainable*100/max(peak_total_at_task,1):.2f}% of {peak_total_at_task:,} at that task)")
        logging.info(f"[Summary]   Final task trainable                   : {last_trainable:,}  ({last_trainable*100/max(total_params,1):.2f}% of total at final task)")
        logging.info("[Summary]")

        # Alignment effect — only when alignment was used (task > 0 ran)
        if self._pre_align_faa is not None:
            logging.info(f"[Summary] ── Alignment effect (end of run) ──────────────────────────────────")
            logging.info(f"[Summary]   Post-merge  : FAA={self._pre_align_faa:.2f}  FFM={self._pre_align_ffm:.2f}  ASA={self._pre_align_asa:.2f}")
            logging.info(f"[Summary]   Post-align  : FAA={self._mlp_faa:.2f}  FFM={self._mlp_ffm:.2f}  ASA={self._mlp_asa:.2f}")
            logging.info(f"[Summary]   Δ alignment : ΔFAA={self._mlp_faa-self._pre_align_faa:+.2f}  ΔFFM={self._mlp_ffm-self._pre_align_ffm:+.2f}  ΔASA={self._mlp_asa-self._pre_align_asa:+.2f}")
            logging.info("[Summary]")
        logging.info("=" * 80)

    def before_task(self, task, data_manager):
        self._total_classes = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_task = task

        for clz in range(self._known_classes, self._total_classes):
            self._cls_to_task_idx[clz] = self._cur_task

        if task > 0 and self._config.get("train_merge", False):
            prev_merged = self.merged_checkpoint(task - 1)
            if os.path.exists(prev_merged):
                self.load_backbone(torch.load(prev_merged))
                logging.info(f"[Training] Loaded merged backbone (task {task - 1}) as init for task {task}")

    def after_task(self):
        self._known_classes = self._total_classes

    def eval(self):
        acc, faa, ffm, asa, grouped = self._eval_task_metrics()
        self._record_eval(acc, faa, ffm, asa, grouped)

    def train(self):
        t_train_start = time.time()
        trainset = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        train_loader = DataLoader(
            trainset,
            batch_size=self._config["train_batch_size"],
            shuffle=True,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g,
        )

        backbone_has_norm = self._config.get("model_use_norm", True)
        classifier_has_norm = not backbone_has_norm

        self.model.update_classifier(
            self._total_classes - self._known_classes,
            with_norm=classifier_has_norm,
            with_bias=False,
            freeze_old=True,
            norm_layer="ln",
        )
        self.model.cuda()

        epochs = int(self._config["train_epochs"])
        base_lr = self._config["train_base_lr"]
        weight_decay = self._config["train_weight_decay"]

        parameters = [
            {
                "params": [p for p in self.model.backbone.parameters() if p.requires_grad],
                "lr": base_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [p for p in self.model.classifier.heads[self._cur_task].parameters() if p.requires_grad],
                "lr": base_lr,
                "weight_decay": weight_decay,
            },
        ]

        optimizer = optim.SGD(parameters, lr=base_lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        self.model.train()
        # Record peak trainable params for this task (backbone adapters + current head)
        backbone_trainable = count_parameters(self.model.backbone, trainable=True)
        head_trainable = count_parameters(self.model.classifier.heads[self._cur_task], trainable=True)
        self._task_trainable_params.append({
            "task": self._cur_task,
            "backbone": backbone_trainable,
            "head": head_trainable,
            "total": backbone_trainable + head_trainable,
            "model_total": count_parameters(self.model),
        })
        logging.info(f"[Training] Task {self._cur_task}")
        logging.info(f"[Training] {self.model}")

        for epoch in range(epochs):
            total_loss, total_acc, total = 0, 0, 0

            _t_batch_start = time.time()
            for _, (_, _, x, y) in enumerate(train_loader):
                self._total_data_time += time.time() - _t_batch_start
                x, y = x.cuda(), y.cuda()

                features = self.model.get_features(x)

                y_local = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                logits = self.model.classifier.heads[-1](features)
                loss = F.cross_entropy(logits, y_local, ignore_index=-100)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_acc += (logits.argmax(dim=1) == y_local).sum().item()
                total += len(y_local)
                total_loss += loss.item() * len(y_local)
                _t_batch_start = time.time()

            scheduler.step()

            logging.info(
                f"[Training] Epoch {epoch + 1}/{epochs}, "
                f"Total Loss: {total_loss / total:.4f}, "
                f"Acc: {total_acc / total:.4f}"
            )

        self._save(self.model.get_backbone_trainable_params(), self.backbone_checkpoint(self._cur_task))

        self._train_time = time.time() - t_train_start
        self._total_train_time += self._train_time

        if self._config.get("train_merge", False):
            t_merge = time.time()
            self.merge()
            self._total_merge_time += time.time() - t_merge

    def compute_multivariate_normal(self):
        _t_gauss = time.time()
        sample_method = self._config.get("train_ca_sample_method", "covariance")
        logging.info(
            f"[Alignment] Compute class statistics ({sample_method}) for classes "
            f"{self._known_classes} - {self._total_classes - 1}"
        )
        total_class = self._total_classes
        feature_dim = self.model.feature_dim

        # Shape of _class_covs depends on what we need to store:
        #   "covariance" → [C, D, D]  full matrix
        #   "diagonal"   → [C, D]     per-dim variances
        #   "variance"   → [C]        single scalar variance per class
        if sample_method == "diagonal":
            cov_shape = (total_class, feature_dim)
        elif sample_method == "variance":
            cov_shape = (total_class,)
        else:  # "covariance"
            cov_shape = (total_class, feature_dim, feature_dim)

        if not hasattr(self, "_class_means") or not hasattr(self, "_class_covs"):
            self._class_means = torch.zeros((total_class, feature_dim))
            self._class_covs = torch.zeros(cov_shape)
        else:
            new_class_means = torch.zeros((total_class, feature_dim))
            new_class_means[: self._known_classes] = self._class_means
            self._class_means = new_class_means
            new_class_covs = torch.zeros(cov_shape)
            new_class_covs[: self._known_classes] = self._class_covs
            self._class_covs = new_class_covs

        for cls_idx in range(self._known_classes, self._total_classes):
            proto_set = self.data_manager.get_dataset(
                np.arange(cls_idx, cls_idx + 1), source="train", mode="test"
            )
            proto_loader = DataLoader(
                proto_set, batch_size=512, shuffle=False,
                num_workers=4, worker_init_fn=seed_worker, generator=g
            )

            features_list = []
            self.model.eval()
            with torch.no_grad():
                for _, (_, _, x, y) in enumerate(proto_loader):
                    x = x.cuda()
                    f = self.model.get_features(x)
                    features_list.append(f.cpu())

            features_list = torch.cat(features_list, dim=0)
            class_mean = torch.mean(features_list, dim=0)
            self._class_means[cls_idx] = class_mean

            # var(unbiased=True) returns NaN with <2 samples; clamp does not fix NaN
            if len(features_list) < 2:
                logging.warning(f"[Alignment] Class {cls_idx}: only {len(features_list)} sample(s), using identity fallback")
                if sample_method == "diagonal":
                    self._class_covs[cls_idx] = torch.full((feature_dim,), 1e-4)
                elif sample_method == "variance":
                    self._class_covs[cls_idx] = torch.tensor(1e-4)
                else:
                    self._class_covs[cls_idx] = torch.eye(feature_dim) * 1e-4
                continue

            if sample_method == "diagonal":
                class_var = features_list.var(dim=0, unbiased=True).clamp(min=1e-6)
                self._class_covs[cls_idx] = class_var
            elif sample_method == "variance":
                scalar_var = features_list.var(dim=0, unbiased=True).mean().clamp(min=1e-6)
                self._class_covs[cls_idx] = scalar_var
            else:  # "covariance"
                class_cov = torch.cov(features_list.T)
                # Ensure positive definiteness
                min_eigenval = torch.linalg.eigvals(class_cov).real.min()
                reg_term = (abs(min_eigenval.item()) + 1e-3) if min_eigenval <= 0 else 1e-4
                class_cov = class_cov + torch.eye(feature_dim) * reg_term
                # Verify with Cholesky; fall back to diagonal if still not PD
                try:
                    torch.linalg.cholesky(class_cov)
                except RuntimeError:
                    class_var = features_list.var(dim=0, unbiased=True).clamp(min=1e-6)
                    class_cov = torch.diag(class_var + 1e-3)
                self._class_covs[cls_idx] = class_cov

        self._total_gauss_time += time.time() - _t_gauss

    def _record_eval(self, acc, faa, ffm, asa, grouped):
        """Append grouped accuracies to mlp_matrix, update metrics, and log."""
        self.mlp_matrix.append(grouped)
        self._mlp_faa, self._mlp_ffm, self._mlp_asa = faa, ffm, asa
        logging.info(f"[Evaluation] Task {self._cur_task}: FAA={faa:.2f}, FFM={ffm:.2f}, ASA={asa:.2f}")
        if self._total_classes == self._total_classnum:
            label = "(post-align)" if self._config.get("train_ca", False) else ""
            logging.info(f"[Evaluation] Accuracy matrix {label}:".strip())
            for row in self.mlp_matrix:
                logging.info(f"  {[round(float(v), 2) for v in row]}")
            if len(self.mlp_matrix) > 1:
                diag = [float(self.mlp_matrix[i][i]) for i in range(len(self.mlp_matrix))]
                final = [float(v) for v in self.mlp_matrix[-1]]
                retention = [final[j] / diag[j] if diag[j] > 0 else 0.0 for j in range(len(diag))]
                logging.info(f"[Evaluation] Worst-task retention: {min(retention):.4f} (per-task: {[round(r, 4) for r in retention]})")

    @torch.no_grad()
    def _eval_task_metrics(self):
        """Evaluate on real test data — returns (acc_total, faa, ffm, asa, grouped)."""
        test_set = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4,
                                 worker_init_fn=seed_worker, generator=g)
        self.model.eval()
        y_true, y_pred = [], []
        for _, (_, _, x, y) in enumerate(test_loader):
            x, y = x.cuda(), y.cuda()
            features = self.model.get_features(x)
            logits = self.model.classifier(features, return_dict=True)["logits"]
            y_pred.append(logits.argmax(1).cpu().numpy())
            y_true.append(y.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc_total, grouped = accuracy(y_pred.T, y_true, self._class_increments)
        provisional = self.mlp_matrix + [grouped]
        n = len(provisional)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                mat[i, j] = provisional[i][j]
        faa, ffm, _, asa = compute_metrics(mat)
        return float(acc_total), float(faa), float(ffm), float(asa), grouped

    def align_classifier(self):
        self.compute_multivariate_normal()

        if self._cur_task == 0:
            return  # No alignment needed for first task

        # ----------------------------
        # 1) Synthetic sampler — fresh draw each call to avoid fixed-feature exploitation
        # ----------------------------
        num_sampled_pcls = self._config.get("train_ca_samples_per_class", 100)
        # "covariance" — MultivariateNormal with full stored [D,D] covariance (correlated dims)
        # "diagonal"   — independent Normal per dim using stored [D] variances
        # "variance"   — isotropic Normal using stored scalar variance per class
        sample_method = self._config.get("train_ca_sample_method", "covariance")

        # Precompute GPU sampling tensors once — avoids per-call Cholesky and CPU→GPU transfers.
        # _class_means: [C, D],  _class_covs: [C,D] | [C] | [C,D,D] depending on sample_method.
        C = self._total_classes
        D = self._class_means.shape[1]
        _all_means = self._class_means[:C].cuda().float()           # [C, D]
        _labels_base = torch.arange(C, device=_all_means.device)   # [C]

        if sample_method == "diagonal":
            _all_stds = torch.sqrt(self._class_covs[:C].cuda().float())  # [C, D]
        elif sample_method == "variance":
            _all_stds = torch.sqrt(self._class_covs[:C].cuda().float()).unsqueeze(1)  # [C, 1]
        else:  # "covariance" — Cholesky once for all classes: [C, D, D]
            try:
                _all_L = torch.linalg.cholesky(self._class_covs[:C].cuda().float())  # [C, D, D]
            except Exception as e:
                logging.warning(f"[Alignment] Batched Cholesky failed ({e}), falling back to per-class")
                _all_L = None

        def sample_features(n=None):
            k = n if n is not None else num_sampled_pcls
            try:
                z = torch.randn(C, k, D, device=_all_means.device)   # [C, k, D]
                if sample_method in ("diagonal", "variance"):
                    feats = _all_means.unsqueeze(1) + _all_stds.unsqueeze(1) * z  # [C, k, D]
                else:  # covariance: mean + L @ z^T, broadcast over batch
                    if _all_L is not None:
                        feats = _all_means.unsqueeze(1) + (_all_L @ z.transpose(1, 2)).transpose(1, 2)
                    else:  # per-class fallback if batched Cholesky failed
                        feats = torch.stack([
                            MultivariateNormal(_all_means[c], self._class_covs[c].cuda().float()).sample((k,))
                            for c in range(C)
                        ])  # [C, k, D]
                feats = feats.reshape(C * k, D)                       # [C*k, D]
                feats = F.layer_norm(feats, (D,))
                if torch.isnan(feats).any():
                    raise ValueError("NaN in sampled features")
                labels = _labels_base.repeat_interleave(k)            # [C*k]
                return feats, labels
            except Exception as e:
                logging.warning(f"[Alignment] Failed to sample: {e}")
                return None, None

        logging.info(f"[Alignment] Synthetic sampler ready: {C * num_sampled_pcls} samples total, {C} classes")

        # ----------------------------
        # 2) Head → class range mapping and stitched logits
        # ----------------------------
        head_class_ranges = {t: (rng[0], rng[1]) for t, rng in enumerate(self._class_increments[: self._cur_task + 1])}
        num_global_classes = self._total_classes

        @torch.no_grad()
        def stitched_logits_from_heads(features):
            B = features.size(0)
            out = features.new_full((B, num_global_classes), float("-inf"))
            for t, head in enumerate(self.model.classifier.heads):
                if t > self._cur_task:
                    continue
                c_lo, c_hi = head_class_ranges[t]
                logits_t = head(features)  # [B, C_t]
                out[:, c_lo: c_hi + 1] = logits_t
            return out

        # ----------------------------
        # 4) Macro-class accuracy helper (vectorized via scatter_add)
        #    Balanced synthetic data → equivalent to micro accuracy, but macro is
        #    kept for correctness under real imbalance.
        # ----------------------------
        @torch.no_grad()
        def macro_class_accuracy(logits, labels):
            preds = logits.argmax(1)
            correct = (preds == labels).float()
            counts = torch.zeros(num_global_classes, device=labels.device)
            hits = torch.zeros(num_global_classes, device=labels.device)
            counts.scatter_add_(0, labels, torch.ones_like(correct))
            hits.scatter_add_(0, labels, correct)
            present = counts > 0
            per_class = (hits[present] / counts[present]).cpu().tolist()
            return float(np.mean(per_class)) if per_class else 0.0, per_class

        # ----------------------------
        # 5) Route to alignment method
        # ----------------------------
        ca_method = self._config.get("train_ca_method", "nes")
        t_align_start = time.time()

        if ca_method == "nes":
            # ── Build genome ──────────────────────────────────────────────────
            head_params = []
            head_param_slices = {}
            offset = 0

            for task_idx in range(self._cur_task + 1):
                head = self.model.classifier.heads[task_idx]
                head_start = offset
                for p in head.parameters():
                    head_params.append(p.data.detach().cpu().numpy().flatten())
                    offset += p.data.numel()
                head_param_slices[task_idx] = (head_start, offset)

            theta_np = np.concatenate(head_params).astype(np.float32)
            D_total = theta_np.size

            # ── Diagnostics ───────────────────────────────────────────────────
            total_model_params = count_parameters(self.model)
            backbone_params = count_parameters(self.model.backbone)
            K = self._config.get("train_ca_samples_per_class", 512)
            gaussian_mem_mb = (self._class_means.numel() + self._class_covs.numel()) * 4 / 1024 ** 2
            adapter_mem_mb = sum(
                os.path.getsize(self.backbone_checkpoint(t)) / 1024 ** 2
                for t in range(self._cur_task + 1)
                if os.path.exists(self.backbone_checkpoint(t))
            )
            logging.info("[Alignment][NES] === Alignment Cost Report ===")
            logging.info(f"[Alignment][NES]   Task: {self._cur_task} | Classes seen: {self._total_classes}")
            logging.info(f"[Alignment][NES]   Synthetic samples: K={K}/class × {self._total_classes} classes = {K * self._total_classes:,} total")
            logging.info(f"[Alignment][NES]   Head params (theta): {D_total:,} / model total {total_model_params:,} ({D_total * 100 / total_model_params:.3f}%)")
            logging.info(f"[Alignment][NES]   Backbone params: {backbone_params:,}")
            logging.info(f"[Alignment][NES]   Gaussian statistics memory: {gaussian_mem_mb:.2f} MB ({self._total_classes} classes × {self.model.feature_dim}-dim)")
            logging.info(f"[Alignment][NES]   Stored adapter checkpoints: {adapter_mem_mb:.2f} MB ({self._cur_task + 1} tasks)")

            # ── Move theta to GPU — no numpy↔cuda transfers in hot path ──────
            theta_gpu = torch.from_numpy(theta_np).cuda()
            label_smoothing = float(self._config.get("train_ca_nes_label_smoothing", 0.1))

            # ── Batched population objective ──────────────────────────────────
            # Replaces the Python `for i in range(pop)` loop with GPU einsum.
            # Memory is controlled by two chunk sizes:
            #   POP_CHUNK  — population members per GPU call (controls [N,F,C] peak)
            #   FEAT_CHUNK — feature rows per mini-batch    (controls [N,F,C] peak)
            # Peak tensor: [POP_CHUNK, FEAT_CHUNK, C_total] × 4 bytes.
            # Defaults (16, 2048) → ~25 MB for C_total=200.
            POP_CHUNK  = 16
            FEAT_CHUNK = 2048

            def batched_objective(
                params_batch: torch.Tensor,  # [N, D_total]
                features: torch.Tensor,       # [B, D_feat]
                labels: torch.Tensor,         # [B]
            ) -> torch.Tensor:               # [N]  CE loss per candidate
                N = params_batch.size(0)
                B_total = features.size(0)
                total_nll = torch.zeros(N, device=features.device)
                for f0 in range(0, B_total, FEAT_CHUNK):
                    fc = features[f0: f0 + FEAT_CHUNK]  # [F, D]
                    lc = labels  [f0: f0 + FEAT_CHUNK]  # [F]
                    F_sz = fc.size(0)
                    logits = torch.zeros(N, F_sz, num_global_classes, device=features.device)
                    for t in range(self._cur_task + 1):
                        c_lo, c_hi = head_class_ranges[t]
                        s_lo, s_hi = head_param_slices[t]
                        n_cls = c_hi - c_lo + 1
                        # W[n,c,d] × fc[f,d] → logits[n,f,c]
                        W = params_batch[:, s_lo:s_hi].view(N, n_cls, -1)  # [N, C_t, D]
                        logits[:, :, c_lo: c_hi + 1] = torch.einsum("fd,ncd->nfc", fc, W)
                    nll = F.cross_entropy(
                        logits.reshape(N * F_sz, num_global_classes),
                        lc.repeat(N),
                        reduction="none",
                        label_smoothing=label_smoothing,
                    ).view(N, F_sz).sum(dim=1)
                    total_nll += nll
                return total_nll / B_total

            def eval_population(
                params_batch: torch.Tensor,
                features: torch.Tensor,
                labels: torch.Tensor,
            ) -> torch.Tensor:
                N = params_batch.size(0)
                losses = torch.empty(N, device=features.device)
                for n0 in range(0, N, POP_CHUNK):
                    n1 = min(n0 + POP_CHUNK, N)
                    losses[n0:n1] = batched_objective(params_batch[n0:n1], features, labels)
                return losses

            # ── Sigma scheduling ──────────────────────────────────────────────
            base_sigma_init  = float(self._config.get("train_ca_nes_sigma_init",  1e-3))
            base_sigma_final = float(self._config.get("train_ca_nes_sigma_final", 1e-4))

            def get_sigma(iteration, total_iters):
                return float(base_sigma_init * (base_sigma_final / base_sigma_init) ** (
                    iteration / max(total_iters - 1, 1)
                ))

            # ── NES loop ──────────────────────────────────────────────────────
            lr    = float(self._config.get("train_ca_nes_lr",         0.01))
            iters = int  (self._config.get("train_ca_nes_iterations",  100))
            pop   = int  (self._config.get("train_ca_nes_popsize",     200))

            logging.info(f"[Alignment][NES] T={iters}, P={pop}, K={K}, lr={lr}")
            logging.info(f"[Alignment][NES] sigma schedule: init={base_sigma_init:.3e}, final={base_sigma_final:.3e}")
            logging.info(f"[Alignment][NES] pop chunks: {POP_CHUNK}, feat chunks: {FEAT_CHUNK}")

            best_theta_gpu = theta_gpu.clone()
            best_loss = float("inf")
            iter_times = []

            for it in range(iters):
                t_iter_start = time.time()

                # Fresh feature batch every iteration — unbiased gradient estimate
                _feats, _labels = sample_features()
                if _feats is None:
                    continue
                sigma = get_sigma(it, iters)

                # Antithetic perturbations — all on GPU, no numpy round-trips
                eps_gpu   = torch.randn(pop, D_total, device="cuda", dtype=torch.float32)
                theta_pos = theta_gpu.unsqueeze(0) + sigma * eps_gpu   # [P, D]
                theta_neg = theta_gpu.unsqueeze(0) - sigma * eps_gpu   # [P, D]

                with torch.no_grad():
                    losses_2p = eval_population(
                        torch.cat([theta_pos, theta_neg], dim=0), _feats, _labels
                    )  # [2P]

                iter_times.append(time.time() - t_iter_start)

                losses_pos_t = losses_2p[:pop]   # [P]
                losses_neg_t = losses_2p[pop:]   # [P]

                # Fitness shaping: rank-normalise to [-0.5, 0.5]
                raw_delta = (losses_pos_t - losses_neg_t).cpu().numpy()
                ranks = np.argsort(np.argsort(raw_delta)).astype(np.float32)
                shaped = torch.from_numpy(ranks / (pop - 1) - 0.5).to(eps_gpu)  # [P]

                # NES gradient estimate + plain SGD step
                # No /sigma: fitness shaping already gives scale-invariant signal;
                # dividing by sigma (1e-3) would amplify the gradient 1000×.
                grad = (shaped[:, None] * eps_gpu).mean(dim=0)  # [D]
                theta_gpu = theta_gpu - lr * grad

                # Track best on the same batch used for the gradient
                with torch.no_grad():
                    cur_loss = eval_population(theta_gpu.unsqueeze(0), _feats, _labels)[0].item()
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_theta_gpu = theta_gpu.clone()

                if it % 10 == 0 or it == iters - 1:
                    logging.info(
                        f"[Alignment][NES-diag] iter {it}: best_loss={best_loss:.6f}, sigma={sigma:.3e}"
                    )

            # ── Apply best solution ───────────────────────────────────────────
            param_idx = 0
            for t in range(self._cur_task + 1):
                head = self.model.classifier.heads[t]
                for p in head.parameters():
                    sz = p.numel()
                    p.data = best_theta_gpu[param_idx: param_idx + sz].view(p.shape).clone()
                    param_idx += sz

            final_features, final_labels = sample_features()
            with torch.no_grad():
                logits = stitched_logits_from_heads(final_features)
                final_macro, _ = macro_class_accuracy(logits, final_labels)

            t_align_total = time.time() - t_align_start
            self._total_align_time += t_align_total
            actual_iters = len(iter_times)

            logging.info("[Alignment][NES] === Convergence & Cost Summary ===")
            logging.info(f"[Alignment][NES]   Iterations run: {actual_iters} / {iters}")
            logging.info(f"[Alignment][NES]   Synthetic CE loss: {best_loss:.4f} → final macro-acc: {final_macro:.4f}")
            logging.info(f"[Alignment][NES]   Wall-clock: total={t_align_total:.1f}s, mean/iter={np.mean(iter_times):.2f}s")
            if hasattr(self, "_train_time") and self._train_time > 0:
                logging.info(f"[Alignment][NES]   Alignment overhead: {t_align_total:.1f}s / train {self._train_time:.1f}s = {t_align_total / self._train_time:.2f}x")
            if self._total_classes == self._total_classnum:
                logging.info(f"[Alignment][NES]   TOTAL across all tasks — alignment: {self._total_align_time:.1f}s, training: {self._total_train_time:.1f}s, ratio: {self._total_align_time / max(self._total_train_time, 1):.2f}x")

        elif ca_method == "ce":
            # ── CE gradient-based alignment ───────────────────────────────────
            ce_samples_pcls = int(self._config.get("train_ce_samples_per_class", num_sampled_pcls))
            ce_epochs = int(self._config.get("train_ce_epochs", 20))
            ce_lr = float(self._config.get("train_ce_lr", 1e-3))

            all_head_params = []
            for t in range(self._cur_task + 1):
                for p in self.model.classifier.heads[t].parameters():
                    p.requires_grad_(True)
                    all_head_params.append(p)

            optimizer = torch.optim.Adam(all_head_params, lr=ce_lr)

            logging.info(f"[Alignment][CE] epochs={ce_epochs}, lr={ce_lr}, samples/class={ce_samples_pcls}")

            for epoch in range(ce_epochs):
                features, labels = sample_features(n=ce_samples_pcls)
                if features is None:
                    break
                optimizer.zero_grad()
                logits = torch.cat(
                    [self.model.classifier.heads[t](features) for t in range(self._cur_task + 1)], dim=1
                )
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 5 == 0 or epoch == ce_epochs - 1:
                    acc = (logits.detach().argmax(1) == labels).float().mean().item()
                    logging.info(
                        f"[Alignment][CE] epoch {epoch + 1}/{ce_epochs}: "
                        f"loss={loss.item():.4f}, train_acc={acc:.4f}"
                    )

            t_align_total = time.time() - t_align_start
            self._total_align_time += t_align_total
            logging.info(f"[Alignment][CE] Done | Wall-clock: {t_align_total:.1f}s")
            if self._total_classes == self._total_classnum:
                logging.info(f"[Alignment][CE] TOTAL — alignment: {self._total_align_time:.1f}s, training: {self._total_train_time:.1f}s, ratio: {self._total_align_time / max(self._total_train_time, 1):.2f}x")

    def merge(self):
        logging.info(f"[Merging] Task {self._cur_task}")

        if self._cur_task > 0:
            base_params = torch.load(self.backbone_checkpoint(-1))
            logging.info(f"[Merging] Method {self._config['model_merge']}")

            if self._config.get("model_merge_incremental", False):
                task_params = [
                    torch.load(self.merged_checkpoint(self._cur_task - 1)),
                    torch.load(self.backbone_checkpoint(self._cur_task)),
                ]
            else:
                task_params = [
                    torch.load(self.backbone_checkpoint(task))
                    for task in range(self._cur_task + 1)
                ]
            logging.info(f"[Merging] Loaded {len(task_params)} tasks for merging")

            backbone_params = merge(
                base_params,
                task_params,
                method=self._config["model_merge"],
                lamb=self._config["model_merge_coef"],
                topk=self._config["model_merge_topk"],
            )
            self.load_backbone(backbone_params)

        logging.info(f"[Merging] Save merged checkpoint for task {self._cur_task}")
        self._save(self.model.get_backbone_trainable_params(), self.merged_checkpoint(self._cur_task))

    def load_backbone(self, backbone_params):
        self.model.backbone.load_state_dict(backbone_params, strict=False)

    def prefix(self):
        prefix_parts = [
            str(self._config['seed']),
            self._config['dataset_name'],
            self._config['model_backbone'],
        ]
        train_prefix = self._config.get("train_prefix", "")
        if train_prefix:
            prefix_parts.append(train_prefix)
        return "_".join(prefix_parts)

    def _ckpt_path(self, filename):
        ckpt_dir = CHECKPOINT_ABLATION_DIR if self._config.get("train_ablation", False) else CHECKPOINT_DIR
        return os.path.join(ckpt_dir, filename)

    def backbone_checkpoint(self, task=-1):
        filename = f"{self.prefix()}_backbone" + (
            f"_{task}.pt" if task >= 0 else "_base.pt"
        )
        return self._ckpt_path(filename)

    def model_checkpoint(self):
        filename = f"{self.prefix()}_model.pt"
        return self._ckpt_path(filename)

    def merged_checkpoint(self, task):
        filename = f"{self.prefix()}_merged_{self._config['model_merge']}_{task}.pt"
        return self._ckpt_path(filename)



def run_single_experiment(dataset_name, config_name, experiment_config, seed):
    config = copy.deepcopy(BASE_CONFIG)
    config["seed"] = seed

    set_random(seed)

    dataset_num_task, dataset_init_cls, dataset_increment = DATA_TABLE[dataset_name][0]
    config.update({
        "dataset_name": dataset_name,
        "dataset_num_task": dataset_num_task,
        "dataset_init_cls": dataset_init_cls,
        "dataset_increment": dataset_increment,
    })

    data_manager = DataManager(
        dataset_name,
        True,
        seed,
        dataset_init_cls,
        dataset_increment,
        False,
    )
    logging.info(f"Dataset Training Size: {data_manager.train_set_size}")

    config.update(experiment_config)

    if dataset_name == "imageneta":
        config["train_batch_size"] = 48

    result = {
        "mlp_faa": 0.0, "mlp_ffm": 0.0, "mlp_asa": 0.0,
        "pre_align_faa": None, "pre_align_ffm": None, "pre_align_asa": None,
    }
    try:
        logging.info("Configuration:")
        for key, value in config.items():
            logging.info(f"  {key}: {value}")

        config["config_name"] = config_name
        learner = Learner(config)
        learner.learn(data_manager)

        result["mlp_faa"] = learner._mlp_faa
        result["mlp_ffm"] = learner._mlp_ffm
        result["mlp_asa"] = learner._mlp_asa
        result["pre_align_faa"] = learner._pre_align_faa
        result["pre_align_ffm"] = learner._pre_align_ffm
        result["pre_align_asa"] = learner._pre_align_asa

        del learner
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        import traceback
        logging.error(f"[Experiment {dataset_name}_{config_name}] {type(e).__name__}: {e}")
        logging.error(traceback.format_exc())

    # ── Per-seed result block — grep for [RESULT] to extract ─────────────────
    has_align = result["pre_align_faa"] is not None
    logging.info("")
    logging.info("[RESULT] " + "─" * 60)
    logging.info(f"[RESULT]  method  : {config_name}")
    logging.info(f"[RESULT]  dataset : {dataset_name}")
    logging.info(f"[RESULT]  seed    : {seed}")
    if has_align:
        logging.info(f"[RESULT]  FAA_pre : {result['pre_align_faa']:.4f}  (post-merge)")
        logging.info(f"[RESULT]  FFM_pre : {result['pre_align_ffm']:.4f}  (post-merge)")
        logging.info(f"[RESULT]  ASA_pre : {result['pre_align_asa']:.4f}  (post-merge)")
    logging.info(f"[RESULT]  FAA     : {result['mlp_faa']:.4f}  {'(post-align)' if has_align else ''}")
    logging.info(f"[RESULT]  FFM     : {result['mlp_ffm']:.4f}  {'(post-align)' if has_align else ''}")
    logging.info(f"[RESULT]  ASA     : {result['mlp_asa']:.4f}  {'(post-align)' if has_align else ''}")
    logging.info("[RESULT] " + "─" * 60)

    return result


def run_experiments():
    seeds = BASE_CONFIG["seed"]

    # Build the final set of named experiment configs.
    # If PARAM_SWEEP has entries, expand each base experiment over the Cartesian
    # product of all active sweep parameters (grid search).
    if PARAM_SWEEP:
        params = list(PARAM_SWEEP.keys())
        values = list(PARAM_SWEEP.values())
        ablation_mode = BASE_CONFIG.get("train_ablation", False)
        experiment_configs = {}
        for base_name, base_cfg in EXPERIMENT_CONFIGS.items():
            for combo in itertools.product(*values):
                suffix = "__".join(f"{p}_{v}" for p, v in zip(params, combo))
                run_name = f"{base_name}__{suffix}"
                cfg = copy.deepcopy(base_cfg)
                for p, v in zip(params, combo):
                    cfg[p] = v
                if ablation_mode:
                    base_prefix = cfg.get("train_prefix", base_name)
                    cfg["train_prefix"] = "__".join([base_prefix] + params)
                experiment_configs[run_name] = cfg
        print(f"\nParam grid: {dict(zip(params, values))}")
        print(f"Generated {len(experiment_configs)} run(s): {list(experiment_configs.keys())}")
    else:
        experiment_configs = EXPERIMENT_CONFIGS

    for dataset_name in DATA_TABLE.keys():
        print(f"\n{'='*60}")
        print(f"Starting experiments for dataset: {dataset_name}")
        print(f"{'='*60}")

        dataset_results = {}

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        for config_name, experiment_config in experiment_configs.items():
            ablation = BASE_CONFIG.get("train_ablation", False) or experiment_config.get("train_ablation", False)
            active_log_dir = LOG_ABLATION_DIR if ablation else LOG_DIR
            dir_path = os.path.join(active_log_dir, dataset_name)
            os.makedirs(dir_path, exist_ok=True)
            logfilename = os.path.join(dir_path, config_name + ".log")
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(filename)s] => %(message)s",
                handlers=[
                    logging.FileHandler(filename=logfilename),
                    logging.StreamHandler(sys.stdout),
                ],
                force=True,
            )

            for seed in seeds:
                logging.info("\n" + "=" * 80)
                logging.info(f"Starting experiment: {dataset_name} - {config_name} - seed {seed}")
                t0 = time.time()
                result = run_single_experiment(dataset_name, config_name, experiment_config, seed)
                logging.info(f"Experiment {dataset_name}_{config_name}_seed{seed} time: {time.time() - t0:.2f}s")

                if config_name not in dataset_results:
                    dataset_results[config_name] = {
                        "seeds": [],
                        "mlp_faa": [], "mlp_ffm": [], "mlp_asa": [],
                        "pre_align_faa": [], "pre_align_ffm": [], "pre_align_asa": [],
                    }

                dataset_results[config_name]["seeds"].append(seed)
                dataset_results[config_name]["mlp_faa"].append(result["mlp_faa"])
                dataset_results[config_name]["mlp_ffm"].append(result["mlp_ffm"])
                dataset_results[config_name]["mlp_asa"].append(result["mlp_asa"])
                if result["pre_align_faa"] is not None:
                    dataset_results[config_name]["pre_align_faa"].append(result["pre_align_faa"])
                    dataset_results[config_name]["pre_align_ffm"].append(result["pre_align_ffm"])
                    dataset_results[config_name]["pre_align_asa"].append(result["pre_align_asa"])

            vals = dataset_results[config_name]
            has_align = len(vals["pre_align_faa"]) > 0
            # ── Aggregate result block — grep for [AGGREGATE] to extract ──────
            logging.info("")
            logging.info("[AGGREGATE] " + "─" * 57)
            logging.info(f"[AGGREGATE]  method  : {config_name}")
            logging.info(f"[AGGREGATE]  dataset : {dataset_name}")
            logging.info(f"[AGGREGATE]  seeds   : {vals['seeds']}")
            if vals["mlp_faa"]:
                faa_arr = np.array(vals["mlp_faa"])
                ffm_arr = np.array(vals["mlp_ffm"])
                asa_arr = np.array(vals["mlp_asa"])
                if has_align:
                    pre_faa = np.array(vals["pre_align_faa"])
                    pre_ffm = np.array(vals["pre_align_ffm"])
                    pre_asa = np.array(vals["pre_align_asa"])
                    logging.info(f"[AGGREGATE]  FAA_pre : {np.mean(pre_faa):.4f} ± {np.std(pre_faa):.4f}  {[round(v,4) for v in pre_faa.tolist()]}  (post-merge)")
                    logging.info(f"[AGGREGATE]  FFM_pre : {np.mean(pre_ffm):.4f} ± {np.std(pre_ffm):.4f}  {[round(v,4) for v in pre_ffm.tolist()]}  (post-merge)")
                    logging.info(f"[AGGREGATE]  ASA_pre : {np.mean(pre_asa):.4f} ± {np.std(pre_asa):.4f}  {[round(v,4) for v in pre_asa.tolist()]}  (post-merge)")
                logging.info(f"[AGGREGATE]  FAA     : {np.mean(faa_arr):.4f} ± {np.std(faa_arr):.4f}  {[round(v,4) for v in faa_arr.tolist()]}  {'(post-align)' if has_align else ''}")
                logging.info(f"[AGGREGATE]  FFM     : {np.mean(ffm_arr):.4f} ± {np.std(ffm_arr):.4f}  {[round(v,4) for v in ffm_arr.tolist()]}  {'(post-align)' if has_align else ''}")
                logging.info(f"[AGGREGATE]  ASA     : {np.mean(asa_arr):.4f} ± {np.std(asa_arr):.4f}  {[round(v,4) for v in asa_arr.tolist()]}  {'(post-align)' if has_align else ''}")
                if has_align:
                    logging.info(f"[AGGREGATE]  ΔFAA    : {np.mean(faa_arr-pre_faa):+.4f} ± {np.std(faa_arr-pre_faa):.4f}  (align effect)")
                    logging.info(f"[AGGREGATE]  ΔFFM    : {np.mean(ffm_arr-pre_ffm):+.4f} ± {np.std(ffm_arr-pre_ffm):.4f}  (align effect)")
                    logging.info(f"[AGGREGATE]  ΔASA    : {np.mean(asa_arr-pre_asa):+.4f} ± {np.std(asa_arr-pre_asa):.4f}  (align effect)")
            logging.info("[AGGREGATE] " + "─" * 57)

        logging.info("=" * 80 + "\n")


if __name__ == "__main__":
    start_time = time.time()
    run_experiments()
    print(f"Total time: {time.time() - start_time:.2f}s")

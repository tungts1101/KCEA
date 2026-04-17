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
        torch.save(self.model.get_backbone_trainable_params(), self.backbone_checkpoint())
        self._mlp_faa = 0.0
        self._mlp_ffm = 0.0
        self._mlp_asa = 0.0
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
            self.train()
            t_eval = time.time()
            self.eval()
            self._total_eval_time += time.time() - t_eval
            self.after_task()
            self._peak_storage_bytes = max(self._peak_storage_bytes, self._current_storage_bytes())

        torch.save(
            self.model.state_dict(),
            self.model_checkpoint(),
        )

        self._log_final_summary()

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
        final_bytes = self._current_storage_bytes()
        pfx = self.prefix()
        ckpt_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith(pfx)] if os.path.isdir(CHECKPOINT_DIR) else []

        def _bytes_matching(keyword):
            return sum(
                os.path.getsize(os.path.join(CHECKPOINT_DIR, f))
                for f in ckpt_files if keyword in f
            )

        backbone_bytes  = _bytes_matching("_backbone_")
        merged_bytes    = _bytes_matching("_merged_")
        head_bytes      = sum(
            os.path.getsize(os.path.join(CHECKPOINT_DIR, f))
            for f in ckpt_files
            if "_head_" in f and "_alignment" not in f
        )
        align_head_bytes = _bytes_matching("_alignment")

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
        model_weight_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        total_disk_bytes   = backbone_bytes + merged_bytes + head_bytes + align_head_bytes
        num_tasks_run      = self._cur_task + 1

        logging.info(f"[Summary] ── Storage ─────────────────────────────────────────────────────")
        logging.info(f"[Summary]   Model weights in RAM (final task)      : {_mb(model_weight_bytes):.1f} MB")
        logging.info(f"[Summary]   Peak RAM (weights + statistics)        : {_mb(model_weight_bytes + gauss_bytes):.1f} MB")
        logging.info(f"[Summary]   On-disk checkpoints                    : {_mb(total_disk_bytes):.1f} MB  ({len(ckpt_files)} files)")
        logging.info(f"[Summary]     Task adapter checkpoints (_backbone_T)  : {_mb(backbone_bytes):.1f} MB  ({num_tasks_run} files)")
        logging.info(f"[Summary]     Merged backbone checkpoints (_merged_T) : {_mb(merged_bytes):.1f} MB  ({num_tasks_run} files)")
        logging.info(f"[Summary]     Classifier heads (_head_T)              : {_mb(head_bytes):.1f} MB  ({num_tasks_run} files)")
        if align_head_bytes > 0:
            n_align = len([f for f in ckpt_files if "_alignment" in f])
            logging.info(f"[Summary]     Post-alignment heads (_head_T_align)  : {_mb(align_head_bytes):.1f} MB  ({n_align} files)")
        logging.info(f"[Summary]   RAM statistics (means/covs/etc.)       : {_mb(gauss_bytes):.1f} MB  ({self._total_classes} classes × {self.model.feature_dim}-dim)")
        logging.info("[Summary]")

        # Parameters
        last_trainable = self._task_trainable_params[-1]["total"] if self._task_trainable_params else 0

        logging.info(f"[Summary] ── Trainable parameters ─────────────────────────────────────────")
        logging.info(f"[Summary]   Total model params (final task)        : {total_params:,}")
        logging.info(f"[Summary]   Peak trainable (any task)              : {peak_trainable:,}  ({peak_trainable*100/max(peak_total_at_task,1):.2f}% of {peak_total_at_task:,} at that task)")
        logging.info(f"[Summary]   Final task trainable                   : {last_trainable:,}  ({last_trainable*100/max(total_params,1):.2f}% of total at final task)")
        logging.info("[Summary]")
        logging.info("=" * 80)

    def before_task(self, task, data_manager):
        self._total_classes = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_task = task

        for clz in range(self._known_classes, self._total_classes):
            self._cls_to_task_idx[clz] = self._cur_task

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

        epochs = self._config["train_epochs"]
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

        torch.save(self.model.get_backbone_trainable_params(), self.backbone_checkpoint(self._cur_task))

        self._train_time = time.time() - t_train_start
        self._total_train_time += self._train_time

        if self._config.get("train_merge", False):
            t_merge = time.time()
            self.merge()
            self._total_merge_time += time.time() - t_merge

        if self._config["train_ca"]:
            self.align_classifier()

    def compute_multivariate_normal(self):
        _t_gauss = time.time()
        logging.info(
            f"[Alignment] Compute class mean and cov for classes {self._known_classes} - {self._total_classes - 1}"
        )
        total_class = self._total_classes
        feature_dim = self.model.feature_dim
        if not hasattr(self, "_class_means") or not hasattr(self, "_class_covs"):
            self._class_means = torch.zeros((total_class, feature_dim))
            self._class_covs = torch.zeros((total_class, feature_dim, feature_dim))
        else:
            new_class_means = torch.zeros((total_class, feature_dim))
            new_class_means[: self._known_classes] = self._class_means
            self._class_means = new_class_means
            new_class_covs = torch.zeros((total_class, feature_dim, feature_dim))
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
            class_cov = torch.cov(features_list.T)

            # Ensure positive definiteness
            min_eigenval = torch.linalg.eigvals(class_cov).real.min()
            reg_term = (abs(min_eigenval.item()) + 1e-3) if min_eigenval <= 0 else 1e-4
            class_cov = class_cov + torch.eye(feature_dim) * reg_term

            # Verify with Cholesky; fall back to diagonal if still not PD
            try:
                torch.linalg.cholesky(class_cov)
            except RuntimeError:
                class_var = features_list.var(dim=0, unbiased=True)
                class_cov = torch.diag(class_var + 1e-3)

            self._class_means[cls_idx, :] = class_mean
            self._class_covs[cls_idx, ...] = class_cov

        self._total_gauss_time += time.time() - _t_gauss

    def _record_eval(self, acc, faa, ffm, asa, grouped):
        """Append grouped accuracies to mlp_matrix, update metrics, and log."""
        self.mlp_matrix.append(grouped)
        self._mlp_faa, self._mlp_ffm, self._mlp_asa = faa, ffm, asa
        logging.info(f"[Evaluation] Task {self._cur_task}: Acc={acc:.2f}, FAA={faa:.2f}, FFM={ffm:.2f}, ASA={asa:.2f}")
        if self._total_classes == self._total_classnum:
            logging.info("[Evaluation] Accuracy matrix:")
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

        def sample_features():
            sampled_data, sampled_label = [], []
            for c_id in range(self._total_classes):
                if hasattr(self, "_class_means") and c_id < self._class_means.shape[0]:
                    try:
                        cls_mean = self._class_means[c_id].cuda().float()
                        cls_cov = self._class_covs[c_id].cuda().float()
                        dist = MultivariateNormal(cls_mean, cls_cov)
                        feats = dist.sample((num_sampled_pcls,))
                        if torch.isnan(feats).any():
                            raise ValueError("NaN in sampled features")
                        feats = F.layer_norm(feats, (feats.shape[-1],))
                        sampled_data.append(feats)
                        sampled_label.extend([c_id] * num_sampled_pcls)
                    except Exception as e:
                        logging.warning(f"[Alignment] Failed to sample from class {c_id}: {e}")
                        continue
            if not sampled_data:
                return None, None
            features = torch.cat(sampled_data, dim=0).float().cuda(non_blocking=True)
            labels = torch.tensor(sampled_label).long().cuda(non_blocking=True)
            return features, labels

        # Validate that sampling works before entering the NES loop
        _check_feats, _ = sample_features()
        if _check_feats is None:
            logging.warning("No samples generated, skipping classifier alignment")
            return
        logging.info(f"[Alignment] Stochastic synthetic sampler ready: {_check_feats.shape[0]} samples/draw, {self._total_classes} classes")

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
        # 4) Build initial genome from all heads
        # ----------------------------
        head_params = []
        param_shapes = []
        head_param_slices = {}
        offset = 0

        for task_idx in range(self._cur_task + 1):
            head = self.model.classifier.heads[task_idx]
            head_start = offset
            for p in head.parameters():
                arr = p.data.detach().cpu().numpy().flatten()
                head_params.append(arr)
                param_shapes.append(p.shape)
                offset += arr.size
            head_end = offset
            head_param_slices[task_idx] = (head_start, head_end)

        theta = np.concatenate(head_params).astype(np.float32)

        # --- Reviewer diagnostics: cost, capacity, convergence info ---
        total_model_params = count_parameters(self.model)
        backbone_params = count_parameters(self.model.backbone)
        K = self._config.get("train_ca_samples_per_class", 512)
        gaussian_mem_mb = (self._class_means.numel() + self._class_covs.numel()) * 4 / 1024 ** 2
        adapter_mem_mb = sum(
            os.path.getsize(self.backbone_checkpoint(t)) / 1024 ** 2
            for t in range(self._cur_task + 1)
            if os.path.exists(self.backbone_checkpoint(t))
        )
        logging.info("[Alignment] === NES Alignment Cost Report ===")
        logging.info(f"[Alignment]   Task: {self._cur_task} | Classes seen: {self._total_classes}")
        logging.info(f"[Alignment]   Synthetic samples: K={K}/class × {self._total_classes} classes = {all_features.shape[0]:,} total")
        logging.info(f"[Alignment]   Head params (theta): {theta.size:,} / model total {total_model_params:,} ({theta.size * 100 / total_model_params:.3f}%)")
        logging.info(f"[Alignment]   Backbone params: {backbone_params:,}")
        logging.info(f"[Alignment]   Gaussian statistics memory: {gaussian_mem_mb:.2f} MB ({self._total_classes} classes × {self.model.feature_dim}-dim)")
        logging.info(f"[Alignment]   Stored adapter checkpoints: {adapter_mem_mb:.2f} MB ({self._cur_task + 1} tasks)")

        # ----------------------------
        # 5) Per-head trust region strengths
        # ----------------------------
        lambda_cfg = self._config.get("train_ca_lambda_head", {})
        default_old = float(self._config.get("train_ca_lambda_old_default", 1e-4))
        default_cur = float(self._config.get("train_ca_lambda_cur_default", 1e-5))

        lambda_task = {}
        for t in range(self._cur_task + 1):
            lam = lambda_cfg.get(str(t), lambda_cfg.get(t, default_cur if t == self._cur_task else default_old))
            lambda_task[t] = float(lam)

        # ----------------------------
        # 6) Macro-class accuracy helper (vectorized via scatter_add)
        #    Synthetic data is balanced (K samples/class), so this is equivalent
        #    to micro accuracy — but macro is kept for correctness under real imbalance.
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
        # 7) Objective (scalar loss to minimize)
        # ----------------------------
        nes_objective = self._config.get("train_ca_nes_objective", "acc")  # "acc" or "ce"

        def objective_function(params_flat: np.ndarray) -> float:
            try:
                # Fresh sample every evaluation to prevent exploiting fixed features
                features, labels = sample_features()
                if features is None:
                    return 1.0

                param_idx = 0
                original_params_snap = {}
                for task_idx in range(self._cur_task + 1):
                    head = self.model.classifier.heads[task_idx]
                    original_params_snap[task_idx] = []
                    for _, p in head.named_parameters():
                        original_params_snap[task_idx].append(p.data.clone())
                        sz = p.numel()
                        new_data = params_flat[param_idx:param_idx + sz]
                        param_idx += sz
                        p.data = torch.from_numpy(new_data).float().cuda().view(p.shape)

                with torch.no_grad():
                    logits = stitched_logits_from_heads(features)
                    # NES maximises fitness; loss = -fitness
                    if nes_objective == "ce":
                        fitness = -F.cross_entropy(logits, labels).item()
                    else:  # "acc"
                        macro_acc, _ = macro_class_accuracy(logits, labels)
                        fitness = macro_acc

                loss = -fitness

                for task_idx in range(self._cur_task + 1):
                    head = self.model.classifier.heads[task_idx]
                    for p, w in zip(head.parameters(), original_params_snap[task_idx]):
                        p.data = w

                return float(loss)

            except Exception as e:
                logging.warning(f"[Alignment] Error in objective function: {e}")
                return 1.0

        # ----------------------------
        # 8) Per-task sigma scheduling
        #    sigma_base decays exponentially; per-task sigma scales by f(lambda_k) = 1/sqrt(lambda_k)
        #    so tightly constrained (high lambda) tasks get less noise, free tasks get more.
        # ----------------------------
        base_sigma_init = float(self._config.get("train_ca_nes_sigma_init", 1e-3))
        base_sigma_final = float(self._config.get("train_ca_nes_sigma_final", 1e-4))
        sigma_min = float(self._config.get("train_ca_nes_sigma_min", 1e-5))
        sigma_max = float(self._config.get("train_ca_nes_sigma_max", 1e-2))
        lambda_eps = float(self._config.get("train_ca_nes_lambda_eps", 1e-8))

        def importance(lam):
            """f(lambda) = 1 / sqrt(max(lambda, eps)) — maps trust-region strength to sigma scale."""
            return 1.0 / np.sqrt(max(float(lam), lambda_eps))

        def get_sigma_vec(iteration, total_iters):
            # Exponential decay: sigma_init * (sigma_final/sigma_init)^(it / max(T-1, 1))
            sigma_base = base_sigma_init * (base_sigma_final / base_sigma_init) ** (
                iteration / max(total_iters - 1, 1)
            )
            sigma = np.empty_like(theta)
            for k in range(self._cur_task + 1):
                s, e = head_param_slices[k]
                sigma[s:e] = np.clip(sigma_base * importance(lambda_task[k]), sigma_min, sigma_max)
            return sigma.astype(np.float32)

        # ----------------------------
        # 9) NES loop
        # ----------------------------
        lr = float(self._config.get("train_ca_nes_lr", 0.05))
        iters = int(self._config.get("train_ca_nes_iterations", 50))
        pop = int(self._config.get("train_ca_nes_popsize", 30))

        logging.info(f"[Alignment][NES] T={iters}, P={pop}, K={K}, lr={lr}, objective={nes_objective}")
        logging.info(f"[Alignment][NES] sigma schedule: init={base_sigma_init:.3e}, final={base_sigma_final:.3e}, clip=[{sigma_min:.1e},{sigma_max:.1e}]")
        logging.info(f"[Alignment][NES] per-task sigma scale f(λ): " + ", ".join(
            f"T{k}={importance(lambda_task[k]):.3f}(λ={lambda_task[k]:.1e})" for k in range(self._cur_task + 1)
        ))
        logging.info(f"[Alignment][NES] Forward passes per iteration: 2P+1={2*pop+1} | Max total: {(2*pop+1)*iters:,}")

        best_theta = theta.copy()
        best_loss = objective_function(best_theta)
        no_improve_loss = 0
        patience = int(self._config.get("train_ca_nes_patience", 30))
        iter_times = []
        t_align_start = time.time()

        for it in range(iters):
            t_iter_start = time.time()
            sigma_vec = get_sigma_vec(it, iters)

            eps = np.random.randn(pop, theta.size).astype(np.float32)

            losses_pos = np.empty(pop, dtype=np.float32)
            losses_neg = np.empty(pop, dtype=np.float32)
            for i in range(pop):
                step = sigma_vec * eps[i]
                losses_pos[i] = objective_function(theta + step)
                losses_neg[i] = objective_function(theta - step)

            iter_times.append(time.time() - t_iter_start)

            deltaL = (losses_pos - losses_neg)[:, None]
            grad = (deltaL * (eps / np.maximum(sigma_vec, 1e-12))).mean(axis=0) / 2.0
            theta = theta - lr * grad.astype(np.float32)

            cur_loss = objective_function(theta)
            if cur_loss < best_loss - 1e-6:
                best_loss = cur_loss
                best_theta = theta.copy()
                no_improve_loss = 0
            else:
                no_improve_loss += 1

            if it % 10 == 0 or it == iters - 1:
                param_idx = 0
                snap = {}
                for t in range(self._cur_task + 1):
                    head = self.model.classifier.heads[t]
                    snap[t] = [p.data.clone() for p in head.parameters()]
                    for p in head.parameters():
                        sz = p.numel()
                        new_data = best_theta[param_idx:param_idx + sz]
                        param_idx += sz
                        p.data = torch.from_numpy(new_data).float().cuda().view(p.shape)
                log_features, log_labels = sample_features()
                with torch.no_grad():
                    logits = stitched_logits_from_heads(log_features)
                    macro_acc, _ = macro_class_accuracy(logits, log_labels)
                for t in range(self._cur_task + 1):
                    head = self.model.classifier.heads[t]
                    for p, w in zip(head.parameters(), snap[t]):
                        p.data = w

                logging.info(
                    f"[Alignment][NES-diag] iter {it}: macro_acc={macro_acc:.4f}, "
                    f"best_loss={best_loss:.6f}, sigma_cur={sigma_vec[head_param_slices[self._cur_task][0]]:.3e}"
                )

            if patience != -1 and no_improve_loss >= patience:
                logging.info(f"[Alignment][NES-diag] Early stop at iter {it}: loss no improvement for {patience} steps.")
                break

        # ----------------------------
        # 10) Apply best solution
        # ----------------------------
        param_idx = 0
        for t in range(self._cur_task + 1):
            head = self.model.classifier.heads[t]
            for p in head.parameters():
                sz = p.numel()
                new_data = best_theta[param_idx:param_idx + sz]
                param_idx += sz
                p.data = torch.from_numpy(new_data).float().cuda().view(p.shape)

        final_features, final_labels = sample_features()
        with torch.no_grad():
            logits = stitched_logits_from_heads(final_features)
            final_macro, _ = macro_class_accuracy(logits, final_labels)

        t_align_total = time.time() - t_align_start
        self._total_align_time += t_align_total
        actual_iters = len(iter_times)

        logging.info("[Alignment][NES] === Convergence & Cost Summary ===")
        logging.info(f"[Alignment][NES]   Iterations run: {actual_iters} / {iters}")
        logging.info(f"[Alignment][NES]   Synthetic macro-acc: {-best_loss:.4f} → {final_macro:.4f}")
        logging.info(f"[Alignment][NES]   Wall-clock: total={t_align_total:.1f}s, mean/iter={np.mean(iter_times):.2f}s")
        if hasattr(self, "_train_time") and self._train_time > 0:
            logging.info(f"[Alignment][NES]   Alignment overhead: {t_align_total:.1f}s / train {self._train_time:.1f}s = {t_align_total / self._train_time:.2f}x")
        if self._total_classes == self._total_classnum:
            logging.info(f"[Alignment][NES]   TOTAL across all tasks — alignment: {self._total_align_time:.1f}s, training: {self._total_train_time:.1f}s, ratio: {self._total_align_time / max(self._total_train_time, 1):.2f}x")

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
        torch.save(self.model.get_backbone_trainable_params(), self.merged_checkpoint(self._cur_task))

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
        config["train_batch_size"] = 32

    result = {"mlp_faa": 0.0, "mlp_ffm": 0.0, "mlp_asa": 0.0}
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

        del learner
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        import traceback
        logging.error(f"[Experiment {dataset_name}_{config_name}] {type(e).__name__}: {e}")
        logging.error(traceback.format_exc())

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
                    dataset_results[config_name] = {"seeds": [], "mlp_faa": [], "mlp_ffm": [], "mlp_asa": []}

                dataset_results[config_name]["seeds"].append(seed)
                dataset_results[config_name]["mlp_faa"].append(result["mlp_faa"])
                dataset_results[config_name]["mlp_ffm"].append(result["mlp_ffm"])
                dataset_results[config_name]["mlp_asa"].append(result["mlp_asa"])

            logging.info("\n" + "=" * 80)
            logging.info(f"SUMMARY FOR {dataset_name.upper()} - {config_name.upper()}")
            logging.info("=" * 80)

            vals = dataset_results[config_name]
            if vals["mlp_asa"]:
                logging.info(f"  MLP - ASA: {np.mean(vals['mlp_asa']):.2f} ± {np.std(vals['mlp_asa']):.2f}")
                logging.info(f"  MLP - FAA: {np.mean(vals['mlp_faa']):.2f} ± {np.std(vals['mlp_faa']):.2f}")
                logging.info(f"  MLP - FFM: {np.mean(vals['mlp_ffm']):.2f} ± {np.std(vals['mlp_ffm']):.2f}")

        logging.info("=" * 80 + "\n")


if __name__ == "__main__":
    start_time = time.time()
    run_experiments()
    print(f"Total time: {time.time() - start_time:.2f}s")

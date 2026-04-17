# =============================================================================
# KCEA experiment configuration
#
# Workflow:
#   1. Edit DATA_TABLE  — pick which dataset(s) to run.
#   2. Edit BASE_CONFIG — adjust shared training hyperparameters.
#   3. Edit EXPERIMENT_CONFIGS — define named experiment variants.
#   4. Edit PARAM_SWEEP — uncomment ONE key to sweep it across multiple values.
#      Each value produces a separate named run:
#          "{experiment_name}__{param_short}_{value}"
#      Leave PARAM_SWEEP empty (all commented) to run EXPERIMENT_CONFIGS as-is.
#
# Run:
#   cd /home/lis/continual_learning/KCEA
#   python learner.py
# =============================================================================


# ── Datasets ──────────────────────────────────────────────────────────────────
# Format: "dataset_name": [(num_tasks, init_cls, increment)]
# Uncomment a line to include that dataset in the run.
DATA_TABLE = {
    "cifar224":         [(10, 10, 10)],
    "imagenetr":        [(10, 20, 20)],
    "imageneta":        [(10, 20, 20)],
    "cub":              [(10, 20, 20)],
    "omnibenchmark":    [(10, 30, 30)],
    "vtab":             [(5,  10, 10)],
    "cars":             [(10, 16, 20)]
}


# ── Base training hyperparameters (shared across all experiments) ─────────────
BASE_CONFIG = {
    "seed": [1993, 1994, 1995],

    # Training
    "train_merge":        False,
    "train_ca":           False,

    "train_ablation":     False,
    "train_epochs":       10,
    "train_batch_size":   64,
    "train_base_lr":      1e-2,
    "train_weight_decay": 5e-4,

    # Backbone / LoRA
    "model_backbone":             "vit_base_patch16_224_lora",
    "model_lora_r":               64,
    "model_lora_alpha":           128,
    "model_lora_dropout":         0.0,
    "model_lora_target_modules":  ["qkv"],
    "model_use_norm":             True,
}


# ── Named experiment variants ─────────────────────────────────────────────────
# Each key is an experiment name; its dict is merged on top of BASE_CONFIG.
# You can define multiple variants here to compare them in a single run.
EXPERIMENT_CONFIGS = {
    "kcea_ft": {
        "train_prefix": "kcea",

        # Parameter merging (TIES)
        "model_merge":             "ties",
        "model_merge_coef":        1.0,
        "model_merge_topk":        100,
        "model_merge_incremental": True,

        # Classifier alignment — method selection
        "train_ca_method":             "nes",  # "nes" | "ce"
        "train_ca_samples_per_class":  512,    # synthetic samples per class (used by NES)

        # Lambda — scales per-head sigma via f(λ) = 1/sqrt(λ):
        #   larger λ → smaller sigma → less exploration → less update (more constrained)
        #   smaller λ → larger sigma → more exploration → more update (freer to move)
        #   λ = 0.0   → uses lambda_eps floor → maximum sigma (fully free)
        "train_ca_lambda_old_default": 1e-2,   # old task heads: constrained (preserves past knowledge)
        "train_ca_lambda_cur_default": 0.0,    # current task head: set 0.0 to let it move freely

        # NES optimisation
        "train_ca_nes_sigma_init":           1e-3,   # initial base exploration std
        "train_ca_nes_sigma_final":          1e-4,   # final base exploration std (exponential decay)
        "train_ca_nes_sigma_min":            1e-5,   # per-task sigma lower clip
        "train_ca_nes_sigma_max":            1e-2,   # per-task sigma upper clip
        "train_ca_nes_lambda_eps":           1e-8,   # epsilon for importance(λ) = 1/sqrt(max(λ, eps))
        "train_ca_nes_lr":                   0.01,
        "train_ca_nes_iterations":           200,
        "train_ca_nes_popsize":              100,
        "train_ca_nes_patience":             30,     # loss early-stop patience (iterations); -1 = run all iterations

        # CE gradient-based alignment (train_ca_method = "ce")
        "train_ce_samples_per_class":        512,    # synthetic samples per class per epoch
        "train_ce_epochs":                   20,     # number of gradient steps
        "train_ce_lr":                       1e-3,   # Adam learning rate for head parameters
    },
}


# ── Hyperparameter sweep ───────────────────────────────────────────────────────
# Uncomment EXACTLY ONE key to sweep it.  All other keys must stay commented.
# Each listed value becomes a separate run named:
#   "{experiment}__{short_param}_{value}"
# Results are summarised side-by-side at the end of the run.
#
# Example: to sweep NES learning rate, uncomment the "train_ca_nes_lr" line.
PARAM_SWEEP = {
    # ── NES core ──────────────────────────────────────────────────────────────
    # "train_ca_nes_lr":                   [0.001, 0.01, 0.1],
    # "train_ca_nes_popsize":              [50, 100, 200],
    # "train_ca_nes_iterations":           [100, 200, 400],
    # "train_ca_nes_sigma_init":           [1e-4, 1e-3, 1e-2],
    # "train_ca_nes_sigma_final":          [1e-5, 1e-4, 1e-3],
    # "train_ca_nes_sigma_min":            [1e-6, 1e-5, 1e-4],
    # "train_ca_nes_sigma_max":            [1e-3, 1e-2, 1e-1],
    # "train_ca_nes_patience":             [-1, 10, 30, 50],

    # ── Classifier alignment ──────────────────────────────────────────────────
    # "train_ca_samples_per_class":        [128, 256, 512],
    # "train_ca_lambda_old_default":       [0.0, 1e-4, 1e-2],
    # "train_ca_lambda_cur_default":       [0.0, 1e-4, 1e-2],

    # ── Model merging ─────────────────────────────────────────────────────────
    # "model_merge_topk":                  [20, 50, 100],
    # "model_merge_coef":                  [0.5, 1.0, 1.5],
    # "model_merge":                       ["none", "ties", "max", "max_abs", "min"],

    # ── Backbone / LoRA ───────────────────────────────────────────────────────
    # "model_lora_r":                      [16, 32, 64],
    # "model_lora_alpha":                  [32, 64, 128],

    # ── Training schedule ─────────────────────────────────────────────────────
    # "train_epochs":                      [5, 10, 20],
    # "train_base_lr":                     [1e-3, 5e-3, 1e-2],
}

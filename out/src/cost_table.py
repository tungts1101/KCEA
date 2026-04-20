"""Shared LaTeX body builders for cost-analysis tables.

Public API
----------
build_params_body(costs, methods, datasets)           -> str
build_memory_body(costs, methods, datasets)           -> str
build_time_body(costs, methods, datasets, kcea_keys)  -> str
build_kcea_breakdown_body(costs, method_key, datasets) -> str
"""
from __future__ import annotations

import numpy as np

from out.src.cost_loader import aggregate_across_datasets, aggregate_per_dataset

# ── Display names ─────────────────────────────────────────────────────────────
COST_BASELINE_KEYS = [
    "APER+Adapter", "APER+SSF", "APER+VPT-deep",
    "CodaPrompt", "DUCT", "DualPrompt", "EASE", "L2P",
    "MOS", "SLCA", "TUNA",
]
COST_KCEA_KEY = "KCEA-FT+Merge+Align"   # full KCEA pipeline

COST_DISPLAY: dict[str, str] = {
    "APER+Adapter":        "APER+Adapter",
    "APER+SSF":            "APER+SSF",
    "APER+VPT-deep":       "APER+VPT-Deep",
    "CodaPrompt":          "CODA-Prompt",
    "DUCT":                "DUCT",
    "DualPrompt":          "DualPrompt",
    "EASE":                "EASE",
    "L2P":                 "L2P",
    "MOS":                 "MOS",
    "SLCA":                "SLCA",
    "TUNA":                "TUNA",
    "KCEA-FT+Merge+Align": "KCEA",
    # Also expose KCEA-FT for the breakdown table
    "KCEA-FT":             r"KCEA$_{\text{FT}}$",
}

# Frozen ViT-B/16 backbone RAM: shared by all methods, excluded from method-owned costs.
# Measured from KCEA logs (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k, fp32).
VIT_BACKBONE_MB: float = 327.3

DATASETS_ORDER = ["CIFAR", "IN-R", "IN-A", "CUB", "OB", "VTAB", "CARS"]
DATASET_DISPLAY = {
    "CIFAR": "CIFAR100", "IN-R": "IN-R", "IN-A": "IN-A",
    "CUB": "CUB", "OB": "OB", "VTAB": "VTAB", "CARS": "CARS",
}

# ── Cell helpers ──────────────────────────────────────────────────────────────

def _cell(mean: float | None, std: float | None, bold: bool,
          fmt: str = ".2f") -> str:
    """Format `mean$_{\\pm std}$` with optional bold."""
    if mean is None:
        return "--"
    s = f"{mean:{fmt}}"
    if bold:
        s = rf"\textbf{{{s}}}"
    sub = f"{std:{fmt}}" if std is not None else "?"
    return rf"{s}$_{{\pm {sub}}}$"


def _best_idx(vals: list[float | None], lower_is_better: bool = True) -> int:
    """Return index of best (lowest or highest) non-None value."""
    valid = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not valid:
        return -1
    valid.sort(key=lambda x: x[1], reverse=not lower_is_better)
    return valid[0][0]


def _tabular_wrap(col_spec: str, header: str, rows: list[str],
                  footnote: str = "") -> str:
    ncols = len(col_spec.split())
    body = [
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ] + rows
    if footnote:
        body.append(
            rf"\multicolumn{{{ncols}}}{{l}}{{\scriptsize {footnote}}} \\"
        )
    body += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
    ]
    return "\n".join(body)


# ── Params table ──────────────────────────────────────────────────────────────

def build_params_body(
    costs: dict,
    methods: list[str] = COST_BASELINE_KEYS,
    kcea_key: str = COST_KCEA_KEY,
    datasets: list[str] = DATASETS_ORDER,
) -> str:
    all_methods = methods + [kcea_key]
    fields = [
        "trainable_params_peak",
        "trainable_params_final",
        "total_params_final",
    ]

    # Collect per-method aggregated values for ranking
    peak_vals, final_vals, total_vals, share_vals = [], [], [], []
    for m in all_methods:
        pk, _ = aggregate_across_datasets(costs, m, datasets, "trainable_params_peak")
        fn, _ = aggregate_across_datasets(costs, m, datasets, "trainable_params_final")
        tt, _ = aggregate_across_datasets(costs, m, datasets, "total_params_final")
        share = (pk / tt * 100) if (pk is not None and tt is not None and tt > 0) else None
        peak_vals.append(pk)
        final_vals.append(fn)
        total_vals.append(tt)
        share_vals.append(share)

    bi_pk    = _best_idx(peak_vals)
    bi_fn    = _best_idx(final_vals)
    bi_tt    = _best_idx(total_vals)
    bi_share = _best_idx(share_vals)

    header = (
        r"Method & \textbf{Peak train. (M)} $\downarrow$ "
        r"& \textbf{Final (M)} $\downarrow$ "
        r"& \textbf{Total (M)} $\downarrow$ "
        r"& \textbf{Share (\%)} $\downarrow$ \\"
    )

    rows = []
    for i, m in enumerate(methods):
        pk_m, pk_s = aggregate_across_datasets(costs, m, datasets, "trainable_params_peak")
        fn_m, fn_s = aggregate_across_datasets(costs, m, datasets, "trainable_params_final")
        tt_m, tt_s = aggregate_across_datasets(costs, m, datasets, "total_params_final")
        sh_m = (pk_m / tt_m * 100) if (pk_m and tt_m) else None
        # std of share propagates; use simple approximation via dataset-level share
        sh_vals_ds = []
        for ds in datasets:
            agg = aggregate_per_dataset(costs, m, ds, fields)
            pk_ds, _ = agg["trainable_params_peak"]
            tt_ds, _ = agg["total_params_final"]
            if pk_ds is not None and tt_ds and tt_ds > 0:
                sh_vals_ds.append(pk_ds / tt_ds * 100)
        sh_s = float(np.std(sh_vals_ds)) if sh_vals_ds else None

        # Convert params to millions
        def to_m(v): return v / 1e6 if v is not None else None
        def s_m(v):  return v / 1e6 if v is not None else None

        cells = [
            COST_DISPLAY.get(m, m),
            _cell(to_m(pk_m), s_m(pk_s), i == bi_pk),
            _cell(to_m(fn_m), s_m(fn_s), i == bi_fn),
            _cell(to_m(tt_m), s_m(tt_s), i == bi_tt),
            _cell(sh_m, sh_s, i == bi_share),
        ]
        rows.append(" & ".join(cells) + r" \\")

    rows.append(r"\midrule")

    i = len(methods)
    m = kcea_key
    pk_m, pk_s = aggregate_across_datasets(costs, m, datasets, "trainable_params_peak")
    fn_m, fn_s = aggregate_across_datasets(costs, m, datasets, "trainable_params_final")
    tt_m, tt_s = aggregate_across_datasets(costs, m, datasets, "total_params_final")
    sh_m = (pk_m / tt_m * 100) if (pk_m and tt_m) else None
    sh_vals_ds = []
    for ds in datasets:
        agg = aggregate_per_dataset(costs, m, ds, fields)
        pk_ds, _ = agg["trainable_params_peak"]
        tt_ds, _ = agg["total_params_final"]
        if pk_ds is not None and tt_ds and tt_ds > 0:
            sh_vals_ds.append(pk_ds / tt_ds * 100)
    sh_s = float(np.std(sh_vals_ds)) if sh_vals_ds else None

    def to_m(v): return v / 1e6 if v is not None else None
    def s_m(v):  return v / 1e6 if v is not None else None

    cells = [
        COST_DISPLAY.get(m, m),
        _cell(to_m(pk_m), s_m(pk_s), i == bi_pk),
        _cell(to_m(fn_m), s_m(fn_s), i == bi_fn),
        _cell(to_m(tt_m), s_m(tt_s), i == bi_tt),
        _cell(sh_m, sh_s, i == bi_share),
    ]
    rows.append(" & ".join(cells) + r" \\")

    return _tabular_wrap("l c c c c", header, rows)


# ── Memory table ──────────────────────────────────────────────────────────────

def build_memory_body(
    costs: dict,
    methods: list[str] = COST_BASELINE_KEYS,
    kcea_key: str = COST_KCEA_KEY,
    datasets: list[str] = DATASETS_ORDER,
) -> str:
    all_methods = methods + [kcea_key]

    # Peak training RAM = PEFT + classifier heads (before Gaussian), backbone excluded.
    # Final RAM = ram_owned_mb (includes Gaussian for aligned variant), backbone excluded.
    # Baselines fall back to storage_cost − VIT_BACKBONE_MB if breakdown fields are absent.
    peak_vals, final_vals = [], []
    for m in all_methods:
        peft_m, _  = aggregate_across_datasets(costs, m, datasets, "ram_peft_mb")
        cls_m, _   = aggregate_across_datasets(costs, m, datasets, "ram_classifier_mb")
        pk = (peft_m or 0) + (cls_m or 0) if (peft_m is not None or cls_m is not None) else None
        if pk is None:
            sc_m, _ = aggregate_across_datasets(costs, m, datasets, "storage_cost")
            pk = max(sc_m - VIT_BACKBONE_MB, 0.0) if sc_m is not None else None
        fn, _ = aggregate_across_datasets(costs, m, datasets, "ram_owned_mb")
        if fn is None:
            sc_m, _ = aggregate_across_datasets(costs, m, datasets, "storage_cost")
            fn = max(sc_m - VIT_BACKBONE_MB, 0.0) if sc_m is not None else None
        peak_vals.append(pk)
        final_vals.append(fn)

    bi_pk = _best_idx(peak_vals)
    bi_fn = _best_idx(final_vals)

    header = (
        r"Method & \textbf{Peak train. RAM (MB)} $\downarrow$ "
        r"& \textbf{Final RAM (MB)} $\downarrow$ \\"
    )
    footnote = (
        rf"All values exclude the shared frozen backbone ({VIT_BACKBONE_MB:.0f}\,MB, "
        r"ViT-B/16). For KCEA, final RAM includes class-conditional Gaussian statistics "
        r"($\mu_c, \Sigma_c$)."
    )

    def _baseline_mem(costs, m, datasets, field):
        """Return (mean, std) for a baseline memory field, subtracting backbone if present.

        Baselines may report `storage_cost` as total RAM (backbone included).
        We subtract VIT_BACKBONE_MB so all methods are backbone-exclusive.
        """
        # Prefer method-specific PEFT breakdown fields when available.
        mn, sd = aggregate_across_datasets(costs, m, datasets, field)
        if mn is None:
            # Fall back to storage_cost − backbone
            sc_m, sc_s = aggregate_across_datasets(costs, m, datasets, "storage_cost")
            if sc_m is not None:
                mn = max(sc_m - VIT_BACKBONE_MB, 0.0)
                sd = sc_s  # backbone is a fixed offset; std is unchanged
        return mn, sd

    rows = []
    for i, m in enumerate(methods):
        peft_m, peft_s = aggregate_across_datasets(costs, m, datasets, "ram_peft_mb")
        cls_m,  cls_s  = aggregate_across_datasets(costs, m, datasets, "ram_classifier_mb")
        pk_m = (peft_m or 0) + (cls_m or 0) if (peft_m is not None or cls_m is not None) else None
        # Propagate std by root-sum-square (independent components)
        pk_s = float(np.sqrt((peft_s or 0)**2 + (cls_s or 0)**2)) if pk_m is not None else None
        if pk_m is None:
            # Fallback: storage_cost − backbone (total in-flight RAM during training)
            pk_m, pk_s = _baseline_mem(costs, m, datasets, "ram_peft_mb")
        fn_m, fn_s = _baseline_mem(costs, m, datasets, "ram_owned_mb")
        cells = [
            COST_DISPLAY.get(m, m),
            _cell(pk_m, pk_s, i == bi_pk),
            _cell(fn_m, fn_s, i == bi_fn),
        ]
        rows.append(" & ".join(cells) + r" \\")

    rows.append(r"\midrule")

    i = len(methods)
    m = kcea_key
    peft_m, peft_s = aggregate_across_datasets(costs, m, datasets, "ram_peft_mb")
    cls_m,  cls_s  = aggregate_across_datasets(costs, m, datasets, "ram_classifier_mb")
    pk_m = (peft_m or 0) + (cls_m or 0) if (peft_m is not None or cls_m is not None) else None
    pk_s = float(np.sqrt((peft_s or 0)**2 + (cls_s or 0)**2)) if pk_m is not None else None
    fn_m, fn_s = aggregate_across_datasets(costs, m, datasets, "ram_owned_mb")
    cells = [
        COST_DISPLAY.get(m, m),
        _cell(pk_m, pk_s, i == bi_pk),
        _cell(fn_m, fn_s, i == bi_fn),
    ]
    rows.append(" & ".join(cells) + r" \\")

    return _tabular_wrap("l c c", header, rows, footnote=footnote)


# ── Time table ────────────────────────────────────────────────────────────────

def build_time_body(
    costs: dict,
    methods: list[str] = COST_BASELINE_KEYS,
    kcea_key: str = COST_KCEA_KEY,
    datasets: list[str] = DATASETS_ORDER,
    kcea_keys: set[str] | None = None,
) -> str:
    """kcea_keys: set of method keys that have alignment time (show nes_s column)."""
    if kcea_keys is None:
        kcea_keys = {"KCEA-FT", "KCEA-FT+Merge", "KCEA-FT+Merge+Align", "KCEA"}

    all_methods = methods + [kcea_key]
    train_vals, eval_vals = [], []
    for m in all_methods:
        tr, _ = aggregate_across_datasets(costs, m, datasets, "training_s")
        ev, _ = aggregate_across_datasets(costs, m, datasets, "evaluation_s")
        train_vals.append(tr)
        eval_vals.append(ev)

    bi_tr = _best_idx(train_vals)
    bi_ev = _best_idx(eval_vals)

    header = (
        r"Method & \textbf{Training (s)} $\downarrow$ "
        r"& \textbf{Evaluation (s)} & \textbf{Alignment (s)} \\"
    )

    rows = []
    for i, m in enumerate(methods):
        tr_m, tr_s = aggregate_across_datasets(costs, m, datasets, "training_s")
        ev_m, ev_s = aggregate_across_datasets(costs, m, datasets, "evaluation_s")
        nes_cell = "--"  # baselines have no alignment step
        cells = [
            COST_DISPLAY.get(m, m),
            _cell(tr_m, tr_s, i == bi_tr),
            _cell(ev_m, ev_s, i == bi_ev),
            nes_cell,
        ]
        rows.append(" & ".join(cells) + r" \\")

    rows.append(r"\midrule")

    i = len(methods)
    m = kcea_key
    tr_m, tr_s   = aggregate_across_datasets(costs, m, datasets, "training_s")
    ev_m, ev_s   = aggregate_across_datasets(costs, m, datasets, "evaluation_s")
    nes_m, nes_s = aggregate_across_datasets(costs, m, datasets, "nes_s")
    nes_cell = _cell(nes_m, nes_s, False) if m in kcea_keys else "--"
    cells = [
        COST_DISPLAY.get(m, m),
        _cell(tr_m, tr_s, i == bi_tr),
        _cell(ev_m, ev_s, i == bi_ev),
        nes_cell,
    ]
    rows.append(" & ".join(cells) + r" \\")

    return _tabular_wrap("l c c c", header, rows)


# ── KCEA breakdown table ──────────────────────────────────────────────────────

def build_kcea_breakdown_body(
    costs: dict,
    method_key: str,
    datasets: list[str] = DATASETS_ORDER,
) -> str:
    """Per-dataset wall-clock breakdown for KCEA.

    Rows: datasets + Mean.
    Columns: Wall-clock (s), Training (s, %), NES alignment (s, %),
             Evaluation (s, %), Alignment/Training (%).
    Cell format for timed columns: `mean$_{\\pm std}$ (pct%)`
    """

    def _timed_cell(mean: float | None, std: float | None,
                    pct: float | None) -> str:
        if mean is None:
            return "--"
        s = f"{mean:.1f}"
        sub = f"{std:.1f}" if std is not None else "?"
        pct_str = f"({pct:.1f}\\%)" if pct is not None else ""
        return rf"{s}$_{{\pm {sub}}}$ {pct_str}".strip()

    def _ratio_cell(mean: float | None, std: float | None) -> str:
        if mean is None:
            return "--"
        s = f"{mean:.1f}"
        sub = f"{std:.1f}" if std is not None else "?"
        return rf"{s}$_{{\pm {sub}}}$"

    header = (
        r"\textbf{Dataset} & \textbf{Wall-clock (s)} "
        r"& \textbf{Training (s)} & \textbf{NES align. (s)} "
        r"& \textbf{Evaluation (s)} & \textbf{Align./Train. (\%)} \\"
    )

    fields = ["wall_clock_s", "training_s", "nes_s", "evaluation_s"]
    rows = []

    # Accumulators for the Mean row
    all_ds_aggs: list[dict] = []

    for ds in datasets:
        agg = aggregate_per_dataset(costs, method_key, ds, fields)
        all_ds_aggs.append(agg)

        wc_m, wc_s   = agg["wall_clock_s"]
        tr_m, tr_s   = agg["training_s"]
        nes_m, nes_s = agg["nes_s"]
        ev_m, ev_s   = agg["evaluation_s"]

        tr_pct  = (tr_m  / wc_m * 100) if (tr_m  is not None and wc_m) else None
        nes_pct = (nes_m / wc_m * 100) if (nes_m is not None and wc_m) else None
        ev_pct  = (ev_m  / wc_m * 100) if (ev_m  is not None and wc_m) else None
        ratio   = (nes_m / tr_m * 100) if (nes_m is not None and tr_m  and tr_m > 0) else None

        # ratio std: propagate via δ(nes/tr) ≈ nes/tr * sqrt((σnes/nes)² + (σtr/tr)²)
        if ratio is not None and nes_m is not None and tr_m:
            rel_nes = (nes_s / nes_m) if (nes_s and nes_m) else 0
            rel_tr  = (tr_s  / tr_m)  if (tr_s  and tr_m)  else 0
            ratio_s = ratio * float(np.sqrt(rel_nes**2 + rel_tr**2))
        else:
            ratio_s = None

        cells = [
            DATASET_DISPLAY.get(ds, ds),
            _timed_cell(wc_m, wc_s, None),
            _timed_cell(tr_m, tr_s, tr_pct),
            _timed_cell(nes_m, nes_s, nes_pct),
            _timed_cell(ev_m, ev_s, ev_pct),
            _ratio_cell(ratio, ratio_s),
        ]
        rows.append(" & ".join(cells) + r" \\")

    # ── Mean row ──────────────────────────────────────────────────────────────
    rows.append(r"\midrule")

    def _mean_across_ds(key):
        vals = [a[key][0] for a in all_ds_aggs if a[key][0] is not None]
        stds = [a[key][1] for a in all_ds_aggs if a[key][1] is not None]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    wc_m, wc_s   = _mean_across_ds("wall_clock_s")
    tr_m, tr_s   = _mean_across_ds("training_s")
    nes_m, nes_s = _mean_across_ds("nes_s")
    ev_m, ev_s   = _mean_across_ds("evaluation_s")

    tr_pct  = (tr_m  / wc_m * 100) if (tr_m  is not None and wc_m) else None
    nes_pct = (nes_m / wc_m * 100) if (nes_m is not None and wc_m) else None
    ev_pct  = (ev_m  / wc_m * 100) if (ev_m  is not None and wc_m) else None
    ratio   = (nes_m / tr_m * 100) if (nes_m is not None and tr_m and tr_m > 0) else None
    if ratio is not None and nes_m is not None and tr_m:
        rel_nes = (nes_s / nes_m) if (nes_s and nes_m) else 0
        rel_tr  = (tr_s  / tr_m)  if (tr_s  and tr_m)  else 0
        ratio_s = ratio * float(np.sqrt(rel_nes**2 + rel_tr**2))
    else:
        ratio_s = None

    cells = [
        r"\textbf{Mean}",
        _timed_cell(wc_m, wc_s, None),
        _timed_cell(tr_m, tr_s, tr_pct),
        _timed_cell(nes_m, nes_s, nes_pct),
        _timed_cell(ev_m, ev_s, ev_pct),
        _ratio_cell(ratio, ratio_s),
    ]
    rows.append(" & ".join(cells) + r" \\")

    lines = [
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\footnotesize",
        r"\begin{tabular}{l c c c c c}",
        r"\toprule",
        header,
        r"\midrule",
    ] + rows + [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
    ]
    return "\n".join(lines)

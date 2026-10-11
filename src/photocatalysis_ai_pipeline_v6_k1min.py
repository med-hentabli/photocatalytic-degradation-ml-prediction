#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
photocatalysis_ai_pipeline_v6_all.py

Complete V6 reviewer-response workflow retaining the full V3/V4/V5 architecture and adding publication-grade cheminformatics + machine-learning diagnostics
for prediction of photocatalytic residual concentration ratio (c/c0).

V6 complete reviewer-response architecture relative to V4/V5
-------------------------------
1. Uses the revised database architecture including pH.
2. Treats irradiation source as a categorical variable:
       0 = sunlight, 1 = UV, 2 = visible light.
   Light_power_W is numerical and may use the author's convention of 0 W for
   sunlight/no artificial lamp.
3. Handles genuinely missing physical variables with TRAINING-ONLY median
   imputation plus missingness indicators.
4. Restricts molecular feature selection to pollutant information only:
       lambda_max + RDKit descriptors from canonicalized SMILES.
5. Ranks descriptors by the hybrid score as the PRIMARY ordering criterion.
6. Adds hybrid-score weight sensitivity analysis.
7. Repeats the complete feature-selection + minimal-k procedure over repeated
   training resamples to quantify stability of both descriptor identity and k*.
8. Adds exhaustive evaluation of every individual pollutant descriptor and every
   pair of two descriptors (optional but enabled by default for publication run).
9. Adds the reviewer-requested leakage-resistant grouped validation using ONLY
   Kinetic_series_ID. A five-fold GroupKFold keeps each complete kinetic series
   exclusively in the outer training or validation subset. In each outer fold,
   imputation, one-hot encoding, scaling, variance filtering, Lasso/MI ranking,
   minimal-k selection and model fitting are re-fitted using outer-training data only.
10. Adds standard QSAR/chemoinformatics validation metrics including Q2F1,
    Q2F2, Q2F3, Lin's CCC, Golbraikh-Tropsha slopes, and rm2 metrics.
11. Uses Wilcoxon signed-rank as the primary paired bias test and adds bootstrap
    95% confidence intervals for mean and median bias.
12. Adds SHAP dependence plots for top predictors.
13. Adds systematic error-cluster analysis by pollutant, catalyst, study,
    kinetic series, catalyst-pollutant system, and irradiation source.
14. Expands kinetic benchmarking to zero-order, pseudo-first-order,
    pseudo-second-order and Weibull/stretched-exponential models.
15. Computes the Williams leverage threshold from the ACTUAL transformed number
    of model columns, including missingness indicators and one-hot light-source
    columns.
16. Exports hyperparameter search ranges, split/group identifiers, Optuna trials,
    descriptor matrix, all validation tables, software versions and model bundle.
17. Adds one paired experimental-vs-predicted figure for EVERY ML model with
    MOD = |experimental - predicted| analysis and an Excel MOD summary.
18. Adds a reviewer-ready "Comparison with classical kinetic models" figure;
    its dashed diagonal is explicitly the 1:1 identity line, not a kinetic fit.
19. Exports the exact kinetic-series groups and outer-fold memberships used in
    validation to validation_groups_and_folds.xlsx; study/pollutant/catalyst IDs remain metadata.
20. Adds hard architecture checks and audit tables proving the intended order:
    SMILES -> RDKit descriptors -> pollutant-only feature selection ->
    concatenation with the untouched fixed catalyst/process/pH/light block.

Primary run
-----------
python photocatalysis_ai_pipeline_v6_all.py \
    --input data_co.xlsx \
    --output-dir outputs_v6_all \
    --n-trials 80

Fast smoke test
---------------
python photocatalysis_ai_pipeline_v6_all.py \
    --input data_co.xlsx \
    --output-dir outputs_v6_smoke \
    --fast-run --no-run-exhaustive --no-run-stability \
    --no-run-y-randomization --no-run-shap --no-run-kinetics \
    --skip-publication-plots

Important interpretation
------------------------
The random 80:20 split remains an INTERNAL held-out interpolation benchmark for
continuity with the submitted manuscript. Leakage-resistant five-fold GroupKFold by Kinetic_series_ID is reported separately and should be emphasized when discussing transferability to unseen kinetic series.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata as importlib_metadata
import itertools
import json
import logging
import os
import platform
import sys
import traceback
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from scipy.optimize import curve_fit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    RepeatedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

import optuna
from optuna.samplers import TPESampler
from xgboost import XGBRegressor

warnings.filterwarnings("default")
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import shap
    SHAP_AVAILABLE = True
    SHAP_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    shap = None
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = exc

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except Exception:  # pragma: no cover
    sns = None
    SEABORN_AVAILABLE = False

try:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except Exception:  # pragma: no cover
    OPENPYXL_AVAILABLE = False


# =============================================================================
# Core configuration and canonical schema
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_N_TRIALS = 80
MI_TOP_K = 10
MAX_POLLUTANT_FEATURES = 10
STABILITY_SPLITS = 5
STABILITY_REPEATS = 10
BOOTSTRAP_REPEATS = 2000
YRANDOM_REPEATS = 100
WEIGHT_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

TARGET_COL = "c_c0"
SMILES_COL = "Smiles"
LAMBDA_COL = "lambda_max_nm"
POLLUTANT_COL = "Pollutant"
CATALYST_COL = "Catalyst"
IRRADIATION_COL = "Irradiation_source"
REF_COL = "Ref"
STUDY_COL = "Study_ID"
SERIES_COL = "Kinetic_series_ID"

FIXED_NUMERIC = [
    "MW",
    "Band_gap_Eg_eV",
    "BET_m2_g",
    "Catalyst_dose_ug_mL",
    "Solution_volume_mL",
    "Light_power_W",
    "Initial_concentration_mg_L",
    "Time_min",
    "pH",
]

# Irradiation source is categorical, NOT a continuous ordinal quantity.
IRRADIATION_MAPPING = {0: "Sunlight", 1: "UV", 2: "Visible"}

COLUMN_ALIASES = {
    # Target
    "c/c0": TARGET_COL,
    "c_c0": TARGET_COL,
    "C/C0": TARGET_COL,
    # Pollutant / catalyst
    "Pollutants": POLLUTANT_COL,
    "pollutant": POLLUTANT_COL,
    "pollutants": POLLUTANT_COL,
    "Photocatalyst": CATALYST_COL,
    # SMILES
    "SMILES": SMILES_COL,
    "smiles": SMILES_COL,
    # Physicochemical variables
    "Band gap Eg (eV)": "Band_gap_Eg_eV",
    "Band_gap_Eg_eV": "Band_gap_Eg_eV",
    "Catalyst dose  µg mL⁻¹": "Catalyst_dose_ug_mL",
    "Catalyst dose µg mL⁻¹": "Catalyst_dose_ug_mL",
    "Catalyst dose μg mL⁻¹": "Catalyst_dose_ug_mL",
    "Catalyst_dose_ug_mL": "Catalyst_dose_ug_mL",
    "Solution volume mL": "Solution_volume_mL",
    "Solution_volume_mL": "Solution_volume_mL",
    "Initial concentration mg L⁻¹": "Initial_concentration_mg_L",
    "Initial_concentration_mg_L": "Initial_concentration_mg_L",
    "Time": "Time_min",
    "Time_min": "Time_min",
    "λmax": LAMBDA_COL,
    "lambda_max_nm": LAMBDA_COL,
    # Irradiation
    "Irradiation source code": IRRADIATION_COL,
    "Irradiation_source": IRRADIATION_COL,
    "coverted_Irradiation_source": IRRADIATION_COL,
    "converted_Irradiation_source": IRRADIATION_COL,
    "power_W": "Light_power_W",
    "Light_power_W": "Light_power_W",
    "UV_power_W": "Light_power_W",
    "UV power W": "Light_power_W",
    # Reference / groups
    "ref": REF_COL,
    "Reference": REF_COL,
    "Study_ID": STUDY_COL,
    "study_id": STUDY_COL,
    "Kinetic_series_ID": SERIES_COL,
    "kinetic_series_id": SERIES_COL,
    "Series_ID": SERIES_COL,
    "Series ID": SERIES_COL,
}


# =============================================================================
# Utilities
# =============================================================================

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.unique(y_true).size < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def pearson_corr(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def save_df(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_excel(path, index=index, engine="openpyxl")
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=index)


def save_figure(fig, stem: Path, dpi: int = 400, vector: bool = False) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if vector:
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        try:
            fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        except Exception:
            pass
    plt.close(fig)
    gc.collect()


def package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "not available"


def software_versions() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scipy": package_version("scipy"),
        "scikit-learn": package_version("scikit-learn"),
        "rdkit": package_version("rdkit"),
        "xgboost": package_version("xgboost"),
        "optuna": package_version("optuna"),
        "shap": package_version("shap"),
        "matplotlib": package_version("matplotlib"),
        "seaborn": package_version("seaborn"),
        "joblib": package_version("joblib"),
    }


def setup_logging(out: Path) -> None:
    (out / "logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(out / "logs" / "pipeline_v6.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def style_excel_workbook(path: Path) -> None:
    if not OPENPYXL_AVAILABLE or not path.exists():
        return
    wb = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin = Side(style="thin", color="D9E2F3")
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for row in ws.iter_rows():
            for cell in row:
                cell.border = Border(bottom=thin)
                if cell.row > 1:
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
        for col_idx in range(1, ws.max_column + 1):
            letter = get_column_letter(col_idx)
            max_len = 0
            for cell in ws[letter]:
                if cell.value is not None:
                    max_len = max(max_len, min(70, len(str(cell.value))))
            ws.column_dimensions[letter].width = min(45, max(11, max_len + 2))
        ws.auto_filter.ref = ws.dimensions
    wb.save(path)


def write_workbook(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            if frame is None or not isinstance(frame, pd.DataFrame):
                frame = pd.DataFrame({"message": ["Not generated"]})
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    style_excel_workbook(path)


# =============================================================================
# Data ingestion, schema harmonization and quality control
# =============================================================================

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        with pd.ExcelFile(path) as xls:
            # Prefer a modelling sheet if the workbook is the cleaned database package.
            preferred = ["Modeling_Data", "Feuil1", "Sheet1"]
            sheet = next((s for s in preferred if s in xls.sheet_names), xls.sheet_names[0])
            return pd.read_excel(xls, sheet_name=sheet)
    raise ValueError("Input must be .xlsx, .xls, or .csv")


def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    for c in df.columns:
        if c in COLUMN_ALIASES:
            rename[c] = COLUMN_ALIASES[c]
    df = df.rename(columns=rename)
    # Remove accidental duplicate columns after aliases, keeping the first.
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def infer_study_ids(df: pd.DataFrame) -> pd.Series:
    if STUDY_COL in df.columns and df[STUDY_COL].notna().any():
        s = df[STUDY_COL].astype(str)
        s = s.where(df[STUDY_COL].notna(), None)
        if s.notna().all():
            return s
    if REF_COL in df.columns:
        refs = df[REF_COL].ffill()
        refs = refs.fillna("Unknown study")
    else:
        refs = pd.Series(["Unknown study"] * len(df), index=df.index)
    order = list(dict.fromkeys(refs.astype(str).tolist()))
    mapping = {r: f"S{i+1:03d}" for i, r in enumerate(order)}
    return refs.astype(str).map(mapping)


def infer_series_ids(df: pd.DataFrame) -> pd.Series:
    if SERIES_COL in df.columns and df[SERIES_COL].notna().all():
        return df[SERIES_COL].astype(str)

    condition_cols = [
        STUDY_COL, CATALYST_COL, POLLUTANT_COL, "MW", "Band_gap_Eg_eV", "BET_m2_g",
        "Catalyst_dose_ug_mL", "Solution_volume_mL", IRRADIATION_COL, "Light_power_W",
        LAMBDA_COL, "Initial_concentration_mg_L", "pH",
    ]
    condition_cols = [c for c in condition_cols if c in df.columns]
    counters = defaultdict(int)
    ids: List[str] = []
    previous_key = None
    previous_time = None
    current_sid = None

    for _, row in df.iterrows():
        study = str(row[STUDY_COL])
        key = tuple(row[c] for c in condition_cols)
        t = pd.to_numeric(pd.Series([row.get("Time_min", np.nan)]), errors="coerce").iloc[0]
        reset = (
            previous_key is None
            or key != previous_key
            or (
                pd.notna(previous_time)
                and pd.notna(t)
                and float(t) <= float(previous_time)
            )
        )
        if reset:
            counters[study] += 1
            current_sid = f"{study}_K{counters[study]:03d}"
        ids.append(current_sid)
        previous_key = key
        previous_time = t
    return pd.Series(ids, index=df.index, name=SERIES_COL)


def quality_control_v5(
    df_raw: pd.DataFrame,
    out: Path,
    zero_eg_bet_as_missing: bool = True,
    drop_out_of_range_target: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    df = harmonize_columns(df_raw)

    required = [
        CATALYST_COL, "MW", "Band_gap_Eg_eV", "BET_m2_g", "Catalyst_dose_ug_mL",
        "Solution_volume_mL", POLLUTANT_COL, SMILES_COL, IRRADIATION_COL,
        "Light_power_W", LAMBDA_COL, "Initial_concentration_mg_L", "Time_min",
        "pH", TARGET_COL,
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError("Missing required columns after schema harmonization: " + ", ".join(missing_cols))

    df = df.copy()
    df["original_index"] = np.arange(len(df), dtype=int)

    # Reference forward-fill is only for traceability, not a predictor.
    if REF_COL not in df.columns:
        df[REF_COL] = np.nan
    df[REF_COL] = df[REF_COL].ffill()

    # Normalize names for grouping while preserving the readable labels.
    df[CATALYST_COL] = df[CATALYST_COL].astype(str).str.strip()
    df[POLLUTANT_COL] = df[POLLUTANT_COL].astype(str).str.strip()
    df[SMILES_COL] = df[SMILES_COL].astype(str).str.strip()

    numeric_cols = FIXED_NUMERIC + [IRRADIATION_COL, LAMBDA_COL, TARGET_COL]
    invalid_numeric_rows = []
    for c in numeric_cols:
        original = df[c].copy()
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        for i in df.index[invalid]:
            invalid_numeric_rows.append({
                "original_index": int(df.loc[i, "original_index"]),
                "column": c,
                "raw_value": original.loc[i],
            })
        df[c] = converted

    converted_zero_rows = []
    if zero_eg_bet_as_missing:
        for c in ["Band_gap_Eg_eV", "BET_m2_g"]:
            mask = df[c].eq(0) & df[c].notna()
            for i in df.index[mask]:
                converted_zero_rows.append({
                    "original_index": int(df.loc[i, "original_index"]),
                    "column": c,
                    "old_value": 0.0,
                    "new_value": "NaN",
                    "reason": "Physical zero treated as legacy missing-value code for revised modelling",
                })
            df.loc[mask, c] = np.nan

    # Validate irradiation coding.
    bad_irr = df.loc[~df[IRRADIATION_COL].isin([0, 1, 2]) & df[IRRADIATION_COL].notna(),
                     ["original_index", IRRADIATION_COL, REF_COL]].copy()
    if not bad_irr.empty:
        logging.warning("Found %d irradiation-source codes outside {0,1,2}.", len(bad_irr))

    sunlight_power_inconsistency = df.loc[
        df[IRRADIATION_COL].eq(0) & ~df["Light_power_W"].eq(0),
        ["original_index", CATALYST_COL, POLLUTANT_COL, IRRADIATION_COL, "Light_power_W", REF_COL],
    ].copy()

    # Heuristic traceability check only: flag possible source-code disagreements
    # suggested by reference titles; never auto-correct from title text.
    keyword_flags = []
    for i, row in df.iterrows():
        ref_text = str(row.get(REF_COL, "")).lower()
        code = row.get(IRRADIATION_COL, np.nan)
        expected = None
        keyword = None
        if "sunlight" in ref_text or "solar light" in ref_text:
            expected, keyword = 0, "sunlight/solar"
        elif "visible" in ref_text:
            expected, keyword = 2, "visible"
        elif "uv" in ref_text or "ultraviolet" in ref_text:
            expected, keyword = 1, "UV"
        if expected is not None and pd.notna(code) and int(code) != expected:
            keyword_flags.append({
                "original_index": int(row["original_index"]),
                "reference_keyword": keyword,
                "expected_code_from_title_keyword": expected,
                "recorded_code": int(code),
                REF_COL: row.get(REF_COL),
                "note": "Heuristic QC flag only; verify the experimental section before changing the code.",
            })
    reference_light_keyword_flags = pd.DataFrame(keyword_flags)

    # Target flags.
    target_flags = df.loc[
        df[TARGET_COL].isna() | (df[TARGET_COL] < 0) | (df[TARGET_COL] > 1),
        ["original_index", TARGET_COL, REF_COL],
    ].copy()

    time0_flags = df.loc[
        df["Time_min"].eq(0) & df[TARGET_COL].notna() & ((df[TARGET_COL] - 1.0).abs() > 0.05),
        ["original_index", CATALYST_COL, POLLUTANT_COL, "Time_min", TARGET_COL, REF_COL],
    ].copy()

    # Create grouping variables before duplicate-condition checks.
    df[STUDY_COL] = infer_study_ids(df)
    df[SERIES_COL] = infer_series_ids(df)
    df["Catalyst_pollutant_ID"] = (
        df[CATALYST_COL].str.replace(r"\s+", "_", regex=True)
        + "__"
        + df[POLLUTANT_COL].str.replace(r"\s+", "_", regex=True)
    )
    df["Irradiation_label"] = df[IRRADIATION_COL].map(IRRADIATION_MAPPING).fillna("Unknown")

    # Duplicate condition conflicts: same recorded predictors/time but different response.
    condition_cols = [
        CATALYST_COL, "MW", "Band_gap_Eg_eV", "BET_m2_g", "Catalyst_dose_ug_mL",
        "Solution_volume_mL", POLLUTANT_COL, SMILES_COL, IRRADIATION_COL,
        "Light_power_W", LAMBDA_COL, "Initial_concentration_mg_L", "Time_min", "pH",
    ]
    dup_conflicts = []
    for _, g in df.groupby(condition_cols, dropna=False, sort=False):
        if len(g) > 1 and g[TARGET_COL].dropna().nunique() > 1:
            for _, row in g.iterrows():
                dup_conflicts.append({
                    "original_index": int(row["original_index"]),
                    STUDY_COL: row[STUDY_COL],
                    SERIES_COL: row[SERIES_COL],
                    CATALYST_COL: row[CATALYST_COL],
                    POLLUTANT_COL: row[POLLUTANT_COL],
                    "Time_min": row["Time_min"],
                    TARGET_COL: row[TARGET_COL],
                    "group_n": len(g),
                    "target_min": g[TARGET_COL].min(),
                    "target_max": g[TARGET_COL].max(),
                    "target_spread": g[TARGET_COL].max() - g[TARGET_COL].min(),
                })
    duplicate_condition_conflicts = pd.DataFrame(dup_conflicts)

    # Missingness BEFORE dropping unusable rows.
    missing_report = pd.DataFrame({
        "column": df.columns,
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_percent": [100.0 * float(df[c].isna().mean()) for c in df.columns],
    })

    # SMILES validation.
    invalid_smiles_rows = []
    for i, smi in df[SMILES_COL].items():
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            invalid_smiles_rows.append({
                "original_index": int(df.loc[i, "original_index"]),
                POLLUTANT_COL: df.loc[i, POLLUTANT_COL],
                SMILES_COL: smi,
            })
    invalid_smiles = pd.DataFrame(invalid_smiles_rows)

    # Only truly unusable records are dropped automatically.
    unusable = df[TARGET_COL].isna() | df[SMILES_COL].isna() | df[SMILES_COL].eq("")
    if invalid_smiles_rows:
        bad_indices = set(invalid_smiles["original_index"].tolist())
        unusable |= df["original_index"].isin(bad_indices)
    excluded = df.loc[unusable].copy()
    if not excluded.empty:
        save_df(excluded, out / "tables" / "rows_excluded_unusable.xlsx")
        logging.warning("Excluding %d rows with missing target/SMILES or invalid SMILES.", len(excluded))
    df = df.loc[~unusable].copy()

    if drop_out_of_range_target:
        out_mask = (df[TARGET_COL] < 0) | (df[TARGET_COL] > 1)
        if out_mask.any():
            save_df(df.loc[out_mask], out / "tables" / "rows_excluded_out_of_range_target.xlsx")
            df = df.loc[~out_mask].copy()

    df = df.reset_index(drop=True)
    df["degradation_percent"] = 100.0 * (1.0 - df[TARGET_COL])

    dataset_structure = pd.DataFrame([
        {"item": "rows_modelled", "value": len(df)},
        {"item": "studies", "value": df[STUDY_COL].nunique()},
        {"item": "kinetic_series", "value": df[SERIES_COL].nunique()},
        {"item": "catalysts", "value": df[CATALYST_COL].nunique()},
        {"item": "pollutants", "value": df[POLLUTANT_COL].str.lower().nunique()},
        {"item": "catalyst_pollutant_pairs", "value": df["Catalyst_pollutant_ID"].nunique()},
        {"item": "pH_coverage_percent", "value": 100.0 * df["pH"].notna().mean()},
        {"item": "BET_coverage_percent", "value": 100.0 * df["BET_m2_g"].notna().mean()},
        {"item": "Eg_coverage_percent", "value": 100.0 * df["Band_gap_Eg_eV"].notna().mean()},
    ])

    irradiation_counts = (
        df.groupby([IRRADIATION_COL, "Irradiation_label"], dropna=False)
        .size().reset_index(name="n")
    )

    reports = {
        "Missing_values": missing_report,
        "Invalid_numeric": pd.DataFrame(invalid_numeric_rows) if invalid_numeric_rows else pd.DataFrame({"message": ["None"]}),
        "Zero_to_missing": pd.DataFrame(converted_zero_rows) if converted_zero_rows else pd.DataFrame({"message": ["None"]}),
        "Bad_irradiation_codes": bad_irr if not bad_irr.empty else pd.DataFrame({"message": ["None"]}),
        "Sunlight_power_inconsistency": sunlight_power_inconsistency if not sunlight_power_inconsistency.empty else pd.DataFrame({"message": ["None"]}),
        "Reference_light_keyword_flags": reference_light_keyword_flags if not reference_light_keyword_flags.empty else pd.DataFrame({"message": ["None"]}),
        "Target_flags": target_flags if not target_flags.empty else pd.DataFrame({"message": ["None"]}),
        "Time0_flags": time0_flags if not time0_flags.empty else pd.DataFrame({"message": ["None"]}),
        "Invalid_SMILES": invalid_smiles if not invalid_smiles.empty else pd.DataFrame({"message": ["None"]}),
        "Duplicate_condition_conflicts": duplicate_condition_conflicts if not duplicate_condition_conflicts.empty else pd.DataFrame({"message": ["None"]}),
        "Dataset_structure": dataset_structure,
        "Irradiation_counts": irradiation_counts,
    }
    write_workbook(out / "data_quality_report_v6.xlsx", reports)
    # Legacy-compatible alias retained so previous manuscript workflows do not break.
    write_workbook(out / "data_quality_report_v5.xlsx", reports)
    return df, reports


# =============================================================================
# RDKit descriptor generation
# =============================================================================

def molecule_descriptors(smiles: str) -> Tuple[Dict[str, float], str]:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    values: Dict[str, float] = {}
    for name, func in Descriptors._descList:
        key = f"RDKit_{name}"
        try:
            v = func(mol)
            values[key] = float(v) if v is not None and np.isfinite(v) else np.nan
        except Exception:
            values[key] = np.nan
    extras = {
        "RDKit_NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
        "RDKit_NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
        "RDKit_NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings,
        "RDKit_FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    }
    for key, func in extras.items():
        try:
            v = func(mol)
            values[key] = float(v) if v is not None and np.isfinite(v) else np.nan
        except Exception:
            values[key] = np.nan
    return values, canonical


def build_pollutant_candidates(df: pd.DataFrame, out: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    cache: Dict[str, Tuple[Dict[str, float], str]] = {}
    rows = []
    smi_rows = []
    for pos, smi in enumerate(df[SMILES_COL].astype(str)):
        key = smi.strip()
        if key not in cache:
            cache[key] = molecule_descriptors(key)
        vals, canonical = cache[key]
        rows.append(vals)
        smi_rows.append({
            "row_position": pos,
            "original_index": int(df.loc[pos, "original_index"]),
            POLLUTANT_COL: df.loc[pos, POLLUTANT_COL],
            "input_smiles": key,
            "canonical_smiles": canonical,
        })
    desc = pd.DataFrame(rows, index=df.index).replace([np.inf, -np.inf], np.nan)
    desc = desc.loc[:, ~desc.columns.duplicated()].copy()
    candidate = pd.concat([df[[LAMBDA_COL]].copy(), desc], axis=1)
    candidate = candidate.loc[:, ~candidate.columns.duplicated()].copy()
    full = pd.concat([df.copy(), desc], axis=1)
    full["canonical_smiles"] = [r["canonical_smiles"] for r in smi_rows]

    save_df(pd.DataFrame(smi_rows), out / "tables" / "canonical_smiles.xlsx")
    save_df(desc, out / "tables" / "full_RDKit_descriptor_matrix.xlsx")
    save_df(full, out / "tables" / "data_with_RDKit_descriptors.xlsx")
    return candidate, pd.DataFrame(smi_rows), list(desc.columns)


# =============================================================================
# Preprocessing: fixed block and pollutant block
# =============================================================================

def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=float)


def fit_fixed_preprocessor(
    df: pd.DataFrame,
    train_idx: Sequence[int],
    transform_idx: Sequence[int],
):
    train = df.loc[list(train_idx)]
    transform = df.loc[list(transform_idx)]

    num_imp = SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True
    )
    num_train_imp = num_imp.fit_transform(train[FIXED_NUMERIC])
    num_transform_imp = num_imp.transform(transform[FIXED_NUMERIC])

    num_scaler = StandardScaler()
    num_train = num_scaler.fit_transform(num_train_imp)
    num_transform = num_scaler.transform(num_transform_imp)

    indicator_features = []
    if getattr(num_imp, "indicator_", None) is not None:
        inds = getattr(num_imp.indicator_, "features_", np.array([], dtype=int))
        indicator_features = [f"{FIXED_NUMERIC[int(i)]}__missing" for i in inds]
    numeric_names = FIXED_NUMERIC + indicator_features

    cat_imp = SimpleImputer(strategy="most_frequent")
    cat_train_imp = cat_imp.fit_transform(train[[IRRADIATION_COL]])
    cat_transform_imp = cat_imp.transform(transform[[IRRADIATION_COL]])
    enc = make_onehot_encoder()
    cat_train = enc.fit_transform(cat_train_imp)
    cat_transform = enc.transform(cat_transform_imp)
    cat_names = [str(x) for x in enc.get_feature_names_out([IRRADIATION_COL])]

    X_train = np.hstack([num_train, cat_train])
    X_transform = np.hstack([num_transform, cat_transform])
    feature_names = numeric_names + cat_names
    objects = {
        "numeric_imputer": num_imp,
        "numeric_scaler": num_scaler,
        "categorical_imputer": cat_imp,
        "categorical_encoder": enc,
        "feature_names": feature_names,
    }
    return X_train, X_transform, feature_names, objects


def fit_pollutant_preprocessor(
    candidate_raw: pd.DataFrame,
    train_idx: Sequence[int],
    transform_idx: Sequence[int],
):
    tr = candidate_raw.loc[list(train_idx)]
    te = candidate_raw.loc[list(transform_idx)]
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    tr_imp = imp.fit_transform(tr)
    te_imp = imp.transform(te)
    scaler = StandardScaler()
    tr_scaled = scaler.fit_transform(tr_imp)
    te_scaled = scaler.transform(te_imp)
    var = VarianceThreshold(threshold=0.0)
    tr_v = var.fit_transform(tr_scaled)
    te_v = var.transform(te_scaled)
    original_names = list(candidate_raw.columns)
    kept = [c for c, keep in zip(original_names, var.get_support()) if keep]
    removed = [c for c, keep in zip(original_names, var.get_support()) if not keep]
    objects = {
        "imputer": imp,
        "scaler": scaler,
        "variance_filter": var,
        "original_names": original_names,
        "kept_names": kept,
        "removed_names": removed,
    }
    return tr_v, te_v, kept, objects


# =============================================================================
# Hybrid LassoCV + mutual information feature selection
# =============================================================================

def hybrid_components(
    X_train: np.ndarray,
    y_train: np.ndarray,
    names: List[str],
    mi_top_k: int = MI_TOP_K,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, LassoCV]:
    n_splits = max(2, min(cv_splits, len(y_train)))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    lasso = LassoCV(alphas=250, cv=cv, max_iter=50000)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lasso.fit(X_train, y_train)
    coef = np.asarray(lasso.coef_, dtype=float)
    abs_coef = np.abs(coef)
    lasso_selected = abs_coef > 1e-10

    mi = mutual_info_regression(X_train, y_train, random_state=random_state)
    mi = np.nan_to_num(np.asarray(mi, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    mi_rank = pd.Series(mi).rank(method="min", ascending=False).to_numpy()
    lasso_rank = pd.Series(abs_coef).rank(method="min", ascending=False).to_numpy()
    mi_top = mi_rank <= min(mi_top_k, len(names))
    both = lasso_selected & mi_top

    lmax = abs_coef.max() if len(abs_coef) else 0.0
    mmax = mi.max() if len(mi) else 0.0
    lnorm = abs_coef / lmax if lmax > 0 else np.zeros_like(abs_coef)
    mnorm = mi / mmax if mmax > 0 else np.zeros_like(mi)

    comp = pd.DataFrame({
        "feature": names,
        "feature_type": ["lambda_max" if n == LAMBDA_COL else "RDKit" for n in names],
        "lasso_coefficient": coef,
        "lasso_abs_coefficient": abs_coef,
        "lasso_rank": lasso_rank,
        "lasso_selected": lasso_selected,
        "mi_score": mi,
        "mi_rank": mi_rank,
        "mi_top10": mi_top,
        "selected_by_both": both,
        "lasso_score_normalized": lnorm,
        "mi_score_normalized": mnorm,
    })
    return comp, lasso


def rank_hybrid(comp: pd.DataFrame, both_weight: float = 2.0) -> pd.DataFrame:
    r = comp.copy()
    r["both_weight"] = float(both_weight)
    r["combined_hybrid_score"] = (
        both_weight * r["selected_by_both"].astype(float)
        + r["mi_top10"].astype(float)
        + r["lasso_selected"].astype(float)
        + r["lasso_score_normalized"]
        + r["mi_score_normalized"]
    )
    # V5: combined score is the PRIMARY ranking criterion; other terms only break ties.
    r = r.sort_values(
        [
            "combined_hybrid_score", "selected_by_both", "mi_score",
            "lasso_abs_coefficient", "mi_top10", "lasso_selected",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    r["final_hybrid_rank"] = np.arange(1, len(r) + 1)
    return r


def cv_splitter_for_selection(
    y_train: np.ndarray,
    groups: Optional[Sequence[Any]] = None,
    desired_splits: int = 5,
):
    if groups is not None:
        groups_arr = np.asarray(groups)
        n_groups = len(pd.unique(groups_arr))
        if n_groups >= 3:
            n_splits = min(desired_splits, n_groups)
            gkf = GroupKFold(n_splits=n_splits)
            dummy = np.zeros(len(y_train))
            return list(gkf.split(dummy, y_train, groups_arr))
    n_splits = min(desired_splits, len(y_train))
    n_splits = max(2, n_splits)
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def optimize_minimal_k(
    X_fixed_train: np.ndarray,
    X_poll_train: np.ndarray,
    y_train: np.ndarray,
    ranking: pd.DataFrame,
    pollutant_names: List[str],
    max_k: int = MAX_POLLUTANT_FEATURES,
    n_estimators: int = 400,
    inner_groups: Optional[Sequence[Any]] = None,
    inner_splits: int = 5,
) -> Tuple[int, List[str], pd.DataFrame]:
    name_to_idx = {n: i for i, n in enumerate(pollutant_names)}
    max_k = min(max_k, len(ranking))
    cv = cv_splitter_for_selection(y_train, inner_groups, desired_splits=inner_splits)
    rows = []
    for k in range(1, max_k + 1):
        selected = ranking["feature"].head(k).tolist()
        cols = [name_to_idx[f] for f in selected]
        Xk = np.hstack([X_fixed_train, X_poll_train[:, cols]])
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        fold = -cross_val_score(
            model, Xk, y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        rows.append({
            "k": k,
            "selected_features": json.dumps(selected, ensure_ascii=False),
            "mean_cv_rmse": float(np.mean(fold)),
            "sd_cv_rmse": float(np.std(fold, ddof=1)) if len(fold) > 1 else 0.0,
        })
    table = pd.DataFrame(rows)
    minimum = float(table["mean_cv_rmse"].min())
    table["difference_from_minimum"] = table["mean_cv_rmse"] - minimum
    table["percent_difference_from_minimum"] = 100.0 * table["difference_from_minimum"] / minimum
    table["within_2pct"] = table["mean_cv_rmse"] <= 1.02 * minimum
    k_star = int(table.loc[table["within_2pct"]].sort_values("k").iloc[0]["k"])
    table["selected_k_star"] = table["k"].eq(k_star)
    selected = ranking["feature"].head(k_star).tolist()
    return k_star, selected, table


# =============================================================================
# Reviewer-requested feature-selection robustness
# =============================================================================

def hybrid_weight_sensitivity(
    comp: pd.DataFrame,
    X_fixed_train: np.ndarray,
    X_poll_train: np.ndarray,
    y_train: np.ndarray,
    pollutant_names: List[str],
    max_k: int,
    weights: Sequence[float],
    n_estimators: int,
    inner_groups: Optional[Sequence[Any]] = None,
) -> Tuple[pd.DataFrame, Dict[float, pd.DataFrame]]:
    ranking_by_weight: Dict[float, pd.DataFrame] = {}
    base_rank = rank_hybrid(comp, 2.0).set_index("feature")["final_hybrid_rank"]
    rows = []
    for w in weights:
        ranking = rank_hybrid(comp, w)
        ranking_by_weight[float(w)] = ranking
        k_star, selected, kdf = optimize_minimal_k(
            X_fixed_train, X_poll_train, y_train, ranking, pollutant_names,
            max_k=max_k, n_estimators=n_estimators,
            inner_groups=inner_groups, inner_splits=5,
        )
        curr = ranking.set_index("feature")["final_hybrid_rank"].reindex(base_rank.index)
        rho = stats.spearmanr(base_rank.to_numpy(), curr.to_numpy()).statistic
        rows.append({
            "both_weight": w,
            "top_descriptor": ranking.iloc[0]["feature"],
            "selected_k": k_star,
            "selected_features": json.dumps(selected, ensure_ascii=False),
            "minimum_cv_rmse": float(kdf["mean_cv_rmse"].min()),
            "selected_k_cv_rmse": float(kdf.loc[kdf["k"].eq(k_star), "mean_cv_rmse"].iloc[0]),
            "spearman_rank_vs_weight_2": rho,
        })
    return pd.DataFrame(rows), ranking_by_weight


def complete_selection_stability(
    df_train: pd.DataFrame,
    candidate_train: pd.DataFrame,
    y_train: np.ndarray,
    max_k: int,
    n_splits: int,
    n_repeats: int,
    n_estimators: int,
    both_weight: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    run_rows = []
    feature_counts = Counter()
    n_runs = 0

    # Reset indexes so fold indices map directly.
    df_train = df_train.reset_index(drop=True)
    candidate_train = candidate_train.reset_index(drop=True)
    y_train = np.asarray(y_train, dtype=float)

    for subtrain, _ in rkf.split(df_train):
        n_runs += 1
        df_sub = df_train.iloc[subtrain].reset_index(drop=True)
        cand_sub = candidate_train.iloc[subtrain].reset_index(drop=True)
        y_sub = y_train[subtrain]
        local_idx = np.arange(len(df_sub))

        Xf, _, _, _ = fit_fixed_preprocessor(df_sub, local_idx, local_idx)
        Cp, _, kept, _ = fit_pollutant_preprocessor(cand_sub, local_idx, local_idx)
        comp, _ = hybrid_components(Cp, y_sub, kept, cv_splits=3, random_state=RANDOM_STATE + n_runs)
        ranking = rank_hybrid(comp, both_weight)
        groups = df_sub[SERIES_COL].to_numpy()
        k_star, selected, kdf = optimize_minimal_k(
            Xf, Cp, y_sub, ranking, kept,
            max_k=max_k,
            n_estimators=n_estimators,
            inner_groups=groups,
            inner_splits=3,
        )
        for f in selected:
            feature_counts[f] += 1
        run_rows.append({
            "resample": n_runs,
            "n_train": len(df_sub),
            "selected_k": k_star,
            "selected_features": json.dumps(selected, ensure_ascii=False),
            "top_descriptor": ranking.iloc[0]["feature"] if len(ranking) else None,
            "minimum_cv_rmse": float(kdf["mean_cv_rmse"].min()),
            "selected_k_cv_rmse": float(kdf.loc[kdf["k"].eq(k_star), "mean_cv_rmse"].iloc[0]),
        })
        if n_runs % 10 == 0:
            logging.info("Complete feature-selection stability: %d/%d resamples", n_runs, n_splits * n_repeats)

    run_table = pd.DataFrame(run_rows)
    all_features = sorted(set(itertools.chain.from_iterable(
        [json.loads(x) for x in run_table["selected_features"]]
    ))) if len(run_table) else []
    freq_rows = []
    for f in all_features:
        freq_rows.append({
            "feature": f,
            "selection_frequency": feature_counts[f] / max(n_runs, 1),
            "selected_count": feature_counts[f],
            "n_resamples": n_runs,
        })
    freq = pd.DataFrame(freq_rows).sort_values("selection_frequency", ascending=False) if freq_rows else pd.DataFrame()
    return run_table, freq


def exhaustive_single_pair_analysis(
    X_fixed_train: np.ndarray,
    X_poll_train: np.ndarray,
    y_train: np.ndarray,
    pollutant_names: List[str],
    n_estimators: int = 100,
    max_pairs: Optional[int] = None,
    inner_groups: Optional[Sequence[Any]] = None,
    desired_splits: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv = cv_splitter_for_selection(y_train, inner_groups, desired_splits=desired_splits)

    def score(cols: List[int]) -> Tuple[float, float]:
        X = np.hstack([X_fixed_train, X_poll_train[:, cols]])
        model = ExtraTreesRegressor(
            n_estimators=n_estimators, max_features="sqrt", min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        s = -cross_val_score(model, X, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
        return float(np.mean(s)), float(np.std(s, ddof=1)) if len(s) > 1 else 0.0

    single_rows = []
    for i, name in enumerate(pollutant_names):
        mean, sd = score([i])
        single_rows.append({"feature": name, "mean_cv_rmse": mean, "sd_cv_rmse": sd})
    singles = pd.DataFrame(single_rows).sort_values("mean_cv_rmse").reset_index(drop=True)
    singles["rank"] = np.arange(1, len(singles) + 1)

    pair_rows = []
    pair_iter = itertools.combinations(range(len(pollutant_names)), 2)
    if max_pairs is not None:
        pair_iter = itertools.islice(pair_iter, max_pairs)
    for count, (i, j) in enumerate(pair_iter, start=1):
        mean, sd = score([i, j])
        pair_rows.append({
            "feature_1": pollutant_names[i],
            "feature_2": pollutant_names[j],
            "mean_cv_rmse": mean,
            "sd_cv_rmse": sd,
        })
        if count % 1000 == 0:
            logging.info("Exhaustive two-feature analysis: %d pairs completed", count)
    pairs = pd.DataFrame(pair_rows).sort_values("mean_cv_rmse").reset_index(drop=True)
    if len(pairs):
        pairs["rank"] = np.arange(1, len(pairs) + 1)

    summary_rows = []
    if len(singles):
        summary_rows.append({
            "analysis": "Best single pollutant feature",
            "features": singles.iloc[0]["feature"],
            "mean_cv_rmse": singles.iloc[0]["mean_cv_rmse"],
        })
    if len(pairs):
        summary_rows.append({
            "analysis": "Best pair of pollutant features",
            "features": f"{pairs.iloc[0]['feature_1']} + {pairs.iloc[0]['feature_2']}",
            "mean_cv_rmse": pairs.iloc[0]["mean_cv_rmse"],
        })
    return singles, pairs, pd.DataFrame(summary_rows)


# =============================================================================
# Model optimization
# =============================================================================

def hyperparameter_search_space_table() -> pd.DataFrame:
    rows = [
        ["XGBoost", "n_estimators", "integer", "200–1600"],
        ["XGBoost", "max_depth", "integer", "2–8"],
        ["XGBoost", "learning_rate", "log-uniform", "0.003–0.2"],
        ["XGBoost", "subsample", "uniform", "0.55–1.0"],
        ["XGBoost", "colsample_bytree", "uniform", "0.55–1.0"],
        ["XGBoost", "min_child_weight", "log-uniform", "0.1–10"],
        ["XGBoost", "reg_alpha", "log-uniform", "1e-8–10"],
        ["XGBoost", "reg_lambda", "log-uniform", "1e-8–10"],
        ["ExtraTrees", "n_estimators", "integer", "250–1200"],
        ["ExtraTrees", "max_depth", "categorical", "None,5,10,20,30"],
        ["ExtraTrees", "min_samples_split", "integer", "2–10"],
        ["ExtraTrees", "min_samples_leaf", "integer", "1–6"],
        ["ExtraTrees", "max_features", "categorical", "sqrt,log2,None"],
        ["RandomForest", "n_estimators", "integer", "250–1200"],
        ["RandomForest", "max_depth", "categorical", "None,5,10,20,30"],
        ["RandomForest", "min_samples_split", "integer", "2–10"],
        ["RandomForest", "min_samples_leaf", "integer", "1–6"],
        ["RandomForest", "max_features", "categorical", "sqrt,log2,None"],
        ["RandomForest", "bootstrap", "categorical", "True,False"],
        ["SVR", "kernel", "categorical", "rbf,poly,sigmoid"],
        ["SVR", "C", "log-uniform", "1e-3–1e4"],
        ["SVR", "epsilon", "log-uniform", "1e-5–0.5"],
        ["SVR", "gamma", "categorical", "scale,auto"],
        ["SVR", "degree", "integer (poly only)", "2–4"],
        ["SVR", "coef0", "uniform (poly only)", "0–3"],
        ["ANN", "hidden_layer_sizes", "categorical", "(32),(64),(128),(64,32),(128,64),(64,32,16)"],
        ["ANN", "activation", "categorical", "relu,tanh"],
        ["ANN", "alpha", "log-uniform", "1e-7–1e-2"],
        ["ANN", "learning_rate_init", "log-uniform", "1e-5–1e-2"],
    ]
    return pd.DataFrame(rows, columns=["model", "hyperparameter", "search_type", "range_or_values"])


def internal_cv_rmse(model, X, y) -> float:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return float(-np.mean(cross_val_score(
        model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1
    )))


def optimize_models(X_train: np.ndarray, y_train: np.ndarray, n_trials: int):
    studies: Dict[str, optuna.Study] = {}
    models: Dict[str, Any] = {}
    summaries = []

    def run_study(name, objective, builder):
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        model = builder(study.best_params)
        model.fit(X_train, y_train)
        studies[name] = study
        models[name] = model
        summaries.append({
            "model": name,
            "best_internal_cv_rmse": float(study.best_value),
            "best_params": json.dumps(study.best_params, default=str),
        })

    def xgb_obj(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1600),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "reg:squarederror",
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        }
        return internal_cv_rmse(XGBRegressor(**p), X_train, y_train)

    run_study(
        "XGBoost", xgb_obj,
        lambda p: XGBRegressor(**p, objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=1),
    )

    def et_obj(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 250, 1200),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        }
        return internal_cv_rmse(ExtraTreesRegressor(**p), X_train, y_train)

    run_study("ExtraTrees", et_obj, lambda p: ExtraTreesRegressor(**p, random_state=RANDOM_STATE, n_jobs=1))

    def rf_obj(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 250, 1200),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        }
        return internal_cv_rmse(RandomForestRegressor(**p), X_train, y_train)

    run_study("RandomForest", rf_obj, lambda p: RandomForestRegressor(**p, random_state=RANDOM_STATE, n_jobs=1))

    def svr_obj(trial):
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
        p = {
            "kernel": kernel,
            "C": trial.suggest_float("C", 1e-3, 1e4, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-5, 0.5, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
        if kernel == "poly":
            p["degree"] = trial.suggest_int("degree", 2, 4)
            p["coef0"] = trial.suggest_float("coef0", 0.0, 3.0)
        return internal_cv_rmse(SVR(**p), X_train, y_train)

    run_study("SVR", svr_obj, lambda p: SVR(**p))

    hidden = {
        "32": (32,), "64": (64,), "128": (128,),
        "64_32": (64, 32), "128_64": (128, 64), "64_32_16": (64, 32, 16),
    }

    def ann_obj(trial):
        key = trial.suggest_categorical("hidden_layer_sizes", list(hidden.keys()))
        p = {
            "hidden_layer_sizes": hidden[key],
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-7, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True),
            "solver": "adam",
            "max_iter": 4000,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "random_state": RANDOM_STATE,
        }
        return internal_cv_rmse(MLPRegressor(**p), X_train, y_train)

    def ann_build(p):
        p = p.copy()
        p["hidden_layer_sizes"] = hidden[p["hidden_layer_sizes"]]
        return MLPRegressor(
            **p,
            solver="adam", max_iter=4000, early_stopping=True,
            validation_fraction=0.15, random_state=RANDOM_STATE,
        )

    run_study("ANN", ann_obj, ann_build)
    return models, studies, pd.DataFrame(summaries)


def build_fast_models() -> Dict[str, Any]:
    return {
        "XGBoost": XGBRegressor(
            n_estimators=180, max_depth=5, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.85,
            min_child_weight=2.0, reg_alpha=1e-5, reg_lambda=1e-4,
            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=180, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=1),
        "RandomForest": RandomForestRegressor(n_estimators=180, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=1),
        "SVR": SVR(kernel="rbf", C=30.0, epsilon=0.04, gamma="scale"),
        "ANN": MLPRegressor(
            hidden_layer_sizes=(64,), activation="relu", alpha=1e-4,
            learning_rate_init=0.003, max_iter=1200, early_stopping=True,
            validation_fraction=0.15, random_state=RANDOM_STATE,
        ),
    }


# =============================================================================
# QSAR / chemoinformatics validation metrics
# =============================================================================

def lin_ccc(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if len(y) < 2:
        return np.nan
    den = np.var(y, ddof=1) + np.var(p, ddof=1) + (np.mean(y) - np.mean(p)) ** 2
    if den <= 0:
        return np.nan
    return float(2 * np.cov(y, p, ddof=1)[0, 1] / den)


def qsar_external_metrics(y_train_ref, y_true, y_pred) -> Dict[str, float]:
    ytr = np.asarray(y_train_ref, dtype=float)
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ntr = len(ytr)
    nt = len(y)
    sse = float(np.sum((y - p) ** 2))
    mean_train = float(np.mean(ytr))
    mean_test = float(np.mean(y))

    den_f1 = float(np.sum((y - mean_train) ** 2))
    den_f2 = float(np.sum((y - mean_test) ** 2))
    den_train = float(np.sum((ytr - mean_train) ** 2))
    q2f1 = 1 - sse / den_f1 if den_f1 > 0 else np.nan
    q2f2 = 1 - sse / den_f2 if den_f2 > 0 else np.nan
    q2f3 = 1 - (sse / nt) / (den_train / ntr) if nt > 0 and ntr > 0 and den_train > 0 else np.nan

    # Regression-through-origin diagnostics.
    k = float(np.dot(y, p) / np.dot(p, p)) if np.dot(p, p) > 0 else np.nan       # y = k*p
    kp = float(np.dot(y, p) / np.dot(y, y)) if np.dot(y, y) > 0 else np.nan      # p = k'*y

    r2_ext = safe_r2(y, p)
    r0_2 = np.nan
    r0p_2 = np.nan
    if np.isfinite(k):
        pred_y0 = k * p
        den = np.sum((y - np.mean(y)) ** 2)
        r0_2 = 1 - np.sum((y - pred_y0) ** 2) / den if den > 0 else np.nan
    if np.isfinite(kp):
        pred_p0 = kp * y
        den = np.sum((p - np.mean(p)) ** 2)
        r0p_2 = 1 - np.sum((p - pred_p0) ** 2) / den if den > 0 else np.nan

    # rm2 metrics use squared correlation coefficient r^2 and zero-intercept R0^2.
    r = pearson_corr(y, p)
    r_sq = r * r if np.isfinite(r) else np.nan
    rm2 = r_sq * (1 - np.sqrt(abs(r_sq - r0_2))) if np.isfinite(r_sq) and np.isfinite(r0_2) else np.nan
    rm2p = r_sq * (1 - np.sqrt(abs(r_sq - r0p_2))) if np.isfinite(r_sq) and np.isfinite(r0p_2) else np.nan
    rm2_mean = np.nanmean([rm2, rm2p]) if np.isfinite(rm2) or np.isfinite(rm2p) else np.nan
    delta_rm2 = abs(rm2 - rm2p) if np.isfinite(rm2) and np.isfinite(rm2p) else np.nan

    gt1 = (r2_ext - r0_2) / r2_ext if np.isfinite(r2_ext) and abs(r2_ext) > 1e-12 and np.isfinite(r0_2) else np.nan
    gt2 = (r2_ext - r0p_2) / r2_ext if np.isfinite(r2_ext) and abs(r2_ext) > 1e-12 and np.isfinite(r0p_2) else np.nan

    return {
        "Q2F1": q2f1,
        "Q2F2": q2f2,
        "Q2F3": q2f3,
        "CCC": lin_ccc(y, p),
        "k": k,
        "k_prime": kp,
        "R0_2": r0_2,
        "R0prime_2": r0p_2,
        "r_squared_correlation": r_sq,
        "rm2": rm2,
        "rm2_prime": rm2p,
        "rm2_mean": rm2_mean,
        "delta_rm2": delta_rm2,
        "GT_ratio_R2_R0": gt1,
        "GT_ratio_R2_R0prime": gt2,
    }


def base_metrics(y_true, y_pred) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    return {
        "R2": safe_r2(y, p),
        "RMSE": rmse(y, p),
        "MAE": float(mean_absolute_error(y, p)),
        "MedAE": float(median_absolute_error(y, p)),
        "Pearson_R": pearson_corr(y, p),
        "CCC": lin_ccc(y, p),
        "Explained_variance": float(explained_variance_score(y, p)) if len(y) > 1 else np.nan,
    }


def evaluate_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    run_loo: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    pred_frames = []
    for name, model in models.items():
        p_train = model.predict(X_train)
        p_test = model.predict(X_test)
        p_cv = cross_val_predict(clone(model), X_train, y_train, cv=cv, n_jobs=1)

        for dataset, yt, yp in [
            ("Train", y_train, p_train),
            ("Held-out random test", y_test, p_test),
            ("5-fold CV", y_train, p_cv),
        ]:
            m = base_metrics(yt, yp)
            if dataset == "Held-out random test":
                m.update(qsar_external_metrics(y_train, yt, yp))
            else:
                for key in ["Q2F1", "Q2F2", "Q2F3", "k", "k_prime", "R0_2", "R0prime_2",
                            "r_squared_correlation", "rm2", "rm2_prime", "rm2_mean", "delta_rm2",
                            "GT_ratio_R2_R0", "GT_ratio_R2_R0prime"]:
                    m.setdefault(key, np.nan)
            m.update({"Model": name, "Dataset": dataset, "Q2_LOO": np.nan})
            rows.append(m)

        if run_loo:
            loo_pred = cross_val_predict(clone(model), X_train, y_train, cv=LeaveOneOut(), n_jobs=1)
            m = base_metrics(y_train, loo_pred)
            den = np.sum((y_train - np.mean(y_train)) ** 2)
            q2loo = 1 - np.sum((y_train - loo_pred) ** 2) / den if den > 0 else np.nan
            m.update({"Model": name, "Dataset": "LOO", "Q2_LOO": q2loo})
            rows.append(m)

        pred_frames.append(pd.DataFrame({
            "Model": name,
            "Dataset": ["Train"] * len(y_train) + ["Held-out random test"] * len(y_test),
            "row_position": np.r_[train_idx, test_idx],
            "experimental_c_c0": np.r_[y_train, y_test],
            "predicted_c_c0": np.r_[p_train, p_test],
        }))

    metrics = pd.DataFrame(rows)
    preds = pd.concat(pred_frames, ignore_index=True)
    preds["residual"] = preds["experimental_c_c0"] - preds["predicted_c_c0"]
    preds["absolute_error"] = preds["residual"].abs()
    return metrics, preds


# =============================================================================
# Grouped / leakage-resistant validation
# =============================================================================

def nested_grouped_validation(
    df: pd.DataFrame,
    candidate_raw: pd.DataFrame,
    y: np.ndarray,
    groups: Sequence[Any],
    scheme_name: str,
    splitter,
    model_template: Any,
    max_k: int,
    both_weight: float,
    fast_run: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    pred_rows = []
    selection_rows = []
    dummy = np.zeros(len(df))

    split_iter = splitter.split(dummy, y, np.asarray(groups))
    for fold_no, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        tr_idx = np.asarray(tr_idx, dtype=int)
        va_idx = np.asarray(va_idx, dtype=int)
        ytr = y[tr_idx]
        yva = y[va_idx]

        train_group_set = set(np.asarray(groups)[tr_idx].astype(str))
        validation_group_set = set(np.asarray(groups)[va_idx].astype(str))
        assert train_group_set.isdisjoint(validation_group_set), (
            f"Grouped-validation leakage detected in {scheme_name}, fold {fold_no}: "
            f"{sorted(train_group_set & validation_group_set)}"
        )

        Xf_tr, Xf_va, fixed_names, _ = fit_fixed_preprocessor(df, tr_idx, va_idx)
        Cp_tr, Cp_va, poll_names, _ = fit_pollutant_preprocessor(candidate_raw, tr_idx, va_idx)
        comp, _ = hybrid_components(Cp_tr, ytr, poll_names, cv_splits=3, random_state=RANDOM_STATE + fold_no)
        ranking = rank_hybrid(comp, both_weight)

        inner_series = df.loc[tr_idx, SERIES_COL].to_numpy()
        k_star, selected, _ = optimize_minimal_k(
            Xf_tr, Cp_tr, ytr, ranking, poll_names,
            max_k=max_k,
            n_estimators=20 if fast_run else 200,
            inner_groups=inner_series,
            inner_splits=3,
        )
        name_to_idx = {n: i for i, n in enumerate(poll_names)}
        sel = [name_to_idx[f] for f in selected]
        Xtr = np.hstack([Xf_tr, Cp_tr[:, sel]]) if sel else Xf_tr
        Xva = np.hstack([Xf_va, Cp_va[:, sel]]) if sel else Xf_va

        model = clone(model_template)
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        m = base_metrics(yva, pred)
        m.update(qsar_external_metrics(ytr, yva, pred))
        held_groups = sorted(set(np.asarray(groups)[va_idx].astype(str)))
        fold_rows.append({
            "scheme": scheme_name,
            "fold": fold_no,
            "n_train": len(tr_idx),
            "n_validation": len(va_idx),
            "n_train_series": df.loc[tr_idx, SERIES_COL].nunique(),
            "n_validation_groups": len(held_groups),
            "held_out_groups": json.dumps(held_groups, ensure_ascii=False),
            "group_overlap_count": 0,
            "selected_k": k_star,
            "selected_features": json.dumps(selected, ensure_ascii=False),
            "n_transformed_model_columns": Xtr.shape[1],
            **m,
        })
        selection_rows.append({
            "scheme": scheme_name,
            "fold": fold_no,
            "selected_k": k_star,
            "selected_features": json.dumps(selected, ensure_ascii=False),
            "top_descriptor": ranking.iloc[0]["feature"] if len(ranking) else None,
        })
        for row_pos, yt, yp in zip(va_idx, yva, pred):
            pred_rows.append({
                "scheme": scheme_name,
                "fold": fold_no,
                "row_position": int(row_pos),
                "experimental_c_c0": float(yt),
                "predicted_c_c0": float(yp),
                "residual": float(yt - yp),
                "absolute_error": float(abs(yt - yp)),
                STUDY_COL: df.loc[row_pos, STUDY_COL],
                SERIES_COL: df.loc[row_pos, SERIES_COL],
                POLLUTANT_COL: df.loc[row_pos, POLLUTANT_COL],
                CATALYST_COL: df.loc[row_pos, CATALYST_COL],
            })
        logging.info("%s: completed outer fold %d", scheme_name, fold_no)

    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(pred_rows)
    selections = pd.DataFrame(selection_rows)
    if not preds.empty:
        overall = base_metrics(preds["experimental_c_c0"], preds["predicted_c_c0"])
        # For grouped OOF predictions, Q2F2 is effectively the standard OOF Q2/R2 denominator.
        summary = pd.DataFrame([{
            "scheme": scheme_name,
            "n_predictions": len(preds),
            "n_folds": folds["fold"].nunique(),
            **overall,
            "selected_k_mode": int(folds["selected_k"].mode().iloc[0]) if len(folds) else np.nan,
            "selected_k_median": float(folds["selected_k"].median()) if len(folds) else np.nan,
            "note": "Outer-fold preprocessing and descriptor selection were re-fitted using outer-training data only; final model hyperparameters were frozen from the primary training-CV optimization.",
        }])
    else:
        summary = pd.DataFrame()
    return folds, preds, summary


def run_kinetic_series_grouped_validation(
    df: pd.DataFrame,
    candidate_raw: pd.DataFrame,
    y: np.ndarray,
    best_model: Any,
    max_k: int,
    both_weight: float,
    fast_run: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reviewer-focused nested grouped validation using Kinetic_series_ID only.

    Every complete kinetic series is held entirely in the outer training or
    outer validation partition. All learned preprocessing and molecular feature
    selection are re-fitted inside the outer training fold.
    """
    series_groups = df[SERIES_COL].astype(str).to_numpy()
    n_series = len(np.unique(series_groups))
    if n_series < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    splitter = GroupKFold(n_splits=min(5, n_series))
    folds, preds, summary = nested_grouped_validation(
        df=df,
        candidate_raw=candidate_raw,
        y=y,
        groups=series_groups,
        scheme_name="GroupKFold_by_kinetic_series",
        splitter=splitter,
        model_template=best_model,
        max_k=max_k,
        both_weight=both_weight,
        fast_run=fast_run,
    )
    selections = (
        folds[["scheme", "fold", "selected_k", "selected_features"]].copy()
        if not folds.empty else pd.DataFrame()
    )
    return folds, preds, summary, selections


# =============================================================================
# Bias, residuals, bootstrap uncertainty, Y-randomization
# =============================================================================

def bootstrap_bias_ci(diff: np.ndarray, n_boot: int = BOOTSTRAP_REPEATS) -> Dict[str, float]:
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(RANDOM_STATE)
    means = []
    medians = []
    n = len(diff)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = diff[idx]
        means.append(float(np.mean(sample)))
        medians.append(float(np.median(sample)))
    return {
        "mean_bias_CI_2.5": float(np.quantile(means, 0.025)),
        "mean_bias_CI_97.5": float(np.quantile(means, 0.975)),
        "median_bias_CI_2.5": float(np.quantile(medians, 0.025)),
        "median_bias_CI_97.5": float(np.quantile(medians, 0.975)),
    }


def paired_agreement_statistics(preds: pd.DataFrame, best_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = preds[preds["Model"].eq(best_name)].copy()
    rows = []
    bland_rows = []
    for dataset in ["Train", "Held-out random test", "All"]:
        sub = d if dataset == "All" else d[d["Dataset"].eq(dataset)]
        y = sub["experimental_c_c0"].to_numpy(float)
        p = sub["predicted_c_c0"].to_numpy(float)
        diff = y - p
        if len(diff) == 0:
            continue
        sh_stat, sh_p = stats.shapiro(diff) if 3 <= len(diff) <= 5000 else (np.nan, np.nan)
        try:
            wil_stat, wil_p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        except Exception:
            wil_stat, wil_p = np.nan, np.nan
        t_stat, t_p = stats.ttest_1samp(diff, 0.0) if len(diff) >= 2 else (np.nan, np.nan)
        ci = bootstrap_bias_ci(diff)
        bias = float(np.mean(diff))
        median_bias = float(np.median(diff))
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan
        lo = bias - 1.96 * sd if np.isfinite(sd) else np.nan
        hi = bias + 1.96 * sd if np.isfinite(sd) else np.nan
        rows.append({
            "Dataset": dataset,
            "n": len(diff),
            "Shapiro_W": sh_stat,
            "Shapiro_p": sh_p,
            "primary_test": "Wilcoxon signed-rank",
            "Wilcoxon_statistic": wil_stat,
            "Wilcoxon_p": wil_p,
            "supplementary_t_statistic": t_stat,
            "supplementary_t_p": t_p,
            "mean_bias_exp_minus_pred": bias,
            "median_bias_exp_minus_pred": median_bias,
            **ci,
            "Lin_CCC": lin_ccc(y, p),
            "Bland_Altman_LoA_lower": lo,
            "Bland_Altman_LoA_upper": hi,
        })
        for yy, pp in zip(y, p):
            bland_rows.append({
                "Dataset": dataset,
                "mean_exp_pred": (yy + pp) / 2,
                "difference_exp_minus_pred": yy - pp,
                "absolute_error": abs(yy - pp),
                "bias": bias,
                "LoA_lower": lo,
                "LoA_upper": hi,
            })
    return pd.DataFrame(rows), pd.DataFrame(bland_rows)


def bootstrap_test_metrics(y_test, pred_test, n_boot: int = BOOTSTRAP_REPEATS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = np.asarray(y_test, dtype=float)
    p = np.asarray(pred_test, dtype=float)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        rows.append({
            "bootstrap": b + 1,
            "R2": safe_r2(y[idx], p[idx]),
            "RMSE": rmse(y[idx], p[idx]),
            "MAE": mean_absolute_error(y[idx], p[idx]),
            "CCC": lin_ccc(y[idx], p[idx]),
        })
    samples = pd.DataFrame(rows)
    ci = []
    for metric in ["R2", "RMSE", "MAE", "CCC"]:
        s = samples[metric].dropna()
        ci.append({
            "metric": metric,
            "bootstrap_mean": s.mean(),
            "CI_2.5%": s.quantile(0.025),
            "CI_97.5%": s.quantile(0.975),
            "n_bootstrap": len(s),
        })
    return samples, pd.DataFrame(ci)


def y_randomization(X_train, y_train, n_perm: int = YRANDOM_REPEATS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    baseline = ExtraTreesRegressor(
        n_estimators=300, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1
    )
    observed = cross_val_predict(clone(baseline), X_train, y_train, cv=cv, n_jobs=1)
    observed_r2 = safe_r2(y_train, observed)
    observed_rmse = rmse(y_train, observed)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for i in range(n_perm):
        yp = rng.permutation(y_train)
        pred = cross_val_predict(clone(baseline), X_train, yp, cv=cv, n_jobs=1)
        rows.append({"permutation": i + 1, "CV_R2": safe_r2(yp, pred), "CV_RMSE": rmse(yp, pred)})
    null = pd.DataFrame(rows)
    p_emp = (1 + int((null["CV_R2"] >= observed_r2).sum())) / (n_perm + 1)
    summary = pd.DataFrame([{
        "observed_CV_R2": observed_r2,
        "observed_CV_RMSE": observed_rmse,
        "null_mean_R2": null["CV_R2"].mean(),
        "null_mean_RMSE": null["CV_RMSE"].mean(),
        "empirical_p_value_R2": p_emp,
        "n_permutations": n_perm,
    }])
    return null, summary


# =============================================================================
# Applicability domain and interpretability
# =============================================================================

def applicability_domain(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    pred_train: np.ndarray,
    pred_test: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    n, p = X_train.shape
    h_star = 3.0 * (p + 1) / n
    Xaug = np.column_stack([np.ones(n), X_train])
    inv_xtx = np.linalg.pinv(Xaug.T @ Xaug)
    h_train = np.diag(Xaug @ inv_xtx @ Xaug.T)
    Xte_aug = np.column_stack([np.ones(len(X_test)), X_test])
    h_test = np.array([row @ inv_xtx @ row.T for row in Xte_aug])

    r_train = y_train - pred_train
    r_test = y_test - pred_test
    sigma = np.std(r_train, ddof=1)
    sr_train = r_train / sigma if sigma > 0 else np.full_like(r_train, np.nan)
    sr_test = r_test / sigma if sigma > 0 else np.full_like(r_test, np.nan)

    mu = np.mean(X_train, axis=0)
    cov = np.cov(X_train, rowvar=False)
    cov_inv = np.linalg.pinv(np.atleast_2d(cov))

    def md(X):
        D = X - mu
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", D, cov_inv, D), 0))

    md_train = md(X_train)
    md_test = md(X_test)
    dist_thr = float(np.quantile(md_train, 0.95))

    ad = pd.DataFrame({
        "dataset": ["Train"] * n + ["Held-out random test"] * len(X_test),
        "row_position": np.r_[train_idx, test_idx],
        "experimental_c_c0": np.r_[y_train, y_test],
        "predicted_c_c0": np.r_[pred_train, pred_test],
        "standardized_residual": np.r_[sr_train, sr_test],
        "leverage": np.r_[h_train, h_test],
        "h_star": h_star,
        "actual_transformed_feature_count_p": p,
        "mahalanobis_distance": np.r_[md_train, md_test],
        "distance_95pct_training_threshold": dist_thr,
    })
    ad["inside_williams"] = (ad["leverage"] <= h_star) & (ad["standardized_residual"].abs() <= 3)
    ad["inside_distance_domain"] = ad["mahalanobis_distance"] <= dist_thr
    ad["inside_both_diagnostics"] = ad["inside_williams"] & ad["inside_distance_domain"]
    return ad


def feature_group(feature: str, selected_pollutant: List[str]) -> str:
    if feature.startswith("Irradiation_source_"):
        return "Irradiation"
    base = feature.replace("__missing", "")
    if base in ["MW", "Band_gap_Eg_eV", "BET_m2_g"]:
        return "Catalyst physicochemical"
    if base in ["Catalyst_dose_ug_mL", "Solution_volume_mL", "Initial_concentration_mg_L", "Time_min", "pH"]:
        return "Operational"
    if base == "Light_power_W":
        return "Irradiation"
    if feature in selected_pollutant:
        return "Pollutant selected information"
    return "Missingness / other"


def feature_importance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    selected_pollutant: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if hasattr(model, "feature_importances_"):
        built = pd.DataFrame({
            "feature": feature_names,
            "built_in_importance": np.asarray(model.feature_importances_, dtype=float),
        }).sort_values("built_in_importance", ascending=False)
    else:
        built = pd.DataFrame({"feature": feature_names, "built_in_importance": np.nan})

    perm = permutation_importance(
        model, X_test, y_test, scoring="neg_root_mean_squared_error",
        n_repeats=30, random_state=RANDOM_STATE, n_jobs=1,
    )
    p = pd.DataFrame({
        "feature": feature_names,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_sd": perm.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)
    p["information_group"] = [feature_group(f, selected_pollutant) for f in p["feature"]]
    temp = p.assign(positive=p["permutation_importance_mean"].clip(lower=0))
    grouped = temp.groupby("information_group", as_index=False)["positive"].sum()
    total = grouped["positive"].sum()
    grouped["relative_grouped_importance_percent"] = 100 * grouped["positive"] / total if total > 0 else np.nan
    return built, p, grouped


def shap_analysis(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    out: Path,
    n_dependence: int = 6,
    make_plots: bool = True,
) -> pd.DataFrame:
    if not SHAP_AVAILABLE:
        (out / "reports" / "shap_error.txt").write_text(f"SHAP unavailable: {SHAP_IMPORT_ERROR}", encoding="utf-8")
        return pd.DataFrame()
    try:
        bg = pd.DataFrame(X_train, columns=feature_names)
        xt = pd.DataFrame(X_test, columns=feature_names)
        if len(bg) > 150:
            bg = bg.sample(150, random_state=RANDOM_STATE)

        # TreeExplainer is faster and exact for tree ensembles; generic Explainer otherwise.
        if isinstance(model, (XGBRegressor, ExtraTreesRegressor, RandomForestRegressor)):
            explainer = shap.TreeExplainer(model)
            raw_values = explainer.shap_values(xt)
            vals = np.asarray(raw_values)
            if vals.ndim == 3:
                vals = vals[:, :, 0]
            base_values = np.repeat(np.nan, len(xt))
            explanation = None
        else:
            explainer = shap.Explainer(model, bg)
            explanation = explainer(xt)
            vals = np.asarray(explanation.values)
            if vals.ndim == 3:
                vals = vals[:, :, 0]
            base_values = np.asarray(explanation.base_values)

        mean_abs = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_SHAP": np.mean(np.abs(vals), axis=0),
        }).sort_values("mean_abs_SHAP", ascending=False).reset_index(drop=True)
        save_df(pd.DataFrame(vals, columns=feature_names), out / "tables" / "shap_values.xlsx")

        if make_plots:
            # Robust legacy plotting API works with ndarray values and DataFrame data.
            plt.figure(figsize=(8, 6))
            shap.summary_plot(vals, xt, plot_type="bar", show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(out / "figures" / "Figure_SHAP_bar.png", dpi=400, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(8, 6))
            shap.summary_plot(vals, xt, show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(out / "figures" / "Figure_SHAP_beeswarm.png", dpi=400, bbox_inches="tight")
            plt.close()

            dep_dir = out / "figures" / "shap_dependence"
            dep_dir.mkdir(parents=True, exist_ok=True)
            for feature in mean_abs["feature"].head(n_dependence):
                plt.figure(figsize=(7, 5.5))
                shap.dependence_plot(feature, vals, xt, interaction_index="auto", show=False)
                plt.tight_layout()
                safe = feature.replace("/", "_").replace(" ", "_").replace("µ", "u")
                plt.savefig(dep_dir / f"SHAP_dependence_{safe}.png", dpi=400, bbox_inches="tight")
                plt.close()
        return mean_abs
    except Exception:
        (out / "reports" / "shap_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return pd.DataFrame()


# =============================================================================
# Systematic error-cluster analysis
# =============================================================================

def error_group_table(data: pd.DataFrame, group_col: str, label: str) -> pd.DataFrame:
    rows = []
    for group, g in data.groupby(group_col, dropna=False, observed=True):
        y = g["experimental_c_c0"].to_numpy(float)
        p = g["predicted_c_c0"].to_numpy(float)
        rows.append({
            "group_type": label,
            "group": str(group),
            "n": len(g),
            "R2": safe_r2(y, p),
            "RMSE": rmse(y, p),
            "MAE": mean_absolute_error(y, p),
            "median_absolute_error": np.median(np.abs(y - p)),
            "max_absolute_error": np.max(np.abs(y - p)),
            "mean_signed_error": np.mean(y - p),
        })
    return pd.DataFrame(rows).sort_values(["MAE", "n"], ascending=[False, False])


def error_cluster_analysis(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    best_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = preds[preds["Model"].eq(best_name)].copy()
    meta = df[[
        STUDY_COL, SERIES_COL, CATALYST_COL, POLLUTANT_COL, "Catalyst_pollutant_ID",
        IRRADIATION_COL, "Irradiation_label", "pH",
    ]].copy()
    meta["row_position"] = meta.index
    d = d.merge(meta, on="row_position", how="left")
    d["pH_bin"] = pd.cut(d["pH"], bins=[-np.inf, 4, 6, 8, 10, np.inf], labels=["≤4", "4–6", "6–8", "8–10", ">10"])

    tables = []
    for col, label in [
        (POLLUTANT_COL, "Pollutant"),
        (CATALYST_COL, "Catalyst"),
        (STUDY_COL, "Study"),
        (SERIES_COL, "Kinetic series"),
        ("Catalyst_pollutant_ID", "Catalyst–pollutant"),
        ("Irradiation_label", "Irradiation source"),
        ("pH_bin", "pH range"),
    ]:
        tables.append(error_group_table(d, col, label))
    grouped = pd.concat(tables, ignore_index=True)
    top_errors = d.sort_values("absolute_error", ascending=False).head(30).copy()
    return grouped, top_errors



# =============================================================================
# pH ablation and molecular chemical-space diagnostics
# =============================================================================

def pH_ablation_v6(
    model_template: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    final_feature_names: List[str],
) -> pd.DataFrame:
    """Compare the same observations/split with and without pH.

    Other transformed columns are identical; model hyperparameters are frozen so
    the comparison isolates the incremental predictive contribution of pH.
    """
    if "pH" not in final_feature_names:
        return pd.DataFrame({"message": ["pH transformed column not found; ablation not performed."]})
    ph_idx = final_feature_names.index("pH")
    keep_idx = [i for i in range(len(final_feature_names)) if i != ph_idx]
    rows = []
    for label, Xtr, Xte in [
        ("With_pH", X_train, X_test),
        ("Without_pH", X_train[:, keep_idx], X_test[:, keep_idx]),
    ]:
        model = clone(model_template)
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = -cross_val_score(
            clone(model_template), Xtr, y_train,
            cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1,
        )
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        rows.append({
            "configuration": label,
            "n_transformed_features": Xtr.shape[1],
            "CV_RMSE_mean": float(np.mean(cv_scores)),
            "CV_RMSE_sd": float(np.std(cv_scores, ddof=1)),
            "Test_R2": safe_r2(y_test, pred),
            "Test_RMSE": rmse(y_test, pred),
            "Test_MAE": float(mean_absolute_error(y_test, pred)),
            "Test_CCC": lin_ccc(y_test, pred),
        })
    out = pd.DataFrame(rows)
    if len(out) == 2:
        with_row = out[out["configuration"].eq("With_pH")].iloc[0]
        without_row = out[out["configuration"].eq("Without_pH")].iloc[0]
        out["Delta_CV_RMSE_vs_without_pH"] = out["CV_RMSE_mean"] - float(without_row["CV_RMSE_mean"])
        out["Delta_Test_RMSE_vs_without_pH"] = out["Test_RMSE"] - float(without_row["Test_RMSE"])
        out["Delta_Test_R2_vs_without_pH"] = out["Test_R2"] - float(without_row["Test_R2"])
    return out


def plot_pH_ablation_v6(table: pd.DataFrame, out: Path) -> None:
    if table.empty or "configuration" not in table.columns or "Test_RMSE" not in table.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    axes[0].bar(table["configuration"], table["CV_RMSE_mean"], yerr=table.get("CV_RMSE_sd"), capsize=4)
    axes[0].set_ylabel("Training 5-fold CV RMSE")
    axes[0].set_title("A. Cross-validated pH ablation")
    axes[1].bar(table["configuration"], table["Test_RMSE"])
    axes[1].set_ylabel("Internal held-out RMSE")
    axes[1].set_title("B. Held-out pH ablation")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_figure(fig, out / "figures" / "Figure_pH_ablation_V6", dpi=600, vector=True)


def chemical_space_mw_slogp_v6(
    df: pd.DataFrame,
    candidate_raw: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    out: Path,
    make_plot: bool = True,
) -> pd.DataFrame:
    """QSAR-style MW versus SlogP chemical-space diagnostic.

    This is descriptive only and does not participate in feature selection.
    """
    slogp_col = "RDKit_MolLogP"
    if slogp_col not in candidate_raw.columns:
        return pd.DataFrame({"message": ["RDKit_MolLogP not available in descriptor matrix."]})
    table = pd.DataFrame({
        "row_position": np.arange(len(df)),
        "original_index": df["original_index"].to_numpy(),
        "MW": pd.to_numeric(df["MW"], errors="coerce").to_numpy(),
        "SlogP_RDKit_MolLogP": pd.to_numeric(candidate_raw[slogp_col], errors="coerce").to_numpy(),
        POLLUTANT_COL: df[POLLUTANT_COL].astype(str).to_numpy(),
        CATALYST_COL: df[CATALYST_COL].astype(str).to_numpy(),
        "split": "",
    })
    table.loc[train_idx, "split"] = "train"
    table.loc[test_idx, "split"] = "held_out_random_test"
    save_df(table, out / "tables" / "MW_SlogP_chemical_space.xlsx")
    if make_plot:
        fig, ax = plt.subplots(figsize=(8.2, 6.3))
        for split, marker in [("train", "o"), ("held_out_random_test", "s")]:
            s = table[table["split"].eq(split)]
            ax.scatter(s["MW"], s["SlogP_RDKit_MolLogP"], marker=marker, alpha=.72, label=split)
        # Label unique pollutant centers to avoid unreadable row-by-row text.
        centers = table.groupby(POLLUTANT_COL, as_index=False)[["MW", "SlogP_RDKit_MolLogP"]].mean()
        for _, r in centers.iterrows():
            ax.annotate(str(r[POLLUTANT_COL]), (r["MW"], r["SlogP_RDKit_MolLogP"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Molecular weight")
        ax.set_ylabel("RDKit MolLogP (SlogP)")
        ax.set_title("Pollutant chemical space: MW versus SlogP")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, out / "figures" / "Figure_MW_SlogP_chemical_space_V6", dpi=600, vector=True)
    return table

# =============================================================================
# Kinetic benchmark: zero-order, PFO, PSO, Weibull
# =============================================================================

def fit_zero_order(t: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
    mask = np.isfinite(t) & np.isfinite(y)
    t = np.asarray(t[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if len(t) < 2 or np.sum(t ** 2) <= 0:
        return None
    k = max(0.0, float(np.dot(t, 1 - y) / np.dot(t, t)))
    return {"k": k}


def predict_zero(t: np.ndarray, pars: Dict[str, float]) -> np.ndarray:
    return 1.0 - pars["k"] * np.asarray(t, dtype=float)


def fit_pfo(t: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t = np.asarray(t[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if len(t) < 2 or np.sum(t ** 2) <= 0:
        return None
    z = np.log(1.0 / np.clip(y, 1e-12, None))
    k = max(0.0, float(np.dot(t, z) / np.dot(t, t)))
    return {"k": k}


def predict_pfo(t: np.ndarray, pars: Dict[str, float]) -> np.ndarray:
    return np.exp(-pars["k"] * np.asarray(t, dtype=float))


def fit_pso(t: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
    # Normalized second-order decay: c/c0 = 1/(1 + k*t)
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t = np.asarray(t[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if len(t) < 2 or np.sum(t ** 2) <= 0:
        return None
    z = 1.0 / np.clip(y, 1e-12, None) - 1.0
    k = max(0.0, float(np.dot(t, z) / np.dot(t, t)))
    return {"k": k}


def predict_pso(t: np.ndarray, pars: Dict[str, float]) -> np.ndarray:
    return 1.0 / (1.0 + pars["k"] * np.asarray(t, dtype=float))


def fit_weibull(t: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t = np.asarray(t[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if len(t) < 3 or np.unique(t).size < 3:
        return None

    def f(tt, k, beta):
        return np.exp(-np.power(np.maximum(k * tt, 0), beta))

    pfo = fit_pfo(t, y)
    k0 = max(1e-6, pfo["k"] if pfo else 0.01)
    try:
        popt, _ = curve_fit(
            f, t, y,
            p0=[k0, 1.0],
            bounds=([1e-10, 0.1], [10.0, 5.0]),
            maxfev=20000,
        )
        return {"k": float(popt[0]), "beta": float(popt[1])}
    except Exception:
        return None


def predict_weibull(t: np.ndarray, pars: Dict[str, float]) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.exp(-np.power(np.maximum(pars["k"] * t, 0), pars["beta"]))


def kinetic_benchmark(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    model_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tmp = df.copy()
    split = pd.Series("", index=tmp.index)
    split.loc[train_idx] = "Train"
    split.loc[test_idx] = "Test"
    tmp["split"] = split
    ml_lookup = pd.Series(index=tmp.index, dtype=float)
    ml_lookup.loc[train_idx] = model.predict(X_train)
    ml_lookup.loc[test_idx] = model.predict(X_test)
    tmp["ML_prediction"] = ml_lookup

    fitters = {
        "Zero_order": (fit_zero_order, predict_zero),
        "PFO": (fit_pfo, predict_pfo),
        "PSO": (fit_pso, predict_pso),
        "Weibull": (fit_weibull, predict_weibull),
    }

    series_rows = []
    point_rows = []
    for sid, g in tmp.groupby(SERIES_COL, sort=False):
        tr = g[g["split"].eq("Train")]
        te = g[g["split"].eq("Test")]
        if len(te) == 0:
            continue
        yt = te[TARGET_COL].to_numpy(float)
        mlp = te["ML_prediction"].to_numpy(float)
        series_rows.append({
            SERIES_COL: sid,
            "model": model_name,
            "n_train_points": len(tr),
            "n_test_points": len(te),
            "fit_reliability": "limited (<3 train points)" if len(tr) < 3 else "adequate (>=3 train points)",
            "R2_test": safe_r2(yt, mlp),
            "RMSE_test": rmse(yt, mlp),
            "MAE_test": mean_absolute_error(yt, mlp),
            CATALYST_COL: g[CATALYST_COL].iloc[0],
            POLLUTANT_COL: g[POLLUTANT_COL].iloc[0],
            "parameters": "ML multivariable model; no single kinetic-rate parameter",
        })
        # Preserve point-level ML predictions so that the ML model and every
        # kinetic equation can be compared on exactly the same held-out points.
        for idx, tt, yy, pp in zip(te.index, te["Time_min"].to_numpy(float), yt, mlp):
            point_rows.append({
                SERIES_COL: sid,
                "row_position": int(idx),
                "time_min": float(tt),
                "model": model_name,
                "experimental_c_c0": float(yy),
                "predicted_c_c0": float(pp),
                "residual": float(yy - pp),
                "absolute_error": float(abs(yy - pp)),
                CATALYST_COL: g[CATALYST_COL].iloc[0],
                POLLUTANT_COL: g[POLLUTANT_COL].iloc[0],
            })

        for kin_name, (fitfun, predfun) in fitters.items():
            pars = fitfun(tr["Time_min"].to_numpy(float), tr[TARGET_COL].to_numpy(float))
            if pars is None:
                continue
            pred = predfun(te["Time_min"].to_numpy(float), pars)
            row = {
                SERIES_COL: sid,
                "model": kin_name,
                "n_train_points": len(tr),
                "n_test_points": len(te),
                "fit_reliability": "limited (<3 train points)" if len(tr) < 3 else "adequate (>=3 train points)",
                "R2_test": safe_r2(yt, pred),
                "RMSE_test": rmse(yt, pred),
                "MAE_test": mean_absolute_error(yt, pred),
                CATALYST_COL: g[CATALYST_COL].iloc[0],
                POLLUTANT_COL: g[POLLUTANT_COL].iloc[0],
                "parameters": json.dumps(pars),
            }
            series_rows.append(row)
            for idx, tt, yy, pp in zip(te.index, te["Time_min"].to_numpy(float), yt, pred):
                point_rows.append({
                    SERIES_COL: sid,
                    "row_position": int(idx),
                    "time_min": float(tt),
                    "model": kin_name,
                    "experimental_c_c0": float(yy),
                    "predicted_c_c0": float(pp),
                    "residual": float(yy - pp),
                    "absolute_error": float(abs(yy - pp)),
                    CATALYST_COL: g[CATALYST_COL].iloc[0],
                    POLLUTANT_COL: g[POLLUTANT_COL].iloc[0],
                })

    series = pd.DataFrame(series_rows)
    points = pd.DataFrame(point_rows)
    comparisons = []
    if not series.empty:
        ml = series[series["model"].eq(model_name)][[SERIES_COL, "RMSE_test", "MAE_test"]].rename(
            columns={"RMSE_test": "ML_RMSE", "MAE_test": "ML_MAE"}
        )
        for kin_name in fitters:
            kin = series[series["model"].eq(kin_name)][[SERIES_COL, "RMSE_test", "MAE_test"]].rename(
                columns={"RMSE_test": "Kinetic_RMSE", "MAE_test": "Kinetic_MAE"}
            )
            merged = ml.merge(kin, on=SERIES_COL, how="inner")
            if len(merged):
                delta = merged["Kinetic_RMSE"] - merged["ML_RMSE"]
                try:
                    stat, p = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
                except Exception:
                    stat, p = np.nan, np.nan
                comparisons.append({
                    "kinetic_model": kin_name,
                    "n_paired_series": len(merged),
                    "median_delta_RMSE_kinetic_minus_ML": float(np.median(delta)),
                    "mean_delta_RMSE_kinetic_minus_ML": float(np.mean(delta)),
                    "ML_better_series_count": int((delta > 0).sum()),
                    "Kinetic_better_series_count": int((delta < 0).sum()),
                    "Wilcoxon_statistic": stat,
                    "Wilcoxon_p": p,
                })
    return series, points, pd.DataFrame(comparisons)


# =============================================================================
# Publication figures
# =============================================================================

def plot_minimal_k(kdf: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(kdf["k"], kdf["mean_cv_rmse"], yerr=kdf["sd_cv_rmse"], marker="o", capsize=3)
    threshold = 1.02 * kdf["mean_cv_rmse"].min()
    ax.axhline(threshold, linestyle="--", label="2% near-optimal threshold")
    sel = kdf[kdf["selected_k_star"]].iloc[0]
    ax.scatter(sel["k"], sel["mean_cv_rmse"], marker="*", s=180, label=f"Selected k*={int(sel['k'])}")
    ax.set_xlabel("Number of selected pollutant features (k)")
    ax.set_ylabel("Cross-validated RMSE")
    ax.set_title("Minimal-k optimization")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out / "figures" / "Figure_minimal_k", dpi=450, vector=True)


def plot_weight_sensitivity(table: pd.DataFrame, out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(table["both_weight"], table["selected_k_cv_rmse"], marker="o")
    ax1.set_xlabel("Weight assigned to joint Lasso+MI support")
    ax1.set_ylabel("CV RMSE at selected k*")
    ax2 = ax1.twinx()
    ax2.plot(table["both_weight"], table["selected_k"], marker="s", linestyle="--")
    ax2.set_ylabel("Selected k*")
    ax1.set_title("Hybrid-score weight sensitivity")
    fig.tight_layout()
    save_figure(fig, out / "figures" / "Figure_hybrid_weight_sensitivity", dpi=450, vector=True)


def plot_grouped_validation(random_test_metrics: pd.Series, grouped_summary: pd.DataFrame, out: Path) -> None:
    rows = [{
        "scheme": "Random_80_20_internal_holdout",
        "R2": random_test_metrics["R2"],
        "RMSE": random_test_metrics["RMSE"],
        "MAE": random_test_metrics["MAE"],
    }]
    if not grouped_summary.empty:
        for _, r in grouped_summary.iterrows():
            rows.append({"scheme": r["scheme"], "R2": r["R2"], "RMSE": r["RMSE"], "MAE": r["MAE"]})
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].barh(d["scheme"], d["R2"])
    axes[0].set_xlabel("R²")
    axes[0].set_title("Validation R²")
    axes[1].barh(d["scheme"], d["RMSE"])
    axes[1].set_xlabel("RMSE")
    axes[1].set_title("Validation RMSE")
    fig.tight_layout()
    save_figure(fig, out / "figures" / "Figure_random_vs_grouped_validation", dpi=450, vector=True)


def plot_williams(ad: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for split, marker in [("Train", "o"), ("Held-out random test", "s")]:
        s = ad[ad["dataset"].eq(split)]
        ax.scatter(s["leverage"], s["standardized_residual"], marker=marker, alpha=0.75, label=split)
    h_star = ad["h_star"].iloc[0]
    ax.axhline(3, linestyle="--")
    ax.axhline(-3, linestyle="--")
    ax.axvline(h_star, linestyle="--", label=f"h*={h_star:.3f}")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized residual")
    ax.set_title("Williams applicability-domain diagnostic")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out / "figures" / "Figure_Williams_AD", dpi=450, vector=True)


def plot_error_heatmap(df: pd.DataFrame, preds: pd.DataFrame, best_name: str, out: Path) -> None:
    if not SEABORN_AVAILABLE:
        return
    d = preds[(preds["Model"].eq(best_name)) & (preds["Dataset"].eq("Held-out random test"))].copy()
    meta = df[[CATALYST_COL, POLLUTANT_COL]].copy()
    meta["row_position"] = meta.index
    d = d.merge(meta, on="row_position", how="left")
    if d.empty:
        return
    pivot = d.pivot_table(index=CATALYST_COL, columns=POLLUTANT_COL, values="absolute_error", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * pivot.shape[1] + 4), max(5, 0.45 * pivot.shape[0] + 3)))
    sns.heatmap(pivot, annot=True, fmt=".3f", ax=ax)
    ax.set_title("Held-out MAE by catalyst–pollutant system")
    fig.tight_layout()
    save_figure(fig, out / "figures" / "Figure_error_cluster_heatmap", dpi=400, vector=True)


def plot_kinetic_deltas(series: pd.DataFrame, model_name: str, out: Path) -> None:
    if series.empty:
        return
    ml = series[series["model"].eq(model_name)][[SERIES_COL, "RMSE_test"]].rename(columns={"RMSE_test": "ML_RMSE"})
    long = []
    for kin in ["Zero_order", "PFO", "PSO", "Weibull"]:
        k = series[series["model"].eq(kin)][[SERIES_COL, "RMSE_test"]].rename(columns={"RMSE_test": "Kinetic_RMSE"})
        m = ml.merge(k, on=SERIES_COL, how="inner")
        for v in (m["Kinetic_RMSE"] - m["ML_RMSE"]):
            long.append({"kinetic_model": kin, "delta_RMSE": v})
    d = pd.DataFrame(long)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if SEABORN_AVAILABLE:
        sns.boxplot(data=d, x="kinetic_model", y="delta_RMSE", ax=ax)
        sns.stripplot(data=d, x="kinetic_model", y="delta_RMSE", ax=ax, alpha=0.6)
    else:
        grouped = [d.loc[d["kinetic_model"].eq(k), "delta_RMSE"].to_numpy() for k in d["kinetic_model"].unique()]
        ax.boxplot(grouped, labels=d["kinetic_model"].unique())
    ax.axhline(0, linestyle="--")
    ax.set_ylabel("ΔRMSE = kinetic RMSE − ML RMSE")
    ax.set_xlabel("")
    ax.set_title("Series-level kinetic benchmark differences")
    fig.tight_layout()
    save_figure(fig, out / "figures" / "Figure_kinetic_delta_RMSE", dpi=450, vector=True)



# =============================================================================
# Reviewer-focused audit, MOD/parity plots, validation-group workbook,
# and expanded kinetic presentation
# =============================================================================

def feature_selection_architecture_audit(candidate_raw, pollutant_names, ranking, selected_pollutant, fixed_names, final_feature_names):
    """Verify: SMILES -> RDKit/lambda block -> feature selection -> append to fixed block."""
    candidate_set=set(candidate_raw.columns); selected_set=set(selected_pollutant); fixed_set=set(FIXED_NUMERIC+[IRRADIATION_COL])
    assert selected_set.issubset(set(pollutant_names)), "Selected pollutant features must come only from filtered lambda_max + RDKit candidates."
    assert not (selected_set & fixed_set), "Fixed catalyst/process/pH/light inputs must not be molecular-feature-selected."
    assert all((f==LAMBDA_COL) or str(f).startswith("RDKit_") for f in candidate_raw.columns), "Pollutant selection block may contain only lambda_max + RDKit descriptors."
    assert len(final_feature_names)==len(fixed_names)+len(selected_pollutant), "Final matrix must equal transformed fixed block + selected pollutant features."
    flow=pd.DataFrame([
        [1,"SMILES input","Pollutant structure",SMILES_COL,"Canonicalize with RDKit"],
        [2,"RDKit generation","Pollutant molecular information",f"{sum(str(c).startswith('RDKit_') for c in candidate_raw.columns)} RDKit descriptors","Generated from canonical SMILES"],
        [3,"Pollutant candidate block","Feature-selection block only",f"{len(candidate_raw.columns)} variables","lambda_max + RDKit descriptors"],
        [4,"Training-only preprocessing","Feature-selection block only",f"{len(pollutant_names)} variables after zero-variance filtering","Median imputation/scaling/variance filter fit on training only"],
        [5,"Parallel selectors","Feature-selection block only","LassoCV || mutual information","Neither pre-filters the other"],
        [6,"Hybrid ranking + minimal-k","Feature-selection block only",", ".join(selected_pollutant) if selected_pollutant else "k*=0","Combined-score-first ranking + 2% minimal-k rule"],
        [7,"Fixed block","Always retained",f"{len(fixed_names)} transformed columns","Catalyst/process/pH/light; irradiation source one-hot encoded"],
        [8,"Final model matrix","Concatenation",f"{len(final_feature_names)} transformed columns","Fixed transformed block + selected pollutant features"],
    ],columns=["stage","operation","information_block","output","rule"])
    cols=[c for c in ["feature","feature_type","lasso_selected","mi_top10","selected_by_both","combined_hybrid_score","final_hybrid_rank"] if c in ranking.columns]
    audit=ranking[cols].copy(); audit["in_candidate_block"]=audit["feature"].isin(candidate_set); audit["selected_final"]=audit["feature"].isin(selected_set); audit["selection_scope"]="pollutant block only"
    return flow,audit


def model_mod_analysis_table(preds):
    rows=[]
    for (model,dataset),g in preds.groupby(["Model","Dataset"],sort=False):
        mod=pd.to_numeric(g["absolute_error"],errors="coerce").dropna().to_numpy(float); res=pd.to_numeric(g["residual"],errors="coerce").dropna().to_numpy(float)
        if not len(mod): continue
        rows.append({"Model":model,"Dataset":dataset,"n":len(mod),"MOD_mean":float(np.mean(mod)),"MOD_median":float(np.median(mod)),"MOD_SD":float(np.std(mod,ddof=1)) if len(mod)>1 else np.nan,"MOD_Q1":float(np.quantile(mod,.25)),"MOD_Q3":float(np.quantile(mod,.75)),"MOD_P95":float(np.quantile(mod,.95)),"MOD_max":float(np.max(mod)),"mean_signed_error_exp_minus_pred":float(np.mean(res)) if len(res) else np.nan})
    return pd.DataFrame(rows)


def plot_each_model_pair_mod_v5(metrics,preds,out):
    outdir=out/"figures"/"paired_model_MOD"; outdir.mkdir(parents=True,exist_ok=True)
    for model_name in preds["Model"].dropna().unique():
        d=preds[preds["Model"].eq(model_name)].copy()
        if d.empty: continue
        safe=str(model_name).replace(" ","_").replace("/","_"); norm=plt.Normalize(0,max(float(d["absolute_error"].max()),1e-8)); fig,axes=plt.subplots(1,2,figsize=(13.2,5.8))
        for dataset,marker in [("Train","o"),("Held-out random test","s")]:
            s=d[d["Dataset"].eq(dataset)]
            if s.empty: continue
            axes[0].scatter(s["experimental_c_c0"],s["predicted_c_c0"],c=s["absolute_error"],cmap="viridis",norm=norm,marker=marker,alpha=.82,edgecolor="black",linewidth=.25,label=dataset)
        lo=min(float(d["experimental_c_c0"].min()),float(d["predicted_c_c0"].min())); hi=max(float(d["experimental_c_c0"].max()),float(d["predicted_c_c0"].max())); axes[0].plot([lo,hi],[lo,hi],"--",color="black",lw=1.2,label="1:1 identity line")
        axes[0].set_xlabel("Experimental c/c₀"); axes[0].set_ylabel("Predicted c/c₀"); axes[0].set_title(f"A. {model_name}: paired prediction plot"); axes[0].legend(fontsize=7)
        mt=metrics[(metrics["Model"].eq(model_name))&(metrics["Dataset"].eq("Held-out random test"))]
        if not mt.empty:
            r=mt.iloc[0]; axes[0].text(.04,.96,f"Held-out test\nR²={r['R2']:.3f}\nRMSE={r['RMSE']:.3f}\nMAE={r['MAE']:.3f}",transform=axes[0].transAxes,va="top",bbox=dict(boxstyle="round",facecolor="white",alpha=.85),fontsize=8)
        dd=d[d["Dataset"].isin(["Train","Held-out random test"])].copy()
        if SEABORN_AVAILABLE:
            sns.boxplot(data=dd,x="Dataset",y="absolute_error",ax=axes[1]); sns.stripplot(data=dd,x="Dataset",y="absolute_error",ax=axes[1],alpha=.35,size=3)
        else:
            axes[1].boxplot([dd.loc[dd["Dataset"].eq(s),"absolute_error"].dropna().to_numpy() for s in ["Train","Held-out random test"]],labels=["Train","Held-out test"])
        axes[1].set_ylabel("MOD = |experimental − predicted|"); axes[1].set_xlabel(""); axes[1].set_title("B. Magnitude-of-deviation analysis")
        sm=plt.cm.ScalarMappable(cmap="viridis",norm=norm); sm.set_array([]); fig.colorbar(sm,ax=axes[0],label="MOD = absolute prediction error"); fig.tight_layout(); save_figure(fig,outdir/f"Figure_pair_MOD_{safe}",dpi=600,vector=True)


def validation_group_tables(df, split_table, grouped_folds, grouped_preds, grouped_selections):
    """Export exact memberships used for random and kinetic-series validation.

    Study, pollutant and catalyst identities are retained as metadata, but they
    are not presented as separate grouped-validation schemes in V6.
    """
    row_groups = df[["original_index", STUDY_COL, SERIES_COL, POLLUTANT_COL, CATALYST_COL, REF_COL]].copy()
    row_groups["row_position"] = np.arange(len(df))
    row_groups = row_groups.merge(split_table[["row_position", "split"]], on="row_position", how="left")

    series_rows = []
    for gid, g in row_groups.groupby(SERIES_COL, dropna=False, sort=False):
        series_rows.append({
            "Kinetic_series_ID": str(gid),
            "n_rows": len(g),
            "row_positions": json.dumps(g["row_position"].astype(int).tolist()),
            "original_indices": json.dumps(g["original_index"].astype(int).tolist()),
            "studies": json.dumps(sorted(g[STUDY_COL].astype(str).unique().tolist())),
            "pollutants": json.dumps(sorted(g[POLLUTANT_COL].astype(str).unique().tolist())),
            "catalysts": json.dumps(sorted(g[CATALYST_COL].astype(str).unique().tolist())),
        })
    series_groups = pd.DataFrame(series_rows)

    return {
        "Row_groups": row_groups,
        "Random_split": split_table.copy(),
        "Kinetic_series_groups": series_groups,
        "GroupKFold_assignments": grouped_preds[["fold", "row_position", SERIES_COL, STUDY_COL, POLLUTANT_COL, CATALYST_COL]].copy()
            if not grouped_preds.empty else pd.DataFrame({"message": ["Grouped validation not run"]}),
        "Outer_fold_metrics": grouped_folds.copy() if not grouped_folds.empty else pd.DataFrame({"message": ["Grouped validation not run"]}),
        "OOF_predictions": grouped_preds.copy() if not grouped_preds.empty else pd.DataFrame({"message": ["Grouped validation not run"]}),
        "Selection_by_outer_fold": grouped_selections.copy() if not grouped_selections.empty else pd.DataFrame({"message": ["Grouped validation not run"]}),
    }


def kinetic_postprocessing_tables(series,points,model_name):
    if series.empty: return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    params=series[~series["model"].eq(model_name)][[SERIES_COL,"model","n_train_points","n_test_points","fit_reliability",CATALYST_COL,POLLUTANT_COL,"parameters"]].copy()
    overall=[]
    if not points.empty:
        for m,g in points.groupby("model",sort=False): overall.append({"model":m,"n_test_points":len(g),"R2_pooled_test":safe_r2(g["experimental_c_c0"],g["predicted_c_c0"]),"RMSE_pooled_test":rmse(g["experimental_c_c0"],g["predicted_c_c0"]),"MAE_pooled_test":float(mean_absolute_error(g["experimental_c_c0"],g["predicted_c_c0"])),"CCC_pooled_test":lin_ccc(g["experimental_c_c0"],g["predicted_c_c0"])})
    counts=series[[SERIES_COL,"model","n_train_points","n_test_points","fit_reliability",CATALYST_COL,POLLUTANT_COL]].copy()
    return params,pd.DataFrame(overall),counts


def plot_kinetic_comparison_reviewer_v5(points,series,model_name,out):
    if points.empty or series.empty: return
    order=[model_name,"Zero_order","PFO","PSO","Weibull"]; available=[m for m in order if m in points["model"].unique()]; fig,axes=plt.subplots(1,2,figsize=(15,6.2)); lo=min(float(points["experimental_c_c0"].min()),float(points["predicted_c_c0"].min())); hi=max(float(points["experimental_c_c0"].max()),float(points["predicted_c_c0"].max()))
    for m,marker in zip(available,["o","s","^","D","P"]):
        g=points[points["model"].eq(m)]; axes[0].scatter(g["experimental_c_c0"],g["predicted_c_c0"],marker=marker,alpha=.68,label=m)
    axes[0].plot([lo,hi],[lo,hi],"--",color="black",lw=1.4,label="1:1 identity line"); axes[0].set_xlabel("Experimental c/c₀"); axes[0].set_ylabel("Predicted c/c₀"); axes[0].set_title("A. Held-out predictions: ML vs classical kinetic models"); axes[0].legend(fontsize=7)
    ml=series[series["model"].eq(model_name)][[SERIES_COL,"RMSE_test"]].rename(columns={"RMSE_test":"ML_RMSE"}); rows=[]
    for kin in ["Zero_order","PFO","PSO","Weibull"]:
        k=series[series["model"].eq(kin)][[SERIES_COL,"RMSE_test"]].rename(columns={"RMSE_test":"Kinetic_RMSE"}); m=ml.merge(k,on=SERIES_COL,how="inner")
        for _,r in m.iterrows(): rows.append({"kinetic_model":kin,"delta_RMSE":float(r["Kinetic_RMSE"]-r["ML_RMSE"])})
    d=pd.DataFrame(rows)
    if not d.empty:
        if SEABORN_AVAILABLE:
            sns.boxplot(data=d,x="kinetic_model",y="delta_RMSE",ax=axes[1]); sns.stripplot(data=d,x="kinetic_model",y="delta_RMSE",ax=axes[1],alpha=.55,size=4)
        else:
            names=list(d["kinetic_model"].unique()); axes[1].boxplot([d.loc[d["kinetic_model"].eq(k),"delta_RMSE"].to_numpy() for k in names],labels=names)
        axes[1].axhline(0,linestyle="--",color="black",lw=1.1,label="Equal RMSE"); axes[1].set_ylabel("ΔRMSE = RMSE(kinetic) − RMSE(ML)"); axes[1].set_xlabel(""); axes[1].set_title("B. Paired series-level error differences"); axes[1].legend(fontsize=7)
    else: axes[1].axis("off")
    fig.suptitle("Comparison with classical kinetic models",fontweight="bold"); fig.tight_layout(); save_figure(fig,out/"figures"/"Figure_ML_vs_classical_kinetic_models",dpi=600,vector=True)

# =============================================================================
# V5 complete publication diagnostics retained from V3/V4 + literature criteria
# =============================================================================

def repeated_lmo_validation(model: Any, X_train: np.ndarray, y_train: np.ndarray,
                            n_splits: int = 5, n_repeats: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated leave-many-out style validation on the training subset.

    Repeated K-fold partitions are used as a reproducible LMO approximation.
    Q2_LMO is evaluated against the calibration-fold mean for every held-out block.
    This is supplementary to the more important kinetic-series/study/pollutant
    grouped validations used for transferability claims.
    """
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    rows=[]
    for i,(tr,va) in enumerate(rkf.split(X_train), start=1):
        m=clone(model); m.fit(X_train[tr], y_train[tr]); p=m.predict(X_train[va])
        den=np.sum((y_train[va]-np.mean(y_train[tr]))**2)
        q2=1-np.sum((y_train[va]-p)**2)/den if den>0 else np.nan
        rows.append({'split':i,'n_calibration':len(tr),'n_validation':len(va),
                     'Q2_LMO':q2,'R2':safe_r2(y_train[va],p),'RMSE':rmse(y_train[va],p),
                     'MAE':mean_absolute_error(y_train[va],p),'CCC':lin_ccc(y_train[va],p)})
    folds=pd.DataFrame(rows)
    summary=pd.DataFrame([{
        'n_splits':n_splits,'n_repeats':n_repeats,'n_validation_blocks':len(folds),
        'Q2_LMO_mean':folds['Q2_LMO'].mean(),'Q2_LMO_sd':folds['Q2_LMO'].std(ddof=1),
        'Q2_LMO_median':folds['Q2_LMO'].median(),'RMSE_LMO_mean':folds['RMSE'].mean(),
        'MAE_LMO_mean':folds['MAE'].mean(),'CCC_LMO_mean':folds['CCC'].mean()
    }])
    return folds,summary


def qsar_literature_criteria_table(metrics: pd.DataFrame, best_name: str,
                                   lmo_summary: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Evaluate commonly reported QSAR criteria from the reviewer-cited literature.

    These thresholds are reported as supplementary chemoinformatics diagnostics.
    They should not replace grouped validation for this heterogeneous kinetic dataset.
    """
    tr=metrics[(metrics['Model'].eq(best_name)) & (metrics['Dataset'].eq('Train'))]
    te=metrics[(metrics['Model'].eq(best_name)) & (metrics['Dataset'].eq('Held-out random test'))]
    loo=metrics[(metrics['Model'].eq(best_name)) & (metrics['Dataset'].eq('LOO'))]
    if te.empty:
        return pd.DataFrame({'criterion':['Held-out test metrics unavailable'],'status':['NOT EVALUATED']})
    tr=tr.iloc[0] if not tr.empty else pd.Series(dtype=float)
    te=te.iloc[0]
    qloo=float(loo.iloc[0].get('Q2_LOO',np.nan)) if not loo.empty else np.nan
    qlmo=float(lmo_summary.iloc[0].get('Q2_LMO_mean',np.nan)) if lmo_summary is not None and not lmo_summary.empty else np.nan
    def status(value, passed):
        if value is None or (isinstance(value,float) and not np.isfinite(value)):
            return 'NOT EVALUATED'
        return 'PASS' if bool(passed) else 'FAIL'
    r2=float(te.get('R2',np.nan)); r0=float(te.get('R0_2',np.nan)); r0p=float(te.get('R0prime_2',np.nan))
    gt1=float(te.get('GT_ratio_R2_R0',np.nan)); gt2=float(te.get('GT_ratio_R2_R0prime',np.nan))
    k=float(te.get('k',np.nan)); kp=float(te.get('k_prime',np.nan))
    qf=[float(te.get(x,np.nan)) for x in ['Q2F1','Q2F2','Q2F3']]
    ccc=float(te.get('CCC',np.nan)); rm=float(te.get('rm2_mean',np.nan)); drm=float(te.get('delta_rm2',np.nan))
    rows=[]
    def add(name,threshold,value,passed,note=''):
        rows.append({'criterion':name,'threshold':threshold,'observed_value':value,
                     'status':status(value,passed),'note':note})
    add('Q2_LOO','> 0.70',qloo, np.isfinite(qloo) and qloo>0.70,
        'Available when --run-loo is enabled.')
    add('R2_ext','> 0.50',r2,np.isfinite(r2) and r2>0.50)
    diff=abs(r0-r0p) if np.isfinite(r0) and np.isfinite(r0p) else np.nan
    add('|R0^2 - R0\'²|','< 0.30',diff,np.isfinite(diff) and diff<0.30)
    or_ok=((np.isfinite(gt1) and gt1<0.10 and np.isfinite(k) and 0.85<k<1.15) or
           (np.isfinite(gt2) and gt2<0.10 and np.isfinite(kp) and 0.85<kp<1.15))
    add('Golbraikh–Tropsha OR criterion','ratio < 0.10 and 0.85 < k or k\' < 1.15',
        min(gt1,gt2) if np.isfinite(gt1) and np.isfinite(gt2) else np.nan,or_ok,
        f'k={k:.4f}; k\'={kp:.4f}; ratios={gt1:.4f}, {gt2:.4f}' if all(np.isfinite(x) for x in [k,kp,gt1,gt2]) else '')
    qmin=np.nanmin(qf) if np.isfinite(qf).any() else np.nan
    add('External predictivity stringent','R2_ext and Q2F1/Q2F2/Q2F3 > 0.70',
        min([r2]+[x for x in qf if np.isfinite(x)]) if np.isfinite(r2) else np.nan,
        np.isfinite(r2) and r2>0.70 and all(np.isfinite(x) and x>0.70 for x in qf))
    gap=abs(float(tr.get('R2',np.nan))-qloo) if np.isfinite(qloo) and np.isfinite(float(tr.get('R2',np.nan))) else np.nan
    add('|R2_train - Q2_LOO|','< 0.10',gap,np.isfinite(gap) and gap<0.10)
    both_ok=(np.isfinite(gt1) and gt1<0.10 and np.isfinite(k) and 0.85<k<1.15 and
             np.isfinite(gt2) and gt2<0.10 and np.isfinite(kp) and 0.85<kp<1.15)
    add('Golbraikh–Tropsha BOTH criteria','both ratios < 0.10 and both slopes 0.85–1.15',
        max(gt1,gt2) if np.isfinite(gt1) and np.isfinite(gt2) else np.nan,both_ok)
    add('mean r_m²','> 0.65',rm,np.isfinite(rm) and rm>0.65)
    add('Δr_m²','< 0.20',drm,np.isfinite(drm) and drm<0.20)
    add('CCC_ext','>= 0.85',ccc,np.isfinite(ccc) and ccc>=0.85)
    add('Q2_LMO (supplementary)','report mean ± SD',qlmo,np.isfinite(qlmo),
        'Repeated 5-fold LMO-style robustness diagnostic; no hard threshold imposed here.')
    out=pd.DataFrame(rows)
    out['literature_context']='Reviewer-cited Stankovic & Marinkovic (2024), with criteria traced to cited QSAR validation literature.'
    out['scope_note']='Supplementary QSAR diagnostics; grouped validation remains primary for transferability.'
    return out


def target_distribution_test_v5(raw: pd.DataFrame) -> pd.DataFrame:
    tr=pd.to_numeric(raw.loc[raw['split'].eq('train'),TARGET_COL],errors='coerce').dropna()
    te=pd.to_numeric(raw.loc[raw['split'].eq('held_out_random_test'),TARGET_COL],errors='coerce').dropna()
    swtr=stats.shapiro(tr) if 3<=len(tr)<=5000 else None; swte=stats.shapiro(te) if 3<=len(te)<=5000 else None
    normal=bool(swtr and swte and swtr.pvalue>=.05 and swte.pvalue>=.05)
    if normal:
        stat,p=stats.ttest_ind(tr,te,equal_var=False); name='Welch independent t-test'
    else:
        stat,p=stats.mannwhitneyu(tr,te,alternative='two-sided'); name='Mann–Whitney U test'
    return pd.DataFrame([{'train_n':len(tr),'test_n':len(te),'train_mean':tr.mean(),'test_mean':te.mean(),
                          'train_median':tr.median(),'test_median':te.median(),
                          'train_Shapiro_p':swtr.pvalue if swtr else np.nan,'test_Shapiro_p':swte.pvalue if swte else np.nan,
                          'selected_test':name,'statistic':stat,'p_value':p,
                          'interpretation_alpha_0.05':'No significant target-location difference' if p>=.05 else 'Significant target-location difference'}])


def residual_variance_test_v5(preds: pd.DataFrame, best_name: str) -> pd.DataFrame:
    d=preds[preds['Model'].eq(best_name)]
    tr=d[d['Dataset'].eq('Train')]['residual'].to_numpy(float)
    te=d[d['Dataset'].eq('Held-out random test')]['residual'].to_numpy(float)
    stat,p=stats.levene(tr,te,center='median')
    return pd.DataFrame([{'test':'Brown–Forsythe (Levene centered at median)','statistic':stat,'p_value':p,
                          'interpretation_alpha_0.05':'No significant residual-spread difference' if p>=.05 else 'Significant residual-spread difference'}])


def pca_diagnostics_v5(X_train: np.ndarray, X_test: np.ndarray, feature_names: List[str], out: Path,
                       train_idx: np.ndarray, test_idx: np.ndarray, make_plot: bool=True) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    ncomp=min(X_train.shape[1],X_train.shape[0])
    pca=PCA(n_components=ncomp,random_state=RANDOM_STATE)
    tr=pca.fit_transform(X_train); te=pca.transform(X_test)
    cols=[f'PC{i+1}' for i in range(ncomp)]
    scores=pd.concat([
        pd.DataFrame(tr,columns=cols).assign(split='train',row_position=train_idx),
        pd.DataFrame(te,columns=cols).assign(split='held_out_random_test',row_position=test_idx)
    ],ignore_index=True)
    load=pd.DataFrame(pca.components_.T,columns=cols); load.insert(0,'feature',feature_names)
    ev=pd.DataFrame({'PC':cols,'explained_variance_ratio':pca.explained_variance_ratio_,
                     'cumulative_explained_variance':np.cumsum(pca.explained_variance_ratio_)})
    save_df(scores,out/'tables'/'pca_final_input_scores.xlsx'); save_df(load,out/'tables'/'pca_final_input_loadings.xlsx'); save_df(ev,out/'tables'/'pca_final_input_explained_variance.xlsx')
    if make_plot and ncomp>=2:
        fig,axes=plt.subplots(1,3,figsize=(17,5.4))
        a=scores[scores['split'].eq('train')]; b=scores[scores['split'].eq('held_out_random_test')]
        axes[0].scatter(a.PC1,a.PC2,alpha=.65,label='Train'); axes[0].scatter(b.PC1,b.PC2,marker='s',alpha=.8,label='Held-out')
        axes[0].set_xlabel(f"PC1 ({100*ev.loc[0,'explained_variance_ratio']:.1f}%)"); axes[0].set_ylabel(f"PC2 ({100*ev.loc[1,'explained_variance_ratio']:.1f}%)"); axes[0].set_title('A. Final input-space PCA'); axes[0].legend()
        for _,r in load.iterrows():
            axes[1].arrow(0,0,r.PC1,r.PC2,head_width=.02,length_includes_head=True,alpha=.65); axes[1].text(r.PC1*1.06,r.PC2*1.06,str(r.feature),fontsize=7)
        axes[1].axhline(0,lw=.7); axes[1].axvline(0,lw=.7); axes[1].set_title('B. PC1/PC2 loadings'); axes[1].set_xlabel('PC1 loading'); axes[1].set_ylabel('PC2 loading')
        m=min(10,len(ev)); axes[2].plot(np.arange(1,m+1),100*ev['explained_variance_ratio'].head(m),marker='o'); axes[2].set_xlabel('Principal component'); axes[2].set_ylabel('Explained variance (%)'); axes[2].set_title('C. Scree plot')
        fig.tight_layout(); save_figure(fig,out/'figures'/'Figure_PCA_final_input_space',dpi=450,vector=True)
    return scores,load,ev


def plot_methodological_workflow_v5(out: Path, n_rows:int, n_studies:int, n_series:int,
                                    n_candidates:int, n_filtered:int, k_star:int, selected:List[str]) -> None:
    fig,ax=plt.subplots(figsize=(24,7.4)); ax.set_xlim(0,24); ax.set_ylim(0,7.4); ax.axis('off')
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    def box(x,y,w,h,title,text):
        p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.08',facecolor='white',edgecolor='black',linewidth=1.1); ax.add_patch(p)
        ax.text(x+w/2,y+h-.25,title,ha='center',va='top',fontsize=9.3,fontweight='bold'); ax.text(x+w/2,y+h/2-.08,text,ha='center',va='center',fontsize=7.5,wrap=True)
    def arrow(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,linewidth=1.0))
    ax.text(12,7.0,'V6 complete reviewer-response workflow: photocatalytic degradation chemometrics and AI',ha='center',fontsize=15,fontweight='bold')
    box(.2,2.0,2.2,3.9,'1. Curated database',f'{n_rows} observations\n{n_studies} studies • {n_series} kinetic series\nCatalyst/process/light/pH\nPollutant SMILES + λmax\nTarget: c/c₀')
    box(2.8,1.6,3.0,4.7,'2. Leakage-safe preprocessing','Random split retained for continuity\nTraining-only median imputation\nMissingness indicators\nIrradiation one-hot encoding\nScaling/variance filtering')
    box(6.2,1.4,4.0,5.1,'3. Pollutant feature selection',f'λmax + RDKit: {n_candidates} candidates\nZero variance → {n_filtered}\nLassoCV ∥ mutual information\nHybrid score + weight sensitivity\nRepeated complete stability\nExhaustive singles and pairs')
    box(10.6,1.8,2.3,4.2,'4. Minimal-k',f'k=1–10\n2% near-optimal rule\nk*={k_star}\nSelected:\n'+(', '.join(selected) if selected else 'none'))
    box(13.3,1.6,2.5,4.7,'5. Models','ANN • SVR\nRandom Forest\nExtraTrees • XGBoost\nOptuna/TPE\nTraining-CV model selection')
    box(16.2,1.25,3.5,5.35,'6. Validation','Random held-out test\nNested GroupKFold: series\nLeave-one-study-out\nLeave-one-pollutant-out\nLOO/LMO diagnostics\nBootstrap + Y-randomization\nWilliams/distance AD')
    box(20.1,1.45,3.6,4.95,'7. Interpretation & benchmarking','Permutation + SHAP dependence\nError-cluster analysis\nResidual/Wilcoxon/Bland–Altman\nQSAR literature criteria\nZero/PFO/PSO/Weibull kinetics\nDeployment + repository package')
    for a,b in [(2.4,2.8),(5.8,6.2),(10.2,10.6),(12.9,13.3),(15.8,16.2),(19.7,20.1)]: arrow(a,3.9,b,3.9)
    fig.tight_layout(); save_figure(fig,out/'figures'/'Figure_methodological_workflow_V5',dpi=600,vector=True)


def plot_feature_selection_four_panel_v5(ranking:pd.DataFrame,kdf:pd.DataFrame,stability:pd.DataFrame,
                                         selected:List[str],out:Path)->None:
    fig,axes=plt.subplots(2,2,figsize=(15,11))
    la=ranking[ranking['lasso_selected']].sort_values('lasso_abs_coefficient').tail(15)
    axes[0,0].barh(la['feature'],la['lasso_coefficient']); axes[0,0].axvline(0,lw=.8); axes[0,0].set_title('A. Non-zero LassoCV coefficients')
    mi=ranking.sort_values('mi_score',ascending=False).head(15).iloc[::-1]
    bars=axes[0,1].barh(mi['feature'],mi['mi_score']);
    for b,f in zip(bars,mi['feature']):
        if f in selected: b.set_hatch('///')
    axes[0,1].set_title('B. Top mutual-information features')
    axes[1,0].errorbar(kdf['k'],kdf['mean_cv_rmse'],yerr=kdf['sd_cv_rmse'],marker='o',capsize=3); thr=1.02*kdf['mean_cv_rmse'].min(); axes[1,0].axhline(thr,ls='--',label='2% threshold')
    sel=kdf[kdf['selected_k_star']].iloc[0]; axes[1,0].scatter(sel['k'],sel['mean_cv_rmse'],marker='*',s=190,zorder=4,label=f"k*={int(sel['k'])}"); axes[1,0].set_xlabel('Number of pollutant features, k'); axes[1,0].set_ylabel('Grouped inner-CV RMSE'); axes[1,0].set_title('C. Minimal-k optimization'); axes[1,0].legend()
    if stability is not None and not stability.empty and 'selection_frequency' in stability.columns:
        st=stability.sort_values('selection_frequency').tail(15); axes[1,1].barh(st['feature'],st['selection_frequency']); axes[1,1].set_xlim(0,1.05); axes[1,1].set_title('D. Complete-selection stability'); axes[1,1].set_xlabel('Selection frequency')
    else:
        axes[1,1].text(.5,.5,'Complete stability not run',ha='center',va='center'); axes[1,1].axis('off')
    fig.tight_layout(); save_figure(fig,out/'figures'/'Figure_feature_selection_four_panel_V5',dpi=450,vector=True)


def plot_model_predictions_v5(metrics:pd.DataFrame,preds:pd.DataFrame,out:Path)->None:
    names=[n for n in ['ANN','SVR','RandomForest','ExtraTrees','XGBoost'] if n in preds['Model'].unique()]
    fig,axes=plt.subplots(2,3,figsize=(16,10)); axes=axes.ravel(); norm=plt.Normalize(preds['absolute_error'].min(),max(preds['absolute_error'].max(),1e-8))
    for ax,name in zip(axes,names):
        d=preds[preds['Model'].eq(name)]
        for split,marker in [('Train','o'),('Held-out random test','s')]:
            s=d[d['Dataset'].eq(split)]; ax.scatter(s['experimental_c_c0'],s['predicted_c_c0'],c=s['absolute_error'],cmap='viridis',norm=norm,marker=marker,alpha=.8,label=split)
        lo=min(d['experimental_c_c0'].min(),d['predicted_c_c0'].min()); hi=max(d['experimental_c_c0'].max(),d['predicted_c_c0'].max()); ax.plot([lo,hi],[lo,hi],'--',lw=1)
        mt=metrics[(metrics['Model'].eq(name))&(metrics['Dataset'].eq('Held-out random test'))]
        if not mt.empty: ax.text(.04,.96,f"R²={mt.iloc[0]['R2']:.3f}\nRMSE={mt.iloc[0]['RMSE']:.3f}",transform=ax.transAxes,va='top')
        ax.set_title(name); ax.set_xlabel('Experimental c/c₀'); ax.set_ylabel('Predicted c/c₀'); ax.legend(fontsize=7)
    for ax in axes[len(names):]: ax.axis('off')
    sm=plt.cm.ScalarMappable(cmap='viridis',norm=norm); sm.set_array([]); fig.colorbar(sm,ax=axes.tolist(),shrink=.76,label='Absolute error')
    fig.suptitle('Experimental versus predicted c/c₀ for optimized models',y=.995,fontweight='bold'); fig.tight_layout(rect=[0,0,.96,.98]); save_figure(fig,out/'figures'/'Figure_all_models_predictions_V5',dpi=450,vector=True)


def plot_model_comparison_v5(metrics:pd.DataFrame,out:Path)->None:
    order=['ANN','SVR','RandomForest','ExtraTrees','XGBoost']; fig,axes=plt.subplots(1,3,figsize=(16,5.4))
    for ax,metric in zip(axes,['R2','RMSE','MAE']):
        for dataset,marker in [('Train','o'),('Held-out random test','s'),('5-fold CV','^')]:
            sub=metrics[metrics['Dataset'].eq(dataset)].set_index('Model').reindex(order).dropna(subset=[metric]);
            if not sub.empty: ax.plot(np.arange(len(sub)),sub[metric],marker=marker,label=dataset)
        ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(order,rotation=35,ha='right'); ax.set_title(metric); ax.set_ylabel(metric); ax.legend(fontsize=7)
    fig.tight_layout(); save_figure(fig,out/'figures'/'Figure_model_comparison_V5',dpi=450,vector=True)


def plot_residual_diagnostics_v5(preds:pd.DataFrame,best_name:str,out:Path)->None:
    d=preds[preds['Model'].eq(best_name)]
    fig,axes=plt.subplots(1,3,figsize=(16,5.2))
    if SEABORN_AVAILABLE: sns.histplot(data=d,x='residual',hue='Dataset',kde=True,bins=24,ax=axes[0])
    else:
        for s in d['Dataset'].unique(): axes[0].hist(d.loc[d['Dataset'].eq(s),'residual'],bins=20,alpha=.5,label=s); axes[0].legend()
    axes[0].set_title('A. Residual distribution')
    for s,m in [('Train','o'),('Held-out random test','s')]:
        z=d[d['Dataset'].eq(s)]; axes[1].scatter(z['predicted_c_c0'],z['residual'],marker=m,alpha=.7,label=s)
    axes[1].axhline(0,ls='--'); axes[1].set_xlabel('Predicted c/c₀'); axes[1].set_ylabel('Residual'); axes[1].set_title('B. Residuals vs predicted'); axes[1].legend()
    stats.probplot(d['residual'].to_numpy(),dist='norm',plot=axes[2]); axes[2].set_title('C. Residual Q–Q plot')
    fig.tight_layout(); save_figure(fig,out/'figures'/'Figure_residual_diagnostics_V5',dpi=450,vector=True)


def plot_bland_altman_v5(bland:pd.DataFrame,out:Path)->None:
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),sharey=True)
    for ax,dataset in zip(axes,['Train','Held-out random test','All']):
        s=bland[bland['Dataset'].eq(dataset)];
        if s.empty: continue
        ax.scatter(s['mean_exp_pred'],s['difference_exp_minus_pred'],c=s['absolute_error'],cmap='viridis',alpha=.8); bias=s['bias'].iloc[0]; lo=s['LoA_lower'].iloc[0]; hi=s['LoA_upper'].iloc[0]; ax.axhline(bias,lw=1.2,label=f'Bias={bias:.3f}'); ax.axhline(lo,ls='--'); ax.axhline(hi,ls='--'); ax.set_title(dataset); ax.set_xlabel('Mean of experimental and predicted'); ax.legend(fontsize=7)
    axes[0].set_ylabel('Experimental − predicted c/c₀'); fig.tight_layout(); save_figure(fig,out/'figures'/'Figure_Bland_Altman_V5',dpi=450,vector=True)


def plot_target_distribution_v5(raw:pd.DataFrame,out:Path)->None:
    fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    if SEABORN_AVAILABLE:
        sns.histplot(data=raw,x=TARGET_COL,hue='split',kde=True,stat='density',common_norm=False,element='step',ax=axes[0]); sns.violinplot(data=raw,x='split',y=TARGET_COL,inner='quart',cut=0,ax=axes[1])
    else:
        for s in raw['split'].unique(): axes[0].hist(raw.loc[raw['split'].eq(s),TARGET_COL],bins=20,alpha=.5,label=s); axes[0].legend(); axes[1].boxplot([raw.loc[raw['split'].eq(s),TARGET_COL] for s in raw['split'].unique()],labels=raw['split'].unique())
    axes[0].set_title('A. Target distribution'); axes[1].set_title('B. Train vs held-out target'); fig.tight_layout(); save_figure(fig,out/'figures'/'Figure_target_train_test_distribution_V5',dpi=450,vector=True)


def plot_violin_features_v5(raw:pd.DataFrame,selected:List[str],out:Path)->None:
    feats=[c for c in FIXED_NUMERIC+selected if c in raw.columns]
    tr=raw[raw['split'].eq('train')]; z=raw[['split']+feats].copy()
    for f in feats:
        mu=pd.to_numeric(tr[f],errors='coerce').mean(); sd=pd.to_numeric(tr[f],errors='coerce').std(ddof=0); z[f]=(pd.to_numeric(raw[f],errors='coerce')-mu)/sd if np.isfinite(sd) and sd>0 else 0
    long=z.melt(id_vars='split',value_vars=feats,var_name='feature',value_name='z_value')
    fig,ax=plt.subplots(figsize=(max(14,len(feats)*.95),7.2))
    if SEABORN_AVAILABLE: sns.violinplot(data=long,x='feature',y='z_value',hue='split',split=True,inner='quart',cut=0,ax=ax)
    else:
        for i,f in enumerate(feats): ax.boxplot([long[(long.feature.eq(f))&(long['split'].eq(s))].z_value.dropna() for s in long['split'].unique()],positions=[i-.15,i+.15],widths=.25)
    ax.set_ylabel('Standardized value using training statistics'); ax.set_xlabel(''); ax.set_title('Train/held-out distributions of final numeric predictors'); ax.tick_params(axis='x',rotation=45); fig.tight_layout(); save_figure(fig,out/'figures'/'Figure_violin_train_test_selected_features_V5',dpi=450,vector=True)


def plot_correlation_mod_v5(raw:pd.DataFrame,preds:pd.DataFrame,best_name:str,selected:List[str],out:Path)->None:
    cols=[c for c in FIXED_NUMERIC+selected+[TARGET_COL] if c in raw.columns]; corr=raw[cols].apply(pd.to_numeric,errors='coerce').corr(); d=preds[preds['Model'].eq(best_name)]
    fig,axes=plt.subplots(1,2,figsize=(18,7.2),gridspec_kw={'width_ratios':[1.2,1]})
    if SEABORN_AVAILABLE: sns.heatmap(corr,cmap='coolwarm',center=0,annot=True,fmt='.2f',annot_kws={'size':6.5},ax=axes[0],cbar_kws={'label':'Pearson r'})
    else: im=axes[0].imshow(corr.to_numpy(),aspect='auto'); axes[0].set_xticks(range(len(corr))); axes[0].set_xticklabels(corr.columns,rotation=90,fontsize=7); axes[0].set_yticks(range(len(corr))); axes[0].set_yticklabels(corr.index,fontsize=7); fig.colorbar(im,ax=axes[0])
    axes[0].set_title('A. Correlation structure (categorical light source excluded)')
    norm=plt.Normalize(d['absolute_error'].min(),max(d['absolute_error'].max(),1e-8))
    for s,m in [('Train','o'),('Held-out random test','s')]:
        q=d[d['Dataset'].eq(s)]; axes[1].scatter(q['experimental_c_c0'],q['predicted_c_c0'],c=q['absolute_error'],cmap='viridis',norm=norm,marker=m,alpha=.82,label=s)
    lo=min(d['experimental_c_c0'].min(),d['predicted_c_c0'].min()); hi=max(d['experimental_c_c0'].max(),d['predicted_c_c0'].max()); axes[1].plot([lo,hi],[lo,hi],'--'); axes[1].set_xlabel('Experimental c/c₀'); axes[1].set_ylabel('Predicted c/c₀'); axes[1].set_title('B. Prediction agreement colored by absolute error'); axes[1].legend(); sm=plt.cm.ScalarMappable(cmap='viridis',norm=norm); sm.set_array([]); fig.colorbar(sm,ax=axes[1],label='Absolute error')
    fig.tight_layout(); save_figure(fig,out/'figures'/'Figure_correlation_and_MOD_V5',dpi=450,vector=True)


def plot_importance_v5(perm:pd.DataFrame,grouped:pd.DataFrame,out:Path)->None:
    fig,axes=plt.subplots(1,2,figsize=(15,6.2)); p=perm.sort_values('permutation_importance_mean').tail(20); axes[0].barh(p['feature'],p['permutation_importance_mean'],xerr=p['permutation_importance_sd']); axes[0].set_title('A. Permutation importance'); axes[0].set_xlabel('Increase in RMSE after permutation')
    g=grouped.sort_values('relative_grouped_importance_percent'); axes[1].barh(g['information_group'],g['relative_grouped_importance_percent']); axes[1].set_title('B. Information-block importance'); axes[1].set_xlabel('Relative grouped importance (%)'); fig.tight_layout(); save_figure(fig,out/'figures'/'Figure_feature_importance_V5',dpi=450,vector=True)


def bubble_feature_summary_v5(raw:pd.DataFrame,perm:pd.DataFrame,shap_imp:pd.DataFrame,selected:List[str],out:Path)->pd.DataFrame:
    perm_map=dict(zip(perm['feature'],perm['permutation_importance_mean'])) if not perm.empty else {}; shap_map=dict(zip(shap_imp['feature'],shap_imp['mean_abs_SHAP'])) if shap_imp is not None and not shap_imp.empty and 'mean_abs_SHAP' in shap_imp else {}
    rows=[]
    for f in [c for c in FIXED_NUMERIC+selected if c in raw.columns]:
        r=pd.to_numeric(raw[f],errors='coerce').corr(pd.to_numeric(raw[TARGET_COL],errors='coerce')); rows.append({'feature':f,'information_group':feature_group(f,selected),'pearson_r':r,'abs_pearson_r':abs(r) if np.isfinite(r) else np.nan,'permutation_importance':perm_map.get(f,np.nan),'mean_abs_SHAP':shap_map.get(f,np.nan)})
    b=pd.DataFrame(rows); save_df(b,out/'tables'/'bubble_plot_feature_summary.xlsx')
    if not b.empty:
        size=b['mean_abs_SHAP'].fillna(0); sizes=120+850*(size-size.min())/(size.max()-size.min()+1e-12) if (size>0).any() else np.full(len(b),300.)
        fig,ax=plt.subplots(figsize=(10,7)); groups=b['information_group'].unique()
        for g in groups:
            m=b['information_group'].eq(g); ax.scatter(b.loc[m,'permutation_importance'],b.loc[m,'abs_pearson_r'],s=sizes[m],alpha=.7,edgecolor='black',label=g)
        for _,r in b.iterrows(): ax.annotate(r['feature'],(r['permutation_importance'],r['abs_pearson_r']),xytext=(4,4),textcoords='offset points',fontsize=7)
        ax.set_xlabel('Permutation importance'); ax.set_ylabel('|Pearson r| with c/c₀'); ax.set_title('Multidimensional feature relevance map'); ax.legend(fontsize=8); fig.tight_layout(); save_figure(fig,out/'figures'/'Figure_bubble_feature_analysis_V5',dpi=450,vector=True)
    return b


def plot_exhaustive_verification_v5(single:pd.DataFrame,pairs:pd.DataFrame,kdf:pd.DataFrame,out:Path)->None:
    if single.empty: return
    fig,axes=plt.subplots(1,3,figsize=(18,6))
    s=single.sort_values('mean_cv_rmse').head(15).sort_values('mean_cv_rmse',ascending=False); axes[0].barh(s['feature'],s['mean_cv_rmse']); axes[0].set_title('A. Best exhaustive single features'); axes[0].set_xlabel('CV RMSE')
    if not pairs.empty:
        p=pairs.sort_values('mean_cv_rmse').head(15).copy(); p['pair']=p['feature_1'].astype(str)+' + '+p['feature_2'].astype(str); p=p.sort_values('mean_cv_rmse',ascending=False); axes[1].barh(p['pair'],p['mean_cv_rmse']); axes[1].set_title('B. Best exhaustive pairs'); axes[1].set_xlabel('CV RMSE')
    else: axes[1].axis('off')
    axes[2].errorbar(kdf['k'],kdf['mean_cv_rmse'],yerr=kdf['sd_cv_rmse'],marker='o',capsize=3); axes[2].axhline(1.02*kdf['mean_cv_rmse'].min(),ls='--'); axes[2].set_xlabel('Hybrid ranked k'); axes[2].set_ylabel('CV RMSE'); axes[2].set_title('C. Hybrid minimal-k')
    fig.tight_layout(); save_figure(fig,out/'figures'/'Figure_exhaustive_vs_hybrid_selection_V5',dpi=450,vector=True)


def figure_index_v6() -> pd.DataFrame:
    return pd.DataFrame([
        ['Figure_methodological_workflow_V5','Methods','Legacy filename retained; V6 workflow with pH, kinetic-series leakage control and QSAR criteria'],
        ['Figure_feature_selection_four_panel_V5','Methods/Results','Parallel LassoCV–MI, minimal-k and stability'],
        ['Figure_hybrid_weight_sensitivity','Results','Reviewer 2: hybrid-weight sensitivity'],
        ['Figure_exhaustive_vs_hybrid_selection_V5','Results','Reviewer 3: all single and two-descriptor verification'],
        ['Figure_PCA_final_input_space','Results','Chemical/experimental input-space diagnostic'],
        ['Figure_violin_train_test_selected_features_V5','Results','Train/test distribution comparison'],
        ['Figure_all_models_predictions_V5','Results','Predicted versus experimental values for all ML models'],
        ['Figure_model_comparison_V5','Results','R²/RMSE/MAE comparison'],
        ['Figure_target_train_test_distribution_V5','Results','Target-distribution balance'],
        ['Figure_residual_diagnostics_V5','Results','Residual histogram, residual-vs-predicted and Q–Q'],
        ['Figure_Bland_Altman_V5','Results','Bias/agreement analysis'],
        ['Figure_Williams_AD','Results','Applicability domain using actual transformed p'],
        ['Figure_feature_importance_V5','Results','Permutation and grouped importance'],
        ['Figure_SHAP_bar / beeswarm / dependence','Results','Reviewer 2: correlated-feature interpretation'],
        ['Figure_error_cluster_heatmap','Results','Reviewer 3: systematic misprediction clusters'],
        ['Figure_random_vs_grouped_validation','Results','Reviewer 2: 5-fold GroupKFold by complete kinetic series'],
        ['Figure_pH_ablation_V6','Results','Reviewer 2: pH incremental-value ablation using identical observations/split'],
        ['Figure_MW_SlogP_chemical_space_V6','Results','QSAR-style descriptive pollutant chemical-space coverage'],
        ['Figure_kinetic_delta_RMSE','Results','Reviewers 2/3: paired series-level ΔRMSE for zero-order/PFO/PSO/Weibull vs ML'],
        ['Figure_ML_vs_classical_kinetic_models','Results','Reviewer 3 Section 3.8: classical kinetic comparison; dashed diagonal = 1:1 identity line'],
        ['figures/paired_model_MOD/Figure_pair_MOD_<model>','Results','Paired prediction + MOD analysis for every developed ML model'],
    ],columns=['figure','section','purpose'])

# =============================================================================
# Deployment bundle
# =============================================================================

def save_deployment_bundle(
    out: Path,
    best_name: str,
    best_model: Any,
    fixed_objects: Dict[str, Any],
    poll_objects: Dict[str, Any],
    selected_pollutant: List[str],
    fixed_feature_names: List[str],
    final_feature_names: List[str],
) -> None:
    bundle = {
        "pipeline_version": "V6-k1min-reviewer-response",
        "best_model_name": best_name,
        "model": best_model,
        "target_col": TARGET_COL,
        "smiles_col": SMILES_COL,
        "fixed_numeric_features": FIXED_NUMERIC,
        "irradiation_source_feature": IRRADIATION_COL,
        "irradiation_mapping": IRRADIATION_MAPPING,
        "selected_pollutant_features": selected_pollutant,
        "fixed_numeric_imputer": fixed_objects["numeric_imputer"],
        "fixed_numeric_scaler": fixed_objects["numeric_scaler"],
        "irradiation_imputer": fixed_objects["categorical_imputer"],
        "irradiation_encoder": fixed_objects["categorical_encoder"],
        "fixed_transformed_feature_names": fixed_feature_names,
        "pollutant_imputer": poll_objects["imputer"],
        "pollutant_scaler": poll_objects["scaler"],
        "pollutant_variance_filter": poll_objects["variance_filter"],
        "pollutant_candidate_original_names": poll_objects["original_names"],
        "pollutant_candidate_kept_names": poll_objects["kept_names"],
        "final_transformed_feature_names": final_feature_names,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "note": "Missing physical variables are imputed using medians fitted on training data only. Irradiation source is one-hot encoded.",
    }
    (out / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "models" / "photocatalysis_v6_deployment_bundle.joblib")


# =============================================================================
# Main V6 run
# =============================================================================

def run_v6(args):
    out = Path(args.output_dir)
    for folder in ["tables", "figures", "models", "pipelines", "reports", "logs", "optuna_studies"]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    logging.info("Starting photocatalysis AI pipeline V6 constrained-k>=1 complete reviewer-response edition")

    # 1. Data QC and grouping.
    df_raw = read_table(Path(args.input))
    df, qc = quality_control_v5(
        df_raw, out,
        zero_eg_bet_as_missing=args.zero_eg_bet_as_missing,
        drop_out_of_range_target=args.drop_out_of_range_target,
    )
    save_df(df, out / "tables" / "cleaned_modeling_database.xlsx")
    save_df(df[["original_index", STUDY_COL, SERIES_COL, POLLUTANT_COL, CATALYST_COL, REF_COL]], out / "tables" / "group_ids_and_references.xlsx")

    # 2. Molecular descriptors.
    candidate_raw, smiles_table, rdkit_cols = build_pollutant_candidates(df, out)
    y = df[TARGET_COL].to_numpy(dtype=float)

    # 3. Primary random split retained for continuity with submitted manuscript.
    positions = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        positions, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE
    )
    split_table = df[["original_index", STUDY_COL, SERIES_COL, POLLUTANT_COL, CATALYST_COL]].copy()
    split_table["row_position"] = np.arange(len(df))
    split_table["split"] = ""
    split_table.loc[train_idx, "split"] = "train"
    split_table.loc[test_idx, "split"] = "held_out_random_test"
    save_df(split_table, out / "tables" / "train_test_split_indices.xlsx")

    y_train = y[train_idx]
    y_test = y[test_idx]

    # 4. Training-only preprocessing.
    Xf_tr, Xf_te, fixed_names, fixed_objects = fit_fixed_preprocessor(df, train_idx, test_idx)
    Cp_tr, Cp_te, pollutant_names, poll_objects = fit_pollutant_preprocessor(candidate_raw, train_idx, test_idx)
    save_df(pd.DataFrame({"removed_zero_variance_pollutant_feature": poll_objects["removed_names"]}), out / "tables" / "zero_variance_removed.xlsx")
    save_df(pd.DataFrame({"pollutant_feature_after_variance": pollutant_names}), out / "tables" / "pollutant_features_after_variance.xlsx")

    # 5. Parallel LassoCV + MI and V5 score-first ranking.
    comp, lasso = hybrid_components(Cp_tr, y_train, pollutant_names, cv_splits=5)
    ranking = rank_hybrid(comp, args.both_weight)
    save_df(ranking, out / "tables" / "pollutant_feature_hybrid_ranking.xlsx")
    joblib.dump(lasso, out / "pipelines" / "pollutant_lassocv_selector.joblib")

    # 6. Hybrid-weight sensitivity requested by reviewer.
    weight_table, _ = hybrid_weight_sensitivity(
        comp, Xf_tr, Cp_tr, y_train, pollutant_names,
        max_k=args.max_pollutant_features,
        weights=WEIGHT_GRID,
        n_estimators=12 if args.fast_run else 250,
        inner_groups=df.loc[train_idx, SERIES_COL].to_numpy(),
    )
    save_df(weight_table, out / "tables" / "hybrid_weight_sensitivity.xlsx")

    # 7. Primary minimal-k, constrained to k>=1 (at least one pollutant-specific feature).
    k_star, selected_pollutant, kdf = optimize_minimal_k(
        Xf_tr, Cp_tr, y_train, ranking, pollutant_names,
        max_k=args.max_pollutant_features,
        n_estimators=20 if args.fast_run else 400,
        inner_groups=df.loc[train_idx, SERIES_COL].to_numpy(),
        inner_splits=5,
    )
    save_df(kdf, out / "tables" / "minimal_k_results.xlsx")
    logging.info("Primary minimal-k selected k*=%d, features=%s", k_star, selected_pollutant)

    name_to_idx = {n: i for i, n in enumerate(pollutant_names)}
    selected_idx = [name_to_idx[f] for f in selected_pollutant]
    X_train = np.hstack([Xf_tr, Cp_tr[:, selected_idx]]) if selected_idx else Xf_tr
    X_test = np.hstack([Xf_te, Cp_te[:, selected_idx]]) if selected_idx else Xf_te
    final_feature_names = fixed_names + selected_pollutant
    conceptual_features = FIXED_NUMERIC + [IRRADIATION_COL] + selected_pollutant

    final_feature_table = pd.DataFrame({
        "transformed_feature": final_feature_names,
        "information_group": [feature_group(f, selected_pollutant) for f in final_feature_names],
    })
    save_df(final_feature_table, out / "tables" / "final_model_features_transformed.xlsx")
    save_df(pd.DataFrame({"conceptual_predictor": FIXED_NUMERIC + [IRRADIATION_COL] + selected_pollutant}), out / "tables" / "final_model_features_conceptual.xlsx")
    feature_flow, feature_audit = feature_selection_architecture_audit(candidate_raw, pollutant_names, ranking, selected_pollutant, fixed_names, final_feature_names)
    save_df(feature_flow, out / "tables" / "feature_selection_pipeline_flow.xlsx")
    save_df(feature_audit, out / "tables" / "feature_selection_architecture_audit.xlsx")

    # 8. Complete feature-selection stability requested by reviewer.
    stability_runs = stability_freq = pd.DataFrame()
    if args.run_stability:
        stability_runs, stability_freq = complete_selection_stability(
            df.loc[train_idx].reset_index(drop=True),
            candidate_raw.loc[train_idx].reset_index(drop=True),
            y_train,
            max_k=args.max_pollutant_features,
            n_splits=STABILITY_SPLITS,
            n_repeats=1 if args.fast_run else STABILITY_REPEATS,
            n_estimators=10 if args.fast_run else 80,
            both_weight=args.both_weight,
        )
        save_df(stability_runs, out / "tables" / "complete_selection_stability_runs.xlsx")
        save_df(stability_freq, out / "tables" / "complete_selection_stability_frequency.xlsx")
        if not stability_runs.empty:
            save_df(
                stability_runs.groupby("selected_k").size().reset_index(name="count").assign(
                    frequency=lambda d: d["count"] / d["count"].sum()
                ),
                out / "tables" / "selected_k_stability_distribution.xlsx",
            )

    # 9. Exhaustive all-single and all-pair descriptor verification.
    single_results = pair_results = exhaustive_summary = pd.DataFrame()
    if args.run_exhaustive:
        max_pairs = None
        single_results, pair_results, exhaustive_summary = exhaustive_single_pair_analysis(
            Xf_tr, Cp_tr, y_train, pollutant_names,
            n_estimators=5 if args.fast_run else args.exhaustive_trees,
            max_pairs=max_pairs,
            inner_groups=df.loc[train_idx, SERIES_COL].to_numpy(),
            desired_splits=3 if args.fast_run else 5,
        )
        save_df(single_results, out / "tables" / "exhaustive_single_descriptor_results.xlsx")
        save_df(pair_results, out / "tables" / "exhaustive_descriptor_pair_results.xlsx")
        save_df(exhaustive_summary, out / "tables" / "exhaustive_descriptor_summary.xlsx")

    # 10. Hyperparameter search spaces + optimization.
    search_space = hyperparameter_search_space_table()
    save_df(search_space, out / "tables" / "hyperparameter_search_spaces.xlsx")

    if args.fast_run:
        models = build_fast_models()
        hrows = []
        for name, model in models.items():
            cv_rmse = internal_cv_rmse(model, X_train, y_train)
            model.fit(X_train, y_train)
            hrows.append({"model": name, "best_internal_cv_rmse": cv_rmse, "best_params": "fast smoke-test settings"})
        hyper = pd.DataFrame(hrows)
        studies = {}
    else:
        models, studies, hyper = optimize_models(X_train, y_train, args.n_trials)
        for name, study in studies.items():
            save_df(study.trials_dataframe(), out / "optuna_studies" / f"{name}_optuna_trials.xlsx")
    save_df(hyper, out / "tables" / "optuna_best_hyperparameters.xlsx")

    best_name = str(hyper.sort_values("best_internal_cv_rmse").iloc[0]["model"])
    best_model = models[best_name]
    logging.info("Best model by training CV: %s", best_name)

    # 11. Random holdout + standard QSAR metrics.
    metrics, preds = evaluate_models(
        models, X_train, X_test, y_train, y_test, train_idx, test_idx,
        run_loo=args.run_loo,
    )
    save_df(metrics, out / "tables" / "model_performance_QSAR_metrics.xlsx")
    save_df(preds, out / "tables" / "predictions_random_holdout.xlsx")
    mod_summary = model_mod_analysis_table(preds)
    save_df(mod_summary, out / "tables" / "all_models_MOD_analysis.xlsx")

    # V5-complete: repeated leave-many-out robustness and literature validation criteria.
    lmo_folds, lmo_summary = repeated_lmo_validation(best_model, X_train, y_train, n_splits=5, n_repeats=1 if args.fast_run else 10)
    save_df(lmo_folds, out / "tables" / "Q2_LMO_repeated_folds.xlsx")
    save_df(lmo_summary, out / "tables" / "Q2_LMO_summary.xlsx")
    literature_criteria = qsar_literature_criteria_table(metrics, best_name, lmo_summary)
    save_df(literature_criteria, out / "tables" / "QSAR_literature_acceptance_criteria.xlsx")

    # 12. pH ablation on exactly the same observations and random split.
    ph_ablation = pH_ablation_v6(best_model, X_train, X_test, y_train, y_test, final_feature_names)
    save_df(ph_ablation, out / "tables" / "pH_ablation_analysis.xlsx")

    # 13. Reviewer-requested grouped validation: Kinetic_series_ID only.
    grouped_folds = grouped_preds = grouped_summary = grouped_selections = pd.DataFrame()
    if args.run_grouped_validation:
        grouped_folds, grouped_preds, grouped_summary, grouped_selections = run_kinetic_series_grouped_validation(
            df, candidate_raw, y, best_model,
            max_k=args.max_pollutant_features,
            both_weight=args.both_weight,
            fast_run=args.fast_run,
        )
        save_df(grouped_folds, out / "tables" / "grouped_validation_fold_metrics.xlsx")
        save_df(grouped_preds, out / "tables" / "grouped_validation_oof_predictions.xlsx")
        save_df(grouped_summary, out / "tables" / "grouped_validation_summary.xlsx")
        save_df(grouped_selections, out / "tables" / "grouped_validation_selection_stability.xlsx")

    validation_sheets = validation_group_tables(df, split_table, grouped_folds, grouped_preds, grouped_selections)
    write_workbook(out / "tables" / "validation_groups_and_folds.xlsx", validation_sheets)

    # 13. Bootstrap uncertainty + bias agreement.
    best_test = preds[(preds["Model"].eq(best_name)) & (preds["Dataset"].eq("Held-out random test"))].sort_values("row_position")
    boot_samples, boot_ci = bootstrap_test_metrics(
        best_test["experimental_c_c0"].to_numpy(),
        best_test["predicted_c_c0"].to_numpy(),
    )
    save_df(boot_samples, out / "tables" / "bootstrap_metric_samples.xlsx")
    save_df(boot_ci, out / "tables" / "bootstrap_metric_confidence_intervals.xlsx")

    paired_stats, bland = paired_agreement_statistics(preds, best_name)
    save_df(paired_stats, out / "tables" / "paired_bias_Wilcoxon_and_bootstrap_CI.xlsx")
    save_df(bland, out / "tables" / "Bland_Altman_data.xlsx")

    # 14. Y-randomization.
    yrand = yrand_summary = pd.DataFrame()
    if args.run_y_randomization:
        yrand, yrand_summary = y_randomization(X_train, y_train, min(args.y_randomization_repeats, 20) if args.fast_run else args.y_randomization_repeats)
        save_df(yrand, out / "tables" / "y_randomization_results.xlsx")
        save_df(yrand_summary, out / "tables" / "y_randomization_summary.xlsx")

    # 15. Applicability domain using actual transformed p.
    pred_train_best = best_model.predict(X_train)
    pred_test_best = best_model.predict(X_test)
    ad = applicability_domain(
        X_train, X_test, y_train, y_test,
        pred_train_best, pred_test_best, train_idx, test_idx,
    )
    save_df(ad, out / "tables" / "applicability_domain.xlsx")

    # 16. Feature importance + SHAP dependence.
    built_imp, perm_imp, grouped_imp = feature_importance(
        best_model, X_test, y_test, final_feature_names, selected_pollutant
    )
    save_df(built_imp, out / "tables" / "built_in_importance.xlsx")
    save_df(perm_imp, out / "tables" / "permutation_importance.xlsx")
    save_df(grouped_imp, out / "tables" / "grouped_permutation_importance.xlsx")
    shap_imp = pd.DataFrame()
    if args.run_shap:
        shap_imp = shap_analysis(
            best_model, X_train, X_test, final_feature_names, out,
            n_dependence=args.shap_dependence_features,
            make_plots=not args.skip_publication_plots,
        )
        save_df(shap_imp if not shap_imp.empty else pd.DataFrame({"message": ["SHAP not generated"]}), out / "tables" / "shap_importance.xlsx")

    # 17. Systematic misprediction clusters.
    error_groups, top_errors = error_cluster_analysis(df, preds, best_name)
    save_df(error_groups, out / "tables" / "systematic_error_group_analysis.xlsx")
    save_df(top_errors, out / "tables" / "largest_prediction_errors.xlsx")

    # 18. Expanded kinetic benchmarking.
    kinetic_series = kinetic_points = kinetic_stats = pd.DataFrame()
    if args.run_kinetics:
        kinetic_series, kinetic_points, kinetic_stats = kinetic_benchmark(
            df, train_idx, test_idx, best_model, X_train, X_test, best_name
        )
        save_df(kinetic_series, out / "tables" / "kinetic_model_comparison_by_series.xlsx")
        save_df(kinetic_points, out / "tables" / "kinetic_model_test_predictions.xlsx")
        save_df(kinetic_stats, out / "tables" / "kinetic_model_paired_Wilcoxon.xlsx")
        kinetic_params, kinetic_overall, kinetic_counts = kinetic_postprocessing_tables(kinetic_series, kinetic_points, best_name)
        save_df(kinetic_params, out / "tables" / "kinetic_model_parameters.xlsx")
        save_df(kinetic_overall, out / "tables" / "kinetic_model_pooled_performance.xlsx")
        save_df(kinetic_counts, out / "tables" / "kinetic_series_training_point_counts.xlsx")
    else:
        kinetic_params = kinetic_overall = kinetic_counts = pd.DataFrame()

    # 19. Descriptive / reproducibility tables.
    raw_selected = df[[
        "original_index", STUDY_COL, SERIES_COL, CATALYST_COL, POLLUTANT_COL,
        *FIXED_NUMERIC, IRRADIATION_COL, TARGET_COL, "degradation_percent",
    ]].copy()
    for f in selected_pollutant:
        raw_selected[f] = candidate_raw[f].values
    raw_selected["split"] = ""
    raw_selected.loc[train_idx, "split"] = "train"
    raw_selected.loc[test_idx, "split"] = "held_out_random_test"
    save_df(raw_selected, out / "tables" / "non_transformed_selected_dataset.xlsx")

    descriptive = raw_selected[[*FIXED_NUMERIC, IRRADIATION_COL, *selected_pollutant, TARGET_COL, "degradation_percent"]].describe(include="all").T.reset_index().rename(columns={"index": "variable"})
    save_df(descriptive, out / "tables" / "descriptive_statistics.xlsx")
    chemical_space = chemical_space_mw_slogp_v6(
        df, candidate_raw, train_idx, test_idx, out,
        make_plot=not args.skip_publication_plots,
    )
    target_dist_test = target_distribution_test_v5(raw_selected)
    residual_var_test = residual_variance_test_v5(preds, best_name)
    save_df(target_dist_test, out / "tables" / "target_train_test_distribution_test.xlsx")
    save_df(residual_var_test, out / "tables" / "residual_variance_Brown_Forsythe.xlsx")
    pca_scores, pca_loadings, pca_ev = pca_diagnostics_v5(
        X_train, X_test, final_feature_names, out, train_idx, test_idx,
        make_plot=not args.skip_publication_plots,
    )

    # 20. Publication figures: V3/V4 figures retained and V5 reviewer figures added.
    if not args.skip_publication_plots:
        plot_methodological_workflow_v5(out, len(df), df[STUDY_COL].nunique(), df[SERIES_COL].nunique(), candidate_raw.shape[1], len(pollutant_names), k_star, selected_pollutant)
        plot_feature_selection_four_panel_v5(ranking, kdf, stability_freq, selected_pollutant, out)
        plot_minimal_k(kdf, out)
        plot_weight_sensitivity(weight_table, out)
        if args.run_exhaustive:
            plot_exhaustive_verification_v5(single_results, pair_results, kdf, out)
        plot_model_predictions_v5(metrics, preds, out)
        plot_model_comparison_v5(metrics, out)
        plot_each_model_pair_mod_v5(metrics, preds, out)
        plot_pH_ablation_v6(ph_ablation, out)
        plot_target_distribution_v5(raw_selected, out)
        plot_violin_features_v5(raw_selected, selected_pollutant, out)
        plot_residual_diagnostics_v5(preds, best_name, out)
        plot_bland_altman_v5(bland, out)
        plot_correlation_mod_v5(raw_selected, preds, best_name, selected_pollutant, out)
        plot_importance_v5(perm_imp, grouped_imp, out)
        plot_williams(ad, out)
        plot_error_heatmap(df, preds, best_name, out)
        if args.run_kinetics:
            plot_kinetic_deltas(kinetic_series, best_name, out)
            plot_kinetic_comparison_reviewer_v5(kinetic_points, kinetic_series, best_name, out)
        random_row = metrics[(metrics["Model"].eq(best_name)) & (metrics["Dataset"].eq("Held-out random test"))].iloc[0]
        if not grouped_summary.empty:
            plot_grouped_validation(random_row, grouped_summary, out)
    bubble_summary = bubble_feature_summary_v5(raw_selected, perm_imp, shap_imp, selected_pollutant, out) if not args.skip_publication_plots else pd.DataFrame()
    fig_index = figure_index_v6()
    save_df(fig_index, out / "tables" / "publication_figure_index.xlsx")

    # 21. Save software versions and preprocessing/model bundle.
    versions = software_versions()
    (out / "software_versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    save_df(pd.DataFrame(list(versions.items()), columns=["package", "version"]), out / "tables" / "software_versions.xlsx")
    joblib.dump(fixed_objects, out / "pipelines" / "fixed_preprocessing_objects.joblib")
    joblib.dump(poll_objects, out / "pipelines" / "pollutant_preprocessing_objects.joblib")
    joblib.dump(selected_pollutant, out / "pipelines" / "selected_pollutant_features.joblib")
    joblib.dump(final_feature_names, out / "pipelines" / "final_transformed_feature_order.joblib")
    # Save every developed model, not only the selected best model.
    all_model_paths = {}
    for model_name, fitted_model in models.items():
        safe_name = model_name.replace(" ", "_").replace("/", "_")
        model_path = out / "models" / f"model_{safe_name}.joblib"
        joblib.dump(fitted_model, model_path)
        all_model_paths[model_name] = str(model_path.name)
    joblib.dump(best_model, out / "models" / "best_overall_model.joblib")
    model_manifest = hyper.copy()
    model_manifest["model_file"] = model_manifest["model"].map(all_model_paths)
    model_manifest["is_primary_best_model"] = model_manifest["model"].eq(best_name)
    save_df(model_manifest, out / "tables" / "saved_model_manifest.xlsx")
    save_deployment_bundle(
        out, best_name, best_model, fixed_objects, poll_objects,
        selected_pollutant, fixed_names, final_feature_names,
    )
    all_models_bundle = {
        "pipeline_version": "V6-k1min-reviewer-response",
        "best_model_name": best_name,
        "models": models,
        "model_files": all_model_paths,
        "target_col": TARGET_COL,
        "smiles_col": SMILES_COL,
        "fixed_numeric_features": FIXED_NUMERIC,
        "irradiation_source_feature": IRRADIATION_COL,
        "irradiation_mapping": IRRADIATION_MAPPING,
        "selected_pollutant_features": selected_pollutant,
        "fixed_preprocessing": fixed_objects,
        "pollutant_preprocessing": poll_objects,
        "fixed_transformed_feature_names": fixed_names,
        "final_transformed_feature_names": final_feature_names,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "note": "All five developed models share the same fitted V6 preprocessing and selected feature space. The primary model is selected only by training CV RMSE."
    }
    joblib.dump(all_models_bundle, out / "models" / "photocatalysis_v6_all_models_deployment_bundle.joblib")

    # 22. Repository manifest for Zenodo/GitHub preparation.
    manifest = pd.DataFrame([
        ["cleaned_modeling_database.xlsx", "Cleaned modelling rows and group identifiers"],
        ["canonical_smiles.xlsx", "Input and canonicalized SMILES"],
        ["full_RDKit_descriptor_matrix.xlsx", "Complete RDKit descriptor matrix"],
        ["train_test_split_indices.xlsx", "Random split indices"],
        ["group_ids_and_references.xlsx", "Study and kinetic-series groups"],
        ["pollutant_feature_hybrid_ranking.xlsx", "Lasso/MI hybrid ranking"],
        ["minimal_k_results.xlsx", "Full k=1..10 CV-RMSE table"],
        ["complete_selection_stability_runs.xlsx", "Repeated complete feature-selection stability"],
        ["hybrid_weight_sensitivity.xlsx", "Reviewer-requested weighting sensitivity"],
        ["exhaustive_single_descriptor_results.xlsx", "All single pollutant features"],
        ["exhaustive_descriptor_pair_results.xlsx", "All two-feature pollutant combinations"],
        ["hyperparameter_search_spaces.xlsx", "Optuna search ranges"],
        ["model_performance_QSAR_metrics.xlsx", "Random holdout and QSAR metrics"],
        ["Q2_LMO_summary.xlsx", "Repeated leave-many-out robustness diagnostic"],
        ["QSAR_literature_acceptance_criteria.xlsx", "Reviewer-cited QSAR validation criteria and pass/fail status"],
        ["publication_figure_index.xlsx", "Method/result figure map for manuscript revision"],
        ["grouped_validation_summary.xlsx", "Leakage-resistant 5-fold GroupKFold by Kinetic_series_ID"],
        ["paired_bias_Wilcoxon_and_bootstrap_CI.xlsx", "Nonparametric bias test + CI"],
        ["applicability_domain.xlsx", "Williams and distance diagnostics"],
        ["shap_values.xlsx", "SHAP values when available"],
        ["systematic_error_group_analysis.xlsx", "Misprediction cluster analysis"],
        ["kinetic_model_comparison_by_series.xlsx", "Expanded kinetic benchmark"],
        ["kinetic_model_parameters.xlsx", "Zero-order/PFO/PSO/Weibull parameters and train-point counts"],
        ["kinetic_model_pooled_performance.xlsx", "Pooled held-out performance for ML and all kinetic models"],
        ["all_models_MOD_analysis.xlsx", "Per-model magnitude-of-deviation statistics"],
        ["validation_groups_and_folds.xlsx", "Exact random/group validation memberships and outer-fold assignments"],
        ["feature_selection_pipeline_flow.xlsx", "SMILES→RDKit→selection→fixed-block concatenation audit"],
        ["saved_model_manifest.xlsx", "All serialized ML model files and primary-model flag"],
        ["pH_ablation_analysis.xlsx", "Same-split comparison with versus without pH"],
        ["MW_SlogP_chemical_space.xlsx", "Descriptive pollutant MW-SlogP chemical space"],
        ["software_versions.xlsx", "Reproducibility environment"],
        ["photocatalysis_v6_deployment_bundle.joblib", "Best-model deployment bundle + preprocessing"],
        ["photocatalysis_v6_all_models_deployment_bundle.joblib", "All developed ML models + shared preprocessing for deployment"],
    ], columns=["file", "description"])
    save_df(manifest, out / "tables" / "repository_manifest.xlsx")

    # 23. Final consolidated workbook.
    workbook_sheets = {
        "Dataset_structure": qc["Dataset_structure"],
        "Group_IDs": df[["original_index", STUDY_COL, SERIES_COL, CATALYST_COL, POLLUTANT_COL, REF_COL]],
        "Hybrid_ranking": ranking,
        "Minimal_k": kdf,
        "Weight_sensitivity": weight_table,
        "Selection_stability": stability_runs if not stability_runs.empty else pd.DataFrame({"message": ["Not run"]}),
        "Exhaustive_summary": exhaustive_summary if not exhaustive_summary.empty else pd.DataFrame({"message": ["Not run"]}),
        "Hyperparameter_ranges": search_space,
        "Best_hyperparameters": hyper,
        "QSAR_metrics": metrics,
        "QSAR_literature_criteria": literature_criteria,
        "Q2_LMO_summary": lmo_summary,
        "Q2_LMO_folds": lmo_folds,
        "Target_distribution_test": target_dist_test,
        "pH_ablation": ph_ablation,
        "MW_SlogP_space": chemical_space,
        "Residual_variance_test": residual_var_test,
        "PCA_explained_variance": pca_ev,
        "PCA_loadings": pca_loadings,
        "Publication_figure_index": fig_index,
        "Grouped_validation": grouped_summary if not grouped_summary.empty else pd.DataFrame({"message": ["Not run"]}),
        "Grouped_fold_metrics": grouped_folds if not grouped_folds.empty else pd.DataFrame({"message": ["Not run"]}),
        "Paired_bias_tests": paired_stats,
        "Bootstrap_CI": boot_ci,
        "Y_randomization": yrand_summary if not yrand_summary.empty else pd.DataFrame({"message": ["Not run"]}),
        "Applicability_domain": ad,
        "Permutation_importance": perm_imp,
        "SHAP_importance": shap_imp if not shap_imp.empty else pd.DataFrame({"message": ["Not generated"]}),
        "Error_clusters": error_groups,
        "Kinetic_Wilcoxon": kinetic_stats if not kinetic_stats.empty else pd.DataFrame({"message": ["Not run"]}),
        "Kinetic_parameters": kinetic_params if not kinetic_params.empty else pd.DataFrame({"message": ["Not run"]}),
        "Kinetic_pooled": kinetic_overall if not kinetic_overall.empty else pd.DataFrame({"message": ["Not run"]}),
        "Kinetic_point_counts": kinetic_counts if not kinetic_counts.empty else pd.DataFrame({"message": ["Not run"]}),
        "All_models_MOD": mod_summary,
        "Feature_selection_flow": feature_flow,
        "Feature_selection_audit": feature_audit,
        "Grouped_validation_groups": grouped_folds if not grouped_folds.empty else pd.DataFrame({"message": ["Not run"]}),
        "Software_versions": pd.DataFrame(list(versions.items()), columns=["package", "version"]),
    }
    if not args.skip_workbook:
        write_workbook(out / "photocatalysis_v6_complete_results.xlsx", workbook_sheets)

    # 23b. Automatic manuscript/reviewer text reports from the actual run.
    def _fmt(v, digits=4):
        try:
            fv = float(v)
            return f"{fv:.{digits}f}" if np.isfinite(fv) else "NA"
        except Exception:
            return str(v)

    best_test_row = metrics[(metrics["Model"].eq(best_name)) & (metrics["Dataset"].eq("Held-out random test"))].iloc[0]
    group_text = "Not run"
    if not grouped_summary.empty:
        group_text = "; ".join(
            f"{r.get('scheme', 'GroupKFold_by_kinetic_series')}: "
            f"R2={_fmt(r.get('R2'))}, RMSE={_fmt(r.get('RMSE'))}, "
            f"MAE={_fmt(r.get('MAE'))}, CCC={_fmt(r.get('CCC'))}"
            for _, r in grouped_summary.iterrows()
        )

    methods_text = f"""METHODS ADDENDUM GENERATED BY V6

The revised workflow preserves the original two-block architecture while adding pH and leakage-resistant validation. The fixed experimental block contains {len(FIXED_NUMERIC)} numerical variables ({', '.join(FIXED_NUMERIC)}) plus the categorical irradiation-source variable ({IRRADIATION_COL}; 0=sunlight, 1=UV, 2=visible). The pollutant-selection block contains lambda_max and RDKit 2D descriptors generated from canonical SMILES. Numerical preprocessing, categorical encoding, descriptor imputation, scaling, variance filtering, LassoCV, mutual information, hybrid ranking, minimal-k selection and model tuning are fitted on training data only.

LassoCV and mutual information are applied independently and in parallel. The hybrid score gives joint support a default weight of {args.both_weight}, and sensitivity is evaluated over {WEIGHT_GRID}. Minimal-k is selected over k=1–10 as the smallest k whose five-fold training CV-RMSE is within 2% of the minimum; at least one pollutant-specific feature is therefore retained by design. Selection stability repeats the full preprocessing -> selection -> minimal-k procedure. Every single pollutant candidate and every possible pair are exhaustively evaluated using an ExtraTrees baseline.

Five regressors (ANN, SVR, Random Forest, ExtraTrees and XGBoost) are developed. Hyperparameters are optimized with Optuna/TPE in the publication run, and the primary model is selected solely by training cross-validation RMSE. In addition to the preserved random 80:20 internal held-out split, transferability to unseen kinetic trajectories is assessed with five-fold GroupKFold using Kinetic_series_ID only. In every outer fold, imputation, one-hot encoding, scaling, variance filtering, LassoCV, mutual information, hybrid ranking, minimal-k selection and model fitting are re-fitted using only the outer-training series.

Model evaluation includes R2, RMSE, MAE, MedAE, Pearson R, explained variance, Q2F1, Q2F2, Q2F3, Lin CCC, origin-constrained slopes k and k', R0²/R0'² diagnostics, Roy rm² diagnostics, repeated Q2LMO, bootstrap confidence intervals, Wilcoxon bias testing when paired residuals are non-normal, Y-randomization, Williams leverage using the actual transformed feature count, distance-based applicability-domain analysis, permutation importance, SHAP summary/dependence analyses, systematic error grouping, a same-observation pH ablation, and MW-SlogP chemical-space analysis. Classical kinetic benchmarking is presented under the revised section title 'Comparison with classical kinetic models' and includes zero-order, pseudo-first-order (PFO), pseudo-second-order (PSO), and Weibull/stretched-exponential equations. Each kinetic equation is fitted only to training observations within a kinetic series and evaluated on held-out observations. PFO is interpreted as a single-apparent-rate-constant exponential time model; it does not imply linear interactions among predictors. In the kinetic parity figure, the dashed diagonal is explicitly the 1:1 identity/reference line, not a fitted kinetic model. Software versions and all preprocessing/model objects are exported for reproducibility and deployment.
"""
    (out / "reports" / "Methods_for_manuscript.txt").write_text(methods_text, encoding="utf-8")

    results_text = f"""RESULTS SUMMARY GENERATED BY V6

Dataset: {len(df)} observations, {df[STUDY_COL].nunique()} studies, {df[SERIES_COL].nunique()} kinetic series, {df[POLLUTANT_COL].str.lower().nunique()} pollutants and {df[CATALYST_COL].nunique()} catalysts. The pollutant candidate block contained {candidate_raw.shape[1]} variables before training-only zero-variance filtering and {len(pollutant_names)} afterward. The 2%-rule minimal-k procedure selected k*={k_star}: {selected_pollutant}.

The best model selected by training CV was {best_name}. On the internal random held-out subset it achieved R2={_fmt(best_test_row.get('R2'))}, RMSE={_fmt(best_test_row.get('RMSE'))}, MAE={_fmt(best_test_row.get('MAE'))}, Q2F1={_fmt(best_test_row.get('Q2F1'))}, Q2F2={_fmt(best_test_row.get('Q2F2'))}, Q2F3={_fmt(best_test_row.get('Q2F3'))}, and CCC={_fmt(best_test_row.get('CCC'))}. These random-split values represent internal interpolation and are not described as independent external validation.

Grouped validation: {group_text}. The grouped result is interpreted specifically as transferability to unseen complete kinetic series; it is not claimed as validation on unseen pollutants or unseen catalysts. The Williams leverage threshold was h*={_fmt(ad['h_star'].iloc[0])}, calculated from the actual transformed model dimension p={X_train.shape[1]}. The same-split pH ablation is reported in pH_ablation_analysis.xlsx.

The reviewer-cited QSAR validation table, repeated Q2LMO, residual/bias tests, SHAP dependence results, systematic error clusters, pH ablation, chemical-space analysis and expanded kinetic comparisons are provided in the exported tables. The revised kinetic comparison evaluates zero-order, PFO, PSO and Weibull models against the ML model on identical held-out kinetic points. In the corresponding parity figure, the dashed diagonal is the 1:1 identity line. PFO is discussed as a constrained exponential time model with one apparent rate constant, not as a model that assumes linear interactions. Interpretation of descriptor importance and SHAP is restricted to predictive association and is not treated as proof of causality or nonlinear physical mechanism.
"""
    (out / "reports" / "Results_for_manuscript.txt").write_text(results_text, encoding="utf-8")

    reviewer_text = f"""COMPUTATIONAL RESPONSE MAP

Reviewer 2 Comment 1 — Full k=1..10 CV-RMSE table, complete selection-stability analysis and selected-k distribution were added. Files: minimal_k_results.xlsx; complete_selection_stability_runs.xlsx; selected_k_stability_distribution.xlsx.
Reviewer 2 Comment 2 — Leakage-resistant five-fold GroupKFold by complete kinetic series was added. A kinetic series cannot occur in both outer training and validation partitions. Files: grouped_validation_summary.xlsx; grouped_validation_fold_metrics.xlsx; grouped_validation_oof_predictions.xlsx; validation_groups_and_folds.xlsx. Results: {group_text}.
Reviewer 2 Comment 3 — Hybrid joint-support weight sensitivity over {WEIGHT_GRID} was added. File: hybrid_weight_sensitivity.xlsx.
Reviewer 2 Comment 4 — The conceptual and transformed feature sets are exported separately, and Williams h* is calculated from actual transformed p={X_train.shape[1]}.
Reviewer 2 Comment 5 — Kinetic benchmarking was expanded beyond PFO to zero-order, PFO, PSO and Weibull; per-series train/test counts, Delta-RMSE distributions and paired Wilcoxon comparisons are exported. The dashed line in the kinetic parity plot is explicitly the 1:1 identity line.
Reviewer 2 Comment 6 — Permutation importance is supplemented by SHAP summary and dependence analyses, with correlated-predictor limitations noted.
Reviewer 2 Comment 7 — pH is included in the revised fixed block and a same-observation pH ablation is exported.
Reviewer 2 Comment 8 — Bias analysis uses residual-normality testing, Wilcoxon where appropriate, and bootstrap 95% confidence intervals.
Reviewer 2 Comment 9 — Code, canonical SMILES, full descriptor matrix, split indices, kinetic-series group IDs, Optuna spaces/trials, software versions and deployment bundles are exported.
Reviewer 3 Section 2.3 — LassoCV and mutual information are implemented independently in parallel and combined by an explicit score whose weight is sensitivity-tested.
Reviewer 3 Section 2.5 — Hyperparameter search ranges and best settings are exported.
Reviewer 3 Section 2.6 — Standard QSAR/chemoinformatics metrics inspired by Stankovic & Marinkovic (2024) are calculated, including Q2F1/F2/F3, CCC, k/k', R0² diagnostics, rm² diagnostics, Q2LOO/Q2LMO, Y-randomization and applicability-domain analysis.
Reviewer 3 Results — All individual pollutant features and all possible pairs are evaluated; systematic error clusters are analysed; a MW-SlogP chemical-space plot is added; the kinetics subsection is revised to 'Comparison with classical kinetic models'; zero-order, PFO, PSO and Weibull equations are compared on identical held-out kinetic points; and PFO is described correctly as a single-rate-constant exponential time model rather than as implying linear predictor interactions. Paired prediction/MOD plots are generated for every ML model. All developed ML models are serialized individually and included in the all-model deployment bundle.
"""
    (out / "reports" / "reviewer_response_computational_summary.txt").write_text(reviewer_text, encoding="utf-8")

    # 24. Final run summary.
    best_train = metrics[(metrics["Model"].eq(best_name)) & (metrics["Dataset"].eq("Train"))].iloc[0]
    best_test_metric = metrics[(metrics["Model"].eq(best_name)) & (metrics["Dataset"].eq("Held-out random test"))].iloc[0]
    summary = {
        "Rows": len(df),
        "Studies": int(df[STUDY_COL].nunique()),
        "Kinetic series": int(df[SERIES_COL].nunique()),
        "Unique pollutants": int(df[POLLUTANT_COL].str.lower().nunique()),
        "Unique catalysts": int(df[CATALYST_COL].nunique()),
        "Training observations": len(train_idx),
        "Held-out random-test observations": len(test_idx),
        "Initial pollutant candidates": int(candidate_raw.shape[1]),
        "After zero-variance filtering": int(len(pollutant_names)),
        "Selected k*": int(k_star),
        "Minimal-k admissible range": "1..max_k (k=0 excluded by design)",
        "Selected pollutant features": selected_pollutant,
        "Conceptual fixed predictors": FIXED_NUMERIC + [IRRADIATION_COL],
        "Actual transformed fixed columns": fixed_names,
        "Final transformed feature count p": int(X_train.shape[1]),
        "Williams h_star": float(ad["h_star"].iloc[0]),
        "Best model": best_name,
        "Random holdout R2": float(best_test_metric["R2"]),
        "Random holdout RMSE": float(best_test_metric["RMSE"]),
        "Random holdout MAE": float(best_test_metric["MAE"]),
        "Random holdout Q2F1": float(best_test_metric.get("Q2F1", np.nan)),
        "Random holdout CCC": float(best_test_metric.get("CCC", np.nan)),
        "Q2_LMO mean": float(lmo_summary.iloc[0]["Q2_LMO_mean"]),
        "QSAR literature criteria passed": int((literature_criteria["status"]=="PASS").sum()),
        "QSAR literature criteria failed": int((literature_criteria["status"]=="FAIL").sum()),
        "Training R2": float(best_train["R2"]),
        "Grouped validation available": bool(not grouped_summary.empty),
        "Grouped validation strategy": "5-fold GroupKFold by Kinetic_series_ID only",
        "Output directory": str(out.resolve()),
        "Interpretation": "Random 80:20 result is internal interpolation; kinetic-series GroupKFold should be used for claims about transferability to unseen kinetic trajectories.",
    }
    (out / "reports" / "run_summary_v6.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 88)
    print("PHOTOCATALYSIS AI PIPELINE V6-k1min — COMPLETE REVIEWER-RESPONSE SUMMARY")
    print("=" * 88)
    for key, value in summary.items():
        print(f"{key}: {value}")
    logging.info("V6 completed successfully: %s", out)
    return summary


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Complete V6 retaining V3/V4/V5 analyses with reviewer-requested kinetic-series grouped validation, QSAR criteria, pH ablation, MOD figures and expanded kinetics"
    )
    p.add_argument("--input", default="data_co.xlsx", help="Input .xlsx/.xls/.csv database")
    p.add_argument("--output-dir", default="outputs_v6_all", help="Output directory")
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS, help="Optuna trials per model")
    p.add_argument("--max-pollutant-features", type=int, default=MAX_POLLUTANT_FEATURES)
    p.add_argument("--both-weight", type=float, default=2.0, help="Primary joint-support hybrid weight")
    p.add_argument("--run-stability", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-exhaustive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--exhaustive-trees", type=int, default=100, help="ExtraTrees per exhaustive single/pair candidate")
    p.add_argument("--run-grouped-validation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-y-randomization", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--y-randomization-repeats", type=int, default=YRANDOM_REPEATS)
    p.add_argument("--run-shap", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--shap-dependence-features", type=int, default=6)
    p.add_argument("--run-kinetics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-loo", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--zero-eg-bet-as-missing", action=argparse.BooleanOptionalAction, default=True,
                   help="Convert legacy Eg=0 and BET=0 to NaN before training")
    p.add_argument("--drop-out-of-range-target", action="store_true", default=False)
    p.add_argument("--fast-run", action="store_true", help="Lightweight smoke test; publication analyses are reduced")
    p.add_argument("--skip-publication-plots", action="store_true")
    p.add_argument("--skip-workbook", action="store_true")
    return p


if __name__ == "__main__":
    run_v6(build_parser().parse_args())

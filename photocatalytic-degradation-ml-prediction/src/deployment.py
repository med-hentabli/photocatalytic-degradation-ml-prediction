from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


IRRADIATION_LABEL_TO_CODE = {
    "Sunlight": 0,
    "UV": 1,
    "Visible": 2,
}


@dataclass
class ADResult:
    leverage: float
    h_star: float
    inside_leverage_domain: bool
    mahalanobis_distance: float
    distance_threshold: float
    inside_distance_domain: bool

    @property
    def inside_both_structural_diagnostics(self) -> bool:
        return self.inside_leverage_domain and self.inside_distance_domain


class PhotocatalysisPredictor:
    """Inference wrapper for the final V6 k*=1 deployment bundle.

    The wrapper reproduces the training-time preprocessing chain and exposes
    all five optimized regression models. Structural applicability-domain
    diagnostics can be computed prospectively. Residual-based Williams limits
    require an experimental response and therefore are not available before
    an experiment is observed.
    """

    def __init__(
        self,
        bundle_path: str | Path,
        reference_data_path: str | Path | None = None,
    ) -> None:
        self.bundle_path = Path(bundle_path)
        self.bundle: Dict[str, Any] = joblib.load(self.bundle_path)

        selected = list(self.bundle["selected_pollutant_features"])
        if selected != ["RDKit_SMR_VSA5"]:
            raise ValueError(
                "This repository expects the final k*=1 bundle with "
                "selected_pollutant_features=['RDKit_SMR_VSA5']; "
                f"found {selected!r}."
            )

        self.models: Mapping[str, Any] = self.bundle["models"]
        self.primary_model_name: str = self.bundle["best_model_name"]
        self.final_feature_names = list(self.bundle["final_transformed_feature_names"])

        if len(self.final_feature_names) != 13:
            raise ValueError(
                f"Expected 13 transformed predictors; found {len(self.final_feature_names)}."
            )

        self.reference_data_path = (
            Path(reference_data_path) if reference_data_path is not None else None
        )
        self._ad_reference = None
        if self.reference_data_path is not None and self.reference_data_path.exists():
            self._ad_reference = self._build_ad_reference(self.reference_data_path)

    @staticmethod
    def molecule_descriptors(smiles: str) -> Dict[str, float]:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            raise ValueError("Invalid SMILES string.")

        values: Dict[str, float] = {}
        for name, func in Descriptors._descList:
            key = f"RDKit_{name}"
            try:
                value = func(mol)
                values[key] = (
                    float(value)
                    if value is not None and np.isfinite(value)
                    else np.nan
                )
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
                value = func(mol)
                values[key] = (
                    float(value)
                    if value is not None and np.isfinite(value)
                    else np.nan
                )
            except Exception:
                values[key] = np.nan

        return values

    def _transform_fixed(self, df: pd.DataFrame) -> np.ndarray:
        fixed = self.bundle["fixed_preprocessing"]
        numeric = list(self.bundle["fixed_numeric_features"])
        irradiation = self.bundle["irradiation_source_feature"]

        num_imp = fixed["numeric_imputer"].transform(df[numeric])
        num_scaled = fixed["numeric_scaler"].transform(num_imp)

        cat_imp = fixed["categorical_imputer"].transform(df[[irradiation]])
        cat_encoded = fixed["categorical_encoder"].transform(cat_imp)

        return np.hstack([num_scaled, cat_encoded])

    def _transform_pollutant_from_smiles(
        self,
        smiles_values: list[str],
        lambda_max_values: list[float | None] | None = None,
    ) -> np.ndarray:
        poll = self.bundle["pollutant_preprocessing"]
        original_names = list(poll["original_names"])
        kept_names = list(poll["kept_names"])
        selected = list(self.bundle["selected_pollutant_features"])

        candidate = pd.DataFrame(
            np.nan,
            index=np.arange(len(smiles_values)),
            columns=original_names,
        )

        for i, smiles in enumerate(smiles_values):
            desc = self.molecule_descriptors(smiles)
            for name, value in desc.items():
                if name in candidate.columns:
                    candidate.at[i, name] = value

        # lambda_max is not selected by the final k*=1 model, but the original
        # pollutant preprocessor contains this candidate column. If a value is
        # supplied, preserve it; otherwise the training-fitted imputer handles it.
        if lambda_max_values is not None and "lambda_max_nm" in candidate.columns:
            for i, value in enumerate(lambda_max_values):
                if value is not None and np.isfinite(value):
                    candidate.at[i, "lambda_max_nm"] = float(value)

        imp = poll["imputer"].transform(candidate)
        scaled = poll["scaler"].transform(imp)
        filtered = poll["variance_filter"].transform(scaled)

        selected_idx = [kept_names.index(name) for name in selected]
        return filtered[:, selected_idx]

    def _transform_pollutant_from_raw_selected(self, values: np.ndarray) -> np.ndarray:
        """Transform already-computed raw RDKit_SMR_VSA5 values.

        Used to reconstruct the training input space for prospective AD checks.
        """
        poll = self.bundle["pollutant_preprocessing"]
        original_names = list(poll["original_names"])
        kept_names = list(poll["kept_names"])
        selected = list(self.bundle["selected_pollutant_features"])

        candidate = pd.DataFrame(
            np.nan,
            index=np.arange(len(values)),
            columns=original_names,
        )
        candidate[selected[0]] = np.asarray(values, dtype=float)

        imp = poll["imputer"].transform(candidate)
        scaled = poll["scaler"].transform(imp)
        filtered = poll["variance_filter"].transform(scaled)
        selected_idx = [kept_names.index(name) for name in selected]
        return filtered[:, selected_idx]

    def transform_input(self, df: pd.DataFrame) -> np.ndarray:
        required = list(self.bundle["fixed_numeric_features"]) + [
            self.bundle["irradiation_source_feature"],
            self.bundle["smiles_col"],
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required input columns: {missing}")

        fixed = self._transform_fixed(df)
        lambdas = (
            df["lambda_max_nm"].tolist()
            if "lambda_max_nm" in df.columns
            else None
        )
        pollutant = self._transform_pollutant_from_smiles(
            df[self.bundle["smiles_col"]].astype(str).tolist(),
            lambdas,
        )
        X = np.hstack([fixed, pollutant])

        if X.shape[1] != len(self.final_feature_names):
            raise RuntimeError(
                f"Transformed matrix has {X.shape[1]} columns but "
                f"{len(self.final_feature_names)} were expected."
            )
        return X

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.transform_input(df)
        out = pd.DataFrame(index=df.index)
        for name, model in self.models.items():
            pred = np.asarray(model.predict(X), dtype=float)
            out[f"{name}_c_c0"] = pred
            out[f"{name}_degradation_percent"] = 100.0 * (1.0 - pred)
        return out

    def predict_one(self, record: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
        pred = self.predict(pd.DataFrame([record])).iloc[0]
        results: Dict[str, Dict[str, float]] = {}
        for name in self.models:
            results[name] = {
                "c_c0": float(pred[f"{name}_c_c0"]),
                "degradation_percent": float(pred[f"{name}_degradation_percent"]),
            }
        return results

    def _build_ad_reference(self, path: Path) -> Dict[str, Any]:
        ref = pd.read_excel(path)
        train = ref.loc[ref["split"].eq("train")].copy()

        fixed = self._transform_fixed(train)
        pollutant = self._transform_pollutant_from_raw_selected(
            train["RDKit_SMR_VSA5"].to_numpy(float)
        )
        X_train = np.hstack([fixed, pollutant])

        n, p = X_train.shape
        Xaug = np.column_stack([np.ones(n), X_train])
        inv_xtx = np.linalg.pinv(Xaug.T @ Xaug)
        h_star = 3.0 * (p + 1) / n

        mu = np.mean(X_train, axis=0)
        cov = np.cov(X_train, rowvar=False)
        cov_inv = np.linalg.pinv(np.atleast_2d(cov))
        D = X_train - mu
        md_train = np.sqrt(
            np.maximum(np.einsum("ij,jk,ik->i", D, cov_inv, D), 0)
        )
        distance_threshold = float(np.quantile(md_train, 0.95))

        return {
            "n_train": n,
            "p": p,
            "inv_xtx": inv_xtx,
            "h_star": h_star,
            "mu": mu,
            "cov_inv": cov_inv,
            "distance_threshold": distance_threshold,
        }

    def applicability_domain(self, df: pd.DataFrame) -> list[ADResult]:
        if self._ad_reference is None:
            raise RuntimeError(
                "Applicability-domain reference was not loaded. "
                "Provide reference_data_path when constructing the predictor."
            )

        X = self.transform_input(df)
        ref = self._ad_reference
        results: list[ADResult] = []

        for x in X:
            xaug = np.r_[1.0, x]
            leverage = float(xaug @ ref["inv_xtx"] @ xaug.T)
            d = x - ref["mu"]
            md = float(np.sqrt(max(d @ ref["cov_inv"] @ d.T, 0.0)))
            results.append(
                ADResult(
                    leverage=leverage,
                    h_star=float(ref["h_star"]),
                    inside_leverage_domain=leverage <= ref["h_star"],
                    mahalanobis_distance=md,
                    distance_threshold=float(ref["distance_threshold"]),
                    inside_distance_domain=md <= ref["distance_threshold"],
                )
            )
        return results

from pathlib import Path
import pandas as pd

from src.deployment import PhotocatalysisPredictor

ROOT = Path(__file__).resolve().parent
p = PhotocatalysisPredictor(
    ROOT / "models" / "photocatalysis_v6_all_models_deployment_bundle.joblib",
    ROOT / "data" / "reference_training_space.xlsx",
)

assert p.bundle["selected_pollutant_features"] == ["RDKit_SMR_VSA5"]
assert len(p.final_feature_names) == 13
assert set(p.models) == {"XGBoost", "ExtraTrees", "RandomForest", "SVR", "ANN"}

ref = pd.read_excel(ROOT / "data" / "reference_training_space.xlsx")
row = ref.iloc[0]
# Carbendazim SMILES used only to exercise the RDKit path. This test verifies
# the inference pipeline shape and model execution, not equality to row 0.
sample = {
    "MW": float(row["MW"]),
    "Band_gap_Eg_eV": float(row["Band_gap_Eg_eV"]),
    "BET_m2_g": float(row["BET_m2_g"]),
    "Catalyst_dose_ug_mL": float(row["Catalyst_dose_ug_mL"]),
    "Solution_volume_mL": float(row["Solution_volume_mL"]),
    "Light_power_W": float(row["Light_power_W"]),
    "Initial_concentration_mg_L": float(row["Initial_concentration_mg_L"]),
    "Time_min": float(row["Time_min"]),
    "pH": float(row["pH"]),
    "Irradiation_source": int(row["Irradiation_source"]),
    "Smiles": "COC1=NC(=NC2=C1N=CN2C)NC3=CC=CC=C3",
}

X = p.transform_input(pd.DataFrame([sample]))
assert X.shape == (1, 13)
result = p.predict_one(sample)
assert set(result) == set(p.models)
ad = p.applicability_domain(pd.DataFrame([sample]))[0]
assert ad.h_star > 0
assert ad.distance_threshold > 0
print("Deployment test passed.")
print("Primary model:", p.primary_model_name)
print("Selected feature:", p.bundle["selected_pollutant_features"])
print("Transformed p:", len(p.final_feature_names))
print("h*:", ad.h_star)
print("Distance threshold:", ad.distance_threshold)

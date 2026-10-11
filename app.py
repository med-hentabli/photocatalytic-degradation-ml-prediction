from pathlib import Path

import pandas as pd
import streamlit as st

from src.deployment import IRRADIATION_LABEL_TO_CODE, PhotocatalysisPredictor

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "models" / "photocatalysis_v6_all_models_deployment_bundle.joblib"
REFERENCE = ROOT / "data" / "reference_training_space.xlsx"

st.set_page_config(
    page_title="Photocatalytic Degradation Predictor",
    page_icon="🧪",
    layout="wide",
)


@st.cache_resource
def load_predictor():
    return PhotocatalysisPredictor(BUNDLE, REFERENCE)


predictor = load_predictor()

st.title("Photocatalytic Degradation Predictor")
st.caption(
    "Final V6 k*=1 framework. Five optimized machine-learning models share the "
    "same training-fitted preprocessing chain; XGBoost is the primary model."
)

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        mw = st.number_input("Catalyst molecular weight, MW (g mol⁻¹)", value=100.0, min_value=0.0)
        eg = st.number_input("Band gap, Eg (eV)", value=3.0, min_value=0.0)
        bet = st.number_input("BET surface area (m² g⁻¹)", value=10.0, min_value=0.0)
        dose = st.number_input("Catalyst dose (µg mL⁻¹)", value=500.0, min_value=0.0)
    with c2:
        volume = st.number_input("Solution volume (mL)", value=200.0, min_value=0.0)
        power = st.number_input("Light power (W)", value=100.0, min_value=0.0)
        concentration = st.number_input("Initial concentration (mg L⁻¹)", value=20.0, min_value=0.0)
        time = st.number_input("Irradiation time (min)", value=60.0, min_value=0.0)
    with c3:
        ph = st.number_input("pH", value=7.0, min_value=0.0, max_value=14.0)
        irradiation_label = st.selectbox("Irradiation source", list(IRRADIATION_LABEL_TO_CODE))
        smiles = st.text_input("Pollutant SMILES", value="CC(=O)Oc1ccccc1C(=O)O")
        lambda_max = st.number_input(
            "λmax (nm, optional; not selected by final k*=1 model)",
            value=0.0,
            min_value=0.0,
            help="The final model retains RDKit_SMR_VSA5 only. λmax is accepted for preprocessing compatibility but is not a final predictor.",
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    record = {
        "MW": mw,
        "Band_gap_Eg_eV": eg,
        "BET_m2_g": bet,
        "Catalyst_dose_ug_mL": dose,
        "Solution_volume_mL": volume,
        "Light_power_W": power,
        "Initial_concentration_mg_L": concentration,
        "Time_min": time,
        "pH": ph,
        "Irradiation_source": IRRADIATION_LABEL_TO_CODE[irradiation_label],
        "Smiles": smiles,
    }
    if lambda_max > 0:
        record["lambda_max_nm"] = lambda_max

    try:
        result = predictor.predict_one(record)
        ad = predictor.applicability_domain(pd.DataFrame([record]))[0]
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    primary = predictor.primary_model_name
    st.subheader("Primary prediction")
    m1, m2 = st.columns(2)
    m1.metric("XGBoost predicted c/c₀", f"{result[primary]['c_c0']:.4f}")
    m2.metric("Predicted degradation", f"{result[primary]['degradation_percent']:.2f}%")

    st.subheader("All five optimized models")
    table = pd.DataFrame(result).T.reset_index(names="Model")
    table = table.rename(columns={
        "c_c0": "Predicted c/c₀",
        "degradation_percent": "Predicted degradation (%)",
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Prospective applicability-domain check")
    a1, a2 = st.columns(2)
    a1.metric("Leverage", f"{ad.leverage:.4f}", help=f"Warning threshold h* = {ad.h_star:.4f}")
    a2.metric(
        "Mahalanobis distance",
        f"{ad.mahalanobis_distance:.4f}",
        help=f"95th-percentile training threshold = {ad.distance_threshold:.4f}",
    )

    if ad.inside_both_structural_diagnostics:
        st.success("Input lies inside both prospective structural applicability-domain diagnostics.")
    else:
        st.warning(
            "Input is outside at least one structural applicability-domain diagnostic. "
            "Treat the prediction as extrapolative and interpret it cautiously."
        )

    st.info(
        "The residual component of the Williams criterion cannot be evaluated prospectively "
        "because it requires an experimentally observed c/c₀ value."
    )

st.divider()
st.caption(
    "Research-use model. Predictions describe associations learned from the published training domain and do not establish causal mechanisms."
)

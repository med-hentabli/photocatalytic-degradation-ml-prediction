# Photocatalytic Degradation ML Prediction

Reproducible deployment package for predicting the residual concentration ratio (`c/c0`) and degradation percentage in heterogeneous photocatalytic systems.

## Final model configuration

The repository corresponds to the **final constrained minimal-k analysis (`k = 1–10`)** used in the revised manuscript:

- 306 observations
- 52 kinetic series
- 11 pollutants
- 14 catalysts
- 10 fixed conceptual predictors
- one selected pollutant descriptor: **RDKit_SMR_VSA5**
- 11 conceptual predictors in total
- 13 transformed predictor columns after one-hot encoding of irradiation source
- five optimized regressors: **ANN, SVR, Random Forest, ExtraTrees, and XGBoost**
- **XGBoost** selected as the primary model by training-based fivefold CV-RMSE

The all-model bundle stores the fitted models and the preprocessing objects learned from the training set.

## Live web application

The public Streamlit deployment is available at:

https://photocatalytic-degradation-ml-prediction-fxexreoiber4vbbrlbhx4.streamlit.app/

## Repository contents

- `app.py` — Streamlit prediction interface
- `src/deployment.py` — reusable inference and prospective applicability-domain code
- `src/photocatalysis_ai_pipeline_v6_k1min.py` — full V6 analysis pipeline
- `models/photocatalysis_v6_all_models_deployment_bundle.joblib` — all five final models + fitted preprocessing
- `data/reference_training_space.xlsx` — reference feature space used for prospective AD calculations
- `results/photocatalysis_v6_complete_results_WITH_STABILITY.xlsx` — consolidated final results
- `example_input.csv` — example input schema

## Required input

The deployed model accepts:

1. catalyst molecular weight (`MW`)
2. band-gap energy (`Band_gap_Eg_eV`)
3. BET surface area (`BET_m2_g`)
4. catalyst dose (`Catalyst_dose_ug_mL`)
5. solution volume (`Solution_volume_mL`)
6. light power (`Light_power_W`)
7. initial pollutant concentration (`Initial_concentration_mg_L`)
8. irradiation time (`Time_min`)
9. pH (`pH`)
10. irradiation source (`0 = Sunlight`, `1 = UV`, `2 = Visible`)
11. pollutant SMILES (`Smiles`)

The SMILES string is converted to RDKit descriptors using the same descriptor-generation procedure as model development. The final molecular predictor retained by minimal-k selection is `RDKit_SMR_VSA5`.

## Local installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Python use

```python
from src.deployment import PhotocatalysisPredictor

predictor = PhotocatalysisPredictor(
    "models/photocatalysis_v6_all_models_deployment_bundle.joblib",
    "data/reference_training_space.xlsx",
)

sample = {
    "MW": 79.866,
    "Band_gap_Eg_eV": 3.2,
    "BET_m2_g": 10.0,
    "Catalyst_dose_ug_mL": 500.0,
    "Solution_volume_mL": 200.0,
    "Light_power_W": 100.0,
    "Initial_concentration_mg_L": 20.0,
    "Time_min": 60.0,
    "pH": 7.0,
    "Irradiation_source": 1,
    "Smiles": "CC(=O)Oc1ccccc1C(=O)O",
}

print(predictor.predict_one(sample))
```

## Applicability domain

The manuscript uses two complementary applicability-domain diagnostics:

- leverage threshold based on the actual transformed feature count (`p = 13`), with `h* = 3(p+1)/n_train`
- Mahalanobis distance with the 95th percentile of the training-distance distribution as threshold

For a new sample, the app reports **prospective structural diagnostics** (leverage and Mahalanobis distance). The residual-based part of the Williams criterion cannot be evaluated until an experimental `c/c0` is available.

Predictions outside either structural diagnostic should be treated as extrapolative and interpreted cautiously.

## Reproducibility note

The deployment bundle is specifically the **final k*=1 configuration**. It retains `RDKit_SMR_VSA5` as the sole pollutant-specific descriptor. Do not substitute earlier k=3 or k=0-allowed model bundles when reproducing the revised manuscript.

## Citation

Please cite the associated manuscript when using this repository. The final bibliographic citation and DOI can be added after publication.

## License

No software license has been assigned yet. Add the license selected by the repository owner before third-party reuse or redistribution.

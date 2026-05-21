# Wildfire Risk Predictor

## What it Does

This project implements a two-head machine learning pipeline for predicting wildfire hazard using the WildfireSpreadTS dataset. The system combines a "P-model" (ignition probability classifier) and an "A-model" (conditional log burned area regressor) to compute a comprehensive hazard score as P(ignition) × E[burned_area | ignition]. The models use CNN embeddings extracted from satellite imagery and multi-modal features from geospatial data to predict both the likelihood of fire ignition and the expected burned area, enabling proactive wildfire risk assessment. The combined model results in the prediction of expected impact of potential wildfires in the US. These predictions aim to support the optimization of US wildfire management by informing fire departments of regions with the highest wildfire hazard risks.

## Quick Start

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Prepare data**: Place WildfireSpreadTS dataset in `data/raw/wildfirespreadts/`
3. **Extract embeddings**: `python src/data/extract_cnn_embeddings.py`
4. **Prepare features**: `python src/data/prepare_wildfirespreadts_features.py`
5. **Split data**: `python src/data/split_data.py`
6. **Train models**: `python src/models/train_model.py`
7. **Evaluate models**: `python scripts/evaluate_both_models.py`
8. **Combine predictions**: `python src/models/combine_predictions.py`

For detailed setup instructions, see [SETUP.md](SETUP.md).

## Video Links

- **Demo Video**: https://www.youtube.com/watch?v=CMLnrTDWifc
- **Technical Walkthrough**: https://www.youtube.com/watch?v=wl1-0oOie_E

## Evaluation

### P-Model (Ignition Classifier)

- **ROC-AUC**: 0.8862
- **PR-AUC**: 0.9357
- **Accuracy**: 0.8128
- **Precision**: 0.9030
- **Recall**: 0.7807
- **F1-Score**: 0.8374
- **False Positive Rate**: 5.18% (101 cases)
- **False Negative Rate**: 13.54% (264 cases)

![P-Model Evaluation](models/final/visualizations/p_model_evaluation.png)

### A-Model (Log Burned Area Regressor)

- **RMSE**: 1.369 (log scale) or 493.21 hectares (original scale)
- **MAE**: 0.939 (log scale) or 121.61 hectares (original scale)
- **R²**: 0.3190
- **Spearman Correlation**: 0.8005
- **Underestimation Rate**: 73.50% (systematic bias)

![A-Model Evaluation](models/final/visualizations/a_model_evaluation.png)

### Combined Model (Hazard Score)
- **RMSE**: 1.597 (log scale) or 389.32 (original scale)
- **MAE**: 1.107 (log scale) or 77.55 (original scale)
- **R²**: 0.4119
- **Spearman Correlation**: 0.8033

![Hazard Score Analysis](models/final/visualizations/hazard_score_analysis.png)

### Baseline Comparison (Linear / Logistic)

On the same test set (from `notebooks/visualize_evaluation_metrics.ipynb`):

| Model    | Metric                  | Baseline | XGBoost |
| -------- | ----------------------- | -------- | ------- |
| P-model  | ROC-AUC                 | 0.7455   | 0.8862  |
| A-model  | RMSE (log1p scale)      | 1.415    | 1.369   |
| A-model  | MAE (log1p scale)       | 1.114    | 0.939   |
| A-model  | R² (log1p scale)        | 0.272    | 0.319   |
| A-model  | RMSE (original scale)   | 582.41   | 493.21  |
| A-model  | MAE (original scale)    | 212.36   | 121.61  |
| Combined | Log RMSE (log1p hazard) | 1.483    | 1.597   |
| Combined | Log MAE (log1p hazard)  | 1.161    | 1.107   |
| Combined | RMSE (original scale)   | 441.18   | 389.32  |
| Combined | MAE (original scale)    | 132.48   | 77.55   |

### Metrics note

`burned_area` is the **count of next-day active-fire pixels** in each sample (derived from the VIIRS active-fire mask), not true mapped hectares unless multiplied by pixel area. README and plots label this as hectares/pixel-units for continuity with the project docs.

For more detailed analysis, see `notebooks/visualize_evaluation_metrics.ipynb`.

## Project Structure

```
├── src/
│   ├── data/          # Data preparation scripts
│   ├── models/        # Model training and evaluation
│   └── visualization/ # Analysis and visualization
├── notebooks/         # Jupyter notebooks for exploration
├── scripts/           # Convenience scripts
├── data/              # Data directory (raw/interim/processed)
└── models/final/      # Trained models and outputs
```

## Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list

# Wildfire Risk Predictor

## What it Does

This project implements a two-head machine learning pipeline for predicting wildfire hazard using the WildfireSpreadTS dataset. The system combines a "P-model" (ignition probability classifier) and an "A-model" (conditional log burned area regressor) to compute a comprehensive hazard score as P(ignition) × E[burned_area | ignition]. The models use CNN embeddings extracted from satellite imagery and multi-modal features from geospatial data to predict both the likelihood of fire ignition and the expected burned area, enabling proactive wildfire risk assessment. The combined model results in the prediction of expected impact of potential wildfires in the US. These predictions aim to support the optimization of US wildfire management by informing fire departments of regions with the highest wildfire hazard risks.

## Quick Start

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Prepare data**: Place WildfireSpreadTS dataset in `data/raw/wildfirespreadts/`
3. **Extract embeddings**: `python src/data/extract_wildfirespreadts_embeddings.py`
4. **Prepare features**: `python src/data/prepare_wildfirespreadts_features.py`
5. **Split data**: `python src/data/split_data.py`
6. **Train models**: `python src/models/train_model.py`
7. **Evaluate models**: `python scripts/evaluate_both_models.py`
8. **Combine predictions**: `python src/models/combine_predictions.py`

For detailed setup instructions, see [SETUP.md](SETUP.md).

## Video Links

- **Demo Video**: [Link to demo video - update with your video URL]
- **Technical Walkthrough**: [Link to technical walkthrough video - update with your video URL]

## Evaluation

### P-Model (Ignition Classifier)

- **ROC-AUC**: 1.000
- **PR-AUC**: 1.000
- **Accuracy**: 1.000
- **F1-Score**: 1.000

![P-Model Evaluation](models/final/visualizations/p_model_evaluation.png)

![P-Model Confusion Matrix](models/final/visualizations/p_model_confusion_matrix.png)

### A-Model (Log Burned Area Regressor)

- **RMSE**: 0.896 (log scale) / 937.13 hectares (original scale)
- **MAE**: 666.22 hectares
- **R²**: 0.057
- **Spearman Correlation**: 0.621

![A-Model Evaluation](models/final/visualizations/a_model_evaluation.png)

![A-Model Residuals](models/final/visualizations/a_model_residuals.png)

### Combined Model (Hazard Score)

- **RMSE**: 647.68 hectares
- **MAE**: 322.22 hectares
- **R²**: 0.752
- **Spearman Correlation**: 0.869
- Hazard scores range from 0.00 to 3,318.11 hectares

![Hazard Score Analysis](models/final/visualizations/hazard_score_analysis.png)

![Combined Model Evaluation](models/final/visualizations/combined_model_evaluation.png)

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

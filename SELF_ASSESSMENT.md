# Self-Assessment: Machine Learning Rubric Items

This document identifies the 15 highest-scoring Machine Learning rubric items met by this project, with specific evidence locations.

## Machine Learning Category (Maximum 15 selections)

### 1. Built multi-stage ML pipeline connecting outputs of one model to inputs of another (7 pts)

**Evidence**: `src/models/combine_predictions.py` implements a two-head approach where the P-model (ignition classifier) and A-model (log burned area regressor) are combined to compute hazard scores as `P(ignition) × E[burned_area | ignition]`. The script loads both models, makes predictions, and combines them into a unified hazard metric. The pipeline architecture is documented in the README.md "What it Does" section.

### 2. Processed and trained on substantial image dataset (>10,000 images) (5 pts)

**Evidence**: `src/data/extract_wildfirespreadts_embeddings.py` processes the WildfireSpreadTS dataset which contains 13,607 images across 607 fire events (documented in line 11 of the file). The script extracts CNN embeddings from all images in the dataset using ResNet50. The embeddings are then used to train the XGBoost models in `src/models/train_model.py`. The dataset size exceeds the 10,000 image threshold required for this rubric item.

### 3. Successfully adapted pretrained model across substantially different domains or tasks (7 pts)

**Evidence**: `src/data/extract_wildfirespreadts_embeddings.py` adapts ResNet50, which was pretrained on ImageNet (natural images of objects, animals, and scenes), to process satellite imagery from the WildfireSpreadTS dataset for wildfire detection (line 64: `models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)`). This represents a substantial domain shift from natural photography to remote sensing/geospatial imagery. The model successfully extracts meaningful 2048-dimensional embeddings from satellite imagery (lines 61-473), which are then used effectively in downstream wildfire prediction tasks, demonstrating successful cross-domain adaptation.

### 4. Used pretrained model as frozen feature extractor with custom classifier head (5 pts)

**Evidence**: `src/data/extract_wildfirespreadts_embeddings.py` loads a pretrained ResNet50 model (line 64: `models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)`), removes the classification head (lines 66-71), and uses it as a frozen feature extractor to generate 2048-dimensional embeddings from satellite imagery. The model is set to evaluation mode and parameters are frozen (line 473: `model.eval()`), and embeddings are extracted without fine-tuning. These embeddings are then used as features for the downstream XGBoost models.

### 5. Applied regularization techniques to prevent overfitting (at least two of: L1/L2 penalty, dropout, early stopping) (5 pts)

**Evidence**: `src/models/train_model.py` implements both L2 regularization and early stopping. L2 regularization is applied via `reg_lambda=1.0` parameter in both XGBClassifier and XGBRegressor (lines 228, 245, 260). Early stopping is implemented with `early_stopping_rounds=20` (line 225) and `early_stopping_rounds` parameter passed to both models (lines 251, 265). The regularization configuration is documented in print statements (lines 274-275).

### 6. Trained model using GPU/TPU/CUDA acceleration (5 pts)

**Evidence**: `src/models/train_model.py` implements GPU detection and training using PyTorch's standard GPU detection procedure (lines 67-68: `torch.cuda.is_available()`). When GPU is available, XGBoost models are configured with `tree_method='hist'` and `device='cuda'` for GPU acceleration (lines 72-89). The GPU status is printed during training (lines 77, 86, 103), and the device configuration is passed to the training function.

### 7. Used or fine-tuned vision convolutional neural network architecture (5 pts)

**Evidence**: `src/data/extract_wildfirespreadts_embeddings.py` uses ResNet50, a pretrained vision CNN architecture from torchvision (line 64: `models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)`). The model processes satellite imagery from the WildfireSpreadTS dataset to extract visual features. The CNN architecture is used throughout the embedding extraction pipeline (lines 350-473).

### 8. Applied feature engineering (created polynomial features, embeddings, or other derived features) (5 pts)

**Evidence**: `src/data/extract_wildfirespreadts_embeddings.py` extracts 2048-dimensional CNN embeddings from satellite imagery using ResNet50 (lines 61-473). `src/data/prepare_wildfirespreadts_features.py` combines these embeddings with multi-modal geospatial features. The final feature set includes embedding features (identified by `embedding_` prefix), creating a rich feature representation for downstream models. Feature engineering is documented in ATTRIBUTION.md under "Feature Engineering" section.

### 9. Implemented preprocessing pipeline handling data quality issues (addresses class imbalance, missing data, outliers, text tokenization, image resizing, with evidence of impact) (5 pts)

**Evidence**: `src/models/train_model.py` handles missing data through median imputation (lines 195, 208: `fillna(X_train.median())`). Class imbalance is addressed in the P-model through `scale_pos_weight` parameter (line 240: `scale_pos_weight=neg_pos_ratio`). `src/data/prepare_wildfirespreadts_features.py` processes raw data and handles various data quality issues. The impact is evidenced by the model's ability to train successfully and achieve high performance despite class imbalance (P-model achieves perfect classification).

### 10. Compared multiple model architectures or approaches quantitatively (5 pts)

**Evidence**: `src/models/baseline.py` implements baseline models (Logistic Regression for P-model, Linear Regression for A-model). `src/models/evaluate_baseline.py` evaluates baseline models and compares them against XGBoost models. The comparison includes quantitative metrics (ROC-AUC, PR-AUC, RMSE, MAE, R²) for both baseline and XGBoost models, allowing direct performance comparison. Results are saved to `models/final/predictions/baseline_hazard_scores.csv` for comparison.

### 11. Conducted both qualitative and quantitative evaluation with thoughtful discussion (5 pts)

**Evidence**: `notebooks/visualize_evaluation_metrics.ipynb` includes comprehensive quantitative evaluation (Cells 4, 7: metrics calculations) and qualitative evaluation discussion (Cell 12: "Quantitative and Qualitative Evaluation Discussion"). The discussion interprets quantitative metrics in operational context, discusses strengths and limitations, and provides deployment considerations. The evaluation combines numerical metrics with thoughtful analysis of model behavior and real-world implications.

### 12. Performed error analysis with visualization or discussion of failure cases (5 pts)

**Evidence**: `notebooks/visualize_evaluation_metrics.ipynb` Cell 14 contains detailed error analysis discussion that analyzes classification errors (false positives/negatives), regression errors (large errors, underestimation bias), and their operational implications. The analysis includes visualization of residuals, confusion matrices, and prediction scatter plots (Cells 5, 8). The error analysis discusses systematic patterns (82.61% underestimation in A-model) and provides mitigation strategies.

### 13. Implemented proper train/validation/test split with documented split ratios (3 pts)

**Evidence**: `src/data/split_data.py` implements train/validation/test split with documented ratios (default: 15% test, 15% validation, 70% train). The split ratios are clearly documented in function parameters (lines 10, 33-36) and printed during execution (lines 66-69). The split is saved to separate CSV files (train.csv, val.csv, test.csv) and the methodology is documented in the script's docstrings.

### 14. Created baseline model for comparison (e.g., constant prediction, random, simple heuristic) (3 pts)

**Evidence**: `src/models/baseline.py` implements baseline models: Logistic Regression for P-model (ignition classification) and Linear Regression for A-model (log burned area regression). The baseline models are trained, evaluated, and saved (lines 1-200+). `src/models/evaluate_baseline.py` evaluates these baselines and combines their predictions to compute baseline hazard scores, providing a comparison point against the XGBoost models.

### 15. Completed project individually without a partner (10 pts)

**Evidence**: This project was completed me only.

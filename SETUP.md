# Setup Instructions

## Prerequisites

- Python 3.9 or higher
- Git (for cloning the repository)
- CUDA-capable GPU (optional, for faster training)

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd ml-wildfire-risk-predictor
```

## Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Some packages may require system libraries:

- `pyhdf` requires HDF4 libraries (install via conda-forge on Windows)
- `rasterio` may require GDAL system libraries

## Step 4: Download Dataset

1. Download the WildfireSpreadTS dataset from [Zenodo](https://zenodo.org/records/8006177)
2. Extract the dataset to `data/raw/wildfirespreadts/`
3. The directory structure should be:
   ```
   data/raw/wildfirespreadts/
   ├── 2018/
   ├── 2019/
   ├── 2020/
   ├── 2021/
   └── WildfireSpreadTS_Documentation.pdf
   ```

## Step 5: Run Data Preparation Pipeline

```bash
# Extract CNN embeddings from images
python src/data/extract_wildfirespreadts_embeddings.py

# Prepare features and labels
python src/data/prepare_wildfirespreadts_features.py

# Split data into train/val/test sets
python src/data/split_data.py
```

## Step 6: Train Models

```bash
# Train both P-model and A-model
python src/models/train_model.py

# Optional: Train baseline models for comparison
python src/models/baseline.py
```

**Note**: Training uses GPU by default if available. To force CPU:

```bash
python src/models/train_model.py --tree-method hist
```

## Step 7: Evaluate Models

```bash
# Evaluate both models
python scripts/evaluate_both_models.py

# Combine predictions to compute hazard scores
python src/models/combine_predictions.py
```

## Step 8: Generate Visualizations and Analysis

```bash
# Use the Jupyter notebook for comprehensive evaluation and analysis
jupyter notebook notebooks/visualize_evaluation_metrics.ipynb
```

The notebook includes:

- Quantitative evaluation metrics (ROC-AUC, PR-AUC, RMSE, MAE, R², etc.)
- Qualitative evaluation discussion
- Error analysis
- Comprehensive visualizations (ROC curves, confusion matrices, scatter plots, residual plots, hazard score distributions)

## Verification

To verify the setup is working correctly:

```bash
# Run tests
pytest tests/

# Check that models were created
ls models/final/*.pkl
```

## Troubleshooting

- **GPU not detected**: Ensure PyTorch with CUDA support is installed and `nvidia-smi` works
- **Missing data files**: Verify dataset is in `data/raw/wildfirespreadts/` with correct structure
- **Import errors**: Ensure virtual environment is activated and all dependencies are installed
- **Memory errors**: Reduce batch size or use CPU training with `--tree-method hist`

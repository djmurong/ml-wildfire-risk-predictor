# Attribution

This document provides detailed attributions for all sources, datasets, libraries, and AI-generated content used in this project.

## Datasets

### WildfireSpreadTS

- **Source**: Zenodo repository
- **Citation**: WildfireSpreadTS Dataset. Zenodo. https://zenodo.org/records/8006177
- **DOI**: https://doi.org/10.5281/zenodo.8006176
- **Usage**: Primary dataset for wildfire prediction training and evaluation
- **URL**: https://zenodo.org/records/8006177

## Software Libraries and Frameworks

### Core Machine Learning

- **XGBoost** (>=1.6): Gradient boosting framework for model training

  - License: Apache 2.0
  - URL: https://github.com/dmlc/xgboost

- **scikit-learn** (>=1.1): Machine learning utilities and metrics

  - License: BSD 3-Clause
  - URL: https://scikit-learn.org/

- **PyTorch** (>=2.0): Deep learning framework for CNN embeddings

  - License: BSD-style
  - URL: https://pytorch.org/

- **torchvision** (>=0.15): Pre-trained vision models (ResNet50 for embeddings)
  - License: BSD 3-Clause
  - URL: https://pytorch.org/vision/

### Data Processing

- **pandas** (>=1.5): Data manipulation and analysis
- **numpy** (>=1.23): Numerical computing
- **rasterio** (>=1.3): Geospatial raster I/O
- **rioxarray** (>=0.14): Geospatial xarray extensions
- **xarray** (>=2023.6): N-dimensional labeled arrays
- **geopandas** (>=0.12): Geospatial data operations
- **pyarrow** (>=10.0): Parquet file format support

### Visualization

- **matplotlib** (>=3.5): Plotting and visualization
- **seaborn** (>=0.12): Statistical data visualization

### Testing

- **pytest** (>=7.0): Testing framework

## AI-Generated Content

### Code Generation

- **AI Assistant (Cursor/Claude)**: Portions of this codebase were generated or refined with AI assistance for:
  - Code structure and organization
  - Documentation and comments
  - Error handling and edge cases
  - Code refactoring and optimization

### AI-Assisted Development

- Model training scripts were developed with AI assistance
- Data preprocessing pipelines were designed with AI guidance
- Evaluation and visualization code was created with AI support

All AI-generated code has been reviewed and tested by me.

## Methodology and Approach

### Two-Head Modeling Approach

The model design was inspired my elements discussed in the literature review in Jain et al. (2020).

- **Citation**: Piyush Jain, Sean C.P. Coogan, Sriram Ganapathi Subramanian, Mark Crowley, Steve Taylor, and Mike D. Flannigan. 2020. A review of machine learning applications in wildfire science and management. _Environmental Reviews_. 28(4): 478-505. https://doi.org/10.1139/er-2020-0019
- **DOI**: https://doi.org/10.1139/er-2020-0019

### Feature Engineering

- CNN embeddings extracted using pre-trained ResNet50 (torchvision)
- Multi-modal feature combination from satellite imagery and geospatial data

## References

Piyush Jain, Sean C.P. Coogan, Sriram Ganapathi Subramanian, Mark Crowley, Steve Taylor, and Mike D. Flannigan. 2020. A review of machine learning applications in wildfire science and management. _Environmental Reviews_. 28(4): 478-505. https://doi.org/10.1139/er-2020-0019

### Datasets

- WildfireSpreadTS dataset (DOI: 10.5281/zenodo.8006177)

### Software Documentation

- XGBoost documentation for model configuration and training

## License

This project is licensed under the MIT License (see LICENSE file).

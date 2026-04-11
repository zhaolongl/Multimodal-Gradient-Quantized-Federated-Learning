# Honey Ripening State Identification with Spectral and Federated Learning Methods

This repository provides code for identifying honey ripening and processing states using spectral preprocessing, single-modality machine learning, and multimodal federated learning.

## Overview

The project contains three main components:

1. **Spectral preprocessing**
   - Savitzky–Golay smoothing
   - Mean centering
   - Export of processed spectra and visualization plots

2. **Optimized single-modality spectral classification**
   - Multiple preprocessing pipelines
   - Feature extraction with MI, PCA, and PLS
   - RF, SVM, XGBoost, and CNN models
   - Five-fold cross-validation and automatic result export

3. **Multimodal centralized and federated learning**
   - Dual-modality fusion of **Raman** and **^1H NMR** data
   - Centralized learning, standard federated learning, and compressed federated learning
   - Comparison of **MultiModalMLP**, **EarlyFusionMLP**, and **MultiModalCNN1D**
   - Communication-cost analysis and summary figure generation

The code is designed for honey samples from three states:

- **PCH**: Partially capped honey
- **IHEH**: In-hive extended honey
- **CTH**: Chamber-treated honey

---


## Requirements

Recommended environment: **Python 3.9+**

Install the main dependencies:

```bash
pip install numpy pandas matplotlib scipy scikit-learn openpyxl xgboost torch tensorflow
```

### Notes
- `scikit-learn` is required for classical machine learning baselines.
- `tensorflow` is required for the CNN in the optimized single-modality script.
- `torch` is required for the multimodal centralized/federated learning script.

---

## Script Descriptions

### `preprocess_spectra.py`
This script performs basic spectral preprocessing for single-modality data.

#### Functions
- Validates input dimensions and missing values
- Applies **Savitzky–Golay smoothing**
- Applies **mean centering**
- Saves processed spectra to Excel files
- Generates spectral plots

#### Run
```bash
python preprocess_spectra.py
```

---

### `optimized_spectral_classification.py`
This script runs optimized single-modality spectral classification with multiple preprocessing, feature extraction, and model configurations.

#### Main settings
- **Preprocessing options**
  - MC
  - SNV
  - SG
  - Baseline + SG
  - Baseline + SNV

- **Feature extraction options**
  - Mutual Information
  - PCA
  - PLS
  - None

- **Models**
  - Random Forest
  - SVM
  - XGBoost
  - 1D CNN

- **Evaluation**
  - Five-fold stratified cross-validation
  - Accuracy
  - Macro-F1
  - Automatic best-configuration export

#### Outputs
Saved in `outputs_optimized/metrics/`:
- `optimized_results.xlsx`
- `best_config.txt`

#### Run
```bash
python optimized_spectral_classification.py
```

---

### `multimodal_federated_learning.py`
This script implements multimodal centralized and federated learning for honey-state classification using Raman and ^1H NMR data.

#### Fusion models
- **MultiModalMLP**
- **EarlyFusionMLP**
- **MultiModalCNN1D**

#### Training paradigms
- **Centralized learning**
- **Federated learning**
- **Federated learning with gradient compression**

#### Simulated clients
The script simulates a multi-laboratory setting:
- **Client 0**: Raman only
- **Client 1**: NMR only
- **Client 2**: Raman + NMR

#### Evaluation
- Test accuracy
- Convergence curves
- Communication volume
- Compression ratio
- Accuracy gap relative to centralized training
- Accuracy–communication trade-off

#### Main output
- `summary_figure.png`

#### Run
```bash
python multimodal_federated_learning.py
```

---

## Workflow

A recommended order is:

1. Prepare and verify the Excel input files
2. Run `preprocess_spectra.py` for initial preprocessing and visualization
3. Run `optimized_spectral_classification.py` for single-modality benchmarking
4. Run `multimodal_federated_learning.py` for multimodal centralized/federated experiments
5. Compare accuracy, convergence, and communication efficiency

---

## Research Focus

This repository supports experiments for:

- identifying honey ripening and processing states
- comparing Raman and ^1H NMR as complementary modalities
- evaluating multimodal fusion strategies
- assessing federated learning under decentralized data conditions
- measuring communication reduction with gradient compression

---

## Customization

You may adapt the code for your own datasets by changing:

- input file paths
- class labels
- preprocessing settings
- feature extraction parameters
- model hyperparameters
- number and modality distribution of federated clients
- compression settings such as sparsity and quantization bits

---

## Notes for Reproducibility

- Fixed random seeds are used in several parts of the code, but exact reproducibility may still depend on library versions and hardware.
- The federated setting in this repository is a simulated experimental setup rather than a production-ready deployment framework.
- Communication cost is estimated at the model-update level and is intended for comparative analysis.

---


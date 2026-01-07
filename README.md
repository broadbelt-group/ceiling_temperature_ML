# ML Thermodynamics: Ceiling Temperature Prediction

This directory contains the main analysis pipeline for predicting thermodynamic properties (enthalpy, entropy, and ceiling temperature) using machine learning models used in the publication "Predicting Ceiling Temperature for Recyclable Polymer Design using Machine Learning".

## Overview

This project uses various ML models (Random Forest, SVR, XGBoost, Gaussian Process, Kernel Ridge Regression) to predict thermodynamic properties (enthalpy and entropy of polymerization and ceiling temperature) for polymer systems. The workflow is organized as a series of numbered notebooks that should be run sequentially.

## Notebooks

**1_dataset_visualization.ipynb**
- Loads and explores the raw featurized dataset
- Generates summary statistics and visualizations about data distribution
- Creates reports on data availability (enthalpy, entropy, ceiling temperature)
- Outputs: Dataset summary figures and statistics reports

**2_generate_datasets.ipynb**
- Creates global test set using Butina clustering and phase stratification
- Performs feature selection using mutual information regression
- Generates split datasets for training different property predictors
- Outputs: Split CSV files in `2_split_datasets/`

**3_model_training_by_split.ipynb** (Main training notebook)
- Trains multiple ML models (RF, SVR, XGBoost, GPR, KRR) for each property
- Uses nested cross-validation with chemistry-cluster-based or phase-based stratification
- Implements transfer learning for ceiling temperature prediction
- Saves trained models to `3_saved_models/`
- Outputs: Model results summary CSVs, trained model files (`.joblib`)

**3_tables_only.ipynb**
- Generates summary tables of model performance without re-training
- Useful for quick reference of results
- Outputs: Summary CSV files

**4_errors.ipynb**
- Creates visualization and analysis of model errors
- Generates prediction vs. actual plots for different properties
- Analyzes error distributions across different chemistry categories and phases
- Outputs: Error analysis figures in `4_images/`

**5_no_PEP_H.ipynb**
- Alternative model training excluding PEP data for enthalpy
- Tests model robustness without specific chemistry category
- Similar structure to notebook 3

## Data Structure

```
0_raw_data_sets/        # Input data
  - featurized_imputed_data.csv
1_dataset_images/       # Dataset visualizations and statistics
2_split_datasets/       # Train/test splits (output from notebook 2)
3_saved_models/         # Trained ML models (output from notebook 3)
3_images_and_csvs/      # Model results summaries
4_images/               # Error analysis plots
```

## Running the Pipeline

1. Make sure data is in `0_raw_data_sets/featurized_imputed_data.csv`
2. Run notebooks in numerical order
3. Trained models will be saved in `3_saved_models/`
4. Results and figures will be generated in respective output directories

## Dependencies

Key libraries:
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - ML models and preprocessing
- `xgboost` - Gradient boosting
- `rdkit` - Molecular fingerprints and clustering
- `matplotlib` - Visualization
- `joblib` - Model serialization
- `shap` - Model interpretability

# SkilioMall Customer Churn Prediction

## Project Overview
Machine learning solution to predict customer churn for SkilioMall e-commerce platform. Identifies at-risk customers for proactive retention campaigns.

## Quick Start

### Prerequisites
- Python 3.8+
- Git

### One-Command Setup and Run
```bash
git clone <repository-url>
cd skilio_mall_churn_prediction
pip install -r requirements.txt
jupyter notebook notebooks/churn_model_training.ipynb
```

Or run cells directly:
```bash
pip install -r requirements.txt
jupyter notebook notebooks/churn_model_training.ipynb
# Then: Run -> Run All Cells
```

### Alternative: Step-by-Step
```bash
# 1. Clone repository
git clone <repository-url>
cd skilio_mall_churn_prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training notebook
jupyter notebook notebooks/churn_model_training.ipynb
# Execute all cells in order

# 4. Check outputs
# Model artifacts: outputs/
# Visualizations: outputs/*.png
```

## Project Structure
```
skilio_mall_churn_prediction/
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   └── skiliomall_data.csv        # Raw customer data
├── notebooks/
│   ├── EDA.ipynb                  # Exploratory data analysis
│   └── churn_model_training.ipynb # Main training pipeline
├── outputs/
│   ├── evaluation_summary.yaml    # Model evaluation metrics
│   ├── selected_features.txt      # Feature list
│   └── *.png                      # Visualization charts
└── src/
    ├── data_loader.py             # Data loading utilities
    ├── preprocessing.py           # Data preprocessing
    ├── features.py                # Feature engineering
    ├── modeling.py                # Model training
    └── evaluate.py                # Evaluation and visualization
```

## Workflow
1. **Data Loading**: Load raw data from `data/skiliomall_data.csv`
2. **Preprocessing**: Handle missing values, split train/val/test
3. **Feature Engineering**: Create tenure, intensity, and interaction features
4. **Model Training**: Train baseline models (LR, RF, XGBoost, LightGBM)
5. **Hyperparameter Tuning**: Optimize best model with RandomizedSearchCV
6. **Evaluation**: Generate metrics and visualizations
7. **Output**: Save results to `outputs/`

## Key Features
- Stratified train/val/test split to prevent data leakage
- Comprehensive feature engineering pipeline
- Multiple baseline model comparison
- Hyperparameter optimization
- Threshold tuning for optimal F1 score
- SHAP interpretability analysis
- 7 professional visualizations (ROC, PR, confusion matrix, etc.)

## Dependencies
- pandas 2.0.3 - Data manipulation
- numpy 1.24.3 - Numerical computing
- scikit-learn 1.3.0 - ML algorithms and preprocessing
- xgboost 1.7.6 - Gradient boosting
- lightgbm 4.1.0 - Fast gradient boosting
- matplotlib 3.7.2 - Plotting
- seaborn 0.12.2 - Statistical visualization
- shap 0.42.1 - Model interpretability
- imbalanced-learn 0.10.1 - Handling imbalanced data
- jupyter 1.0.0 - Interactive notebooks
- pyyaml 6.0.1 - YAML configuration

## Output Files
After running the training notebook:
- `outputs/evaluation_summary.yaml` - Model performance metrics
- `outputs/selected_features.txt` - Final feature list
- `outputs/baseline_comparison.png` - Model comparison chart
- `outputs/roc_curve.png` - ROC curve
- `outputs/pr_curve.png` - Precision-Recall curve
- `outputs/confusion_matrix.png` - Confusion matrix heatmap
- `outputs/metrics_summary.png` - Metrics comparison
- `outputs/lift_chart.png` - Lift chart
- `outputs/probability_distribution.png` - Probability distributions

## Configuration
Edit `config.yaml` to customize:
- Data paths
- Train/val/test split ratios
- Feature lists
- Model hyperparameters
- Random seed for reproducibility

## Notes
- All code is documented in English
- Execution time: ~5-10 minutes on standard laptop
- Outputs are automatically saved to `outputs/` directory
- No GPU required (CPU-only)
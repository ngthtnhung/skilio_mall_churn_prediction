# SkilioMall Customer Churn Prediction

## Project Overview
Machine learning solution to predict customer churn for SkilioMall e-commerce platform. Identifies at-risk customers for proactive retention campaigns.

## Quick Start

### Prerequisites
- Python 3.8+
- Git

### One-Command Setup and Run
```bash
git clone https://github.com/ngthtnhung/skilio_mall_churn_prediction.git
cd skilio_mall_churn_prediction
pip install -r requirements.txt
jupyter notebook notebooks/churn_model_training.ipynb
```

Or run cells directly:
```bash
pip install -r requirements.txt
jupyter notebook notebooks/churn_model_training.ipynb
# Then: Run -> Run All Cells

# Optional: Run ensemble stacking experiment
jupyter notebook notebooks/ensemble_stacking_training.ipynb
```

## Project Structure
```
skilio_mall_churn_prediction/
├── .gitignore                      # Git ignore rules
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   ├── skiliomall_data.csv         # Raw customer data
│   └── processed_data_split.pkl    # Checkpoint: train/val/test splits
├── notebooks/
│   ├── EDA.ipynb                   # Exploratory data analysis
│   ├── churn_model_training.ipynb  # Main baseline training pipeline
│   └── ensemble_stacking_training.ipynb # Advanced ensemble experiment
├── outputs/
│   ├── baseline_models_comparison.png   # 4 models comparison chart
│   ├── confusion_matrix_test.png        # Confusion matrix heatmap
│   ├── evaluation_summary.yaml          # Baseline model metrics
│   ├── final_model.pkl                  # Trained Logistic Regression model
│   ├── lift_chart_test.png              # Lift chart
│   ├── precision_recall_curve_test.png  # PR curve
│   ├── preprocessor.pkl                 # Fitted preprocessing pipeline
│   ├── probability_distribution_test.png # Probability distribution
│   ├── roc_curve_test.png               # ROC curve
│   ├── selected_features.txt            # Feature list
│   ├── shap_summary.png                 # SHAP feature importance
│   ├── test_metrics_summary.png         # Metrics bar chart
│   └── ensemble/                        # Ensemble stacking outputs
│       ├── stacking_model.pkl           # Trained stacking ensemble
│       ├── stacking_evaluation_summary.yaml  # Ensemble metrics
│       ├── roc_curves_comparison.png    # ROC curves comparison
│       └── models_performance_comparison.png # Performance bar chart
└── src/
    ├── data_loader.py              # Data loading utilities
    ├── preprocessing.py            # Data preprocessing
    ├── features.py                 # Feature engineering
    ├── modeling.py                 # Model training
    └── evaluate.py                 # Evaluation and visualization
```

## Workflow

### Main Pipeline (churn_model_training.ipynb)
1. **Data Loading**: Load raw data from `data/skiliomall_data.csv`
2. **Preprocessing**: Handle missing values, split train/val/test (70/15/15)
3. **Feature Engineering**: Create tenure, intensity, and interaction features
4. **Model Training**: Train 4 baseline models (LR, RF, XGBoost, LightGBM)
5. **Hyperparameter Tuning**: Optimize best model with Optuna (900s budget)
6. **Threshold Optimization**: Select optimal threshold via F1 score on validation
7. **Final Retraining**: Retrain on train+val combined for production
8. **Evaluation**: Generate metrics and 7 visualizations
9. **Output**: Save model and results to `outputs/`

**Result**: Logistic Regression achieves 98.41% ROC-AUC on test set

### Advanced Ensemble (ensemble_stacking_training.ipynb) - Optional
1. **Load Checkpoint**: Use preprocessed data from main pipeline
2. **Train Base Models**: 4 diverse algorithms (RF, XGBoost, LightGBM, LR)
3. **Build Stacking**: Meta-learner combines base model predictions (5-fold CV)
4. **Threshold Tuning**: Optimize on validation set
5. **Final Retraining**: Retrain ensemble on train+val
6. **Comparison**: Compare with baseline performance
7. **Output**: Save ensemble artifacts to `outputs/ensemble/`

**Result**: Stacking achieves 98.52% ROC-AUC (+0.11% over baseline)

## Key Features

### Baseline Model
- Stratified train/val/test split (70/15/15) to prevent data leakage
- Comprehensive feature engineering (6 engineered features)
- 4 baseline models comparison (LR, RF, XGBoost, LightGBM)
- Hyperparameter tuning with Optuna (900s time budget)
- Threshold optimization for optimal F1 score
- SHAP interpretability analysis
- 7 professional visualizations (ROC, PR, confusion matrix, lift, etc.)

### Ensemble Stacking (Advanced)
- 4-model stacking architecture with meta-learner
- 5-fold cross-validation to prevent overfitting
- Side-by-side comparison with baseline
- Precision-Recall trade-off analysis
- Production deployment recommendations

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

### Baseline Model Outputs (churn_model_training.ipynb)
- `outputs/evaluation_summary.yaml` - Model performance metrics
- `outputs/final_model.pkl` - Trained Logistic Regression model
- `outputs/preprocessor.pkl` - Fitted preprocessing pipeline
- `outputs/selected_features.txt` - Final feature list
- `outputs/baseline_models_comparison.png` - 4 models comparison
- `outputs/roc_curve_test.png` - ROC curve
- `outputs/precision_recall_curve_test.png` - PR curve
- `outputs/confusion_matrix_test.png` - Confusion matrix heatmap
- `outputs/test_metrics_summary.png` - Metrics bar chart
- `outputs/lift_chart_test.png` - Lift chart
- `outputs/probability_distribution_test.png` - Probability distribution
- `outputs/shap_summary.png` - SHAP feature importance

### Ensemble Stacking Outputs (ensemble_stacking_training.ipynb)
- `outputs/ensemble/stacking_model.pkl` - Trained stacking ensemble
- `outputs/ensemble/stacking_evaluation_summary.yaml` - Ensemble metrics
- `outputs/ensemble/roc_curves_comparison.png` - ROC curves (all models)
- `outputs/ensemble/models_performance_comparison.png` - Performance bar chart

## Configuration
Edit `config.yaml` to customize:
- Data paths
- Train/val/test split ratios
- Feature lists
- Model hyperparameters
- Random seed for reproducibility

## Model Performance Summary

| Model | ROC-AUC | PR-AUC | F1-Score | Recall | Precision |
|-------|---------|--------|----------|--------|-----------|
| **Baseline (Logistic Regression)** | 0.9841 | 0.9562 | 0.8741 | 0.8517 | 0.8977 |
| **Stacking Ensemble** | 0.9852 | 0.9576 | 0.8796 | 0.8725 | 0.8867 |
| **Improvement** | +0.11% | +0.14% | +0.55% | +2.08% | -1.10% |

**Recommendation**: 
- Use **Baseline** for production (simpler, faster, 98.41% already excellent)
- Use **Stacking** if maximizing recall is critical (catches ~40 more churners per 10,000 customers)

## Notes
- Baseline execution time: ~5-10 minutes on standard laptop
- Ensemble execution time: ~15-20 minutes (trains 4 base models + meta-learner)
- Outputs are automatically saved to `outputs/` and `outputs/ensemble/`
- No GPU required (CPU-only)
- Deterministic results with fixed random seed (RANDOM_STATE=42)
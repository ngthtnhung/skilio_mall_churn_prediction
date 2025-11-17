import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve, auc)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(cm: np.ndarray, model_name: str = "", title: str = "Confusion Matrix") -> None:
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm (np.ndarray): Confusion matrix from sklearn.metrics.confusion_matrix
        model_name (str): Name of model (for title)
        title (str): Title for the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    full_title = f"{model_name} - {title}" if model_name else title
    plt.title(full_title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()
    logger.info(f"Confusion matrix plotted: {full_title}")


def plot_roc_curve(y_true: Union[np.ndarray, pd.Series],
                  y_pred_proba: Union[np.ndarray, pd.Series],
                  model_name: str = "",
                  title: str = "ROC Curve") -> None:
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true (Union[np.ndarray, pd.Series]): True labels
        y_pred_proba (Union[np.ndarray, pd.Series]): Predicted probabilities
        model_name (str): Name of model (for title)
        title (str): Title for the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    full_title = f"{model_name} - {title}" if model_name else title
    plt.title(full_title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    logger.info(f"ROC curve plotted: {full_title}")


def plot_pr_curve(y_true: Union[np.ndarray, pd.Series],
                 y_pred_proba: Union[np.ndarray, pd.Series],
                 model_name: str = "",
                 title: str = "Precision-Recall Curve") -> float:
    """
    Plot Precision-Recall curve with AUC score.
    
    PR-AUC is more important than ROC-AUC for imbalanced datasets (Churn prediction).
    
    Args:
        y_true (Union[np.ndarray, pd.Series]): True labels
        y_pred_proba (Union[np.ndarray, pd.Series]): Predicted probabilities
        model_name (str): Name of model (for title)
        title (str): Title for the plot
        
    Returns:
        float: PR-AUC score
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkgreen', lw=2.5, label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # No-skill classifier baseline
    no_skill = np.sum(y_true == 1) / len(y_true)
    plt.axhline(y=no_skill, color='gray', lw=2, linestyle='--', label=f'No-Skill Classifier ({no_skill:.1%})')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    full_title = f"{model_name} - {title}" if model_name else title
    plt.title(full_title, fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    logger.info(f"PR curve plotted: {full_title} (PR-AUC: {pr_auc:.4f})")
    return pr_auc


def plot_feature_importance(feature_importance: pd.DataFrame, 
                           model_name: str = "",
                           top_n: int = 20,
                           title: str = "Feature Importance") -> None:
    """
    Plot feature importance from model.
    
    Args:
        feature_importance (pd.DataFrame): DataFrame with 'feature' and 'importance' columns
        model_name (str): Name of model (for title)
        top_n (int): Number of top features to display
        title (str): Title for the plot
    """
    top_features = feature_importance.head(top_n)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
    plt.xlabel('Importance Score', fontsize=12)
    full_title = f"{model_name} - {title}" if model_name else title
    plt.title(full_title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    logger.info(f"Feature importance plotted: {full_title}")


def plot_models_comparison(all_results: Dict[str, Any], metric: str = 'roc_auc') -> None:
    """
    Plot comparison of multiple models for a given metric.
    
    Args:
        all_results (dict): Results from train_all_models()
        metric (str): Metric to compare ('accuracy', 'precision', 'recall', 'f1_score', 'roc_auc')
    """
    models = []
    train_scores = []
    val_scores = []
    test_scores = []
    
    for model_name, result in all_results.items():
        if 'error' not in result:
            models.append(model_name)
            train_scores.append(result['train_metrics'][metric])
            val_scores.append(result['val_metrics'][metric])
            test_scores.append(result['test_metrics'][metric])
    
    if not models:
        logger.warning("No valid results to plot")
        return
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, train_scores, width, label='Train', alpha=0.85, color='#1f77b4')
    plt.bar(x, val_scores, width, label='Validation', alpha=0.85, color='#ff7f0e')
    plt.bar(x + width, test_scores, width, label='Test', alpha=0.85, color='#2ca02c')
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel(metric.upper(), fontsize=12, fontweight='bold')
    plt.title(f'Model Comparison by {metric.upper()}', fontsize=14, fontweight='bold')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    logger.info(f"Model comparison plot created for {metric}")


def plot_lift_gain_chart(y_true: Union[np.ndarray, pd.Series],
                        y_pred_proba: Union[np.ndarray, pd.Series],
                        model_name: str = "",
                        percentile_bins: int = 10) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Plot Lift and Gain charts for ranking performance.
    
    Lift/Gain answers: "If we can only intervene on top X% of customers (by churn probability),
    how effective is our model compared to random selection?"
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of model (for title)
        percentile_bins: Number of bins for percentile (10 = deciles)
        
    Returns:
        Tuple of (lift_gain_df, summary_dict)
    """
    # Sort by predicted probability (descending)
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = np.array(y_true)[sorted_indices]
    
    # Calculate cumulative positives
    total_positives = np.sum(y_true_sorted)
    n_samples = len(y_true_sorted)
    
    # Create percentile bins
    bin_size = n_samples // percentile_bins
    percentiles = []
    gains = []
    lifts = []
    
    for i in range(1, percentile_bins + 1):
        # Get samples up to this percentile
        percentile_value = (i * 100) / percentile_bins
        bin_end = min(i * bin_size, n_samples)
        
        # Count positives in this bin
        positives_in_bin = np.sum(y_true_sorted[:bin_end])
        
        # Calculate Gain and Lift
        gain = (positives_in_bin / total_positives) * 100 if total_positives > 0 else 0
        baseline = percentile_value  # Random model's gain
        lift = (gain / baseline) if baseline > 0 else 0
        
        percentiles.append(percentile_value)
        gains.append(gain)
        lifts.append(lift)
    
    # Create dataframe
    lift_gain_df = pd.DataFrame({
        'Percentile': percentiles,
        'Gain (%)': gains,
        'Lift': lifts,
    })
    
    # Summary metrics
    summary = {
        'lift_at_10': lifts[0] if len(lifts) > 0 else 0,
        'lift_at_20': lifts[1] if len(lifts) > 1 else 0,
        'lift_at_30': lifts[2] if len(lifts) > 2 else 0,
        'gain_at_10': gains[0] if len(gains) > 0 else 0,
        'gain_at_20': gains[1] if len(gains) > 1 else 0,
        'gain_at_30': gains[2] if len(gains) > 2 else 0,
    }
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gain Chart
    ax1.plot(percentiles, gains, marker='o', linewidth=2.5, markersize=6, color='#2ca02c', label='Model')
    ax1.plot(percentiles, percentiles, linestyle='--', linewidth=2, color='gray', label='No-Skill')
    ax1.fill_between(percentiles, percentiles, gains, alpha=0.2, color='#2ca02c')
    ax1.set_xlabel('Percentile (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Gain (%)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{model_name} - Gain Chart' if model_name else 'Gain Chart', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Lift Chart
    ax2.plot(percentiles, lifts, marker='o', linewidth=2.5, markersize=6, color='#1f77b4', label='Model')
    ax2.axhline(y=1, linestyle='--', linewidth=2, color='gray', label='No-Skill (Lift=1)')
    ax2.fill_between(percentiles, 1, lifts, where=(np.array(lifts) >= 1), alpha=0.2, color='#1f77b4', interpolate=True)
    ax2.set_xlabel('Percentile (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Lift', fontsize=11, fontweight='bold')
    ax2.set_title(f'{model_name} - Lift Chart' if model_name else 'Lift Chart', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Log summary
    logger.info(f"\n{model_name} - Lift/Gain Summary:" if model_name else "\nLift/Gain Summary:")
    logger.info(f"  Lift @ 10%: {summary['lift_at_10']:.2f}x")
    logger.info(f"  Lift @ 20%: {summary['lift_at_20']:.2f}x")
    logger.info(f"  Lift @ 30%: {summary['lift_at_30']:.2f}x")
    logger.info(f"  Gain @ 10%: {summary['gain_at_10']:.1f}%")
    logger.info(f"  Gain @ 20%: {summary['gain_at_20']:.1f}%")
    logger.info(f"  Gain @ 30%: {summary['gain_at_30']:.1f}%")
    
    return lift_gain_df, summary


# ============================================================================
# EVALUATION LOGIC FUNCTIONS
# ============================================================================

def calculate_pr_auc(y_true: Union[np.ndarray, pd.Series],
                    y_pred_proba: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Precision-Recall AUC score.
    
    PR-AUC is more appropriate than ROC-AUC for imbalanced datasets like churn prediction.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        float: PR-AUC score (0-1)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    return pr_auc


def calculate_lift_gain(y_true: Union[np.ndarray, pd.Series],
                       y_pred_proba: Union[np.ndarray, pd.Series],
                       percentiles: list = [10, 20, 30]) -> Dict[str, float]:
    """
    Calculate Lift and Gain at specified percentiles.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        percentiles: List of percentiles to calculate (e.g., [10, 20, 30])
        
    Returns:
        dict: Dictionary with lift and gain values
    """
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = np.array(y_true)[sorted_indices]
    
    total_positives = np.sum(y_true_sorted)
    n_samples = len(y_true_sorted)
    
    results = {}
    
    for percentile in percentiles:
        # Get samples up to this percentile
        n_samples_percentile = max(1, int(n_samples * percentile / 100))
        positives_in_percentile = np.sum(y_true_sorted[:n_samples_percentile])
        
        # Calculate Gain and Lift
        gain = (positives_in_percentile / total_positives) * 100 if total_positives > 0 else 0
        baseline = percentile
        lift = (gain / baseline) if baseline > 0 else 0
        
        results[f'lift_{percentile}'] = lift
        results[f'gain_{percentile}'] = gain
    
    return results


def create_evaluation_summary(all_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create comprehensive summary table of all model evaluation metrics.
    
    Args:
        all_results (dict): Results from train_all_models()
        
    Returns:
        pd.DataFrame: Summary table with all metrics for all models
    """
    summary_data = []
    
    for model_name, result in all_results.items():
        if 'error' not in result:
            test_metrics = result['test_metrics']
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{test_metrics['accuracy']:.4f}",
                'Precision': f"{test_metrics['precision']:.4f}",
                'Recall': f"{test_metrics['recall']:.4f}",
                'F1-Score': f"{test_metrics['f1_score']:.4f}",
                'ROC-AUC': f"{test_metrics['roc_auc']:.4f}",
            })
    
    summary_df = pd.DataFrame(summary_data)
    logger.info("\nEvaluation Summary:")
    logger.info(summary_df.to_string(index=False))
    
    return summary_df


def get_best_model_info(all_results: Dict[str, Any], metric: str = 'roc_auc') -> Dict[str, Any]:
    """
    Get information about the best performing model.
    
    Args:
        all_results (dict): Results from train_all_models()
        metric (str): Metric to use for ranking
        
    Returns:
        dict: Information about the best model
    """
    best_model = None
    best_score = -1
    
    for model_name, result in all_results.items():
        if 'error' not in result:
            score = result['test_metrics'][metric]
            if score > best_score:
                best_score = score
                best_model = model_name
    
    if best_model is None:
        logger.warning("No valid models found")
        return None
    
    best_result = all_results[best_model]
    
    info = {
        'model_name': best_model,
        'metric_used': metric,
        'test_score': best_score,
        'train_metrics': best_result['train_metrics'],
        'val_metrics': best_result['val_metrics'],
        'test_metrics': best_result['test_metrics'],
        'model_object': best_result['model'],
    }
    
    logger.info(f"\nBest Model: {best_model}")
    logger.info(f"  {metric.upper()}: {best_score:.4f}")
    logger.info(f"  Accuracy: {best_result['test_metrics']['accuracy']:.4f}")
    logger.info(f"  F1-Score: {best_result['test_metrics']['f1_score']:.4f}")
    
    return info


def compare_metrics_across_models(all_results: Dict[str, Any], 
                                 metric: str = 'roc_auc') -> pd.DataFrame:
    """
    Compare specific metric across all models on Train/Val/Test sets.
    
    Args:
        all_results (dict): Results from train_all_models()
        metric (str): Metric to compare
        
    Returns:
        pd.DataFrame: Comparison table sorted by Test score
    """
    comparison = []
    
    for model_name, result in all_results.items():
        if 'error' not in result:
            comparison.append({
                'Model': model_name,
                'Train': f"{result['train_metrics'][metric]:.4f}",
                'Validation': f"{result['val_metrics'][metric]:.4f}",
                'Test': f"{result['test_metrics'][metric]:.4f}",
            })
    
    comparison_df = pd.DataFrame(comparison)
    
    logger.info(f"\nMetric Comparison ({metric.upper()}):")
    logger.info(comparison_df.to_string(index=False))
    
    return comparison_df


def get_model_predictions_summary(model_obj, X_test, y_test) -> Dict[str, Any]:
    """
    Get summary of model predictions on test set.
    
    Args:
        model_obj: ChurnModel object (trained)
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Summary statistics of predictions
    """
    predictions, probabilities = model_obj.predict(X_test, probability=True)
    
    summary = {
        'total_samples': len(y_test),
        'predicted_positive': (predictions == 1).sum(),
        'predicted_negative': (predictions == 0).sum(),
        'actual_positive': (y_test == 1).sum(),
        'actual_negative': (y_test == 0).sum(),
        'true_positives': ((predictions == 1) & (y_test == 1)).sum(),
        'true_negatives': ((predictions == 0) & (y_test == 0)).sum(),
        'false_positives': ((predictions == 1) & (y_test == 0)).sum(),
        'false_negatives': ((predictions == 0) & (y_test == 1)).sum(),
        'mean_probability': probabilities.mean(),
        'min_probability': probabilities.min(),
        'max_probability': probabilities.max(),
    }
    
    logger.info("\nPrediction Summary:")
    logger.info(f"  Total Samples: {summary['total_samples']}")
    logger.info(f"  Predicted Positive: {summary['predicted_positive']} ({summary['predicted_positive']/summary['total_samples']:.1%})")
    logger.info(f"  Predicted Negative: {summary['predicted_negative']} ({summary['predicted_negative']/summary['total_samples']:.1%})")
    logger.info(f"  True Positives: {summary['true_positives']}")
    logger.info(f"  True Negatives: {summary['true_negatives']}")
    logger.info(f"  False Positives: {summary['false_positives']}")
    logger.info(f"  False Negatives: {summary['false_negatives']}")
    logger.info(f"  Mean Probability: {summary['mean_probability']:.4f}")
    
    return summary

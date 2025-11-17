# File: src/modeling.py

import pandas as pd
import numpy as np
import logging
import pickle
from typing import Dict, Tuple, Any, Union
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChurnModel:
    """
    Wrapper class for churn prediction model with support for multiple algorithms.
    
    Supported models:
    - LogisticRegression
    - RandomForest
    - XGBoost
    - LightGBM
    
    All models handle imbalanced data with class_weight='balanced' (or scale_pos_weight for XGBoost).
    """
    
    def __init__(self, config: Dict[str, Any], model_name: str = 'LightGBM'):
        """
        Initialize ChurnModel with specified algorithm.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary with model parameters
            model_name (str): Name of model to use ('LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM')
        """
        self.config = config
        self.model = None
        self.model_name = model_name
        self.available_models = ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM']
        
        # Validate model name
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not supported. Choose from: {self.available_models}")
        
        # Get model configuration
        if 'MODELS' in config and model_name in config['MODELS']:
            model_config = config['MODELS'][model_name]
            self.hyperparameters = model_config.get('HYPERPARAMETERS', {})
        else:
            raise ValueError(f"Configuration for '{model_name}' not found in config")
        
        self.random_state = config['GENERAL']['RANDOM_STATE']
        
        logger.info(f"Initializing {self.model_name} model with hyperparameters: {self.hyperparameters}")
        
        # Initialize model
        self._build_model()
    
    def _build_model(self) -> None:
        """Build and initialize the model based on model_name."""
        try:
            if self.model_name == 'LogisticRegression':
                self.model = LogisticRegression(**self.hyperparameters)
                
            elif self.model_name == 'RandomForest':
                self.model = RandomForestClassifier(**self.hyperparameters)
                
            elif self.model_name == 'XGBoost':
                # Remove scale_pos_weight temporarily, add when fitting
                hyperparams = self.hyperparameters.copy()
                self.scale_pos_weight = hyperparams.pop('scale_pos_weight', 3)
                self.model = XGBClassifier(**hyperparams, verbosity=0)
                
            elif self.model_name == 'LightGBM':
                hyperparams = self.hyperparameters.copy()
                self.model = LGBMClassifier(**hyperparams, verbose=-1)
            
            logger.info(f"{self.model_name} model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error building {self.model_name} model: {e}")
            raise
    
    def train(self, X_train: Union[np.ndarray, pd.DataFrame], 
             y_train: Union[np.ndarray, pd.Series],
             X_val: Union[np.ndarray, pd.DataFrame] = None,
             y_val: Union[np.ndarray, pd.Series] = None,
             eval_metric: str = 'auc') -> Dict[str, Any]:
        """
        Train the model with optional validation set.
        
        Note: 
        - LogisticRegression: Does not support eval_set (ignored)
        - RandomForest: Does not support eval_set (ignored)
        - XGBoost & LightGBM: Support early stopping with eval_set
        
        Args:
            X_train (Union[np.ndarray, pd.DataFrame]): Training features
            y_train (Union[np.ndarray, pd.Series]): Training target
            X_val (Union[np.ndarray, pd.DataFrame]): Validation features (optional)
            y_val (Union[np.ndarray, pd.Series]): Validation target (optional)
            eval_metric (str): Evaluation metric for tree-based models
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info(f"Starting {self.model_name} training...")
        logger.info(f"Training set size: {X_train.shape}")
        
        try:
            if self.model_name in ['LogisticRegression', 'RandomForest']:
                # These models don't support eval_set
                self.model.fit(X_train, y_train)
                logger.info(f"{self.model_name} training completed successfully")
                
            else:
                # XGBoost and LightGBM support eval_set
                eval_set = None
                if X_val is not None and y_val is not None:
                    eval_set = [(X_val, y_val)]
                    logger.info(f"Validation set size: {X_val.shape}")
                
                fit_params = {'eval_set': eval_set, 'eval_metric': eval_metric}
                
                if self.model_name == 'XGBoost':
                    fit_params['verbose'] = False
                
                self.model.fit(X_train, y_train, **fit_params)
                logger.info(f"{self.model_name} training completed successfully")
            
            # Get training results
            results = {
                'model_name': self.model_name,
                'training_complete': True,
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during {self.model_name} training: {e}")
            raise
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame], 
               probability: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on new data.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features to predict
            probability (bool): If True, return probabilities instead of class labels
            
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
            - If probability=False: Predicted class labels (0 or 1)
            - If probability=True: Tuple of (class_labels, probabilities)
        """
        if self.model is None:
            logger.error("Model not trained yet")
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            predictions = self.model.predict(X)
            
            if probability:
                probabilities = self.model.predict_proba(X)[:, 1]
                return predictions, probabilities
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during {self.model_name} prediction: {e}")
            raise
    
    def evaluate(self, X_test: Union[np.ndarray, pd.DataFrame],
                y_test: Union[np.ndarray, pd.Series],
                dataset_name: str = "Test") -> Dict[str, Any]:
        """
        Evaluate model performance on a dataset.
        
        Calculates:
        - Accuracy, Precision, Recall, F1 Score
        - ROC-AUC Score
        - Confusion Matrix
        - Classification Report
        
        Args:
            X_test (Union[np.ndarray, pd.DataFrame]): Test features
            y_test (Union[np.ndarray, pd.Series]): Test target
            dataset_name (str): Name of dataset (for logging)
            
        Returns:
            Dict[str, Any]: Dictionary with all evaluation metrics
        """
        logger.info(f"Evaluating {self.model_name} on {dataset_name} set...")
        
        try:
            # Get predictions
            y_pred, y_pred_proba = self.predict(X_test, probability=True)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Package results
            results = {
                'model_name': self.model_name,
                'dataset_name': dataset_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
            }
            
            # Log results
            logger.info(f"{self.model_name} - {dataset_name} Set Results:")
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  F1 Score:  {f1:.4f}")
            logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during {self.model_name} evaluation: {e}")
            raise
    
    def get_feature_importance(self, top_n: int = 20, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Note: LogisticRegression uses absolute values of coefficients
        
        Args:
            top_n (int): Number of top features to return
            feature_names (list): List of original feature names from preprocessing.
                                 If None, uses generic Feature_0, Feature_1, etc.
                                 Typically obtained from OneHotEncoder or preprocessing output.
                                 Example: ['age', 'country_Vietnam', 'country_Thailand', 'app_version_2.0', ...]
            
        Returns:
            pd.DataFrame: DataFrame with features and importance scores, sorted by importance descending
        """
        if self.model is None:
            logger.error("Model not trained yet")
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            n_features = self.model.n_features_in_
            
            # Use provided feature names or generate generic ones
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(n_features)]
            elif len(feature_names) != n_features:
                logger.warning(
                    f"Feature names length ({len(feature_names)}) does not match "
                    f"number of features ({n_features}). Using generic names."
                )
                feature_names = [f"Feature_{i}" for i in range(n_features)]
            
            if self.model_name == 'LogisticRegression':
                # Use absolute values of coefficients
                importances = np.abs(self.model.coef_[0])
                
            elif self.model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
                # Use built-in feature_importances_
                importances = self.model.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            logger.info(f"Top {top_n} most important features retrieved for {self.model_name}")
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            raise
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            logger.error("Model not trained yet")
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"{self.model_name} model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving {self.model_name} model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath (str): Path to load the model from
        """
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"{self.model_name} model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading {self.model_name} model: {e}")
            raise


def train_and_evaluate_model(X_train: Union[np.ndarray, pd.DataFrame],
                           y_train: Union[np.ndarray, pd.Series],
                           X_val: Union[np.ndarray, pd.DataFrame],
                           y_val: Union[np.ndarray, pd.Series],
                           X_test: Union[np.ndarray, pd.DataFrame],
                           y_test: Union[np.ndarray, pd.Series],
                           config: Dict[str, Any],
                           model_name: str = 'LightGBM') -> Dict[str, Any]:
    """
    Complete training and evaluation pipeline for a single model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        config: Configuration dictionary
        model_name: Name of model to train ('LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM')
        
    Returns:
        Dict[str, Any]: Complete results including model and metrics
    """
    logger.info(f"Starting training and evaluation pipeline for {model_name}...")
    
    try:
        # Initialize model
        model = ChurnModel(config, model_name=model_name)
        
        # Train model
        logger.info(f"Training phase for {model_name}...")
        train_results = model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on all sets
        logger.info(f"Evaluation phase for {model_name}...")
        train_eval = model.evaluate(X_train, y_train, dataset_name="Train")
        val_eval = model.evaluate(X_val, y_val, dataset_name="Validation")
        test_eval = model.evaluate(X_test, y_test, dataset_name="Test")
        
        # Get feature importance
        logger.info(f"Extracting feature importance for {model_name}...")
        feature_importance = model.get_feature_importance(top_n=20)
        
        # Package all results
        pipeline_results = {
            'model_name': model_name,
            'model': model,
            'train_metrics': train_eval,
            'val_metrics': val_eval,
            'test_metrics': test_eval,
            'feature_importance': feature_importance,
        }
        
        # Summary
        logger.info(f"{model_name} pipeline completed successfully")
        logger.info(f"{model_name} Test ROC-AUC: {test_eval['roc_auc']:.4f}")
        logger.info(f"{model_name} Test F1 Score: {test_eval['f1_score']:.4f}")
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"{model_name} pipeline failed: {e}")
        raise


def train_all_models(X_train: Union[np.ndarray, pd.DataFrame],
                    y_train: Union[np.ndarray, pd.Series],
                    X_val: Union[np.ndarray, pd.DataFrame],
                    y_val: Union[np.ndarray, pd.Series],
                    X_test: Union[np.ndarray, pd.DataFrame],
                    y_test: Union[np.ndarray, pd.Series],
                    config: Dict[str, Any],
                    models: list = None) -> Dict[str, Any]:
    """
    Train and evaluate multiple models in sequence.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        config: Configuration dictionary
        models: List of model names to train. If None, trains all available models.
               Options: 'LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM'
        
    Returns:
        Dict[str, Any]: Results from all trained models
    """
    if models is None:
        models = ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM']
    
    logger.info(f"Starting training pipeline for {len(models)} models: {models}")
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Validation set size: {X_val.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    
    all_results = {}
    
    for model_name in models:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training model: {model_name}")
            logger.info(f"{'='*60}\n")
            
            result = train_and_evaluate_model(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                config,
                model_name=model_name
            )
            
            all_results[model_name] = result
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}
    
    # Print summary comparison
    logger.info(f"\n{'='*60}")
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info(f"{'='*60}\n")
    
    for model_name, result in all_results.items():
        if 'error' not in result:
            test_metrics = result['test_metrics']
            logger.info(f"{model_name:20s} - Accuracy: {test_metrics['accuracy']:.4f}, "
                       f"Precision: {test_metrics['precision']:.4f}, "
                       f"Recall: {test_metrics['recall']:.4f}, "
                       f"F1: {test_metrics['f1_score']:.4f}, "
                       f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
        else:
            logger.info(f"{model_name:20s} - ERROR: {result['error']}")
    
    logger.info(f"{'='*60}\n")
    
    return all_results


def compare_models_metrics(all_results: Dict[str, Any], metric: str = 'roc_auc') -> pd.DataFrame:
    """
    Compare metrics across all trained models.
    
    Args:
        all_results (Dict[str, Any]): Results from train_all_models()
        metric (str): Metric to compare ('accuracy', 'precision', 'recall', 'f1_score', 'roc_auc')
        
    Returns:
        pd.DataFrame: Comparison DataFrame with models and metrics on different sets
    """
    comparison = []
    
    for model_name, result in all_results.items():
        if 'error' not in result:
            comparison.append({
                'Model': model_name,
                'Train': result['train_metrics'][metric],
                'Validation': result['val_metrics'][metric],
                'Test': result['test_metrics'][metric],
            })
    
    comparison_df = pd.DataFrame(comparison).sort_values('Test', ascending=False)
    logger.info(f"\nModel Comparison by {metric.upper()}:")
    logger.info(comparison_df.to_string())
    
    return comparison_df

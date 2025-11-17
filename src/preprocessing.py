import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Union
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_column_lists(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Extract numerical and categorical feature lists from config and validate.
    
    Args:
        df: DataFrame to analyze
        config: Configuration dictionary with feature lists
        
    Returns:
        Tuple of numerical and categorical feature lists
        
    Raises:
        KeyError: If required config keys missing
        ValueError: If features not found in DataFrame
    """
    if 'FEATURE_ENGINEERING' not in config:
        logger.error("Config missing 'FEATURE_ENGINEERING' key")
        raise KeyError("Config missing 'FEATURE_ENGINEERING' key")
    
    num_feat = config['FEATURE_ENGINEERING'].get('NUMERICAL_FEATURES', [])
    cat_feat = config['FEATURE_ENGINEERING'].get('CATEGORICAL_FEATURES', [])
    
    missing_num = [col for col in num_feat if col not in df.columns]
    missing_cat = [col for col in cat_feat if col not in df.columns]
    
    if missing_num or missing_cat:
        error_msg = f"Missing columns - Numerical: {missing_num}, Categorical: {missing_cat}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Column lists validated: {len(num_feat)} numerical, {len(cat_feat)} categorical")
    return num_feat, cat_feat


def create_preprocessor(numerical_features: List[str], 
                       categorical_features: List[str],
                       sparse_output: bool = False) -> ColumnTransformer:
    """
    Create ColumnTransformer for preprocessing features.
    Numerical: median imputation + scaling
    Categorical: constant imputation + one-hot encoding
    
    Args:
        numerical_features: List of numerical column names
        categorical_features: List of categorical column names
        sparse_output: Use sparse matrix for OneHotEncoder (False for tree models)
        
    Returns:
        ColumnTransformer for preprocessing
    """
    logger.info(f"Creating preprocessor with {len(numerical_features)} numerical and {len(categorical_features)} categorical features")
    logger.info(f"Sparse output: {sparse_output}")
    
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=sparse_output))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )
    
    logger.info("Preprocessor created successfully")
    return preprocessor


def handle_missing_values(df: pd.DataFrame, 
                         numerical_strategy: str = 'median',
                         categorical_strategy: str = 'mode') -> pd.DataFrame:
    """
    Handle missing values before preprocessing.
    
    Args:
        df: Input DataFrame with potential missing values
        numerical_strategy: Strategy for numerical columns
        categorical_strategy: Strategy for categorical columns
        
    Returns:
        DataFrame with missing values handled
    """
    logger.info(f"Handling missing values - Numerical: {numerical_strategy}, Categorical: {categorical_strategy}")
    
    df = df.copy()
    
    initial_nan = df.isna().sum().sum()
    if initial_nan > 0:
        logger.warning(f"Detected {initial_nan} missing values before handling")
        logger.info(f"Missing values per column:\n{df.isna().sum()[df.isna().sum() > 0]}")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            if numerical_strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif numerical_strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif numerical_strategy == 'drop':
                df.dropna(subset=[col], inplace=True)
            logger.info(f"Handled {nan_count} missing values in {col} using {numerical_strategy}")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            if categorical_strategy == 'mode':
                mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'missing'
                df[col].fillna(mode_value, inplace=True)
            elif categorical_strategy == 'missing':
                df[col].fillna('missing', inplace=True)
            elif categorical_strategy == 'drop':
                df.dropna(subset=[col], inplace=True)
            logger.info(f"Handled {nan_count} missing values in {col} using {categorical_strategy}")
    
    final_nan = df.isna().sum().sum()
    logger.info(f"Missing value handling complete: {initial_nan} -> {final_nan} NaN values")
    
    if final_nan > 0:
        logger.warning(f"Warning: {final_nan} NaN values remain after handling")
    
    return df


def validate_data_quality(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate data quality before preprocessing.
    Checks for empty DataFrame, duplicates, missing values, and target column.
    
    Args:
        df: DataFrame to validate
        config: Configuration dictionary
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Starting data quality validation...")
    
    validation_results = {
        'is_valid': True,
        'empty': df.empty,
        'shape': df.shape,
        'duplicates': df.duplicated().sum(),
        'nan_count': df.isna().sum().sum(),
        'nan_percent': (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100) if df.shape[0] > 0 else 0,
    }
    
    if validation_results['empty']:
        logger.error("DataFrame is empty")
        validation_results['is_valid'] = False
    
    if validation_results['duplicates'] > 0:
        logger.warning(f"Found {validation_results['duplicates']} duplicate rows")
    
    if validation_results['nan_count'] > 0:
        logger.warning(f"Found {validation_results['nan_count']} ({validation_results['nan_percent']:.2f}%) NaN values")
    target_col = config['GENERAL'].get('TARGET_COLUMN')
    if target_col and target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in DataFrame")
        validation_results['is_valid'] = False
    
    logger.info(f"Validation complete - Valid: {validation_results['is_valid']}")
    return validation_results


def get_clean_split_data(raw_df: pd.DataFrame, config: Dict[str, Any]) -> Tuple:
    """
    Main preprocessing pipeline: validate, split, fit preprocessor, transform.
    Prevents data leakage by fitting only on train set.
    
    Args:
        raw_df: Raw input DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple of processed train/val/test sets, labels, and fitted preprocessor
                
    Raises:
        ValueError: If data validation fails
    """
    logger.info("Starting preprocessing and data splitting pipeline...")
    
    try:
        logger.info("Step 1: Data quality validation")
        validation = validate_data_quality(raw_df, config)
        if not validation['is_valid']:
            raise ValueError("Data validation failed")
        
        logger.info("Step 2: Extracting target and features")
        target_col = config['GENERAL']['TARGET_COLUMN']
        y = raw_df[target_col]
        X = raw_df.drop(columns=[target_col])
        
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        logger.info("Step 3: Splitting data with stratification")
        train_ratio = config['DATA_SPLIT']['TRAIN_RATIO']
        val_ratio = config['DATA_SPLIT']['VAL_RATIO']
        test_ratio = config['DATA_SPLIT']['TEST_RATIO']
        random_state = config['GENERAL']['RANDOM_STATE']
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=y
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size),
            random_state=random_state,
            stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        logger.info("Step 4: Creating and fitting preprocessor on training data")
        num_feats, cat_feats = get_column_lists(X_train, config)
        preprocessor = create_preprocessor(num_feats, cat_feats, sparse_output=False)
        
        preprocessor.fit(X_train)
        logger.info("Preprocessor fitted successfully")
        
        logger.info("Step 5: Transforming all sets with fitted preprocessor")
        X_train_processed = preprocessor.transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        
        logger.info(f"Train set shape: {X_train_processed.shape}")
        logger.info(f"Val set shape: {X_val_processed.shape}")
        logger.info(f"Test set shape: {X_test_processed.shape}")
        
        logger.info("Pipeline execution successful")
        
        return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, preprocessor
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
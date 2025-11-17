import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TENURE_REQUIRED_COLS = ['orders_2024', 'sessions_30d', 'reg_days']
INTENSITY_REQUIRED_COLS = ['sessions_30d', 'sessions_90d', 'refunds_count_2024', 'orders_2024', 'support_tickets_2024']
INTERACTION_REQUIRED_COLS = ['discount_rate_2024', 'aov_2024']


def _validate_columns(df: pd.DataFrame, required_cols: List[str], function_name: str) -> bool:
    """
    Validate DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        function_name: Name of calling function
        
    Returns:
        True if all columns exist
        
    Raises:
        ValueError: If required columns are missing
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        error_msg = f"{function_name}: Missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return True


def _create_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features normalized by user tenure.
    Assumes input has no NaN values.
    
    Args:
        df: Input DataFrame with orders_2024, sessions_30d, reg_days
        
    Returns:
        DataFrame with new tenure features
        
    Raises:
        ValueError: If required columns are missing
    """
    _validate_columns(df, TENURE_REQUIRED_COLS, "_create_tenure_features")
    
    logger.info("Creating tenure features...")
    
    df['orders_per_day'] = df['orders_2024'] / np.maximum(df['reg_days'], 1)
    df['sessions_per_day'] = df['sessions_30d'] / np.maximum(df['reg_days'], 1)
    
    logger.info("Tenure features created successfully")
    return df


def _create_intensity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio features indicating activity intensity and engagement trends.
    Assumes input has no NaN values.
    
    Args:
        df: Input DataFrame with required columns
        
    Returns:
        DataFrame with new ratio features
        
    Raises:
        ValueError: If required columns are missing
    """
    _validate_columns(df, INTENSITY_REQUIRED_COLS, "_create_intensity_ratios")
    
    logger.info("Creating intensity ratio features...")
    
    df['recent_session_ratio'] = df['sessions_30d'] / np.maximum(df['sessions_90d'], 1)
    df['refund_to_order_ratio'] = df['refunds_count_2024'] / np.maximum(df['orders_2024'], 1)
    df['support_per_order'] = df['support_tickets_2024'] / np.maximum(df['orders_2024'], 1)
    
    logger.info("Intensity ratio features created successfully")
    return df


def _create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features capturing combined effects of variables.
    Assumes input has no NaN values.
    
    Args:
        df: Input DataFrame with required columns
        
    Returns:
        DataFrame with new interaction features
        
    Raises:
        ValueError: If required columns are missing
    """
    _validate_columns(df, INTERACTION_REQUIRED_COLS, "_create_interaction_features")
    
    logger.info("Creating interaction features...")
    
    df['discount_aov_interaction'] = df['discount_rate_2024'] * df['aov_2024']
    
    logger.info("Interaction features created successfully")
    return df


def create_all_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Execute complete feature engineering pipeline.
    Creates tenure, intensity ratio, and interaction features.
    Expects clean input with no NaN values.
    
    Args:
        df_raw: Clean input DataFrame
        
    Returns:
        DataFrame with all engineered features
        
    Raises:
        ValueError: If required columns are missing
    """
    logger.info("Starting feature engineering pipeline...")
    
    df = df_raw.copy()
    initial_shape = df.shape
    
    try:
        df = _create_tenure_features(df)
        df = _create_intensity_ratios(df)
        df = _create_interaction_features(df)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        final_nan_count = df.isna().sum().sum()
        logger.info(f"Feature engineering completed")
        logger.info(f"Input shape: {initial_shape}, Output shape: {df.shape}")
        logger.info(f"Total NaN values in output: {final_nan_count}")
        
        return df
        
    except ValueError as e:
        logger.error(f"Feature engineering failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during feature engineering: {e}")
        raise
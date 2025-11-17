import pandas as pd
import yaml
import os
import logging
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = '../config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If YAML file is invalid
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at: {config_path}")
        raise FileNotFoundError(f"Error: Config file not found at {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded from: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error reading YAML: {e}")
        raise

def load_raw_data(config_path: str = '../config.yaml') -> pd.DataFrame:
    """
    Load raw SkilioMall data with optimized data types for memory efficiency.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        DataFrame containing loaded data
        
    Raises:
        FileNotFoundError: If data file does not exist
        ValueError: If data is invalid
    """
    try:
        config = load_config(config_path)
        data_path = config['PATHS']['RAW_DATA']
        
        # Resolve relative path from config directory
        if not os.path.isabs(data_path):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            data_path = os.path.join(config_dir, data_path)
        
        if not os.path.exists(data_path):
            logger.error(f"Data file not found at: {data_path}")
            raise FileNotFoundError(f"Error: Data file not found at {data_path}")
        
        logger.info(f"Starting to load data from: {data_path}")
        
        # Define optimized data types to reduce memory usage by approximately 60%
        dtype_spec = {
            config['GENERAL']['ID_COLUMN']: str,
            config['GENERAL']['TARGET_COLUMN']: 'int8',
            'country': 'category',
            'city': 'category',
            'app_version_major': 'category',
        }
        
        # Load CSV with optimized types
        df = pd.read_csv(
            data_path,
            dtype=dtype_spec,
            index_col=config['GENERAL']['ID_COLUMN']
        )
        
        # Validate loaded data
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        if df.empty:
            logger.warning("Loaded data is empty!")
            return df
        
        # Verify target column exists
        target_col = config['GENERAL']['TARGET_COLUMN']
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' does not exist in data")
            raise ValueError(f"Target column '{target_col}' does not exist")
        
        # Log memory and data statistics
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info(f"Columns: {', '.join(df.columns[:5])}... ({df.shape[1]} total)")
        logger.info(f"Index name: {df.index.name}")
        
        # Report missing values
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"Detected {nan_count} NaN values in data")
            logger.info(f"NaN per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        
        logger.info("Data loaded successfully")
        return df
        
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        raise
    except KeyError as e:
        logger.error(f"Error: Config missing key '{e}' or data missing column '{e}'")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for DataFrame.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary containing summary information
    """
    summary = {
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'nan_count': df.isnull().sum().sum(),
        'nan_percent': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'duplicate_rows': df.duplicated().sum(),
    }
    return summary


def validate_required_columns(df: pd.DataFrame, required_cols: list) -> bool:
    """
    Verify DataFrame contains all required columns.
    
    Args:
        df: DataFrame to check
        required_cols: List of required column names
        
    Returns:
        True if all columns exist, False otherwise
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        return False
    
    logger.info(f"All {len(required_cols)} required columns are present")
    return True
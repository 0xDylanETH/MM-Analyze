import pandas as pd
import numpy as np
from typing import Optional, Tuple

def load_data(csv_file_path: str) -> Optional[pd.DataFrame]:
    """Load data from CSV file with basic validation."""
    try:
        df = pd.read_csv(csv_file_path)
        print(f"✅ Loaded {len(df):,} trader wallets")
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        print("💡 Check:")
        print("- File path is correct?")
        print("- CSV format is valid?")
        return None

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the dataset."""
    initial_count = len(df)
    
    # Remove missing values in critical columns
    critical_cols = ['total_transactions', 'avg_daily_transactions', 'total_pnl', 'total_volume']
    available_cols = [col for col in critical_cols if col in df.columns]
    if available_cols:
        df = df.dropna(subset=available_cols)
    
    # Remove extreme outliers
    if 'pnl_percentage' in df.columns:
        df = df[(df['pnl_percentage'] >= -99) & (df['pnl_percentage'] <= 2000)]
    
    # Ensure positive volume
    if 'total_volume' in df.columns:
        df = df[df['total_volume'] > 0]
    
    # Calculate PnL percentage if not available
    if 'pnl_percentage' not in df.columns and 'total_pnl' in df.columns and 'total_volume' in df.columns:
        df['pnl_percentage'] = (df['total_pnl'] / df['total_volume']) * 100
    
    print(f"📊 Final dataset: {len(df):,} wallets ({len(df)/initial_count*100:.1f}% of original)")
    return df

def prepare_trader_groups(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split traders into active and normal groups."""
    if 'trader_type' in df.columns:
        active_traders = df[df['trader_type'] == 'Active']
        normal_traders = df[df['trader_type'] == 'Normal']
    else:
        threshold = df['avg_daily_transactions'].quantile(0.7)
        active_traders = df[df['avg_daily_transactions'] >= threshold]
        normal_traders = df[df['avg_daily_transactions'] < threshold]
    
    return active_traders, normal_traders 
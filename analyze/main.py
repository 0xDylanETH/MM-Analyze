from typing import Optional, Tuple, Dict
import pandas as pd

from .data_loader import load_data, clean_data
from .statistical_analysis import analyze_trading_patterns
from .visualization import create_analysis_plots
from .insights import generate_insights

def run_analysis(csv_file_path: str) -> Optional[Tuple[pd.DataFrame, Dict]]:
    """
    Run complete trading pattern analysis.
    
    Args:
        csv_file_path: Path to the CSV file containing trading data
        
    Returns:
        Tuple of (processed DataFrame, analysis results) if successful, None otherwise
    """
    print("🎯 SOLANA BLOCKCHAIN TRADING PATTERN ANALYSIS")
    print("🔬 Hypothesis: Active Traders có performance khác biệt so với Normal Traders")
    print("📊 Data source: ClickHouse query từ Solana blockchain")
    print()
    
    # Step 1: Load data
    df = load_data(csv_file_path)
    if df is None:
        return None
    
    # Step 2: Clean data
    df = clean_data(df)
    if len(df) < 50:
        print("❌ Không đủ dữ liệu để phân tích (cần ít nhất 50 records)")
        return None
    
    # Step 3: Statistical analysis
    results = analyze_trading_patterns(df)
    if results is None:
        return None
    
    # Step 4: Visualizations
    create_analysis_plots(df, results)
    
    # Step 5: Additional insights
    insights = generate_insights(df, results)
    
    print(f"\n" + "=" * 60)
    print("✅ PHÂN TÍCH HOÀN THÀNH!")
    print(f"📊 Đã phân tích {len(df):,} ví trader từ Solana blockchain")
    print(f"🔬 Kết quả kiểm định hypothesis đã được hiển thị ở trên")
    print(f"📈 Biểu đồ và insights chi tiết đã được tạo")
    
    return df, {**results, 'insights': insights}

if __name__ == "__main__":
    # Example usage
    csv_file_path = "datawallet2.csv"  # Replace with your data file path
    run_analysis(csv_file_path) 
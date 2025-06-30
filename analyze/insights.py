from typing import Dict
import pandas as pd
import numpy as np

def analyze_top_performers(df: pd.DataFrame, pnl_col: str) -> Dict:
    """Analyze characteristics of top performing traders."""
    top_10_percent = df.nlargest(int(len(df) * 0.1), pnl_col)
    
    if 'trader_type' in df.columns:
        top_active_ratio = (top_10_percent['trader_type'] == 'Active').mean() * 100
    else:
        top_active_ratio = (top_10_percent['avg_daily_transactions'] >= 
                          df['avg_daily_transactions'].quantile(0.7)).mean() * 100
    
    return {
        'count': len(top_10_percent),
        'active_ratio': top_active_ratio,
        'mean_pnl': top_10_percent[pnl_col].mean(),
        'mean_tx': top_10_percent['avg_daily_transactions'].mean(),
        'mean_volume': top_10_percent['total_volume'].mean()
    }

def calculate_correlations(df: pd.DataFrame, pnl_col: str) -> Dict:
    """Calculate correlations between key metrics."""
    return {
        'tx_pnl': df['avg_daily_transactions'].corr(df[pnl_col]),
        'volume_pnl': df['total_volume'].corr(df[pnl_col]),
        'tx_volume': df['avg_daily_transactions'].corr(df['total_volume'])
    }

def analyze_high_volume_traders(df: pd.DataFrame, pnl_col: str) -> Dict:
    """Analyze characteristics of high volume traders."""
    high_volume_threshold = df['total_volume'].quantile(0.8)
    high_volume_traders = df[df['total_volume'] >= high_volume_threshold]
    
    if 'trader_type' in df.columns:
        hv_active_ratio = (high_volume_traders['trader_type'] == 'Active').mean() * 100
    else:
        hv_active_ratio = (high_volume_traders['avg_daily_transactions'] >= 
                          df['avg_daily_transactions'].quantile(0.7)).mean() * 100
    
    return {
        'count': len(high_volume_traders),
        'active_ratio': hv_active_ratio,
        'mean_pnl': high_volume_traders[pnl_col].mean(),
        'volume_threshold': high_volume_threshold
    }

def analyze_performance_distribution(df: pd.DataFrame, pnl_col: str) -> Dict:
    """Analyze distribution of profitable vs loss-making traders."""
    profit_traders = df[df[pnl_col] > 0]
    loss_traders = df[df[pnl_col] <= 0]
    
    if 'trader_type' in df.columns:
        profit_active_ratio = (profit_traders['trader_type'] == 'Active').mean() * 100
        loss_active_ratio = (loss_traders['trader_type'] == 'Active').mean() * 100
    else:
        threshold = df['avg_daily_transactions'].quantile(0.7)
        profit_active_ratio = (profit_traders['avg_daily_transactions'] >= threshold).mean() * 100
        loss_active_ratio = (loss_traders['avg_daily_transactions'] >= threshold).mean() * 100
    
    return {
        'profit_count': len(profit_traders),
        'profit_ratio': len(profit_traders) / len(df) * 100,
        'profit_active_ratio': profit_active_ratio,
        'loss_count': len(loss_traders),
        'loss_ratio': len(loss_traders) / len(df) * 100,
        'loss_active_ratio': loss_active_ratio
    }

def calculate_risk_reward(df: pd.DataFrame, pnl_col: str) -> Dict:
    """Calculate risk-reward metrics."""
    profit_traders = df[df[pnl_col] > 0]
    loss_traders = df[df[pnl_col] <= 0]
    
    if len(profit_traders) > 0 and len(loss_traders) > 0:
        avg_profit = profit_traders[pnl_col].mean()
        avg_loss = abs(loss_traders[pnl_col].mean())
        risk_reward_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')
        
        return {
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'risk_reward_ratio': risk_reward_ratio
        }
    return None

def generate_insights(df: pd.DataFrame, results: Dict) -> Dict:
    """Generate comprehensive insights from the analysis."""
    pnl_col = results['pnl_column']
    
    insights = {
        'top_performers': analyze_top_performers(df, pnl_col),
        'correlations': calculate_correlations(df, pnl_col),
        'high_volume': analyze_high_volume_traders(df, pnl_col),
        'performance': analyze_performance_distribution(df, pnl_col),
        'risk_reward': calculate_risk_reward(df, pnl_col)
    }
    
    # Print insights
    print("\n" + "=" * 60)
    print("💡 ADDITIONAL INSIGHTS")
    print("=" * 60)
    
    # Top performers
    print(f"\n🏆 TOP 10% PERFORMERS ANALYSIS:")
    print(f"  📊 Total top performers: {insights['top_performers']['count']:,}")
    print(f"  🟢 Active traders in top 10%: {insights['top_performers']['active_ratio']:.1f}%")
    print(f"  💰 Average {pnl_col}: {insights['top_performers']['mean_pnl']:.2f}")
    print(f"  📈 Average daily transactions: {insights['top_performers']['mean_tx']:.2f}")
    print(f"  💵 Average volume: ${insights['top_performers']['mean_volume']:,.0f}")
    
    # Correlations
    print(f"\n📈 CORRELATION ANALYSIS:")
    print(f"  🔄 Trading Frequency ↔ {pnl_col}: {insights['correlations']['tx_pnl']:.4f}")
    print(f"  💰 Volume ↔ {pnl_col}: {insights['correlations']['volume_pnl']:.4f}")
    print(f"  🔄 Trading Frequency ↔ Volume: {insights['correlations']['tx_volume']:.4f}")
    
    # High volume traders
    print(f"\n💰 HIGH VOLUME TRADERS (Top 20% by volume):")
    print(f"  📊 Total high volume traders: {insights['high_volume']['count']:,}")
    print(f"  🟢 Active traders ratio: {insights['high_volume']['active_ratio']:.1f}%")
    print(f"  💰 Average {pnl_col}: {insights['high_volume']['mean_pnl']:.2f}")
    print(f"  💵 Volume threshold: ${insights['high_volume']['volume_threshold']:,.0f}")
    
    # Performance distribution
    print(f"\n📊 PERFORMANCE DISTRIBUTION:")
    print(f"  🟢 Profitable traders: {insights['performance']['profit_count']:,} "
          f"({insights['performance']['profit_ratio']:.1f}%)")
    print(f"     Active ratio in profitable: {insights['performance']['profit_active_ratio']:.1f}%")
    print(f"  🔴 Loss-making traders: {insights['performance']['loss_count']:,} "
          f"({insights['performance']['loss_ratio']:.1f}%)")
    print(f"     Active ratio in losses: {insights['performance']['loss_active_ratio']:.1f}%")
    
    # Risk-reward analysis
    if insights['risk_reward']:
        print(f"\n⚖️  RISK-REWARD ANALYSIS:")
        print(f"  📈 Average profit: {insights['risk_reward']['avg_profit']:.2f}")
        print(f"  📉 Average loss: {insights['risk_reward']['avg_loss']:.2f}")
        print(f"  ⚖️  Risk-Reward ratio: {insights['risk_reward']['risk_reward_ratio']:.2f}")
    
    return insights 
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import pandas as pd
import numpy as np

def setup_plot_style():
    """Setup consistent plot styling."""
    plt.style.use('default')
    sns.set_palette("husl")

def create_distribution_plot(ax, active_traders: pd.DataFrame, 
                           normal_traders: pd.DataFrame, pnl_col: str):
    """Create distribution comparison plot."""
    ax.hist(normal_traders[pnl_col], bins=50, alpha=0.7,
            label=f'Normal Traders (n={len(normal_traders):,})',
            color='skyblue', density=True)
    ax.hist(active_traders[pnl_col], bins=50, alpha=0.7,
            label=f'Active Traders (n={len(active_traders):,})',
            color='orange', density=True)
    
    ax.axvline(normal_traders[pnl_col].mean(), color='blue', linestyle='--',
               label=f'Normal Mean: {normal_traders[pnl_col].mean():.2f}')
    ax.axvline(active_traders[pnl_col].mean(), color='red', linestyle='--',
               label=f'Active Mean: {active_traders[pnl_col].mean():.2f}')
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xlabel(pnl_col)
    ax.set_ylabel('Density')
    ax.set_title('📊 Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_box_plot(ax, active_traders: pd.DataFrame, 
                   normal_traders: pd.DataFrame, pnl_col: str):
    """Create box plot comparison."""
    box_data = [normal_traders[pnl_col], active_traders[pnl_col]]
    box_plot = ax.boxplot(box_data, labels=['Normal\nTraders', 'Active\nTraders'],
                         patch_artist=True, showfliers=False)
    
    box_plot['boxes'][0].set_facecolor('skyblue')
    box_plot['boxes'][1].set_facecolor('orange')
    
    ax.set_ylabel(pnl_col)
    ax.set_title('📦 Box Plot Comparison')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)

def create_scatter_plot(ax, df: pd.DataFrame, pnl_col: str):
    """Create scatter plot of trading frequency vs PnL."""
    scatter = ax.scatter(df['avg_daily_transactions'], df[pnl_col],
                        alpha=0.6, s=30, c=df['total_volume'], cmap='viridis')
    
    threshold = df['avg_daily_transactions'].quantile(0.7)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'Active Threshold ({threshold:.1f} tx/day)')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Average Daily Transactions')
    ax.set_ylabel(pnl_col)
    ax.set_title('🎯 Trading Frequency vs Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return scatter

def create_success_rate_plot(ax, active_stats: Dict, normal_stats: Dict):
    """Create success rate comparison bar plot."""
    categories = ['Normal\nTraders', 'Active\nTraders']
    success_rates = [normal_stats['success_rate'], active_stats['success_rate']]
    
    bars = ax.bar(categories, success_rates, color=['skyblue', 'orange'],
                  alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('🎯 Success Rate Comparison')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

def create_volume_distribution(ax, active_traders: pd.DataFrame, 
                             normal_traders: pd.DataFrame):
    """Create volume distribution plot."""
    ax.hist(np.log10(normal_traders['total_volume']), bins=30, alpha=0.7,
            label='Normal Traders', color='skyblue', density=True)
    ax.hist(np.log10(active_traders['total_volume']), bins=30, alpha=0.7,
            label='Active Traders', color='orange', density=True)
    
    ax.set_xlabel('Log10(Total Volume)')
    ax.set_ylabel('Density')
    ax.set_title('💰 Volume Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_transaction_distribution(ax, active_traders: pd.DataFrame, 
                                  normal_traders: pd.DataFrame):
    """Create transaction count distribution plot."""
    ax.hist(normal_traders['total_transactions'], bins=30, alpha=0.7,
            label='Normal Traders', color='skyblue', density=True)
    ax.hist(active_traders['total_transactions'], bins=30, alpha=0.7,
            label='Active Traders', color='orange', density=True)
    
    ax.set_xlabel('Total Transactions')
    ax.set_ylabel('Density')
    ax.set_title('🔄 Transaction Count Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_statistical_summary(ax, results: Dict):
    """Create statistical summary text box."""
    ax.axis('off')
    
    summary_text = f"""
    📊 STATISTICAL SUMMARY

🧪 Test: {results['test_name']}
📈 P-value: {results['p_value']:.6f}
✅ Significant: {'YES' if results['p_value'] < 0.05 else 'NO'}

🟢 Active Traders ({len(results['active_traders']):,} wallets)
   Mean {results['pnl_column']}: {results['active_stats']['mean_pnl']:.2f}
   Success Rate: {results['active_stats']['success_rate']:.1f}%
   Avg Daily TX: {results['active_stats']['mean_tx']:.2f}

🔵 Normal Traders ({len(results['normal_traders']):,} wallets)
   Mean {results['pnl_column']}: {results['normal_stats']['mean_pnl']:.2f}
   Success Rate: {results['normal_stats']['success_rate']:.1f}%
   Avg Daily TX: {results['normal_stats']['mean_tx']:.2f}

💡 Conclusion:
{'Active traders perform significantly better' if results['p_value'] < 0.05 and results['active_stats']['mean_pnl'] > results['normal_stats']['mean_pnl'] else 'No significant performance difference' if results['p_value'] >= 0.05 else 'Normal traders perform better'}
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

def create_analysis_plots(df: pd.DataFrame, results: Dict):
    """Create comprehensive visualization of analysis results."""
    setup_plot_style()
    
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Create all subplots
    create_distribution_plot(fig.add_subplot(gs[0, :2]), 
                           results['active_traders'], 
                           results['normal_traders'], 
                           results['pnl_column'])
    
    create_box_plot(fig.add_subplot(gs[0, 2]), 
                   results['active_traders'], 
                   results['normal_traders'], 
                   results['pnl_column'])
    
    scatter = create_scatter_plot(fig.add_subplot(gs[1, :2]), df, results['pnl_column'])
    plt.colorbar(scatter, ax=fig.add_subplot(gs[1, :2])).set_label('Total Volume ($)')
    
    create_success_rate_plot(fig.add_subplot(gs[1, 2]), 
                           results['active_stats'], 
                           results['normal_stats'])
    
    create_volume_distribution(fig.add_subplot(gs[2, 0]), 
                             results['active_traders'], 
                             results['normal_traders'])
    
    create_transaction_distribution(fig.add_subplot(gs[2, 1]), 
                                  results['active_traders'], 
                                  results['normal_traders'])
    
    create_statistical_summary(fig.add_subplot(gs[2, 2]), results)
    
    # Add main title
    fig.suptitle(f'🔬 SOLANA TRADING ANALYSIS - {results["test_name"]} (p={results["p_value"]:.4f})',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show() 
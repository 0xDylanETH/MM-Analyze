from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats

def calculate_group_stats(group: pd.DataFrame, pnl_col: str) -> Dict:
    """Calculate statistics for a trader group."""
    stats_dict = {
        'count': len(group),
        'mean_pnl': group[pnl_col].mean(),
        'median_pnl': group[pnl_col].median(),
        'std_pnl': group[pnl_col].std(),
        'mean_tx': group['avg_daily_transactions'].mean(),
        'mean_volume': group['total_volume'].mean(),
    }
    
    # Calculate success rate
    if 'success_rate' in group.columns:
        stats_dict['success_rate'] = group['success_rate'].mean() * 100
    else:
        stats_dict['success_rate'] = (group[pnl_col] > 0).mean() * 100
    
    return stats_dict

def perform_hypothesis_test(active_traders: pd.DataFrame, 
                          normal_traders: pd.DataFrame,
                          pnl_col: str) -> Tuple[str, float, float]:
    """Perform hypothesis test between active and normal traders."""
    sample_size = min(100, len(active_traders), len(normal_traders))
    
    if sample_size >= 30:
        try:
            # Test normality
            _, p_active = stats.shapiro(active_traders[pnl_col].sample(min(50, len(active_traders))))
            _, p_normal = stats.shapiro(normal_traders[pnl_col].sample(min(50, len(normal_traders))))
            
            if p_active > 0.05 and p_normal > 0.05:
                # Use t-test for normal distributions
                t_stat, p_value = stats.ttest_ind(active_traders[pnl_col], normal_traders[pnl_col])
                test_name = "Independent t-test"
            else:
                # Use Mann-Whitney U for non-normal distributions
                t_stat, p_value = stats.mannwhitneyu(active_traders[pnl_col], 
                                                   normal_traders[pnl_col], 
                                                   alternative='two-sided')
                test_name = "Mann-Whitney U test"
        except:
            # Fallback to Mann-Whitney U
            t_stat, p_value = stats.mannwhitneyu(active_traders[pnl_col], 
                                               normal_traders[pnl_col], 
                                               alternative='two-sided')
            test_name = "Mann-Whitney U test"
    else:
        # Small sample - use Mann-Whitney U
        t_stat, p_value = stats.mannwhitneyu(active_traders[pnl_col], 
                                           normal_traders[pnl_col], 
                                           alternative='two-sided')
        test_name = "Mann-Whitney U test"
    
    return test_name, t_stat, p_value

def calculate_effect_size(active_stats: Dict, normal_stats: Dict,
                         active_count: int, normal_count: int) -> Optional[float]:
    """Calculate Cohen's d effect size."""
    try:
        pooled_std = np.sqrt(((active_count - 1) * active_stats['std_pnl']**2 +
                            (normal_count - 1) * normal_stats['std_pnl']**2) /
                           (active_count + normal_count - 2))
        cohens_d = (active_stats['mean_pnl'] - normal_stats['mean_pnl']) / pooled_std
        return cohens_d
    except:
        return None

def analyze_trading_patterns(df: pd.DataFrame) -> Optional[Dict]:
    """Main function to perform statistical analysis."""
    # Determine PnL column to use
    pnl_col = 'pnl_percentage' if 'pnl_percentage' in df.columns else 'total_pnl'
    
    # Split traders into groups
    active_traders, normal_traders = df[df['avg_daily_transactions'] >= df['avg_daily_transactions'].quantile(0.7)], \
                                   df[df['avg_daily_transactions'] < df['avg_daily_transactions'].quantile(0.7)]
    
    if len(active_traders) < 10 or len(normal_traders) < 10:
        print("⚠️ Warning: One group has too few samples")
        return None
    
    # Calculate statistics
    active_stats = calculate_group_stats(active_traders, pnl_col)
    normal_stats = calculate_group_stats(normal_traders, pnl_col)
    
    # Perform hypothesis test
    test_name, t_stat, p_value = perform_hypothesis_test(active_traders, normal_traders, pnl_col)
    
    # Calculate effect size
    effect_size = calculate_effect_size(active_stats, normal_stats, 
                                      len(active_traders), len(normal_traders))
    
    return {
        'active_traders': active_traders,
        'normal_traders': normal_traders,
        'active_stats': active_stats,
        'normal_stats': normal_stats,
        'test_name': test_name,
        't_stat': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'pnl_column': pnl_col
    } 
# === REAL DATA SMA STRATEGY ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import sys
import os

# Add parent directory to path to import crypto_data_loader
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from crypto_data_loader import DataHandler, load_multi_symbol_data, get_coingecko_metadata, merge_metadata
    print("✅ Successfully imported crypto_data_loader")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running from the correct directory")
    sys.exit(1)

warnings.filterwarnings('ignore')

print("🚀 BTC SMA-Based Dynamic Spread Strategy")
print("📊 Using Real Data from Binance API")
print("=" * 60)

# === CONFIGURATION ===
CONFIG = {
    # Data Parameters
    'SYMBOL': 'BTCUSDT',
    'INTERVAL': '5m',   # Đổi sang nến 5 phút
    'DAYS_BACK': 7,     # Lấy 7 ngày gần nhất để tránh quá nhiều dữ liệu
    
    # Strategy Parameters
    'LEVERAGE': 5,
    'POSITION_SIZE': 0.00035,  # BTC per trade
    'TAKE_PROFIT': 0.0008,     # 0.08%
    'STOP_LOSS': 0.0003,       # 0.03%
    'STARTING_BALANCE': 1000,  # USDT

    # SMA Parameters
    'SMA_PERIODS': [5, 15, 30],
    'TREND_THRESHOLD': 0.0005,  # 0.05%

    # Trading Fees Configuration
    'TAKER_FEE': 0.00045,      # 0.045% taker fee
    'MAKER_FEE': 0.00015,      # 0.015% maker fee

    # Funding Rate Configuration (estimated)
    'INTEREST_RATE': 0.0001,   # 0.01% per funding interval
    'FUNDING_INTERVAL_HOURS': 1,  # Funding paid every hour
    'PREMIUM_VOLATILITY': 0.0002,  # Simulated premium volatility
}

# Display configuration
print("📋 Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

# === REAL DATA LOADER CLASS ===
class RealDataLoader:
    """Load real BTC price data from Binance API"""
    
    def __init__(self, config):
        self.config = config
        self.data_handler = DataHandler()  # No API keys needed for public data
    
    def load_btc_data(self):
        """Load real BTC price data from Binance"""
        print(f"\n📊 Loading real BTC data from Binance...")
        print(f"   Symbol: {self.config['SYMBOL']}")
        print(f"   Interval: {self.config['INTERVAL']}")
        print(f"   Days back: {self.config['DAYS_BACK']}")
        
        # Calculate start date
        start_str = f"{self.config['DAYS_BACK']} days ago UTC"
        
        try:
            # Load BTC data
            df = self.data_handler.get_historical_klines(
                symbol=self.config['SYMBOL'],
                interval=self.config['INTERVAL'],
                start_str=start_str
            )
            
            # Reset index to get timestamp as column
            df = df.reset_index()
            
            # Rename columns to match our expected format
            df = df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add additional columns for compatibility
            df['candle'] = range(1, len(df) + 1)
            df['time'] = df['timestamp'].dt.strftime('%H:%M')
            df['change'] = df['close'] - df['open']
            df['change_pct'] = (df['change'] / df['open']) * 100
            
            # Simulate funding rate data (since Binance doesn't provide this in historical klines)
            df['funding_rate'] = np.random.normal(0.0001, 0.0002, len(df))
            df['funding_rate_interval'] = df['funding_rate'].apply(
                lambda x: x if np.random.random() < 0.25 else 0  # 25% chance of funding time
            )
            df['premium_rate'] = np.random.normal(0, 0.0002, len(df))
            
            print(f"✅ Loaded {len(df)} candles")
            print(f"📈 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"💰 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            print(f"💹 Total movement: {((df.iloc[-1]['close'] / df.iloc[0]['open']) - 1) * 100:.2f}%")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

# === SMA CALCULATOR CLASS ===
class SMACalculator:
    """Calculate Simple Moving Averages and classify trends"""

    @staticmethod
    def calculate_sma(df, periods):
        """Calculate SMAs for given periods"""
        print("\n📊 Calculating SMAs...")

        for period in periods:
            df[f'sma{period}'] = df['close'].rolling(window=period).mean().round(2)

        # Count valid SMA data
        for period in periods:
            valid_count = df[f'sma{period}'].notna().sum()
            print(f"   SMA{period}: {valid_count} valid values")

        return df

    @staticmethod
    def classify_trend(row, threshold=0.0005):
        """Classify trend based on SMA relationships"""
        sma5, sma15, sma30 = row['sma5'], row['sma15'], row['sma30']

        if pd.isna(sma5) or pd.isna(sma15) or pd.isna(sma30):
            return 'INSUFFICIENT_DATA'

        # Calculate gaps
        gap5_15 = abs(sma5 - sma15) / sma15
        gap15_30 = abs(sma15 - sma30) / sma30

        # Classify trend
        if sma5 > sma15 > sma30:
            if gap5_15 > threshold and gap15_30 > threshold:
                return 'STRONG_UPTREND'
            else:
                return 'WEAK_UPTREND'
        elif sma5 < sma15 < sma30:
            if gap5_15 > threshold and gap15_30 > threshold:
                return 'STRONG_DOWNTREND'
            else:
                return 'WEAK_DOWNTREND'
        else:
            return 'SIDEWAYS'

    @staticmethod
    def add_trend_classification(df, threshold=0.0005):
        """Add trend classification to dataframe"""
        print("\n🎯 Classifying trends...")

        df['trend'] = df.apply(lambda row: SMACalculator.classify_trend(row, threshold), axis=1)

        # Display trend distribution
        trend_counts = df['trend'].value_counts()
        print("📈 Trend Distribution:")
        for trend, count in trend_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {trend}: {count} candles ({percentage:.1f}%)")

        return df

# === SPREAD STRATEGY CLASS ===
class SpreadStrategy:
    """Implement dynamic spread strategy based on trends"""

    SPREAD_CONFIG = {
        'STRONG_UPTREND': {'bid': 0.0005, 'ask': 0.002},      # Favor Long
        'WEAK_UPTREND': {'bid': 0.0007, 'ask': 0.0015},       # Slight Long bias
        'SIDEWAYS': {'bid': 0.001, 'ask': 0.001},             # Balanced
        'WEAK_DOWNTREND': {'bid': 0.0015, 'ask': 0.0007},     # Slight Short bias
        'STRONG_DOWNTREND': {'bid': 0.002, 'ask': 0.0005},    # Favor Short
        'INSUFFICIENT_DATA': {'bid': 0.001, 'ask': 0.001}     # Default
    }

    @classmethod
    def apply_dynamic_spreads(cls, df):
        """Apply dynamic spreads based on trend classification"""
        print("\n💰 Applying dynamic spread strategy...")

        # Add spread configuration
        df['bid_spread'] = df['trend'].map(lambda t: cls.SPREAD_CONFIG[t]['bid'])
        df['ask_spread'] = df['trend'].map(lambda t: cls.SPREAD_CONFIG[t]['ask'])

        # Calculate bid/ask prices
        df['bid_price'] = (df['close'] * (1 - df['bid_spread'])).round(2)
        df['ask_price'] = (df['close'] * (1 + df['ask_spread'])).round(2)
        df['spread_pct'] = ((df['ask_price'] - df['bid_price']) / df['close'] * 100).round(3)

        print("✅ Dynamic spreads applied")

        # Display spread strategy summary
        print("\n📊 Spread Strategy Configuration:")
        for trend, config in cls.SPREAD_CONFIG.items():
            bid_pct = config['bid'] * 100
            ask_pct = config['ask'] * 100
            print(f"   {trend}: Bid={bid_pct:.2f}%, Ask={ask_pct:.2f}%")

        return df

# === TRADING SIMULATOR CLASS ===
class TradingSimulator:
    """Simulate trading based on dynamic spread strategy with fees and funding"""

    def __init__(self, config):
        self.config = config
        self.balance = config['STARTING_BALANCE']
        self.trades = []
        self.active_position = None
        self.trade_id = 0
        # Track fees and funding
        self.total_trading_fees = 0
        self.total_funding_fees = 0

    def calculate_trading_fee(self, trade_value, is_maker=False):
        """Calculate trading fee based on maker/taker"""
        fee_rate = self.config['MAKER_FEE'] if is_maker else self.config['TAKER_FEE']
        return trade_value * fee_rate

    def calculate_funding_fee(self, position, funding_rate_interval):
        """Calculate funding fee for a position"""
        position_value = position['size'] * position['entry_price'] * self.config['LEVERAGE']

        # Funding fee direction depends on position side and funding rate
        if position['side'] == 'LONG':
            # Long positions pay when funding rate is positive
            funding_fee = position_value * funding_rate_interval
        else:  # SHORT
            # Short positions receive when funding rate is positive (so negative fee for them)
            funding_fee = -position_value * funding_rate_interval

        return funding_fee

    def simulate_order_fill(self, candle):
        """Simulate order fill probability with maker/taker determination"""
        # Calculate volatility factor
        price_movement = candle['close'] - candle['open']
        volatility_factor = abs(price_movement) / candle['open']

        # Base fill probability scaled by volatility
        base_prob = 0.35 + (volatility_factor * 600)
        fill_probability = min(0.75, base_prob)

        if np.random.random() < fill_probability:
            # Determine if order is maker or taker (simplified)
            # Assume 30% of orders are maker orders (providing liquidity)
            is_maker = np.random.random() < 0.3

            # Determine direction based on trend bias
            trend = candle['trend']
            if 'UPTREND' in trend:
                trend_bias = 0.65  # Favor Long
            elif 'DOWNTREND' in trend:
                trend_bias = 0.35  # Favor Short
            else:
                trend_bias = 0.5   # Balanced

            if np.random.random() < trend_bias:
                # For maker orders, we get better price (closer to mid)
                if is_maker:
                    # Maker gets price between bid and mid
                    fill_price = candle['bid_price'] + (candle['close'] - candle['bid_price']) * 0.3
                else:
                    fill_price = candle['bid_price']
                return {'side': 'LONG', 'price': fill_price, 'is_maker': is_maker}
            else:
                # For maker orders, we get better price (closer to mid)
                if is_maker:
                    # Maker gets price between mid and ask
                    fill_price = candle['ask_price'] - (candle['ask_price'] - candle['close']) * 0.3
                else:
                    fill_price = candle['ask_price']
                return {'side': 'SHORT', 'price': fill_price, 'is_maker': is_maker}

        return None

    def calculate_pnl(self, position, current_price):
        """Calculate position PnL"""
        if position['side'] == 'LONG':
            price_diff = current_price - position['entry_price']
        else:  # SHORT
            price_diff = position['entry_price'] - current_price

        unrealized_pnl = (price_diff / position['entry_price']) * \
                        position['size'] * position['entry_price'] * self.config['LEVERAGE']

        pnl_percentage = (price_diff / position['entry_price']) * self.config['LEVERAGE']

        return {'unrealized_pnl': unrealized_pnl, 'pnl_percentage': pnl_percentage}

    def run_simulation(self, df):
        """Run complete trading simulation with fees and funding"""
        print("\n🤖 Running trading simulation...")
        print(f"💰 Starting balance: ${self.balance:.2f} USDT")
        print(f"⚡ Leverage: {self.config['LEVERAGE']}x")
        print(f"📦 Position size: {self.config['POSITION_SIZE']} BTC")
        print(f"💸 Taker fee: {self.config['TAKER_FEE']*100:.3f}%")
        print(f"💸 Maker fee: {self.config['MAKER_FEE']*100:.3f}%")

        # Start from candle 30 (when SMA30 is available)
        for i, row in df.iterrows():
            if i < 29:  # Skip first 29 candles
                continue

            candle = row.to_dict()

            if candle['trend'] == 'INSUFFICIENT_DATA':
                continue

            # === FUNDING FEE CALCULATION ===
            if self.active_position and candle['funding_rate_interval'] != 0:
                funding_fee = self.calculate_funding_fee(
                    self.active_position,
                    candle['funding_rate_interval']
                )

                # Apply funding fee to balance
                self.balance -= funding_fee
                self.total_funding_fees += funding_fee

                # Store funding fee in position for tracking
                if 'funding_fees' not in self.active_position:
                    self.active_position['funding_fees'] = 0
                self.active_position['funding_fees'] += funding_fee

            # === EXIT LOGIC ===
            if self.active_position:
                pnl = self.calculate_pnl(self.active_position, candle['close'])

                should_exit = False
                exit_reason = ''

                # Check take profit
                if pnl['pnl_percentage'] >= self.config['TAKE_PROFIT']:
                    should_exit = True
                    exit_reason = 'TAKE_PROFIT'
                # Check stop loss
                elif pnl['pnl_percentage'] <= -self.config['STOP_LOSS']:
                    should_exit = True
                    exit_reason = 'STOP_LOSS'

                if should_exit:
                    # Calculate exit trading fee
                    exit_trade_value = self.active_position['size'] * candle['close']
                    exit_fee = self.calculate_trading_fee(exit_trade_value, is_maker=False)  # Assume taker on exit

                    # Close position
                    self.active_position['exit_price'] = candle['close']
                    self.active_position['exit_time'] = candle['timestamp']
                    self.active_position['exit_reason'] = exit_reason
                    self.active_position['exit_fee'] = exit_fee
                    self.active_position['final_pnl'] = pnl['unrealized_pnl']
                    self.active_position['pnl_percentage'] = pnl['pnl_percentage']

                    # Calculate net P&L after all fees
                    total_fees = self.active_position['entry_fee'] + exit_fee + self.active_position.get('funding_fees', 0)
                    self.active_position['total_fees'] = total_fees
                    self.active_position['net_pnl'] = pnl['unrealized_pnl'] - exit_fee  # Entry fee and funding already deducted

                    self.balance += pnl['unrealized_pnl'] - exit_fee
                    self.total_trading_fees += exit_fee

                    self.trades.append(self.active_position.copy())
                    self.active_position = None

            # === ENTRY LOGIC ===
            if not self.active_position:
                fill = self.simulate_order_fill(candle)

                if fill:
                    # Calculate entry trading fee
                    entry_trade_value = self.config['POSITION_SIZE'] * fill['price']
                    entry_fee = self.calculate_trading_fee(entry_trade_value, fill['is_maker'])

                    # Deduct entry fee from balance
                    self.balance -= entry_fee
                    self.total_trading_fees += entry_fee

                    self.trade_id += 1
                    self.active_position = {
                        'id': self.trade_id,
                        'side': fill['side'],
                        'entry_price': fill['price'],
                        'entry_time': candle['timestamp'],
                        'size': self.config['POSITION_SIZE'],
                        'trend_at_entry': candle['trend'],
                        'candle_number': candle['candle'],
                        'is_maker_entry': fill['is_maker'],
                        'entry_fee': entry_fee,
                        'funding_fees': 0
                    }

        # Close any remaining position
        if self.active_position:
            last_candle = df.iloc[-1].to_dict()
            pnl = self.calculate_pnl(self.active_position, last_candle['close'])

            # Calculate exit fee
            exit_trade_value = self.active_position['size'] * last_candle['close']
            exit_fee = self.calculate_trading_fee(exit_trade_value, is_maker=False)

            self.active_position['exit_price'] = last_candle['close']
            self.active_position['exit_time'] = last_candle['timestamp']
            self.active_position['exit_reason'] = 'END_OF_DAY'
            self.active_position['exit_fee'] = exit_fee
            self.active_position['final_pnl'] = pnl['unrealized_pnl']
            self.active_position['pnl_percentage'] = pnl['pnl_percentage']

            # Calculate net P&L after all fees
            total_fees = self.active_position['entry_fee'] + exit_fee + self.active_position.get('funding_fees', 0)
            self.active_position['total_fees'] = total_fees
            self.active_position['net_pnl'] = pnl['unrealized_pnl'] - exit_fee

            self.balance += pnl['unrealized_pnl'] - exit_fee
            self.total_trading_fees += exit_fee

            self.trades.append(self.active_position.copy())

        print(f"✅ Simulation complete: {len(self.trades)} trades executed")
        print(f"💸 Total trading fees: ${self.total_trading_fees:.2f}")
        print(f"💰 Total funding fees: ${self.total_funding_fees:.2f}")

        return self.trades

# === PERFORMANCE ANALYZER CLASS ===
class PerformanceAnalyzer:
    """Analyze trading performance and generate metrics"""

    @staticmethod
    def calculate_metrics(trades, starting_balance, final_balance, total_trading_fees, total_funding_fees):
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE METRICS ANALYSIS (Including Fees)")
        print("=" * 60)

        if not trades:
            print("❌ No trades to analyze")
            return {}

        # Basic metrics
        total_trades = len(trades)
        total_pnl = final_balance - starting_balance
        return_pct = (total_pnl / starting_balance) * 100

        # Calculate gross P&L (before fees)
        gross_pnl = sum(trade['final_pnl'] for trade in trades)
        total_fees = total_trading_fees + total_funding_fees

        # Win/Loss analysis (using net P&L)
        winners = [t for t in trades if t.get('net_pnl', t['final_pnl']) > 0]
        losers = [t for t in trades if t.get('net_pnl', t['final_pnl']) <= 0]

        win_rate = (len(winners) / total_trades) * 100 if total_trades > 0 else 0
        avg_win = sum(t.get('net_pnl', t['final_pnl']) for t in winners) / len(winners) if winners else 0
        avg_loss = sum(t.get('net_pnl', t['final_pnl']) for t in losers) / len(losers) if losers else 0

        # Profit factor (using net P&L)
        gross_profit = sum(t.get('net_pnl', t['final_pnl']) for t in winners)
        gross_loss = abs(sum(t.get('net_pnl', t['final_pnl']) for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Fee analysis
        avg_trading_fee_per_trade = total_trading_fees / total_trades if total_trades > 0 else 0
        avg_funding_fee_per_trade = total_funding_fees / total_trades if total_trades > 0 else 0

        metrics = {
            'total_trades': total_trades,
            'final_balance': final_balance,
            'total_pnl': total_pnl,
            'gross_pnl': gross_pnl,
            'total_fees': total_fees,
            'total_trading_fees': total_trading_fees,
            'total_funding_fees': total_funding_fees,
            'return_pct': return_pct,
            'win_rate': win_rate,
            'winners': len(winners),
            'losers': len(losers),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trading_fee_per_trade': avg_trading_fee_per_trade,
            'avg_funding_fee_per_trade': avg_funding_fee_per_trade
        }

        # Display metrics
        print("🎯 KEY PERFORMANCE METRICS:")
        print(f"📈 Total Trades: {total_trades}")
        print(f"💰 Final Balance: ${final_balance:.2f} USDT")
        print(f"📊 Net P&L: {'+' if total_pnl >= 0 else ''}${total_pnl:.2f} USDT")
        print(f"📊 Gross P&L: {'+' if gross_pnl >= 0 else ''}${gross_pnl:.2f} USDT")
        print(f"💸 Total Fees: ${total_fees:.2f} USDT")
        print(f"   - Trading Fees: ${total_trading_fees:.2f} USDT")
        print(f"   - Funding Fees: ${total_funding_fees:.2f} USDT")
        print(f"📈 Net Return: {'+' if return_pct >= 0 else ''}{return_pct:.2f}%")
        print(f"🏆 Win Rate: {win_rate:.1f}% ({len(winners)} wins, {len(losers)} losses)")
        print(f"💚 Average Win: ${avg_win:.2f} USDT")
        print(f"🔴 Average Loss: ${avg_loss:.2f} USDT")
        print(f"⚖️ Profit Factor: {profit_factor:.2f}")
        print(f"💸 Avg Trading Fee/Trade: ${avg_trading_fee_per_trade:.2f} USDT")
        print(f"💰 Avg Funding Fee/Trade: ${avg_funding_fee_per_trade:.2f} USDT")

        return metrics

    @staticmethod
    def analyze_by_trend(trades):
        """Analyze performance by trend type"""
        print("\n📈 PERFORMANCE BY TREND TYPE:")
        print("-" * 50)

        # Group trades by trend
        trend_groups = {}
        for trade in trades:
            trend = trade['trend_at_entry']
            if trend not in trend_groups:
                trend_groups[trend] = []
            trend_groups[trend].append(trade)

        # Analyze each trend
        for trend, trend_trades in trend_groups.items():
            total_trades = len(trend_trades)
            total_pnl = sum(t.get('net_pnl', t['final_pnl']) for t in trend_trades)
            winners = [t for t in trend_trades if t.get('net_pnl', t['final_pnl']) > 0]
            win_rate = (len(winners) / total_trades) * 100

            print(f"{trend}:")
            print(f"   Trades: {total_trades} | Win Rate: {win_rate:.1f}% | Total P&L: ${total_pnl:.2f}")

    @staticmethod
    def display_sample_trades(trades, n=5):
        """Display sample trades for inspection"""
        print(f"\n📋 SAMPLE TRADES (Last {min(n, len(trades))}):")
        print("-" * 80)
        
        for i, trade in enumerate(trades[-n:]):
            print(f"Trade {trade['id']}:")
            print(f"   Side: {trade['side']} | Entry: ${trade['entry_price']:.2f} | Exit: ${trade['exit_price']:.2f}")
            print(f"   P&L: ${trade.get('net_pnl', trade['final_pnl']):.2f} | Trend: {trade['trend_at_entry']}")
            print(f"   Reason: {trade['exit_reason']} | Fees: ${trade.get('total_fees', 0):.2f}")
            print()

# === MAIN EXECUTION ===
def main():
    """Main execution function"""
    print("\n🚀 STARTING REAL DATA SMA STRATEGY SIMULATION")
    print("=" * 60)
    
    # Step 1: Load real data
    data_loader = RealDataLoader(CONFIG)
    df = data_loader.load_btc_data()
    
    if df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Step 2: Calculate SMAs
    df = SMACalculator.calculate_sma(df, CONFIG['SMA_PERIODS'])
    
    # Step 3: Classify trends
    df = SMACalculator.add_trend_classification(df, CONFIG['TREND_THRESHOLD'])
    
    # Step 4: Apply dynamic spreads
    df = SpreadStrategy.apply_dynamic_spreads(df)
    
    # Step 5: Run trading simulation
    simulator = TradingSimulator(CONFIG)
    trades = simulator.run_simulation(df)
    
    # Step 6: Analyze performance
    if trades:
        metrics = PerformanceAnalyzer.calculate_metrics(
            trades, 
            CONFIG['STARTING_BALANCE'], 
            simulator.balance,
            simulator.total_trading_fees,
            simulator.total_funding_fees
        )
        
        PerformanceAnalyzer.analyze_by_trend(trades)
        PerformanceAnalyzer.display_sample_trades(trades)
        
        # Step 7: Visualization
        print("\n📊 Creating visualizations...")
        create_visualizations(df, trades, metrics)
    else:
        print("❌ No trades executed. Cannot analyze performance.")

def create_visualizations(df, trades, metrics):
    """Create comprehensive visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('BTC SMA Strategy - Real Data Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Price and SMAs
    axes[0, 0].plot(df['timestamp'], df['close'], label='BTC Price', alpha=0.7)
    for period in CONFIG['SMA_PERIODS']:
        axes[0, 0].plot(df['timestamp'], df[f'sma{period}'], label=f'SMA{period}', alpha=0.8)
    axes[0, 0].set_title('BTC Price with SMAs')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Trend distribution
    trend_counts = df['trend'].value_counts()
    axes[0, 1].pie(trend_counts.values, labels=trend_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Trend Distribution')
    
    # Plot 3: P&L over time (if trades exist)
    if trades:
        trade_pnls = [t.get('net_pnl', t['final_pnl']) for t in trades]
        cumulative_pnl = np.cumsum(trade_pnls)
        axes[1, 0].plot(range(len(cumulative_pnl)), cumulative_pnl, marker='o')
        axes[1, 0].set_title('Cumulative P&L')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Cumulative P&L ($)')
    
    # Plot 4: Performance metrics
    if trades:
        win_rate = metrics['win_rate']
        profit_factor = min(metrics['profit_factor'], 5)  # Cap for visualization
        return_pct = metrics['return_pct']
        
        bars = ['Win Rate (%)', 'Profit Factor', 'Return (%)']
        values = [win_rate, profit_factor, return_pct]
        colors = ['green' if v > 0 else 'red' for v in values]
        
        axes[1, 1].bar(bars, values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Key Performance Metrics')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Run the simulation
if __name__ == "__main__":
    main() 
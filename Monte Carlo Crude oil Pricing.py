import numpy as np
import pandas as pd
import yfinance as yf
import ssl
from typing import Tuple, Optional
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')



def add_profitability_analysis(final_prices: np.ndarray, strike_price: float, 
                             option_premium: float, paths: np.ndarray = None) -> dict:
    """
    Add comprehensive profitability analysis to existing Monte Carlo results.
    
    Args:
        final_prices: Array of final prices from Monte Carlo simulation
        strike_price: Option strike price
        option_premium: Option price/premium (from Black-Scholes or Monte Carlo)
        paths: Optional - full price paths for additional analysis
        
    Returns:
        Dictionary with profitability metrics
    """
    
    # Basic option calculations
    payoffs = np.maximum(final_prices - strike_price, 0)
    
    # Key profitability metrics
    prob_exercise = np.mean(final_prices > strike_price)  # ITM probability
    breakeven_price = strike_price + option_premium
    prob_profit = np.mean(final_prices > breakeven_price)  # Probability of profit
    
    # Profit/Loss analysis
    net_profits = payoffs - option_premium
    expected_profit = np.mean(net_profits)
    
    # Separate profitable and unprofitable scenarios
    profitable_scenarios = net_profits > 0
    loss_scenarios = net_profits <= 0
    
    avg_profit_when_profitable = np.mean(net_profits[profitable_scenarios]) if np.any(profitable_scenarios) else 0
    avg_loss_when_unprofitable = np.mean(net_profits[loss_scenarios]) if np.any(loss_scenarios) else 0
    
    # Return metrics
    roi_expected = expected_profit / option_premium if option_premium > 0 else 0
    
    # Risk metrics
    max_loss = -option_premium
    var_95 = np.percentile(net_profits, 5)  # 95% Value at Risk
    
    # Return probability analysis
    prob_positive_return = prob_profit
    prob_50pct_return = np.mean(net_profits > 0.5 * option_premium)
    prob_100pct_return = np.mean(net_profits > option_premium)  # Double your money
    prob_200pct_return = np.mean(net_profits > 2 * option_premium)  # Triple your money
    
    # Conditional probability
    if prob_exercise > 0:
        prob_profit_given_itm = np.mean(net_profits[final_prices > strike_price] > 0)
    else:
        prob_profit_given_itm = 0
    
    results = {
        # Basic metrics
        'prob_exercise': prob_exercise,
        'prob_profit': prob_profit,
        'breakeven_price': breakeven_price,
        
        # Profit/Loss metrics
        'expected_profit': expected_profit,
        'avg_profit_when_profitable': avg_profit_when_profitable,
        'avg_loss_when_unprofitable': avg_loss_when_unprofitable,
        'prob_profit_given_itm': prob_profit_given_itm,
        
        # Return metrics
        'roi_expected': roi_expected,
        'prob_positive_return': prob_positive_return,
        'prob_50pct_return': prob_50pct_return,
        'prob_100pct_return': prob_100pct_return,
        'prob_200pct_return': prob_200pct_return,
        
        # Risk metrics
        'max_loss': max_loss,
        'var_95': var_95,
        
        # Data for further analysis
        'net_profits': net_profits,
        'payoffs': payoffs
    }
    
    # Print results
    print_profitability_results(results, option_premium)
    
    return results

def print_profitability_results(results: dict, option_premium: float) -> None:
    """Print formatted profitability analysis results."""
    
    print(f"\n{'='*60}")
    print("PROFITABILITY ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\n📊 BASIC OPTION METRICS:")
    print(f"Option Premium (Cost):              ${option_premium:.4f}")
    print(f"Breakeven Price:                    ${results['breakeven_price']:.2f}")
    
    print(f"\n🎯 PROBABILITY ANALYSIS:")
    print(f"Probability of Exercise (ITM):      {results['prob_exercise']:.4f} ({results['prob_exercise']*100:.2f}%)")
    print(f"Probability of Profit:              {results['prob_profit']:.4f} ({results['prob_profit']*100:.2f}%)")
    print(f"Prob. of Profit (given ITM):        {results['prob_profit_given_itm']:.4f} ({results['prob_profit_given_itm']*100:.2f}%)")
    
    print(f"\n💰 PROFIT/LOSS ANALYSIS:")
    print(f"Expected Profit/Loss:               ${results['expected_profit']:.4f}")
    print(f"Expected ROI:                       {results['roi_expected']*100:.2f}%")
    print(f"Average Profit (when profitable):   ${results['avg_profit_when_profitable']:.4f}")
    print(f"Average Loss (when unprofitable):   ${results['avg_loss_when_unprofitable']:.4f}")
    
    print(f"\n⚠️  RISK METRICS:")
    print(f"Maximum Possible Loss:              ${results['max_loss']:.4f}")
    print(f"Value at Risk (95%):                ${results['var_95']:.4f}")
    
    print(f"\n📈 RETURN PROBABILITIES:")
    print(f"Probability of Positive Return:     {results['prob_positive_return']*100:.2f}%")
    print(f"Probability of 50%+ Return:         {results['prob_50pct_return']*100:.2f}%")
    print(f"Probability of 100%+ Return:        {results['prob_100pct_return']*100:.2f}%")
    print(f"Probability of 200%+ Return:        {results['prob_200pct_return']*100:.2f}%")

def create_profitability_summary_table(results: dict, option_premium: float) -> pd.DataFrame:
    """Create a summary table of key profitability metrics."""
    
    summary_data = {
        'Metric': [
            'Option Premium',
            'Breakeven Price',
            'Probability of Exercise',
            'Probability of Profit',
            'Expected Profit/Loss',
            'Expected ROI',
            'Max Possible Loss',
            'Prob. of 50%+ Return',
            'Prob. of 100%+ Return'
        ],
        'Value': [
            f"${option_premium:.4f}",
            f"${results['breakeven_price']:.2f}",
            f"{results['prob_exercise']*100:.2f}%",
            f"{results['prob_profit']*100:.2f}%",
            f"${results['expected_profit']:.4f}",
            f"{results['roi_expected']*100:.2f}%",
            f"${results['max_loss']:.4f}",
            f"{results['prob_50pct_return']*100:.2f}%",
            f"{results['prob_100pct_return']*100:.2f}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df


    # ===================================================================
    # NEW: STEP 5 - PROFITABILITY ANALYSIS
    # ===================================================================
    print("\nSTEP 5: PROFITABILITY ANALYSIS")
    print("-" * 30)
    
    # Extract final prices from Monte Carlo paths
    final_prices = paths[:, -1]
    
    # Use Black-Scholes price as the option premium (market price proxy)
    option_premium = bs_price
    
    # Perform profitability analysis
    profitability_results = add_profitability_analysis(
        final_prices=final_prices,
        strike_price=STRIKE_PRICE,
        option_premium=option_premium,
        paths=paths
    )
    
    # Create summary table
    summary_table = create_profitability_summary_table(profitability_results, option_premium)
    print(f"\n📋 PROFITABILITY SUMMARY TABLE:")
    print(summary_table.to_string(index=False))
    
    # Enhanced return results
    return {
        'parameters': {
            'spot_price': SPOT_PRICE,
            'strike_price': STRIKE_PRICE,
            'time_to_expiry': TIME_TO_EXPIRY,
            'risk_free_rate': risk_free_rate,
            'volatility': volatility
        },
        'monte_carlo': mc_results,
        'black_scholes': bs_results,
        'binomial': binomial_price,
        'merton': merton_price,
        'profitability': profitability_results,  # NEW: Added profitability results
        'summary_table': summary_table,          # NEW: Added summary table
        'data': {
            'wti_data': wti_data,
            'log_returns': log_returns,
            'paths': paths
        }
    }


    
    # Optional: Create charts
    # create_simple_profitability_chart(results)
class WTIDataFetcher:
    """Handles data fetching and preparation for WTI options pricing."""
    
    def __init__(self) -> None:
        """Initialize data fetcher with SSL fix."""
        self._fix_ssl()
    
    @staticmethod
    def _fix_ssl() -> None:
        """Fix SSL certificate issues for yfinance."""
        ssl._create_default_https_context = ssl._create_unverified_context
    
    def fetch_wti_data(self, period: str = "3y") -> pd.DataFrame:
        """
        Fetch WTI crude oil futures data.
        
        Args:
            period: Time period for data ('1y', '2y', '3y', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker("CL=F")
            data = ticker.history(period=period)
            
        
            
            print(f"✓ Fetched {len(data)} WTI observations")
            print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
            
            return data
            
        except Exception as e:
            print(f"⚠ Error fetching WTI data: {e}")
            return self._generate_synthetic_data()
    
    def fetch_treasury_rate(self) -> float:
        """
        Fetch 6-month Treasury bill rate.
        
        Returns:
            Risk-free rate as decimal (e.g., 0.0525 for 5.25%)
        """
        try:
            # Use 13-week Treasury as proxy for 6-month
            ticker = yf.Ticker("^IRX")
            data = ticker.history(period="5d")
            
            if len(data) > 0:
                rate = data['Close'].iloc[-1] / 100
                print(f"✓ Treasury rate: {rate:.4f} ({rate*100:.2f}%)")
                return rate
            else:
                raise ValueError("No Treasury data")
                
        except Exception as e:
            print(f"⚠ Treasury fetch failed: {e}")
            rate = 0.0525  # Default 5.25%
            print(f"  Using default rate: {rate:.4f}")
            return rate
    
    def calculate_volatility(self, price_data: pd.DataFrame, 
                           period: str = "1y") -> Tuple[float, pd.Series]:
        """
        Calculate volatility from log returns.
        
        Args:
            price_data: DataFrame with 'Close' column
            period: Period for volatility calculation
            
        Returns:
            Tuple of (annualized_volatility, log_returns)
        """
        # Use last 1 year of data for volatility
        if period == "1y":
            cutoff_date = price_data.index[-1] - pd.Timedelta(days=365)
            recent_data = price_data[price_data.index >= cutoff_date]
        else:
            recent_data = price_data
        
        # Calculate log returns
        log_returns = np.log(recent_data['Close'] / recent_data['Close'].shift(1)).dropna()
        
        # Annualize volatility (252 trading days)
        daily_vol = log_returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        print(f"✓ Volatility from {len(log_returns)} observations: {annual_vol:.4f}")
        
        return annual_vol, log_returns
    
    def test_stationarity(self, log_returns: pd.Series) -> dict:
        """
        Test stationarity of log returns using ADF test.
        
        Args:
            log_returns: Series of log returns
            
        Returns:
            Dictionary with test results
        """
        try:
            adf_result = adfuller(log_returns)
            
            results = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
            
            print(f"ADF Test Results:")
            print(f"  Statistic: {results['adf_statistic']:.4f}")
            print(f"  p-value: {results['p_value']:.4f}")
            print(f"  Stationary: {results['is_stationary']}")
            
            return results
            
        except Exception as e:
            print(f"⚠ Stationarity test failed: {e}")
            return {'error': str(e)}
    


## Monte Carlo Engine


"""
Monte Carlo simulation engine for WTI options pricing.
Implements GBM price paths and European call option pricing.
"""

import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class MonteCarloEngine:
    """Monte Carlo simulation engine for options pricing."""
    
    def __init__(self, spot_price: float, strike_price: float, 
                 time_to_expiry: float, risk_free_rate: float, 
                 volatility: float) -> None:
        """
        Initialize Monte Carlo engine.
        
        Args:
            spot_price: Current price of underlying ($70)
            strike_price: Strike price of option ($80)
            time_to_expiry: Time to expiry in years (180/252)
            risk_free_rate: Risk-free rate (from Treasury)
            volatility: Annualized volatility (from data)
        """
        self.S0 = spot_price
        self.K = strike_price
        self.T = time_to_expiry
        self.r = risk_free_rate
        self.sigma = volatility
        
        # Validate inputs
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        if self.S0 <= 0:
            raise ValueError("Spot price must be positive")
        if self.K <= 0:
            raise ValueError("Strike price must be positive")
        if self.T <= 0:
            raise ValueError("Time to expiry must be positive")
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive")
    
    def generate_price_paths(self, num_paths: int = 10000, 
                           num_steps: int = 180) -> np.ndarray:
        """
        Generate price paths using Geometric Brownian Motion.
        
        Formula: S_{t+Δt} = S_t * exp[(r - ½σ²)Δt + σ√Δt*Z]
        
        Args:
            num_paths: Number of simulation paths (10,000)
            num_steps: Number of time steps (180 daily steps)
            
        Returns:
            Array of shape (num_paths, num_steps+1) with price paths
        """
        # Time grid
        dt = self.T / num_steps
        
        # Pre-calculate constants
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        # Generate random numbers
        np.random.seed(42)  # For reproducibility
        Z = np.random.standard_normal((num_paths, num_steps))
        
        # Initialize price paths
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.S0
        
        # Generate paths using vectorized operations
        for t in range(num_steps):
            paths[:, t+1] = paths[:, t] * np.exp(drift + diffusion * Z[:, t])
        
        print(f"✓ Generated {num_paths:,} price paths")
        print(f"  Final price range: ${paths[:, -1].min():.2f} - ${paths[:, -1].max():.2f}")
        print(f"  Mean final price: ${paths[:, -1].mean():.2f}")
        
        return paths
    
    def calculate_option_price(self, paths: np.ndarray) -> Dict[str, float]:
        """
        Calculate European call option price from paths.
        
        Args:
            paths: Array of price paths
            
        Returns:
            Dictionary with pricing results
        """
        # Extract final prices
        final_prices = paths[:, -1]
        
        # Calculate payoffs: max(S_T - K, 0)
        payoffs = np.maximum(final_prices - self.K, 0)
        
        # Calculate statistics
        mean_payoff = np.mean(payoffs)
        std_payoff = np.std(payoffs)
        
        # Discount to present value
        discount_factor = np.exp(-self.r * self.T)
        option_price = discount_factor * mean_payoff
        
        # Calculate confidence interval
        n_sims = len(payoffs)
        std_error = std_payoff * discount_factor / np.sqrt(n_sims)
        ci_95 = 1.96 * std_error
        
        # Additional metrics
        itm_probability = np.mean(final_prices > self.K)
        
        results = {
            'option_price': option_price,
            'std_error': std_error,
            'ci_95': ci_95,
            'lower_bound': option_price - ci_95,
            'upper_bound': option_price + ci_95,
            'itm_probability': itm_probability,
            'mean_payoff': mean_payoff,
            'discount_factor': discount_factor
        }
        
        print(f"✓ Monte Carlo option price: ${option_price:.4f}")
        print(f"  95% CI: [${results['lower_bound']:.4f}, ${results['upper_bound']:.4f}]")
        print(f"  ITM probability: {itm_probability:.4f}")
        
        return results
    
    def run_simulation(self, num_paths: int = 10000) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Run complete Monte Carlo simulation.
        
        Args:
            num_paths: Number of simulation paths
            
        Returns:
            Tuple of (price_paths, option_results)
        """
        print("Running Monte Carlo simulation...")
        
        # Generate price paths
        paths = self.generate_price_paths(num_paths)
        
        # Calculate option price
        results = self.calculate_option_price(paths)
        
        return paths, results


## Step 3: Benchmark Models


"""
Benchmark models for options pricing comparison.
Includes Black-Scholes, Binomial Tree, and Merton Jump-Diffusion.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict

class BenchmarkModels:
    """Collection of benchmark pricing models."""
    
    def __init__(self, spot_price: float, strike_price: float,
                 time_to_expiry: float, risk_free_rate: float,
                 volatility: float) -> None:
        """Initialize benchmark models with same parameters as Monte Carlo."""
        self.S0 = spot_price
        self.K = strike_price
        self.T = time_to_expiry
        self.r = risk_free_rate
        self.sigma = volatility
    
    def black_scholes_call(self) -> Dict[str, float]:
        """
        Black-Scholes European call option pricing.
        
        Returns:
            Dictionary with price and Greeks
        """
        # Calculate d1 and d2
        d1 = (np.log(self.S0/self.K) + (self.r + self.sigma**2/2)*self.T) / (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        
        # Call option price
        call_price = self.S0*norm.cdf(d1) - self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        
        # Calculate Greeks
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T))
        theta = (-self.S0 * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) - 
                 self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(d2)) / 365
        vega = self.S0 * norm.pdf(d1) * np.sqrt(self.T) / 100
        rho = self.K * self.T * np.exp(-self.r*self.T) * norm.cdf(d2) / 100
        
        return {
            'price': call_price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def binomial_tree(self, steps: int = 180) -> float:
        """
        Cox-Ross-Rubinstein binomial tree pricing.
        
        Args:
            steps: Number of time steps
            
        Returns:
            Option price
        """
        # Tree parameters
        dt = self.T / steps
        u = np.exp(self.sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(self.r * dt) - d) / (u - d)  # Risk-neutral probability
        
        # Asset prices at expiration
        asset_prices = np.array([self.S0 * (u**(steps-i)) * (d**i) 
                                for i in range(steps+1)])
        
        # Option values at expiration
        option_values = np.maximum(asset_prices - self.K, 0)
        
        # Backward induction
        for step in range(steps-1, -1, -1):
            for i in range(step+1):
                option_values[i] = (np.exp(-self.r*dt) * 
                                  (p*option_values[i] + (1-p)*option_values[i+1]))
        
        return option_values[0]
    
    def merton_jump_diffusion(self, num_paths: int = 10000,
                             jump_intensity: float = 0.1,
                             jump_mean: float = -0.05,
                             jump_std: float = 0.15) -> float:
        """
        Merton Jump-Diffusion model (stretch goal).
        
        Args:
            num_paths: Number of simulation paths
            jump_intensity: Poisson jump intensity (λ)
            jump_mean: Mean jump size (μ_J)
            jump_std: Jump size standard deviation (σ_J)
            
        Returns:
            Option price
        """
        steps = 180
        dt = self.T / steps
        
        # Adjust drift for jump compensation
        jump_compensator = jump_intensity * (np.exp(jump_mean + 0.5*jump_std**2) - 1)
        adjusted_drift = self.r - 0.5*self.sigma**2 - jump_compensator
        
        np.random.seed(42)
        
        # Initialize paths
        paths = np.zeros((num_paths, steps+1))
        paths[:, 0] = self.S0
        
        for t in range(steps):
            # Brownian motion component
            dW = np.random.normal(0, np.sqrt(dt), num_paths)
            
            # Jump component
            jump_occur = np.random.poisson(jump_intensity * dt, num_paths)
            jump_sizes = np.where(jump_occur > 0,
                                 np.random.normal(jump_mean, jump_std, num_paths), 0)
            
            # Update prices
            paths[:, t+1] = paths[:, t] * np.exp(adjusted_drift*dt + 
                                               self.sigma*dW + jump_sizes)
        
        # Calculate option price
        final_prices = paths[:, -1]
        payoffs = np.maximum(final_prices - self.K, 0)
        option_price = np.exp(-self.r*self.T) * np.mean(payoffs)
        
        return option_price



## Step 4: Main Application

### Create `main.py`

"""
Main application for WTI Monte Carlo options pricing.
Coordinates all components and produces final results.
"""


import numpy as np

def main():
    """Main execution function."""
    print("WTI CRUDE OIL EUROPEAN CALL OPTION PRICING")
    print("=" * 60)
    
    # Project parameters
    SPOT_PRICE = 66.55  # $70/bbl
    STRIKE_PRICE = 68.0  # $80/bbl
    TIME_TO_EXPIRY = 105/252  # 180 trading days ≈ 0.71 years
    
    # Step 1: Data Preparation
    print("\nSTEP 1: DATA PREPARATION")
    print("-" * 30)
    
    fetcher = WTIDataFetcher()
    
    # Fetch WTI data (2-3 years)
    wti_data = fetcher.fetch_wti_data(period="3y")
    
    # Get risk-free rate (6-month T-bill)
    risk_free_rate = fetcher.fetch_treasury_rate()
    
    # Calculate volatility from 1-year returns
    volatility, log_returns = fetcher.calculate_volatility(wti_data, period="1y")
    
    # Test stationarity
    stationarity_results = fetcher.test_stationarity(log_returns)
    
    # Step 2: Monte Carlo Simulation
    print("\nSTEP 2: MONTE CARLO SIMULATION")
    print("-" * 30)
    
    mc_engine = MonteCarloEngine(
        spot_price=SPOT_PRICE,
        strike_price=STRIKE_PRICE,
        time_to_expiry=TIME_TO_EXPIRY,
        risk_free_rate=risk_free_rate,
        volatility=volatility
    )
    
    # Run simulation with 10k paths
    paths, mc_results = mc_engine.run_simulation(num_paths=10000)
    
    # Step 3: Benchmark Models
    print("\nSTEP 3: BENCHMARK MODELS")
    print("-" * 30)
    
    benchmarks = BenchmarkModels(
        spot_price=SPOT_PRICE,
        strike_price=STRIKE_PRICE,
        time_to_expiry=TIME_TO_EXPIRY,
        risk_free_rate=risk_free_rate,
        volatility=volatility
    )
    
    # Calculate benchmarks
    bs_results = benchmarks.black_scholes_call()
    binomial_price = benchmarks.binomial_tree()
    
    try:
        merton_price = benchmarks.merton_jump_diffusion()
        print(f"✓ Merton Jump-Diffusion: ${merton_price:.4f}")
    except:
        merton_price = np.nan
        print("⚠ Merton model failed")
    
    print(f"✓ Black-Scholes: ${bs_results['price']:.4f}")
    print(f"✓ Binomial Tree: ${binomial_price:.4f}")
    
    # Step 4: Results Summary
    print("\nSTEP 4: RESULTS SUMMARY")
    print("-" * 30)
    
    mc_price = mc_results['option_price']
    bs_price = bs_results['price']
    
    print(f"Monte Carlo:     ${mc_price:.4f} ± ${mc_results['ci_95']:.4f}")
    print(f"Black-Scholes:   ${bs_price:.4f}")
    print(f"Binomial Tree:   ${binomial_price:.4f}")
    print(f"Difference (MC-BS): ${mc_price - bs_price:.4f} ({((mc_price - bs_price)/bs_price)*100:.2f}%)")

    print("\nSTEP 5: PROFITABILITY ANALYSIS")
    print("-" * 30)
    
    # Extract final prices from Monte Carlo paths
    final_prices = paths[:, -1]
    
    # Use Black-Scholes price as the option premium (market price proxy)
    option_premium = bs_price
    
    # Perform profitability analysis
    profitability_results = add_profitability_analysis(
        final_prices=final_prices,
        strike_price=STRIKE_PRICE,
        option_premium=option_premium,
        paths=paths
    )
    
    # Create summary table
    summary_table = create_profitability_summary_table(profitability_results, option_premium)
    print(f"\n📋 PROFITABILITY SUMMARY TABLE:")
    print(summary_table.to_string(index=False))
    
    # Enhanced return results
    return {
        'parameters': {
            'spot_price': SPOT_PRICE,
            'strike_price': STRIKE_PRICE,
            'time_to_expiry': TIME_TO_EXPIRY,
            'risk_free_rate': risk_free_rate,
            'volatility': volatility
        },
        'monte_carlo': mc_results,
        'black_scholes': bs_results,
        'binomial': binomial_price,
        'merton': merton_price,
        'profitability': profitability_results,  # NEW: Added profitability results
        'summary_table': summary_table,          # NEW: Added summary table
        'data': {
            'wti_data': wti_data,
            'log_returns': log_returns,
            'paths': paths
        }
    } 
   

    
    # Return results for further analysis
    return {
        'parameters': {
            'spot_price': SPOT_PRICE,
            'strike_price': STRIKE_PRICE,
            'time_to_expiry': TIME_TO_EXPIRY,
            'risk_free_rate': risk_free_rate,
            'volatility': volatility
        },
        'monte_carlo': mc_results,
        'black_scholes': bs_results,
        'binomial': binomial_price,
        'merton': merton_price,
        'data': {
            'wti_data': wti_data,
            'log_returns': log_returns,
            'paths': paths
        }
    }

if __name__ == "__main__":
    results = main()

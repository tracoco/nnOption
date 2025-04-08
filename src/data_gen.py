import numpy as np
import pandas as pd
from scipy.stats import norm
import argparse
from sklearn.model_selection import train_test_split

def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price and Greeks."""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Option Price
    price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    # Greeks
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - 
             r*K*np.exp(-r*T)*norm.cdf(d2))
    vega = S*np.sqrt(T)*norm.pdf(d1)
    
    return price, delta, gamma, theta, vega

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price and Greeks using put-call parity."""
    call_price, call_delta, gamma, call_theta, vega = black_scholes_call(S, K, T, r, sigma)
    
    # Put-call parity
    put_price = call_price - S + K*np.exp(-r*T)
    put_delta = call_delta - 1
    put_theta = call_theta + r*K*np.exp(-r*T)
    
    return put_price, put_delta, gamma, put_theta, vega

def generate_random_parameters(num_samples):
    """Generate random option parameters."""
    S = np.random.uniform(80, 120, num_samples)  # Spot price
    K = np.random.uniform(80, 120, num_samples)  # Strike price
    T = np.random.uniform(0.1, 1.0, num_samples)  # Time to maturity
    r = np.random.uniform(0.01, 0.05, num_samples)  # Risk-free rate
    sigma = np.random.uniform(0.1, 0.5, num_samples)  # Volatility
    option_type = np.random.choice([0, 1], num_samples)  # 0=Put, 1=Call
    
    return S, K, T, r, sigma, option_type

def generate_dataset(num_samples):
    """Generate option pricing dataset."""
    # Generate random parameters
    S, K, T, r, sigma, option_type = generate_random_parameters(num_samples)
    
    # Initialize arrays for prices and Greeks
    prices = np.zeros(num_samples)
    deltas = np.zeros(num_samples)
    gammas = np.zeros(num_samples)
    thetas = np.zeros(num_samples)
    vegas = np.zeros(num_samples)
    
    # Calculate prices and Greeks
    for i in range(num_samples):
        if option_type[i] == 1:  # Call option
            price, delta, gamma, theta, vega = black_scholes_call(
                S[i], K[i], T[i], r[i], sigma[i]
            )
        else:  # Put option
            price, delta, gamma, theta, vega = black_scholes_put(
                S[i], K[i], T[i], r[i], sigma[i]
            )
        
        prices[i] = price
        deltas[i] = delta
        gammas[i] = gamma
        thetas[i] = theta
        vegas[i] = vega
    
    # Create DataFrame
    df = pd.DataFrame({
        'spot_price': S,
        'strike_price': K,
        'time_to_maturity': T,
        'risk_free_rate': r,
        'volatility': sigma,
        'option_type': option_type,
        'price': prices,
        'delta': deltas,
        'gamma': gammas,
        'theta': thetas,
        'vega': vegas
    })
    
    return df

def save_datasets(df, train_ratio=0.8, val_ratio=0.1):
    """Split and save datasets."""
    # First split: training and temporary set
    train_data, temp_data = train_test_split(df, test_size=(1-train_ratio), random_state=42)
    
    # Second split: validation and test sets
    val_ratio_adjusted = val_ratio / (1-train_ratio)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Save datasets
    train_data.to_csv('data/train.csv', index=False)
    val_data.to_csv('data/val.csv', index=False)
    test_data.to_csv('data/test.csv', index=False)
    
    print(f"Datasets saved successfully:")
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

def main():
    parser = argparse.ArgumentParser(description='Generate option pricing dataset')
    parser.add_argument('--num_samples', type=int, default=100000,
                      help='Number of samples to generate')
    args = parser.parse_args()
    
    print("Generating dataset...")
    df = generate_dataset(args.num_samples)
    
    print("Splitting and saving datasets...")
    save_datasets(df)
    
    print("Done!")

if __name__ == "__main__":
    main()
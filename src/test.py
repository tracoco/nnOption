import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from model import OptionPricingModel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import os

def load_test_data():
    """Load test dataset."""
    return pd.read_csv('data/test.csv')

def evaluate_model(model, test_features, test_sequences, test_targets):
    """Evaluate model performance."""
    predictions = model.predict([test_features, test_sequences])
    
    metrics = {}
    target_names = ['price', 'delta', 'gamma', 'theta', 'vega']
    
    for name in target_names:
        true_values = test_targets[name]
        pred_values = predictions[name]
        
        mse = mean_squared_error(true_values, pred_values)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(true_values, pred_values)
        r2 = r2_score(true_values, pred_values)
        
        metrics[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
    
    return metrics, predictions

def plot_predictions(true_values, predictions, target_names):
    """Create prediction vs actual plots."""
    os.makedirs('plots', exist_ok=True)
    
    n_targets = len(target_names)
    fig, axes = plt.subplots(n_targets, 1, figsize=(12, 4*n_targets))
    
    for name, ax in zip(target_names, axes):
        true_val = true_values[name].reshape(-1)
        pred_val = predictions[name].reshape(-1)
        
        # Scatter plot
        ax.scatter(true_val, pred_val, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(true_val.min(), pred_val.min())
        max_val = max(true_val.max(), pred_val.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name} Predictions vs Actual')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('plots/prediction_results.png')
    plt.close()

def plot_error_distribution(true_values, predictions, target_names):
    """Plot error distribution for each target."""
    os.makedirs('plots', exist_ok=True)
    
    n_targets = len(target_names)
    fig, axes = plt.subplots(n_targets, 1, figsize=(12, 4*n_targets))
    
    for name, ax in zip(target_names, axes):
        true_val = true_values[name].reshape(-1)
        pred_val = predictions[name].reshape(-1)
        
        # Calculate percentage errors
        errors = (pred_val - true_val) / true_val * 100
        
        # Plot histogram
        ax.hist(errors, bins=50, alpha=0.75)
        ax.axvline(x=0, color='r', linestyle='--', label='Zero Error')
        
        ax.set_xlabel('Percentage Error (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} Error Distribution')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('plots/error_distribution.png')
    plt.close()

def print_metrics(metrics):
    """Print evaluation metrics."""
    print("\nModel Performance Metrics:")
    print("=" * 50)
    
    for target, metric_values in metrics.items():
        print(f"\n{target.upper()} Metrics:")
        print("-" * 20)
        for metric_name, value in metric_values.items():
            print(f"{metric_name}: {value:.4f}")

def main():
    # Load test data
    print("Loading test data...")
    test_data = load_test_data()
    
    # Initialize model
    print("Loading model...")
    model = OptionPricingModel()
    full_model = model.build_full_model()
    
    # Load trained weights
    checkpoint_path = "checkpoints/model.weights.h5"
    full_model.load_weights(checkpoint_path)
    
    # Preprocess features
    test_features = model.preprocess_data(test_data)
    test_sequences = model.create_sequences(test_features)
    
    # Prepare targets
    target_names = ['price', 'delta', 'gamma', 'theta', 'vega']
    test_targets = {name: test_data[name].values[10-1:].reshape(-1, 1) 
                   for name in target_names}
    
    # Evaluate model
    print("Evaluating model...")
    metrics, predictions = evaluate_model(
        full_model, 
        test_features[10-1:],
        test_sequences,
        test_targets
    )
    
    # Print metrics
    print_metrics(metrics)
    
    # Create visualization plots
    print("Generating plots...")
    plot_predictions(test_targets, predictions, target_names)
    plot_error_distribution(test_targets, predictions, target_names)
    
    print("\nTest completed!")
    print("Results plots saved in 'plots' directory")

if __name__ == "__main__":
    main()
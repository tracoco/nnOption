import numpy as np
import pandas as pd
import tensorflow as tf
from model import OptionPricingModel
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data():
    """Load train and validation datasets."""
    train_data = pd.read_csv('data/train.csv')
    val_data = pd.read_csv('data/val.csv')
    return train_data, val_data

def prepare_targets(data):
    """
    Prepare target variables with normalization to improve training stability.
    
    Normalizes each target variable separately to address scale differences.
    Gamma values are typically much smaller than other Greeks, so special 
    normalization is applied to bring them to a comparable scale.
    """
    # Initialize scalers dictionary if not exists
    if not hasattr(prepare_targets, 'scalers'):
        prepare_targets.scalers = {}
    
    targets = {}
    target_columns = ['price', 'delta', 'gamma', 'theta', 'vega']
    
    for col in target_columns:
        values = data[col].values.reshape(-1, 1)
        
        if col not in prepare_targets.scalers:
            # For gamma, use MinMaxScaler to ensure values aren't too small
            if col == 'gamma':
                scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                scaler = StandardScaler()
                
            scaler.fit(values)
            prepare_targets.scalers[col] = scaler
        
        # Transform the values
        normalized_values = prepare_targets.scalers[col].transform(values)
        targets[col] = normalized_values
    
    return targets

def denormalize_predictions(predictions):
    """
    Denormalize model predictions back to original scale.
    """
    if not hasattr(prepare_targets, 'scalers'):
        # If no scalers defined, return predictions as is
        return predictions
        
    denormalized = {}
    for key, values in predictions.items():
        if key in prepare_targets.scalers:
            denormalized[key] = prepare_targets.scalers[key].inverse_transform(values)
        else:
            denormalized[key] = values
            
    return denormalized

def evaluate_predictions(model, X_test, y_test):
    """
    Evaluate model predictions with focus on gamma accuracy.
    """
    # Get predictions
    predictions = model.predict(X_test)
    
    # Denormalize if necessary
    if hasattr(prepare_targets, 'scalers'):
        predictions = denormalize_predictions(predictions)
        
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Plot actual vs predicted for each output
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    outputs = ['price', 'delta', 'gamma', 'theta', 'vega']
    for i, output in enumerate(outputs):
        if i >= len(axes):
            break
            
        ax = axes[i]
        y_true = y_test[output].flatten()
        y_pred = predictions[output].flatten()
        
        # Calculate metrics
        mse = np.mean((y_true - y_pred)**2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Handle potential division by zero in MAPE calculation
        mask = np.abs(y_true) > 1e-10
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else float('nan')
        
        corr = np.corrcoef(y_true, y_pred)[0, 1] if len(np.unique(y_true)) > 1 else float('nan')
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_title(f'{output.capitalize()}')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        
        # Add metrics to plot
        ax.text(0.05, 0.95, f"MSE: {mse:.6f}\nMAE: {mae:.6f}\nMAPE: {mape:.2f}%\nCorr: {corr:.4f}",
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    plt.tight_layout()
    plt.savefig('results/prediction_evaluation.png')
    plt.close()
    
    # Special focus on gamma
    plt.figure(figsize=(10, 8))
    y_true = y_test['gamma'].flatten()
    y_pred = predictions['gamma'].flatten()
    
    plt.scatter(y_true, y_pred, alpha=0.5, s=15)
    
    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Gamma Prediction Analysis')
    plt.xlabel('Actual Gamma')
    plt.ylabel('Predicted Gamma')
    
    # Add error histogram inset
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(plt.gca(), width="40%", height="30%", loc=4)
    
    errors = y_pred - y_true
    axins.hist(errors, bins=50, alpha=0.75)
    axins.set_title('Error Distribution')
    axins.set_xlabel('Prediction Error')
    
    plt.savefig('results/gamma_analysis.png')
    plt.close()
    
    return predictions

def create_callbacks():
    """Create training callbacks."""
    checkpoint_path = "checkpoints/model.weights.h5"
    os.makedirs('checkpoints', exist_ok=True)
    
    callbacks = [
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=1
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        # Learning rate scheduler
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]
    return callbacks

def plot_training_history(history):
    """Plot training history."""
    os.makedirs('plots', exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    # Plot losses for each output
    output_names = ['price', 'delta', 'gamma', 'theta', 'vega']
    for i, name in enumerate(output_names):
        ax = axes[i]
        ax.plot(history.history[f'{name}_loss'], label=f'Train {name}')
        ax.plot(history.history[f'val_{name}_loss'], label=f'Val {name}')
        ax.set_title(f'{name.capitalize()} Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
    
    # Plot total loss
    ax = axes[-1]
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Val Loss')
    ax.set_title('Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    train_data, val_data = load_data()
    
    # Initialize model
    print("Building model...")
    model = OptionPricingModel()
    
    # Preprocess features
    train_features = model.preprocess_data(train_data)
    val_features = model.preprocess_data(val_data)
    
    # Create sequences for RNN
    train_sequences = model.create_sequences(train_features)
    val_sequences = model.create_sequences(val_features)
    
    # Prepare targets with normalization
    train_targets = prepare_targets(train_data[10-1:])  # Adjust for sequence length
    val_targets = prepare_targets(val_data[10-1:])
    
    # Build and compile model with custom loss weights
    full_model = model.build_full_model(gamma_weight=1.0)  # Increased weight for gamma
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train model
    print("Training model...")
    history = full_model.fit(
        [train_features[10-1:]], # TBN , train_sequences],
        train_targets,
        validation_data=([val_features[10-1:]], val_targets),
        # TBN validation_data=([val_features[10-1:], val_sequences], val_targets),
        epochs=150,  # Increased epochs for better convergence
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    # Evaluate predictions with special focus on gamma
    print("Evaluating predictions...")
    evaluate_predictions(full_model, [val_features[10-1:]], val_targets)
    
    print("Training completed!")
    print("Model checkpoints saved in 'checkpoints' directory")
    print("Training plots saved in 'plots' directory")
    print("Prediction evaluations saved in 'results' directory")

if __name__ == "__main__":
    main()
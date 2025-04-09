import numpy as np
import pandas as pd
import tensorflow as tf
from model import OptionPricingModel
import matplotlib.pyplot as plt
import os

def load_data():
    """Load train and validation datasets."""
    train_data = pd.read_csv('data/train.csv')
    val_data = pd.read_csv('data/val.csv')
    return train_data, val_data

def prepare_targets(data):
    """Prepare target variables."""
    targets = {
        'price': data['price'].values.reshape(-1, 1),
        'delta': data['delta'].values.reshape(-1, 1),
        'gamma': data['gamma'].values.reshape(-1, 1),
        'theta': data['theta'].values.reshape(-1, 1),
        'vega': data['vega'].values.reshape(-1, 1)
    }
    return targets

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
    
    # Prepare targets
    train_targets = prepare_targets(train_data[10-1:])  # Adjust for sequence length
    val_targets = prepare_targets(val_data[10-1:])
    
    # Build and compile model
    full_model = model.build_full_model()
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train model
    print("Training model...")
    history = full_model.fit(
        [train_features[10-1:]], # TBN , train_sequences],  # Market and temporal inputs
        train_targets,
        validation_data=([val_features[10-1:]], val_targets),
        # TBN validation_data=([val_features[10-1:], val_sequences], val_targets),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    print("Training completed!")
    print("Model checkpoints saved in 'checkpoints' directory")
    print("Training plots saved in 'plots' directory")

if __name__ == "__main__":
    main()
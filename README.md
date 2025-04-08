# TensorFlow/Keras Option Pricing Model

This project implements a deep learning approach to option pricing using TensorFlow/Keras. The model combines feedforward neural networks (FNN), recurrent neural networks (RNN), and decoder networks to accurately price options based on the Black-Scholes model.

## Project Structure

```
├── data/              # Generated datasets
│   ├── train.csv     # Training data (80%)
│   ├── val.csv       # Validation data (10%) 
│   └── test.csv      # Test data (10%)
├── src/              # Source code
│   ├── data_gen.py   # Dataset generation script
│   ├── model.py      # Neural network model implementation
│   ├── train.py      # Training script
│   └── test.py       # Testing and visualization script
└── README.md         # Project documentation
```

## Installation

```bash
pip install tensorflow numpy pandas matplotlib scipy scikit-learn
```

## Usage

### 1. Generate Dataset

Generate sample option pricing dataset using Black-Scholes formula:

```bash
python src/data_gen.py --num_samples 100000
```

This will create train/validation/test datasets in the data/ directory with the following features:
- Spot price (S)
- Strike price (K)
- Time to maturity (T)
- Risk-free rate (r)
- Volatility (σ)
- Option type (Call=1, Put=0)

And these targets:
- Option price
- Delta
- Gamma
- Theta
- Vega

### 2. Train Model

Train the neural network model:

```bash
python src/train.py
```

The model architecture consists of:
1. A feedforward neural network (FNN) that processes global market information
2. A recurrent neural network (RNN) that handles temporal dependencies
3. A decoder network that generates final option prices

### 3. Test Model

Test the model and generate visualization plots:

```bash
python src/test.py
```

This will:
1. Load the trained model
2. Evaluate performance on test dataset
3. Generate comparison plots between predicted and actual prices
4. Calculate and display error metrics

## Model Architecture

### Feedforward Neural Network (FNN)
- Processes global market features
- Multiple dense layers with ReLU activation
- Batch normalization for training stability

### Recurrent Neural Network (RNN)
- LSTM layers for temporal dependencies
- Processes market state evolution
- Handles time-series aspects of option pricing

### Decoder Network
- Converts latent space representations to option prices
- Dense layers with appropriate activation functions
- Final layer outputs option price and Greeks

## Training Process

1. Data preprocessing and normalization
2. Mini-batch training with Adam optimizer
3. Early stopping to prevent overfitting
4. Learning rate scheduling for optimal convergence

## Performance Metrics

The model is evaluated on:
- Mean Squared Error (MSE)
- Mean Absolute Percentage Error (MAPE)
- R-squared score
- Greeks accuracy comparison

## References

- Black-Scholes Option Pricing Model
- Deep Learning for Option Pricing literature
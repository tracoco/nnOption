import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from sklearn.preprocessing import StandardScaler
import numpy as np

class OptionPricingModel:
    """
    Neural network model for option pricing and Greeks calculation.
    
    The model uses a hybrid architecture combining:
    - Feedforward Neural Network (FNN): Processes static features
    - Recurrent Neural Network (RNN): Captures temporal dependencies
    - Decoder: Maps latent representation to option prices and Greeks
    
    Mathematical foundation is based on extending the Black-Scholes model:
    dS_t = μS_t dt + σS_t dW_t
    where S_t is the underlying price, μ is drift, σ is volatility, and W_t is Brownian motion.
    """
    def __init__(self, input_dim=6):
        self.input_dim = input_dim
        
    def build_fnn(self):
        """
        Build feedforward neural network for processing global information.
        
        The FNN approximates a function f: R^n → R^m that maps input features to a latent space:
        f(x) = W_L · σ(W_{L-1} · ... σ(W_1 · x + b_1) ... + b_{L-1}) + b_L
        where σ is the ReLU activation function: σ(x) = max(0, x)
        """
        inputs = layers.Input(shape=(self.input_dim,))
        
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        
        return Model(inputs=inputs, outputs=x, name='fnn')
    
    def build_rnn(self, sequence_length=10):
        """
        Build RNN for temporal dependencies.
        
        The LSTM cell computes:
        f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # forget gate
        i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # input gate
        o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # output gate
        c'_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  # candidate cell state
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ c'_t  # cell state update
        h_t = o_t ⊙ tanh(c_t)  # hidden state
        
        Where ⊙ denotes element-wise multiplication.
        """
        inputs = layers.Input(shape=(sequence_length, self.input_dim))
        
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        
        x = layers.LSTM(32)(x)
        x = layers.BatchNormalization()(x)
        
        return Model(inputs=inputs, outputs=x, name='rnn')
    
    def build_decoder(self, latent_dim):
        """
        Build decoder network.
        
        Maps latent space to option price and Greeks:
        - Price: Direct prediction of option value
        - Delta (Δ): ∂V/∂S (price sensitivity to underlying)
        - Gamma (Γ): ∂²V/∂S² (delta sensitivity to underlying)
        - Theta (Θ): ∂V/∂t (price sensitivity to time)
        - Vega: ∂V/∂σ (price sensitivity to volatility)
        """
        inputs = layers.Input(shape=(latent_dim,))
        
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # TBN: Add more layers if needed, for RNN temp disabled 
        # x = layers.Dense(64, activation='relu')(x)
        # x = layers.BatchNormalization()(x)
        
        # Create a dictionary of outputs
        outputs = {}
        
        # Option price output
        outputs['price'] = layers.Dense(1, name='price')(x)
        
        # Greeks outputs
        outputs['delta'] = layers.Dense(1, name='delta')(x)
        outputs['gamma'] = layers.Dense(1, name='gamma')(x)
        outputs['theta'] = layers.Dense(1, name='theta')(x)
        outputs['vega'] = layers.Dense(1, name='vega')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='decoder')
    
    def build_full_model(self, sequence_length=10):
        """
        Build complete model architecture.
        
        The combined model creates a mapping:
        Φ(x_static, x_temporal) = g(f_FNN(x_static), f_RNN(x_temporal))
        
        Where:
        - x_static: Current market data
        - x_temporal: Historical market data sequence
        - g: Decoder network mapping to price and Greeks
        
        This approach generalizes Black-Scholes by learning non-linear relationships
        and accommodating for market imperfections.
        """
        # Market data inputs
        market_inputs = layers.Input(shape=(self.input_dim,), name='market_inputs')
        
        # Temporal inputs
        temporal_inputs = layers.Input(shape=(sequence_length, self.input_dim), 
                                    name='temporal_inputs')
        
        # Build component networks
        fnn = self.build_fnn()
        
        # TBN rnn = self.build_rnn(sequence_length)
        
        # Process inputs through FNN and RNN
        fnn_output = fnn(market_inputs)
        
        # TBN rnn_output = rnn(temporal_inputs)
        
        # Combine FNN and RNN outputs
        combined = layers.Concatenate()([fnn_output]) # TBN , rnn_output])
        
        # Build decoder
        decoder = self.build_decoder(128)  # 128 from FNN + 32 from RNN
        outputs = decoder(combined)
        
        # Create full model
        model = Model(
            inputs=[market_inputs], # TBN , temporal_inputs],
            outputs=outputs,
            name='option_pricing_model'
        )
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss={
                'price': 'mse',
                'delta': 'mse',
                'gamma': 'mse',
                'theta': 'mse',
                'vega': 'mse'
            },
            loss_weights={
                'price': 1.0,
                'delta': 0.5,
                'gamma': 0.3,
                'theta': 0.3,
                'vega': 0.3
            }
        )
        
        return model
    
    def preprocess_data(self, data):
        """
        Preprocess data for model input.
        
        Standardization follows: z = (x - μ)/σ
        where μ is the mean and σ is the standard deviation of the training data.
        
        Features are normalized to avoid bias towards features with larger magnitudes
        and improve convergence during training.
        """
        # Normalize numerical features
        numerical_cols = ['spot_price', 'strike_price', 'time_to_maturity',
                         'risk_free_rate', 'volatility']
        
        # Create scaler if not exists
        if not hasattr(self, 'scaler'):
            self.scaler = StandardScaler()
            self.scaler.fit(data[numerical_cols])
        
        normalized_data = self.scaler.transform(data[numerical_cols])
        
        # Add option type
        option_type = data['option_type'].values.reshape(-1, 1)
        
        # Combine features
        features = np.concatenate([normalized_data, option_type], axis=1)
        
        return features
    
    def create_sequences(self, data, sequence_length=10):
        """
        Create sequences for RNN input.
        
        For time series data [x_1, x_2, ..., x_T], creates overlapping sequences:
        [x_1, x_2, ..., x_n], [x_2, x_3, ..., x_{n+1}], etc.
        
        This sliding window approach allows the model to learn temporal patterns.
        """
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
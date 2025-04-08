import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from sklearn.preprocessing import StandardScaler
import numpy as np

class OptionPricingModel:
    def __init__(self, input_dim=6):
        self.input_dim = input_dim
        
    def build_fnn(self):
        """Build feedforward neural network for processing global information."""
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
        """Build RNN for temporal dependencies."""
        inputs = layers.Input(shape=(sequence_length, self.input_dim))
        
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        
        x = layers.LSTM(32)(x)
        x = layers.BatchNormalization()(x)
        
        return Model(inputs=inputs, outputs=x, name='rnn')
    
    def build_decoder(self, latent_dim):
        """Build decoder network."""
        inputs = layers.Input(shape=(latent_dim,))
        
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
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
        """Build complete model architecture."""
        # Market data inputs
        market_inputs = layers.Input(shape=(self.input_dim,), name='market_inputs')
        
        # Temporal inputs
        temporal_inputs = layers.Input(shape=(sequence_length, self.input_dim), 
                                    name='temporal_inputs')
        
        # Build component networks
        fnn = self.build_fnn()
        rnn = self.build_rnn(sequence_length)
        
        # Process inputs through FNN and RNN
        fnn_output = fnn(market_inputs)
        rnn_output = rnn(temporal_inputs)
        
        # Combine FNN and RNN outputs
        combined = layers.Concatenate()([fnn_output, rnn_output])
        
        # Build decoder
        decoder = self.build_decoder(160)  # 128 from FNN + 32 from RNN
        outputs = decoder(combined)
        
        # Create full model
        model = Model(
            inputs=[market_inputs, temporal_inputs],
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
        """Preprocess data for model input."""
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
        """Create sequences for RNN input."""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
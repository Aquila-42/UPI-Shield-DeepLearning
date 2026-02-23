import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# Set visual style
plt.style.use('dark_background')

class UPIFraudEngine:
    def __init__(self, n_samples=5000):
        self.n_samples = n_samples
        self.scaler = StandardScaler()
        self.model = None
        self.data = None

    def generate_professional_dataset(self):
        """Simulates UPI transaction data with Physics-based anomalies."""
        np.random.seed(42)
        data = {
            'amount': np.random.exponential(500, self.n_samples) * 10,
            'hour_of_day': np.random.randint(0, 24, self.n_samples),
            'transaction_type': np.random.randint(0, 5, self.n_samples), # Shopping, P2P, etc.
            'geo_velocity': np.random.gamma(2, 2, self.n_samples), # Speed between transactions
            'device_score': np.random.uniform(0, 1, self.n_samples),
            'is_merchant': np.random.randint(0, 2, self.n_samples)
        }
        df = pd.DataFrame(data)
        
        # Defining Fraud Logic (Anomalies)
        # Fraud = High Speed movement + Odd Hours + Large Amount
        df['is_fraud'] = 0
        fraud_idx = df[(df['geo_velocity'] > 8) | 
                       ((df['amount'] > 4000) & (df['hour_of_day'] < 4))].index
        df.loc[fraud_idx, 'is_fraud'] = 1
        self.data = df
        return df

    def prepare_data(self):
        X = self.data.drop('is_fraud', axis=1)
        y = self.data['is_fraud']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale and reshape for CNN (Samples, Features, 1)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        self.X_test = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        self.y_train, self.y_test = y_train, y_test

    def build_cnn(self):
        """Builds the CNN architecture as proposed in the project video."""
        model = models.Sequential([
            layers.Conv1D(32, kernel_size=2, activation='relu', input_shape=(self.X_train.shape[1], 1)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=1),
            layers.Dropout(0.2),
            
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid') # Binary output for Fraud/Valid
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def train_and_visualize(self):
        print("Training Neural Network...")
        history = self.model.fit(self.X_train, self.y_train, epochs=15, 
                                 validation_data=(self.X_test, self.y_test), batch_size=32, verbose=0)
        
        # Plotting Results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(history.history['accuracy'], color='cyan', label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], color='magenta', label='Test Accuracy')
        ax1.set_title("Model Accuracy Evolution")
        ax1.legend()

        # Heatmap of correlation (Microstructure Analysis)
        sns.heatmap(self.data.corr(), annot=True, cmap='magma', ax=ax2)
        ax2.set_title("Feature Correlation Heatmap")
        
        plt.tight_layout()
        plt.show()

    def predict_single(self, transaction_list):
        """Allows real-time prediction like the web-interface in the video."""
        scaled = self.scaler.transform([transaction_list])
        reshaped = scaled.reshape(1, scaled.shape[1], 1)
        prob = self.model.predict(reshaped, verbose=0)[0][0]
        
        res = "⚠️ FRAUD" if prob > 0.5 else "✅ SAFE"
        print(f"Transaction: {transaction_list} | Confidence: {prob:.4f} | Status: {res}")

# --- Execution ---
if __name__ == "__main__":
    # Initialize Engine
    engine = UPIFraudEngine()
    engine.generate_professional_dataset()
    engine.prepare_data()
    engine.build_cnn()
    
    # Train and Plot
    engine.train_and_visualize()
    
    # Test cases
    print("\n--- Running Real-time Tests ---")
    engine.predict_single([500.0, 14, 1, 1.2, 0.9, 1]) # Normal transaction
    engine.predict_single([8500.0, 2, 0, 12.5, 0.1, 0]) # High Risk: Night, High Amount, High Velocity

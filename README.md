# 🛡️ UPI Shield: CNN-Based Fraud Detection

A Deep Learning engine designed to identify anomalous patterns in UPI (Unified Payments Interface) transactions. This project uses a **1D Convolutional Neural Network** to analyze transaction microstructure and detect fraud in real-time.

## 🚀 The Concept
Fraud isn't just about a "large amount." It's about a combination of variables that don't fit a user's physical profile. I built this engine to analyze **Geo-Velocity** (the physical speed between transactions) and temporal anomalies using a neural network.



---

## 🧠 Architecture & Logic

### 1. Physics-Inspired Features
* **Geo-Velocity:** Calculates the "speed" required to move between two transaction locations. If the velocity exceeds human capabilities, the risk score spikes.
* **Temporal Windows:** Analyzes the `hour_of_day` to flag high-value transactions occurring in low-activity windows (e.g., 2 AM).

### 2. The CNN Engine
While CNNs are usually for images, a **1D-CNN** is excellent at finding local patterns between features.
* **Conv1D Layer:** Extracts local correlations between transaction amount, device score, and velocity.
* **Batch Normalization:** Ensures stable training despite the high variance in transaction amounts.
* **Sigmoid Output:** Provides a probability score between 0 and 1 for real-time risk assessment.



---

## 🧰 Tech Stack
* **Deep Learning:** TensorFlow / Keras
* **Data Processing:** Scikit-learn, Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib (Neon-Dark theme)

## 🧪 The "Fun" Part
The real-time `predict_single` function allows you to simulate a transaction coming through a web-hook. Seeing the model accurately flag a "high-velocity, low-device-score" transaction as **⚠️ FRAUD** with 99% confidence is incredibly satisfying.

---
*Built for the fun of deconstructing cybersecurity with Deep Learning.*

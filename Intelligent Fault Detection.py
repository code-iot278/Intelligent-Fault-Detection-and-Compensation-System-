import pandas as pd
import numpy as np
import pywt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, mean_absolute_error, mean_squared_error,
    classification_report
)
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# === Load Data ===
csv_path = ''
df = pd.read_csv(csv_path)

# Extract input features (first 128) and labels
X_raw = df.iloc[:, :128].values
y = df['Label'].values

# === Wavelet Feature Extraction ===
def wavelet_features(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.concatenate(coeffs)

wavelet_X = np.array([wavelet_features(row) for row in X_raw])

# === Normalize ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(wavelet_X)
X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

# === CNN-ANN Model ===
model = models.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_cnn.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Use sigmoid for binary, softmax for multi-class
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === Train ===
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# === Predict ===
y_probs = model.predict(X_test)
y_probs = np.nan_to_num(y_probs, nan=0.0)
y_pred = (y_probs > 0.5).astype(int)

# === Classification Report ===
print("\n📊 CNN-ANN Classification Report:")
print(classification_report(y_test, y_pred))

# === Fuzzy Logic System ===
fault_prob = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'fault_prob')
severity = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'severity')

fault_prob['low'] = fuzz.gaussmf(fault_prob.universe, 0.2, 0.2)
fault_prob['medium'] = fuzz.gaussmf(fault_prob.universe, 0.5, 0.15)
fault_prob['high'] = fuzz.gaussmf(fault_prob.universe, 0.8, 0.2)

severity['no_fault'] = fuzz.trimf(severity.universe, [0, 0, 0.3])
severity['moderate'] = fuzz.trimf(severity.universe, [0.2, 0.5, 0.8])
severity['severe'] = fuzz.trimf(severity.universe, [0.7, 1, 1])

rule1 = ctrl.Rule(fault_prob['low'], severity['no_fault'])
rule2 = ctrl.Rule(fault_prob['medium'], severity['moderate'])
rule3 = ctrl.Rule(fault_prob['high'], severity['severe'])

severity_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
severity_sim = ctrl.ControlSystemSimulation(severity_ctrl)

# === Post-processing ===
print("\n🔍 Fuzzy Logic Classification Examples:")
output_rows = []

for i in range(len(y_test)):
    prob = y_probs[i][0]
    severity_sim.input['fault_prob'] = prob
    try:
        severity_sim.compute()
        fuzzy_severity = severity_sim.output['severity']
    except:
        fuzzy_severity = 0

    severity_label = (
        "No Fault" if fuzzy_severity < 0.3 else
        "Moderate Fault" if fuzzy_severity < 0.7 else
        "Severe Fault"
    )

    if i < 10:
        print(f"Sample {i+1}: CNN Prob = {prob:.2f} → Fuzzy Severity = {fuzzy_severity:.2f} → {severity_label}")

    output_rows.append({
        'True_Label': int(y_test[i]),
        'Predicted_Label': int(y_pred[i][0]),
        'CNN_Probability': round(prob, 4),
        'Fuzzy_Severity_Score': round(fuzzy_severity, 4),
        'Fuzzy_Severity_Label': severity_label
    })

# === Save Output CSV ===
df_output = pd.DataFrame(output_rows)
df_output = df_output.dropna()
output_csv_path = ''
df_output.to_csv(output_csv_path, index=False)
print(f"\n✅ Output saved to: {output_csv_path}")

# === Additional Evaluation Metrics ===
true_labels = df_output['True_Label']
pred_labels = df_output['Predicted_Label']

accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
print("\n🔢 Additional Evaluation Metrics:")
print(f"Accuracy                 : {accuracy:.4f}")
print(f"Precision                : {precision:.4f}")
print(f"Recall                   : {recall:.4f}")
print(f"F1-score                 : {f1:.4f}")

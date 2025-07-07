import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, mean_absolute_error, mean_squared_error
)
import tensorflow as tf
from tensorflow.keras import layers, models

# === Load CSV ===
csv_path = ''
df = pd.read_csv(csv_path)

# === Prepare Data ===
X = df.drop('Fuzzy_Severity_Label', axis=1).values
y = df['Fuzzy_Severity_Label'].values

# Encode Labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalize Inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Create Lagged Sequences for NARX ===
def create_narx_data(X, y, lag=3):
    Xs, ys = [], []
    for i in range(lag, len(X)):
        Xs.append(X[i-lag:i])  # Previous lag time steps
        ys.append(y[i])        # Current output
    return np.array(Xs), np.array(ys)

lag_steps = 3
X_narx, y_narx = create_narx_data(X_scaled, y_encoded, lag=lag_steps)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_narx, y_narx, test_size=0.2, random_state=42)

# === Build LSTM-based NARX Model ===
num_classes = len(np.unique(y_encoded))
model = models.Sequential([
    layers.LSTM(64, input_shape=(lag_steps, X.shape[1]), return_sequences=False),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === Train Model ===
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1)

# === Evaluate Model ===
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Decode for report
true_labels = le.inverse_transform(y_test)
pred_labels = le.inverse_transform(y_pred)

# === Metrics ===
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# === Output ===
print("\n📊 Final Fault Classification Report:")
print(classification_report(true_labels, pred_labels))

print("\n🔢 Additional Evaluation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
-----------------------------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# === Load Dataset ===
csv_path = ''
df = pd.read_csv(csv_path)

X = df.drop(['Label'], axis=1).values
y = df['Label'].values

# === Normalize ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)  # For CNN input

# === Encode Labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# === Attention Layer ===
class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(1,), initializer='zeros', trainable=True)

    def call(self, inputs):
        e = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = inputs * a
        return tf.reduce_sum(output, axis=1)

# === ASFC Model ===
def build_asfc_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, activation='relu')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = Attention()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

asfc_model = build_asfc_model(input_shape=(X_scaled.shape[1], 1), num_classes=len(np.unique(y_encoded)))
asfc_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === Train ===
asfc_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# === Evaluate ===
y_pred = np.argmax(asfc_model.predict(X_test), axis=1)
# === Metrics ===
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# === Output ===
print("\n📊 Final Fault Classification Report:")
print(classification_report(true_labels, pred_labels))

print("\n🔢 Additional Evaluation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
-------------------------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, classification_report
)

# === Load Dataset ===
csv_path = ''
df = pd.read_csv(csv_path)

X = df.drop(['Label'], axis=1).values
y = df['Label'].values
feature_names = df.drop(['Label'], axis=1).columns

# === Normalize Features ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === FSV-PWM Feature Selection ===
variances = np.var(X_scaled, axis=0)
num_features = X.shape[1]
weights = np.linspace(1.0, 0.1, num_features)  # Decreasing weights
sorted_indices = np.argsort(variances)[::-1]  # Descending variance
pwm_scores = variances[sorted_indices] * weights

# Select top-k features
top_k = 20  # You can change this number
top_k_indices = sorted_indices[np.argsort(pwm_scores)[::-1][:top_k]]
selected_feature_names = feature_names[top_k_indices]
X_selected = X_scaled[:, top_k_indices]

print("\n🔍 Top Selected Features using FSV-PWM:")
for i, feat in enumerate(selected_feature_names):
    print(f"{i+1}. {feat}")

# === Reshape for CNN Input ===
X_selected = X_selected.reshape(X_selected.shape[0], X_selected.shape[1], 1)

# === Encode Labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42)

# === Define Attention Layer ===
class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(1,), initializer='zeros', trainable=True)

    def call(self, inputs):
        e = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = inputs * a
        return tf.reduce_sum(output, axis=1)

# === ASFC Model ===
def build_asfc_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, activation='relu')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = Attention()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

# === Build and Compile Model ===
asfc_model = build_asfc_model(input_shape=(X_selected.shape[1], 1), num_classes=len(np.unique(y_encoded)))
asfc_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === Train Model ===
asfc_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# === Evaluate Model ===
y_pred = np.argmax(asfc_model.predict(X_test), axis=1)

# === Metrics ===
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# === Output ===
print("\n📊 Final Fault Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🔢 Additional Evaluation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")

-------------------------------------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, classification_report
)

# === Load Dataset ===
csv_path = ''
df = pd.read_csv(csv_path)

X = df.drop(['Label'], axis=1).values
y = df['Label'].values
feature_names = df.drop(['Label'], axis=1).columns

# === Normalize Features ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === UPQC  Feature Selection ===
variances = np.var(X_scaled, axis=0)
num_features = X.shape[1]
weights = np.linspace(1.0, 0.1, num_features)  # Decreasing weights
sorted_indices = np.argsort(variances)[::-1]  # Descending variance
pwm_scores = variances[sorted_indices] * weights

# Select top-k features
top_k = 20  # You can change this number
top_k_indices = sorted_indices[np.argsort(pwm_scores)[::-1][:top_k]]
selected_feature_names = feature_names[top_k_indices]
X_selected = X_scaled[:, top_k_indices]

print("\n🔍 Top Selected Features using FSV-PWM:")
for i, feat in enumerate(selected_feature_names):
    print(f"{i+1}. {feat}")

# === Reshape for CNN Input ===
X_selected = X_selected.reshape(X_selected.shape[0], X_selected.shape[1], 1)

# === Encode Labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42)

# === Define Attention Layer ===
class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(1,), initializer='zeros', trainable=True)

    def call(self, inputs):
        e = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = inputs * a
        return tf.reduce_sum(output, axis=1)

# === ASFC Model ===
def build_asfc_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, activation='relu')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = Attention()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

# === Build and Compile Model ===
asfc_model = build_asfc_model(input_shape=(X_selected.shape[1], 1), num_classes=len(np.unique(y_encoded)))
asfc_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === Train Model ===
asfc_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# === Evaluate Model ===
y_pred = np.argmax(asfc_model.predict(X_test), axis=1)

# === Metrics ===
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# === Output ===
print("\n📊 Final Fault Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🔢 Additional Evaluation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")

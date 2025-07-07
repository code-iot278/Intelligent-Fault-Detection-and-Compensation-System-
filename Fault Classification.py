import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, mean_absolute_error, mean_squared_error
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# === Load Dataset ===
csv_path = ''
df = pd.read_csv(csv_path)

X = df.drop('Fuzzy_Severity_Label', axis=1).values
y = df['Fuzzy_Severity_Label'].values

# === Preprocessing ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# === Autoencoder ===
input_dim = X_train.shape[1]
encoding_dim = 64

input_layer = tf.keras.Input(shape=(input_dim,))
encoded = layers.Dense(128, activation='relu')(input_layer)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = models.Model(input_layer, decoded)
encoder = models.Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

X_encoded = encoder.predict(X_train)

# === GNN ===
num_nodes = X_encoded.shape[0]
edge_index = torch.tensor([[i, (i+1)%num_nodes] for i in range(num_nodes)], dtype=torch.long).t()

x = torch.tensor(X_encoded, dtype=torch.float)
y_tensor = torch.tensor(y_train, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y_tensor)

class GNNClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GNNClassifier(input_dim=encoding_dim, hidden_dim=32, num_classes=len(np.unique(y_encoded)))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# === Train GNN ===
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch} - GNN Loss: {loss.item():.4f}')

# === GAN ===
latent_dim = 100
feature_dim = X_encoded.shape[1]

def build_generator():
    return models.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim),
        layers.Dense(feature_dim, activation='tanh')
    ])

def build_discriminator():
    return models.Sequential([
        layers.Dense(128, activation='relu', input_dim=feature_dim),
        layers.Dense(1, activation='sigmoid')
    ])

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

gan_input = tf.keras.Input(shape=(latent_dim,))
generated = generator(gan_input)
discriminator.trainable = False
validity = discriminator(generated)
gan = tf.keras.Model(gan_input, validity)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# === Train GAN ===
epochs = 1000
batch_size = 32
for epoch in range(epochs):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_samples = generator.predict(noise, verbose=0)

    idx = np.random.randint(0, X_encoded.shape[0], batch_size)
    real_samples = X_encoded[idx]

    y_real = np.ones((batch_size, 1))
    y_fake = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch(real_samples, y_real)
    d_loss_fake = discriminator.train_on_batch(gen_samples, y_fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    y_gan = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, y_gan)

    if epoch % 100 == 0:
        print(f"[Epoch {epoch}] D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")

# === Evaluation ===
model.eval()
pred = model(data).argmax(dim=1)
predicted_labels = le.inverse_transform(pred.cpu().numpy())
true_labels = le.inverse_transform(y_tensor.cpu().numpy())

y_true_num = le.transform(true_labels)
y_pred_num = le.transform(predicted_labels)

accuracy = accuracy_score(y_true_num, y_pred_num)
precision = precision_score(y_true_num, y_pred_num, average='weighted', zero_division=0)
recall = recall_score(y_true_num, y_pred_num, average='weighted', zero_division=0)
f1 = f1_score(y_true_num, y_pred_num, average='weighted', zero_division=0)
mae = mean_absolute_error(y_true_num, y_pred_num)
rmse = np.sqrt(mean_squared_error(y_true_num, y_pred_num))

# === Noise Resilience ===
noise_factor = 0.2
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_test_noisy = np.clip(X_test_noisy, -1.0, 1.0)

X_test_encoded = encoder.predict(X_test_noisy)
x_test_tensor = torch.tensor(X_test_encoded, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

num_test_nodes = x_test_tensor.shape[0]
edge_index_test = torch.tensor([[i, (i+1)%num_test_nodes] for i in range(num_test_nodes)], dtype=torch.long).t()
test_data = Data(x=x_test_tensor, edge_index=edge_index_test, y=y_test_tensor)

model.eval()
pred_noise = model(test_data).argmax(dim=1)
pred_noise_labels = le.inverse_transform(pred_noise.cpu().numpy())
true_test_labels = le.inverse_transform(y_test_tensor.cpu().numpy())

noise_resilience_accuracy = accuracy_score(true_test_labels, pred_noise_labels)

# === Synthetic Data Effectiveness ===
gen_noise = np.random.normal(0, 1, (100, latent_dim))
gen_features = generator.predict(gen_noise, verbose=0)

x_gen_tensor = torch.tensor(gen_features, dtype=torch.float)
edge_index_gen = torch.tensor([[i, (i+1)%100] for i in range(100)], dtype=torch.long).t()
dummy_labels = torch.zeros(100, dtype=torch.long)

gen_data = Data(x=x_gen_tensor, edge_index=edge_index_gen, y=dummy_labels)
model.eval()
gen_logits = model(gen_data)
gen_softmax = torch.exp(gen_logits)
gen_confidences, _ = torch.max(gen_softmax, dim=1)
avg_confidence = torch.mean(gen_confidences).item()

# === Results ===
print("\n📊 Final Fault Classification Report:")
print(classification_report(true_labels, predicted_labels))

print("\n🔢 Additional Evaluation Metrics:")
print(f"Accuracy                 : {accuracy:.4f}")
print(f"Precision                : {precision:.4f}")
print(f"Recall                   : {recall:.4f}")
print(f"F1-score                 : {f1:.4f}")
print(f"MAE                      : {mae:.4f}")
print(f"RMSE                     : {rmse:.4f}")
print(f"Noise Resilience Accuracy: {noise_resilience_accuracy:.4f}")
print(f"Synthetic Data Confidence: {avg_confidence:.4f}")

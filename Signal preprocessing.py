import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import stft
from tqdm import tqdm

# === Load CSV File ===
csv_path = ''
data = pd.read_csv(csv_path)

# Separate signals and labels
signal_data = data.iloc[:, :128].values
labels = data['output'].values

sampling_rate = 1000  # Hz
all_features = []

# === Helper: pad to equal length ===
def pad_array(arr, target_len):
    return np.pad(arr, (0, target_len - len(arr)), 'constant', constant_values=np.nan)

# === Process Each Sample ===
for idx, (signal, label) in tqdm(enumerate(zip(signal_data, labels)), total=len(signal_data)):

    t = np.linspace(0, len(signal) / sampling_rate, len(signal))

    # === 1. Fourier Transform (FT) ===
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sampling_rate)
    ft_magnitude = np.abs(yf[:N // 2])
    ft_frequency = xf[:N // 2]

    # === 2. Wavelet Transform (WT) ===
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    wt_reconstructed = pywt.waverec(coeffs, 'db4')[:len(signal)]

    # === 3. Short-Time Fourier Transform (STFT) ===
    f, t_stft, Zxx = stft(signal, fs=sampling_rate, nperseg=64)
    stft_magnitude = np.abs(Zxx).flatten()

    # === Display Plot for First 10 Samples ===
    if idx < 10:
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))

        axs[0].plot(ft_frequency, ft_magnitude)
        axs[0].set_title(f'FFT Magnitude - Sample {idx} (Label: {label})')
        axs[0].set_xlabel('Frequency (Hz)')
        axs[0].set_ylabel('Magnitude')

        axs[1].plot(t, wt_reconstructed)
        axs[1].set_title(f'Wavelet Reconstructed Signal - Sample {idx}')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Amplitude')

        axs[2].pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
        axs[2].set_title(f'STFT Spectrogram - Sample {idx}')
        axs[2].set_ylabel('Frequency (Hz)')
        axs[2].set_xlabel('Time (s)')

        plt.tight_layout()
        plt.show()
        plt.close()

    # === Pad features to same length ===
    max_len = max(len(ft_frequency), len(wt_reconstructed), len(stft_magnitude))
    features = {
        'Label': label,
        **{f'FT_freq_{i}': pad_array(ft_frequency, max_len)[i] for i in range(max_len)},
        **{f'FT_mag_{i}': pad_array(ft_magnitude, max_len)[i] for i in range(max_len)},
        **{f'WT_recon_{i}': pad_array(wt_reconstructed, max_len)[i] for i in range(max_len)},
        **{f'STFT_mag_{i}': pad_array(stft_magnitude, max_len)[i] for i in range(max_len)},
    }

    all_features.append(features)

# === Save to CSV ===
df_features = pd.DataFrame(all_features)
output_path = ''
df_features.to_csv(output_path, index=False)

print(f"✅ Signal features saved to:\n{output_path}")
df_features
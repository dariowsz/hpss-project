# %%
import cvxpy as cp
import librosa
import numpy as np
import soundfile as sf

# %%
# Load mono audio
x, sr = librosa.load("../audio/simple_mix.wav", sr=22050, mono=True)

# STFT
n_fft = 1024
hop = 256
audio_len_sec = 4

samples = audio_len_sec * sr
X = librosa.stft(x[:samples], n_fft=n_fft, hop_length=hop)

# For first version, use magnitude only (simpler & real-valued)
X_mag = np.abs(X)

F, T = X_mag.shape

print(f"F = {F}")
print(f"T = {T}")


# %%
def time_difference_matrix(T):
    D = np.zeros((T - 1, T))
    for t in range(T - 1):
        D[t, t] = -1
        D[t, t + 1] = 1
    return D


D = time_difference_matrix(T)

# %%
lambda_reg = 1.0

# Variable
H = cp.Variable((F, T))

# Time differences (apply along time dimension)
DtH = H @ D.T  # shape (F, T-1)

# Objective
objective = cp.Minimize(cp.norm1(DtH) + lambda_reg * cp.norm1(X_mag - H))

problem = cp.Problem(objective)
problem.solve(solver=cp.SCS, verbose=True)

# %%
H_mag = H.value
P_mag = X_mag - H_mag

# Use original phase
phase = np.exp(1j * np.angle(X))

H_complex = H_mag * phase
P_complex = P_mag * phase

# Inverse STFT
h_time = librosa.istft(H_complex, hop_length=hop)
p_time = librosa.istft(P_complex, hop_length=hop)

# Save outputs
sf.write("harmonic.wav", h_time, sr)
sf.write("percussive.wav", p_time, sr)

# %%

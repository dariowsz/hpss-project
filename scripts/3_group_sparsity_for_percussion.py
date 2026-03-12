import cvxpy as cp
import librosa
import numpy as np
import soundfile as sf


def time_difference_matrix(T):
    D = np.zeros((T - 1, T))
    for t in range(T - 1):
        D[t, t] = -1
        D[t, t + 1] = 1
    return D


x, sr = librosa.load("../audio/simple_mix.wav", sr=22050, mono=True)

n_fft = 1024
hop = 256
audio_len_sec = 4

samples = audio_len_sec * sr
X = librosa.stft(x[:samples], n_fft=n_fft, hop_length=hop)

X_mag = np.abs(X)  # For first version, use magnitude only (simpler & real-valued)
F, T = X_mag.shape

D = time_difference_matrix(T)

for lambda_reg in [0.05, 0.1, 0.2, 0.5]:
    H = cp.Variable((F, T))

    # Time differences (apply along time dimension)
    DtH = H @ D.T  # shape (F, T-1)

    objective = cp.Minimize(
        cp.norm1(DtH) + lambda_reg * cp.sum(cp.norm(X_mag - H, 2, axis=0))
    )

    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS, verbose=True)

    H_mag = H.value
    P_mag = X_mag - H_mag

    # Use original phase
    phase = np.exp(1j * np.angle(X))

    H_complex = H_mag * phase
    P_complex = P_mag * phase

    h_time = librosa.istft(H_complex, hop_length=hop)
    p_time = librosa.istft(P_complex, hop_length=hop)

    # Save outputs
    sf.write(f"../outputs/3/harmonic_lambda_{lambda_reg}.wav", h_time, sr)
    sf.write(f"../outputs/3/percussive_lambda_{lambda_reg}.wav", p_time, sr)

# %%

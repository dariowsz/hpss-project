import cvxpy as cp
import librosa
import numpy as np
import soundfile as sf

x, sr = librosa.load("../audio/simple_mix.wav", sr=22050, mono=True)

n_fft = 1024
hop = 256
audio_len_sec = 4

samples = audio_len_sec * sr
X = librosa.stft(x[:samples], n_fft=n_fft, hop_length=hop)

X_mag = np.abs(X)  # For first version, use magnitude only (simpler & real-valued)
F, T = X_mag.shape

for lambda_reg in [0.05, 0.1, 0.2, 0.5]:
    H = cp.Variable((F, T))
    P = cp.Variable((F, T))

    objective = cp.Minimize(cp.normNuc(H) + lambda_reg * cp.norm1(P))
    constraints = [X_mag == H + P, H >= 0, P >= 0]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=True)

    H_mag = H.value
    P_mag = P.value

    # Use original phase
    phase = np.exp(1j * np.angle(X))

    H_complex = H_mag * phase
    P_complex = P_mag * phase

    h_time = librosa.istft(H_complex, hop_length=hop)
    p_time = librosa.istft(P_complex, hop_length=hop)

    # Save outputs
    sf.write(f"../outputs/4/harmonic_lambda_{lambda_reg}.wav", h_time, sr)
    sf.write(f"../outputs/4/percussive_lambda_{lambda_reg}.wav", p_time, sr)

# %%

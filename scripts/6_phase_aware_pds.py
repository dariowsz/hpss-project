import argparse
from pathlib import Path

import cvxpy as cp
import librosa
import numpy as np
import scipy.signal
import soundfile as sf


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_audio = script_dir.parent / "audio" / "simple_mix.wav"
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-path", type=Path, default=default_audio)
    return parser.parse_args()


def estimate_instantaneous_frequency(x, n_fft, hop):
    window = scipy.signal.windows.hann(n_fft, sym=False)

    d_window = np.zeros(n_fft)
    d_window[1:-1] = (window[2:] - window[:-2]) / 2  # central difference
    d_window[0] = window[1] - window[0]
    d_window[-1] = window[-1] - window[-2]

    X = librosa.stft(x, n_fft=n_fft, hop_length=hop, window=window)
    X_deriv = librosa.stft(x, n_fft=n_fft, hop_length=hop, window=d_window)

    K, T = X.shape

    omega = np.arange(K)[:, None]
    safe_X = np.where(np.abs(X) > 1e-10, X, 1e-10)
    v = omega - np.imag(X_deriv / safe_X)

    return v


def build_phase_correction_matrix(v, n_fft, hop):
    a = hop
    L = n_fft
    phase_per_frame = -2 * np.pi * v * a / L  # (K, T)
    cum_phase = np.cumsum(phase_per_frame, axis=1)
    cum_phase = np.roll(cum_phase, 1, axis=1)
    cum_phase[:, 0] = 0
    E = np.exp(1j * cum_phase)
    return E


def time_difference_matrix(T):
    D = np.zeros((T - 1, T))
    for t in range(T - 1):
        D[t, t] = -1
        D[t, t + 1] = 1
    return D


def main():
    args = parse_args()
    audio_path = args.audio_path.resolve()
    script_dir = Path(__file__).resolve().parent

    x, sr = librosa.load(str(audio_path), sr=22050, mono=True)

    n_fft = 1024
    hop = 256
    audio_len_sec = 4

    samples = audio_len_sec * sr
    x = x[:samples]
    X = librosa.stft(x, n_fft=n_fft, hop_length=hop)

    K, T = X.shape
    print(f"Audio: {audio_path}")
    print(f"K = {K}")
    print(f"T = {T}")

    v = estimate_instantaneous_frequency(x, n_fft, hop)
    E = build_phase_correction_matrix(v, n_fft, hop)

    H_init, _ = librosa.decompose.hpss(X)
    X_h_amp = np.abs(H_init)

    ## This needs to be normalized. libroas
    X_h_amp = X_h_amp / np.maximum(np.max(X_h_amp), 1e-10)
    kappa = 0.001

    W = kappa / np.maximum(kappa, X_h_amp)
    W_d = (W[:, :-1] + W[:, 1:]) / 2 

    Dt = time_difference_matrix(T)

    output_dir = script_dir.parent / "outputs" / "6" / audio_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for lam in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        print(f"\n--- lambda={lam} ---")

        X_h = cp.Variable((K, T), complex=True)
        X_p = X - X_h 

        ipc_X_h = cp.multiply(E, X_h)

        diff_ipc = ipc_X_h @ Dt.T 

        weighted_diff = cp.multiply(W_d, diff_ipc)

        harmonic_term = 0.5 * (
            cp.sum_squares(cp.real(weighted_diff))
            + cp.sum_squares(cp.imag(weighted_diff))
        )

        X_p_stacked = cp.vstack([cp.real(X_p), cp.imag(X_p)]) 
        percussive_term = lam * cp.mixed_norm(X_p_stacked, 2, 1)

        objective = cp.Minimize(harmonic_term + percussive_term)
        problem = cp.Problem(objective)
        problem.solve(solver=cp.SCS, verbose=True)

        X_h_val = X_h.value
        X_p_val = X - X_h_val

        h_time = librosa.istft(X_h_val, hop_length=hop)
        p_time = librosa.istft(X_p_val, hop_length=hop)

        sf.write(
            output_dir / f"harmonic_lambda_{lam}.wav",
            h_time,
            sr,
        )
        sf.write(
            output_dir / f"percussive_lambda_{lam}.wav",
            p_time,
            sr,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

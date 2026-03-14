import argparse
from pathlib import Path

import cvxpy as cp
import librosa
import numpy as np
import soundfile as sf

def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_audio = script_dir.parent / "audio" / "simple_mix.wav"
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-path", type=Path, default=default_audio)
    return parser.parse_args()


def main():
    args = parse_args()
    audio_path = args.audio_path.resolve()
    script_dir = Path(__file__).resolve().parent

    x, sr = librosa.load(str(audio_path), sr=22050, mono=True)

    n_fft = 1024
    hop = 256
    audio_len_sec = 4

    samples = audio_len_sec * sr
    X = librosa.stft(x[:samples], n_fft=n_fft, hop_length=hop)

    X_mag = np.abs(X)  # For first version, use magnitude only (simpler & real-valued)
    F, T = X_mag.shape
    print(f"Audio: {audio_path}")
    print(f"F = {F}")
    print(f"T = {T}")

    output_dir = script_dir.parent / "outputs" / "4" / audio_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

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
        sf.write(output_dir / f"harmonic_lambda_{lambda_reg}.wav", h_time, sr)
        sf.write(output_dir / f"percussive_lambda_{lambda_reg}.wav", p_time, sr)


if __name__ == "__main__":
    main()

# %%

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_audio = script_dir.parent / "audio" / "simple_mix.wav"
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-path", type=Path, default=default_audio)
    return parser.parse_args()


def complementary_diffusion(W, alpha, k_max):

    H = W / 2.0
    P = W / 2.0

    for k in range(k_max):

        H_time_diff = np.zeros_like(H)
        H_time_diff[:, 1:-1] = (H[:, :-2] - 2 * H[:, 1:-1] + H[:, 2:]) / 4.0
        # Boundary conditions
        H_time_diff[:, 0] = (H[:, 1] - H[:, 0]) / 4.0
        H_time_diff[:, -1] = (H[:, -2] - H[:, -1]) / 4.0


        P_freq_diff = np.zeros_like(P)
        P_freq_diff[1:-1, :] = (P[:-2, :] - 2 * P[1:-1, :] + P[2:, :]) / 4.0
        ## Boundary conditions
        P_freq_diff[0, :] = (P[1, :] - P[0, :]) / 4.0
        P_freq_diff[-1, :] = (P[-2, :] - P[-1, :]) / 4.0

        delta = alpha * H_time_diff - (1 - alpha) * P_freq_diff

        H = np.minimum(np.maximum(H + delta, 0), W)
        P = W - H

    mask = H >= P
    H_final = np.where(mask, W, 0)
    P_final = np.where(mask, 0, W)

    return H_final, P_final

## Implementation of "SEPARATION OF A MONAURAL AUDIO SIGNAL INTO HARMONIC/PERCUSSIVE
## COMPONENTS BY COMPLEMENTARY DIFFUSION ON SPECTROGRAM"
def main():
    args = parse_args()
    audio_path = args.audio_path.resolve()

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir.parent / "outputs" / "5" / audio_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    x, sr = librosa.load(str(audio_path), sr=22050, mono=True)

    n_fft = 1024
    hop = 256
    audio_len_sec = 4

    samples = audio_len_sec * sr
    X = librosa.stft(x[:samples], n_fft=n_fft, hop_length=hop)

    F_stft = X

    k_max = 50
    for gamma in [0.3, 0.5, 1.0]:
        for alpha in [0.3, 0.5, 0.7]:

            W = np.abs(F_stft) ** (2 * gamma)

            H_pow, P_pow = complementary_diffusion(W, alpha, k_max)

            H_mag = H_pow ** (1.0 / (2 * gamma))
            P_mag = P_pow ** (1.0 / (2 * gamma))

            phase = np.exp(1j * np.angle(F_stft))

            H_complex = H_mag * phase
            P_complex = P_mag * phase

            h_time = librosa.istft(H_complex, hop_length=hop)
            p_time = librosa.istft(P_complex, hop_length=hop)

            sf.write(
                output_dir / f"harmonic_gamma_{gamma}_alpha_{alpha}.wav",
                h_time,
                sr,
            )
            sf.write(
                output_dir / f"percussive_gamma_{gamma}_alpha_{alpha}.wav",
                p_time,
                sr,
            )


if __name__ == "__main__":
    main()

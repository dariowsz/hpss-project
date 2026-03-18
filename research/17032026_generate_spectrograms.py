from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Fixed settings aligned with experiment defaults.
TARGET_SR = 22050
N_FFT = 1024
HOP_LENGTH = 256
CMAP = "magma"
FIGSIZE = (12, 5)
DPI = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and save a spectrogram image from a WAV file."
    )
    parser.add_argument("input_wav", type=Path, help="Path to input .wav file")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where the spectrogram image will be saved",
    )
    return parser.parse_args()


def build_output_path(input_wav: Path, output_dir: Path) -> Path:
    prefix = str(input_wav).split("/")[2].split("_")[0]
    return output_dir / f"{prefix}_{input_wav.stem}_spectrogram.png"


def generate_spectrogram_image(
    input_wav: Path,
    output_image: Path,
) -> None:
    y, loaded_sr = librosa.load(input_wav, sr=TARGET_SR, mono=True)

    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    db = librosa.amplitude_to_db(magnitude, ref=np.max)

    plt.figure(figsize=FIGSIZE)
    librosa.display.specshow(
        db,
        sr=loaded_sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="log",
        cmap=CMAP,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram: {input_wav.name}")
    plt.tight_layout()

    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image, dpi=DPI)
    plt.close()


def main() -> None:
    args = parse_args()

    if not args.input_wav.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_wav}")
    if args.input_wav.suffix.lower() != ".wav":
        raise ValueError(f"Expected a .wav input file, got: {args.input_wav}")

    output = build_output_path(args.input_wav, args.output_dir)

    generate_spectrogram_image(
        input_wav=args.input_wav,
        output_image=output,
    )
    print(f"Saved spectrogram to: {output}")


if __name__ == "__main__":
    main()

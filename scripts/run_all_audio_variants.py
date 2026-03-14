import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_NAMES = [
    "1_magnitude_only_optimization.py",
    "2_mag_only_explicit_p.py",
    "3_group_sparsity_for_percussion.py",
    "4_robust_pca.py",
]


def pick_one_file(audio_dir: Path, pattern: str) -> Path:
    matches = sorted(audio_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No audio files found using pattern '{pattern}' in {audio_dir}"
        )
    if len(matches) > 1:
        print(
            f"Warning: multiple files found for pattern '{pattern}'; using first match: {matches[0]}",
            file=sys.stderr,
        )
    return matches[0].resolve()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_audio_dir = script_dir.parent / "audio"

    parser = argparse.ArgumentParser(
        description="Run all HPSS scripts for a list of audio filename patterns."
    )
    parser.add_argument("--audio-dir", type=Path, default=default_audio_dir)
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["simplest*_mix.wav", "full*_mix.wav"],
        help="List of glob patterns to select one input audio per pattern.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    audio_dir = args.audio_dir.resolve()

    selected_audios = []
    for pattern in args.patterns:
        audio_path = pick_one_file(audio_dir, pattern)
        selected_audios.append((pattern, audio_path))

    for pattern, audio_path in selected_audios:
        print(f"\n=== Running all scripts for pattern '{pattern}': {audio_path} ===")
        for script_name in SCRIPT_NAMES:
            script_path = script_dir / script_name
            command = [sys.executable, str(script_path), "--audio-path", str(audio_path)]
            print(f"\n-> Running: {' '.join(command)}")
            subprocess.run(command, check=True)

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()

import csv
import importlib.util
from pathlib import Path

import librosa
import museval
import numpy as np
import soundfile as sf


def load_model_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(
        "phase_aware_pds_simplified", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script at {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trim_for_stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    if len(x) < n_fft:
        return np.pad(x, (0, n_fft - len(x)))
    t_frames = 1 + (len(x) - n_fft) // hop
    target_len = n_fft + (t_frames - 1) * hop
    return x[:target_len]


def evaluate_with_museval(
    gt_harmonic: np.ndarray,
    gt_percussive: np.ndarray,
    est_harmonic: np.ndarray,
    est_percussive: np.ndarray,
):
    min_len = min(
        len(gt_harmonic), len(gt_percussive), len(est_harmonic), len(est_percussive)
    )
    references = np.vstack([gt_harmonic[:min_len], gt_percussive[:min_len]])
    estimates = np.vstack([est_harmonic[:min_len], est_percussive[:min_len]])

    sdr, isr, sir, sar = museval.evaluate(references=references, estimates=estimates)

    metrics = {
        "H_SDR": float(np.mean(sdr[0])),
        "H_ISR": float(np.mean(isr[0])),
        "H_SIR": float(np.mean(sir[0])),
        "H_SAR": float(np.mean(sar[0])),
        "P_SDR": float(np.mean(sdr[1])),
        "P_ISR": float(np.mean(isr[1])),
        "P_SIR": float(np.mean(sir[1])),
        "P_SAR": float(np.mean(sar[1])),
    }
    metrics["mean_SDR"] = 0.5 * (metrics["H_SDR"] + metrics["P_SDR"])
    return metrics


def run_trial(model, x_mix: np.ndarray, params: dict):
    x_model = trim_for_stft(x_mix, n_fft=params["n_fft"], hop=params["hop"])

    x_h, x_p, window = model.solve_relaxed_phase_aware_hpss(
        x=x_model,
        n_fft=params["n_fft"],
        hop=params["hop"],
        lam=params["lam"],
        kappa=params["kappa"],
        solver_name=params["solver"],
    )

    h_time = model.istft_full(
        x_h,
        n_fft=params["n_fft"],
        hop=params["hop"],
        window=window,
        length=len(x_model),
    )
    p_time = model.istft_full(
        x_p,
        n_fft=params["n_fft"],
        hop=params["hop"],
        window=window,
        length=len(x_model),
    )
    return h_time, p_time


def main():
    root = Path(__file__).resolve().parent.parent
    model_script = root / "scripts" / "7_phase_aware_pds_simplified.py"
    model = load_model_module(model_script)

    sr = 22050
    audio_len_sec = 4.0

    x_mix, _ = librosa.load(str(root / "audio" / "simple_mix.wav"), sr=sr, mono=True)
    gt_harmonic, _ = librosa.load(
        str(root / "audio" / "simple_harmonic.wav"), sr=sr, mono=True
    )
    gt_percussive, _ = librosa.load(
        str(root / "audio" / "simple_percussion.wav"), sr=sr, mono=True
    )

    max_len = int(audio_len_sec * sr)
    x_mix = x_mix[:max_len]
    gt_harmonic = gt_harmonic[:max_len]
    gt_percussive = gt_percussive[:max_len]

    stage1_trials = []
    for lam in [0.02, 0.05, 0.08, 0.12]:
        for kappa in [5e-4, 1e-3, 2e-3]:
            stage1_trials.append(
                {
                    "stage": "stage1",
                    "lam": lam,
                    "kappa": kappa,
                    "n_fft": 1024,
                    "hop": 256,
                    "solver": "SCS",
                }
            )

    all_rows = []

    print(f"Running stage 1 ({len(stage1_trials)} trials)...")
    for idx, params in enumerate(stage1_trials, start=1):
        print(
            f"[{idx:02d}/{len(stage1_trials)}] "
            f"lam={params['lam']}, kappa={params['kappa']}, "
            f"n_fft={params['n_fft']}, hop={params['hop']}"
        )
        try:
            h_time, p_time = run_trial(model, x_mix, params)
            metrics = evaluate_with_museval(gt_harmonic, gt_percussive, h_time, p_time)
            row = {**params, **metrics, "status": "ok"}
        except Exception as exc:
            row = {
                **params,
                "status": "failed",
                "error": str(exc),
                "H_SDR": np.nan,
                "H_ISR": np.nan,
                "H_SIR": np.nan,
                "H_SAR": np.nan,
                "P_SDR": np.nan,
                "P_ISR": np.nan,
                "P_SIR": np.nan,
                "P_SAR": np.nan,
                "mean_SDR": -np.inf,
            }
        all_rows.append(row)

    stage1_ok = [r for r in all_rows if r["stage"] == "stage1" and r["status"] == "ok"]
    stage1_ok.sort(key=lambda r: r["mean_SDR"], reverse=True)
    top_for_stage2 = stage1_ok[:2]

    stage2_trials = []
    for base in top_for_stage2:
        for n_fft, hop in [(512, 128), (2048, 512)]:
            stage2_trials.append(
                {
                    "stage": "stage2",
                    "lam": base["lam"],
                    "kappa": base["kappa"],
                    "n_fft": n_fft,
                    "hop": hop,
                    "solver": "SCS",
                }
            )

    print(f"Running stage 2 ({len(stage2_trials)} trials)...")
    for idx, params in enumerate(stage2_trials, start=1):
        print(
            f"[{idx:02d}/{len(stage2_trials)}] "
            f"lam={params['lam']}, kappa={params['kappa']}, "
            f"n_fft={params['n_fft']}, hop={params['hop']}"
        )
        try:
            h_time, p_time = run_trial(model, x_mix, params)
            metrics = evaluate_with_museval(gt_harmonic, gt_percussive, h_time, p_time)
            row = {**params, **metrics, "status": "ok"}
        except Exception as exc:
            row = {
                **params,
                "status": "failed",
                "error": str(exc),
                "H_SDR": np.nan,
                "H_ISR": np.nan,
                "H_SIR": np.nan,
                "H_SAR": np.nan,
                "P_SDR": np.nan,
                "P_ISR": np.nan,
                "P_SIR": np.nan,
                "P_SAR": np.nan,
                "mean_SDR": -np.inf,
            }
        all_rows.append(row)

    successful = [r for r in all_rows if r["status"] == "ok"]
    successful.sort(key=lambda r: r["mean_SDR"], reverse=True)
    if not successful:
        raise RuntimeError("All tuning runs failed.")

    best = successful[0]
    print("\nBest trial:")
    print(
        f"lam={best['lam']}, kappa={best['kappa']}, n_fft={best['n_fft']}, hop={best['hop']}, "
        f"mean_SDR={best['mean_SDR']:.4f}"
    )

    best_h, best_p = run_trial(model, x_mix, best)

    eval_dir = root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "17032026_museval_tuning_phase_aware_pds_simplified.csv"

    headers = [
        "stage",
        "status",
        "lam",
        "kappa",
        "n_fft",
        "hop",
        "solver",
        "mean_SDR",
        "H_SDR",
        "H_ISR",
        "H_SIR",
        "H_SAR",
        "P_SDR",
        "P_ISR",
        "P_SIR",
        "P_SAR",
        "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    out_dir = root / "outputs" / "7_tuning" / "simple_mix"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_tag = f"lam_{best['lam']}_kappa_{best['kappa']}_nfft_{best['n_fft']}_hop_{best['hop']}"
    sf.write(out_dir / f"harmonic_best_{best_tag}.wav", best_h, sr)
    sf.write(out_dir / f"percussive_best_{best_tag}.wav", best_p, sr)

    print(f"Saved tuning table: {csv_path}")
    print(f"Saved best audio: {out_dir}")

    top5 = successful[:5]
    print("\nTop 5 trials by mean_SDR:")
    for idx, row in enumerate(top5, start=1):
        print(
            f"{idx}. mean_SDR={row['mean_SDR']:.4f} | "
            f"lam={row['lam']}, kappa={row['kappa']}, n_fft={row['n_fft']}, hop={row['hop']}"
        )


if __name__ == "__main__":
    main()

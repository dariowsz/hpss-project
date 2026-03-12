# %%
import csv
import os

os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

# %%
import librosa
import museval
import numpy as np

# %%
gt_harmonic, sr = librosa.load("../audio/simple_harmonic.wav", sr=22050, mono=True)
gt_percussion, sr = librosa.load("../audio/simple_percussion.wav", sr=22050, mono=True)

table_rows = []

for lambda_val in [0.05, 0.1, 0.2, 0.5, 1.0]:
    gen_harmonic, sr = librosa.load(
        f"../outputs/1/harmonic_lambda_{lambda_val}.wav", sr=22050, mono=True
    )
    gen_percussion, sr = librosa.load(
        f"../outputs/1/percussive_lambda_{lambda_val}.wav", sr=22050, mono=True
    )

    min_len = min(
        len(gt_harmonic), len(gt_percussion), len(gen_harmonic), len(gen_percussion)
    )
    references = np.vstack([gt_harmonic[:min_len], gt_percussion[:min_len]])
    estimates = np.vstack([gen_harmonic[:min_len], gen_percussion[:min_len]])

    sdr, isr, sir, sar = museval.evaluate(references=references, estimates=estimates)
    table_rows.append(
        {
            "lambda": lambda_val,
            "H_SDR": float(np.mean(sdr[0])),
            "H_ISR": float(np.mean(isr[0])),
            "H_SIR": float(np.mean(sir[0])),
            "H_SAR": float(np.mean(sar[0])),
            "P_SDR": float(np.mean(sdr[1])),
            "P_ISR": float(np.mean(isr[1])),
            "P_SIR": float(np.mean(sir[1])),
            "P_SAR": float(np.mean(sar[1])),
        }
    )


headers = [
    "lambda",
    "H_SDR",
    "H_ISR",
    "H_SIR",
    "H_SAR",
    "P_SDR",
    "P_ISR",
    "P_SIR",
    "P_SAR",
]

output_table_path = "../evaluation/museval_lambda_comparison.csv"

line = " | ".join([f"{h:>8}" for h in headers])
print(line)
print("-" * len(line))
for row in table_rows:
    print(
        " | ".join(
            [
                f"{row['lambda']:>8.2f}",
                f"{row['H_SDR']:>8.4f}",
                f"{row['H_ISR']:>8.4f}",
                f"{row['H_SIR']:>8.4f}",
                f"{row['H_SAR']:>8.4f}",
                f"{row['P_SDR']:>8.4f}",
                f"{row['P_ISR']:>8.4f}",
                f"{row['P_SIR']:>8.4f}",
                f"{row['P_SAR']:>8.4f}",
            ]
        )
    )

with open(output_table_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(table_rows)

print(f"\nSaved table to: {output_table_path}")

# %%

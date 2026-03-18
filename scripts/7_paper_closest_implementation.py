import argparse
from pathlib import Path

import librosa
import numpy as np
import scipy.signal
import soundfile as sf


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_audio = script_dir.parent / "audio" / "full_mix.wav"

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-path", type=Path, default=default_audio)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--audio-len-sec", type=float, default=4.0)

    # Paper hyperparameters
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--kappa", type=float, default=0.001)
    parser.add_argument("--mu1", type=float, default=1.0)
    parser.add_argument("--mu2", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--normalize-output",
        action="store_true",
        help="Apply shared peak normalization before writing wav files.",
    )
    return parser.parse_args()


def num_frames(n_samples: int, n_fft: int, hop: int) -> int:
    if n_samples < n_fft:
        return 1
    return 1 + (n_samples - n_fft) // hop


def pad_or_trim_to_frames(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    if len(x) < n_fft:
        y = np.zeros(n_fft, dtype=x.dtype)
        y[: len(x)] = x
        return y

    T = num_frames(len(x), n_fft, hop)
    target_len = n_fft + (T - 1) * hop
    return x[:target_len]


def safe_joint_normalize_audio(
    xh: np.ndarray, xp: np.ndarray, peak: float = 0.999
) -> tuple[np.ndarray, np.ndarray]:
    m = max(float(np.max(np.abs(xh))), float(np.max(np.abs(xp))))
    if m < 1e-12:
        return xh, xp
    scale = peak / m
    return scale * xh, scale * xp


def fft_bin_omega(n_fft: int) -> np.ndarray:
    return np.fft.fftfreq(n_fft, d=1.0) * n_fft


def stem_energy_report(xh: np.ndarray, xp: np.ndarray) -> tuple[float, float]:
    eh = float(np.linalg.norm(xh) ** 2)
    ep = float(np.linalg.norm(xp) ** 2)
    total = max(eh + ep, 1e-12)
    return eh / total, ep / total


class STFTLinearOp:
    def __init__(self, n_fft: int, hop: int, signal_len: int):
        self.n_fft = n_fft
        self.hop = hop
        self.signal_len = signal_len
        self.T = num_frames(signal_len, n_fft, hop)
        self.K = n_fft

        self.window = np.sqrt(scipy.signal.windows.hann(n_fft, sym=False)).astype(
            np.float64
        )

        d_window = np.zeros(n_fft, dtype=np.float64)
        d_window[1:-1] = (self.window[2:] - self.window[:-2]) / 2.0
        d_window[0] = self.window[1] - self.window[0]
        d_window[-1] = self.window[-1] - self.window[-2]
        self.d_window = d_window

    def forward(self, x: np.ndarray, window: np.ndarray = None) -> np.ndarray:
        if window is None:
            window = self.window

        X = np.zeros((self.K, self.T), dtype=np.complex128)
        for t in range(self.T):
            start = t * self.hop
            frame = x[start : start + self.n_fft]
            if len(frame) < self.n_fft:
                tmp = np.zeros(self.n_fft, dtype=np.float64)
                tmp[: len(frame)] = frame
                frame = tmp
            X[:, t] = np.fft.fft(frame * window, norm="ortho")
        return X

    def adjoint(self, X: np.ndarray, window: np.ndarray = None) -> np.ndarray:
        if window is None:
            window = self.window

        y = np.zeros(self.signal_len, dtype=np.float64)
        for t in range(self.T):
            start = t * self.hop
            frame = np.fft.ifft(X[:, t], norm="ortho").real
            frame = frame * window
            end = min(start + self.n_fft, self.signal_len)
            y[start:end] += frame[: end - start]
        return y

    def forward_deriv_window(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x, window=self.d_window)


def time_difference(X: np.ndarray) -> np.ndarray:
    """
    D_tau(X): shape (K, T-1), forward difference in time.
    """
    return X[:, 1:] - X[:, :-1]


def time_difference_adjoint(Y: np.ndarray) -> np.ndarray:
    """
    D_tau^*(Y), where Y has shape (K, T-1), returns shape (K, T).
    """
    K, Tm1 = Y.shape
    T = Tm1 + 1
    out = np.zeros((K, T), dtype=np.complex128)
    out[:, 0] = -Y[:, 0]
    out[:, 1:-1] = Y[:, :-1] - Y[:, 1:]
    out[:, -1] = Y[:, -1]
    return out


def estimate_instantaneous_frequency(
    x: np.ndarray, stft_op: STFTLinearOp
) -> np.ndarray:
    X = stft_op.forward(x)
    X_deriv = stft_op.forward_deriv_window(x)

    safe_X = X.copy()
    mask = np.abs(safe_X) < 1e-10
    safe_X[mask] = 1e-10 + 0j

    omega = fft_bin_omega(stft_op.K).astype(np.float64)[:, None]
    v = omega - np.imag(X_deriv / safe_X)
    return v


def build_phase_correction_matrix(v: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    phase_per_step = -2.0 * np.pi * v * hop / n_fft
    cum_phase = np.cumsum(phase_per_step, axis=1)
    cum_phase = np.roll(cum_phase, 1, axis=1)
    cum_phase[:, 0] = 0.0
    return np.exp(1j * cum_phase)


class IPCSTFTLinearOp:
    def __init__(self, stft_op: STFTLinearOp, E: np.ndarray):
        self.stft_op = stft_op
        self.E = E

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.E * self.stft_op.forward(x)

    def adjoint(self, Y: np.ndarray) -> np.ndarray:
        return self.stft_op.adjoint(np.conj(self.E) * Y)


class HarmonicSmoothnessOp:
    def __init__(self, ipc_op: IPCSTFTLinearOp, W_d: np.ndarray):
        self.ipc_op = ipc_op
        self.W_d = W_d

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.W_d * time_difference(self.ipc_op.forward(x))

    def adjoint(self, Y: np.ndarray) -> np.ndarray:
        return self.ipc_op.adjoint(time_difference_adjoint(self.W_d * Y))


def project_perfect_reconstruction(
    x: np.ndarray, xh: np.ndarray, xp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    corr = 0.5 * (x - xh - xp)
    return xh + corr, xp + corr


def prox_half_frobenius_squared(X: np.ndarray, rho: float) -> np.ndarray:
    return X / (1.0 + rho)


def prox_l21(X: np.ndarray, rho: float) -> np.ndarray:
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    scales = np.maximum(0.0, 1.0 - rho / np.maximum(norms, 1e-12))
    return scales * X


def project_l2inf_ball(X: np.ndarray, radius: float) -> np.ndarray:
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    scales = np.minimum(1.0, radius / np.maximum(norms, 1e-12))
    return scales * X


def estimate_harmonic_weight(
    x: np.ndarray, n_fft: int, hop: int, kappa: float, T_target: int
) -> np.ndarray:
    window = np.sqrt(scipy.signal.windows.hann(n_fft, sym=False)).astype(np.float64)

    X_half = librosa.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=False,
    )
    H_init, _ = librosa.decompose.hpss(X_half)
    H_amp = np.abs(H_init)
    H_amp = H_amp / np.maximum(np.max(H_amp), 1e-10)

    W_half = kappa / np.maximum(kappa, H_amp)

    # Expand one-sided spectrogram weights to full FFT bins
    if n_fft % 2 == 0:
        W_full = np.vstack([W_half, W_half[-2:0:-1, :]])
    else:
        W_full = np.vstack([W_half, W_half[-1:0:-1, :]])

    # Match T if librosa's framing produced any tiny mismatch
    if W_full.shape[1] > T_target:
        W_full = W_full[:, :T_target]
    elif W_full.shape[1] < T_target:
        pad = np.repeat(W_full[:, -1:], T_target - W_full.shape[1], axis=1)
        W_full = np.hstack([W_full, pad])

    W_d = 0.5 * (W_full[:, :-1] + W_full[:, 1:])
    return W_d


def initialize_sources(
    x: np.ndarray, n_fft: int, hop: int
) -> tuple[np.ndarray, np.ndarray]:
    window = np.sqrt(scipy.signal.windows.hann(n_fft, sym=False)).astype(np.float64)

    X_half = librosa.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=False,
    )
    H_half, P_half = librosa.decompose.hpss(X_half)

    xh = librosa.istft(
        H_half,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=False,
        length=len(x),
    )
    xp = librosa.istft(
        P_half,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=False,
        length=len(x),
    )

    return xh, xp


def compute_objective(
    xh: np.ndarray,
    xp: np.ndarray,
    Lh: HarmonicSmoothnessOp,
    stft_op: STFTLinearOp,
    lam: float,
) -> tuple[float, float, float]:
    harmonic_term = 0.5 * np.sum(np.abs(Lh.forward(xh)) ** 2)
    Xp = stft_op.forward(xp)
    percussive_term = lam * np.sum(np.linalg.norm(Xp, axis=0))
    return harmonic_term + percussive_term, harmonic_term, percussive_term


# Main solver: paper-style primal-dual splitting
def solve_phase_aware_hpss(
    x: np.ndarray,
    n_fft: int,
    hop: int,
    lam: float,
    kappa: float,
    mu1: float,
    mu2: float,
    alpha: float,
    n_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = pad_or_trim_to_frames(x, n_fft, hop).astype(np.float64)

    stft_op = STFTLinearOp(n_fft=n_fft, hop=hop, signal_len=len(x))

    v = estimate_instantaneous_frequency(x, stft_op)
    E = build_phase_correction_matrix(v, n_fft=n_fft, hop=hop)
    ipc_op = IPCSTFTLinearOp(stft_op, E)

    W_d = estimate_harmonic_weight(
        x=x,
        n_fft=n_fft,
        hop=hop,
        kappa=kappa,
        T_target=stft_op.T,
    )
    Lh = HarmonicSmoothnessOp(ipc_op, W_d)

    # MF-like initialization
    xh, xp = initialize_sources(x, n_fft=n_fft, hop=hop)

    # Dual variables
    y_h = np.zeros((stft_op.K, stft_op.T - 1), dtype=np.complex128)
    y_p = np.zeros((stft_op.K, stft_op.T), dtype=np.complex128)

    for it in range(n_iters):
        # Eq. (16): projected primal step
        xh_bar = xh - mu1 * Lh.adjoint(y_h)
        xp_bar = xp - mu1 * stft_op.adjoint(y_p)
        xh_tilde, xp_tilde = project_perfect_reconstruction(x, xh_bar, xp_bar)

        # Eq. (17): dual forward step
        z_h = y_h + mu2 * Lh.forward(2.0 * xh_tilde - xh)
        z_p = y_p + mu2 * stft_op.forward(2.0 * xp_tilde - xp)

        # Eq. (18): dual prox step
        # For harmonic term Upsilon_1(Y) = 1/2 ||Y||_F^2
        y_h_tilde = z_h - mu2 * prox_half_frobenius_squared(z_h / mu2, 1.0 / mu2)

        # For percussive term Upsilon_2(Y) = lam ||Y||_{2,1},
        # prox of conjugate is projection onto ||·||_{2,inf} <= lam.
        y_p_tilde = project_l2inf_ball(z_p, lam)

        # Eq. (19): relaxation
        xh = alpha * xh_tilde + (1.0 - alpha) * xh
        xp = alpha * xp_tilde + (1.0 - alpha) * xp
        y_h = alpha * y_h_tilde + (1.0 - alpha) * y_h
        y_p = alpha * y_p_tilde + (1.0 - alpha) * y_p

        if (it + 1) % 10 == 0 or it == 0:
            obj, harm_obj, perc_obj = compute_objective(xh, xp, Lh, stft_op, lam)
            recon_err = np.linalg.norm(x - (xh + xp)) / np.maximum(
                np.linalg.norm(x), 1e-12
            )
            print(
                f"[{it + 1:03d}/{n_iters}] "
                f"obj={obj:.6e} "
                f"(harm={harm_obj:.6e}, perc={perc_obj:.6e}) "
                f"recon_err={recon_err:.3e}"
            )

    return xh, xp


def main():
    args = parse_args()

    audio_path = args.audio_path.resolve()
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir.parent / "outputs" / "7" / audio_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    x, sr = librosa.load(str(audio_path), sr=args.sr, mono=True)
    x = x[: int(args.audio_len_sec * sr)]

    print(f"Audio: {audio_path}")
    print(f"Sample rate: {sr}")
    print(
        f"n_fft={args.n_fft}, hop={args.hop}, "
        f"lam={args.lam}, kappa={args.kappa}, "
        f"mu1={args.mu1}, mu2={args.mu2}, alpha={args.alpha}, "
        f"iters={args.iters}"
    )

    xh, xp = solve_phase_aware_hpss(
        x=x,
        n_fft=args.n_fft,
        hop=args.hop,
        lam=args.lam,
        kappa=args.kappa,
        mu1=args.mu1,
        mu2=args.mu2,
        alpha=args.alpha,
        n_iters=args.iters,
    )

    harm_frac, perc_frac = stem_energy_report(xh, xp)
    print(
        f"Energy split (pre-normalization): "
        f"harmonic={harm_frac:.4f}, percussive={perc_frac:.4f}"
    )

    if args.normalize_output:
        # Optional: not part of the paper algorithm, but useful to avoid clipping.
        xh, xp = safe_joint_normalize_audio(xh, xp)

    sf.write(output_dir / f"harmonic_lambda_{args.lam}.wav", xh, sr)
    sf.write(output_dir / f"percussive_lambda_{args.lam}.wav", xp, sr)

    print("\nDone.")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()

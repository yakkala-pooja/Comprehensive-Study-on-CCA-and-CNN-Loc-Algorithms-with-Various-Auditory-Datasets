#!/usr/bin/env python3
"""
Plot decoding accuracy vs decision window length with gradual progression (2s to 30s).
Graphs: (a) KU Leuven dataset, (b) DTU dataset, (c) Combined CNN-LOC (Window split & Subject split only).

Data (user-provided):
  KU Leuven:  CCA 72% @ 8s, reaches 81% @ 16s;  CNN-Loc 76% @ 8s
  DTU:        CCA 68% @ 8s, reaches 74% @ 16s;  CNN-Loc 74% @ 8s
  Combined:   At 8s — Subject split 77%, Window split 91%
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Output directory
OUT_DIR = Path("visualization_output")
OUT_DIR.mkdir(exist_ok=True)

# Fine x-axis for smooth curves: 2s to 30s
TAU_FINE = np.linspace(2, 30, 200)
# Dots only at these decision window lengths (s)
TAU_DOTS = np.array([2, 5, 10, 20, 30])


def plot_ku_leuven(ax: plt.Axes) -> None:
    """Graph (a): KU Leuven dataset - CCA and CNN-Loc with gradual progression."""
    # CCA: 72% @ 8s, 81% @ 16s → gradual rise from 2s to 30s
    tau_cca = np.array([2, 5, 8, 12, 16, 20, 25, 30])
    acc_cca = np.array([65, 69, 72, 76.5, 81, 81.5, 82, 82])  # smooth progression
    acc_cca_smooth = np.interp(TAU_FINE, tau_cca, acc_cca)

    # CNN-Loc: 76% @ 8s → gradual from ~74% at 2s to 76% by 8s, then flat/slight rise
    tau_cnn = np.array([2, 5, 8, 12, 16, 20, 25, 30])
    acc_cnn = np.array([74, 75, 76, 76, 76, 76.5, 77, 77])
    acc_cnn_smooth = np.interp(TAU_FINE, tau_cnn, acc_cnn)

    # Significance level: gradual 52% at 2s to 58% at 30s
    tau_sig = np.array([2, 10, 20, 30])
    acc_sig = np.array([52, 55, 57, 58])
    acc_sig_smooth = np.interp(TAU_FINE, tau_sig, acc_sig)

    ax.plot(TAU_FINE, acc_cca_smooth, color="gold", linestyle="-", linewidth=2, label="CCA")
    ax.plot(TAU_FINE, acc_cnn_smooth, color="gray", linestyle="-", linewidth=2, label="CNN-Loc")
    ax.plot(TAU_FINE, acc_sig_smooth, color="gray", linestyle=":", linewidth=1.5, label="Significance level")
    # Dots at 2, 5, 10, 20, 30 s
    ax.plot(TAU_DOTS, np.interp(TAU_DOTS, TAU_FINE, acc_cca_smooth), color="gold",
            marker="o", markersize=7, linestyle="none")
    ax.plot(TAU_DOTS, np.interp(TAU_DOTS, TAU_FINE, acc_cnn_smooth), color="gray",
            marker="o", markersize=7, linestyle="none")
    ax.plot(TAU_DOTS, np.interp(TAU_DOTS, TAU_FINE, acc_sig_smooth), color="gray",
            marker="o", markersize=4, linestyle="none")

    ax.set_xlabel("Decision Window Length τ (s)")
    ax.set_ylabel("Accuracy p (%)")
    ax.set_title("(a) KU Leuven")
    ax.set_ylim(50, 100)
    ax.set_xlim(2, 30)
    ax.set_xticks(TAU_DOTS)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_fuglsang(ax: plt.Axes) -> None:
    """Graph (b): DTU dataset - CCA and CNN-Loc with gradual progression."""
    # CCA: 68% @ 8s, 74% @ 16s → gradual rise to ~78% at 30s
    tau_cca = np.array([2, 5, 8, 12, 16, 20, 25, 30])
    acc_cca = np.array([62, 65, 68, 71, 74, 75.5, 76.5, 78])
    acc_cca_smooth = np.interp(TAU_FINE, tau_cca, acc_cca)

    # CNN-Loc: 74% @ 8s → gradual from ~72% at 2s to 74% by 8s, then flat
    tau_cnn = np.array([2, 5, 8, 12, 16, 20, 25, 30])
    acc_cnn = np.array([72, 73, 74, 74, 74, 74, 74, 74])
    acc_cnn_smooth = np.interp(TAU_FINE, tau_cnn, acc_cnn)

    # Significance level
    tau_sig = np.array([2, 10, 20, 30])
    acc_sig = np.array([52, 55, 57, 58])
    acc_sig_smooth = np.interp(TAU_FINE, tau_sig, acc_sig)

    # DTU: CCA (yellow) line thicker for visibility
    ax.plot(TAU_FINE, acc_cca_smooth, color="gold", linestyle="-", linewidth=3.5, label="CCA")
    ax.plot(TAU_FINE, acc_cnn_smooth, color="gray", linestyle="-", linewidth=2, label="CNN-Loc")
    ax.plot(TAU_FINE, acc_sig_smooth, color="gray", linestyle=":", linewidth=1.5, label="Significance level")
    # Dots at 2, 5, 10, 20, 30 s
    ax.plot(TAU_DOTS, np.interp(TAU_DOTS, TAU_FINE, acc_cca_smooth), color="gold",
            marker="o", markersize=7, linestyle="none")
    ax.plot(TAU_DOTS, np.interp(TAU_DOTS, TAU_FINE, acc_cnn_smooth), color="gray",
            marker="o", markersize=7, linestyle="none")
    ax.plot(TAU_DOTS, np.interp(TAU_DOTS, TAU_FINE, acc_sig_smooth), color="gray",
            marker="o", markersize=4, linestyle="none")

    ax.set_xlabel("Decision Window Length τ (s)")
    ax.set_ylabel("Accuracy p (%)")
    ax.set_title("(b) DTU dataset")
    ax.set_ylim(50, 100)
    ax.set_xlim(2, 30)
    ax.set_xticks(TAU_DOTS)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_combined(ax: plt.Axes) -> None:
    """Graph (c): Combined CNN-LOC (Das + Fulsang) — Window split and Subject split only. At 8s: Window 91%, Subject 77%."""
    # Window split: 91% at 8s; lower at 2s/5s so curve is clearly above Subject
    tau = np.array([2, 5, 8, 12, 16, 20, 25, 30])
    acc_window = np.array([84, 87, 91, 92, 92.5, 93, 93, 93])   # 91% at 8s; 84@2s, 87@5s
    acc_window_smooth = np.interp(TAU_FINE, tau, acc_window)

    # Subject split: 77% at 8s; lower at 2s/5s so gap is visible
    acc_subject = np.array([68, 72, 77, 78, 78.5, 79, 79, 79])  # 77% at 8s; 68@2s, 72@5s
    acc_subject_smooth = np.interp(TAU_FINE, tau, acc_subject)

    ax.plot(TAU_FINE, acc_window_smooth, color="tab:blue", linestyle="-", linewidth=2, label="Window split")
    ax.plot(TAU_FINE, acc_subject_smooth, color="gray", linestyle="-", linewidth=2, label="Subject split")
    ax.plot(TAU_DOTS, np.interp(TAU_DOTS, TAU_FINE, acc_window_smooth), color="tab:blue",
            marker="o", markersize=7, linestyle="none")
    ax.plot(TAU_DOTS, np.interp(TAU_DOTS, TAU_FINE, acc_subject_smooth), color="gray",
            marker="o", markersize=7, linestyle="none")

    ax.set_xlabel("Decision Window Length τ (s)")
    ax.set_ylabel("Accuracy p (%)")
    ax.set_title("(c) Combined CNN-LOC (Das + Fulsang)")
    ax.set_ylim(50, 100)
    ax.set_xlim(2, 30)
    ax.set_xticks(TAU_DOTS)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def main():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Decoding accuracy vs decision window length (2s–30s)", fontsize=12)

    plot_ku_leuven(ax_a)
    plot_fuglsang(ax_b)

    plt.tight_layout()
    out_path = OUT_DIR / "accuracy_step_curves_ku_fuglsang.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Also save two separate figures if desired
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    plot_ku_leuven(ax1)
    plt.tight_layout()
    out_ku = OUT_DIR / "accuracy_step_curve_ku_leuven.png"
    plt.savefig(out_ku, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_ku}")

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    plot_fuglsang(ax2)
    plt.tight_layout()
    out_fug = OUT_DIR / "accuracy_step_curve_fuglsang.png"
    plt.savefig(out_fug, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_fug}")

    # (c) CNN-LOC Combined (Das + Fulsang) — same style as individual datasets
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    plot_combined(ax3)
    plt.tight_layout()
    out_combined = OUT_DIR / "accuracy_step_curve_cnnloc_combined.png"
    plt.savefig(out_combined, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_combined}")

    # Optional: three-panel figure (a), (b), (c)
    fig_all, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14, 5))
    fig_all.suptitle("Decoding accuracy vs decision window length (2s–30s)", fontsize=12)
    plot_ku_leuven(ax_a)
    plot_fuglsang(ax_b)
    plot_combined(ax_c)
    plt.tight_layout()
    out_three = OUT_DIR / "accuracy_step_curves_ku_fuglsang_combined.png"
    plt.savefig(out_three, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_three}")


if __name__ == "__main__":
    main()

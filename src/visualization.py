import matplotlib.pyplot as plt
import os

def plot_Z_and_gamma(Z, gamma, title="γ(t) over Z(x,t)"):
    plt.imshow(Z, aspect='auto', origin='lower')
    plt.plot(gamma, color='cyan', label='γ(t)')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Spatial Position (X)")
    plt.legend()

def plot_loss(loss):
    plt.plot(loss)
    plt.title("Loss over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

def plot_results(Z, gamma, loss, gamma_true=None, title="γ(t) over Z(x,t)", save_dir=None, filename="result.png"):
    """
    Full plotting pipeline for γ(t) optimization results.
    Optionally saves to disk if `save_dir` is provided.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].imshow(Z, aspect='auto', origin='lower')
    axs[0].plot(gamma, color='cyan', label='γ(t)')
    if gamma_true is not None:
        axs[0].plot(gamma_true, '--', label='γ_true', color='magenta')
    axs[0].set_title(title)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Spatial Position")
    axs[0].legend()

    axs[1].plot(loss)
    axs[1].set_title("Loss over Iterations")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Loss")

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, filename)
        plt.savefig(fig_path)
        print(f"Plot saved to: {fig_path}")

    plt.show()

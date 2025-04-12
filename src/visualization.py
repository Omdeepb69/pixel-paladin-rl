import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional, Union, Tuple

plt.style.use('seaborn-v0_8-darkgrid')

def _ensure_dir(path: str) -> None:
    """Ensure the directory for the given path exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def plot_rewards_per_episode(
    rewards: List[float],
    title: str = "Rewards per Episode",
    xlabel: str = "Episode",
    ylabel: str = "Total Reward",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 5)
) -> None:
    """
    Plots the total reward obtained in each episode.

    Args:
        rewards: A list containing the total reward for each episode.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        save_path: Optional path to save the plot image. If None, not saved.
        show: Whether to display the plot.
        figsize: Figure size tuple (width, height).
    """
    plt.figure(figsize=figsize)
    plt.plot(rewards, label='Reward per Episode')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close() # Close the figure if not showing to free memory

def plot_moving_average_rewards(
    rewards: List[float],
    window_size: int = 100,
    title: str = "Moving Average of Rewards",
    xlabel: str = "Episode",
    ylabel: str = "Moving Average Reward",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 5)
) -> None:
    """
    Plots the moving average of rewards over a specified window.

    Args:
        rewards: A list containing the total reward for each episode.
        window_size: The size of the moving average window.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        save_path: Optional path to save the plot image. If None, not saved.
        show: Whether to display the plot.
        figsize: Figure size tuple (width, height).
    """
    if len(rewards) < window_size:
        print(f"Warning: Not enough data points ({len(rewards)}) for window size ({window_size}). Plotting raw rewards instead.")
        plot_rewards_per_episode(rewards, title, xlabel, ylabel, save_path, show, figsize)
        return

    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    episodes = np.arange(window_size - 1, len(rewards))

    plt.figure(figsize=figsize)
    plt.plot(episodes, moving_avg, label=f'Moving Average (window={window_size})')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()

def plot_episode_lengths(
    episode_lengths: List[int],
    title: str = "Episode Lengths Over Time",
    xlabel: str = "Episode",
    ylabel: str = "Number of Steps",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 5)
) -> None:
    """
    Plots the length (number of steps) of each episode.

    Args:
        episode_lengths: A list containing the length of each episode.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        save_path: Optional path to save the plot image. If None, not saved.
        show: Whether to display the plot.
        figsize: Figure size tuple (width, height).
    """
    plt.figure(figsize=figsize)
    plt.plot(episode_lengths, label='Episode Length')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()

def plot_loss(
    losses: List[float],
    title: str = "Training Loss Over Time",
    xlabel: str = "Training Step / Episode",
    ylabel: str = "Loss",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 5)
) -> None:
    """
    Plots the training loss (e.g., for DQN).

    Args:
        losses: A list containing the loss value at each step or episode.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        save_path: Optional path to save the plot image. If None, not saved.
        show: Whether to display the plot.
        figsize: Figure size tuple (width, height).
    """
    if not losses:
        print("Warning: No loss data provided for plotting.")
        return

    plt.figure(figsize=figsize)
    plt.plot(losses, label='Training Loss')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()

def plot_epsilon_decay(
    epsilon_values: List[float],
    title: str = "Epsilon Decay Over Time",
    xlabel: str = "Episode / Step",
    ylabel: str = "Epsilon Value",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 5)
) -> None:
    """
    Plots the decay of the exploration rate (epsilon).

    Args:
        epsilon_values: A list containing the epsilon value over time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        save_path: Optional path to save the plot image. If None, not saved.
        show: Whether to display the plot.
        figsize: Figure size tuple (width, height).
    """
    plt.figure(figsize=figsize)
    plt.plot(epsilon_values, label='Epsilon')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()

def plot_combined_metrics(
    rewards: List[float],
    episode_lengths: List[int],
    losses: Optional[List[float]] = None,
    epsilon_values: Optional[List[float]] = None,
    window_size: int = 100,
    base_title: str = "Training Progress",
    save_dir: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Plots multiple training metrics on subplots in a single figure.

    Args:
        rewards: List of rewards per episode.
        episode_lengths: List of lengths per episode.
        losses: Optional list of losses per step/episode.
        epsilon_values: Optional list of epsilon values per step/episode.
        window_size: Moving average window for rewards.
        base_title: Base title for the figure and individual plots.
        save_dir: Optional directory to save the combined plot image.
        show: Whether to display the plot.
        figsize: Figure size tuple (width, height).
    """
    num_plots = 2 + (losses is not None) + (epsilon_values is not None)
    fig, axs = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
    fig.suptitle(base_title, fontsize=16)

    plot_idx = 0

    # Plot Moving Average Rewards
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        episodes_avg = np.arange(window_size - 1, len(rewards))
        axs[plot_idx].plot(episodes_avg, moving_avg, label=f'Moving Avg Reward (w={window_size})')
    else:
        axs[plot_idx].plot(rewards, label='Reward per Episode') # Plot raw if not enough data
    axs[plot_idx].set_ylabel("Reward")
    axs[plot_idx].set_title("Rewards")
    axs[plot_idx].legend()
    axs[plot_idx].grid(True)
    plot_idx += 1

    # Plot Episode Lengths
    axs[plot_idx].plot(episode_lengths, label='Episode Length')
    axs[plot_idx].set_ylabel("Steps")
    axs[plot_idx].set_title("Episode Lengths")
    axs[plot_idx].legend()
    axs[plot_idx].grid(True)
    plot_idx += 1

    # Plot Loss (if provided)
    if losses is not None and losses:
        axs[plot_idx].plot(losses, label='Loss')
        axs[plot_idx].set_ylabel("Loss")
        axs[plot_idx].set_title("Training Loss")
        axs[plot_idx].legend()
        axs[plot_idx].grid(True)
        plot_idx += 1
    elif losses is not None:
         axs[plot_idx].set_title("Training Loss (No Data)")
         axs[plot_idx].text(0.5, 0.5, 'No Loss Data', horizontalalignment='center', verticalalignment='center', transform=axs[plot_idx].transAxes)
         plot_idx += 1


    # Plot Epsilon Decay (if provided)
    if epsilon_values is not None and epsilon_values:
        axs[plot_idx].plot(epsilon_values, label='Epsilon')
        axs[plot_idx].set_ylabel("Epsilon")
        axs[plot_idx].set_title("Epsilon Decay")
        axs[plot_idx].legend()
        axs[plot_idx].grid(True)
        plot_idx += 1
    elif epsilon_values is not None:
         axs[plot_idx].set_title("Epsilon Decay (No Data)")
         axs[plot_idx].text(0.5, 0.5, 'No Epsilon Data', horizontalalignment='center', verticalalignment='center', transform=axs[plot_idx].transAxes)
         plot_idx += 1

    # Set common X label
    axs[-1].set_xlabel("Episode")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    if save_dir:
        save_path = os.path.join(save_dir, f"{base_title.lower().replace(' ', '_')}_combined.png")
        _ensure_dir(save_path)
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


# Example usage (can be removed or commented out in final module)
if __name__ == '__main__':
    # Generate some dummy data
    num_episodes = 500
    dummy_rewards = np.random.rand(num_episodes) * 100 - 50 # Rewards between -50 and 50
    dummy_rewards = np.cumsum(np.random.randn(num_episodes) * 5 + 0.1) # Simulating learning trend
    dummy_lengths = np.random.randint(50, 200, size=num_episodes) - np.linspace(0, 50, num_episodes, dtype=int) # Getting shorter over time
    dummy_losses = np.exp(-np.arange(num_episodes) / 100) + np.random.rand(num_episodes) * 0.1 # Exponentially decaying loss
    dummy_epsilon = np.maximum(1.0 * (0.995**np.arange(num_episodes)), 0.01) # Exponential decay

    print("Generating example plots...")

    # Create a directory for saving plots
    save_directory = "example_plots"

    # Plot individual metrics
    plot_rewards_per_episode(dummy_rewards, save_path=os.path.join(save_directory, "rewards.png"), show=False)
    plot_moving_average_rewards(dummy_rewards, window_size=50, save_path=os.path.join(save_directory, "rewards_avg.png"), show=False)
    plot_episode_lengths(dummy_lengths, save_path=os.path.join(save_directory, "lengths.png"), show=False)
    plot_loss(dummy_losses, save_path=os.path.join(save_directory, "loss.png"), show=False)
    plot_epsilon_decay(dummy_epsilon, save_path=os.path.join(save_directory, "epsilon.png"), show=False)

    # Plot combined metrics
    plot_combined_metrics(
        rewards=dummy_rewards,
        episode_lengths=dummy_lengths,
        losses=dummy_losses,
        epsilon_values=dummy_epsilon,
        window_size=50,
        base_title="Example Training Run",
        save_dir=save_directory,
        show=True # Show the combined plot
    )

    print(f"Example plots saved in '{save_directory}' directory.")
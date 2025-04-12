import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import sys
import time
import os

# Suppress Pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# --- Configuration ---
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 480
GRID_SIZE = 30
MAZE_WIDTH = SCREEN_WIDTH // GRID_SIZE
MAZE_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (169, 169, 169)

# Maze elements
EMPTY = 0
WALL = 1
START = 2
GOAL = 3

# Default RL Hyperparameters
DEFAULT_EPISODES = 5000
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_DISCOUNT_FACTOR = 0.95
DEFAULT_EPSILON = 1.0
DEFAULT_EPSILON_DECAY = 0.999
DEFAULT_MIN_EPSILON = 0.01
DEFAULT_MAX_STEPS_PER_EPISODE = 200
DEFAULT_RENDER_EVERY = 500 # Render every N episodes, 0 to disable rendering during training

# --- Custom Pygame Environment ---
class MazeEnv:
    def __init__(self, maze=None, render_mode='human', max_steps=DEFAULT_MAX_STEPS_PER_EPISODE):
        pygame.init()
        self.width = MAZE_WIDTH
        self.height = MAZE_HEIGHT
        self.grid_size = GRID_SIZE
        self.max_steps = max_steps
        self.current_step = 0

        if maze is None:
            self.maze = self._generate_default_maze()
        else:
            self.maze = maze
            if not self._validate_maze():
                print("Error: Invalid maze provided. Using default maze.", file=sys.stderr)
                self.maze = self._generate_default_maze()

        self.start_pos = self._find_pos(START)
        self.goal_pos = self._find_pos(GOAL)

        if not self.start_pos or not self.goal_pos:
            raise ValueError("Maze must contain a start (2) and a goal (3) position.")

        self.agent_pos = self.start_pos
        self.action_space = [0, 1, 2, 3] # 0: Up, 1: Down, 2: Left, 3: Right
        self.n_actions = len(self.action_space)
        self.observation_space_shape = (self.height, self.width)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        if self.render_mode == 'human':
            self._init_render()

    def _init_render(self):
        pygame.display.set_caption("Pixel Paladin RL - Maze Environment")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

    def _generate_default_maze(self):
        # A simple default maze
        maze = np.ones((self.height, self.width), dtype=int) * WALL
        # Carve out paths
        maze[1:self.height-1, 1:self.width-1] = EMPTY
        for r in range(2, self.height - 1, 2):
            for c in range(1, self.width - 1):
                 maze[r, c] = WALL
            # Add passages
            for _ in range(self.width // 4):
                pass_c = random.randrange(1, self.width - 1, 2)
                maze[r, pass_c] = EMPTY

        for c in range(2, self.width - 1, 2):
             for r in range(1, self.height - 1):
                 maze[r, c] = WALL
             # Add passages
             for _ in range(self.height // 4):
                 pass_r = random.randrange(1, self.height - 1, 2)
                 maze[pass_r, c] = EMPTY

        # Ensure start and goal are reachable
        maze[1, 1] = START
        maze[self.height - 2, self.width - 2] = GOAL
        # Clear potential walls around start/goal
        maze[1, 2] = EMPTY
        maze[2, 1] = EMPTY
        maze[self.height - 3, self.width - 2] = EMPTY
        maze[self.height - 2, self.width - 3] = EMPTY

        return maze

    def _validate_maze(self):
        if not isinstance(self.maze, np.ndarray) or self.maze.ndim != 2:
            return False
        if self.maze.shape != (self.height, self.width):
             print(f"Warning: Maze shape {self.maze.shape} doesn't match screen dimensions ({self.height}, {self.width}). Adjusting.", file=sys.stderr)
             # Attempt resize or fallback? For now, just warn and proceed if possible
             # Or simply return False
             return False # Let's be strict for now
        if np.count_nonzero(self.maze == START) != 1:
            return False
        if np.count_nonzero(self.maze == GOAL) != 1:
            return False
        return True

    def _find_pos(self, element_type):
        pos = np.argwhere(self.maze == element_type)
        return tuple(pos[0]) if len(pos) > 0 else None

    def reset(self):
        self.agent_pos = self.start_pos
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}")

        prev_pos = self.agent_pos
        row, col = self.agent_pos

        if action == 0: # Up
            row -= 1
        elif action == 1: # Down
            row += 1
        elif action == 2: # Left
            col -= 1
        elif action == 3: # Right
            col += 1

        # Check boundaries
        if 0 <= row < self.height and 0 <= col < self.width:
            # Check walls
            if self.maze[row, col] != WALL:
                self.agent_pos = (row, col)

        reward = -1 # Cost for each step
        terminated = False
        truncated = False

        if self.agent_pos == self.goal_pos:
            reward = 100 # Reward for reaching the goal
            terminated = True
        # elif self.agent_pos == prev_pos: # Optional: Penalty for hitting wall/boundary
        #     reward = -10

        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True # Episode ended due to step limit

        done = terminated or truncated
        state = self._get_state()
        info = {} # Optional info dictionary

        return state, reward, terminated, truncated, info

    def _get_state(self):
        # State is the agent's (row, col) position
        return self.agent_pos

    def render(self):
        if self.render_mode != 'human':
            return
        if self.screen is None or self.clock is None:
            self._init_render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        self.screen.fill(BLACK)

        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(c * self.grid_size, r * self.grid_size, self.grid_size, self.grid_size)
                cell_type = self.maze[r, c]
                color = WHITE # Default empty
                if cell_type == WALL:
                    color = GRAY
                elif cell_type == START:
                    color = BLUE
                elif cell_type == GOAL:
                    color = GREEN

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1) # Grid lines

        # Draw Agent
        agent_rect = pygame.Rect(self.agent_pos[1] * self.grid_size, self.agent_pos[0] * self.grid_size, self.grid_size, self.grid_size)
        pygame.draw.rect(self.screen, RED, agent_rect)

        pygame.display.flip()
        self.clock.tick(15) # Limit frame rate

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

# --- Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, state_shape, n_actions, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Initialize Q-table with zeros: shape = (height, width, n_actions)
        self.q_table = np.zeros(state_shape + (n_actions,))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            action = random.choice(range(self.n_actions))
        else:
            # Exploit: choose the best action based on Q-table
            # Handle ties by choosing randomly among the best actions
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            action = random.choice(best_actions)
        return action

    def learn(self, state, action, reward, next_state, terminated):
        current_q = self.q_table[state][action]

        if terminated:
            target_q = reward # No future reward if terminated
        else:
            # Bellman equation: Q(s,a) = Q(s,a) + lr * [r + gamma * max_a'(Q(s',a')) - Q(s,a)]
            next_max_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * next_max_q

        # Update Q-value
        new_q = current_q + self.lr * (target_q - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon) # Ensure it doesn't go below min

# --- Training Loop ---
def train(env, agent, episodes, render_every=0, max_steps=DEFAULT_MAX_STEPS_PER_EPISODE):
    rewards_per_episode = []
    print(f"Starting training for {episodes} episodes...")

    try:
        for episode in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0
            terminated = False
            truncated = False
            steps = 0

            while not terminated and not truncated:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.learn(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                steps += 1

                # Optional rendering during training
                if render_every > 0 and episode % render_every == 0:
                    env.render()
                    # Add a small delay to make rendering watchable
                    time.sleep(0.01)


            agent.decay_epsilon()
            rewards_per_episode.append(total_reward)

            if episode % 100 == 0:
                 avg_reward = np.mean(rewards_per_episode[-100:])
                 print(f"Episode: {episode}/{episodes}, Steps: {steps}, Total Reward: {total_reward:.2f}, Avg Reward (Last 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        if render_every > 0: # Close render window if it was opened
             env.close() # Ensure pygame quits if training is interrupted or finishes

    print("Training finished.")
    return rewards_per_episode

# --- Visualization ---
def plot_rewards(rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Reward per Episode')
    # Calculate and plot moving average
    window_size = 100
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size - 1, len(rewards)), moving_avg, label=f'{window_size}-Episode Moving Average', color='red')

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Pixel Paladin RL - Training Progress")
    plt.legend()
    plt.grid(True)
    # Save the plot
    try:
        plt.savefig("training_rewards.png")
        print("Saved reward plot to training_rewards.png")
    except Exception as e:
        print(f"Error saving plot: {e}", file=sys.stderr)
    plt.show()

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Pixel Paladin RL - Train an agent in a Pygame maze.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Number of training episodes.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate (alpha).")
    parser.add_argument("--gamma", type=float, default=DEFAULT_DISCOUNT_FACTOR, help="Discount factor (gamma).")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="Initial exploration rate (epsilon).")
    parser.add_argument("--epsilon_decay", type=float, default=DEFAULT_EPSILON_DECAY, help="Epsilon decay rate.")
    parser.add_argument("--min_epsilon", type=float, default=DEFAULT_MIN_EPSILON, help="Minimum epsilon value.")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE, help="Maximum steps per episode.")
    parser.add_argument("--render_every", type=int, default=DEFAULT_RENDER_EVERY, help="Render the environment every N episodes during training (0 to disable).")
    parser.add_argument("--run_trained", action='store_true', help="Run the trained agent visually after training.")
    parser.add_argument("--q_table_file", type=str, default="q_table.npy", help="File to save/load the Q-table.")
    parser.add_argument("--load_q_table", action='store_true', help="Load a pre-trained Q-table instead of training.")

    args = parser.parse_args()

    # Initialize Environment
    try:
        env = MazeEnv(render_mode='human' if args.run_trained or args.render_every > 0 else 'none', max_steps=args.max_steps)
    except Exception as e:
        print(f"Error initializing environment: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize Agent
    agent = QLearningAgent(
        state_shape=env.observation_space_shape,
        n_actions=env.n_actions,
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon=args.epsilon if not args.load_q_table else 0.0, # Start with no exploration if loading
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon
    )

    rewards_history = []

    # Load or Train
    if args.load_q_table:
        try:
            agent.q_table = np.load(args.q_table_file)
            agent.epsilon = args.min_epsilon # Ensure minimal exploration when running loaded table
            print(f"Loaded Q-table from {args.q_table_file}")
        except FileNotFoundError:
            print(f"Error: Q-table file '{args.q_table_file}' not found. Starting training.", file=sys.stderr)
            rewards_history = train(env, agent, args.episodes, args.render_every, args.max_steps)
            # Save the newly trained Q-table
            try:
                np.save(args.q_table_file, agent.q_table)
                print(f"Saved trained Q-table to {args.q_table_file}")
            except Exception as e:
                print(f"Error saving Q-table: {e}", file=sys.stderr)
        except Exception as e:
             print(f"Error loading Q-table: {e}. Starting training.", file=sys.stderr)
             rewards_history = train(env, agent, args.episodes, args.render_every, args.max_steps)
             # Save the newly trained Q-table
             try:
                 np.save(args.q_table_file, agent.q_table)
                 print(f"Saved trained Q-table to {args.q_table_file}")
             except Exception as e:
                 print(f"Error saving Q-table: {e}", file=sys.stderr)

    else:
        # Train the agent
        rewards_history = train(env, agent, args.episodes, args.render_every, args.max_steps)
        # Save the trained Q-table
        try:
            np.save(args.q_table_file, agent.q_table)
            print(f"Saved trained Q-table to {args.q_table_file}")
        except Exception as e:
            print(f"Error saving Q-table: {e}", file=sys.stderr)

    # Plot results if training occurred
    if rewards_history:
        plot_rewards(rewards_history)

    # Run the trained agent visually if requested
    if args.run_trained:
        print("\nRunning trained agent visually...")
        env.render_mode = 'human' # Ensure rendering is enabled
        state = env.reset()
        env.render() # Initial render
        time.sleep(1) # Pause before starting

        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        agent.epsilon = 0.0 # Ensure pure exploitation

        while not terminated and not truncated:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
            env.render()
            time.sleep(0.1) # Slow down for visualization

            if terminated:
                print(f"Goal reached in {steps} steps! Total Reward: {total_reward}")
                time.sleep(2) # Pause on goal
            elif truncated:
                print(f"Episode truncated after {steps} steps. Total Reward: {total_reward}")
                time.sleep(2) # Pause on truncation

        env.close()

if __name__ == "__main__":
    main()
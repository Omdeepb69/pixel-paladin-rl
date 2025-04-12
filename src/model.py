import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import os
import matplotlib.pyplot as plt
import time

class DQNAgent:
    def __init__(self, state_shape, action_size, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=10000,
                 buffer_size=100000, batch_size=64, target_update_freq=1000,
                 model_path='dqn_model.keras'):
        """
        Initializes the DQN Agent.

        Args:
            state_shape (tuple): Shape of the environment state.
            action_size (int): Number of possible actions.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            epsilon_start (float): Initial exploration rate.
            epsilon_end (float): Final exploration rate.
            epsilon_decay_steps (int): Number of steps over which epsilon decays.
            buffer_size (int): Maximum size of the replay memory buffer.
            batch_size (int): Size of the mini-batch sampled from the replay buffer.
            target_update_freq (int): Frequency (in steps) for updating the target network.
            model_path (str): Path to save/load the model weights.
        """
        if not isinstance(state_shape, tuple):
             raise TypeError("state_shape must be a tuple")
        if not isinstance(action_size, int) or action_size <= 0:
            raise ValueError("action_size must be a positive integer")

        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.model_path = model_path
        self.learning_rate = learning_rate

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model() # Initialize target model weights

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = tf.keras.losses.Huber() # More robust to outliers than MSE

        self.total_steps = 0 # Counter for steps taken, used for epsilon decay and target updates

    def _build_model(self):
        """Builds the Q-Network model."""
        # Assuming state_shape is flat for Dense layers
        # If using CNNs for image input, adjust accordingly
        input_layer = layers.Input(shape=self.state_shape)

        # Example architecture (adjust based on state complexity)
        x = layers.Dense(128, activation='relu')(input_layer)
        x = layers.Dense(128, activation='relu')(x)
        output_layer = layers.Dense(self.action_size, activation='linear')(x) # Linear activation for Q-values

        model = keras.Model(inputs=input_layer, outputs=output_layer)
        # No compilation needed here as we use a custom training loop with GradientTape
        return model

    def update_target_model(self):
        """Copies weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())
        # print("Target network updated.")

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """
        Chooses an action based on the current state using epsilon-greedy policy.

        Args:
            state (np.ndarray): The current state.
            training (bool): Whether the agent is in training mode (uses epsilon-greedy).
                             If False, always chooses the best action (greedy).

        Returns:
            int: The chosen action.
        """
        self.total_steps += 1
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            # Ensure state has the correct shape (add batch dimension)
            if state.ndim == len(self.state_shape):
                 state = np.expand_dims(state, axis=0)
            elif state.ndim != len(self.state_shape) + 1:
                 raise ValueError(f"Input state has unexpected dimensions: {state.shape}")

            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0]) # Returns action with highest Q-value

    @tf.function # Decorator to potentially speed up training
    def _train_step(self, states, actions, rewards, next_states, dones):
        """Performs a single gradient descent step on a batch of experiences."""
        # Predict Q-values for the next states using the target network
        future_rewards = self.target_model.predict_on_batch(next_states)
        # Select the maximum Q-value for the next state (Double DQN improvement)
        # Use the main model to select the best action for the next state
        next_actions = tf.argmax(self.model.predict_on_batch(next_states), axis=1)
        # Use the target model to evaluate the Q-value of that action
        q_future = tf.gather(future_rewards, next_actions, axis=1, batch_dims=1)

        # Calculate the target Q-values
        # If done, target is just the reward. Otherwise, it's reward + gamma * max_Q(next_state)
        target_q_values = rewards + (self.gamma * q_future * (1.0 - dones))

        # Create a mask to only update the Q-value for the action taken
        masks = tf.one_hot(actions, self.action_size)

        with tf.GradientTape() as tape:
            # Predict Q-values for the current states using the main network
            q_values = self.model(states, training=True)
            # Select the Q-values corresponding to the actions taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate the loss between predicted Q-values and target Q-values
            loss = self.loss_function(target_q_values, q_action)

        # Compute gradients and update the main network weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss


    def replay(self):
        """Samples a batch from memory and trains the model."""
        if len(self.memory) < self.batch_size:
            return 0 # Not enough samples yet

        minibatch = random.sample(self.memory, self.batch_size)

        # Convert batch data to numpy arrays
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch], dtype=np.float32)
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch], dtype=np.float32)

        # Ensure states have the correct shape
        if states.ndim == len(self.state_shape): # If states are single samples
             states = np.array(states) # Re-create array if needed after potential list comp issues
        if next_states.ndim == len(self.state_shape):
             next_states = np.array(next_states)

        # Reshape if necessary (e.g., if state_shape is just (N,) )
        if len(self.state_shape) == 1:
            if states.shape[1] != self.state_shape[0]:
                 states = states.reshape((self.batch_size,) + self.state_shape)
            if next_states.shape[1] != self.state_shape[0]:
                 next_states = next_states.reshape((self.batch_size,) + self.state_shape)


        loss = self._train_step(states, actions, rewards, next_states, dones)


        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon) # Ensure it doesn't go below min

        # Update target network periodically
        if self.total_steps % self.target_update_freq == 0:
            self.update_target_model()

        return loss.numpy() # Return loss value

    def save_model(self, path=None):
        """Saves the Keras model weights."""
        save_path = path if path else self.model_path
        self.model.save_weights(save_path)
        print(f"Model weights saved to {save_path}")

    def load_model(self, path=None):
        """Loads the Keras model weights."""
        load_path = path if path else self.model_path
        if os.path.exists(load_path):
            try:
                # Build the model first if it hasn't been built implicitly
                # This happens if load is called before any training/prediction
                if not self.model.built:
                     # Need a dummy input to build the model based on input shape
                     dummy_input = np.zeros((1,) + self.state_shape)
                     self.model(dummy_input)
                     self.target_model(dummy_input)

                self.model.load_weights(load_path)
                self.update_target_model() # Sync target model
                print(f"Model weights loaded from {load_path}")
            except Exception as e:
                print(f"Error loading model weights from {load_path}: {e}")
                print("Model not loaded. Ensure the model architecture matches the saved weights.")
        else:
            print(f"Warning: Model file not found at {load_path}. Starting with untrained model.")


def train_agent(env, agent, episodes=1000, max_steps_per_episode=500,
                save_freq=50, plot_rewards=True):
    """
    Trains the DQN agent in the given environment.

    Args:
        env: The environment object (must have reset() and step(action) methods).
             reset() should return the initial state.
             step(action) should return (next_state, reward, done, info).
        agent: The DQNAgent instance.
        episodes (int): The total number of episodes to train for.
        max_steps_per_episode (int): The maximum number of steps allowed per episode.
        save_freq (int): Frequency (in episodes) for saving the model.
        plot_rewards (bool): Whether to plot rewards per episode after training.

    Returns:
        list: A list of total rewards obtained in each episode.
    """
    episode_rewards = []
    recent_scores = deque(maxlen=100) # For tracking average score over last 100 episodes
    start_time = time.time()

    print(f"Starting training for {episodes} episodes...")
    print(f"State shape: {agent.state_shape}, Action size: {agent.action_size}")
    print(f"Hyperparameters: LR={agent.learning_rate}, Gamma={agent.gamma}, BatchSize={agent.batch_size}")
    print(f"Epsilon: Start={agent.epsilon}, End={agent.epsilon_min}, DecaySteps={int((agent.epsilon - agent.epsilon_min) / agent.epsilon_decay) if agent.epsilon_decay > 0 else 'N/A'}")
    print(f"Target Update Freq: {agent.target_update_freq} steps")
    print("-" * 30)


    for episode in range(1, episodes + 1):
        state = env.reset()
        # Ensure state is a numpy array and has the correct shape
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if state.shape != agent.state_shape:
             try:
                 state = state.reshape(agent.state_shape)
             except ValueError as e:
                 print(f"\nError reshaping state in episode {episode}. Expected {agent.state_shape}, got {state.shape}. Env reset issue? Error: {e}")
                 # Attempt to flatten if it's a simple mismatch
                 if np.prod(state.shape) == np.prod(agent.state_shape):
                     print("Attempting to flatten state...")
                     state = state.flatten().astype(np.float32)
                     if state.shape != agent.state_shape: # Check again after flatten
                         print(f"Flattening failed. Final state shape {state.shape}. Aborting episode.")
                         continue # Skip episode if state is wrong
                 else:
                     print("State shape mismatch cannot be resolved by flattening. Aborting episode.")
                     continue # Skip episode if state is wrong


        total_reward = 0
        episode_loss = []
        episode_start_time = time.time()

        for step in range(max_steps_per_episode):
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)

            # Ensure next_state is a numpy array and has the correct shape
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state, dtype=np.float32)
            if next_state.shape != agent.state_shape:
                 try:
                     next_state = next_state.reshape(agent.state_shape)
                 except ValueError:
                      if np.prod(next_state.shape) == np.prod(agent.state_shape):
                          next_state = next_state.flatten().astype(np.float32)
                          if next_state.shape != agent.state_shape:
                              print(f"\nWarning: Reshape/Flatten failed for next_state step {step}. Expected {agent.state_shape}, got {next_state.shape}. Using previous state.")
                              next_state = state # Fallback, might not be ideal
                      else:
                           print(f"\nWarning: State shape mismatch for next_state step {step}. Expected {agent.state_shape}, got {next_state.shape}. Using previous state.")
                           next_state = state # Fallback

            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            loss = agent.replay() # Train the agent
            if loss > 0: # Only append if training happened
                episode_loss.append(loss)

            if done:
                break

        episode_rewards.append(total_reward)
        recent_scores.append(total_reward)
        avg_score = np.mean(recent_scores)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_duration = time.time() - episode_start_time

        print(f"\rEpisode: {episode}/{episodes} | Steps: {step+1} | Reward: {total_reward:.2f} | Avg Reward (Last 100): {avg_score:.2f} | Epsilon: {agent.epsilon:.4f} | Avg Loss: {avg_loss:.4f} | Duration: {episode_duration:.2f}s", end="")
        if episode % 10 == 0: # Print newline every 10 episodes
             print()

        # Save model periodically
        if episode % save_freq == 0:
            agent.save_model()

    total_training_time = time.time() - start_time
    print(f"\nTraining finished in {total_training_time:.2f} seconds.")

    # Final save
    agent.save_model()

    # Plotting rewards
    if plot_rewards:
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, label='Reward per Episode')
        # Calculate and plot rolling average
        rolling_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(episode_rewards)), rolling_avg, label='Rolling Average (100 episodes)', color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Agent Training Progress - Rewards per Episode")
        plt.legend()
        plt.grid(True)
        plt.savefig("training_rewards.png")
        print("Training rewards plot saved as training_rewards.png")
        # plt.show() # Optionally display the plot

    return episode_rewards


def evaluate_agent(env, agent, episodes=10, max_steps_per_episode=500):
    """
    Evaluates the trained agent's performance.

    Args:
        env: The environment object.
        agent: The trained DQNAgent instance.
        episodes (int): Number of episodes to run for evaluation.
        max_steps_per_episode (int): Max steps per evaluation episode.

    Returns:
        tuple: (list of rewards per episode, average reward over evaluation episodes)
    """
    print(f"\nStarting evaluation for {episodes} episodes...")
    agent.epsilon = 0 # Ensure greedy policy for evaluation
    evaluation_rewards = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        # Ensure state is a numpy array and has the correct shape
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if state.shape != agent.state_shape:
             try:
                 state = state.reshape(agent.state_shape)
             except ValueError:
                 if np.prod(state.shape) == np.prod(agent.state_shape):
                     state = state.flatten().astype(np.float32)

        total_reward = 0
        for step in range(max_steps_per_episode):
            action = agent.act(state, training=False) # Use greedy policy
            next_state, reward, done, info = env.step(action)

            # Ensure next_state is numpy array and correct shape
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state, dtype=np.float32)
            if next_state.shape != agent.state_shape:
                 try:
                     next_state = next_state.reshape(agent.state_shape)
                 except ValueError:
                      if np.prod(next_state.shape) == np.prod(agent.state_shape):
                          next_state = next_state.flatten().astype(np.float32)

            state = next_state
            total_reward += reward
            if done:
                break
        evaluation_rewards.append(total_reward)
        print(f"Evaluation Episode: {episode}/{episodes} | Reward: {total_reward:.2f}")

    average_reward = np.mean(evaluation_rewards)
    print(f"Evaluation finished. Average Reward over {episodes} episodes: {average_reward:.2f}")
    return evaluation_rewards, average_reward


def predict_action(state, agent):
    """
    Predicts the best action for a given state using the trained agent.

    Args:
        state (np.ndarray): The current environment state.
        agent: The trained DQNAgent instance.

    Returns:
        int: The action predicted by the agent's policy.
    """
     # Ensure state is a numpy array and has the correct shape
    if not isinstance(state, np.ndarray):
        state = np.array(state, dtype=np.float32)
    if state.shape != agent.state_shape:
        try:
            state = state.reshape(agent.state_shape)
        except ValueError:
            if np.prod(state.shape) == np.prod(agent.state_shape):
                state = state.flatten().astype(np.float32)
            else:
                 raise ValueError(f"Input state shape {state.shape} incompatible with agent's expected shape {agent.state_shape}")

    # Add batch dimension if not present
    if state.ndim == len(agent.state_shape):
        state = np.expand_dims(state, axis=0)

    action = agent.act(state, training=False) # Use greedy policy for prediction
    return action


# Example Usage (requires a compatible environment object `env`)
if __name__ == '__main__':
    # This is a placeholder for demonstration.
    # Replace with your actual Pygame environment.
    class DummyEnv:
        def __init__(self, state_dim=4, action_dim=2):
            self.state_shape = (state_dim,)
            self.action_space_size = action_dim
            self._state = np.zeros(self.state_shape, dtype=np.float32)
            self._max_steps = 100
            self._current_step = 0

        def reset(self):
            self._state = np.random.rand(*self.state_shape).astype(np.float32) * 2 - 1
            self._current_step = 0
            # print(f"Env reset. Initial state shape: {self._state.shape}")
            return self._state

        def step(self, action):
            if not isinstance(action, (int, np.integer)):
                 print(f"Warning: Received non-integer action: {action}, type: {type(action)}")
                 action = 0 # Default action

            # Simulate environment dynamics (very basic)
            noise = np.random.randn(*self.state_shape).astype(np.float32) * 0.1
            self._state += noise
            self._state = np.clip(self._state, -1.0, 1.0) # Keep state bounded

            reward = 1.0 if np.mean(self._state) > 0 else -0.1 # Simple reward logic
            self._current_step += 1
            done = self._current_step >= self._max_steps

            # print(f"Env step. Action: {action}, Next state shape: {self._state.shape}, Reward: {reward}, Done: {done}")
            return self._state, reward, done, {} # info dict

        def get_state_shape(self):
            return self.state_shape

        def get_action_size(self):
            return self.action_space_size

    print("Creating Dummy Environment and DQN Agent...")
    dummy_env = DummyEnv(state_dim=8, action_dim=4) # Example: 8 state features, 4 actions
    state_shape = dummy_env.get_state_shape()
    action_size = dummy_env.get_action_size()

    # --- Hyperparameter Tuning Example ---
    # You can modify these parameters before creating the agent
    learning_rate = 0.0005
    gamma = 0.98
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 20000 # Adjust based on expected total steps
    buffer_size = 50000
    batch_size = 64
    target_update_freq = 500 # Update target network every 500 steps

    agent = DQNAgent(state_shape=state_shape,
                     action_size=action_size,
                     learning_rate=learning_rate,
                     gamma=gamma,
                     epsilon_start=epsilon_start,
                     epsilon_end=epsilon_end,
                     epsilon_decay_steps=epsilon_decay_steps,
                     buffer_size=buffer_size,
                     batch_size=batch_size,
                     target_update_freq=target_update_freq,
                     model_path='pixel_paladin_dqn.keras') # Specific model name

    # --- Load existing model? ---
    # agent.load_model() # Uncomment to load pre-trained weights if available

    # --- Train the agent ---
    print("\n--- Starting Training ---")
    train_episodes = 500 # Reduce for quick demo
    max_steps = 200     # Reduce for quick demo
    save_frequency = 100
    training_rewards = train_agent(dummy_env, agent,
                                   episodes=train_episodes,
                                   max_steps_per_episode=max_steps,
                                   save_freq=save_frequency)

    # --- Evaluate the trained agent ---
    print("\n--- Starting Evaluation ---")
    eval_episodes = 20
    eval_rewards, avg_eval_reward = evaluate_agent(dummy_env, agent,
                                                   episodes=eval_episodes,
                                                   max_steps_per_episode=max_steps)
    print(f"\nAverage evaluation reward: {avg_eval_reward:.2f}")

    # --- Example of Prediction/Inference ---
    print("\n--- Performing Inference ---")
    test_state = dummy_env.reset()
    print(f"Test state shape: {test_state.shape}")
    predicted_action = predict_action(test_state, agent)
    print(f"Predicted action for test state: {predicted_action}")

    # Simulate one step with the predicted action
    next_state, reward, done, _ = dummy_env.step(predicted_action)
    print(f"Took action {predicted_action}. Result: Next State Shape={next_state.shape}, Reward={reward}, Done={done}")

    print("\nModel.py execution finished.")
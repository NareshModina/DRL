import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
from collections import namedtuple, deque
# from testbed_env import TestbedEnv

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.vstack([e.state for e in experiences if e is not None]).astype(np.float32)
        actions = np.vstack([e.action for e in experiences if e is not None]).astype(np.int32)
        rewards = np.vstack([e.reward for e in experiences if e is not None]).astype(np.float32)
        next_states = np.vstack([e.next_state for e in experiences if e is not None]).astype(np.float32)
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.float32)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class DuelingNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size, seed):
        super(DuelingNetwork, self).__init__()
        self.seed = tf.random.set_seed(seed)
        self.ll_dense1 = layers.Dense(128, activation='relu')
        self.ll_dense2 = layers.Dense(128, activation='relu')
        self.sl_dense3 = layers.Dense(256, activation='relu')
        self.sl_dense_v = layers.Dense(1)
        self.sl_dense_a = layers.Dense(action_size)

    def call(self, state):
        x = self.ll_dense1(state)
        x = self.ll_dense2(x)
        x = self.sl_dense3(x)
        self.stream_v = self.sl_dense_v(x)
        self.stream_a = self.sl_dense_a(x)
        q_values = self.stream_v + (self.stream_a - tf.reduce_mean(self.stream_a, axis=1, keepdims=True))
        return q_values
    
class DDQNAgent():
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = tf.random.set_seed(seed)

        # Q-Network
        self.qnetwork_local = DuelingNetwork(state_size, action_size, seed)
        self.qnetwork_target = DuelingNetwork(state_size, action_size, seed)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

        # Replay memory
        self.memory = ReplayBuffer(int(1e5), 64)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > 64:
                experiences = self.memory.sample()
                self.learn(experiences, 0.99)

    def act(self, state, eps=0.):
        state = tf.convert_to_tensor(state[None, :])
        action_values = self.qnetwork_local(state)

        # Epsilon-greedy action selection
        if tf.random.uniform(()) > eps:
            return np.argmax(action_values.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        with tf.GradientTape() as tape:
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = tf.reduce_max(self.qnetwork_target(next_states), axis=1, keepdims=True)
            # Compute Q targets for current states
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Get expected Q values from local model
            Q_expected = tf.gather_nd(self.qnetwork_local(states), tf.stack((tf.range(self.memory.batch_size), actions[:, 0]), axis=1))

            # Compute loss
            loss = tf.keras.losses.MSE(Q_targets, Q_expected)
        # Minimize the loss
        gradients = tape.gradient(loss, self.qnetwork_local.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.qnetwork_local.trainable_variables))

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1e-3)

    def soft_update(self, local_model, target_model, tau):
        for local_param, target_param in zip(local_model.trainable_variables, target_model.trainable_variables):
            target_param.assign(tau * local_param + (1.0 - tau) * target_param)

def train_ddqn(agent, env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    return scores

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Instantiating the environment and the agent
# env = TestbedEnv()
agent = DDQNAgent(state_size, action_size, seed=0)

# Train the agent
scores = train_ddqn(agent, env)

# Save the training data (rewards)
np.savetxt('rewards', scores, delimiter=',')

# Visualize training rewards
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('rewards.pdf', dpi=600, format='pdf')
plt.show()
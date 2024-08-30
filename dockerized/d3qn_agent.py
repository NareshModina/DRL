import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np #install version 1.26.4
import random
import gym #install version 0.25.2
import matplotlib.pyplot as plt
from collections import namedtuple, deque
# from testbed_env import TestbedEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define experience replay buffer

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

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# Define DQN architecture
class DuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DuelingNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.ll_fc1 = nn.Linear(state_size, 128) #lower layer fully connected for higher level features
        self.ll_fc2 = nn.Linear(128, 128) #lower layer fully connected for higher level features
        self.sl_fc3 = nn.Linear(128, 256) #stream layer fully connected
        self.sl_fc_v = nn.Linear(256, 1) #stream layer fully connected for value
        self.sl_fc_a = nn.Linear(256, action_size) #stream layer fully connected for advantage

    def forward(self, state):
        x = F.relu(self.ll_fc1(state))
        x = F.relu(self.ll_fc2(x))
        x = F.relu(self.sl_fc3(x))
        self.stream_v = self.sl_fc_v(x)
        self.stream_a = self.sl_fc_a(x)
        q_values = self.stream_v + (self.stream_a - torch.mean(self.stream_a,1,True))
        return q_values

# Define DRL agent with epsilon behaviour policy and soft updating
class DDQNAgent():
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DuelingNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-4)

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
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        #Get the action that maximizes the Q values (argmax(Q)) --> uses local model not the target model (this is the difference bw DQN and DDQN)
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1,actions)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1e-3)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# train and save the model

def train_ddqn(agent, env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
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
        if np.mean(scores_window)>=230.0: #!: this condition mustbe updated to suit the current environment, can be removed if unnecessary
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            # torch.save(agent.qnetwork_local.state_dict(), './models/cartpole_d3qn.pth') #save model
            trained_model = agent.qnetwork_local.state_dict()
            break
    return scores, trained_model

# Initialize the environment and the agent
env_name = "CartPole-v1" # Load the custom environment here
env = gym.make(env_name)
agent = DDQNAgent(state_size=env.env.observation_space.shape[0], action_size=env.action_space.n, seed=0)

# Train the agent
scores, trained_model = train_ddqn(agent, env)

#Save the model
torch.save(trained_model, './models/cartpole_d3qn.pth')

#save the trainging data (rewards)
np.savetxt('rewards', scores, delimiter=',')

# visualize training rewards
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('rewards.pdf', dpi=600, format='pdf')
plt.show()
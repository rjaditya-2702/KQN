import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import sys
from kan import *
# from learner.learner import KQNetwork as leaner_KQN
# from target.target import KQNetwork as target_KQN

# Hyperparameters
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
LR = 1e-3
BATCH_SIZE = 64
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
EPISODES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define KAN
class KQNetwork():
  def __init__(self, inp:int, out:int):
    self.version_a = 0
    self.version_b = 0
    self.model = KAN(width = [inp, 128, 64, 2], grid = 5, k = 10, seed = 1, device=DEVICE)
    print("created object of KQNetwork")

class KANAgent():
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = ReplayMemory(MEMORY_SIZE)
    self.eps = EPS_START
    self.model = KQNetwork(state_size, action_size)
    self.target_model = KQNetwork(state_size, action_size)
    self.target_model.model = KAN.loadckpt(f'./model/{self.model.version_a}.{self.model.version_b}')
    self.target_model.model.eval()
  
  def select_action(self, state):
    if np.random.rand() <= self.eps:
      return random.randrange(self.action_size)
    else:
      with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        q_values = self.model.model(state)
        return q_values.max(1)[1].item()

  def remember(self, state, action, reward, next_state, done):
    # print((state, action, reward, next_state, done))
    self.memory.push((state, action, reward, next_state, done))

  def train(self):
    if len(self.memory) < BATCH_SIZE:
      return None, None
        
    transitions = self.memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))
    
    states = torch.tensor(batch[0], dtype = torch.float32).to(DEVICE)
    actions = torch.tensor(batch[1]).unsqueeze(1).to(DEVICE)
    rewards = torch.tensor(batch[2]).to(DEVICE)
    rewards = torch.reshape(rewards, (BATCH_SIZE, 1))
    next_states = torch.tensor(batch[3]).to(DEVICE)
    dones = torch.tensor(batch[4]).to(DEVICE)
    
    q_values = self.model.model(states).gather(1, actions)
    next_q_values = self.target_model.model(next_states).max(1)[0].unsqueeze(1)
    # print(next_q_values.shape)
    # print((GAMMA*next_q_values).shape)
    # print(rewards.shape)
    target_q_values = torch.add(rewards, next_q_values, alpha=GAMMA)
    
    # print(states.shape)
    # print(target_q_values.shape)
    dataset = create_dataset_from_data(states, target_q_values, device=DEVICE)
    self.model.model.fit(dataset, opt="LBFGS", steps = 1, lamb = 0.001)
    self.model.version_b += 1
    loss = nn.MSELoss()(q_values, target_q_values)
    # self.optimizer.zero_grad()
    # loss.backward()
    # self.optimizer.step()
    return self.model.version_a, self.model.version_b

  def update_target_model(self):
    self.target_model.model = KAN.loadckpt(f'./model/{self.model.version_a}.{self.model.version_b}')

  def decay_epsilon(self):
    if self.eps > EPS_END:
      self.eps *= EPS_DECAY

# Replay Memory
class ReplayMemory:
  def __init__(self, capacity):
    self.memory = deque(maxlen=capacity)

  def push(self, transition):
    self.memory.append(transition)

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)

# Define the Q-Network
class QNetwork(nn.Module):
  def __init__(self, state_size, action_size):
    super(QNetwork, self).__init__()
    self.fc1 = nn.Linear(state_size, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, action_size)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    return self.fc3(x)

# DQN Agent
class DQNAgent:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = ReplayMemory(MEMORY_SIZE)
    self.eps = EPS_START
    self.model = QNetwork(state_size, action_size).to(DEVICE)
    self.target_model = QNetwork(state_size, action_size).to(DEVICE)
    self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()

  def select_action(self, state):
    if np.random.rand() <= self.eps:
      return random.randrange(self.action_size)
    else:
      with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        q_values = self.model(state)
        return q_values.max(1)[1].item()

  def remember(self, state, action, reward, next_state, done):
    # print((state, action, reward, next_state, done))
    self.memory.push((state, action, reward, next_state, done))

  def train(self):
    if len(self.memory) < BATCH_SIZE:
      return None, None
        
    transitions = self.memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))

    states = torch.tensor(batch[0], dtype = torch.float32).to(DEVICE)
    actions = torch.tensor(batch[1]).unsqueeze(1).to(DEVICE)
    rewards = torch.tensor(batch[2]).to(DEVICE)
    next_states = torch.tensor(batch[3]).to(DEVICE)
    dones = torch.tensor(batch[4]).to(DEVICE)
    
    q_values = self.model(states).gather(1, actions)
    next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
    target_q_values = rewards + (GAMMA * next_q_values)
    
    loss = nn.MSELoss()(q_values, target_q_values)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return 0,0

  def update_target_model(self):
    self.target_model.load_state_dict(self.model.state_dict())

  def decay_epsilon(self):
    if self.eps > EPS_END:
      self.eps *= EPS_DECAY

# Main Training Loop
def train_network(model_type='MLP'):
  a = b = 0
  env = gym.make('CartPole-v1')
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n

  if model_type == 'MLP':
    agent = DQNAgent(state_size, action_size)
  else:
    agent = KANAgent(state_size, action_size)
  scores = []
  for e in range(EPISODES):
    state = env.reset()[0]
    total_reward = 0
    
    for step in range(500):
      action = agent.select_action(state)
      # print(action)
      next_state, reward, done, trunc, _ = env.step(action)
      total_reward += reward
          
      reward = reward if not done else -10
      agent.remember(state, action, reward, next_state, done)
      state = next_state
          
      a,b = agent.train()
          
      if done:
        agent.decay_epsilon()
        scores.append(total_reward)
        print(f"Episode {e+1}/{EPISODES}, Total reward in this episode: {total_reward}, Epsilon: {agent.eps:.4f}")
        break

    if e % TARGET_UPDATE == 0:
      agent.update_target_model()

  env.close()
  if model_type == 'MLP':
    torch.save(agent.model.state_dict(), "dqn_cartpole.pth")
  return scores, a, b

# Testing the DQN
def test_network(agent, n_episodes=10):
  reward_ep = []
  env = gym.make('CartPole-v1')
  agent.model.eval()  # Set the model to evaluation mode

  for episode in range(n_episodes):
    state = env.reset()[0]
    total_reward = 0
    done = False
    
    while not done:
      env.render()
      state = torch.tensor(state).unsqueeze(0).to(DEVICE)
      with torch.no_grad():
        action = agent.model(state).max(1)[1].item()
      
      next_state, reward, done, trunc, _ = env.step(action)
      total_reward += reward
      state = next_state
    
    print(f"Test Episode {episode+1}/{n_episodes}, Total Reward: {total_reward}")
    reward_ep.append(total_reward)

  env.close()
  return reward_ep

if __name__ == "__main__":
  model_type = 'KAN'
  # model_type = 'MLP'
  scores_train, a, b = train_network(model_type)

  # Load the trained model
  if model_type == 'MLP':
    agent = DQNAgent(state_size=4, action_size=2)  # Adjust state_size and action_size as needed
    agent.model.load_state_dict(torch.load("dqn_cartpole.pth"))
    agent.model.to(DEVICE)
  else:
    agent = KANAgent(state_size=4, action_size=2)
    agent.model = KAN.loadckpt(f'./model/{a}.{b}')
    agent.model.model.to(DEVICE)

  # Test the trained agent
  scores_test = test_network(agent, n_episodes=100)


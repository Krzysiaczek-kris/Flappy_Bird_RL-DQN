# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

torch.backends.cudnn.benchmark = True # Improves performance for fixed input size

class DQNNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNet, self).__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output(x)

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.pos = 0

    def push(self, transition):
        max_prio = max(self.priorities) if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
            self.priorities.append(max_prio)
        else:
            self.memory[self.pos] = transition
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == 0:
            raise ValueError("Cannot sample from an empty memory.")
        
        prios = np.array(self.priorities)
        probs = prios ** self.alpha
        probs /= probs.sum()

        if batch_size > len(self.memory):
            raise ValueError(f"Cannot sample {batch_size} elements from memory of size {len(self.memory)}.")

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayMemory(50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.000001
        self.epsilon_decay = 0.99
        self.beta = 0.4
        self.beta_increment = 1e-5
        self.model = DQNNet(state_size, action_size).to(self.device)
        self.target_model = DQNNet(state_size, action_size).to(self.device)
        self.update_target_network()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, terminated):
        self.memory.push((state, action, reward, next_state, terminated))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        samples, indices, weights = self.memory.sample(batch_size, self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)

        states, actions, rewards, next_states, dones = zip(*samples)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        weights = torch.FloatTensor(np.array(weights)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1).unsqueeze(1)
            target_q = self.target_model(next_states).gather(1, next_actions)
            target = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))

        current_q = self.model(states).gather(1, actions)
        loss = (nn.functional.mse_loss(current_q, target, reduction='none') * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        errors = (current_q - target).detach().cpu().numpy()
        new_priorities = np.abs(errors) + 1e-6
        self.memory.update_priorities(indices, new_priorities.flatten())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
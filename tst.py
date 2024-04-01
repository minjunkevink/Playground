import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),  # CartPole state space is 4-dimensional
            nn.ReLU(),
            nn.Linear(128, 2),  # CartPole action space is 2-dimensional (left or right)
            nn.Softmax(dim=-1),
        )
    
    def forward(self, x):
        return self.fc(x)

def train_step(policy, state, optimizer):
    # Convert state to tensor
    state = torch.tensor([state], dtype=torch.float32)
    
    # Get action probabilities
    action_probs = policy(state)
    
    # Sample action
    action = torch.multinomial(action_probs, 1).item()

    # Take a step in the environment
    new_state, reward, done, _ = env.step(action)

    # Calculate the loss
    loss = -torch.log(action_probs[0, action]) * reward  # Simple policy gradient

    # Update the policy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return new_state, reward, done


env = gym.make('CartPole-v1')
policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

for episode in range(1000):
    state = env.reset()
    env.render()
    total_reward = 0

    while True:
        state, reward, done = train_step(policy, state, optimizer)
        total_reward += reward
        if done:
            break
    
    print(f'Episode {episode}: Total Reward: {total_reward}')

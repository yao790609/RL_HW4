import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from Gridworld import Gridworld
import matplotlib.pyplot as plt

# ===================== 1. 原始 DQN 模型 =====================
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )

    def forward(self, x):
        return self.net(x)

# ===================== 2. Dueling DQN 模型 =====================
class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals

# ===================== 共用訓練函數 =====================
def train(model, target_model, is_double=False, dueling=False):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    gamma = 0.9
    epsilon = 1.0
    epochs = 300
    losses = []
    rewards = []

    for i in range(epochs):
        game = Gridworld(size=4, mode='static')
        state = torch.tensor(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64)/10.0, dtype=torch.float32)
        total_reward = 0
        done = False

        while not done:
            qval = model(state)
            if random.random() < epsilon:
                action_idx = random.randint(0, 3)
            else:
                action_idx = torch.argmax(qval).item()
            action = ['u', 'd', 'l', 'r'][action_idx]

            game.makeMove(action)
            reward = game.reward()
            total_reward += reward
            next_state = torch.tensor(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64)/10.0, dtype=torch.float32)

            with torch.no_grad():
                if is_double:
                    next_q = model(next_state)
                    next_action = torch.argmax(next_q).item()
                    target_q = target_model(next_state)
                    max_q = target_q[0][next_action]
                else:
                    target_q = target_model(next_state)
                    max_q = torch.max(target_q)

                y = reward if reward != -1 else reward + gamma * max_q

            y = torch.tensor([y], dtype=torch.float32)
            x = qval[0][action_idx]

            loss = loss_fn(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            state = next_state

            if reward != -1:
                done = True

        rewards.append(total_reward)
        if epsilon > 0.1:
            epsilon -= 1 / epochs

        if is_double and i % 10 == 0:
            target_model.load_state_dict(model.state_dict())

    return rewards

# ===================== 執行訓練 =====================
# 原始 DQN
dqn = DQN()
rewards_dqn = train(dqn, dqn, is_double=False)

# Double DQN
double_dqn = DQN()
double_dqn_target = DQN()
double_dqn_target.load_state_dict(double_dqn.state_dict())
rewards_double = train(double_dqn, double_dqn_target, is_double=True)

# Dueling Double DQN
dueling_dqn = DuelingDQN()
dueling_dqn_target = DuelingDQN()
dueling_dqn_target.load_state_dict(dueling_dqn.state_dict())
rewards_dueling = train(dueling_dqn, dueling_dqn_target, is_double=True, dueling=True)

# ===================== 繪製 Performance 比較圖表 =====================
plt.figure(figsize=(12, 6))
plt.plot(rewards_dqn, label='DQN')
plt.plot(rewards_double, label='Double DQN')
plt.plot(rewards_dueling, label='Dueling Double DQN')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Comparison of DQN Variants')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

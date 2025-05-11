import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from Gridworld import Gridworld
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class DQNModel(nn.Module):
    def __init__(self):
        super(DQNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )

    def forward(self, x):
        return self.model(x)

class DQNLightning(pl.LightningModule):
    def __init__(self, gamma=0.9, lr=1e-3, epsilon=1.0, epsilon_min=0.1, epsilon_decay=1e-3):
        super().__init__()
        self.model = DQNModel()
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        self.losses = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9),
            'interval': 'epoch'
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        game = Gridworld(size=4, mode='random')
        state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.tensor(state_, dtype=torch.float32)
        status = 1
        episode_loss = 0

        while status == 1:
            qval = self.model(state)
            qval_np = qval.detach().numpy()

            if random.random() < self.epsilon:
                action_idx = np.random.randint(0, 4)
            else:
                action_idx = np.argmax(qval_np)

            action = self.action_set[action_idx]
            game.makeMove(action)

            next_state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            next_state = torch.tensor(next_state_, dtype=torch.float32)
            reward = game.reward()

            with torch.no_grad():
                next_qval = self.model(next_state)
            maxQ = torch.max(next_qval)

            Y = reward + self.gamma * maxQ if reward == -1 else reward
            Y = torch.tensor([Y], dtype=torch.float32)

            X = qval.squeeze()[action_idx]
            loss = self.loss_fn(X, Y)
            episode_loss += loss.item()

            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.trainer.optimizers[0].step()
            self.trainer.optimizers[0].zero_grad()

            state = next_state
            if reward != -1:
                status = 0

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        self.losses.append(episode_loss)
        self.log('train_loss', episode_loss)
        return episode_loss

    def train_dataloader(self):
        # Dummy DataLoader just to call training_step multiple times
        return DataLoader([0] * 1000, batch_size=1)

    def plot_losses(self):
        plt.plot(self.losses)
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("DQN Lightning Training Loss")
        plt.savefig("dqn_lightning_loss.png")
        plt.show()

if __name__ == '__main__':
    model = DQNLightning()
    trainer = pl.Trainer(max_epochs=1000, enable_progress_bar=True, log_every_n_steps=1)
    trainer.fit(model)
    model.plot_losses()

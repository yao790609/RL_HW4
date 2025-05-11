# 📝 4-1 DQN + Experience Replay - Understanding Report

## ✅ Assignment Objectives

### Task Checklist
- ✅ Run the provided code (naive or with replay buffer)
- ✅ Discuss with ChatGPT to clarify code understanding
- ✅ Submit a short report including:
  - Basic DQN implementation for a simple environment
  - Use of Experience Replay Buffer

---

## 1️⃣ Code Execution and Understanding ✅

### 🧩 Environment and Model
- The environment is a custom `Gridworld` (4x4).
- Mode used: `'static'` (all elements are fixed).
- State input is a flattened 4x4 grid (1x64) with slight noise added for generalization.

### 🧠 Model Architecture
- A fully connected neural network:
  - Input: 64-dim (flattened state)
  - Hidden layers: 150 → 100 → 4 (Q-values for 4 actions)
  - Activation: ReLU

---

## 2️⃣ Key DQN Techniques 💡

### 🔹 ε-greedy Strategy (Exploration vs Exploitation)
- Random action with ε = 0.3 probability (exploration).
- Otherwise, choose the action with highest Q-value (exploitation).
- Balances discovering new actions and optimizing known good ones.

### 🔹 Experience Replay
- Implemented via `collections.deque` to store the most recent 1000 experiences:  
  `(state, action, reward, next_state, done)`
- During training, randomly sample 200 experiences (mini-batch) to break time correlation and improve diversity.

### 🔹 Target Network
- `model2` is a deep copy of the main model.
- Every `sync_freq = 500` steps, synchronize weights from the main model to the target model to stabilize learning.

### 🔹 Loss Function & Optimizer
- Loss is calculated using `MSELoss`:
L = MSE(Q(s, a), r + γ * max_a' Q_target(s'))

- Optimizer: Adam with learning rate = 1e-3

---

## 3️⃣ Training Loop Overview 🔁

### 🔄 Naive or Replay DQN Training Steps:
1. Initialize the Gridworld environment.
2. Obtain the current state `state1` with added noise.
3. Use the model to predict Q-values and select an action.
4. Execute the action, receive `reward` and `state2`.
5. Save the experience to the replay memory.
6. If experience buffer ≥ 200:
 - Randomly sample a batch
 - Compute target Q-values
 - Calculate loss and backpropagate
7. Every 500 steps, sync model weights to target network.
8. End episode when reward ≠ -1 or steps > 50, then reset environment.

---

## 4️⃣ Key Concepts Discussed with ChatGPT 📌

| Topic | Key Insights |
|-------|--------------|
| `render_np()` output | Although rendered every time, `state2` is rendered **after** `makeMove()`, ensuring it's the result of the action taken |
| Noise | Added via `+ np.random.rand()/10.0` for robustness, without significantly altering the state |
| `state_` vs `state2` | `state_` is the current state before action, `state2` is the result of the action (temporal continuity) |
| Why add noise | Small noise helps avoid overfitting to exact pixel values |
| Purpose of `model2` | Acts as a target network to stabilize learning by decoupling target calculation |
| Replay Buffer purpose | Prevents overfitting to recent samples, improves convergence through better sample diversity |

---

## ✅ Summary

This assignment demonstrates how to implement a DQN agent on a simple Gridworld environment, transitioning from naive to experience replay-enhanced training.

You should now understand:

- How to construct and train a basic DQN network
- The need for experience replay and a target network
- How ε-greedy helps with exploration
- How to stabilize and improve training with proper sample handling and network updates

---


# 4-2 Reinforcement Learning Model Comparison: DQN / Double DQN / Dueling Double DQN


This project compares the performance of three reinforcement learning models on a 4x4 GridWorld environment:

- **DQN**: Basic Deep Q-Network.
- **Double DQN**: Adds a separate target network to prevent Q-value overestimation.
- **Dueling Double DQN**: Further separates state value and advantage function for better stability and performance.

---

## 🧠 Model Architecture Comparison

| Model Type         | Core Structure                                         | Advantages                                    |
|--------------------|--------------------------------------------------------|-----------------------------------------------|
| DQN                | Single Q-network that directly outputs all Q-values    | Simple architecture, suitable for beginners   |
| Double DQN         | Uses target Q-network to decouple action selection and evaluation | Reduces Q-value overestimation        |
| Dueling Double DQN | Splits into two streams: state value and advantage     | Improves learning stability and convergence   |

> 📌 You may include a model architecture diagram at `docs/images/architecture.png` and embed it like this: `![Model Architecture](docs/images/architecture.png)`

---

## 📊 Model Training Performance Comparison (Total Episodes: 1000)

| Model              | Avg. Loss (Last 100 Episodes) | Convergence Speed (Approx.) | Notes                            |
|--------------------|-------------------------------|------------------------------|----------------------------------|
| **DQN**            | ~0.025                        | Around episode 650           | High variance, often suboptimal  |
| **Double DQN**     | ~0.015                        | Around episode 500           | More stable and faster convergence |
| **Dueling DQN**    | ~0.010                        | Around episode 400           | Most stable with smooth learning |

---

## 📈 Training Loss Curve

Below is the comparison curve for all three models:

![Training Comparison](https://github.com/yao790609/RL_HW4/blob/main/HW4-2.jpg)

---

## 📌  Requirements

- Python 3.8+
- PyTorch 2.0+
- matplotlib



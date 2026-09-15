# 🌿 Grid Explorer — Q-Learning Maze Solver

Grid Explorer is an interactive **Reinforcement Learning** project where an AI agent 🐝 learns to navigate a grid-based maze and find a path from the **Start 🟢** to the **Goal 🌸** using **Tabular Q-Learning**.

The project allows users to experiment with different maze difficulties, create custom mazes, train the agent, and visualize its learning process.

---

## ✨ Features

- 🧠 Tabular Q-Learning implemented from scratch
- 🗺️ Easy, Medium, and Hard maze environments
- 🧩 Custom maze creation
- 🎬 Visual AI learning replay
- 📊 Q-table visualization
- 🧭 Learned policy visualization
- 📈 Training performance visualization
- ⚙️ Adjustable Q-Learning parameters
- 🌿 Interactive Streamlit interface

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **NumPy**
- **Pandas**
- **HTML / CSS / JavaScript**

---

## 🧠 How It Works

The AI agent learns through **trial and error**.

At each step, the agent:

1. Observes its current state.
2. Selects an action.
3. Receives a reward or penalty.
4. Moves to a new state.
5. Updates its Q-value.
6. Repeats the process until it learns an effective path.

The learned Q-table is then used to determine the best action for each state.

### Q-Learning Update

```text
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]
```

Where:

- `s` → Current state
- `a` → Action
- `r` → Reward
- `s'` → Next state
- `α` → Learning rate
- `γ` → Discount factor

---

## 🗺️ Maze Environment

The maze is represented as a **7 × 7 grid**.

```text
S . . . . . .
. # # # . . .
. . . # . . .
. # # # . # .
. . . . . # .
. # # # # # .
. . . . . . G
```

### Symbols

| Symbol | Meaning |
|---|---|
| 🟢 `S` | Start |
| 🌸 `G` | Goal |
| 🪨 `#` | Wall |
| 🌿 `.` | Open path |
| 🐝 | AI Agent |

The agent can move **up, down, left, or right**.

---

## 🎮 Maze Modes

### 🌱 Preset Mazes

The application provides three difficulty levels:

- **Easy** — fewer obstacles and simpler routes
- **Medium** — more barriers and exploration
- **Hard** — denser obstacles and more challenging routes

Each difficulty contains multiple maze environments.

### 🧩 Custom Maze

Users can create their own maze by selecting:

- Start position
- Goal position
- Obstacles

The custom maze is checked for a valid route before training.

---

## ⚙️ Q-Learning Parameters

The training process can be customized using parameters such as:

- **Learning Rate (α)** — controls how quickly the agent learns from new information.
- **Discount Factor (γ)** — controls the importance of future rewards.
- **Training Episodes** — controls how many times the agent interacts with the environment.
- **Maximum Steps** — limits the number of steps per episode.
- **Epsilon Decay** — controls the transition from exploration to exploitation.

---

## 🎬 AI Learning Replay

Grid Explorer includes a visual replay of the agent's learning process.

The 🐝 agent explores the maze during training and gradually learns how to reach the 🌸 goal.

This provides a visual representation of how an RL agent improves through repeated interaction with its environment.

---

## 📁 Project Structure

```text
WhimsyGrid/
├── app.py
├── q_learning.py
├── maze_environment.py
├── mazes.py
├── requirements.txt
└── README.md
```

### File Overview

| File | Description |
|---|---|
| `app.py` | Main Streamlit application and interface |
| `q_learning.py` | Q-Learning agent implementation |
| `maze_environment.py` | Maze environment, movement, rewards, and states |
| `mazes.py` | Preset maze definitions |
| `requirements.txt` | Required Python dependencies |
| `README.md` | Project documentation |

---


---

## 🔮 Future Improvements

Possible extensions include:

- 🗺️ Larger maze environments
- 🎲 Random maze generation
- 🤖 Comparison with SARSA
- 🧠 Deep Q-Networks (DQN)
- 🚧 Dynamic obstacles
- 👥 Multiple agents
- 📊 More detailed training analysis

---

## 🌿 Project Goal

The goal of Grid Explorer is to provide a simple and interactive way to understand **how reinforcement learning agents learn to make decisions through experience**.

Instead of being given the correct path, the 🐝 agent discovers it through exploration, rewards, and repeated learning.

---

### 🌿 Learn. Explore. Navigate. 🐝

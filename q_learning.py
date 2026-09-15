import numpy as np


class QLearningAgent:
    """Tabular Q-learning agent for the WhimsyGrid maze environment."""

    def __init__(
        self,
        environment,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ):
        self.environment = environment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros(
            (environment.num_states, environment.num_actions),
            dtype=float,
        )

    # ------------------------------------------------------------
    # Environment compatibility helper
    # ------------------------------------------------------------
    def _step(self, state, action):
        """Support both common MazeEnvironment.step signatures.

        Some versions use step(action), while others use
        step(state, action). This keeps the Q-learning agent compatible
        with either version without changing the maze display.
        """
        try:
            return self.environment.step(action)
        except TypeError as first_error:
            try:
                return self.environment.step(state, action)
            except TypeError:
                raise first_error

    # ------------------------------------------------------------
    # Epsilon-greedy action selection
    # ------------------------------------------------------------
    def choose_action(self, state_index):
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.environment.num_actions))
        return self.choose_best_action(state_index)

    def choose_best_action(self, state_index):
        values = self.q_table[state_index]
        best_actions = np.flatnonzero(values == np.max(values))
        return int(np.random.choice(best_actions))

    # ------------------------------------------------------------
    # Q-learning training
    # ------------------------------------------------------------
    def train(self, episodes=1000, max_steps=200, callback=None):
        rewards = []
        steps_history = []
        successful_episodes = 0
        episode_trajectories = {}

        # Keep a small set of representative episodes for the learning replay.
        replay_episodes = set(np.unique(np.linspace(1, episodes, min(10, episodes), dtype=int)).tolist())

        for episode in range(1, episodes + 1):
            state = self.environment.reset()
            state = tuple(state)

            total_reward = 0.0
            reached_goal = False
            steps_taken = 0
            trajectory = [state]

            for step in range(1, max_steps + 1):
                state_index = self.environment.state_to_index(state)
                action = self.choose_action(state_index)

                next_state, reward, done = self._step(state, action)
                next_state = tuple(next_state)
                next_index = self.environment.state_to_index(next_state)

                # Q-learning update:
                # Q(s,a) <- Q(s,a) + alpha *
                # [r + gamma * max Q(s',a') - Q(s,a)]
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * np.max(self.q_table[next_index])

                self.q_table[state_index, action] += self.alpha * (
                    target - self.q_table[state_index, action]
                )

                total_reward += reward
                steps_taken = step
                state = next_state
                trajectory.append(state)

                if done:
                    reached_goal = True
                    successful_episodes += 1
                    break

            rewards.append(total_reward)
            steps_history.append(steps_taken)

            if episode in replay_episodes:
                episode_trajectories[episode] = trajectory

            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay,
            )

            if callback is not None:
                try:
                    callback(
                        episode,
                        total_reward,
                        steps_taken,
                        self.epsilon,
                        reached_goal,
                        trajectory,
                    )
                except TypeError as error:
                    # Preserve compatibility with callbacks using the old 5-argument form.
                    if "positional argument" in str(error) or "required positional" in str(error):
                        callback(
                            episode,
                            total_reward,
                            steps_taken,
                            self.epsilon,
                            reached_goal,
                        )
                    else:
                        raise

        return {
            "rewards": rewards,
            "steps": steps_history,
            "successful_episodes": successful_episodes,
            "success_rate": (
                successful_episodes / episodes if episodes else 0.0
            ),
            "episode_trajectories": episode_trajectories,
        }

    # ------------------------------------------------------------
    # Greedy learned path
    # ------------------------------------------------------------
    def get_learned_path(self):
        state = tuple(self.environment.reset())
        path = [state]
        visited = {state}
        goal = tuple(self.environment.find_symbol("G"))

        for _ in range(self.environment.num_states * 2):
            if state == goal:
                break

            state_index = self.environment.state_to_index(state)
            action_order = np.argsort(self.q_table[state_index])[::-1]
            moved = False

            for action in action_order:
                candidate, reward, done = self._step(state, int(action))
                candidate = tuple(candidate)

                # Prefer reaching the goal immediately.
                if done and candidate == goal:
                    state = candidate
                    path.append(candidate)
                    moved = True
                    break

                # Ignore walls, boundaries and loops.
                if candidate != state and candidate not in visited:
                    state = candidate
                    path.append(candidate)
                    visited.add(candidate)
                    moved = True
                    break

            if not moved:
                break

        return path

    # ------------------------------------------------------------
    # Human-readable path actions
    # ------------------------------------------------------------
    def get_path_actions(self):
        path = self.get_learned_path()
        names = []

        direction_names = {
            (-1, 0): "↑ Up",
            (1, 0): "↓ Down",
            (0, -1): "← Left",
            (0, 1): "→ Right",
        }

        for current, nxt in zip(path, path[1:]):
            dr = nxt[0] - current[0]
            dc = nxt[1] - current[1]
            names.append(direction_names.get((dr, dc), "?"))

        return names

    # ------------------------------------------------------------
    # Learned policy
    # ------------------------------------------------------------
    def get_policy(self):
        policy = np.full(
            (self.environment.rows, self.environment.cols),
            "",
            dtype=object,
        )

        symbols = {
            0: "↑",
            1: "↓",
            2: "←",
            3: "→",
        }

        for r in range(self.environment.rows):
            for c in range(self.environment.cols):
                value = self.environment.maze[r, c]

                if value == "#":
                    policy[r, c] = "🪨"
                elif value == "G":
                    policy[r, c] = "🌸"
                elif value == "S":
                    policy[r, c] = "🟢"
                else:
                    idx = self.environment.state_to_index((r, c))
                    policy[r, c] = symbols[self.choose_best_action(idx)]

        return policy

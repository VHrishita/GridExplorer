import numpy as np
from collections import deque


class MazeEnvironment:
    """
    Custom Gridworld environment for tabular Q-learning.

    Symbols:
        S = Start
        G = Goal
        # = Obstacle
        . = Walkable cell
    """

    ACTIONS = {
        0: (-1, 0),   # Up
        1: (1, 0),    # Down
        2: (0, -1),   # Left
        3: (0, 1)     # Right
    }

    ACTION_NAMES = {
        0: "Up",
        1: "Down",
        2: "Left",
        3: "Right"
    }

    ACTION_SYMBOLS = {
        0: "↑",
        1: "↓",
        2: "←",
        3: "→"
    }

    def __init__(self, maze):
        self.maze = np.array(maze, dtype=str)

        self.rows, self.cols = self.maze.shape

        self.start_state = self.find_symbol("S")
        self.goal_state = self.find_symbol("G")

        self.num_states = self.rows * self.cols
        self.num_actions = 4

    # ---------------------------------------------------------
    # Find a symbol in the maze
    # ---------------------------------------------------------

    def find_symbol(self, symbol):

        positions = np.argwhere(self.maze == symbol)

        if len(positions) == 0:
            return None

        row, col = positions[0]

        return (int(row), int(col))

    # ---------------------------------------------------------
    # Convert state to Q-table index
    # ---------------------------------------------------------

    def state_to_index(self, state):

        row, col = state

        return row * self.cols + col

    # ---------------------------------------------------------
    # Convert Q-table index back to state
    # ---------------------------------------------------------

    def index_to_state(self, index):

        row = index // self.cols
        col = index % self.cols

        return (row, col)

    # ---------------------------------------------------------
    # Check whether a position is valid
    # ---------------------------------------------------------

    def is_valid_position(self, row, col):

        if row < 0 or row >= self.rows:
            return False

        if col < 0 or col >= self.cols:
            return False

        if self.maze[row, col] == "#":
            return False

        return True

    # ---------------------------------------------------------
    # Take an action
    # ---------------------------------------------------------

    def step(self, state, action):

        row, col = state

        dr, dc = self.ACTIONS[action]

        new_row = row + dr
        new_col = col + dc

        # Boundary or obstacle
        if not self.is_valid_position(new_row, new_col):

            return state, -5, False

        next_state = (new_row, new_col)

        # Goal
        if next_state == self.goal_state:

            return next_state, 100, True

        # Normal movement
        return next_state, -1, False

    # ---------------------------------------------------------
    # Reset environment
    # ---------------------------------------------------------

    def reset(self):

        return self.start_state

    # ---------------------------------------------------------
    # Find whether a path exists using BFS
    # ---------------------------------------------------------

    def is_solvable(self):

        if self.start_state is None or self.goal_state is None:
            return False

        queue = deque([self.start_state])

        visited = {self.start_state}

        while queue:

            current = queue.popleft()

            if current == self.goal_state:
                return True

            row, col = current

            for dr, dc in self.ACTIONS.values():

                new_row = row + dr
                new_col = col + dc

                next_state = (new_row, new_col)

                if (
                    self.is_valid_position(new_row, new_col)
                    and next_state not in visited
                ):

                    visited.add(next_state)

                    queue.append(next_state)

        return False

    # ---------------------------------------------------------
    # Get shortest possible path using BFS
    # ---------------------------------------------------------

    def shortest_path_bfs(self):

        if not self.is_solvable():
            return []

        queue = deque([self.start_state])

        parent = {
            self.start_state: None
        }

        while queue:

            current = queue.popleft()

            if current == self.goal_state:
                break

            row, col = current

            for dr, dc in self.ACTIONS.values():

                next_state = (
                    row + dr,
                    col + dc
                )

                if (
                    self.is_valid_position(*next_state)
                    and next_state not in parent
                ):

                    parent[next_state] = current

                    queue.append(next_state)

        if self.goal_state not in parent:
            return []

        path = []

        current = self.goal_state

        while current is not None:

            path.append(current)

            current = parent[current]

        path.reverse()

        return path
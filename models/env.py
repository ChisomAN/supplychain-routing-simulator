import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx


class RoutingEnv(gym.Env):
    """Minimal toy env: agent moves along nodes to reach target. Reward = -edge distance."""
    metadata = {"render_modes": []}

    def __init__(self, G: nx.Graph, start: int | None = None, goal: int | None = None):
        super().__init__()
        self.G = G
        self.nodes = list(G.nodes())
        self.node_index = {n: i for i, n in enumerate(self.nodes)}
        self.start = self.nodes[0] if start is None else start
        self.goal = self.nodes[-1] if goal is None else goal
        self.current = self.start
        self.action_space = spaces.Discrete(len(self.nodes))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.nodes),), dtype=np.float32)
        self.max_steps = len(self.nodes) * 2
        self.steps = 0

    def _obs(self):
        obs = np.zeros(len(self.nodes), dtype=np.float32)
        obs[self.node_index[self.current]] = 1.0
        obs[self.node_index[self.goal]] = 1.0
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current = self.start
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        next_node = self.nodes[action % len(self.nodes)]
        if self.G.has_edge(self.current, next_node):
            cost = float(self.G[self.current]
                         [next_node].get("distance_km", 1.0))
            self.current = next_node
            reward = -cost
        else:
            reward = -5.0
        terminated = self.current == self.goal
        truncated = self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}

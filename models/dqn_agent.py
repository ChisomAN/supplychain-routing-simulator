from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import tempfile
import os
import json


LOG_DIR = os.path.join("artifacts", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
REWARD_LOG_PATH = os.path.join(LOG_DIR, "dqn_reward_history.json")


class SaveEveryCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
        return True


class RewardLoggerCallback(BaseCallback):
    """
    Logs (timesteps, mean_episode_reward) at the end of each rollout.
    """

    def __init__(self, buffer, verbose=0):
        super().__init__(verbose)
        self.buffer = buffer

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            mean_r = float(
                sum(info["r"] for info in self.model.ep_info_buffer)
                / len(self.model.ep_info_buffer)
            )
            self.buffer.append((int(self.num_timesteps), mean_r))
        return True


def train_dqn(
    env,
    total_timesteps: int = 2000,
    learning_rate: float = 5e-4,
    batch_size: int = 64,
    gamma: float = 0.99,
    exploration_fraction: float = 0.1,
    save_path: str | None = None,
) -> str:
    """
    Train a DQN agent and:
      - save the model (path returned)
      - write reward history to artifacts/logs/dqn_reward_history.json
    """
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        verbose=0,
    )

    out_path = save_path or tempfile.mktemp(suffix="_dqn_model.zip")

    # buffer to collect (timesteps, mean_reward) during learning
    reward_history = []

    callbacks = CallbackList([
        SaveEveryCallback(save_freq=500, save_path=out_path),
        RewardLoggerCallback(reward_history),
    ])

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(out_path)

    # persist reward history for the Streamlit app
    try:
        with open(REWARD_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(
                [{"timesteps": int(t), "mean_reward": float(r)} for t, r in reward_history],
                f,
            )
    except Exception:
        # fail silently if logging can't be written; training still succeeded
        pass

    return out_path


def infer_dqn(model_path: str, env, episodes: int = 5) -> dict:
    model = DQN.load(model_path, env=env)
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, trunc = False, False
        ep_rew = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(int(action))
            ep_rew += float(r)
        rewards.append(ep_rew)
    return {"episodes": episodes, "mean_reward": float(sum(rewards) / len(rewards))}

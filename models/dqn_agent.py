from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import tempfile


class SaveEveryCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
        return True


def train_dqn(env, total_timesteps: int = 2000, learning_rate: float = 5e-4,
              batch_size: int = 64, gamma: float = 0.99, exploration_fraction: float = 0.1,
              save_path: str | None = None) -> str:
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
    cb = SaveEveryCallback(save_freq=500, save_path=out_path)
    model.learn(total_timesteps=total_timesteps, callback=cb)
    model.save(out_path)
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

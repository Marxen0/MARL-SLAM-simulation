import time
import numpy as np
import torch
import torch.nn as nn

from helpers.map_generator import OccupancyViewer
import Environment


# =========================
# ACTOR
# =========================
class Actor(nn.Module):
    def __init__(self, obs_shape=(5, 6), action_dim=5):
        super().__init__()

        input_dim = obs_shape[0] * obs_shape[1]

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


def test():

    AGENTS = 3
    RAYS = 16
    W, H = 50, 50

    env = Environment.environment(AGENTS, RAYS, W, H)

    actor = Actor()

    actor.load_state_dict(
        torch.load("models2/actor_final.pth", map_location="cpu")
    )

    actor.eval()

    viewer = OccupancyViewer(
        env.world_widht,
        env.world_height,
        cell_size=20
    )

    obs = env.reset()
    done = False

    total_reward = 0

    while not done:

        actions = np.zeros(env.ag_num, dtype=int)

        for i in range(env.ag_num):

            if len(env.ag_paths[i]) == 0:

                with torch.no_grad():

                    o = torch.FloatTensor(obs[i]).unsqueeze(0)

                    logits = actor(o)

                    action = torch.argmax(logits, dim=1)

                    actions[i] = action.item()

            else:
                actions[i] = -1

        obs, rewards, done = env.step(actions)

        total_reward += np.sum(rewards)

        viewer.render(
            env.ag_occ[0],
            env.ag_pos,
            obs[0],
        )

        time.sleep(0.5)

    print(f"Finished! Reward: {total_reward}")
    print(f"Steps: {env.time}")


if __name__ == "__main__":
    test()
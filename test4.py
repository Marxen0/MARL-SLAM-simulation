import time
import numpy as np
import torch
import torch.nn as nn

import Environment

from helpers.map_generator import OccupancyViewer
from observation_encoder import encode_observation


# ==================================================
# ACTOR
# ==================================================

class Actor(nn.Module):

    def __init__(self, input_dim, action_dim=5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ==================================================
# TEST
# ==================================================

def test():

    # -----------------------------
    # SETTINGS
    # -----------------------------

    AGENTS = 3
    RAYS = 120

    W = 50
    H = 50

    VERSION = "Version 23 Training"

    # Choose ONE of these

    MODEL_PATH = f"{VERSION}/models/actor_final.pth"

    # MODEL_PATH = f"{VERSION}/models/checkpoint_4900.pth"

    RENDER = True

    # -----------------------------
    # ENVIRONMENT
    # -----------------------------

    env = Environment.environment(
        AGENTS,
        RAYS,
        W,
        H,
        estimate_grid_size=4
    )

    obs, action_masks = env.reset()

    sample_input = encode_observation(
        obs[0],
        env.time
    )

    input_dim = len(sample_input)

    print("Actor input dim:", input_dim)

    actor = Actor(input_dim)

    # -----------------------------
    # LOAD MODEL
    # -----------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    if isinstance(checkpoint, dict) and "actor_state_dict" in checkpoint:

        actor.load_state_dict(
            checkpoint["actor_state_dict"]
        )

    else:

        actor.load_state_dict(checkpoint)

    actor.eval()

    print("Loaded:", MODEL_PATH)

    # -----------------------------
    # VIEWER
    # -----------------------------

    if RENDER:

        viewer = OccupancyViewer(
            env.world_widht,
            env.world_height,
            cell_size=20
        )

    # -----------------------------
    # RUN EPISODE
    # -----------------------------

    done = False

    episode_reward = 0

    while not done:

        agents_need_action = [

            i

            for i in range(env.ag_num)

            if len(env.ag_paths[i]) == 0

        ]

        actions = np.full(
            env.ag_num,
            -1,
            dtype=int
        )

        # -------------------------------------
        # ACTION SELECTION
        # -------------------------------------

        for i in agents_need_action:

            encoded_obs = encode_observation(
                obs[i],
                env.time
            )

            with torch.no_grad():

                logits = actor(

                    torch.FloatTensor(
                        encoded_obs
                    ).unsqueeze(0)

                )

            for a in range(5):

                if not action_masks[i][a]:

                    logits[0, a] = -float("inf")

            if torch.all(torch.isinf(logits)):

                action = 0

            else:

                action = torch.argmax(
                    logits,
                    dim=1
                ).item()

            actions[i] = action

        # -------------------------------------
        # STEP
        # -------------------------------------

        (

            obs,

            action_masks,

            rewards,

            done,

            finished_agents,

        ) = env.step(actions)

        episode_reward += np.sum(rewards)

        if RENDER:

            viewer.render(
                env.global_map,
                env.ag_pos
            )

            time.sleep(0.05)

    # -----------------------------
    # RESULT
    # -----------------------------

    print()
    print("=" * 40)
    print("Episode Finished")
    print("=" * 40)
    print(f"Reward : {episode_reward:.2f}")
    print(f"Steps  : {env.time}")
    print("=" * 40)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    test()
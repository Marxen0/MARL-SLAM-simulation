import os
import csv
import numpy as np
import torch

import Environment
from observation_encoder import encode_observation
from train4 import Actor      # wherever your Actor class is located


# ==========================================
# SETTINGS
# ==========================================

AGENTS = 3
RAYS = 120

W = 50
H = 50

EPISODES = 1000

MODEL_PATH = "Version 17 Training/models/checkpoint_2500.pth"
LOG_FILE = "Version 17 Training/test_results.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

env = Environment.environment(
    AGENTS,
    RAYS,
    W,
    H,
    estimate_grid_size=12
)

obs, masks = env.reset()

input_dim = len(
    encode_observation(
        obs[0],
        env.time
    )
)

actor = Actor(input_dim=input_dim).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if "actor_state_dict" in checkpoint:
    actor.load_state_dict(checkpoint["actor_state_dict"])
    print(f"Loaded checkpoint from episode {checkpoint['episode']}")
else:
    actor.load_state_dict(checkpoint)
    print("Loaded final actor model")

actor.eval()

actor.eval()

print("Loaded model.")


# ==========================================
# CREATE CSV
# ==========================================

with open(LOG_FILE, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "episode",
        "reward",
        "steps",
        "exploration_reward",
        "decision_count"
    ])


# ==========================================
# TEST LOOP
# ==========================================

for ep in range(EPISODES):

    obs, action_masks = env.reset()

    done = False

    episode_reward = 0
    episode_exploration = 0
    episode_decision = 0

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

        # -------------------------
        # Greedy action selection
        # -------------------------

        with torch.no_grad():

            for i in agents_need_action:

                encoded = encode_observation(
                    obs[i],
                    env.time
                )

                x = torch.FloatTensor(encoded).unsqueeze(0).to(DEVICE)

                logits = actor(x)

                for a in range(5):

                    if not action_masks[i][a]:

                        logits[0, a] = -float("inf")

                if torch.all(torch.isinf(logits)):

                    actions[i] = 0

                else:

                    actions[i] = torch.argmax(
                        logits,
                        dim=1
                    ).item()

        # -------------------------
        # Update paths
        # -------------------------

        for i in agents_need_action:

            frontier = obs[i]["frontiers"][actions[i]]

            env.ag_paths[i] = frontier["cached_path"]

            env.ag_target[i] = frontier["frontier_position"]

        # -------------------------
        # Step environment
        # -------------------------

        obs, action_masks, rewards, done, exploration_reward = env.step(
            actions
        )

        episode_reward += np.sum(rewards)

        episode_exploration += exploration_reward

        episode_decision += 1

    print(
        f"Episode {ep:3d} | "
        f"Reward {episode_reward:8.2f} | "
        f"Steps {env.time:4d} | "
        f"Exploration {episode_exploration}"
    )

    with open(LOG_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            ep,
            episode_reward,
            env.time,
            episode_exploration,
            episode_decision
        ])

print("Testing complete.")
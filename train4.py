import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
import time

import Environment

from helpers.map_generator import OccupancyViewer
from observation_encoder import encode_observation


# =========================
# ACTOR
# =========================

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


# =========================
# CRITIC
# =========================

class Critic(nn.Module):

    def __init__(self, state_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(state_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, x):

        return self.net(x).squeeze(-1)


# =========================
# GLOBAL STATE
# =========================

from helpers.frontier_finder import (
    get_direction,
    frontier_information_gain
)
def compute_unknown_density_features(global_map, unknown_value=-1, grid_size=4):
    """
    Compute unknown-cell density features.

    Returns:
        List containing:
        - global unknown density
        - unknown density for each grid section (row-major order)
    """
    features = []

    h, w = global_map.shape

    # Global unknown density
    unknown_count = np.sum(global_map == unknown_value)
    features.append(unknown_count / (h * w))

    cell_h = h // grid_size
    cell_w = w // grid_size

    for gy in range(grid_size):
        for gx in range(grid_size):
            y0 = gy * cell_h
            y1 = h if gy == grid_size - 1 else (gy + 1) * cell_h

            x0 = gx * cell_w
            x1 = w if gx == grid_size - 1 else (gx + 1) * cell_w

            section = global_map[y0:y1, x0:x1]
            features.append(np.mean(section == unknown_value))

    return features

def get_global_state(env, current_agent, grid_size=4, unknown_value=-1):

    features = []

    # ==========================================
    # GLOBAL MAP FEATURES
    # ==========================================
    global_map = env.global_map
    h, w = global_map.shape

    # Global unknown density
    unknown_count = np.sum(global_map == unknown_value)
    features.append(unknown_count / (h * w))

    cell_h = h // grid_size
    cell_w = w // grid_size
    features = compute_unknown_density_features(global_map, unknown_value, grid_size)
    agent_order = [current_agent]

    for i in range(env.ag_num):

        if i != current_agent:

            agent_order.append(i)

    # ==========================================
    # PATH INFORMATION
    # ==========================================

    path_lengths = [

        len(path)
        for path in env.ag_paths

    ]

    max_dist = h + w

    features.append(min(path_lengths) / max_dist)

    features.extend([
        p / max_dist
        for p in path_lengths
])

    # ==========================================
    # AGENT FEATURES
    # ==========================================

    for i in agent_order:

        x, y = env.ag_pos[i]

        # ------------------------------
        # Position
        # ------------------------------

        features.extend([
            x / w,
            y / h
        ])

        # ------------------------------
        # Which 3x3 sector?
        # ------------------------------

        sector_x = min(2, int(x / cell_w))
        sector_y = min(2, int(y / cell_h))

        features.extend([
            sector_x,
            sector_y
        ])

        # ------------------------------
        # Target direction
        # ------------------------------

        if env.ag_target[i] is not None:

            dx, dy = get_direction(
                env.ag_pos[i],
                env.ag_target[i]
            )

        else:

            dx, dy = 0, 0

        features.extend([dx, dy])

        # ------------------------------
        # Frontier value
        # ------------------------------

        if env.ag_target[i] is not None:

            gain = frontier_information_gain(
                env.ag_target[i],
                env.global_map
            )

        else:

            gain = 0

        features.append(
            gain / (h * w)
        )

        # ------------------------------
        # Frontier sector
        # ------------------------------

        if env.ag_target[i] is not None:

            tx, ty = env.ag_target[i]

            target_sector_x = min(
                2,
                int(tx / cell_w)
            )

            target_sector_y = min(
                2,
                int(ty / cell_h)
            )

        else:

            target_sector_x = 0
            target_sector_y = 0

        features.extend([
            target_sector_x,
            target_sector_y
        ])

        # ------------------------------
        # Distance to other agents
        # ------------------------------

        features.extend([
            d / (h + w)
            for d in env.observation[i]["other_agent_dijkstra"]
        ])

    return np.array(
        features,
        dtype=np.float32
    )


# =========================
# TRAIN
# =========================

def train():

    AGENTS = 3
    RAYS = 120

    W = 50
    H = 50

    EPISODES = 5000

    GAMMA = 0.99

    RENDER = False
    VERSION = "Version 23 Training"

    MODEL_FOLDER = os.path.join(
        VERSION,
        "models"
    )

    os.makedirs(
        MODEL_FOLDER,
        exist_ok=True
    )
    LOG_FILE = os.path.join(
        VERSION,
        "training_log.csv"
    )

    env = Environment.environment(
        AGENTS,
        RAYS,
        W,
        H,
        estimate_grid_size=4
    )

    # --------------------------------
    # Get input size automatically
    # --------------------------------

    obs, masks = env.reset()

    sample_input = encode_observation(
        obs[0],
        env.time
    )

    input_dim = len(
        sample_input
    )

    print(
        "Actor input dim:",
        input_dim
    )

    actor = Actor(
        input_dim=input_dim
    )

    sample_state = get_global_state(env,0)

    print(
        "Critic input dim:",
        len(sample_state)
    )

    critic = Critic(
        state_dim=len(sample_state)
    )

    actor_opt = optim.Adam(
        actor.parameters(),
        lr=1e-4
    )

    critic_opt = optim.Adam(
        critic.parameters(),
        lr=5e-4
    )

    if RENDER:

        viewer = OccupancyViewer(
            env.world_widht,
            env.world_height,
            cell_size=20
        )

    print("Starting training...")

    # ==========================================
    # EPISODES
    # ==========================================
    
    for ep in range(EPISODES):
        episode_exploration = 0
        episode_decission = 0
        obs, action_masks = env.reset()

        done = False

        trajectory = []
        pending = [None for _ in range(env.ag_num)]

        ep_reward = 0

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

            # ==================================
            # ACTION SELECTION
            # ==================================

            for i in agents_need_action:

                encoded_obs = encode_observation(
                    obs[i],
                    env.time
                )

                o = torch.FloatTensor(
                    encoded_obs
                ).unsqueeze(0)

                logits = actor(o)

                for a in range(5):

                    if not action_masks[i][a]:

                        logits[0, a] = -float("inf")

                if torch.all(torch.isinf(logits)):

                    action = torch.tensor(0)
                    log_prob = torch.tensor(0.0)

                else:

                    dist = torch.distributions.Categorical(
                        logits=logits
                    )

                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                actions[i] = action.item()
                if pending[i] is not None:
                    raise RuntimeError(
                        f"Pending transition for agent {i} already exists."
                    )
                pending[i] = {

                    "agent": i,

                    "obs": torch.FloatTensor(
                        encoded_obs
                    ),

                    "global_state": torch.FloatTensor(
                        get_global_state(env, i)
                    ).unsqueeze(0),

                    "log_prob": log_prob,

                    "action": action.item()
                }
            
            # ==================================
            # ENVIRONMENT STEP
            # ==================================

            (
                next_obs,
                next_masks,
                rewards,
                done,
                finished_agents,
             )= env.step(actions)

            if RENDER:

                viewer.render(
                    env.ag_occ[0],
                    env.ag_pos
                )

                time.sleep(0.1)

            for agent in finished_agents:

                if pending[agent] is None:
                    continue

                transition = pending[agent]

                transition["next_global_state"] = torch.FloatTensor(
                    get_global_state(
                        env,
                        current_agent=agent
                    )
                ).unsqueeze(0)

                transition["reward"] = rewards[agent]

                transition["done"] = done

                trajectory.append(
                    transition
                )

                pending[agent] = None

                ep_reward += rewards[agent]

        obs = next_obs
        action_masks = next_masks
        # ==================================
        # A2C UPDATE
        # ==================================
        if len(trajectory) == 0:
            continue
        advantages = []
        actor_log_probs = []
        critic_losses = []

        for t in trajectory:

            value = critic(
                t["global_state"]
            )

            with torch.no_grad():

                next_value = critic(
                    t["next_global_state"]
                )

            reward = t["reward"]
            done = t["done"]

            target = reward + (

                GAMMA
                * next_value
                * (1 - done)

            )

            advantage = target - value

            advantages.append(
                advantage
            )

            actor_log_probs.append(
                t["log_prob"]
            )

            critic_losses.append(
                advantage.pow(2)
            )

        advantages = torch.stack(
            advantages
        ).squeeze()

        if len(advantages.shape) > 0 and advantages.numel() > 1:

            advantages = (

                advantages
                - advantages.mean()

            ) / (

                advantages.std()
                + 1e-8
            )

        actor_losses = []

        for log_prob, advantage in zip(
            actor_log_probs,
            advantages
        ):

            actor_losses.append(

                -log_prob
                * advantage.detach()

            )

        loss_actor = torch.stack(
            actor_losses
        ).mean()

        loss_critic = torch.stack(
            critic_losses
        ).mean()

        loss = (

            loss_actor
            + 0.5 * loss_critic

        )

        actor_opt.zero_grad()
        critic_opt.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            actor.parameters(),
            0.5
        )

        torch.nn.utils.clip_grad_norm_(
            critic.parameters(),
            0.5
        )

        actor_opt.step()
        critic_opt.step()

        # ==================================
        # LOGGING
        # ==================================

        if ep % 10 == 0:

            print(

                f"Episode {ep} | "
                f"Reward: {ep_reward:.2f} | "
                f"Steps: {env.time} |"
                f"episode exploration : {episode_exploration} |"
                f"episode decssion : {episode_decission}"

            )

        with open(
            LOG_FILE,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([

                ep,

                ep_reward,

                env.time

            ])

        if ep % 100 == 0:

            checkpoint = {

                "episode":
                    ep,

                "actor_state_dict":
                    actor.state_dict(),

                "critic_state_dict":
                    critic.state_dict(),

                "actor_optimizer_state_dict":
                    actor_opt.state_dict(),

                "critic_optimizer_state_dict":
                    critic_opt.state_dict()
            }

            torch.save(

                checkpoint,

                f"{MODEL_FOLDER}/checkpoint_{ep}.pth"

            )

            print(

                f"Saved {MODEL_FOLDER}/checkpoint_{ep}.pth"

            )

    torch.save(

        actor.state_dict(),

        f"{MODEL_FOLDER}/actor_final.pth"
    )

    torch.save(

        critic.state_dict(),

        f"{MODEL_FOLDER}/critic_final.pth"
    )

    print("Training complete.")


if __name__ == "__main__":

    train()
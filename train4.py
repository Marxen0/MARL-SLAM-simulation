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

def get_global_state(env):

    return np.concatenate([

        env.global_map.flatten(),

        np.array(
            env.ag_pos
        ).flatten()

    ])


# =========================
# TRAIN
# =========================

def train():

    AGENTS = 3
    RAYS = 16

    W = 50
    H = 50

    EPISODES = 200

    GAMMA = 0.99

    RENDER = False
    MODELSFOLDER = "models7"
    os.makedirs(
        MODELSFOLDER,
        exist_ok=True
    )

    env = Environment.environment(
        AGENTS,
        RAYS,
        W,
        H
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

    critic = Critic(
        state_dim=W * H + AGENTS * 2
    )

    actor_opt = optim.Adam(
        actor.parameters(),
        lr=1e-4
    )

    critic_opt = optim.Adam(
        critic.parameters(),
        lr=1e-3
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

        obs, action_masks = env.reset()

        done = False

        trajectory = []

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

            log_probs = {}

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

                # -----------------------------
                # MASK INVALID ACTIONS
                # -----------------------------

                for a in range(5):

                    if not action_masks[i][a]:

                        logits[0, a] = -float("inf")

                # Safety

                if torch.all(
                    torch.isinf(logits)
                ):

                    actions[i] = 0

                    log_probs[i] = torch.tensor(
                        0.0
                    )

                else:

                    dist = torch.distributions.Categorical(
                        logits=logits
                    )

                    action = dist.sample()

                    actions[i] = action.item()

                    log_probs[i] = dist.log_prob(
                        action
                    )

            global_state = torch.FloatTensor(
                get_global_state(env)
            ).unsqueeze(0)

            # ==================================
            # ENV STEP
            # ==================================

            next_obs, next_masks, rewards, done = env.step(
                actions
            )

            if RENDER:

                viewer.render(
                    env.ag_occ[0],
                    env.ag_pos
                )

                time.sleep(0.1)

            reward = np.sum(
                rewards
            )

            ep_reward += reward

            next_global_state = torch.FloatTensor(
                get_global_state(env)
            ).unsqueeze(0)

            # ==================================
            # STORE TRANSITIONS
            # ==================================

            for i in agents_need_action:

                encoded_obs = encode_observation(
                    obs[i],
                    env.time
                )

                trajectory.append({

                    "obs":
                        torch.FloatTensor(
                            encoded_obs
                        ),

                    "global_state":
                        global_state,

                    "next_global_state":
                        next_global_state,

                    "log_prob":
                        log_probs[i],

                    "reward":
                        reward,

                    "done":
                        done
                })

            obs = next_obs
            action_masks = next_masks

        # ==================================
        # A2C UPDATE
        # ==================================

        actor_losses = []

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

            actor_loss = (

                -t["log_prob"]
                * advantage.detach()

            )

            critic_loss = advantage.pow(
                2
            )

            actor_losses.append(
                actor_loss
            )

            critic_losses.append(
                critic_loss
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

        actor_opt.step()

        critic_opt.step()

        # ==================================
        # LOGGING
        # ==================================

        if ep % 10 == 0:

            print(

                f"Episode {ep} | "
                f"Reward: {ep_reward:.2f} | "
                f"Steps: {env.time}"

            )

        with open(
            "training_log9.csv",
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

                f"{MODELSFOLDER}/checkpoint_{ep}.pth"

            )

            print(

                f"Saved {MODELSFOLDER}/checkpoint_{ep}.pth"

            )

    torch.save(

        actor.state_dict(),

        f"{MODELSFOLDER}/actor_final.pth"
    )

    torch.save(

        critic.state_dict(),

        f"{MODELSFOLDER}/critic_final.pth"
    )

    print("Training complete.")


if __name__ == "__main__":

    train()
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from helpers.map_generator import OccupancyViewer
import time
import Environment
import os


# =========================
# ACTOR (decentralized)
# =========================
class Actor(nn.Module):
    def __init__(self, obs_shape=(5, 6), extra_dim=3, action_dim=5):
        super().__init__()

        # (5 * 6) + ((AGENTS - 1) distances) + (1 time feature)
        input_dim = obs_shape[0] * obs_shape[1] + extra_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# CENTRAL CRITIC (global)
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
        np.array(env.ag_pos).flatten()
    ])


# =========================
# TRAINING LOOP
# =========================
def train():

    AGENTS = 3
    RAYS = 16
    W, H = 50, 50
    EPISODES = 200
    GAMMA = 0.99
    RENDER = False

    # extra = (AGENTS-1 distances) + time
    extra_dim = (AGENTS - 1) + 1

    env = Environment.environment(AGENTS, RAYS, W, H)

    os.makedirs("models5", exist_ok=True)

    actor = Actor(extra_dim=extra_dim)
    critic = Critic(state_dim=W * H + AGENTS * 2)

    actor_opt = optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

    if RENDER:
        viewer = OccupancyViewer(
            env.world_widht,
            env.world_height,
            cell_size=20
        )

    print("Starting A2C training...")

    for ep in range(EPISODES):

        # extra example:
        # [
        #   [ag1->ag2, ag1->ag3],
        #   [ag2->ag1, ag2->ag3],
        #   [ag3->ag1, ag3->ag2]
        # ]
        obs, extra = env.reset()

        done = False
        trajectory = []
        ep_reward = 0

        while not done:
            agents_need_action = [
                i for i in range(env.ag_num)
                if len(env.ag_paths[i]) == 0
            ]

            actions = np.zeros(env.ag_num, dtype=int)
            log_probs = {}

            # =========================
            # ACTION SELECTION
            # =========================
            for i in range(env.ag_num):

                if i in agents_need_action:

                    # Add time as an extra feature
                    combined_extra = np.append(
                        extra[i],
                        env.time / 1000.0
                    )

                    combined_obs = np.concatenate([
                        obs[i].flatten(),
                        combined_extra
                    ])

                    o = torch.FloatTensor(
                        combined_obs
                    ).unsqueeze(0)

                    logits = actor(o)

                    # =========================
                    # MASK INVALID FRONTIERS
                    # =========================
                    for a in range(5):

                        # Invalid frontier = [0,0,0,0,0,0]
                        if np.all(obs[i][a] == 0):
                            logits[0, a] = -float("inf")
                    ###print("logits",logits)

                    # Safety fallback
                    if torch.all(torch.isinf(logits)):

                        actions[i] = 0
                        log_probs[i] = torch.tensor(
                            0.0,
                            dtype=torch.float32
                        )

                    else:

                        dist = torch.distributions.Categorical(
                            logits=logits
                        )

                        action = dist.sample()

                        log_prob = dist.log_prob(action)

                        actions[i] = action.item()
                        log_probs[i] = log_prob

                else:
                    actions[i] = -1

            global_state = torch.FloatTensor(
                get_global_state(env)
            ).unsqueeze(0)

            # =========================
            # ENV STEP
            # =========================
           #print(1)
            next_obs, next_extra, rewards, done = env.step(actions)
           #print(2)

            if RENDER:
                viewer.render(
                    env.ag_occ[0],
                    env.ag_pos,
                    obs[0],
                )
                time.sleep(1)

            reward = np.sum(rewards)
            ep_reward += reward

            next_global_state = torch.FloatTensor(
                get_global_state(env)
            ).unsqueeze(0)

            # =========================
            # STORE TRANSITIONS
            # =========================
            for i in agents_need_action:

                combined_extra = np.append(
                    extra[i],
                    env.time / 1000.0
                )

                combined_obs = np.concatenate([
                    obs[i].flatten(),
                    combined_extra
                ])

                trajectory.append({
                    "obs": torch.FloatTensor(combined_obs),
                    "global_state": global_state,
                    "next_global_state": next_global_state,
                    "log_prob": log_probs[i],
                    "reward": reward,
                    "done": done
                })

            # IMPORTANT
            obs = next_obs
            extra = next_extra

        # =========================
        # A2C UPDATE
        # =========================
        actor_losses = []
        critic_losses = []

        for t in trajectory:

            value = critic(t["global_state"])

            with torch.no_grad():
                next_value = critic(
                    t["next_global_state"]
                )

            reward = t["reward"]
            done = t["done"]

            target = reward + (
                GAMMA * next_value * (1 - done)
            )

            advantage = target - value

            actor_loss = (
                -t["log_prob"] * advantage.detach()
            )

            critic_loss = advantage.pow(2)

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        loss_actor = torch.stack(
            actor_losses
        ).mean()

        loss_critic = torch.stack(
            critic_losses
        ).mean()

        loss = loss_actor + 0.5 * loss_critic

        actor_opt.zero_grad()
        critic_opt.zero_grad()

        loss.backward()

        actor_opt.step()
        critic_opt.step()

        # =========================
        # LOGGING
        # =========================
        if ep % 10 == 0:
            print(
                f"Episode {ep} | "
                f"Reward: {ep_reward} | "
                f"Steps: {env.time}"
            )

        with open(
            "training_log7.csv",
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

            torch.save({
                "episode": ep,
                "actor_state_dict":
                    actor.state_dict(),
                "critic_state_dict":
                    critic.state_dict(),
                "actor_optimizer_state_dict":
                    actor_opt.state_dict(),
                "critic_optimizer_state_dict":
                    critic_opt.state_dict(),
            }, f"models5/a2c_checkpoint_{ep}.pth")

            print(
                f"Saved checkpoint: "
                f"models5/a2c_checkpoint_{ep}.pth"
            )

    torch.save(
        actor.state_dict(),
        "models5/actor_final.pth"
    )

    torch.save(
        critic.state_dict(),
        "models5/critic_final.pth"
    )

    print("Training finished. Final models saved.")


if __name__ == "__main__":
    train()
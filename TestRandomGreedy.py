import os
import csv
import numpy as np

import Environment


# ==========================================
# SETTINGS
# ==========================================

AGENTS = 3
RAYS = 120

W = 50
H = 50

EPISODES = 1000

SAVE_FOLDER = "Version 17 Training"

os.makedirs(SAVE_FOLDER, exist_ok=True)


# ==========================================
# ENVIRONMENT
# ==========================================

env = Environment.environment(
    AGENTS,
    RAYS,
    W,
    H,
    estimate_grid_size=12
)


# ==========================================
# POLICIES TO TEST
# ==========================================

POLICIES = {
    "random": "baseline_random.csv",
    "action0": "baseline_action0.csv",
    "action2": "baseline_action2.csv",
    "action4": "baseline_action4.csv",
    "fixed_agents": "baseline_fixed_agents.csv"
}


# ==========================================
# ACTION SELECTION
# ==========================================

def select_action(policy, agent_idx, action_mask):

    valid_actions = [
        i
        for i, valid in enumerate(action_mask)
        if valid
    ]

    # Should never happen, but just in case
    if len(valid_actions) == 0:
        return 0

    if policy == "random":

        return np.random.choice(valid_actions)

    elif policy == "action0":

        return 0 if action_mask[0] else valid_actions[0]

    elif policy == "action2":

        return 2 if action_mask[2] else valid_actions[0]

    elif policy == "action4":

        return 4 if action_mask[4] else valid_actions[0]

    elif policy == "fixed_agents":

        preferred = {
            0: 0,
            1: 2,
            2: 4
        }

        action = preferred[agent_idx]

        return action if action_mask[action] else valid_actions[0]

    else:

        raise ValueError(f"Unknown policy: {policy}")


# ==========================================
# RUN ALL POLICIES
# ==========================================

for policy, filename in POLICIES.items():

    print(f"\n==============================")
    print(f"Testing policy: {policy}")
    print(f"==============================")

    log_file = os.path.join(
        SAVE_FOLDER,
        filename
    )

    with open(log_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "episode",
            "reward",
            "steps",
            "exploration_reward",
            "decision_count"
        ])

    # --------------------------------------

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
            # Select actions
            # -------------------------

            for i in agents_need_action:

                actions[i] = select_action(
                    policy,
                    i,
                    action_masks[i]
                )

            # -------------------------
            # Assign frontier
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
            f"{policy:15s} | "
            f"Episode {ep:4d} | "
            f"Reward {episode_reward:8.2f} | "
            f"Steps {env.time:4d}"
        )

        with open(log_file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                ep,
                episode_reward,
                env.time,
                episode_exploration,
                episode_decision
            ])

print("\nAll baseline tests complete.")
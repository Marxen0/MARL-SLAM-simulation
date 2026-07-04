import time
import random
import csv
import numpy as np
import os
import Environment


# ==================================================
# RANDOM VALID ACTION
# ==================================================
def random_valid_action(mask):

    valid_actions = [

        i
        for i, valid in enumerate(mask)
        if valid

    ]

    if len(valid_actions) == 0:
        return 0

    return random.choice(valid_actions)


# ==================================================
# TEST RANDOM BASELINE
# ==================================================
def test_random():

    AGENTS = 3
    RAYS = 16
    W = 50
    H = 50

    EPISODES = 500
    RENDER = False

    env = Environment.environment(
        AGENTS,
        RAYS,
        W,
        H,
        render=RENDER
    )
    VERSION = "Version 1 Training"
    os.makedirs(
        VERSION,
        exist_ok=True
    )
    LOG_FILE = os.path.join(
        VERSION,
        "random_test.csv"
    )
    with open(
        LOG_FILE,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "episode",
            "reward",
            "steps"
        ])

        # ==========================================
        # EPISODES
        # ==========================================
        for ep in range(EPISODES):

            obs, action_mask = env.reset()

            done = False
            total_reward = 0

            while not done:

                actions = np.full(
                    env.ag_num,
                    -1,
                    dtype=int
                )

                agents_need_action = [

                    i
                    for i in range(env.ag_num)
                    if len(env.ag_paths[i]) == 0

                ]

                # ==========================================
                # RANDOM ACTIONS
                # ==========================================
                for i in agents_need_action:

                    actions[i] = random_valid_action(
                        action_mask[i]
                    )

                # ==========================================
                # STEP
                # ==========================================
                obs, action_mask, rewards, done = env.step(
                    actions
                )

                total_reward += np.sum(rewards)

                if RENDER:
                    time.sleep(0.2)

            # ==========================================
            # LOGGING
            # ==========================================
            print(
                f"Episode {ep} | "
                f"Reward: {total_reward:.2f} | "
                f"Steps: {env.time} | "
                f"Seed: {env.seed}"
            )

            writer.writerow([
                ep,
                total_reward,
                env.time
            ])

    print()
    print("Random benchmark finished!")
    print("Saved to random_test.csv")


if __name__ == "__main__":
    test_random()
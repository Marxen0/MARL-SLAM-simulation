import time
import numpy as np
import torch
import torch.nn as nn
import os
import Environment
from helpers.map_generator import OccupancyViewer


# ==================================================
# OBSERVATION -> NN INPUT
# ==================================================
def observation_to_tensor(obs, current_time):

    features = []

    # Other agent dijkstra
    features.extend(obs["other_agent_dijkstra"])

    # Frontiers
    for frontier in obs["frontiers"]:

        features.extend([
            frontier["frontier_value"],
            frontier["self_dx"],
            frontier["self_dy"],
            frontier["self_dijkstra"],
        ])

        for ag in frontier["agents"]:

            features.extend([
                ag["target_to_frontier_euclidean"],
                ag["agent_to_frontier_euclidean"],
                ag["dijkstra_sum_ag"],
                ag["dijkstra_overlap_percent_ag"],
                ag["ag_target_dx"],
                ag["ag_target_dy"],
                ag["ag_to_frontier_dx"],
                ag["ag_to_frontier_dy"],
            ])

    # ADD TIME FEATURE
    features.append(current_time / 1000.0)

    return np.array(
        features,
        dtype=np.float32
    )


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

    AGENTS = 3
    RAYS = 16
    W = 50
    H = 50

    env = Environment.environment(
        AGENTS,
        RAYS,
        W,
        H,
        render=True
    )

    obs, action_mask = env.reset()
    x = observation_to_tensor(
        obs[0],
        env.time
    )

    print(len(x))
    input_dim = len(
        observation_to_tensor(obs[0], env.time)
    )

    print("Input dimension:", input_dim)

    actor = Actor(input_dim)

    VERSION = "Version 3 Training"

    MODEL_FOLDER = os.path.join(
        VERSION,
        "models"
    )
    actor.load_state_dict(
        torch.load(
            f"{MODEL_FOLDER}/actor_final.pth",
            map_location="cpu"
        )
    )

    actor.eval()

    #viewer = OccupancyViewer(
     #   env.world_widht,
      #  env.world_height,
       # cell_size=20
    #)

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
        # ACTION SELECTION
        # ==========================================
        for i in agents_need_action:

            with torch.no_grad():

                x = observation_to_tensor(
                    obs[i],
                    env.time
                )

                x = torch.FloatTensor(
                    x
                ).unsqueeze(0)

                logits = actor(x)

                # -----------------------------
                # Apply action mask
                # -----------------------------
                for a in range(5):

                    if not action_mask[i][a]:

                        logits[0, a] = -float("inf")

                # Safety fallback
                if torch.all(torch.isinf(logits)):

                    print(
                        f"WARNING: Agent {i} "
                        "has no valid actions"
                    )

                    actions[i] = 0

                else:

                    action = torch.argmax(
                        logits,
                        dim=1
                    )

                    actions[i] = action.item()

        # ==========================================
        # STEP
        # ==========================================
        obs, action_mask, rewards, done = env.step(
            actions
        )

        total_reward += np.sum(rewards)

        # ==========================================
        # RENDER
        # ==========================================
   #     viewer.render(
   #         env.ag_occ[0],
   #         env.ag_pos
   #     )

        # Debug collisions
        for i in range(env.ag_num):
            for j in range(i + 1, env.ag_num):

                if tuple(env.ag_pos[i]) == tuple(env.ag_pos[j]):

                    print(
                        f"Collision: "
                        f"{i} and {j}"
                    )

        time.sleep(0.2)

    print()
    print("seed : ",env.seed)
    print("Finished")
    print("Reward:", total_reward)
    print("Steps:", env.time)


if __name__ == "__main__":
    test()
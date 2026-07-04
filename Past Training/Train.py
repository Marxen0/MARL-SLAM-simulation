import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import Environment
from helpers.map_generator import OccupancyViewer
from helpers.frontier_finder import get_frontiers
import csv

# Import the environment class we just designed
# from your_env_file import environment

# --- Mock Policy Network for Demonstration ---
class MultiAgentPolicy(nn.Module):
    def __init__(self, observation_shape=(5, 7), action_dim=5):
        super(MultiAgentPolicy, self).__init__()
        # Input: Flattened observation matrix (5 targets * 7 metrics = 35)
        input_dim = observation_shape[0] * observation_shape[1]
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        # Flatten the (5, 7) observation to (35,)
        obs_flat = obs.view(obs.size(0), -1)
        return self.network(obs_flat)

# --- Training Script ---
def train_marl_async():
    # 1. Hyperparameters & Environment Setup
    AGENT_NUM = 3
    RAY_COUNT = 16
    WORLD_WIDTH = 50
    WORLD_HEIGHT = 50
    EPISODES = 500
    
    env = Environment.environment(AGENT_NUM, RAY_COUNT, WORLD_WIDTH, WORLD_HEIGHT)
   # viewer = OccupancyViewer(
   #     env.world_widht,
   #     env.world_height,
   #     cell_size=20
   # )
    # Initialize a policy and optimizer for the agents
    # (Shared policy layout common in Parameter Sharing MARL)
    policy = MultiAgentPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    print("Starting Asynchronous MARL Training Loop...")
    print("--------------------------------------------")

    for episode in range(1, EPISODES + 1):
        # Reset returns a list of initial observations for ALL agents
        observations = env.reset()
        done = False
        episode_reward = 0
        
        # We start with a full layout where all agents need a decision immediately
        while not done:
            # Re-verify which agents actually need a decision right now
            # (Their current path buffer in the env is empty)
            agents_needing_decision = [i for i in range(env.ag_num) if len(env.ag_paths[i]) == 0]
            
            # Create a placeholder array for actions. 
            # Active agents will keep their current path, while idle agents will get overwritten.
            actions = np.zeros(env.ag_num, dtype=int)
            
            # 2. Action Selection Phase (Only for idle agents)
            for agent_idx in range(env.ag_num):
                if agent_idx in agents_needing_decision:
                    # Convert agent's unique (5, 7) frontier observation to tensor
                    obs_tensor = torch.FloatTensor(observations[agent_idx]).unsqueeze(0) 
                    
                    with torch.no_grad():
                        action_probs = policy(obs_tensor)
                        
                    # Sample action from probabilities
                    action = torch.multinomial(action_probs, 1).item()
                    actions[agent_idx] = action
                else:
                    # For agents currently traveling, pass a dummy index.
                    # The env's updated step() will completely ignore this index anyway.
                    actions[agent_idx] = -1 

            # 3. Environment Step Phase
            # The env will process the new actions for idle agents, progress the clock,
            # and pause as soon as AT LEAST ONE agent reaches its destination.
    #        viewer.render(
    #            env.global_map,
    #            env.ag_pos,
    #        )
            next_observations, rewards, done = env.step(actions)
            
            # Track rewards accumulated during this step interval
            episode_reward += np.sum(rewards)
            
            # 4. Optimization / Transition Storage Phase
            # In a production script, you would append data to a Replay Buffer here.
            # CRITICAL: Only save transitions (obs, action, reward, next_obs) 
            # for the specific agent indices listed in 'agents_needing_decision'.
            
            # Advance state forward
            observations = next_observations

        with open("training_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, episode_reward, env.time])
        # Logging metrics at the end of every episode
        if episode % 10 == 0:
            print(f"Episode {episode:03d} | Total Step Penalty (Reward): {episode_reward:.1f} | Env Simulation Clock: {env.time} steps")
        

if __name__ == "__main__":
    # To run this script seamlessly, copy the updated environment class from the previous 
    # response and make sure 'walk_agent' and 'check_done' are imported/defined!
    print(3)
    train_marl_async()
import numpy as np
# Import your standalone helper functions from the helpers package
from helpers.map_generator import create_house, visualize_occ
from helpers.frontier_finder import get_frontiers, filter_frontiers
# Tambahkan import ini di bagian atas file environment Anda
from helpers.agent_movement import walk_agent, check_done, check_agent_movement, proximity_penalty
from collections import deque


def get_start_positions(world, num_agents):
    """
    Find num_agents nearby free starting positions.

    Args:
        world (np.ndarray):
            0 = free
            1 = wall
        num_agents (int):
            Number of agents.

    Returns:
        list[tuple[int, int]]:
            List of (x, y) starting positions.
    """

    width, height = world.shape

    # Find first valid border position
    start = None

    for x in range(2, width - 2):
        for y in range(2, height - 2):

            if x == 2 or x == width - 3 or y == 2 or y == height - 3:

                if world[x, y] == 0:
                    start = (x, y)
                    break

        if start is not None:
            break

    if start is None:
        raise ValueError("No free starting position found.")

    # BFS to collect nearby positions
    queue = deque([start])
    visited = {start}
    positions = []

    directions = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1)
    ]

    while queue and len(positions) < num_agents:

        x, y = queue.popleft()
        positions.append((x, y))

        for dx, dy in directions:

            nx, ny = x + dx, y + dy

            if (
                0 <= nx < width
                and 0 <= ny < height
                and world[nx, ny] == 0
                and (nx, ny) not in visited
            ):
                visited.add((nx, ny))
                queue.append((nx, ny))

    if len(positions) < num_agents:
        raise ValueError(
            f"Could only find {len(positions)} connected free cells, "
            f"but {num_agents} agents were requested."
        )

    return positions
def combine_occ(ag_occ):
    """
    Combines individual occupancy grids from multiple agents into a single global map.
    
    Args:
        ag_occ (np.ndarray): 3D array of shape (agent_num, world_width, world_height)
                             containing the occupancy grid for each agent.
                             
    Returns:
        np.ndarray: 2D array of shape (world_width, world_height) representing 
                    the merged global occupancy map.
    """
    # Ensure the input is a NumPy array
    ag_occ_array = np.asarray(ag_occ)
    
    # Merge individual agent grids by taking the maximum value along the agent axis (axis=0)
    # This preserves discovered spaces (1s) over undiscovered spaces (0s)
    global_map = np.max(ag_occ_array, axis=0)
    
    return global_map
def agent_target(actions, agent_observations):
    """
    Maps each agent's discrete action to its chosen frontier profile.
    
    Args:
        actions: List or 1D array of shape (agent_num,) containing 
                 integer actions between 0 and 4.
        agent_observations: Array-like object or NumPy array of shape 
                            (agent_num, 5,   containing frontier choices.
                            
    Returns:
        np.ndarray: Selected targets of shape (agent_num, 7)
    """
    # Convert inputs to reliable numpy arrays
    obs_array = np.asarray(agent_observations)
    actions_array = np.asarray(actions, dtype=int)
    
    # Generate an array of row indices: [0, 1, ..., agent_num - 1]
    agent_indices = np.arange(len(actions_array))
    
    # Advanced slicing picks the specific action row for each agent row
    selected_targets = obs_array[agent_indices, actions_array]
    
    return selected_targets
class environment():
    def __init__(self, agent_num, agent_ray_count, world_widht, world_height):
        self.world_widht = world_widht
        self.world_height = world_height
        self.world = np.zeros((self.world_widht, self.world_height))

        self.ag_num = agent_num
        self.ag_ray_count = agent_ray_count
        self.cached_paths = {}
    def reset(self, seed=None):
        self.world = create_house(self.world_widht, self.world_height, seed)

        
        start_pos = get_start_positions(self.world, self.ag_num)
        self.ag_pos = start_pos
        self.ag_target = np.zeros((self.ag_num, 6))
        self.ag_occ = np.full((self.ag_num, self.world_widht, self.world_height), -1)
        # Inside __init__ AND reset
        self.ag_paths = [[] for _ in range(self.ag_num)]
        self.time = 0
        self.ag_observations = []
        self.cached_paths = {}
        self.ag_observations = []
        self.global_map = np.full((self.world_widht, self.world_height), -1)
        self.ag_occ, self.ag_pos, walk_penalty = walk_agent(self.world, self.ag_occ, self.ag_ray_count, start_pos)
        for agent in range(self.ag_num):
            obs, ag_cached_paht = self.agent_observation(agent)
            self.ag_observations.append(obs)
            self.cached_paths[agent] = ag_cached_paht
      #  visualize_occ(self.ag_occ[0])
        return self.ag_observations
    def agent_observation(self, agent):
        agent_frontiers = get_frontiers(self.ag_occ[agent])
        agent_frontiers_choice, ag_cached_path = filter_frontiers(agent, agent_frontiers, self.ag_pos, self.ag_occ[agent], self.ag_target)
        observation = agent_frontiers_choice
        return observation, ag_cached_path
    def critic_observation(self):
        global_observation = []
        global_observation.append(self.ag_target)
        global_observation.append(self.ag_pos)
        self.global_map = combine_occ(self.ag_occ)
        global_observation.append(self.global_map)
        return global_observation
    def step(self, actions):
        """
        Expects 'actions' only for agents that actually need a decision.
        """
        # 1. Assign new paths ONLY to agents that are idle (empty path list)
        for agent_idx in range(self.ag_num):
            if len(self.ag_paths[agent_idx]) == 0:
                chosen_action = actions[agent_idx]
                
                # CRITICAL: Force convert the cached path to a list of coordinates
                # shapes should look like: [(x1,y1), (x2,y2), ...]
                self.ag_paths[agent_idx] = list(self.cached_paths[agent_idx][chosen_action])
                
                # Update its target profile matrix
                self.ag_target[agent_idx] = self.ag_observations[agent_idx][chosen_action]

        # 2. Environment Simulation Loop
        done = False
        new_time = 0
        while True:
            current_step_paths = []
            for agent_idx in range(self.ag_num):
                if len(self.ag_paths[agent_idx]) > 0:
                    # Pop the next step sequence coordinate pair
                    current_step_paths.append(self.ag_paths[agent_idx].pop(0))
                else:
                    # If empty, they hold their current ground position
                    current_step_paths.append(self.ag_pos[agent_idx])
            
            # Pass single-step coordinates to your walk_agent helper
            self.ag_occ, self.ag_pos, walk_penalty = walk_agent(self.world, self.ag_occ, self.ag_ray_count, current_step_paths)
            new_time += 1
            
            self.global_map = combine_occ(self.ag_occ)
            done = check_done(self.global_map)
            if done:
                break
                
            # BREAK CONDITION: Check if any agent has completely finished their path
            agents_needing_decision = [i for i in range(self.ag_num) if len(self.ag_paths[i]) == 0]
            if len(agents_needing_decision) > 0:
                break

        # 3. Rewards tracking
        proximity_rewards = np.array(proximity_penalty(self.ag_pos))
        time_rewards = np.zeros(self.ag_num)
        time_rewards[0] = -new_time * 0.1

        rewards = proximity_rewards + time_rewards + np.array(walk_penalty)
        self.time += new_time
        
        # Clear old tracking cache
        self.cached_paths.clear()
        
        # 4. Generate new observations
        self.ag_observations = []
        for agent in range(self.ag_num):
            obs, ag_cached_paht = self.agent_observation(agent)
            self.ag_observations.append(obs)
            self.cached_paths[agent] = ag_cached_paht
        return self.ag_observations, rewards, done
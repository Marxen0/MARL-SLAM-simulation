import numpy as np
# Import your standalone helper functions from the helpers package
from helpers.map_generator import create_house, OccupancyViewer
from helpers.frontier_finder import observation
# Tambahkan import ini di bagian atas file environment Anda
from helpers.agent_movement import walk_agent, check_done, check_agent_movement, proximity_penalty, proximity_penalty_dijkstra
from collections import deque
import time
import random

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
class environment():
    def __init__(self, agent_num, agent_ray_count, world_widht, world_height, render=False):
        self.render = render
        self.world_widht = world_widht
        self.world_height = world_height
        self.world = np.zeros((self.world_widht, self.world_height))

        self.ag_num = agent_num
        self.ag_ray_count = agent_ray_count
        self.cached_paths = {}
        if render == True:
            self.viewer = OccupancyViewer(
            self.world_widht,
            self.world_height,
            cell_size=20)
    def reset(self, seed=None):
        if seed==None:
            seed = random.randint(0, 99999999)
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        self.world = create_house(self.world_widht, self.world_height, self.seed)

        
        start_pos = get_start_positions(self.world, self.ag_num)
        self.ag_pos = list(start_pos)
        self.ag_target = start_pos
        self.ag_occ = np.full((self.ag_num, self.world_widht, self.world_height), -1)
        # Inside __init__ AND reset
        self.ag_paths = [[] for _ in range(self.ag_num)]
        self.time = 0
        self.cached_paths = {}
        self.observation = {}
        self.masked_action = {}
        self.global_map = np.full((self.world_widht, self.world_height), -1)
        self.prev_global_map = self.global_map.copy()
        self.ag_occ, self.ag_pos, walk_penalty = walk_agent(self.world, self.ag_occ, self.ag_ray_count, start_pos)
        self.update_agent_observation()
      #  visualize_occ(self.ag_occ[0])
        return self.observation, self.masked_action
    def update_agent_observation(self, agents=None):
        if agents == None:
            agents = [i for i in range(self.ag_num)]
        for agent in agents:

            obs, mask = self.agent_observation(agent)

            self.observation[agent] = obs
            self.masked_action[agent] = mask

        return self.observation, self.masked_action
    def agent_observation(self, agent):
        agent_obs, agent_mask = observation(agent, self.ag_pos, self.ag_occ[agent], self.ag_target, self.ag_num)
        return agent_obs, agent_mask
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
                if not self.masked_action[agent_idx][chosen_action]:
                    raise ValueError(
                        f"Agent {agent_idx} picked invalid action {chosen_action}"
                    )
                
                # CRITICAL: Force convert the cached path to a list of coordinates
                # shapes should look like: [(x1,y1), (x2,y2), ...]
                self.ag_paths[agent_idx] = self.observation[agent_idx]["frontiers"][chosen_action]["cached_path"]
                
                # Update its target profile matrix
                self.ag_target[agent_idx] = self.observation[agent_idx]["frontiers"][chosen_action]["frontier_position"]

                if self.render and agent_idx == 0:
                    to_render = [self.observation[0]["frontiers"][i]["frontier_position"] for i in range(5)]
                    self.viewer.render(
                    self.ag_occ[0],
                    self.ag_pos,
                    to_render)
                    time.sleep(0.5)
                    print(self.observation[0]["frontiers"][actions[0]])
        # 2. Environment Simulation Loop
        done = False
        new_time = 0
        agents_needing_decision = []
        proximity_reward = np.zeros(self.ag_num)
        final_walk_penalty = np.zeros(self.ag_num)
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
            final_walk_penalty = final_walk_penalty + np.asarray(walk_penalty)
            new_time += 1
            
            self.global_map = combine_occ(self.ag_occ)
            done = check_done(self.global_map)
            if done:
                break
            if self.render:
                    self.viewer.render(
                    self.ag_occ[0],
                    self.ag_pos,
                    [self.observation[0]["frontiers"][i]["frontier_position"] for i in range(5)])
                    time.sleep(0.5)
       #     if self.render:
        #        self.viewer.render(
         #           self.ag_occ[0],
          #          self.ag_pos,
           #         self.ag_target,
            #    )
             #   time.sleep(0.1)
            proximity_reward = proximity_reward + np.asarray(proximity_penalty(self.ag_pos))
                
            # BREAK CONDITION: Check if any agent has completely finished their path
            agents_needing_decision = [i for i in range(self.ag_num) if len(self.ag_paths[i]) == 0]
            if len(agents_needing_decision) > 0:
                break

        # 3. Rewards tracking
        time_rewards = -new_time 
        # Count unknown cells
        prev_unknown = np.sum(self.prev_global_map == -1)
        curr_unknown = np.sum(self.global_map == -1)

        # Positive if we explored new cells
        exploration_reward = prev_unknown - curr_unknown
        rewards =  time_rewards + (exploration_reward/(-time_rewards))

        self.time += new_time
        self.prev_global_map = self.global_map.copy()
        if not done:
            self.update_agent_observation()
        return self.observation, self.masked_action, rewards, done
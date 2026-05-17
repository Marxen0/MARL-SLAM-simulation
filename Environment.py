import numpy as np
# Import your standalone helper functions from the helpers package
from helpers.map_generator import create_house
from helpers.frontier_finder import get_frontiers, filter_frontiers

class environment():
    def __init__(self, agent_num, agent_ray_count, world_widht, world_height):
        self.world_widht = world_widht
        self.world_height = world_height
        self.world = np.zero((self.world_widht, self.world_height))

        self.ag_num = agent_num
        self.ag_ray_count = agent_ray_count
    
    def reset(self, seed=None):
        self.world = create_house(self.world_widht, self.world_height, seed)

        self.ag_pos = np.zeros((self.ag_num, 2))
        self.ag_target = np.zeros((self.ag_num, 4))
        self.ag_occ = np.zeros((self.ag_num, self.world_widht, self.world_height))
        self.ag_path = {}
        self.time = 0
        self.ag_observations = []
        for agent in range(self.ag_num):
            self.ag_observations.append(self.agent_observation(agent))
        return self.ag_observations
    def agent_observation(self, agent):
        agent_frontiers = get_frontiers(self.ag_occ[agent])
        agent_frontiers_choice = filter_frontiers(agent, agent_frontiers, self.ag_pos, self.ag_target)
        observation = agent_frontiers_choice
        return observation
    def critic_observation(self):
        global_observation = []
        global_observation.append(self.ag_target)
        global_observation.append(self.ag_pos)
        global_map = combine_occ(self.ag_occ)
        global_observation.append(global_map)
        return global_observation
    def step(self, actions):
        self.ag_paths, ammount_run = create_path(self.ag_pos, self.ag_target, actions)
        done = False
        new_time = 0
        for _ in range(ammount_run):
            self.ag_occ, self.ag_pos = walk_agent(self.world, self.ag_ray_count, self.ag_paths)
            new_time += 1
            done = check_done(combine_occ(self.ag_occ))
            if done:
                break
        reward = np.full(self.ag_num, -new_time)
        self.time += new_time
        self.ag_observations = []
        for agent in range(self.ag_num):
            self.ag_observations.append(self.agent_observation(agent))
        return self.ag_observations, reward, done
        
    
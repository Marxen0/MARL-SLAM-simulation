import numpy as np

from scipy import ndimage

def get_frontiers(occupancy_grid):
    """
    Identify frontier cells using convolution (faster for large grids).
    
    Args:
        occupancy_grid: 2D numpy array (0=unexplored, 1=free, -1=obstacle)
    
    Returns:
        numpy array of shape (N, 2) with [x, y] coordinates of frontiers
    """
    # Binary mask of free cells
    free_mask = (occupancy_grid == 1).astype(int)
    
    # Binary mask of unexplored cells
    unexplored_mask = (occupancy_grid == 0).astype(int)
    
    # Dilate unexplored region to find cells adjacent to unexplored
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])
    adjacent_to_unexplored = ndimage.binary_dilation(unexplored_mask, kernel)
    
    # Frontiers = free cells that are adjacent to unexplored
    frontier_mask = free_mask & adjacent_to_unexplored
    
    # Get coordinates (returns (y_indices, x_indices))
    ys, xs = np.where(frontier_mask)
    
    # Return as (N, 2) array with [x, y] format
    return np.column_stack((xs, ys)) if len(xs) > 0 else np.empty((0, 2), dtype=int)

def filter_frontiers(agent_idx, frontiers, ag_pos, ag_occ):
    """
    Filters and selects 5 specific frontiers for a given agent.
    
    Returns a 5x5 numpy array. Each row contains:
    [x_direction, y_direction, frontier_value, dist_to_agent, closest_dist_to_other_agent]
    
    If fewer than 5 frontiers exist, the array is padded with zeros.
    """
    # 1. Handle edge case where no frontiers are found
    if len(frontiers) == 0:
        return np.zeros((5, 5))
    
    current_agent_pos = ag_pos[agent_idx]
    other_agents_pos = np.delete(ag_pos, agent_idx, axis=0) if len(ag_pos) > 1 else None
    
    # Extract map limits for bounds checking
    map_occ = ag_occ[agent_idx]
    height, width = map_occ.shape
    
    # 2. Compute metrics for ALL found frontiers
    frontier_metrics = []
    
    for f in frontiers:
        fx, fy = f[0], f[1]
        
        # --- Metric A: Distance to current agent ---
        dist_to_agent = np.linalg.norm(current_agent_pos - [fx, fy])
        
        # --- Metric B: Closest distance to any OTHER agent ---
        if other_agents_pos is not None and len(other_agents_pos) > 0:
            dists_to_others = np.linalg.norm(other_agents_pos - [fx, fy], axis=1)
            closest_other_dist = np.min(dists_to_others)
        else:
            closest_other_dist = 999.0  # Default large distance if it's a solo agent
            
        # --- Metric C: Frontier Value (Count neighboring unexplored cells '0') ---
        # Look at the 8-connected neighborhood
        value = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = fx + dx, fy + dy
                # Ensure the neighbor is within map boundaries
                if 0 <= nx < width and 0 <= ny < height:
                    if map_occ[ny, nx] == 0:  # 0 is unexplored
                        value += 1
                        
        frontier_metrics.append({
            'coord': [fx, fy],
            'value': value,
            'dist_to_agent': dist_to_agent,
            'closest_other_dist': closest_other_dist,
            'original_idx': len(frontier_metrics)
        })
        
    # Keep track of selected frontier indices to avoid picking the exact same one twice
    selected_indices = []
    
    def get_best_available(sorted_list, count):
        added = []
        for item in sorted_list:
            if item['original_idx'] not in selected_indices:
                selected_indices.append(item['original_idx'])
                added.append(item)
                if len(added) == count:
                    break
        return added

    # 3. Apply selection criteria
    # Criteria 1: 2 Closest to the current agent
    sorted_by_closeness = sorted(frontier_metrics, key=lambda k: k['dist_to_agent'])
    closest_2 = get_best_available(sorted_by_closeness, 2)
    
    # Criteria 2: 2 Highest value (most unexplored adjacent space)
    sorted_by_value = sorted(frontier_metrics, key=lambda k: k['value'], reverse=True)
    highest_2 = get_best_available(sorted_by_value, 2)
    
    # Criteria 3: 1 Farthest from any other agent
    sorted_by_isolation = sorted(frontier_metrics, key=lambda k: k['closest_other_dist'], reverse=True)
    farthest_1 = get_best_available(sorted_by_isolation, 1)
    
    # Combine the chosen items
    chosen_frontiers = closest_2 + highest_2 + farthest_1
    
    # 4. Format into a 5x5 array
    output_array = np.zeros((5, 5))
    
    for i, f_item in enumerate(chosen_frontiers):
        fx, fy = f_item['coord']
        
        # Directions calculated relative to current agent position
        xdirection = fx - current_agent_pos[0]
        ydirection = fy - current_agent_pos[1]
        
        output_array[i] = [
            xdirection,
            ydirection,
            f_item['value'],
            f_item['dist_to_agent'],
            f_item['closest_other_dist']
        ]
        
    return output_array

class environment():
    def __init__(self, agent_num, agent_ray_count, world_widht, world_height):
        self.world_widht = world_widht
        self.world_height = world_height
        self.world = np.zero((self.world_widht, self.world_height))

        self.ag_num = agent_num
        self.ag_ray_count = agent_ray_count
    
    def reset(self, seed=None):
        self.world = create_house(seed)

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
        observation = [agent_frontiers_choice]
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
        
    
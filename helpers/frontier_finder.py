import heapq
import numpy as np
from helpers.map_generator import visualize_occ
from scipy import ndimage
def dijkstra_path(world, start, goal):
    """
    Find the shortest path between two points using Dijkstra.

    Args:
        world (np.ndarray):
            World map.
            0 = free space
            1 = wall

        start (tuple):
            Starting coordinate (x, y).

        goal (tuple):
            Goal coordinate (x, y).

    Returns:
        list[tuple]:
            Path from start to goal.

            Example:
                [(1,1), (1,2), (2,2), (3,2)]

            Returns an empty list if no path exists.
    """

    start = tuple(map(int, start))
    goal = tuple(map(int, goal))

    width, height = world.shape

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    pq = [(0, start)]
    distances = {start: 0}
    previous = {}

    while pq:

        current_dist, current = heapq.heappop(pq)

        if current == goal:
            break

        x, y = current

        for dx, dy in directions:

            nx = x + dx
            ny = y + dy

            if nx < 0 or nx >= width:
                continue

            if ny < 0 or ny >= height:
                continue

            if world[nx, ny] != 0:
                continue

            neighbor = (nx, ny)
            new_dist = current_dist + 1

            if neighbor not in distances:
                distances[neighbor] = new_dist
                previous[neighbor] = current
                heapq.heappush(
                    pq,
                    (new_dist, neighbor)
                )

    if goal not in previous and goal != start:
        return []

    path = []
    current = goal

    while current != start:
        path.append(current)
        current = previous[current]

    path.reverse()

    return path
def get_frontiers(occ):

    frontiers = []

    width, height = occ.shape

    for x in range(1, width - 1):
        for y in range(1, height - 1):

            if occ[x, y] != 0:
                continue

            neighbors = [
                occ[x+1,y],
                occ[x-1,y],
                occ[x,y+1],
                occ[x,y-1]
            ]

            if -1 in neighbors:
                frontiers.append((x, y))

    return np.array(frontiers)
def compute_cached_paths(
    ag_occ,
    current_agent_pos,
    chosen_frontiers
):
    """
    Compute paths for all selected frontiers.

    Args:
        world (np.ndarray):
            Ground truth map.

        current_agent_pos (tuple):
            Current agent position.

        chosen_frontiers (list):
            Selected frontier metric dictionaries.

    Returns:
        list[list[tuple]]:
            Cached paths.

            The path at index i corresponds to
            chosen_frontiers[i].
    """

    cached_paths = []
    for frontier in chosen_frontiers:

        fx, fy = frontier["coord"]
        path = dijkstra_path(
            ag_occ,
            current_agent_pos,
            (fx, fy)
        )
        if (path == []): 
            print("EMPTY PATH : ", current_agent_pos, "to", (fx,fy), "filled with ", ag_occ[fx][fy])
            visualize_occ(ag_occ)
        cached_paths.append(path)

    while len(cached_paths) < 5:
        cached_paths.append([])
    
    return cached_paths

def get_best_available(sorted_list, count):
    added = []
    selected_indices = []
    for item in sorted_list:
        if item['original_idx'] not in selected_indices:
            selected_indices.append(item['original_idx'])
            added.append(item)
            if len(added) == count:
                break
    return added
def filter_frontiers(agent_idx, frontiers, ag_pos, ag_occ):
    """
    Filters and selects 5 specific frontiers for a given agent.
    
    Returns a 5x7 numpy array. Each row contains:
    [x_direction, y_direction, frontier_x, frontier_y, frontier_value, dist_to_agent, closest_dist_to_other_agent]
    
    If fewer than 5 frontiers exist, the array is padded with zeros.
    """
    # 1. Handle edge case where no frontiers are found (now returning 5x7)
    if len(frontiers) == 0:
        return np.zeros((5, 7)), [[] for _ in range(5)]
    
    current_agent_pos = ag_pos[agent_idx]
    other_agents_pos = np.delete(ag_pos, agent_idx, axis=0) if len(ag_pos) > 1 else None
    
    # Extract map limits for bounds checking
    height, width = ag_occ.shape
    
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
        value = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if ag_occ[ny, nx] == 0:  # 0 is unexplored
                        value += 1
                        
        frontier_metrics.append({
            'coord': [fx, fy],
            'value': value,
            'dist_to_agent': dist_to_agent,
            'closest_other_dist': closest_other_dist,
            'original_idx': len(frontier_metrics)
        })
    # Keep track of selected frontier indices to avoid duplicates 
    # 3. Apply selection criteria
    # Criteria 1: 2 Closest to the current agent
    sorted_by_closeness = sorted(
    frontier_metrics,
    key=lambda k: k['dist_to_agent']
    )
    closest_2 = sorted_by_closeness[:2]

    # Criteria 2: 2 Highest value
    sorted_by_value = sorted(
        frontier_metrics,
        key=lambda k: k['value'],
        reverse=True
    )
    highest_2 = sorted_by_value[:2]

    # Criteria 3: 1 Farthest from any other agent
    sorted_by_isolation = sorted(
        frontier_metrics,
        key=lambda k: k['closest_other_dist'],
        reverse=True
    )
    farthest_1 = sorted_by_isolation[:1]
    
    # Combine the chosen items
    chosen_frontiers = closest_2 + highest_2 + farthest_1
    cached_paths = compute_cached_paths(
        ag_occ,
        current_agent_pos,
        chosen_frontiers
    )
    
    # 4. Format into a 5x7 array
    output_array = np.zeros((5, 7))
    
    for i, f_item in enumerate(chosen_frontiers):
        fx, fy = f_item['coord']
        
        # Directions calculated relative to current agent position
        xdirection = fx - current_agent_pos[0]
        ydirection = fy - current_agent_pos[1]
        
        output_array[i] = [
            xdirection,
            ydirection,
            fx,                   # Absolute frontier X coordinate
            fy,                   # Absolute frontier Y coordinate
            f_item['value'],
            f_item['dist_to_agent'],
            f_item['closest_other_dist']
        ]
        
    return output_array, cached_paths

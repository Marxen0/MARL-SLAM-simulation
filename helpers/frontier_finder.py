import heapq
import numpy as np
from helpers.map_generator import visualize_occ
from scipy import ndimage
import heapq
import numpy as np


def dijkstra_all(start, occ):
    """
    Compute shortest paths from start to every reachable free cell.

    Args:
        start (tuple):
            (x, y) starting position.

        occ (np.ndarray):
            Occupancy map:
                -1 = unknown
                 0 = free
                 1 = wall

    Returns:
        tuple:
            dist:
                dict[(x,y)] -> shortest distance

            parent:
                dict[(x,y)] -> previous node
    """

    start = tuple(map(int, start))

    width, height = occ.shape

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    pq = [(0, start)]

    dist = {start: 0}
    parent = {}

    while pq:

        current_dist, current = heapq.heappop(pq)

        x, y = current

        for dx, dy in directions:

            nx = x + dx
            ny = y + dy

            if nx < 0 or nx >= width:
                continue

            if ny < 0 or ny >= height:
                continue

            # Only walk on known free space
            if occ[nx, ny] != 0:
                continue

            neighbor = (nx, ny)
            new_dist = current_dist + 1

            if neighbor not in dist:

                dist[neighbor] = new_dist
                parent[neighbor] = current

                heapq.heappush(
                    pq,
                    (new_dist, neighbor)
                )

    return dist, parent


def reconstruct_path(parent, start, goal):
    """
    Reconstruct a path using the parent dictionary.
    """

    if start == goal:
        return []

    if goal not in parent:
        return []

    path = []

    current = goal

    while current != start:

        path.append(current)
        current = parent[current]

    path.reverse()

    return path


def compute_frontier_value(fx, fy, occ):
    """
    Count unknown neighbors around a frontier.
    """

    value = 0

    width, height = occ.shape

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:

            nx = fx + dx
            ny = fy + dy

            if nx < 0 or nx >= width:
                continue

            if ny < 0 or ny >= height:
                continue

            if occ[nx, ny] == -1:
                value += 1

    return value
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

def filter_frontiers(agent_idx, frontiers, ag_pos, ag_occ, ag_target):
    """
    Select 5 frontier choices and compute cached paths.

    Returns:
        observation:
            shape (5, 7)

        cached_paths:
            list of length 5
    """

    if len(frontiers) == 0:
        return np.zeros((5, 7)), [[] for _ in range(5)]

    current_agent_pos = tuple(map(int, ag_pos[agent_idx]))

    other_agents_pos = np.delete(
        ag_pos,
        agent_idx,
        axis=0
    )

    map_occ = ag_occ

    # ==================================================
    # STEP 1:
    # Run Dijkstra ONCE
    # ==================================================

    dist, parent = dijkstra_all(
        current_agent_pos,
        map_occ
    )

    # ==================================================
    # STEP 2:
    # Keep only reachable frontiers
    # ==================================================

    reachable_frontiers = []

    for frontier in frontiers:

        fx, fy = map(int, frontier)

        if (fx, fy) in dist:
            reachable_frontiers.append((fx, fy))

    if len(reachable_frontiers) == 0:

        return np.zeros((5, 7)), [[] for _ in range(5)]

    # ==================================================
    # STEP 3:
    # Compute metrics
    # ==================================================

    frontier_metrics = []

    for fx, fy in reachable_frontiers:

        dist_to_agent = dist[(fx, fy)]

        # Distance to OTHER agents' targets
        other_targets = [
            [target[2], target[3]]   # Extract (pos_x, pos_y)
            for i, target in enumerate(ag_target)
            if i != agent_idx and target is not None
        ]

        if len(other_targets) > 0:

            other_targets = np.array(other_targets)

            dists_to_targets = np.linalg.norm(
                other_targets - [fx, fy],
                axis=1
            )

            closest_other_dist = np.min(dists_to_targets)

        else:

            closest_other_dist = 999

        value = compute_frontier_value(
            fx,
            fy,
            map_occ
        )

        frontier_metrics.append({
            "coord": (fx, fy),
            "value": value,
            "dist_to_agent": dist_to_agent,
            "closest_other_dist": closest_other_dist
        })

    # ==================================================
    # STEP 4:
    # Select frontiers
    # (duplicates allowed)
    # ==================================================

    closest_2 = sorted(
        frontier_metrics,
        key=lambda x: x["dist_to_agent"]
    )[:2]

    highest_2 = sorted(
        frontier_metrics,
        key=lambda x: x["value"],
        reverse=True
    )[:2]

    farthest_1 = sorted(
        frontier_metrics,
        key=lambda x: x["closest_other_dist"],
        reverse=True
    )[:1]

    chosen_frontiers = (
        closest_2 +
        highest_2 +
        farthest_1
    )

    # ==================================================
    # STEP 5:
    # Build observation + cached paths
    # ==================================================

    observation = np.zeros((5, 7))

    cached_paths = []

    for i, frontier in enumerate(chosen_frontiers):

        fx, fy = frontier["coord"]

        x_direction = fx - current_agent_pos[0]
        y_direction = fy - current_agent_pos[1]

        observation[i] = [
            x_direction,
            y_direction,
            fx,
            fy,
            frontier["value"],
            frontier["dist_to_agent"],
            frontier["closest_other_dist"]
        ]

        path = reconstruct_path(
            parent,
            current_agent_pos,
            (fx, fy)
        )

        cached_paths.append(path)
        if (path == []):
            print("EMPTY PATH: ", current_agent_pos, "to" , (fx,fy), "with value", ag_occ[fx][fy])
            visualize_occ(ag_occ)

    # Pad to exactly 5 actions
    while len(cached_paths) < 5:

        cached_paths.append([])

    return observation, cached_paths

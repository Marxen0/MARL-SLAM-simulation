import heapq
import numpy as np
from map_generator import visualize_occ
from scipy import ndimage
import heapq
import numpy as np


def dijkstra_all(start, occ, free_value=(0,2)):
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
            if occ[nx, ny] not in free_value:
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
    if occ[fx][fy] == 2: return 1 #if the frontier is on a friend path return the value to 1
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
def get_frontiers(occ, Free_value=(0,2), unknown_value=-1):

    frontiers = []

    width, height = occ.shape

    for x in range(1, width - 1):
        for y in range(1, height - 1):

            if occ[x, y] not in Free_value:
                continue

            neighbors = [
                occ[x+1,y],
                occ[x-1,y],
                occ[x,y+1],
                occ[x,y-1]
            ]

            if unknown_value in neighbors:
                frontiers.append((x, y))

    return np.array(frontiers)
def euclidean_distance_calculation(pos1, pos2):

    pos1 = np.array(pos1)
    pos2 = np.array(pos2)

    return np.linalg.norm(pos1 - pos2)
def filter_reachable_frontiers(
    frontiers,
    dist
):


    reachable_frontiers = []

    for fx, fy in frontiers:

        if (fx, fy) in dist:
            reachable_frontiers.append((fx, fy))

    return reachable_frontiers
def get_frontier_metrics(
    reachable_frontiers,
    dist,
    ag_pos,
    ag_target,
    agent_idx,
    map_occ
):

    frontier_metrics = []
    # --------------------------
    # Distance to other agents
    # --------------------------
    other_positions = [
        pos
        for i, pos in enumerate(ag_pos)
        if i != agent_idx
    ]
    # --------------------------
    # Distance to other targets
    # --------------------------
    other_targets = [
        target
        for i, target in enumerate(ag_target)
        if i != agent_idx and target is not None
    ]
    for fx, fy in reachable_frontiers:

        dist_to_agent = dist[(fx, fy)]

        if len(other_targets) > 0:

            closest_other_target_dist = min(
                euclidean_distance_calculation(
                    (fx, fy),
                    target
                )
                for target in other_targets
            )

        else:

            closest_other_target_dist = 999

        

        if len(other_positions) > 0:

            closest_other_pos_dist = min(
                euclidean_distance_calculation(
                    (fx, fy),
                    pos
                )
                for pos in other_positions
            )

        else:

            closest_other_pos_dist = 999

        value = compute_frontier_value(
            fx,
            fy,
            map_occ
        )

        frontier_metrics.append({
            "coord": (fx, fy),
            "value": value,
            "dist_to_agent_dijkstra": dist_to_agent,
            "closest_other_pos_dist": closest_other_pos_dist,
            "closest_other_target_dist": closest_other_target_dist
        })

    return frontier_metrics
def get_chosen_frontiers(frontier_metrics):

    closest_2 = sorted(
        frontier_metrics,
        key=lambda x: x["dist_to_agent_dijkstra"]
    )[:2]

    highest_2 = sorted(
        frontier_metrics,
        key=lambda x: x["value"],
        reverse=True
    )[:2]

    farthest_1 = sorted(
        frontier_metrics,
        key=lambda x: x["closest_other_pos_dist"],
        reverse=True
    )[:1]

    return (
        closest_2 +
        highest_2 +
        farthest_1
    )
def get_direction(from_pos, to_pos):
    """
    Compute direction vector from one position to another.

    Args:
        from_pos: (x, y)
        to_pos: (x, y)

    Returns:
        (dx, dy)
    """

    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]

    return dx, dy
def get_other_agent_dijkstra(dist, agent_idx, ag_pos):
    other_agent_dijkstra = []

    for i, pos in enumerate(ag_pos):

        if i == agent_idx:
            continue

        pos = tuple(map(int, pos))

        if pos in dist:
            other_agent_dijkstra.append(dist[pos])
        else:
            other_agent_dijkstra.append(999)
    return other_agent_dijkstra
def dijkstra_overlap_percentage(path_a, path_b):
    """
    Compute how much path_a overlaps with path_b.

    Args:
        path_a: [(x, y), ...]
        path_b: [(x, y), ...]

    Returns:
        float in [0, 1]
    """

    if len(path_a) == 0 or len(path_b) == 0:
        return 0.0

    set_a = set(path_a)
    set_b = set(path_b)

    overlap = len(set_a & set_b)

    # Percentage relative to path_a
    return overlap / len(path_a)
def observation(agent_idx, ag_pos, ag_occ, ag_target):
    #SAFETY
    if len(ag_target)<len(ag_pos): print("ag_target is smaller than ag_pos")
    for x in ag_target:
        if len(x) == 0 or x == None: print("ag_target is empty?? ",ag_target)

    current_agent_pos = tuple(map(int, ag_pos[agent_idx]))
    dist, parrent = dijkstra_all(current_agent_pos, ag_occ)
    
    ag_dis_ag = get_other_agent_dijkstra(dist, agent_idx, ag_pos)
    
    frontiers = get_frontiers(ag_occ)
    if len(frontiers)==0: 
        frontier_features = []
        action_mask = []
        for _ in range(5):
            frontier_features.append(
                empty_frontier_feature(agent_idx)
            )
            action_mask.append(False)

        obs = build_observation(
            other_agent_dijkstra=ag_dis_ag,
            frontier_features=frontier_features
        )
        return obs, action_mask
    
    reachable_frontiers =filter_reachable_frontiers(frontiers, dist)
    if len(reachable_frontiers)==0:
        frontier_features = []
        action_mask = []
        for _ in range(5):
            frontier_features.append(
                empty_frontier_feature(agent_idx)
            )
            action_mask.append(False)

        obs = build_observation(
            other_agent_dijkstra=ag_dis_ag,
            frontier_features=frontier_features
        )
        return obs, action_mask
    
    frontier_metrics = get_frontier_metrics(
        reachable_frontiers,
        dist,
        ag_pos,
        ag_target,
        agent_idx,
        ag_occ
    )
    chosen_frontiers = get_chosen_frontiers(frontier_metrics)

    action_mask = []
    frontier_features = []
    for frontier in chosen_frontiers:
        fx, fy = frontier["coord"]
        path = reconstruct_path(parrent, current_agent_pos, (fx,fy))
        agent_features = []
        for i, pos in enumerate(ag_pos):
            if i==agent_idx : continue
            pos_tuple = tuple(map(int, pos))
            ag_to_ag_path = reconstruct_path(parrent, current_agent_pos, pos_tuple)
            ag_target_dir = get_direction(pos_tuple, ag_target[i])
            ag_frontier_dir = get_direction(pos_tuple, (fx,fy))
            agent_feature = build_agent_feature(
                target_to_frontier_euclidean=euclidean_distance_calculation(ag_target[i],(fx,fy)),
                agent_to_frontier_euclidean=euclidean_distance_calculation(pos_tuple, (fx,fy)),
                dijkstra_sum_ag=frontier["dist_to_agent_dijkstra"]+dist[pos_tuple],
                dijkstra_overlap_percent_ag=dijkstra_overlap_percentage(path, ag_to_ag_path),
                ag_target_dx=ag_target_dir[0],
                ag_target_dy=ag_target_dir[1],
                ag_to_frontier_dx=ag_frontier_dir[0],
                ag_to_frontier_dy=ag_frontier_dir[1]

            )
            agent_features.append(agent_feature)
        self_direction = get_direction(current_agent_pos, (fx,fy))
        frontier_feature = build_frontier_feature(
            agent_self=agent_idx,
            frontier_value=frontier["value"],
            frontier_position=frontier["coord"],
            self_dx= self_direction[0],
            self_dy= self_direction[1],
            self_dijkstra=frontier["dist_to_agent_dijkstra"],
            cached_path = path,
            agent_features=agent_features
        )
        frontier_features.append(frontier_feature)
        action_mask.append(True)

    while len(frontier_features) < 5:

        frontier_features.append(
            empty_frontier_feature(agent_idx)
        )

        action_mask.append(False)

    obs = build_observation(
        other_agent_dijkstra=ag_dis_ag,
        frontier_features=frontier_features
    )

    return obs

def build_agent_feature(
    target_to_frontier_euclidean,
    agent_to_frontier_euclidean,
    dijkstra_sum_ag,
    dijkstra_overlap_percent_ag,
    ag_target_dx,
    ag_target_dy,
    ag_to_frontier_dx,
    ag_to_frontier_dy
):
    return {
        "target_to_frontier_euclidean": target_to_frontier_euclidean,
        "agent_to_frontier_euclidean": agent_to_frontier_euclidean,
        "dijkstra_sum_ag": dijkstra_sum_ag,
        "dijkstra_overlap_percent_ag": dijkstra_overlap_percent_ag,
        "ag_target_dx": ag_target_dx,
        "ag_target_dy": ag_target_dy,
        "ag_to_frontier_dx": ag_to_frontier_dx,
        "ag_to_frontier_dy": ag_to_frontier_dy,
    }

def empty_frontier_feature(agent_idx):
    return build_frontier_feature(
        agent_self=agent_idx,
        frontier_value=0,
        frontier_position=(-1, -1),
        self_dx=0,
        self_dy=0,
        self_dijkstra=999,
        cached_path=[],
        agent_features=[]
    )
def build_frontier_feature(
    agent_self,
    frontier_value,
    frontier_position,
    self_dx,
    self_dy,
    self_dijkstra,
    cached_path,
    agent_features
):
    return {
        "agent_self" : agent_self,
        "frontier_value": frontier_value,
        "self_dx": self_dx,
        "self_dy": self_dy,
        "self_dijkstra": self_dijkstra,
        "frontier_position" : frontier_position,

        "agents": agent_features,
        "cached_paht" : cached_path
    }


def build_observation(
    other_agent_dijkstra,
    frontier_features
):
    return {
        "other_agent_dijkstra": other_agent_dijkstra,
        "frontiers": frontier_features,
    }

if __name__ == "__main__":

    # ----------------------------
    # Test map
    # ----------------------------
    #
    # X = wall
    # . = free
    #
    #  0 1 2 3 4
    #0 X X X X X
    #1 X . . . X
    #2 X . X . X
    #3 X . . . X
    #4 X X X X X
    #
    # Agent0 = (1,1)
    # Agent1 = (3,3)
    #

    occ = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ])

    ag_pos = np.array([
        [1, 1],   # Agent 0
        [3, 3],   # Agent 1
    ])

    # Pretend agent 1 targets (3,1)
    ag_target = [
        [],       # self (agent 0)
        (3, 1)
    ]

    print("=" * 50)
    print("DIJKSTRA TEST")
    print("=" * 50)

    dist, parent = dijkstra_all((1, 1), occ)

    print("Distance to (3,3):", dist[(3, 3)])
    print("Expected: 4")

    path = reconstruct_path(
        parent,
        (1, 1),
        (3, 3)
    )

    print("Path:")
    print(path)

    print()

    print("=" * 50)
    print("EUCLIDEAN TEST")
    print("=" * 50)

    d = euclidean_distance_calculation(
        (1, 1),
        (3, 3)
    )

    print("Euclidean distance:", d)
    print("Expected:", np.sqrt(8))

    print()

    print("=" * 50)
    print("DIRECTION TEST")
    print("=" * 50)

    dx, dy = get_direction(
        (1, 1),
        (3, 3)
    )

    print("Direction:", (dx, dy))
    print("Expected: (2,2)")

    print()

    print("=" * 50)
    print("OVERLAP TEST")
    print("=" * 50)

    path_a = [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 3),
    ]

    path_b = [
        (1, 1),
        (1, 2),
        (2, 2),
        (3, 2),
    ]

    overlap = dijkstra_overlap_percentage(
        path_a,
        path_b
    )

    print("Overlap:", overlap)
    print("Expected: 0.5")

    print()

    print("=" * 50)
    print("FRONTIER TEST")
    print("=" * 50)

    # Unknown area to create frontiers
    occ_frontier = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, -1, -1, 1],
        [1, 0, 0, -1, -1, 1],
        [1, 0, 0,  0,  0, 1],
        [1, 1, 1,  1,  1, 1],
    ])

    frontiers = get_frontiers(occ_frontier)

    print("Frontiers found:")
    print(frontiers)

    print()

    print("=" * 50)
    print("FULL OBSERVATION TEST")
    print("=" * 50)

    ag_pos = np.array([
        [1, 1],   # self
        [3, 3],   # other agent
    ])

    ag_target = [
        [2, 1],       # self has no target yet
        (3, 1)    # agent 1 target
    ]

    obs = observation(
        agent_idx=0,
        ag_pos=ag_pos,
        ag_occ=occ_frontier,
        ag_target=ag_target
    )

    print("Observation:")
    print(obs)

    print()

    print("=" * 50)
    print("CHECK FRONTIER FEATURES")
    print("=" * 50)

    for i, frontier in enumerate(obs["frontiers"]):

        print(f"\nFrontier {i}")

        print("Position:",
              frontier["frontier_position"])

        print("Value:",
              frontier["frontier_value"])

        print("Self direction:",
              (frontier["self_dx"],
               frontier["self_dy"]))

        print("Self dijkstra:",
              frontier["self_dijkstra"])

        print("Cached path:")
        print(frontier["cached_paht"])

        print("\nOther agents:")

        for j, agent in enumerate(frontier["agents"]):

            print(f"Agent {j}")

            print(
                "target_to_frontier_euclidean:",
                agent["target_to_frontier_euclidean"]
            )

            print(
                "agent_to_frontier_euclidean:",
                agent["agent_to_frontier_euclidean"]
            )

            print(
                "dijkstra_sum_ag:",
                agent["dijkstra_sum_ag"]
            )

            print(
                "dijkstra_overlap_percent_ag:",
                agent["dijkstra_overlap_percent_ag"]
            )

            print(
                "ag_target_dir:",
                (
                    agent["ag_target_dx"],
                    agent["ag_target_dy"]
                )
            )

            print(
                "ag_frontier_dir:",
                (
                    agent["ag_to_frontier_dx"],
                    agent["ag_to_frontier_dy"]
                )
            )

    print()
    print("=" * 50)
    print("ALL TESTS FINISHED")
    print("=" * 50)
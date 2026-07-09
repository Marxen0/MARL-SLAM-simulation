import numpy as np


def encode_agent_feature(agent_feature):

    return [
        agent_feature["target_to_frontier_euclidean"],
        agent_feature["agent_to_frontier_euclidean"],
        agent_feature["dijkstra_sum_ag"],
        agent_feature["dijkstra_overlap_percent_ag"],
        agent_feature["ag_target_dx"],
        agent_feature["ag_target_dy"],
        agent_feature["ag_to_frontier_dx"],
        agent_feature["ag_to_frontier_dy"],
    ]


def encode_frontier_feature(frontier):

    features = [

        frontier["frontier_value"],

        frontier["self_distance"],
        frontier["self_dx"],
        frontier["self_dy"],

        frontier["self_dijkstra"],
    ]

    for agent in frontier["agents"]:
        features.extend(
            encode_agent_feature(agent)
        )

    return features


def encode_observation(obs, time_step):

    features = []

    # Other agent distances
    features.extend(
        obs["other_agent_dijkstra"]
    )
    max_value = np.max(obs["other_estimate_occ"])
    if max_value > 0:
        obs["other_estimate_occ"] /= max_value
    features.extend(
        obs["other_estimate_occ"].ravel() ### HOW DO I ADD THIS UPDATE THIS
    )
    features.extend(
        obs["agents_pos"]
    )
    features.extend(
        obs["agents_target_pos"]
    )

    # Frontiers
    for frontier in obs["frontiers"]:

        features.extend(
            encode_frontier_feature(frontier)
        )

    # Time feature
    features.append(
        time_step / 1000.0
    )

    return np.array(
        features,
        dtype=np.float32
    )
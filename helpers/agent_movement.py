import numpy as np

def walk_agent(world, ag_occ, ray_count, current_step_paths):
    """
    Menggerakkan setiap agent ke koordinat berikutnya dan mensimulasikan pembacaan sensor (raycasting).
    
    Args:
        world (np.ndarray): Peta asli rumah (0: kosong/jalan, 1: dinding/halangan).
        ray_count (int): Jumlah arah sinar sensor yang ditembakkan agent (360 derajat / ray_count).
        current_step_paths (list): List koordinat (x, y) tujuan langkah saat ini untuk masing-masing agent.
                                  Bentuknya: [(x1, y1), (x2, y2), ...] seukuran agent_num.
                                  
    Returns:
        tuple: (updated_ag_occ, updated_ag_pos)
            - updated_ag_occ: Peta okupansi terbaru untuk masing-masing agent (shape: agent_num, width, height)
            - updated_ag_pos: Koordinat posisi terbaru semua agent (shape: agent_num, 2)
    """
    world_width, world_height = world.shape
    agent_num = len(current_step_paths)
    
    # Inisialisasi array untuk posisi baru dan peta okupansi baru langkah ini
    updated_ag_pos = np.zeros((agent_num, 2), dtype=int)
    updated_ag_occ = ag_occ.copy()
    
    # Jarak maksimum jangkauan sensor agent (dalam pixel/grid)
    max_sensor_range = 15
    for x in range(len(ag_occ)):
        for pos in current_step_paths:
            updated_ag_occ[x][pos[0]][pos[1]] = 0
    for agent_idx, next_pos in enumerate(current_step_paths):

        # ----------------------------------
        # Update position
        # ----------------------------------
        x = int(np.clip(
            next_pos[0],
            0,
            world_width - 1
        ))

        y = int(np.clip(
            next_pos[1],
            0,
            world_height - 1
        ))

        # Prevent walking into walls
        if world[x, y] == 1:
            continue

        updated_ag_pos[agent_idx] = [x, y]

        # The agent always knows its own position is free
        updated_ag_occ[agent_idx, x, y] = 0

        # ----------------------------------
        # Raycasting
        # ----------------------------------

        angles = np.linspace(
            0,
            2 * np.pi,
            ray_count,
            endpoint=False
        )

        for angle in angles:

            for r in range(1, max_sensor_range + 1):

                ray_x = int(
                    x + r * np.cos(angle)
                )

                ray_y = int(
                    y + r * np.sin(angle)
                )

                # Out of bounds
                if (
                    ray_x < 0
                    or ray_x >= world_width
                    or ray_y < 0
                    or ray_y >= world_height
                ):
                    break

                # Store the ACTUAL occupancy value
                updated_ag_occ[
                    agent_idx,
                    ray_x,
                    ray_y
                ] = world[ray_x, ray_y]

                # Stop if a wall is hit
                if world[ray_x, ray_y] == 1:
                    break

    return updated_ag_occ, updated_ag_pos


def check_done(global_map):
    """
    Memeriksa apakah simulasi selesai berdasarkan rasio area yang sudah dijelajahi.
    Selesai (True) jika nilai bukan nol (sudah diketahui) mencapai >= 90% dari total map.
    
    Args:
        global_map (np.ndarray): Map hasil gabungan okupansi semua agent (2D array).
        
    Returns:
        bool: True jika eksplorasi mencapai >= 90%, False jika belum.
    """
    total_cells = global_map.size
    
    known_cells = np.count_nonzero(global_map != -1)
    
    # Hitung persentase area yang sudah diketahui
    coverage_ratio = known_cells / total_cells
    
    # Jika sudah mencapai atau melewati 90% (0.90), return True
    return coverage_ratio >= 0.90

import numpy as np

previous_ag_pos = None
def check_agent_movement(ag_pos):
    problem = False
    """
    Check that agents move exactly one cell per update.

    Flags:
        - Teleporting (>1 cell movement)
        - Standing still (0 cell movement)

    Args:
        ag_pos (np.ndarray):
            Shape (agent_num, 2)
    """

    global previous_ag_pos

    # First call
    if previous_ag_pos is None:
        previous_ag_pos = ag_pos.copy()
        return

    for agent_idx in range(len(ag_pos)):

        old_pos = previous_ag_pos[agent_idx]
        new_pos = ag_pos[agent_idx]

        dx = new_pos[0] - old_pos[0]
        dy = new_pos[1] - old_pos[1]

        distance = abs(dx) + abs(dy)

        if distance > 1:

            print(
                f"[TELEPORT] Agent {agent_idx}: "
                f"{tuple(old_pos)} -> {tuple(new_pos)}"
            )
            problem = True

        elif distance == 0:

            print(
                f"[STUCK] Agent {agent_idx}: "
                f"at {tuple(new_pos)}"
            )
            problem = True
    

    previous_ag_pos = ag_pos.copy()
    return problem
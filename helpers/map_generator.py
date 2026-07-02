import random
import numpy as np
def create_house(width, height, seed=None):
    """
    Generates a house environment layout using a Binary Space Partitioning (BSP) tree.
    
    Args:
        width (int): Width of the map.
        height (int): Height of the map.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        numpy.ndarray: 2D array where 1 = free space, -1 = obstacle / wall.
    """
    if seed is not None:
        random.seed(seed)

    # Initialize everything as walls (-1)
    world = np.ones((width, height), dtype=int)

    MIN_ROOM = 6
    MAX_DEPTH = 4
    DOOR_MARGIN = 2

    rooms = []

    class Node:
        def __init__(self, x, y, w, h, depth=0):
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.depth = depth
            self.left = None
            self.right = None
            self.split_vertical = None
            self.split_pos = None

        def split(self):
            if self.depth >= MAX_DEPTH:
                rooms.append(self)
                return

            if self.w < MIN_ROOM * 2 and self.h < MIN_ROOM * 2:
                rooms.append(self)
                return

            self.split_vertical = random.random() > 0.5

            if self.w > self.h:
                self.split_vertical = True
            elif self.h > self.w:
                self.split_vertical = False

            if self.split_vertical:
                if self.w < MIN_ROOM * 2:
                    rooms.append(self)
                    return

                split = random.randint(MIN_ROOM, self.w - MIN_ROOM)
                self.split_pos = self.x + split

                self.left = Node(self.x, self.y, split, self.h, self.depth + 1)
                self.right = Node(self.x + split, self.y, self.w - split, self.h, self.depth + 1)
            else:
                if self.h < MIN_ROOM * 2:
                    rooms.append(self)
                    return

                split = random.randint(MIN_ROOM, self.h - MIN_ROOM)
                self.split_pos = self.y + split

                self.left = Node(self.x, self.y, self.w, split, self.depth + 1)
                self.right = Node(self.x, self.y + split, self.w, self.h - split, self.depth + 1)

            self.left.split()
            self.right.split()

        def carve_rooms(self):
            if self.left or self.right:
                if self.left:
                    self.left.carve_rooms()
                if self.right:
                    self.right.carve_rooms()
            else:
                # Leaf room: Carve out floor paths (1 = free space)
                for x in range(self.x + 1, self.x + self.w - 1):
                    for y in range(self.y + 1, self.y + self.h - 1):
                        world[x, y] = 0

        def create_doors(self):
            if not self.left or not self.right:
                return

            if self.split_vertical:
                min_y = self.y + DOOR_MARGIN
                max_y = self.y + self.h - DOOR_MARGIN - 1

                if max_y > min_y:
                    y = random.randint(min_y, max_y)
                    x = self.split_pos

                    # Open doors in the walls (1 = free space)
                    world[x, y] = 0
                    world[x - 1, y] = 0
                    world[x - 2, y] = 0
                    world[x + 1, y] = 0
                    world[x + 2, y] = 0
            else:
                min_x = self.x + DOOR_MARGIN
                max_x = self.x + self.w - DOOR_MARGIN - 1

                if max_x > min_x:
                    x = random.randint(min_x, max_x)
                    y = self.split_pos

                    # Open doors in the walls (1 = free space)
                    world[x, y] = 0
                    world[x, y - 1] = 0
                    world[x, y - 2] = 0
                    world[x, y + 1] = 0
                    world[x, y + 2] = 0

            self.left.create_doors()
            self.right.create_doors()

    # Root node leaves a 1-cell border padding around the map bounds
    root = Node(1, 1, width - 2, height - 2)
    root.split()
    root.carve_rooms()
    root.create_doors()

    # Explicitly enforce outer perimeter walls (-1 = obstacle)
    world[0, :] = 1
    world[-1, :] = 1
    world[:, 0] = 1
    world[:, -1] = 1

    return world
import pygame
import numpy as np


class OccupancyViewer:
    """
    Real-time occupancy grid visualizer.

    Cell encoding:
        -1 = unknown (gray)
         0 = free (white)
         1 = wall (black)
    """

    def __init__(self, width, height, cell_size=20):

        pygame.init()

        self.cell_size = cell_size
        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode(
            (width * cell_size, height * cell_size)
        )

        pygame.display.set_caption("Occupancy Grid")

        self.colors = {
            -1: (128, 128, 128),   # Unknown
             0: (255, 255, 255),  # Free
             1: (0, 0, 0),         # Wall
             2: (255,255,0)
        }

    def render(
        self,
        occ,
        agent_positions=None,
        frontiers=None
    ):
        """
        Render an occupancy grid.

        Args:
            occ (np.ndarray):
                Shape (width, height)

            agent_positions (np.ndarray, optional):
                Shape (agent_num, 2)

            frontiers (np.ndarray, optional):
                Shape (N, 2) or (N, 7).

                If shape is (N, 7), columns 2 and 3 are assumed
                to be frontier_x and frontier_y.
        """

        # Keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        self.screen.fill((255, 255, 255))

        width, height = occ.shape

        # Draw occupancy grid
        for x in range(width):
            for y in range(height):

                value = int(occ[x, y])

                color = self.colors.get(
                    value,
                    (255, 0, 255)  # Invalid values
                )

                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                )

                pygame.draw.rect(
                    self.screen,
                    (200, 200, 200),
                    (
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    ),
                    1
                )

        # Draw frontiers (GREEN)
       # Draw frontiers (GREEN)
        if frontiers is not None:

            frontiers = np.asarray(frontiers)

            for frontier in frontiers:

                # Observation format:
                # [dx, dy, fx, fy, value, dist, closest_other_dist]
                if frontier.shape[0] == 6:

                    fx = int(frontier[2])
                    fy = int(frontier[3])

                # Simple frontier format:
                # [x, y]
                elif frontier.shape[0] == 2:

                    fx = int(frontier[0])
                    fy = int(frontier[1])

                else:

                    print(
                        f"Warning: Unsupported frontier format {frontier.shape}"
                    )
                    continue

                center = (
                    int(fx * self.cell_size + self.cell_size / 2),
                    int(fy * self.cell_size + self.cell_size / 2)
                )

                pygame.draw.circle(
                    self.screen,
                    (0, 255, 0),
                    center,
                    self.cell_size // 4
                )

        # Draw agents (RED)
        if agent_positions is not None:

            for i in range(len(agent_positions)):
                x,y = agent_positions[i]
                color = (255,0,0)
                if i == 0: color = (0,0,255)
                center = (
                    int(x * self.cell_size + self.cell_size / 2),
                    int(y * self.cell_size + self.cell_size / 2)
                )
                pygame.draw.circle(
                    self.screen,
                    color,
                    center,
                    self.cell_size // 3
                )

        pygame.display.flip()

        return True
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def visualize_occ(occ):
    """
    Visualize an occupancy grid with coordinates and cell values.

    Occupancy encoding:
        -1 = unknown (gray)
         0 = free (white)
         1 = wall (black)

    Args:
        occ (np.ndarray):
            2D occupancy grid.
    """

    height, width = occ.shape

    # Convert values to color indices
    # -1 -> 0 (gray)
    #  0 -> 1 (white)
    #  1 -> 2 (black)
    display_grid = occ + 1

    cmap = ListedColormap([
        "gray",   # unknown (-1)
        "white",  # free (0)
        "black"   # wall (1)
    ])

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(
        display_grid,
        cmap=cmap,
        origin="lower",
        vmin=0,
        vmax=2
    )

    # Show coordinates
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)

    ax.grid(which="minor")

    # Print the actual cell value inside each square
    for x in range(width):
        for y in range(height):

            value = int(occ[y, x])

            # White text on black walls
            text_color = "white" if value == 1 else "black"

            ax.text(
                x,
                y,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=8
            )

    ax.set_title("Occupancy Grid Debug View")
    plt.show()
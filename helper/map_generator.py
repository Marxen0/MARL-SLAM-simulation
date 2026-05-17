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
    world = np.full((width, height), -1)

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
                        world[x, y] = 1

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
                    world[x, y] = 1
                    world[x - 1, y] = 1
                    world[x - 2, y] = 1
                    world[x + 1, y] = 1
                    world[x + 2, y] = 1
            else:
                min_x = self.x + DOOR_MARGIN
                max_x = self.x + self.w - DOOR_MARGIN - 1

                if max_x > min_x:
                    x = random.randint(min_x, max_x)
                    y = self.split_pos

                    # Open doors in the walls (1 = free space)
                    world[x, y] = 1
                    world[x, y - 1] = 1
                    world[x, y - 2] = 1
                    world[x, y + 1] = 1
                    world[x, y + 2] = 1

            self.left.create_doors()
            self.right.create_doors()

    # Root node leaves a 1-cell border padding around the map bounds
    root = Node(1, 1, width - 2, height - 2)
    root.split()
    root.carve_rooms()
    root.create_doors()

    # Explicitly enforce outer perimeter walls (-1 = obstacle)
    world[0, :] = -1
    world[-1, :] = -1
    world[:, 0] = -1
    world[:, -1] = -1

    return world
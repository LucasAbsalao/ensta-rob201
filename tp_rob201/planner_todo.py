"""
Planner class
Implementation of A*
"""

import copy
import heapq
import math
from collections import defaultdict
from typing import Tuple

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.map_walls = None

    def get_neighbors(self, current_cell):
        """ Return list of free (i.e. not obstacle) neighbour cells 
            with the format of current_cell: (i, j) in the map frame
        """
        neighbor_list = []
        # TODO for TP5: iterate through neighbors and add free ones to neighbor_list
        x, y = current_cell

        for i in range(-1,2):
            for j in range(-1,2):
                nx = x + i
                ny = y + j

                if 0 <= nx < self.grid.x_max_map and 0 <= ny <self.grid.y_max_map:

                    if i==0 and j == 0:
                        continue

                    if(self.grid.occupancy_map[nx][ny] <= 0):
                        neighbor_list.append((nx, ny))
        return neighbor_list

    def heuristic(self, cell_1: Tuple[int, int], cell_2: Tuple[int, int]):
        """ Return heuristic goal distance """
        h = 0
            # TODO for TP5: compute heuristic distance between cell_1 and cell_2
        h = np.sqrt((cell_1[0] - cell_2[0])**2 + (cell_1[1] - cell_2[1])**2)
        return h

    def reconstruct_path(self, came_from, goal):
        """ Extract path after cost computation """
        total_path = [goal]
        cell = goal
        while cell in came_from.keys():
            cell = came_from[cell]
            total_path.insert(0, cell)

        total_path = np.array(total_path)
        traj_world_x, traj_world_y = self.grid.conv_map_to_world(total_path[:, 0], total_path[:, 1])
        return np.vstack((traj_world_x, traj_world_y))

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """

        start: Tuple[int, int] = self.grid.conv_world_to_map(start[0], start[1])
        goal: Tuple[int, int] = self.grid.conv_world_to_map(goal[0], goal[1])

        # creates a copy of occupancy map to modify it and take into account
        # a margin in the walls
        self.map_walls = copy.deepcopy(self.grid.occupancy_map)

        # TODO for TP5: dilate walls in self.map_walls to take into account a margin around obstacles
        #Creating binary map
        binary_map = (self.map_walls > 0).astype(np.uint8)

        #Dilating this binary map
        kernel_dilatation = np.ones((15,15), dtype=np.uint8)
        self.map_walls = cv2.dilate(binary_map, kernel_dilatation, iterations=1)
        cv2.imshow("map_walls", self.map_walls * 255)
        cv2.waitKey(1)

        # min heap to contain values to explore next
        open_set = [(0.0, start)]
        heapq.heapify(open_set)

        # dictionary to trace back route
        came_from = {}

        # cost to get to each cell
        g_score = defaultdict(lambda: math.inf)
        g_score[start] = 0.0

        # best guess of cost for each cell (cost + heuristic)
        f_score = defaultdict(lambda: math.inf)
        f_score[start] = 0.0 + self.heuristic(start, goal)

        while len(open_set) > 0:
            current = heapq.heappop(open_set)
            current_f, current_cell = current
            # lazy deletion: skip stale entries
            if current_f > f_score[current_cell]:
                continue
            if current_cell == goal:
                return self.reconstruct_path(came_from, goal)

            neighbours = self.get_neighbors(current_cell)
            for cell in neighbours:
                tentative_g_score = g_score[current_cell] + self.heuristic(current_cell, cell)
                if tentative_g_score < g_score[cell]:
                    # better path, recording it
                    came_from[cell] = current_cell
                    g_score[cell] = tentative_g_score
                    f_score[cell] = tentative_g_score + self.heuristic(cell, goal)
                    heapq.heappush(open_set, (f_score[cell], cell))

        # goal was never reached
        print('failed getting to objective')
        return None

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal

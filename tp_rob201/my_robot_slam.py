"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.simulation.robot.robot_abstract import RobotAbstract
from place_bot.simulation.robot.odometer import OdometerParams
from place_bot.simulation.ray_sensors.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0
        self.best_score = 0

        # Counter to threshold
        self.counter_threshold = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        #Safe distance to avoid obstacles
        self.d_safe = 50.0

        #Goal
        self.goal = np.array([-450,-450,0])

    def control(self):
        """
        Main control function executed at each time step
        """
        pose = self.odometer_values()

        if self.counter>=30:
            score = self.tiny_slam.localise(self.lidar(), pose)
            #print("Score: ", score)
            if score > self.best_score or self.counter_threshold > 40:
                self.best_score = score 
            
            threshold_score = 3000
            #print("Threshold: ", threshold_score)
            
            if score>threshold_score:
                self.corrected_pose = self.tiny_slam.get_corrected_pose(pose)
                self.tiny_slam.update_map_offset(self.lidar(), self.corrected_pose)
                self.counter_threshold = 0
            else:
                self.counter_threshold += 1
        else:
            self.counter+=1
            self.corrected_pose = self.tiny_slam.get_corrected_pose(pose)
            self.tiny_slam.update_map_offset(self.lidar(), self.corrected_pose)


        return self.control_tp2()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        #self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.corrected_pose

        self.occupancy_grid.display_cv(pose, self.goal)

        if(self.arrived_at_goal(pose, self.goal)):
            self.goal = self.new_goal()

        #print("Pose: ", pose, "| Goal: ", self.goal)

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(lidar = self.lidar(), current_pose = pose, goal_pose = self.goal, d_safe = self.d_safe)

        return command
    
    def arrived_at_goal(self, pose, goal):
        distance = np.linalg.norm(pose[:2] - goal[:2])
        return distance < 10


    def new_goal(self):
        laser_dist = self.lidar().get_sensor_values()
        laser_angles = self.lidar().get_ray_angles()

        laser_dist_prob = laser_dist/np.sum(laser_dist)

        new_direction = np.random.choice(len(laser_dist), 1, p = laser_dist_prob)

        print("New direction: ", new_direction, laser_angles[new_direction])

        #5.0 is a margin to avoid putting a goal in a coordinate very close to the safe distance from an obstacle
        distance_to_goal = np.random.uniform(0, max(laser_dist[new_direction] - self.d_safe - 5.0, 0)) 

        x_world = distance_to_goal * np.sin(laser_angles[new_direction])
        y_world = distance_to_goal * np.sin(laser_angles[new_direction])

        x_coord = x_world.squeeze()
        y_coord = y_world.squeeze()

        return np.array([self.corrected_pose[0] + x_coord, self.corrected_pose[1] + y_coord, 0])



        
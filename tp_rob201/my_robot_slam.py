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
from planner_todo import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams(),
                 gaussian_modele_probabiliste: bool = False):
        # Passing parameter to parent class
        super().__init__(lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0
        self.best_score = 0

        # Counter to threshold
        self.counter_threshold = 0

        # Exploring boolean variable
        self.exploring = True

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

        self.tiny_slam = TinySlam(self.occupancy_grid, gaussian_b=gaussian_modele_probabiliste)
        self.planner = Planner(self.occupancy_grid)
        self.plan = None

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        #Safe distance to avoid obstacles
        self.d_safe = 30.0

        #Goal
        self.goal = np.array([-50,-500,0])
        self.goal_count = 0

    def control(self):
        """
        Main control function executed at each time step
        """
        pose = self.odometer_values()

        if self.exploring:
            if self.counter>=30:
                score = self.tiny_slam.localise(self.lidar(), pose)
                #print("Score: ", score)
                if score > self.best_score:
                    self.best_score = score 
                
                threshold_score = 6000
                #print("Threshold: ", threshold_score)
                
                if score>threshold_score:
                    self.corrected_pose = self.tiny_slam.get_corrected_pose(pose)
                    self.tiny_slam.update_map_offset(self.lidar(), self.corrected_pose)

            else:
                self.corrected_pose = self.tiny_slam.get_corrected_pose(pose)
                self.tiny_slam.update_map_offset(self.lidar(), self.corrected_pose)

            self.counter+=1
        else:
            if self.plan is None:
                self.plan = self.planner.plan(self.corrected_pose, np.array([0,0,0]))

                if self.plan is not None:
                    idx = np.arange(self.plan.shape[1]) % 10    
                    idx = idx==0
                    idx[-1] = True
                    self.plan = self.plan[:,idx]
                    print(self.plan)

                else:
                    print("Path didn't find")
                    self.plan = np.array([[],[]])


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
        pose = self.odometer_values()

        #EXPLORING MODE
        if self.exploring:
            pose = self.corrected_pose

            if(self.arrived_at_goal(pose, self.goal)):
                if self.goal_count >= 5:
                    self.goal = np.array([0,0,0])
                    self.exploring = False
                else:
                    self.goal = self.new_goal()
                    self.goal_count += 1
            
            # Compute new command speed to perform obstacle avoidance
            command = potential_field_control(lidar = self.lidar(), current_pose = pose, goal_pose = self.goal, d_safe = self.d_safe)

        else:
            if self.plan is not None and self.plan.shape[1] > 0:
                pose = self.odometer_values()

                waypoint_tolerance = 10.0 
                
                while self.plan.shape[1] > 0:
                    target_pose = np.array([self.plan[0,0], self.plan[1,0], 0.0])
                    dist_to_wp = np.linalg.norm(pose[:2] - target_pose[:2])
                    
                    if dist_to_wp < waypoint_tolerance:
                        # This point is very close to our actual position
                        self.plan = self.plan[:, 1:]
                    else:
                        # The point is sufficiently far
                        break

                if self.plan.shape[1] == 0:
                    command = {"forward": 0.0, "rotation": 0.0}

                else:
                    next_target = np.array([self.plan[0,0], self.plan[1,0], 0.0])
                    command = potential_field_control(lidar=self.lidar(), current_pose=pose, goal_pose=next_target, d_safe=self.d_safe/3)

            else:
                # Se não tem plano ou a lista já acabou, fica totalmente parado
                command = {"forward": 0.0, "rotation": 0.0}

        if self.plan is not None:
            self.occupancy_grid.display_cv(pose, self.goal, self.plan)
        else:
            self.occupancy_grid.display_cv(pose, self.goal)

        return command
    
    def arrived_at_goal(self, pose, goal):
        distance = np.linalg.norm(pose[:2] - goal[:2])
        return distance < 10


    def new_goal(self):
        laser_dist = self.lidar().get_sensor_values()
        laser_angles = self.lidar().get_ray_angles()

        sum_dist = np.sum(laser_dist)
        if sum_dist == 0:
            laser_dist_prob = np.ones(len(laser_dist)) / len(laser_dist)
        else:
            laser_dist_prob = laser_dist / sum_dist

        new_direction = np.random.choice(len(laser_dist), p = laser_dist_prob)

        angle_local = laser_angles[new_direction]
        chosen_dist = laser_dist[new_direction]
        print("New direction: ", chosen_dist, angle_local)

        #5.0 is a margin to avoid putting a goal in a coordinate very close to the safe distance from an obstacle
        distance_to_goal = np.random.uniform(0, max(chosen_dist - self.d_safe - 5.0, 0)) 

        x_world = self.corrected_pose[0] + distance_to_goal * np.cos(angle_local + self.corrected_pose[2])
        y_world = self.corrected_pose[1] + distance_to_goal * np.sin(angle_local + self.corrected_pose[2])

        return np.array([x_world, y_world, 0])


    

        
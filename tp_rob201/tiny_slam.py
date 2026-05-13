""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid, gaussian_b: bool):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

        self.gaussian = gaussian_b


    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4
        
        distances = lidar.get_sensor_values()
        idx = distances<lidar.max_range
        distances = distances[idx]
        angles = lidar.get_ray_angles()[idx]

        x_o = distances * np.cos(angles + pose[2])
        y_o = distances * np.sin(angles + pose[2])

        x_absolu = pose[0] + x_o 
        y_absolu = pose[1] + y_o

        x_map, y_map = self.grid.conv_world_to_map(x_absolu, y_absolu)
        # print("Without constraints: ")
        # print(x_map)
        # print(y_map)
        # print("limits: ", self.grid.x_max_map, self.grid.y_max_map)

        x_idx = (x_map<self.grid.x_max_map) & (x_map>=0)
        y_idx = (y_map<self.grid.y_max_map) & (y_map>=0)
        # print("After constraints: ")
        # print(x_idx)
        # print(y_idx)
        idx = x_idx & y_idx
        #print("idx: ", idx)

        score = np.sum(self.grid.occupancy_map[x_map[idx], y_map[idx]])

        # print("Calculated: ", score)
        # print()

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref
        
        if len(odom_pose.shape)>1:
            x_o, y_o, theta_o = odom_pose[:,0], odom_pose[:,1], odom_pose[:,2] 
        else:
            x_o, y_o, theta_o = odom_pose

        x_o_ref, y_o_ref, theta_o_ref = odom_pose_ref


        alpha = np.arctan2(y_o , x_o )
        d = np.hypot(y_o, x_o)  

        x = x_o_ref + d * np.cos(theta_o_ref + alpha)
        y = y_o_ref + d * np.sin(theta_o_ref + alpha)
        theta = theta_o + theta_o_ref

        if len(odom_pose.shape)>1:
            return np.column_stack((x,y,theta))
        else:
            return np.array([x,y,theta])

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        corrected_raw_pose = self.get_corrected_pose(raw_odom_pose)

        best_score = self._score(lidar, corrected_raw_pose)
        #print("Initial_score: ", best_score)

        iterations = 300
        #odom_pose_ref_base = np.copy(self.odom_pose_ref)
        for i in range(iterations):
            samples = np.random.normal(0, [5.0,5.0,np.pi/5])
            odom_pose_ref_test = self.odom_pose_ref + samples

            corrected_pose = self.get_corrected_pose(raw_odom_pose, odom_pose_ref_test)

            new_score = self._score(lidar, corrected_pose)

            if new_score > best_score:
                self.odom_pose_ref = odom_pose_ref_test
                best_score = new_score
                
            
        return best_score


    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3
        idx = lidar.get_sensor_values()<lidar.max_range
        values = lidar.get_sensor_values()[idx] #+ np.random.normal(0, 0.5, size=len(lidar.get_sensor_values()))
        x = np.cos(lidar.get_ray_angles()[idx]+pose[2]) * values + pose[0]
        y = np.sin(lidar.get_ray_angles()[idx]+pose[2]) * values + pose[1]

        for i in range(len(x)):
            self.grid.add_value_along_line(pose[0], pose[1], x[i], y[i], -0.95)
        self.grid.add_map_points(x, y, 6)


        np.clip(self.grid.occupancy_map, -40, 40, out=self.grid.occupancy_map)

    def update_map_offset(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3

        distances = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()

        idx = distances<lidar.max_range

        safe_distances = np.minimum(distances, lidar.max_range)

        x = np.cos(angles + pose[2]) * distances + pose[0]
        y = np.sin(angles + pose[2]) * distances + pose[1]

        for i in range(len(x)):
            if idx[i]:
                if self.gaussian:
                    self.grid.add_value_along_line_gaussian(pose[0], pose[1], x[i], y[i], -0.95, val_wall=6, offset=2, sigma=0.5)
                else: 
                    self.grid.add_value_along_line_offset(pose[0], pose[1], x[i], y[i], -0.95, val_wall=0, offset=5)
            else:
                self.grid.add_value_along_line(pose[0], pose[1], x[i], y[i], -0.95)

        if not self.gaussian:
            self.grid.add_map_points(x[idx], y[idx], 6)

        np.clip(self.grid.occupancy_map, -40, 40, out=self.grid.occupancy_map)
        

    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

        # Poor implementation of polar to cartesian conversion
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x, pt_y])

        # pt_x = ranges * np.cos(ray_angles)
        # pt_y = ranges * np.sin(ray_angles)
        # points = np.vstack((pt_x, pt_y))

""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    laser_dist = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    speed = 1.0
    rotation_speed = 0.0

    indexes = []
    for i in range(len(angles)):
        if angles[i] < 0.2 and angles[i] > -0.2:
            indexes.append(i)

    for i in indexes:
        if laser_dist[i] < 50.0:
            rotation_speed = np.random.uniform(0, 1)
    
    print(rotation_speed)

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command


def potential_field_control(lidar, current_pose, goal_pose, d_safe = 50.0):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2
    dlim = 40
    anglelim = np.pi*3/4

    k_attraction = 1.0
    k_repulsion  = 50000

    #Calculating the distance between the current pose and the goal
    diff = goal_pose[:2] - current_pose[:2]
    distance_absolut = np.linalg.norm(diff)

    # Calculating attraction gradient 
    if distance_absolut > dlim:
        gradient = k_attraction*diff/distance_absolut
    else:
        gradient = k_attraction*diff/dlim

    #Calculating repulsion gradient
    closest_obstacle_index = np.argmin(lidar.get_sensor_values())
    closest_distance = lidar.get_sensor_values()[closest_obstacle_index]

    closest_angle = lidar.get_ray_angles()[closest_obstacle_index]
    angle_global = closest_angle + current_pose[2]


    negative_gradient = 0
    if closest_distance <= d_safe:
        x_obstacle = - np.cos(angle_global) * closest_distance
        y_obstacle = - np.sin(angle_global) * closest_distance
        diff_repulsion = np.array([x_obstacle, y_obstacle], dtype=np.float64)
        negative_gradient = k_repulsion/(closest_distance**3) * ((1/closest_distance) - (1/d_safe)) * diff_repulsion


    gradient = gradient + negative_gradient

    #Calculating rotation speed based in the gradient
    rotation_speed = np.arctan2(gradient[1],gradient[0]) - current_pose[2]
    rotation_speed = (rotation_speed + np.pi) % (2 * np.pi) - np.pi #Normalizing

    #Setting new speed. If the curve is too strict, reduce speed through k_vitesse
    k_vitesse = anglelim/abs(rotation_speed) if abs(rotation_speed)>anglelim else 1
    forward_speed = 0.4 * np.linalg.norm(gradient) * k_vitesse

    #Clip speed and rotations
    forward_speed = np.clip(forward_speed, -1.0, 1.0)
    rotation_speed = np.clip(rotation_speed, -1.0 ,1.0)


    command = {"forward": forward_speed,
               "rotation": rotation_speed}

    return command

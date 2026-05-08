import numpy as np

raw_odom_pose = np.array([1.2,2.2,3.2])

new_angles = np.random.normal(raw_odom_pose[2], [np.pi/8,4], size=(2,50)) 
#print(new_angles)

print(new_angles)
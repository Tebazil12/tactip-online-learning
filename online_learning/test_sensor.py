import tactip_toolkit_dobot.experiments.min_example.common as common
from tactip_toolkit_dobot.experiments.online_learning.contour_following_2d import (
    make_meta,
)

meta = make_meta()

with common.make_sensor(meta) as sensor:
    keypoints = sensor.process(meta["num_frames"])
    print("this is: ", keypoints)
    print(len(keypoints[0]))

# import matplotlib.pyplot as plt
#
# for i in range(0,10):
#     plt.scatter(keypoints[i][:,0],keypoints[i][:,1] )
# # plt.plot(keypoints[1][:,0],keypoints[1][:,1] )
# plt.scatter(keypoints[0][:,0],keypoints[0][:,1] )
# plt.axis('equal')
# plt.ion()
# # plt.show(block = False)
# plt.pause(1)
# print("now done")

import matplotlib.pyplot as plt

for i in range(0, len(keypoints[0])):
    plt.scatter(keypoints[:, i, 0], keypoints[:, i, 1])
    plt.plot(keypoints[:, i, 0], keypoints[:, i, 1])
# plt.plot(keypoints[1][:,0],keypoints[1][:,1] )
plt.scatter(keypoints[0][:, 0], keypoints[0][:, 1])
plt.axis("equal")
# plt.ion()
# plt.show(block = False)
# plt.pause(1)
plt.show()
print("now done")

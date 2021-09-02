import numpy as np
import matplotlib.pyplot as plt

import tactip_toolkit_dobot.experiments.min_example.common as common
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp

data = common.load_data("/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_3d_2021y-09m-01d_10h29m14s/data_line_010.json")
data = np.array(data)
neutral_tap = common.load_data("/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_3d_2021y-09m-01d_10h29m14s/neutral_tap.json")
neutral_tap = np.array(neutral_tap)
neutral_tap= np.reshape(neutral_tap, [int(np.shape(neutral_tap)[0]/2),2])

ref_tap = common.load_data("/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_3d_2021y-09m-01d_10h29m14s/ref_tap.json")
ref_tap = np.array(ref_tap)
ref_tap= np.reshape(ref_tap, [int(np.shape(ref_tap)[0]/2),2])

print(np.shape(data))
print(np.shape(neutral_tap))
# show plot of taps taken (note, this pauses program until graph is closed)
for i, tap in enumerate(data):
    plt.subplot(2, np.ceil(np.shape(data)[0]/2), i+1)
    plt.scatter(neutral_tap[:, 0], neutral_tap[:, 1], marker='x')
    plt.scatter(ref_tap[:, 0]+neutral_tap[:, 0], ref_tap[:, 1]+neutral_tap[:, 1], marker='+')
    for frame in tap:

        plt.scatter(frame[:, 0], frame[:, 1])
    plt.axis("equal")
plt.show()

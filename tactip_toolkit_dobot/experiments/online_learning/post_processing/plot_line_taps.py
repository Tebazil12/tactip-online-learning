import numpy as np
import matplotlib.pyplot as plt

import tactip_toolkit_dobot.experiments.min_example.common as common
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp

data = common.load_data("/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_2d_01m-18d_16h02m52s/data_line_001.json")
data = np.array(data)
neutral_tap = common.load_data("/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_2d_01m-18d_16h02m52s/neutral_tap.json")
neutral_tap = np.array(neutral_tap)
neutral_tap= np.reshape(neutral_tap, [int(np.shape(neutral_tap)[0]/2),2])

print(np.shape(data))
print(np.shape(neutral_tap))
# show plot of taps taken (note, this pauses program until graph is closed)
for i, tap in enumerate(data):
    plt.subplot(2, round(np.shape(data)[0]/2), i+1)
    plt.scatter(neutral_tap[:, 0], neutral_tap[:, 1], marker='x')
    for frame in tap:

        plt.scatter(frame[:, 0], frame[:, 1])
    plt.axis("equal")
plt.show()

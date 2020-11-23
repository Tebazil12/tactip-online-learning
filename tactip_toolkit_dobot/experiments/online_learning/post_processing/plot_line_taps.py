import numpy as np
import matplotlib.pyplot as plt

import tactip_toolkit_dobot.experiments.min_example.common as common
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp

data = common.load_data("/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_2d_11m-17d_17h20m10s/data_line_001.json")
data = np.array(data)

print(np.shape(data))
# show plot of taps taken (note, this pauses program until graph is closed)
for frame in data[0]:
    plt.scatter(frame[:, 0], frame[:, 1])
plt.axis("equal")
plt.show()

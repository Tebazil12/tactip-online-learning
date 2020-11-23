import numpy as np
import matplotlib.pyplot as plt

import tactip_toolkit_dobot.experiments.min_example.common as common
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp

# here both data and ref tap are relative to neutral tap (NB, these are from different datasets, where neurtal tap will be different
data = common.load_data(
    "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_2d_11m-17d_14h19m40s/data.json"
)
ref_tap = common.load_data(
    "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_2d_11m-13d_12h00m22s/ref_tap.json"
)
meta = common.load_data(
    "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_2d_11m-17d_14h19m40s/meta.json"
)


dissim_profile = dp.calc_dissims(np.array(data), np.array(ref_tap))



corrected_disps, offset = dp.align_radius(np.array(meta["line_range"]), dissim_profile)

# print(data)
# print(np.shape(data))

print(dissim_profile)
print(corrected_disps)
print(offset)

plt.plot(meta["line_range"], dissim_profile)
plt.plot(corrected_disps, dissim_profile)
plt.show()

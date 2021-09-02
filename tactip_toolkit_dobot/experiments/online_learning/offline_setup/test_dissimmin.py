import numpy as np
import matplotlib.pyplot as plt

import tactip_toolkit_dobot.experiments.min_example.common as common
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp

# here both data and ref tap are relative to neutral tap (NB, these are from different datasets, where neurtal tap will be different
data = common.load_data(
    # "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_2d_11m-17d_14h19m40s/data.json"
    "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_3d_2021y-09m-01d_11h18m48s/gplvm_000.json"
)
data= data['y']



ref_tap = common.load_data(
    # "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_2d_11m-13d_12h00m22s/ref_tap.json"
    "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_3d_2021y-09m-01d_11h18m48s/ref_tap.json"

)
meta = common.load_data(
    # "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_2d_11m-17d_14h19m40s/meta.json"
    "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_3d_2021y-09m-01d_11h18m48s/meta.json"

)

len_height = len(meta["height_range"])
# print(len_height)

dissim_profile = dp.calc_dissims(np.array(data)[:-len_height], np.array(ref_tap))



corrected_disps1, offset = dp.align_radius(np.array(meta["line_range"]), dissim_profile, gp_extrap=True, show_graph=True)

# print(data)
# print(np.shape(data))

print(dissim_profile)

print(f"Offeset predicted: {offset} , giving disps of: {corrected_disps1}")
# print(corrected_disps)
# print(offset)



corrected_disps2, offset = dp.align_radius(np.array(meta["line_range"]), dissim_profile, gp_extrap=False, show_graph=True)

# print(data)
# print(np.shape(data))

print(dissim_profile)

print(f"Offeset predicted: {offset} , giving disps of: {corrected_disps2}")
# print(corrected_disps)
# print(offset)

# plt.plot(meta["line_range"], dissim_profile, 'g')
plt.plot(corrected_disps1, dissim_profile, 'r')
plt.plot(meta["line_range"], dissim_profile, 'g')
plt.plot(corrected_disps2, dissim_profile, 'b')
ax = plt.gca()
ax.axhline(y=0, color="k")
ax.axvline(x=0, color="k")

plt.show()

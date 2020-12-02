import tactip_toolkit_dobot.experiments.min_example.common as common
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.gplvm as gplvm
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp
import numpy as np

# load gplvm from json file
data = common.load_data(
    "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/contour_following_2d_11m-30d_16h16m55s/gplvm_000.json"
)


# print(old_model_data)
# print(old_model_data['x'])

# init model with data from file
model = gplvm.GPLVM(
    np.array(data["x"]),
    np.array(data["y"]),
    data["sigma_f"],
    np.array(data["ls"]),
)

# load new data (as though taking a second line elsewhere)
# for now, use the same data just to check shapes of everything, to make sure
# no huge errors
disps = np.array([
    [-10.0],
    [-8.0],
    [-6.0],
    [-4.0],
    [-2.0],
    [0.0],
    [2.0],
    [4.0],
    [6.0],
    [8.0],
    [10.0],
])

y = np.array(data["y"])
y.sort() # try to get some variation from the same dataset...
y[2] = y[6] # try for more variation...

# y =

print(np.shape(y))

# test optim_line_mu
# optimise mu of line given old data and hyperpars
optm_mu = model.optim_line_mu(disps, y)

# print(optm_mu)

x_line = dp.add_line_mu(disps, optm_mu)

print(f"final x for line: {x_line}")

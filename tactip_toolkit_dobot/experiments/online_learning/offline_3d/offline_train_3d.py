import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.gplvm as gplvm
import tactip_toolkit_dobot.experiments.min_example.common as common
from tactip_toolkit_dobot.experiments.online_learning.contour_following_2d import (
    Experiment,
    make_meta,
    plot_gplvm,
    State,
    parse_exp_name,
)

def at_line_extract(local, data, meta):
    # return the data at given height and angle, local=[height, angle(in deg)]

    heights = np.array(meta["height_range"])
    num_heights = len(heights)
    angles = np.array(meta["angle_range"])
    num_angles = len(angles)
    real_disp = np.array(meta["line_range"])
    num_disps = len(real_disp)

    height_index = np.where(heights == local[0])[0][0]
    print(f"height index {height_index}")
    angle_index =np.where(angles == local[1])[0][0]
    print(f"angle index {angle_index}")

    index = angle_index + ((height_index)*num_angles)

    print(f"getting index {index}")

    the_slice = data[index]
    return the_slice


def at_plane_extract(local, data, meta, method):
    pass

def main(ex, meta):

    path = data_home + current_experiment

    neutral_tap = np.array(common.load_data(path + "neutral_tap.json"))
    ref_tap = np.array(common.load_data(path + "ref_tap.json"))

    locations = np.array(common.load_data(path + "post_processing/all_locations.json"))
    lines = np.array(common.load_data(path + "post_processing/all_lines.json"))
    dissims = np.array(common.load_data(path + "post_processing/dissims.json"))

    optm_disps = np.array(
        common.load_data(path + "post_processing/corrected_disps_basic.json")
    )


    heights = meta["height_range"]
    num_heights = len(heights)
    angles = meta["angle_range"]
    num_angles = len(angles)
    real_disp = meta["line_range"]  # nb, not copied so that reverse is persistent
    real_disp.reverse()  # to match previous works (-ve on obj, +ve free)
    num_disps = len(real_disp)

    training_local_1 = [0,0]
    new_taps = at_line_extract(training_local_1, lines, meta)
    adjusted_disps = at_line_extract(training_local_1, optm_disps, meta)

    print(f"lengths are {len(new_taps)} and {len(adjusted_disps)}")

    # if state.model is None:
    #     print("Model is None, mu will be 1")
    #     # set mus to 1 for first line only - elsewhere mu is optimised
    #     x_line = dp.add_line_mu(adjusted_disps, 1)
    #
    #     # init model (sets hyperpars)
    #     state.model = gplvm.GPLVM(x_line, np.array(new_taps))
    #     model = state.model


    # best_frames = dp.best_frame()

    # print(model["ls"])
    # state.model = gplvm.GPLVM(
    #     np.array(model["x"]),
    #     np.array(model["y"]),
    #     sigma_f=model["sigma_f"],
    #     ls=model["ls"],
    # )
    # print(state.model.x)
    # #
    # #
    # plot_gplvm(state.model, meta)


if __name__ == "__main__":

    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
    )
    current_experiment = "collect_dataset_3d_21y-03m-03d_15h18m06s/"

    state = State(meta=common.load_data(data_home + current_experiment + "meta.json"))

    print(f"Dataset: {current_experiment} using {state.meta['stimuli_name']}")

    state.ex = Experiment()
    main(state.ex, state.meta)

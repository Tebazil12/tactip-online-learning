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


class Plane:
    disps = None  # known/estimated disps to edge
    heights = None  # robot / estimated z location
    phis = None  # known / estimated sensor angle wrt edge

    x = None  # combination -> x = [locs, phis, heights]

    y = None  # the (processed) pin data at these points

    def __init__(self):
        pass

    def make_x(self):
        # combine locs and phis (if needed, heights) - reuse other funciotn?
        self.x = np.concatenate(
            (np.array([self.disps]).T, np.array([self.heights]).T, np.array([self.phis]).T),
            axis=1,
        )
        print(f"x is shape: {np.shape(self.x)}")

    def make_all_heights(self, height):
        # make heights the same length as disp and set to given height

        if self.disps is None:
            raise NameError("disps must be defined before all_heights can be used")

        self.heights = [height] * len(self.disps)

    def make_all_phis(self, phi):
        # make heights the same length as disp and set to given height

        if self.disps is None:
            raise NameError("disps must be defined before all_phis can be used")

        self.phis = [phi] * len(self.disps)


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
    angle_index = np.where(angles == local[1])[0][0]
    print(f"angle index {angle_index}")

    index = angle_index + ((height_index) * num_angles)

    print(f"getting index {index}")

    the_line = Plane()
    the_line.y = data[index]
    the_line.disps = real_disp
    the_line.make_all_phis(local[1])
    the_line.make_all_heights(local[0])
    the_line.make_x()

    return the_line


def at_plane_extract(local, data, meta, method="cross", cross_length=None):
    base_line = at_line_extract(local, data, meta)

    local = np.array(local)

    if method == "cross":
        # take the points cross_length up and cross_length down (in height)
        # from center of line

        other_lines = []
        other_ys = []
        height_step = 0.5 # todo, extract from meta

        for i in np.arange(height_step, (cross_length*height_step)+0.000001, height_step): #todo, map to .5mm prooperly
            other_lines.append(at_line_extract(local + np.array([i, 0]), data, meta))
            other_ys.append(other_lines[-1].y)
            other_lines.append(at_line_extract(local + np.array([-i, 0]), data, meta))
            other_ys.append(other_lines[-1].y)

        other_ys = np.array(other_ys)

        index_to_take = int(np.ceil(
            (len(base_line.y) / 2) - 1
        ))  # rounds down in case of even num of taps (which really shouldn't be the case

        print(f"index: {index_to_take}")

        print(f"other ys shape: {np.shape(other_ys)} slice: {np.shape(other_ys[:,index_to_take])}")
        print(f"shape of base_line.y {np.shape(base_line.y)}")


        for line in other_lines:
            # print("---")
            # print(f"line.y is shape {np.shape(line.y)}")
            # new_point = np.array([line.y[index_to_take]])
            # print(f"new point {np.shape(new_point)}")

            base_line.y = np.concatenate((base_line.y, np.array([line.y[index_to_take]])), axis=0)
            #todo same for disp etc


            # print(f"shape of y : {np.shape(base_line.y)}")



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

    training_local_1 = [0, 0]
    new_taps = at_line_extract(training_local_1, lines, meta).y
    adjusted_disps = at_line_extract(training_local_1, optm_disps, meta).y

    print(f"lengths are {len(new_taps)} and {len(adjusted_disps)}")

    new_taps_plane = at_plane_extract(
        training_local_1, lines, meta, method="cross", cross_length=2
    )

    print(f"new taps plane done")

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

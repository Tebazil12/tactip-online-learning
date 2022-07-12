# To use this file, data must be pre-processed by running the following files:
# > graph_dataset.py

# NB this file will overwrite graphs in main dataset, so graph_dataset.py needs
# to be rerun afterwards to correct the graphs

import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import interpolate

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

from tactip_toolkit_dobot.experiments.online_learning.contour_following_3d import (
    plot_dissim_grid,
)

from tactip_toolkit_dobot.experiments.online_learning.offline_3d.offline_train_3d import (
    Plane,
    get_calibrated_plane,
    at_plane_extract,
    extract_point_at,
)

import tactip_toolkit_dobot.experiments.online_learning.offline_3d.graph_dataset as graph_dataset


def main(ex, meta, original_ref=True, grid_graphs_on=True):

    # load data
    path = data_home + current_experiment

    # neutral_tap = np.array(common.load_data(path + "neutral_tap.json"))
    #
    # locations = np.array(common.load_data(path + "post_processing/all_locations.json"))
    lines = np.array(common.load_data(path + "post_processing/all_lines.json"))

    # if original_ref:
    #     ref_tap = np.array(common.load_data(path + "ref_tap.json"))
    #     dissims = np.array(common.load_data(path + "post_processing/dissims.json"))
    # else:
    #     pass
    #
    # optm_disps = np.array(
    #     common.load_data(path + "post_processing/corrected_disps_basic.json")
    # )

    # heights = meta["height_range"]
    # num_heights = len(heights)
    # angles = meta["angle_range"]
    # num_angles = len(angles)
    # real_disp = meta["line_range"]
    # num_disps = len(real_disp)

    # Find index location of disp minima
    # training_local_1 = [0, 5]  # [height(in mm), angle(in deg)]

    heights = np.array(meta["height_range"])
    num_heights = len(heights)
    heights_step = heights[1] - heights[0]

    angles = np.array(meta["angle_range"])
    num_angles = len(angles)
    angle_step = angles[1] - angles[0]

    real_disp = np.array(meta["line_range"])
    num_disps = len(real_disp)
    disps_step = real_disp[1] - real_disp[0]

    print(f"heights = {heights}")
    print(f"angles = {angles}")
    print(f"disps = {real_disp}")

    for k in range(-1, 1 + 1):  # how many indexs to go either side of where 0 is
        for j in ["h", "a", "d"]:
            if j == "h":
                i = k * heights_step
                local = [i, 0, 0]  # height, angle, disp (in mm and deg)
            elif j == "a":
                i = k * angle_step
                local = [0, i, 0]
            elif j == "d":
                i = k * disps_step
                local = [0, 0, i]
            # print(f"local: {local} local[1]: {local[1]}")

            height_index = np.where(heights == local[0])[0][0]
            # print(f"height index {height_index}")

            angle_index = np.where(angles == local[1])[0][0]
            # print(f"angle index {angle_index}")

            disp_index = np.where(real_disp == local[2])[0][0]
            # print(f"angle index {disp_index}")

            index = angle_index + ((height_index) * num_angles)

            # print(f"getting index {index}")
            #
            # print(f"shape of lines {lines.shape}")
            # print(f"data:  {lines[index]}")

            new_ref_tap = lines[index][disp_index]

            # print(f"new ref tap {new_ref_tap}")

            # save ref tap to seperate file
            part_path, _ = os.path.split(meta["meta_file"])
            full_path = os.path.join(
                meta["home_dir"], part_path, "post_processing/alt_ref_taps/"
            )
            isExist = os.path.exists(full_path)
            if not isExist:
                os.makedirs(full_path)

            ref_file_name = (
                "post_processing/alt_ref_taps/ref_tap_" + j + str(i) + ".json"
            )
            common.save_data(
                new_ref_tap,
                meta,
                ref_file_name,
            )

            # call code in graph_dataset.py with the new reftap
            # WARNING: overwrites graphs with new ref tap
            graph_dataset.main(
                ex,
                meta,
                data_home=data_home,
                current_experiment=current_experiment,
                alt_ref=ref_file_name,
                show_figs=False
            )


if __name__ == "__main__":

    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
    )
    # current_experiment = "collect_dataset_3d_21y-03m-03d_15h18m06s/"
    # current_experiment = "collect_dataset_3d_21y-11m-19d_12h24m42s/"
    # current_experiment = "collect_dataset_3d_21y-11m-22d_16h10m54s/"
    #
    # current_experiment = "collect_dataset_3d_21y-12m-07d_16h00m01s/"
    # current_experiment = "collect_dataset_3d_21y-12m-07d_15h24m32s/"
    current_experiment = "collect_dataset_3d_21y-12m-07d_12h33m47s/"

    state = State(meta=common.load_data(data_home + current_experiment + "meta.json"))

    print(f"Dataset: {current_experiment} using {state.meta['stimuli_name']}")

    # reverse real displacements so when main is run twice, the change is not reverted
    real_disp = state.meta["line_range"]  # nb, not copied so that reverse is persistent
    real_disp.reverse()  # to match previous works (-ve on obj, +ve free)

    state.ex = Experiment()

    # main(state.ex, state.meta,train_or_test="train", train_folder="model_two_cross/")
    # main(state.ex, state.meta,train_or_test="test_line_angles",train_folder="model_two_cross/")
    # main(state.ex, state.meta,train_or_test="test_single_taps", train_folder="model_two_cross/")

    # main(state.ex, state.meta,train_or_test="train", train_folder="model_one_grid/")
    # main(state.ex, state.meta,train_or_test="test_line_angles",train_folder="model_one_grid/")
    # main(state.ex, state.meta,train_or_test="test_single_taps", train_folder="model_one_grid/")

    # main(state.ex, state.meta,train_or_test="train", train_folder="model_two_grid/")
    # main(state.ex, state.meta,train_or_test="test_line_angles",train_folder="model_two_grid/")
    main(state.ex, state.meta)

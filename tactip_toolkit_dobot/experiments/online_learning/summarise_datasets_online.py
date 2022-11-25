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

from tabulate import tabulate


def main(ex, meta):

    pass

    # # load data
    # path = data_home + current_experiment
    #
    # neutral_tap = np.array(common.load_data(path + "neutral_tap.json"))
    # ref_tap = np.array(common.load_data(path + "ref_tap.json"))
    #
    # locations = np.array(common.load_data(path + "post_processing/all_locations.json"))
    # lines = np.array(common.load_data(path + "post_processing/all_lines.json"))
    # dissims = np.array(common.load_data(path + "post_processing/dissims.json"))
    #
    # optm_disps = np.array(
    #     common.load_data(path + "post_processing/corrected_disps_basic.json")
    # )
    #
    # heights = meta["height_range"]
    # num_heights = len(heights)
    # angles = meta["angle_range"]
    # num_angles = len(angles)
    # real_disp = meta["line_range"]
    # num_disps = len(real_disp)
    #
    # # Find index location of disp minima
    # # training_local_1 = [0, 5]  # [height(in mm), angle(in deg)]
    #
    # heights_at_mins = []
    # disps_at_mins = []


if __name__ == "__main__":

    data_home = (
        # "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2023/"
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
    )



    datasets = []
    for subdir, dirs, files in os.walk(data_home):
        # print(subdir)
        if subdir.split('/')[-1] != "post_processing" and subdir.split('/')[-1] != "":
            # print(subdir.split('/')[-1])
            current_experiment = subdir.split('/')[-1] + "/"
            # print(current_experiment)
            datasets.append(current_experiment)

    # datasets = [
    #     # "collect_dataset_3d_21y-03m-03d_15h18m06s/",
    #     "collect_dataset_3d_21y-11m-19d_12h24m42s/",
    #     "collect_dataset_3d_21y-11m-22d_16h10m54s/",
    #     "collect_dataset_3d_21y-12m-07d_16h00m01s/",
    #     "collect_dataset_3d_21y-12m-07d_15h24m32s/",
    #     "collect_dataset_3d_21y-12m-07d_12h33m47s/",
    #     "collect_dataset_3d_22y-08m-09d_10h18m30s/",
    # ]

    to_tabulate = []

    for i in datasets:
        try:
            current_experiment = i
            # print(i)

            meta = common.load_data(data_home + current_experiment + "meta.json")

            gplvm = common.load_data(data_home + current_experiment + "gplvm_final.json")

            edge_data = common.load_data(data_home + current_experiment + "all_edge_locs_final.json")
            # print(len(edge_data))

            if True:#
            # if len(edge_data) == meta["MAX_STEPS"]: # if successful experiment


                headers = [
                    "name",
                    "stimuli_name",
                    "stimuli_height",
                    "ref_plat_height",
                    "work_frame_offset",
                    "line_range & step",
                    "height_range & step",
                    "step_length",
                    "plane_method",
                    "total_taps_in_gplvm",
                    "taps_per_plane",
                    "num_of_planes",
                    "tol",
                    "height_tol"
                ]

                metrics = [
                    i,
                    meta["stimuli_name"],
                    meta["stimuli_height"],
                    meta["ref_plat_height"],
                    f"x:{meta['work_frame_offset'][0]} y:{meta['work_frame_offset'][1]}",
                    f"{meta['line_range'][0]} to {meta['line_range'][-1]}, {meta['line_range'][1] - meta['line_range'][0]} ",
                    f"{meta['height_range'][0]} to {meta['height_range'][-1]}, {meta['height_range'][1] - meta['height_range'][0]} ",
                    # f"{meta['angle_range'][0]} to {meta['angle_range'][-1]}, {meta['angle_range'][1] - meta['angle_range'][0]} ",
                    meta["STEP_LENGTH"],
                    meta["plane_method"],
                    len(gplvm['x']),
                    len(meta["line_range"]) * len(meta["height_range"]),
                    len(gplvm['x']) / (len(meta["line_range"]) * len(meta["height_range"])),
                    meta["tol"],
                    meta["tol_height"]

                ]
                to_tabulate.append(metrics)
        except:
            "failed, oh well"
        #     data = [
        #     [1, "Liquid", 24, 12],
        #     [2, "Virtus.pro", 19, 14],
        #     [3, "PSG.LGD", 15, 19],
        #     [4, "Team Secret", 10, 20],
        # ]

    to_tabulate_sorted = sorted(to_tabulate, key=lambda x: x[1], reverse=True)

    print(tabulate(to_tabulate_sorted, headers=headers))

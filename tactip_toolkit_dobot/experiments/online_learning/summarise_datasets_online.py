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
        if subdir.split('/')[-1] != "post_processing" and subdir.split('/')[-1] != "" and subdir.split('/')[-1].split('_')[2] == "3d":
            # print(subdir.split('/')[-1])
            current_experiment = subdir.split('/')[-1] + "/"
            # print(current_experiment)
            datasets.append(current_experiment)

    # datasets = [
    #     # "collect_dataset_3d_21y-03m-03d_15h18m06s/",
    #     # "collect_dataset_3d_21y-11m-19d_12h24m42s/",
    #     # "collect_dataset_3d_21y-11m-22d_16h10m54s/",
    #     # "collect_dataset_3d_21y-12m-07d_16h00m01s/",
    #     # "collect_dataset_3d_21y-12m-07d_15h24m32s/",
    #     # "collect_dataset_3d_21y-12m-07d_12h33m47s/",
    #     # "collect_dataset_3d_22y-08m-09d_10h18m30s/",
    #
    #     # "contour_following_3d_2023y-02m-01d_15h20m26s/",
    #     # "contour_following_3d_2023y-02m-01d_15h03m56s/",
    #     # "contour_following_3d_2023y-02m-01d_15h10m31s/",
    #     # "contour_following_3d_2023y-02m-01d_15h26m36s/",
    #     # "contour_following_3d_2022y-08m-08d_15h05m19s/",
    #     # "contour_following_3d_2022y-08m-08d_15h15m18s/",
    #     # "contour_following_3d_2023y-02m-01d_10h56m31s/",
    #     # "contour_following_3d_2023y-02m-01d_11h05m16s/",
    #     # "contour_following_3d_2022y-08m-04d_15h44m53s/",
    #     # "contour_following_3d_2022y-09m-13d_14h38m49s/",
    #     # "contour_following_3d_2023y-02m-20d_15h11m38s/",
    #     # "contour_following_3d_2023y-02m-20d_15h25m58s/"
    #
    #
    #      "contour_following_2d_01m-19d_10h47m37s/",
    # "contour_following_2d_01m-18d_17h41m48s/",
    # "contour_following_2d_01m-22d_14h58m05s/",
    # "contour_following_2d_2021y-01m-25d_17h37m24s/",
    # "contour_following_2d_2021y-01m-25d_18h08m31s/",
    # "contour_following_2d_2021y-01m-26d_15h13m00s/",
    # "contour_following_3d_2021y-08m-11d_11h54m37s/",
    #  "contour_following_3d_2021y-08m-13d_15h51m47s/",
    #  "contour_following_3d_2021y-08m-16d_15h44m12s/", #20 deg down
    #  "contour_following_3d_2021y-08m-13d_15h51m47s/", #10 deg down
    # "contour_following_3d_2021y-09m-02d_18h40m35s/", #slide
    #
    # "contour_following_3d_2022y-07m-29d_11h50m44s/",
    # "contour_following_3d_2022y-07m-29d_14h29m56s/",
    #
    # "contour_following_3d_2022y-08m-02d_14h03m55s/",
    # "contour_following_3d_2022y-08m-03d_16h57m30s/",
    # "contour_following_3d_2022y-09m-05d_16h39m47s/",
    #
    # "contour_following_3d_2022y-09m-08d_10h25m19s/",
    # "contour_following_3d_2022y-09m-08d_12h54m35s/",
    # "contour_following_3d_2022y-09m-08d_12h47m46s/",
    # "contour_following_3d_2022y-09m-08d_12h44m04s/",
    # "contour_following_3d_2022y-09m-05d_15h23m19s/",
    # "contour_following_3d_2022y-09m-09d_17h05m32s/",
    # "contour_following_3d_2022y-09m-09d_15h49m55s/",
    # "contour_following_3d_2022y-09m-10d_11h22m36s/",
    # "contour_following_3d_2022y-09m-10d_12h30m36s/",
    # "contour_following_3d_2022y-09m-10d_16h29m37s/",
    # "contour_following_3d_2022y-08m-08d_15h05m19s/",
    # "contour_following_3d_2022y-08m-05d_14h58m18s/",
    #
    # ### white lid thesis ###
    # "contour_following_3d_2022y-09m-13d_14h38m49s/",
    #
    # "contour_following_3d_2022y-08m-04d_15h44m53s/",
    #
    #
    # "contour_following_3d_2022y-09m-05d_14h08m17s/",
    # "contour_following_3d_2022y-08m-16d_14h06m41s/",
    # "contour_following_3d_2022y-08m-16d_14h17m02s/",
    # "contour_following_3d_2022y-08m-08d_15h15m18s/",
    # ]

    to_tabulate_full = []
    to_tabulate_part_cross = []
    to_tabulate_part_grid = []

    for i in datasets:
        try:
            current_experiment = i
            print(i)

            meta = common.load_data(data_home + current_experiment + "meta.json")

            gplvm = common.load_data(data_home + current_experiment + "gplvm_final.json")

            edge_data = common.load_data(data_home + current_experiment + "all_edge_locs_final.json")
            # print(len(edge_data))

            edge_data_height = common.load_data(data_home + current_experiment + "all_edge_heights_final.json")


            if meta["stimuli_name"] == "wavy-line-thin" or \
                    meta["stimuli_name"] == "wavy-line-thick" or\
                    meta["stimuli_name"] == "banana-screwed" or\
                    meta["stimuli_name"] == "105mm-circle" or\
                    meta["stimuli_name"] == "flower" or\
                    meta["stimuli_name"] == "squishy-brick":
                # print(" this is a 2d wave")
                # print(edge_data)
                # print(edge_data_height)
                error_in_height = np.round(np.mean(np.abs(edge_data_height)), 2)
                # print("hello")
                # print(error_in_height)
                # print("world")
                print(f"stats {meta['stimuli_name']} {meta['plane_method']} error in height = {error_in_height}")

            if True:#
            # if len(edge_data) == meta["MAX_STEPS"]: # if successful experiment

            # if current_experiment.split('-')[2].split("_")[0] == "16d":
                if meta["stimuli_name"] == "high-squishy-brick":
                    meta["stimuli_name"] = "squishy brick"
                elif meta["stimuli_name"] == "strapped-banana":
                    meta["stimuli_name"] = "banana"
                elif meta["stimuli_name"] == "105mm-circle":
                    meta["stimuli_name"] = "circle (105mm)"

                if meta["plane_method"] == "cross":
                    taps_per_plane = len(meta["line_range"]) + len(meta["height_range"])

                    headers_part_cross = [
                        "stimuli_name",
                        "vertical_error",
                        "num_of_planes",
                        "taps_per_plane",
                        "total_taps_in_gplvm",


                        "line_range & step",
                        "height_range & step",
                        "dataset name",
                    ]

                    metrics_part_cross = [
                        meta["stimuli_name"] ,#+ ' {tiny ' + i +"}",
                        error_in_height,
                        len(gplvm['x']) / taps_per_plane,
                        taps_per_plane,
                        len(gplvm['x']),


                        f"{meta['line_range'][0]} to {meta['line_range'][-1]}, {meta['line_range'][1] - meta['line_range'][0]} ",
                        f"{meta['height_range'][0]} to {meta['height_range'][-1]}, {meta['height_range'][1] - meta['height_range'][0]} ",
                        i,
                    ]

                elif meta["plane_method"] == "full_grid":
                    taps_per_plane = len(meta["line_range"]) * len(meta["height_range"])

                    headers_part_grid = [
                        "stimuli_name",
                        "vertical_error",
                        "num_of_planes",
                        "taps_per_plane",
                        "total_taps_in_gplvm",


                        "line_range & step",
                        "height_range & step",
                        # "dataset name",
                    ]

                    metrics_part_grid = [
                        meta["stimuli_name"] ,#+ ' {tiny ' + i +"}",
                        error_in_height,
                        len(gplvm['x']) / taps_per_plane,
                        taps_per_plane,
                        len(gplvm['x']),


                        f"{meta['line_range'][0]} to {meta['line_range'][-1]}, {meta['line_range'][1] - meta['line_range'][0]} ",
                        f"{meta['height_range'][0]} to {meta['height_range'][-1]}, {meta['height_range'][1] - meta['height_range'][0]} ",
                        # i,
                    ]
                else:
                    asdf



                headers_full = [
                    "dataset name",
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

                metrics_full = [
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
                    taps_per_plane,
                    len(gplvm['x']) / taps_per_plane,
                    meta["tol"],
                    meta["tol_height"]

                ]

                # Stimuli &
                # No. of Training phases &
                # No. of taps per Training phase &
                # Total no. of taps in model &
                # Lateral Range and Resolution (mm) &
                # Vertical Range and Resolution (mm)\\



                to_tabulate_full.append(metrics_full)
                to_tabulate_part_cross.append(metrics_part_cross)
                to_tabulate_part_grid.append(metrics_part_grid)
        except Exception as e:
            print(f"failed, oh well{e}")
        #     data = [
        #     [1, "Liquid", 24, 12],
        #     [2, "Virtus.pro", 19, 14],
        #     [3, "PSG.LGD", 15, 19],
        #     [4, "Team Secret", 10, 20],
        # ]

    np.set_printoptions(precision=1)

    # to_tabulate_sorted = sorted(to_tabulate_full, key=lambda x: x[1], reverse=True)
    to_tabulate_sorted_full = sorted(to_tabulate_full, key=lambda x: x[0], reverse=True)

    tmp = set()
    # a = [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]]
    for i in to_tabulate_part_grid:
        tmp.add(tuple(i))
    to_tabulate_part_grid =list(tmp)

    tmp = set()
    # a = [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]]
    for i in to_tabulate_part_cross:
        tmp.add(tuple(i))
    to_tabulate_part_cross =list(tmp)

    to_tabulate_sorted_part_cross = sorted(to_tabulate_part_cross, key=lambda x: (x[0], -x[3]))
    to_tabulate_sorted_part_grid = sorted(to_tabulate_part_grid, key=lambda x: (x[0], -x[3]))

    # print(tabulate(to_tabulate_sorted_full, headers=headers_full, tablefmt="latex"))
    print("full grid")
    print(tabulate(to_tabulate_sorted_part_grid, headers=headers_part_grid, tablefmt="latex"))
    print("cross")
    print(tabulate(to_tabulate_sorted_part_cross, headers=headers_part_cross, tablefmt="latex"))

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

    # data = [
    #     [1, "Liquid", 24, 12],
    #     [2, "Virtus.pro", 19, 14],
    #     [3, "PSG.LGD", 15, 19],
    #     [4, "Team Secret", 10, 20],
    # ]
    # print(tabulate(data, headers=["Pos", "Team", "Win", "Lose"]))

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
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
    )

    current_experiment = "collect_dataset_3d_21y-03m-03d_15h18m06s/"

    datasets = ["model_one_grid/", "model_two_grid/", "model_one_cross/", "model_two_cross/"]

    means=[]
    perc50 = []
    perc75 = []
    perc90 = []

    means_r=[]
    perc50_r = []
    perc75_r = []
    perc90_r = []

    to_tabulate = []
    for i in datasets:
        data_set = i
        # print(i)

        results = common.load_data(data_home + current_experiment + "post_processing/" + data_set + "single_tap_results.json")

        real = np.array(results["real"])
        optm = np.array(results["optm"])

        # # 0 = height 1 = angle 2 = disp
        # things = np.where((real[:, 1] < -15) | (real[:, 1] > 15))[0] ## original
        # things = np.where((real[:, 0] < -1) | (real[:, 0] > 0.5))[0]
        things = np.where((real[:, 2] < -4) | (real[:, 2] > 4))[0]
        print(things)



        #map angle to phi
        # -45:45 , 0:2
        real[:,1] = (real[:,1]+45) /45
        # optm[:,1] = (optm[:,1]*45) - 45
        # reduced_real[:,1] = (reduced_real[:,1]+45) /45

        reduced_real = np.delete(real, things, axis=0)
        reduced_optm =np.delete(optm, things, axis=0)
        print("optm",optm)
        print(real)
        print(reduced_real)

        errors =real- optm
        reduced_errors = reduced_real-reduced_optm

        print("len errors", len(errors))
        print("len reduced", len(reduced_errors))


        means.append( np.mean(np.abs(errors),axis=0))
        perc50.append( np.percentile(np.abs(errors), 50,axis=0))
        perc75.append( np.percentile(np.abs(errors), 75,axis=0))
        perc90.append(  np.percentile(np.abs(errors), 90,axis=0))


        means_r.append( np.mean(np.abs(reduced_errors),axis=0))
        perc50_r.append( np.percentile(np.abs(reduced_errors), 50,axis=0))
        perc75_r.append( np.percentile(np.abs(reduced_errors), 75,axis=0))
        perc90_r.append(  np.percentile(np.abs(reduced_errors), 90,axis=0))


        # print(errors)
        # print(i )
    # print(np.array(means).T, "mean - height, angle, disp")
    # print(np.array( perc50).T, "50% - height, angle, disp")
    # print(np.array( perc75).T, "75% - height, angle, disp")
    # print(np.array( perc90).T, "90% - height, angle, disp")
    # print("\n")
    np.set_printoptions(precision=1)
    print("all")
    # all_end =np.array(means).T
    all_end =np.concatenate((np.array(means).T ,np.array( perc50).T,np.array( perc75).T,np.array( perc90).T), axis=1)

    print( all_end)
    print(tabulate(np.round(all_end,1), tablefmt="latex"))
    print("reduced")
    # all_end_reduced = np.array(means_r).T
    all_end_reduced =np.concatenate((np.array(means_r).T ,np.array( perc50_r).T,np.array( perc75_r).T,np.array( perc90_r).T), axis=1)
    print( all_end_reduced)
    print(tabulate(np.round(all_end_reduced,1), tablefmt="latex"))
        # print(np.array([means, perc50, perc75, perc90]).T)
    print("diff")
    # reduced_diff = all_end_reduced- all_end
    # print(reduced_diff)
    #
    # # reduced_diff[ np.where((reduced_diff <= -0.05))] = 'a'
    # print(type(reduced_diff))
    # reduced_diff[reduced_diff >= 0.05] = 1
    # reduced_diff[reduced_diff <= -0.05] = -1
    # print(reduced_diff)

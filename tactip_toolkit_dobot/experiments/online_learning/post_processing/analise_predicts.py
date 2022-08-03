import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Wedge
from mpl_toolkits.mplot3d import Axes3D

import tactip_toolkit_dobot.experiments.min_example.common as common
# from tactip_toolkit_dobot.experiments.online_learning.contour_following_3d import (
#     Experiment,
#     make_meta,
#     State,
#     parse_exp_name
# )
import tactip_toolkit_dobot.experiments.online_learning.contour_following_3d as online



def main(ex,meta, data_home, current_experiment, show_figs=False):

    for num in ["001","002","003","004","005"]:
        stuff = common.load_data(data_home + current_experiment + "predictions_" + num +".json")
        print(stuff)


    # ex.all_tap_positions = common.load_data(data_home + current_experiment + "all_positions_final.json")
    # ex.all_tap_positions = np.array(ex.all_tap_positions)
    #
    # ex.line_locations = common.load_data(data_home + current_experiment + "location_line_001.json")
    # ex.line_locations = np.array([ex.line_locations])
    #
    # print(ex.line_locations)
    # print(type(ex.line_locations))
    # print(np.shape(ex.line_locations))
    #
    # ex.edge_locations = common.load_data(data_home + current_experiment + "all_edge_locs_final.json")
    # ex.edge_locations = np.array(ex.edge_locations)
    #
    # ex.edge_height = common.load_data(data_home + current_experiment + "all_edge_heights_final.json")
    # ex.edge_height = np.array(ex.edge_height)
    #
    # # online.plot_all_movements(ex,meta, show_figs)
    # # online.plot_all_movements_3d(ex,meta, show_figs)
    #
    # # plot_edge_predictions(ex, meta, show_figs)
    # plot_edge_predictions_ver(ex, meta, show_figs)
    # plot_edge_predictions_lat(ex, meta, show_figs)

if __name__ == "__main__":

    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
        # "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2022/"
    )
    # current_experiment = "contour_following_2d_01m-19d_10h47m37s/"
    # current_experiment = "contour_following_2d_01m-18d_17h41m48s/"
    # current_experiment = "contour_following_2d_01m-22d_14h58m05s/"
    # current_experiment = "contour_following_2d_2021y-01m-25d_17h37m24s/"
    # current_experiment = "contour_following_2d_2021y-01m-25d_18h08m31s/"
    # current_experiment = "contour_following_2d_2021y-01m-26d_15h13m00s/"
    # current_experiment = "contour_following_3d_2021y-08m-11d_11h54m37s/"
    # current_experiment =  "contour_following_3d_2021y-08m-13d_15h51m47s/"
    # current_experiment =  "contour_following_3d_2021y-08m-16d_15h44m12s/" #20 deg down
    # current_experiment =  "contour_following_3d_2021y-08m-13d_15h51m47s/" #10 deg down
    # current_experiment = "contour_following_3d_2021y-09m-02d_18h40m35s/" #slide

    # current_experiment = "contour_following_3d_2022y-07m-29d_11h50m44s/"
    # current_experiment = "contour_following_3d_2022y-07m-29d_14h29m56s/"

    # current_experiment = "contour_following_3d_2022y-08m-02d_14h03m55s/"
    current_experiment = "contour_following_3d_2022y-08m-03d_16h57m30s/"


    state = online.State(meta=common.load_data(data_home + current_experiment + "meta.json"))

    print(state.meta["stimuli_name"])

    state.ex = online.Experiment()
    main(state.ex,state.meta, data_home, current_experiment, show_figs=True)

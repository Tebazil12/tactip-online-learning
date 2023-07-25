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


if __name__ == "__main__":

    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
    )

    datasets = [
        "collect_dataset_3d_21y-03m-03d_15h18m06s/"
    ]

    meta = common.load_data(data_home + datasets[0] + "meta.json")

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    ax=[None]*4
    (ax[0], ax[1]), (ax[2], ax[3]) = gs.subplots(sharex='col', sharey='row')
    # fig.suptitle('Sharing x per column, y per row')

    for subdir, dirs, files in os.walk(data_home + datasets[0]):
        # print(subdir)
        if subdir.split('/')[-1] != "post_processing" and subdir.split('/')[-1] != "":
            # print(subdir.split('/')[-1])
            current_experiment = subdir.split('/')[-1] + "/"
            # print(current_experiment)
            print(current_experiment)


            phi_data = common.load_data(data_home + datasets[0] + "post_processing/" + current_experiment + "optm_plane_mus.json")
            gplvm_model = common.load_data(data_home + datasets[0] + "post_processing/" + current_experiment + "gplvm_model.json")

            second_point = gplvm_model['x'][-1][2]

            # print(phi_data)
            plt.rcParams['axes.titley'] = 1.0
            plt.rcParams['axes.titlepad'] = -14

            if current_experiment == "model_one_grid/":
                i = 0
                ax[i].set_title("One Full Grid")
                ax[i].scatter([0],[1], color='brown')
            elif current_experiment == "model_two_grid/":
                i = 1
                ax[i].set_title("Two Full Grid")
                ax[i].scatter([0],[1], color='brown')
                ax[i].scatter([45],[second_point], color='orange')
                ax[i].plot([45, 0 - 45],[second_point, 2 - second_point],'k:')
            elif current_experiment == "model_one_cross/":
                i = 2
                ax[i].set_title("One Cross")
                ax[i].scatter([0],[1],color='brown')
            elif current_experiment == "model_two_cross/":
                i = 3
                ax[i].set_title("Two Cross")
                ax[i].scatter([0],[1], color='brown')
                ax[i].scatter([45],[second_point], color='orange')
                ax[i].plot([45, 0 - 45],[second_point, 2 - second_point],'k:')

            # ax[i].plot(meta["angle_range"], phi_data, marker='+',ms=7)
            # ax[i].plot(meta["angle_range"], phi_data)
            ax[i].plot(meta["angle_range"], phi_data, '+')

            # lim_y = ax[i].get_ylim()
            # ax[i].plot([0,0], lim_y, 'k')
            # ax[i].set_xlim(lim_y)

            # lim_x = ax[i].get_xlim()
            # ax[i].plot(lim_x, [0,0], 'k')

            # ax[i].set_xlim(lim_x)

            # ax[i].plot([-47,47], [0,0], 'k')
            # ax[i].plot([0,0], [-1,3.5], 'k')


    for a in fig.get_axes():
        a.label_outer()

    # fig.supxlabel("test")
    # fig.xlabel("Angle")
    # fig.ylabel("Predicted Phi")

    fig.text(0.5, 0.04, "Angle ($\degree$)", ha='center', va='center')
    fig.text(0.06, 0.5, "Predicted $\phi$", ha='center', va='center', rotation='vertical')

    full_path_png = os.path.join(
        data_home + datasets[0] + "post_processing/" + "plane_predicts.png"
    )
    full_path_svg = os.path.join(
        data_home + datasets[0] + "post_processing/" + "plane_predicts.svg"
    )
    plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
    plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    plt.show()

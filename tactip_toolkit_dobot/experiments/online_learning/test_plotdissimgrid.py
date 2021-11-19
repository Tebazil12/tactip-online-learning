import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt

import tactip_toolkit_dobot.experiments.online_learning.offline_setup.gplvm as gplvm
import tactip_toolkit_dobot.experiments.min_example.common as common
from tactip_toolkit_dobot.experiments.online_learning.contour_following_3d import (
    Experiment,
    make_meta,
    plot_dissim_grid,
    State,
)
from tactip_toolkit_dobot.experiments.online_learning.offline_3d.offline_train_3d import Plane


# np.set_printoptions(precision=2)#, suppress=True)


def main(ex, meta):
    for num in ["000","005","006"]:
        stuff = common.load_data(data_home + current_experiment + "plane_" + num +".json")

        plane = Plane()

        plane.disps = stuff["disps"]
        plane.dissims = stuff["dissims"]
        plane.x = stuff["x"]
        plane.y = stuff["y"]
        plane.heights = stuff["heights"]
        plane.phis = stuff["phis"]
        plane.x_no_mu = stuff["x_no_mu"]

        plot_dissim_grid(plane, meta, step_num_str=num, show_fig=False)

        plt.clf()
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
    # current_experiment = "contour_following_2d_01m-19d_10h47m37s/"
    # current_experiment = "contour_following_2d_01m-18d_17h41m48s/"
    # current_experiment = "contour_following_2d_01m-22d_14h58m05s/"
    # current_experiment = "contour_following_2d_2021y-01m-25d_17h37m24s/"
    # current_experiment = "contour_following_2d_2021y-01m-25d_18h08m31s/"
    # current_experiment = "contour_following_2d_2021y-01m-26d_15h13m00s/"
    # current_experiment = "contour_following_2d_01m-25d_14h42m04s/"
    # current_experiment = "contour_following_3d_2021y-08m-18d_14h53m19s/"
    # current_experiment = "contour_following_3d_2021y-08m-18d_14h25m12s/"
    # current_experiment = "contour_following_3d_2021y-08m-16d_15h44m12s/"

    current_experiment = "contour_following_3d_2021y-11m-08d_16h33m58s/"

    state = State(meta=common.load_data(data_home + current_experiment + "meta.json"))

    print(state.meta["stimuli_name"])

    state.ex = Experiment()
    main(state.ex, state.meta)

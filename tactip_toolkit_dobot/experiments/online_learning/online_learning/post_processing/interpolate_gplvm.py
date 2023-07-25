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
    plot_gplvm,
    State,
)
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.gp as gp
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp

# np.set_printoptions(precision=2)#, suppress=True)


def main(ex, meta):

    model = common.load_data(data_home + current_experiment + "gplvm_final.json")
    ref_tap = np.array(
        common.load_data(data_home + current_experiment + "ref_tap.json")
    )

    print(model["ls"])
    state.model = gplvm.GPLVM(
        np.array(model["x"]),
        np.array(model["y"]),
        sigma_f=model["sigma_f"],
        ls=model["ls"],
    )
    print(state.model.x)
    #
    #
    plot_gplvm(state.model, meta, show_fig=False)

    dissims = dp.calc_dissims(state.model.y, ref_tap, )

    disp_stars, dissim_stars = gp.interpolate(
        state.model.x,
        dissims,
        np.array(state.model.sigma_f),
        np.array(state.model.ls),
        0.379,
        x_limits=[-10, 10],
        mu=0,
        height=1
    )

    # todo, interpolate should take and return y predictions , not dissim


    # plt.plot(state.model.x, dissims)
    print(f"disp_Stars {disp_stars.shape} dissim_stars {dissim_stars.shape} ")
    plt.plot(disp_stars, dissim_stars, zs=np.ones(disp_stars.shape)) # todo why is dissim stars 3 columns? should be 1 or ~37
    # plt.plot(np.ones(10), np.ones(10), zs=np.ones(10))
    plt.show()


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
    current_experiment = "contour_following_3d_2021y-09m-23d_11h49m36s/"

    state = State(meta=common.load_data(data_home + current_experiment + "meta.json"))

    print(state.meta["stimuli_name"])

    state.ex = Experiment()
    main(state.ex, state.meta)

import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt

import tactip_toolkit_dobot.experiments.online_learning.offline_setup.gplvm as gplvm
import tactip_toolkit_dobot.experiments.min_example.common as common
from tactip_toolkit_dobot.experiments.online_learning.contour_following_2d import (
    Experiment,
    make_meta,
    plot_gplvm,
    State,
)


# np.set_printoptions(precision=2)#, suppress=True)


def main(ex, meta):

    model = common.load_data(data_home + current_experiment + "gplvm_final.json")

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
    plot_gplvm(state.model, meta)


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
    current_experiment = "contour_following_2d_01m-25d_14h42m04s/"

    state = State(meta=common.load_data(data_home + current_experiment + "meta.json"))

    print(state.meta["stimuli_name"])

    state.ex = Experiment()
    main(state.ex, state.meta)

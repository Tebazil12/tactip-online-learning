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
import tactip_toolkit_dobot.experiments.online_learning.post_processing.plot_all_movements as plot_all_movements





if __name__ == "__main__":

    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
    )

    for subdir, dirs, files in os.walk(data_home):
        print(subdir)
        if subdir.split('/')[-1] != "post_processing" and subdir.split('/')[-1] != "":
            print(subdir.split('/')[-1])
            current_experiment = subdir.split('/')[-1] + "/"
            print(current_experiment)

            try:

                state = State(meta=common.load_data(data_home + current_experiment + "meta.json"))

                print(state.meta["stimuli_name"])

                state.ex = Experiment()

                plot_all_movements.main(state.ex, state.meta, data_home, current_experiment, show_figs=False)
            except:
                print("Plot all failed, moving on")

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
import tactip_toolkit_dobot.experiments.online_learning.offline_3d.graph_dataset as graph_dataset






if __name__ == "__main__":

    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
    )
    thesis_dir = "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/thesis/"

    experiments = {
    "contour_following_3d_2023y-02m-01d_15h20m26s/",
    "contour_following_3d_2023y-02m-01d_15h03m56s/",
    "contour_following_3d_2023y-02m-01d_15h10m31s/",
    "contour_following_3d_2023y-02m-01d_15h26m36s/",
    }

    for current_experiment in experiments:

        for subdir, dirs, files in os.walk(data_home + current_experiment):
            # print(f" subs {subdir}, dirs {dirs}, files {files} ")

            for file_name in files:
                # print(file_name)
                if file_name.split('_')[0] == 'all' and  file_name.split('_')[1] == 'movements':
                    # print(file_name)
                    # print( file_name.split('.'))
                    if  file_name.split('.')[1] == 'svg':
                        print(file_name)
                        src = data_home + current_experiment + file_name
                        dst = thesis_dir + file_name

                        os.unlink(dst)
                        os.symlink(src, dst)

        print ("#####")

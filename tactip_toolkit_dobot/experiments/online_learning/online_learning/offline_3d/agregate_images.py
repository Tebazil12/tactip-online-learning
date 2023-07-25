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
    agg_dir = "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/agregated_images/"

    for subdir, dirs, files in os.walk(data_home):
        if dirs == ["post_processing"]: # TODO  make this a has rather than ==
            print(subdir)
            for file in files:
                f_name ,extension = file.split(".")
                print(extension)
                if extension == "png" or extension == "svg":

                    src = os.path.join(subdir , file)
                    print(src)
                    exp_name =  subdir.split("/")[-1]
                    if "collect_dataset_3d" in exp_name:
                        sub_dir_name = "collect_dataset_3d/"
                    dst = agg_dir +sub_dir_name+ f_name + "__" + exp_name[19:] + "." + extension
                    print(dst)

                    if not os.path.isfile(dst):
                        os.symlink(src, dst)

        # if subdir.split('/')[-1] == "post_processing":
        #     print(subdir)
        #     current_experiment = subdir.split('/')[-2] + "/"
        #     print(current_experiment)
        #     print(files)


            # state = State(meta=common.load_data(data_home + current_experiment + "meta.json"))
            #
            # print(state.meta["stimuli_name"])
            #
            # state.ex = Experiment()
            # graph_dataset.main(state.ex, state.meta, data_home=data_home,current_experiment=current_experiment, show_figs=False)

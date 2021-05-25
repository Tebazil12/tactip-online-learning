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




def main(ex, meta, train_folder="model_oneline/"):

    # load data
    path = data_home + current_experiment

    results = common.load_data(path + "post_processing/"+train_folder+"single_tap_results.json")

    # print(results)

    real = np.array(results["real"])
    optm = np.array(results["optm"])

    # remove some datapoints (change for desired analysis

    where_to_remove = np.where(
        (real[:,2] >= 8)
        | (real[:,2] <= -6)
        | (real[:,1] <= -25)
        | (real[:,1] >= 25)
    )

    print(where_to_remove)

    real = np.delete(real, where_to_remove, axis=0)
    optm = np.delete(optm, where_to_remove, axis=0)

    print(f"size of real={np.shape(real)} ")


    errors = optm - real
    print(errors)

    height_error = errors[:,0]
    angle_error = errors[:,1]
    disp_error = errors[:,2]
    print(disp_error)

    fig, axs = plt.subplots(1,3)

    for i in range(np.shape(errors)[1]):
        print(i)
        axs[i].scatter(real[:,i],optm[:,i], marker="+")

        if i == 0:
            print("### height ###")
            axs[i].set_title("Height")
            axs[i].plot([-1,0,1],[-1,0,1],'k:') #plot ideal relation # TODO extract from data
            axs[i].set_xlabel("real height (mm)")
            axs[i].set_ylabel("predicted height")
        elif i == 1:
            print("### angle ###")
            axs[i].set_title("Angle")
            axs[i].plot([-45,0,45],[0,1,2],'k:') #plot ideal relation # TODO extract from data
            axs[i].set_xlabel("real angle (degrees)")
            axs[i].set_ylabel("predicted phi")
        elif i == 2:
            print("### disp ###")
            axs[i].set_title("Disp.")
            axs[i].plot([-10,0,10],[-10,0,10],'k:') #plot ideal relation # TODO extract from data
            axs[i].set_xlabel("real disp (mm)")
            axs[i].set_ylabel("predicted disp")
        this_error = errors[:,i]

        print(this_error)
        # print(f"mean = {np.mean(this_error)}")

        print(f"abs max = {np.max(np.abs(this_error))}")
        print(f"percentile 90  = {np.percentile(np.abs(this_error),90)}")
        print(f"percentile 75  = {np.percentile(np.abs(this_error),75)}")
        print(f"abs mean = {np.mean(np.abs(this_error))}")
        print(f"percentile 25  = {np.percentile(np.abs(this_error),25)}")
        print(f"abs min = {np.min(np.abs(this_error))}")


    plt.show()


if __name__ == "__main__":

    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
    )
    current_experiment = "collect_dataset_3d_21y-03m-03d_15h18m06s/"

    state = State(meta=common.load_data(data_home + current_experiment + "meta.json"))

    print(f"Dataset: {current_experiment} using {state.meta['stimuli_name']}")

    # reverse real displacements so when main is run twice, the change is not reverted
    real_disp = state.meta["line_range"]  # nb, not copied so that reverse is persistent
    real_disp.reverse()  # to match previous works (-ve on obj, +ve free)

    state.ex = Experiment()

    # main(state.ex, state.meta,train_or_test="train")
    # main(state.ex, state.meta,train_or_test="test_line_angles")
    main(state.ex, state.meta)

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




def main(ex, meta, train_folder=""):

    # load data
    path = data_home + current_experiment

    results = common.load_data(path + "post_processing/"+train_folder+"single_tap_results.json")

    # print(results)

    real = np.array(results["real"])
    real[:,1] = (-real[:,1] / 45) *2 +1 # normalise angle so can do error thing
    optm = np.array(results["optm"])

    # remove some datapoints (change for desired analysis

    # where_to_remove = np.where(
    #     (real[:,2] >= 8)
    #     | (real[:,2] <= -6)
    #     | (real[:,1] <= -25)
    #     | (real[:,1] >= 25)
    # )
    #
    # print(where_to_remove)
    #
    # real = np.delete(real, where_to_remove, axis=0)
    # optm = np.delete(optm, where_to_remove, axis=0)

    print(f"size of real={np.shape(real)} ")


    errors = optm - real
    print(errors)

    height_error = errors[:,0]
    angle_error = errors[:,1]
    disp_error = errors[:,2]
    print(disp_error)

    height_where = np.where(
        (height_error > 1)
        | (height_error < -1)
    )

    angle_where = np.where(
        (angle_error > 2)
        | (angle_error < -2)
    )

    disp_where = np.where(
        (disp_error > 10)
        | (disp_error < -10)
    )


    # the_figure = plt.figure()
    fig, axs = plt.subplots(1,3, figsize=(30, 10))

    for i in range(np.shape(errors)[1]):
        print(i)
        axs[i].scatter(real[:,i],optm[:,i], marker="+", color="orange") # show all data

        if i != 0:
            axs[i].scatter(real[height_where,i],optm[height_where,i], marker="1", color="blue") # highlight where height is very bad
        if i != 1:
            axs[i].scatter(real[angle_where,i],optm[angle_where,i], marker="2", color="red")
        if i != 2:
            axs[i].scatter(real[disp_where,i],optm[disp_where,i], marker="3", color="green")




        if i == 0:
            print("### height ###")
            axs[i].set_title("Height")
            axs[i].plot([-1,0,1],[-1,0,1],'k:') #plot ideal relation # TODO extract from data
            axs[i].set_xlabel("real height (mm)")
            axs[i].set_ylabel("predicted height")
            axs[i].axis([ -1.2, 1.2 ,-20, 55])

            # Show the major grid lines with dark grey lines
            axs[i].grid(b=True, which="major", color="#666666", linestyle="-", alpha=0.5)
            axs[i].xaxis.set_major_locator(ticker.MultipleLocator(1))
            axs[i].yaxis.set_major_locator(ticker.MultipleLocator(1))

            # Show the minor grid lines with very faint and almost transparent grey lines
            axs[i].minorticks_on()
            axs[i].grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
            axs[i].xaxis.set_minor_locator(ticker.MultipleLocator(.5))
            axs[i].yaxis.set_minor_locator(ticker.MultipleLocator(.5))

        elif i == 1:
            print("### angle ###")
            axs[i].set_title("Angle")
            # axs[i].plot([-45,0,45],[0,1,2],'k:') #plot ideal relation # TODO extract from data
            # axs[i].plot([45,0,-45],[0,1,2],'k:') #plot ideal relation # TODO extract from data
            # axs[i].set_xlabel("real angle (degrees)")
            # axs[i].set_ylabel("predicted phi")
            # axs[i].axis([ -50, 50 ,-25, 25])
            #
            # # Show the major grid lines with dark grey lines
            # axs[i].grid(b=True, which="major", color="#666666", linestyle="-", alpha=0.5)
            # axs[i].xaxis.set_major_locator(ticker.MultipleLocator(15))
            # axs[i].yaxis.set_major_locator(ticker.MultipleLocator(2))
            #
            # # Show the minor grid lines with very faint and almost transparent grey lines
            # axs[i].minorticks_on()
            # axs[i].grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
            # axs[i].xaxis.set_minor_locator(ticker.MultipleLocator(5))
            # axs[i].yaxis.set_minor_locator(ticker.MultipleLocator(.5))

            # axs[i].plot([-1,3],[-1,3],'k:') #plot ideal relation # TODO extract from data
            axs[i].plot([3,-1],[-1,3],'k:') #plot ideal relation # TODO extract from data
            axs[i].set_xlabel("normalised real angle")
            axs[i].set_ylabel("predicted phi")
            axs[i].axis([ -1.1, 3.1 ,-25, 25])

            # Show the major grid lines with dark grey lines
            axs[i].grid(b=True, which="major", color="#666666", linestyle="-", alpha=0.5)
            axs[i].xaxis.set_major_locator(ticker.MultipleLocator(2))
            axs[i].yaxis.set_major_locator(ticker.MultipleLocator(2))

            # Show the minor grid lines with very faint and almost transparent grey lines
            axs[i].minorticks_on()
            axs[i].grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
            axs[i].xaxis.set_minor_locator(ticker.MultipleLocator(.5))
            axs[i].yaxis.set_minor_locator(ticker.MultipleLocator(.5))

        elif i == 2:
            print("### disp ###")
            axs[i].set_title("Disp.")
            axs[i].plot([-10,0,10],[-10,0,10],'k:') #plot ideal relation # TODO extract from data
            axs[i].set_xlabel("real disp (mm)")
            axs[i].set_ylabel("predicted disp")
            axs[i].axis([ -12, 12 ,-25, 25])

            # Show the major grid lines with dark grey lines
            axs[i].grid(b=True, which="major", color="#666666", linestyle="-", alpha=0.5)
            axs[i].xaxis.set_major_locator(ticker.MultipleLocator(10))
            axs[i].yaxis.set_major_locator(ticker.MultipleLocator(10))

            # Show the minor grid lines with very faint and almost transparent grey lines
            axs[i].minorticks_on()
            axs[i].grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
            axs[i].xaxis.set_minor_locator(ticker.MultipleLocator(1))
            axs[i].yaxis.set_minor_locator(ticker.MultipleLocator(1))

        this_error = errors[:,i]

        print(this_error)
        # print(f"mean = {np.mean(this_error)}")

        print(f"abs max = {np.max(np.abs(this_error))}")
        print(f"percentile 90  = {np.percentile(np.abs(this_error),90)}")
        print(f"percentile 75  = {np.percentile(np.abs(this_error),75)}")
        print(f"abs mean = {np.mean(np.abs(this_error))}")
        print(f"percentile 25  = {np.percentile(np.abs(this_error),25)}")
        print(f"abs min = {np.min(np.abs(this_error))}")



    # save graphs automatically
    part_path, _ = os.path.split(meta["meta_file"])
    full_path_png = os.path.join(meta["home_dir"], part_path, "post_processing/"+train_folder+"single_tap_predictions.png")
    full_path_svg = os.path.join(meta["home_dir"], part_path, "post_processing/"+train_folder+"single_tap_predictions.svg")
    plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
    plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)
    plt.show()

    # plt.show()


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

    main(state.ex, state.meta, train_folder="model_one_cross/")
    main(state.ex, state.meta, train_folder="model_two_cross/")

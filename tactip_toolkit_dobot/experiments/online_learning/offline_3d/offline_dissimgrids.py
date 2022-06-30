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

from tactip_toolkit_dobot.experiments.online_learning.contour_following_3d import (
    plot_dissim_grid
)

from tactip_toolkit_dobot.experiments.online_learning.offline_3d.offline_train_3d import (
    Plane,
    get_calibrated_plane,
    at_plane_extract,
    extract_point_at
)

def plot_minimas_graph(heights, disps, angles, meta, show_fig=True):
    plt.clf()
    plt.close()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    # fig.suptitle('Axes values are scaled individually by default')


    ax1.plot(angles, heights)

    ax2.plot(angles, disps)

    magnitude = np.sqrt(np.array(heights)**2 + np.array(disps)**2)
    ax3.plot(angles, magnitude)

    # TODO plotting magnitude but with visual representation of angle too


    ax4.plot(angles * something, magnitude * somethingelse)

    plt.xlabel("Angle (deg)")
    ax1.set(ylabel="Height (mm)")
    ax2.set(ylabel="Disp (mm)")
    ax3.set(ylabel="Magnitude (mm)")

    ax = plt.gca()


    # save graphs automatically
    part_path, _ = os.path.split(meta["meta_file"])

    full_path_png = os.path.join(meta["home_dir"], part_path, "min_drift.png")
    full_path_svg = os.path.join(meta["home_dir"], part_path, "min_drift.svg")

    plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
    plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)
    if show_fig:
        plt.show()
    plt.clf()
    plt.close()


def main(ex, meta):

    # load data
    path = data_home + current_experiment

    neutral_tap = np.array(common.load_data(path + "neutral_tap.json"))
    ref_tap = np.array(common.load_data(path + "ref_tap.json"))

    locations = np.array(common.load_data(path + "post_processing/all_locations.json"))
    lines = np.array(common.load_data(path + "post_processing/all_lines.json"))
    dissims = np.array(common.load_data(path + "post_processing/dissims.json"))

    optm_disps = np.array(
        common.load_data(path + "post_processing/corrected_disps_basic.json")
    )

    heights = meta["height_range"]
    num_heights = len(heights)
    angles = meta["angle_range"]
    num_angles = len(angles)
    real_disp = meta["line_range"]
    num_disps = len(real_disp)

    # Find index location of disp minima
    # training_local_1 = [0, 5]  # [height(in mm), angle(in deg)]

    heights_at_mins = []
    disps_at_mins = []

    for local in np.concatenate((np.zeros((num_angles,1),), np.array([np.arange(angles[0],angles[-1]+1,5,dtype=int)]).T), axis=1).tolist():
        # new_taps = extract_line_at(training_local_1, lines, meta).y

        # ready_plane = get_calibrated_plane(
        #     local, meta, lines, optm_disps, ref_tap, num_disps
        # )


        ready_plane = at_plane_extract(
            local,
            lines,
            meta,
            method="full",
            cross_length=2,
            cross_disp=0,
            dissims=dissims
        )


        ready_plane.make_all_phis(1)
        ready_plane.make_x()

        # make the grid graphs
        # plot_dissim_grid(ready_plane, meta, step_num_str="0"+str(int(local[1])), show_fig=False, filled=True)

        print(f"calibrated plane is: {ready_plane.__dict__}")

        plt.clf()

        # find where in height/displacement the minima dissim is
        # currently using simple method, not interpolating in any way
        index = np.argmin(ready_plane.dissims)
        height_at_min = ready_plane.heights[index]
        disp_at_min = ready_plane.disps[index]

        # save to arrays etc
        heights_at_mins.append(height_at_min)
        disps_at_mins.append(disp_at_min)


    plot_minimas_graph(heights_at_mins, disps_at_mins, angles, meta)




if __name__ == "__main__":

    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
    )
    # current_experiment = "collect_dataset_3d_21y-03m-03d_15h18m06s/"
    # current_experiment = "collect_dataset_3d_21y-11m-19d_12h24m42s/"
    # current_experiment = "collect_dataset_3d_21y-11m-22d_16h10m54s/"

    # current_experiment = "collect_dataset_3d_21y-12m-07d_16h00m01s/"
    # current_experiment = "collect_dataset_3d_21y-12m-07d_15h24m32s/"
    current_experiment = "collect_dataset_3d_21y-12m-07d_12h33m47s/"

    state = State(meta=common.load_data(data_home + current_experiment + "meta.json"))

    print(f"Dataset: {current_experiment} using {state.meta['stimuli_name']}")

    # reverse real displacements so when main is run twice, the change is not reverted
    real_disp = state.meta["line_range"]  # nb, not copied so that reverse is persistent
    real_disp.reverse()  # to match previous works (-ve on obj, +ve free)

    state.ex = Experiment()

    # main(state.ex, state.meta,train_or_test="train", train_folder="model_two_cross/")
    # main(state.ex, state.meta,train_or_test="test_line_angles",train_folder="model_two_cross/")
    # main(state.ex, state.meta,train_or_test="test_single_taps", train_folder="model_two_cross/")

    # main(state.ex, state.meta,train_or_test="train", train_folder="model_one_grid/")
    # main(state.ex, state.meta,train_or_test="test_line_angles",train_folder="model_one_grid/")
    # main(state.ex, state.meta,train_or_test="test_single_taps", train_folder="model_one_grid/")

    # main(state.ex, state.meta,train_or_test="train", train_folder="model_two_grid/")
    # main(state.ex, state.meta,train_or_test="test_line_angles",train_folder="model_two_grid/")
    main(
        state.ex,
        state.meta
    )

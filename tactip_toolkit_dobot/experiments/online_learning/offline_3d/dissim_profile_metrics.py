# To use this file, data must be pre-processed by running the following files:
# > graph_dataset.py

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





def main(ex, meta, alt_ref=None, grid_graphs_on=True, data_home=None, current_experiment=None):

    # load data
    path = data_home + current_experiment

    neutral_tap = np.array(common.load_data(path + "neutral_tap.json"))

    locations = np.array(common.load_data(path + "post_processing/all_locations.json"))
    lines = np.array(common.load_data(path + "post_processing/all_lines.json"))

    if alt_ref is None:
        ref_tap = np.array(common.load_data(path + "ref_tap.json"))
        dissims = np.array(common.load_data(path + "post_processing/dissims.json"))
    else:
        ref_tap = np.array(common.load_data(path + alt_ref))
        dissims = np.array(
            common.load_data(path + alt_ref.replace("ref_tap_", "dissims_"))
        )


    # optm_disps = np.array(
    #     common.load_data(path + "post_processing/corrected_disps_basic.json")
    # )

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

    for local in np.concatenate(
        (
            np.zeros(
                (num_angles, 1),
            ),
            np.array([np.arange(angles[0], angles[-1] + 1, 5, dtype=int)]).T,
        ),
        axis=1,
    ).tolist():
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
            dissims=dissims,
        )

        # ready_plane.make_all_phis(1)
        ready_plane.make_x()

        # if grid_graphs_on:
        #     # make the grid graphs
        #     plot_dissim_grid(
        #         ready_plane,
        #         meta,
        #         step_num_str="0" + str(int(local[1])),
        #         show_fig=False,
        #         filled=True,
        #     )

        # print(f"calibrated plane is: {ready_plane.__dict__}")

        # plt.clf()

        # print(f"ready plane {ready_plane.dissims}")
        # plt.plot(ready_plane.disps,ready_plane.dissims)

        for i in range(int(len(ready_plane.dissims)/num_disps)):
            # print(i)
            current_disps = ready_plane.disps[num_disps*i:num_disps*(i+1)]
            current_dissims = ready_plane.dissims[num_disps*i:num_disps*(i+1)]
            colour = [1- (i/num_heights),0,i/num_heights]
            plt.plot(current_disps, current_dissims, c=colour)

            dissim_average = np.mean(current_dissims)
            print(dissim_average)
            plt.plot([current_disps[0],current_disps[-1]], [dissim_average,dissim_average], c=colour)

            index = np.argmin(current_dissims)
            dissim_at_min = current_dissims[index]
            disp_at_min = current_disps[index]

            dissim_diff = dissim_average - dissim_at_min

            print(dissim_diff)
        plt.show()

    #     if True:
    #         # find where in height/displacement the minima dissim is
    #         # currently using simple method, not interpolating in any way
    #         index = np.argmin(ready_plane.dissims)
    #         height_at_min = ready_plane.heights[index]
    #         disp_at_min = ready_plane.disps[index]
    #
    #     if False:
    #         # interpolating method
    #         # can't use dp.align_radius() because we need 2d rather than 2 sets of 1D
    #         mesh_shape = (num_heights, num_disps)
    #
    #         # disps_meshed = np.reshape(ready_plane.disps , mesh_shape)
    #         # heights_meshed = np.reshape(ready_plane.heights, mesh_shape)
    #         # dissims_meshed = np.reshape(ready_plane.dissims, mesh_shape)
    #         #
    #         # print(disps_meshed)
    #         # print(heights_meshed)
    #         # print(dissims_meshed)
    #
    #         # plt.contourf(disps_meshed, heights_meshed, dissims_meshed, 100, cmap="turbo")
    #         # plt.show()
    #
    #         # f = interpolate.interp2d(ready_plane.disps.T, ready_plane.heights.T, ready_plane.dissims.T, kind='linear')
    #         # f = interpolate.interp2d(disps_meshed, heights_meshed, dissims_meshed, kind='linear')
    #         # f = interpolate.LinearNDInterpolator(ready_plane.x[:,:2], ready_plane.dissims)
    #
    #         # print(f"ready_plane.disps.T[0] {ready_plane.disps.T[0]}")
    #         # print(f"before interp {list(zip(ready_plane.disps.T[0], ready_plane.heights.T[0]))}")
    #         # print(f"ready_plane.dissims{ready_plane.dissims.T[0]}")
    #         print(f"ready_plane.x[:,:2]  {ready_plane.x[:,:2]}")
    #
    #         f = interpolate.LinearNDInterpolator(
    #             ready_plane.x[:, :2], ready_plane.dissims.T[0]
    #         )
    #
    #         xnew = np.linspace(np.min(real_disp), np.max(real_disp), 100)
    #
    #         ynew = np.linspace(np.min(heights), np.max(heights), 100)
    #
    #         print(f"y new {ynew} length {len(ynew)}")
    #
    #         znew = f(xnew, ynew)
    #         print(f"after using f {znew} length {len(znew)}")
    #
    #         plt.plot(ynew, znew, "b-", ready_plane.heights, ready_plane.dissims, "ro-")
    #         plt.show()
    #
    #         plt.plot(xnew, znew, "b-", ready_plane.disps, ready_plane.dissims, "ro-")
    #         plt.show()
    #
    #     # save to arrays etc
    #     heights_at_mins.append(height_at_min)
    #     disps_at_mins.append(disp_at_min)
    #
    # plot_minimas_graph(heights_at_mins, disps_at_mins, angles, meta, alt_ref=alt_ref)


if __name__ == "__main__":

    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
    )
    # current_experiment = "collect_dataset_3d_21y-03m-03d_15h18m06s/"
    # current_experiment = "collect_dataset_3d_21y-11m-19d_12h24m42s/"
    # current_experiment = "collect_dataset_3d_21y-11m-22d_16h10m54s/"
    #
    # current_experiment = "collect_dataset_3d_21y-12m-07d_16h00m01s/"
    # current_experiment = "collect_dataset_3d_21y-12m-07d_15h24m32s/"
    current_experiment = "collect_dataset_3d_21y-12m-07d_12h33m47s/"

    state = State(meta=common.load_data(data_home + current_experiment + "meta.json"))

    print(f"Dataset: {current_experiment} using {state.meta['stimuli_name']}")

    # reverse real displacements so when main is run twice, the change is not reverted
    real_disp = state.meta["line_range"]  # nb, not copied so that reverse is persistent
    real_disp.reverse()  # to match previous works (-ve on obj, +ve free)

    state.ex = Experiment()

    main(state.ex, state.meta, data_home=data_home, current_experiment=current_experiment)

import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Wedge
from mpl_toolkits.mplot3d import Axes3D

import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.gplvm as gplvm
import tactip_toolkit_dobot.experiments.min_example.common as common
from tactip_toolkit_dobot.experiments.online_learning.contour_following_2d import (
#     Experiment,
#     make_meta,
#     plot_gplvm,
#     State,
    parse_exp_name,
)
# import tactip_toolkit_dobot.experiments.online_learning.post_processing.plot_all_movements as plot_all_movements
# from tactip_toolkit_dobot.experiments.online_learning.contour_following_3d import (
#     Experiment,
#     make_meta,
#     # plot_all_movements,
#     # plot_all_movements_3d,
#     State
# )

import tactip_toolkit_dobot.experiments.online_learning.contour_following_3d as online
import tactip_toolkit_dobot.experiments.online_learning.plots_test as plots_test


if __name__ == "__main__":

    both_averages = []
    total_taps_in_gplvm =[]
    taps_per_plane= []
    num_of_planes = []

    save_as_sub_name = ""

    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2023/"
    )

    stimuli_to_graph = "waves3d"
    # stimuli_to_graph = "tilt"
    # method_type = "full_grid"
    # method_type = "cross"
    method_type = "both"


    base_x_location= -16
    averages_array =[]
    if stimuli_to_graph == "waves3d":
        plot_size = [75,20,90]
    elif stimuli_to_graph == "tilt":
        plot_size = [7,27,45]

    fig, ax = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [plot_size[0], plot_size[1]]}
        )

    for subdir, dirs, files in os.walk(data_home):
        # print(subdir)
        if subdir.split('/')[-1] != "post_processing" and subdir.split('/')[-1] != "":
            # print(subdir.split('/')[-1])
            current_experiment = subdir.split('/')[-1] + "/"
            print(current_experiment)

            # try:

            state = online.State(meta=common.load_data(data_home + current_experiment + "meta.json"))

            print(state.meta["stimuli_name"])

            state.ex = online.Experiment()

            ex = state.ex
            meta = state.meta
            show_figs=False

            ex.all_tap_positions = common.load_data(data_home + current_experiment + "all_positions_final.json")
            ex.all_tap_positions = np.array(ex.all_tap_positions) + np.array([meta['work_frame_offset'][0] - base_x_location, 0,0,0])
            # print(ex.all_tap_positions)

            ex.line_locations = common.load_data(data_home + current_experiment + "location_line_001.json")
            ex.line_locations = np.array([ex.line_locations]) + np.array([meta['work_frame_offset'][0] - base_x_location,0])

            # print("line locations:")
            # print(f"lines: {ex.line_locations}")
            # print(f"type : {type(ex.line_locations)}")
            # print(f"shape: {np.shape(ex.line_locations)}")

            ex.edge_locations = common.load_data(data_home + current_experiment + "all_edge_locs_final.json")
            if meta["stimuli_name"].split('-')[0] == "wavy" and meta["stimuli_name"].split('-')[-1] == "3d":
                ex.edge_locations = np.array(ex.edge_locations) + np.array([meta['work_frame_offset'][0] - base_x_location, 0])
                if meta["stimuli_name"] == "wavy-raised-3d":
                    ex.edge_height = common.load_data(data_home + current_experiment + "all_edge_heights_final.json")
                    ex.edge_height = np.array(ex.edge_height) +2
                else:
                    ex.edge_height = common.load_data(data_home + current_experiment + "all_edge_heights_final.json")
                    ex.edge_height = np.array(ex.edge_height)
            else:
                ex.edge_locations = np.array(ex.edge_locations)# + np.array([meta['work_frame_offset'][0] - base_x_location, 0])
                ex.edge_height = common.load_data(data_home + current_experiment + "all_edge_heights_final.json")
                ex.edge_height = np.array(ex.edge_height)



            if method_type == "both" or meta["plane_method"] == method_type:
                if stimuli_to_graph == "waves3d":
                    stimuli = meta["stimuli_name"]
                    print("hee")
                    print(stimuli.split("-"))

                    if meta["stimuli_name"].split('-')[0] == "wavy" and meta["stimuli_name"].split('-')[-1] == "3d":
                        save_as_sub_name = "wavy-3d"
                        print("making graphs")
                        plots_test.plot_all_movements_both(ex,meta, show_figs, save_figs=False, fig_ax=(fig,ax), plot_size=plot_size)

                        # online.plot_all_movements_3d(ex,meta, show_figs, save_figs=False)


                elif stimuli_to_graph == "tilt":
                    if meta["stimuli_name"].split('-')[0] == "tilt":
                        save_as_sub_name = "tilt"
                        # if save_as_name == "all_movements_final":
                        # if meta["plane_method"] == "cross":
                        plots_test.plot_all_movements_both(ex,meta, show_figs, save_figs=False, fig_ax=(fig,ax),plot_size=plot_size)

                        # gplvm = common.load_data(data_home + current_experiment + "gplvm_final.json")
                        # print(f"len gplvm x {len(gplvm['x'])}")
                        # total_taps_in_gplvm.append((averages_array[-1][0],len(gplvm['x']) ))
                        #
                        # if meta["plane_method"] == "cross":
                        #     num_taps = len(meta["line_range"]) + len(meta["height_range"])
                        #     taps_per_plane.append((averages_array[-1][0], num_taps))
                        #
                        #     num_planes = total_taps_in_gplvm[-1][1] / taps_per_plane[-1][1]
                        #     num_of_planes.append((averages_array[-1][0], num_planes))
                        #
                        # elif meta["plane_method"] == "full_grid":
                        #     num_taps = len(meta["line_range"]) * len(meta["height_range"])
                        #     taps_per_plane.append((averages_array[-1][0], num_taps))
                        #
                        #     num_planes = total_taps_in_gplvm[-1][1] / taps_per_plane[-1][1]
                        #     num_of_planes.append((averages_array[-1][0], num_planes))

                        # else:
                        #     if  meta["plane_method"] == "cross":
                        #         plots_test.plot_all_movements_3d(ex,meta, show_figs, save_figs=False)

                # except:
                #     print("Plot all failed, moving on")

    if False:
        print(f"averages array: {averages_array}")
        print("sorted:")
        print(sorted(averages_array, key=lambda x: x[0]))

        xmin, xmax, ymin, ymax = plt.axis()
        # print(xmax)
        # plt.axis([xmin+2, xmax + 1, ymin +2, ymax -1])
        plt.axis([xmin+2, xmax-2 ,  ymax+1, ymin-1 ])
        both_averages.append(sorted(averages_array, key=lambda x: x[0]))

    # plt.legend(loc="upper left",fontsize=5)

    # ax=plt.gca()
    # # handles, labels = ax.get_legend_handles_labels()
    # # # sort both labels and handles by labels
    # # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # # ax.legend(handles, labels)
    #
    # handles,labels = ax.get_legend_handles_labels()
    # sorted_legends= [x for _,x in sorted(zip(labels,labels),reverse=True)]
    # #sort the labels based on the list k
    # #reverse=True sorts it in descending order
    # sorted_handles=[x for _,x in sorted(zip(labels,handles),reverse=True)]
    # #to sort the colored handles
    # ax.legend(sorted_handles,sorted_legends,loc="upper left",fontsize=5)
    #display the legend on the side of your plot.
    print(f"here4 {plt.rcParams['figure.figsize']}")

    full_path_png = os.path.join(data_home, save_as_sub_name + "all_movements_final_both_"+method_type+".png")
    full_path_svg = os.path.join(data_home, save_as_sub_name + "all_movements_final_both_"+method_type+".svg")
    plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
    plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    plt.clf()
    plt.close()


    print(f"both averages: {both_averages}")
    print(f"total_taps in gplvm {sorted(total_taps_in_gplvm, key=lambda x: x[0])}")
    print(f"taps per plane {sorted(taps_per_plane, key=lambda x: x[0])}")
    print(f"no of planes {sorted(num_of_planes, key=lambda x: x[0])}")

    # both_averages = []
    # total_taps_in_gplvm =[]
    # taps_per_plane= []
    # num_of_planes = []


    # list_3d = []
    # for i, item in enumerate(both_averages[0]):
    #
    #     average_3d = np.mean([both_averages[0][i][1], both_averages[1][i][1]])
    #     list_3d.append((both_averages[0][i][0],average_3d))
    # print(list_3d)



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


def plot_all_movements(ex, meta, show_figs=True, save_figs=True):
    line_width = 1
    marker_size = 1
    ax = plt.gca()
    # if meta["stimuli_name"] == "70mm-circle":
    #     # print small circle location
    #     radius = 35
    #     x_offset = 35 - 35
    #     y_offset = 0 + 35
    #     # --- https://uk.mathworks.com/matlabcentral/answers/3058-plotting-circles
    #     ang = np.linspace(np.pi / 2, -np.pi / 2, 100)
    #     x = x_offset + radius * -np.cos(ang)
    #     y = y_offset + radius * np.sin(ang)
    #     plt.plot(x, y, "tab:brown", linewidth=line_width)
    #     # y=y*.8
    #     # plt.plot(x, y,'tab:brown',linewidth=line_width, linestyle='dashed')
    #
    #     # Arc(xy, width, height, angle=0.0, theta1=0.0, theta2=360.0, **kwargs
    #     w2 = Wedge((x_offset, y_offset), radius, 90, -90, fc="tab:brown", alpha=0.5)
    #     ax.add_artist(w2)
    # elif meta["stimuli_name"] == "flower":
    #     img = plt.imread("/home/lizzie/Pictures/stimulus-flower.png")
    #     img_cropped = img[:, 0 : int(img.shape[0] / 2), :]
    #     f_size = 126
    #     f_y_offset = -5.2
    #     ax.imshow(
    #         img_cropped,
    #         extent=[-f_size / 2, 0, 0 + f_y_offset, f_size + f_y_offset],
    #         alpha=0.5,
    #     )

    if meta["stimuli_name"].split("-")[0] == "tilt":
        # # plt.plot([0, 0, 100],[0, 80, 80])
        # # plt.plot([0, 60],[0, 0], 'k:')
        #
        # # x1 = (31.97--34.82)/0.765
        # # x1_smaller = x1* (60/x1)
        # # y1 = (170.84-168.58) * (60/x1)
        # #
        # # plt.plot([0, x1_smaller],[0, y1], 'k:')
        #
        # x1 = (56.42--12.05)/0.765
        # x1_smaller = x1* (60/x1)
        # y1 = (171.67-167.95) * (60/x1)
        #
        #
        # plt.plot([0, x1_smaller],[0, y1], 'k:')
        x1_smaller = 39
        y1 =1
        plt.plot([0, x1_smaller],[0, y1], 'k:')

    # # print all tap locations
    # all_tap_positions_np = np.array(ex.all_tap_positions)
    # pos_xs = all_tap_positions_np[2:, 0]
    # pos_ys = all_tap_positions_np[2:, 1]
    # # pos_ys = pos_ys/0.8
    # n = range(len(pos_xs))
    # plt.plot(
    #     pos_xs, pos_ys, "k", marker="o", markersize=marker_size, linewidth=line_width
    # )
    # plt.scatter(pos_xs, pos_ys, color="k", s=marker_size)

    # [
    #     ax.annotate(
    #         int(x[0]), (x[1], x[2]), fontsize=1, ha="center", va="center", color="grey"
    #     )
    #     for x in np.array([n, pos_xs, pos_ys]).T
    # ]

    # # print data collection lines
    # for line in ex.line_locations:
    #     line_locations_np = np.array(line)
    #     plt.plot(
    #         line_locations_np[:, 0],
    #         line_locations_np[:, 1],
    #         "r",
    #         marker="o",
    #         markersize=marker_size,
    #         linewidth=line_width,
    #     )
        # plt.scatter(line_locations_np[:, 0], line_locations_np[:, 1], color="g",s=marker_size)
    tilt_angle = int(meta["stimuli_name"].split("-")[1].split("deg")[0])
    try:
        if meta["stimuli_name"].split("-")[2] == "down":
            tilt_angle = -tilt_angle

    except:
        print("moving on")

    if ex.edge_locations is not None:
        # print predicted edge locations
        all_edge_np = np.array(ex.edge_locations)
        pos_xs = all_edge_np[:, 1]
        pos_ys = all_edge_np[:, 0]
        pos_ys = pos_ys - pos_ys[0]
        # pos_ys = pos_ys/0.8
        n = range(len(pos_xs))

        color_line = (tilt_angle + 20) / 65
        plt.plot(
            pos_xs,
            pos_ys,
            # color="#15b01a",
            color=[1-color_line, 0, color_line],
            marker="",
            markersize=marker_size + 1,
            linewidth=line_width,
        )
    # plt.scatter(pos_xs, pos_ys, color="r",marker='+',s=marker_size)
    plt.gca().set_aspect("equal")

    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which="major", color="#666666", linestyle="-", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    # plt.yticks([-4,-2,0.0,2,4])

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    # set axis font size
    plt.tick_params(labelsize=5)

    # axis labels
    plt.xlabel("y displacement (mm)", fontsize=5, va="top")
    plt.ylabel("x displacement (mm)", fontsize=5, va="top",labelpad=10 )

    # add identifier labels
    part_path, _ = os.path.split(meta["meta_file"])

    exp_name = part_path.split("/")
    readable_name = parse_exp_name(exp_name[1])

    # plt.gcf().text(
    #     0.01, 1.01, meta["stimuli_name"], transform=ax.transAxes, fontsize=4, alpha=0.2
    # )
    # plt.gcf().text(
    #     1,
    #     1.01,
    #     readable_name,
    #     transform=ax.transAxes,
    #     fontsize=4,
    #     alpha=0.2,
    #     ha="right",
    # )
    #     # Don't allow the axis to be on top of your data
    ax.set_axisbelow(True)

    # ax.set(auto=True)
    # xmin, xmax, ymin, ymax = plt.axis()
    # print(xmax)
    # plt.axis([xmin, xmax + 2, ymin, ymax])

    # ax.annotate(
    #     str(tilt_angle)+"$\degree$", (pos_xs[-1]+1.75,pos_ys[-1]), fontsize=5, ha="center", va="center", color="black"
    # )

    #
    # # Turn on the minor TICKS, which are required for the minor GRID
    # ax.minorticks_on()
    #
    # # Customize the major grid
    # ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # # Customize the minor grid
    # ax.grid(which='minor', linestyle='-', linewidth='0.5', color='grey')
    #
    # # Turn off the display of all ticks.
    # ax.tick_params(which='both', # Options for both major and minor ticks
    #                 top='off', # turn off top ticks
    #                 left='off', # turn off left ticks
    #                 right='off',  # turn off right ticks
    #                 bottom='off') # turn off bottom ticks

    plt.axis([ -2, 43, 5,-5])

    if save_figs:
        # save graphs automatically
        part_path, _ = os.path.split(meta["meta_file"])
        full_path_png = os.path.join(meta["home_dir"], part_path, "all_movements_final.png")
        full_path_svg = os.path.join(meta["home_dir"], part_path, "all_movements_final.svg")
        plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
        plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    if show_figs:
        plt.show()

    if save_figs:
        plt.clf()

    p1 = np.array([0,0])
    p2 = np.array([x1_smaller,y1])
    p3 = np.concatenate(([pos_xs],[pos_ys])).T
    # print(p3)

    # d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
    d = np.abs(np.cross(p2-p1, p3-p1)) / np.linalg.norm(p2-p1)
    print(f"##### distance = {d} #####")
    averages = np.mean(np.abs(d))
    print(f"#### mean = {np.around(averages,1)} ####")
    return (tilt_angle, np.around(averages,1))

def plot_all_movements_3d(ex, meta, show_figs=True, save_figs=True):
    # print(ex.all_tap_positions)

    line_width = 1
    marker_size = 1
    ax = plt.gca()

    tilt_angle = int(meta["stimuli_name"].split("-")[1].split("deg")[0])
    try:
        if meta["stimuli_name"].split("-")[2] == "down":
            tilt_angle = -tilt_angle

    except:
        print("moving on")

    if tilt_angle <26:
        x_distance = 40
    elif tilt_angle <41:
        x_distance = 40
    else:
        x_distance = 40


    y_distance = np.tan(np.deg2rad(tilt_angle)) * x_distance

    plt.plot([0,x_distance],[0, y_distance], ':', linewidth=line_width, color=[0, 0, 0] )

    # if meta["stimuli_name"] == "tilt-05deg-down":
    #     plt.plot([0,100],[0, -8.7])
    # elif meta["stimuli_name"] == "tilt-10deg-down":
    #     plt.plot([0,100],[0, -17.6])
    # elif meta["stimuli_name"] == "tilt-20deg-down":
    #     plt.plot([0,100],[0, -36.4])

    # print all tap locations
    all_tap_positions_np = np.array(ex.all_tap_positions)
    pos_xs = all_tap_positions_np[2:, 0] # remove ref and neutral taps
    pos_ys = all_tap_positions_np[2:, 1]
    heights = all_tap_positions_np[2:, 3]
    heights = heights - heights[0]
    # pos_ys = pos_ys/0.8
    n = range(len(pos_xs))
    # plt.plot(
    #      pos_ys, heights, "k", marker="o", markersize=marker_size, linewidth=line_width
    # )
    # plt.scatter(pos_xs, pos_ys, color="k", s=marker_size)

    # [
    #     ax.annotate(
    #         int(x[0]), (x[1], x[2]), fontsize=1, ha="center", va="center", color="grey"
    #     )
    #     for x in np.array([n,  pos_ys, heights]).T
    # ]
    p1 = np.array([0,0])
    p2 = np.array([x_distance,y_distance])
    p3 = np.concatenate(([pos_ys],[heights])).T
    # print(p3)

    # d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
    d = np.abs(np.cross(p2-p1, p3-p1)) / np.linalg.norm(p2-p1)
    print(f"##### distance = {d} #####")
    averages = np.mean(np.abs(d))
    print(f"#### 3d mean = {np.around(averages,1)} ####")

    ax.annotate(
        str(tilt_angle)+"$\degree$", (x_distance+1.5, y_distance), fontsize=10, ha="center", va="center", color="black"
    )


    # # print data collection lines
    # for line in ex.line_locations:
    #     line_locations_np = np.array(line)
    #     plt.plot(
    #         line_locations_np[:, 0],
    #         line_locations_np[:, 1],
    #         "r",
    #         marker="o",
    #         markersize=marker_size,
    #         linewidth=line_width,
    #     )
    #     # plt.scatter(line_locations_np[:, 0], line_locations_np[:, 1], color="g",s=marker_size)

    if ex.edge_locations is not None:
        # print predicted edge locations
        all_edge_np = np.array(ex.edge_locations)
        pos_xs = all_edge_np[:, 0]
        pos_ys = all_edge_np[:, 1]
        heights = ex.edge_height
        heights = heights - heights[0]
        # pos_ys = pos_ys/0.8
        n = range(len(pos_xs))

        color_line = (tilt_angle + 20) / 65
        # if meta["plane_method"] == "cross":
        #     line_style = ":"
        # elif meta["plane_method"] == "full_grid":
        #     line_style = "-"

        plt.plot(
            pos_ys,
            heights,
            # color="#15b01a",
            color=[1-color_line, 0, color_line],
            marker="",
            markersize=marker_size + .5,
            linewidth=line_width,
            # linestyle=line_style,
            # label=f"{tilt_angle:>3}"
            label=tilt_angle
        )
    # plt.scatter(pos_xs, pos_ys, color="r",marker='+',s=marker_size)
    plt.gca().set_aspect("equal")

    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which="major", color="#666666", linestyle="-", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    # set axis font size
    plt.tick_params(labelsize=5)

    # axis labels
    plt.xlabel("y displacement (mm)", fontsize=5, va="top")
    plt.ylabel("z displacement (mm)", fontsize=5, va="top")

    # add identifier labels
    part_path, _ = os.path.split(meta["meta_file"])

    exp_name = part_path.split("/")
    readable_name = parse_exp_name(exp_name[1])

    # plt.gcf().text(
    #     0.01, 1.01, meta["stimuli_name"], transform=ax.transAxes, fontsize=4, alpha=0.2
    # )
    # plt.gcf().text(
    #     1,
    #     1.01,
    #     readable_name,
    #     transform=ax.transAxes,
    #     fontsize=4,
    #     alpha=0.2,
    #     ha="right",
    # )
    #     # Don't allow the axis to be on top of your data
    ax.set_axisbelow(True)

    # ax.set(auto=True)
    # xmin, xmax, ymin, ymax = plt.axis()
    # # print(xmax)
    # plt.axis([xmin, xmax + 2, ymin, ymax])

    #
    # # Turn on the minor TICKS, which are required for the minor GRID
    # ax.minorticks_on()
    #
    # # Customize the major grid
    # ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # # Customize the minor grid
    # ax.grid(which='minor', linestyle='-', linewidth='0.5', color='grey')
    #
    # # Turn off the display of all ticks.
    # ax.tick_params(which='both', # Options for both major and minor ticks
    #                 top='off', # turn off top ticks
    #                 left='off', # turn off left ticks
    #                 right='off',  # turn off right ticks
    #                 bottom='off') # turn off bottom ticks

    plt.axis([ -2, 43, -10,20-3])

    if save_figs:
        # save graphs automatically
        part_path, _ = os.path.split(meta["meta_file"])
        full_path_png = os.path.join(meta["home_dir"], part_path, "all_movements_3d_final.png")
        full_path_svg = os.path.join(meta["home_dir"], part_path, "all_movements_3d_final.svg")
        plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
        plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    if show_figs:
        plt.show()

    if save_figs:
        plt.clf()

    return (tilt_angle, np.around(averages,1))



if __name__ == "__main__":

    both_averages = []
    total_taps_in_gplvm =[]
    taps_per_plane= []
    num_of_planes = []

    for save_as_name in ["all_movements_final", "all_movements_final_3d"]:
        save_as_sub_name = ""

        data_home = (
            "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2023/"
        )

        base_x_location= -16
        averages_array =[]

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
                else:
                    ex.edge_locations = np.array(ex.edge_locations)# + np.array([meta['work_frame_offset'][0] - base_x_location, 0])



                ex.edge_height = common.load_data(data_home + current_experiment + "all_edge_heights_final.json")
                ex.edge_height = np.array(ex.edge_height)

                # stimuli = meta["stimuli_name"]
                # print("hee")
                # print(stimuli.split("-"))

                # if meta["stimuli_name"].split('-')[0] == "wavy" and meta["stimuli_name"].split('-')[-1] == "3d":
                #     save_as_sub_name = "wavy-3d"
                #     if save_as_name == "all_movements_final":
                #         averages_array.append(online.plot_all_movements(ex,meta, show_figs, save_figs=False))
                #     else:
                #         averages_array.append(online.plot_all_movements_3d(ex,meta, show_figs, save_figs=False))
                if meta["stimuli_name"].split('-')[0] == "tilt":
                    save_as_sub_name = "tilt"
                    if save_as_name == "all_movements_final":
                        if meta["plane_method"] == "cross":
                            averages_array.append(plot_all_movements(ex,meta, show_figs, save_figs=False))

                            gplvm = common.load_data(data_home + current_experiment + "gplvm_final.json")
                            print(f"len gplvm x {len(gplvm['x'])}")
                            total_taps_in_gplvm.append((averages_array[-1][0],len(gplvm['x']) ))

                            if meta["plane_method"] == "cross":
                                num_taps = len(meta["line_range"]) + len(meta["height_range"])
                                taps_per_plane.append((averages_array[-1][0], num_taps))

                                num_planes = total_taps_in_gplvm[-1][1] / taps_per_plane[-1][1]
                                num_of_planes.append((averages_array[-1][0], num_planes))

                            elif meta["plane_method"] == "full_grid":
                                num_taps = len(meta["line_range"]) * len(meta["height_range"])
                                taps_per_plane.append((averages_array[-1][0], num_taps))

                                num_planes = total_taps_in_gplvm[-1][1] / taps_per_plane[-1][1]
                                num_of_planes.append((averages_array[-1][0], num_planes))

                    else:
                        if  meta["plane_method"] == "cross":
                            averages_array.append(plot_all_movements_3d(ex,meta, show_figs, save_figs=False))

                # except:
                #     print("Plot all failed, moving on")

        if True:
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

        full_path_png = os.path.join(data_home, save_as_sub_name + save_as_name + ".png")
        full_path_svg = os.path.join(data_home, save_as_sub_name + save_as_name + ".svg")
        plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
        plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

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



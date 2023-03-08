import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Wedge
from mpl_toolkits.mplot3d import Axes3D

import tactip_toolkit_dobot.experiments.min_example.common as common
# from tactip_toolkit_dobot.experiments.online_learning.contour_following_3d import (
#     Experiment,
#     make_meta,
#     State,
#     parse_exp_name
# )
import tactip_toolkit_dobot.experiments.online_learning.contour_following_3d as online
import tactip_toolkit_dobot.experiments.online_learning.plots_test as plots_test

def plot_all_movements(ex, meta, show_figs=True, save_figs=True):
    line_width = 0.5
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
#200, -35, -175
    x = np.array([
        205.88,
        205.32,
        204.88,
        204.99,
        204.69,
        204.06,
        204.10,
        203.77,
        203.58,
        203.60
    ]) - 205.88#200
    y = (np.array([
        -38.71,
        -27.73,
        -19.82,
        -9.27,
        -3.86,
        -1.31,
        8.95,
        16.46,
        23.96,
        31.15
    ]) - -35)/0.765

    z = np.array([
        -173.21,
        -172.91,
        -170.64,
        -163.08,
        -161.54,
        -159.90,
        -158.57,
        -159.68,
        -161.67,
        -165.42
    ]) --173.21 #-175
    plt.plot(y,x, 'k:')

    if meta["stimuli_name"].split("-")[0] == "tilt":
        # plt.plot([0, 0, 100],[0, 80, 80])
        # plt.plot([0, 60],[0, 0], 'k:')

        # x1 = (31.97--34.82)/0.765
        # x1_smaller = x1* (60/x1)
        # y1 = (170.84-168.58) * (60/x1)
        #
        # plt.plot([0, x1_smaller],[0, y1], 'k:')

        x1 = (56.42--12.05)/0.765
        x1_smaller = x1* (60/x1)
        y1 = (171.67-167.95) * (60/x1)


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

        # color_line = (tilt_angle + 20) / 65
        plt.plot(
            pos_xs,
            pos_ys,
            color="#15b01a",
            # color=[1-color_line, 0, color_line],
            marker="+",
            markersize=marker_size + 1,
            linewidth=line_width,
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
    plt.ylabel("x displacement (mm)", fontsize=5, va="top")

    # add identifier labels
    part_path, _ = os.path.split(meta["meta_file"])

    exp_name = part_path.split("/")
    readable_name = online.parse_exp_name(exp_name[1])

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
    xmin, xmax, ymin, ymax = plt.axis()
    # print(xmax)
    plt.axis([xmin, xmax, ymax+2, ymin-2])

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

    if save_figs:
        # save graphs automatically
        # part_path, _ = os.path.split(meta["meta_file"])
        full_path_png = os.path.join("/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2022/", "all_movements_final_slide.png")
        full_path_svg = os.path.join("/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2022/", "all_movements_final_slide.svg")
        plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
        plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    if show_figs:
        plt.show()

    if save_figs:
        plt.clf()

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    a =[]
    distances = []
    for i in range(0, len(x)-1):
        p1 = np.array([y[i],x[i]])
        p2 = np.array([y[i+1],x[i+1]])
        p3 = np.concatenate(([pos_xs],[pos_ys])).T
    # print(p3)

    # d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
        d = np.abs(np.cross(p2-p1, p3-p1)) / np.linalg.norm(p2-p1)
        distances.append(d)

    distances = np.array(distances)
    print(f"##### distance = {distances} #####")
    min_distnaces = np.min(distances, axis=0)
    print(min_distnaces)

    averages = np.mean(np.abs(min_distnaces))
    print(f"#### mean = {np.around(averages,1)} ####")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


def plot_all_movements_3d(ex, meta, show_figs=True, save_figs=True):
    # print(ex.all_tap_positions)

    line_width = 1
    marker_size = 1
    ax = plt.gca()

    if meta["stimuli_name"].split("-")[0] == "tilt":
        tilt_angle = int(meta["stimuli_name"].split("-")[1].split("deg")[0])
        try:
            if meta["stimuli_name"].split("-")[2] == "down":
                tilt_angle = -tilt_angle

        except:
            print("moving on")

        if tilt_angle <26:
            x_distance = 60
        elif tilt_angle <41:
            x_distance = 50
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
#200, -35, -175
    x = np.array([
        205.88,
        205.32,
        204.88,
        204.99,
        204.69,
        204.06,
        204.10,
        203.77,
        203.58,
        203.60
    ]) - 200
    y = (np.array([
        -38.71,
        -27.73,
        -19.82,
        -9.27,
        -3.86,
        -1.31,
        8.95,
        16.46,
        23.96,
        31.15
    ]) - -35)/0.765

    z = np.array([
        -173.21,
        -172.91,
        -170.64,
        -163.08,
        -161.54,
        -159.90,
        -158.57,
        -159.68,
        -161.67,
        -165.42
    ]) --173.21 #-175

    plt.plot(y, z, ':', linewidth=line_width, color=[0, 0, 0] )

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


    # ax.annotate(
    #     str(tilt_angle)+"$\degree$", (x_distance+1.75, y_distance), fontsize=5, ha="center", va="center", color="black"
    # )


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

        # color_line = (tilt_angle + 20) / 65

        plt.plot(
            pos_ys,
            heights,
            color="#15b01a",
            # color=[1-color_line, 0, color_line],
            marker="+",
            markersize=marker_size + .5,
            linewidth=line_width,
            # label=f"{tilt_angle:>3}"
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
    readable_name = online.parse_exp_name(exp_name[1])

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

    if save_figs:
        # save graphs automatically
        # part_path, _ = os.path.split(meta["meta_file"])
        full_path_png = os.path.join("/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2022/", "all_movements_3d_final_slide.png")
        full_path_svg = os.path.join("/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2022/", "all_movements_3d_final_slide.svg")
        plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
        plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    if show_figs:
        plt.show()

    if save_figs:
        plt.clf()

    a =[]
    distances = []
    for i in range(0, len(x)-1):
        p1 = np.array([y[i],z[i]])
        p2 = np.array([y[i+1],z[i+1]])
        p3 = np.concatenate(([pos_ys],[heights])).T
    # print(p3)

    # d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
        d = np.abs(np.cross(p2-p1, p3-p1)) / np.linalg.norm(p2-p1)
        distances.append(d)

    distances = np.array(distances)
    print(f"##### distance = {distances} #####")
    min_distnaces = np.min(distances, axis=0)
    print(min_distnaces)

    averages = np.mean(np.abs(min_distnaces))
    print(f"#### mean = {np.around(averages,1)} ####")
    # a.append(averages)
    # print(f"averges = {a}")


# np.set_printoptions(precision=2)#, suppress=True)


def main(ex,meta, data_home, current_experiment, show_figs=False):

    ex.all_tap_positions = common.load_data(data_home + current_experiment + "all_positions_final.json")
    ex.all_tap_positions = np.array(ex.all_tap_positions)

    ex.line_locations = common.load_data(data_home + current_experiment + "location_line_001.json")
    ex.line_locations = np.array([ex.line_locations])

    print(ex.line_locations)
    print(type(ex.line_locations))
    print(np.shape(ex.line_locations))

    if False:
        ex.edge_locations = common.load_data(data_home + current_experiment + "all_edge_locs_final.json")
        ex.edge_locations = np.array(ex.edge_locations)

        ex.edge_height = common.load_data(data_home + current_experiment + "all_edge_heights_final.json")
        ex.edge_height = np.array(ex.edge_height)

        plots_test.plot_all_movements_both(ex,meta, show_figs)
    else:
        all_positions_final = common.load_data(data_home + current_experiment + "all_positions_final.json")
        all_positions_final = np.array(all_positions_final[2:])

        print(all_positions_final)
        # x_positions = all_positions_final[:,0]
        # y_positions = all_positions_final[:,1]
        # z_positions = all_positions_final[:,3]
        # angles_positions = all_positions_final[:,2]

        ex.edge_locations = all_positions_final[:,0:2]
        ex.edge_height = all_positions_final[:,3]

        meta["plane_method"] = "full_grid"

        # height_range = np.array(meta["height_range"])
        # line_range =np.array( meta["line_range"])
        # angle_range =np.array( meta["angle_range"])
        #
        # # grid_thing = np.meshgrid(line_range,angle_range, height_range)
        #
        # grid_thing = []
        # for i in height_range:
        #     for j in angle_range:
        #         for k in line_range:
        #             grid_thing.append([i,j, i])
        #
        # print(grid_thing)
        # asdf

        # ex.edge_locations = common.load_data(data_home + current_experiment + "all_edge_locs_final.json")
        # ex.edge_locations = np.array(ex.edge_locations)
        #
        # ex.edge_height = common.load_data(data_home + current_experiment + "all_edge_heights_final.json")
        # ex.edge_height = np.array(ex.edge_height)
        plots_test.plot_all_movements_both(ex,meta, show_figs)


    # plots_test.plot_all_movements_both(ex,meta, show_figs)
    # plots_test.plot_all_movements_basic_test(ex,meta, show_figs)
    # online.plot_all_movements(ex,meta, show_figs)
    # online.plot_all_movements_3d(ex,meta, show_figs)


if __name__ == "__main__":

    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
        # "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2022/"
        # "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2023/"
    )
    # current_experiment = "contour_following_2d_01m-19d_10h47m37s/"
    # current_experiment = "contour_following_2d_01m-18d_17h41m48s/"
    # current_experiment = "contour_following_2d_01m-22d_14h58m05s/"
    # current_experiment = "contour_following_2d_2021y-01m-25d_17h37m24s/"
    # current_experiment = "contour_following_2d_2021y-01m-25d_18h08m31s/"
    # current_experiment = "contour_following_2d_2021y-01m-26d_15h13m00s/"
    # current_experiment = "contour_following_3d_2021y-08m-11d_11h54m37s/"
    # current_experiment =  "contour_following_3d_2021y-08m-13d_15h51m47s/"
    # current_experiment =  "contour_following_3d_2021y-08m-16d_15h44m12s/" #20 deg down
    # current_experiment =  "contour_following_3d_2021y-08m-13d_15h51m47s/" #10 deg down
    # current_experiment = "contour_following_3d_2021y-09m-02d_18h40m35s/" #slide

    # current_experiment = "contour_following_3d_2022y-07m-29d_11h50m44s/"
    # current_experiment = "contour_following_3d_2022y-07m-29d_14h29m56s/"

    # current_experiment = "contour_following_3d_2022y-08m-02d_14h03m55s/"
    # current_experiment = "contour_following_3d_2022y-08m-03d_16h57m30s/"
    # current_experiment = "contour_following_3d_2022y-09m-05d_16h39m47s/"

    # current_experiment = "contour_following_3d_2022y-09m-08d_10h25m19s/"
    # current_experiment = "contour_following_3d_2022y-09m-08d_12h54m35s/"
    # current_experiment = "contour_following_3d_2022y-09m-08d_12h47m46s/"
    # current_experiment = "contour_following_3d_2022y-09m-08d_12h44m04s/"
    # current_experiment = "contour_following_3d_2022y-09m-05d_15h23m19s/"
    # current_experiment = "contour_following_3d_2022y-09m-09d_17h05m32s/"
    # current_experiment = "contour_following_3d_2022y-09m-09d_15h49m55s/"
    # current_experiment = "contour_following_3d_2022y-09m-10d_11h22m36s/"
    # current_experiment = "contour_following_3d_2022y-09m-10d_12h30m36s/"
    # current_experiment = "contour_following_3d_2022y-09m-10d_16h29m37s/"
    # current_experiment = "contour_following_3d_2022y-08m-08d_15h05m19s/"
    # current_experiment = "contour_following_3d_2022y-08m-05d_14h58m18s/"

    ### white lid thesis ###
    # current_experiment = "contour_following_3d_2022y-09m-13d_14h38m49s/"

    # current_experiment = "contour_following_3d_2022y-08m-04d_15h44m53s/"


    # current_experiment = "contour_following_3d_2022y-09m-05d_14h08m17s/"
    # current_experiment = "contour_following_3d_2022y-08m-16d_14h06m41s/"
    # current_experiment = "contour_following_3d_2022y-08m-16d_14h17m02s/"
    # current_experiment = "contour_following_3d_2022y-08m-08d_15h15m18s/"

    ### Groove 2d thesis ###
    # current_experiment = "contour_following_3d_2023y-02m-01d_15h20m26s/"
    # current_experiment = "contour_following_3d_2023y-02m-01d_15h03m56s/"
    # current_experiment = "contour_following_3d_2023y-02m-01d_15h10m31s/"
    # current_experiment = "contour_following_3d_2023y-02m-01d_15h26m36s/"

    ### Failed saddle thesis ###
    # current_experiment = "contour_following_3d_2022y-08m-04d_15h44m53s/"

    ### Banana 3d thesisi ###
    # current_experiment = "contour_following_3d_2023y-02m-01d_10h56m31s/"
    # current_experiment = "contour_following_3d_2023y-02m-01d_11h05m16s/"

    experiments = {
        "collect_dataset_3d_21y-03m-03d_15h18m06s/"
        #
        # "contour_following_3d_2023y-02m-01d_15h20m26s/",
        # "contour_following_3d_2023y-02m-01d_15h03m56s/",
        # "contour_following_3d_2023y-02m-01d_15h10m31s/",
        # "contour_following_3d_2023y-02m-01d_15h26m36s/",
        # "contour_following_3d_2022y-08m-08d_15h05m19s/",
        # "contour_following_3d_2022y-08m-08d_15h15m18s/",
        # "contour_following_3d_2023y-02m-01d_10h56m31s/",
        # "contour_following_3d_2023y-02m-01d_11h05m16s/",
        # "contour_following_3d_2022y-08m-04d_15h44m53s/",
        # "contour_following_3d_2022y-09m-13d_14h38m49s/",
        # "contour_following_3d_2023y-02m-20d_15h11m38s/",
        # "contour_following_3d_2023y-02m-20d_15h25m58s/"

    }

    for current_experiment in experiments:

        state = online.State(meta=common.load_data(data_home + current_experiment + "meta.json"))

        print(state.meta["stimuli_name"])

        state.ex = online.Experiment()
        main(state.ex,state.meta, data_home, current_experiment, show_figs=False)

# To use this file, data must be pre-processed by running the following files:
# > graph_dataset.py
# > generate_reftaps.py

# NB this file will overwrite graphs in main dataset, so graph_dataset.py needs
# to be rerun afterwards to correct the graphs

import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import interpolate
from PIL import Image

import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp
# import tactip_toolkit_dobot.experiments.online_learning.offline_setup.gplvm as gplvm
import tactip_toolkit_dobot.experiments.min_example.common as common
from tactip_toolkit_dobot.experiments.online_learning.contour_following_2d import (
    Experiment,
    # make_meta,
    # plot_gplvm,
    State,
    # parse_exp_name,
)


import tactip_toolkit_dobot.experiments.online_learning.offline_3d.offline_dissimgrids as offline_dissimgrids

def extract_val(e):
    answer = e.split('/')[-1]
    answer = answer.replace("min_drift_", '')
    answer = answer.replace(".png",'')
    answer = answer[1:]
    print(f"answer {answer}")
    return float(answer)


def main(ex, meta, make_graphs=True, combine_graphs=True):
    if make_graphs:
        #  loop over file to get all ref taps

        for subdir, dirs, files in os.walk(
            data_home + current_experiment + "post_processing/alt_ref_taps/"
        ):
            print(files)

            for file in files:
                if file.split("_")[0] == "ref":
                    print(file)

                    # loc_name = file.replace('ref_tap_','')
                    # loc_name = loc_name.replace('.json','')
                    #
                    # print(f"loc name = {loc_name}")

                    # todo: make it so that can save minimas graphs (original and alt)
                    # alt_ref shoudl be in format= "post_processing/alt_ref_taps/ref_tap_" + j + str(i) + ".json"
                    offline_dissimgrids.main(
                        ex,
                        meta,
                        alt_ref="post_processing/alt_ref_taps/" + file,
                        grid_graphs_on=False,
                        data_home=data_home,
                        current_experiment=current_experiment,
                        show_fig=False
                    )
    if combine_graphs:
        # create summary image for each of height, angle and disp
        list_im_h = []
        list_im_a = []
        list_im_d = []

        for subdir, dirs, files in os.walk(
            data_home + current_experiment + "post_processing/alt_ref_taps/"
        ):

            for file in files:
                if file.split("_")[0] == "min" and file.split(".")[-1] == "png":
                    print(file)

                    h_a_d = file.split('_')[2][0]
                    print(h_a_d)
                    if h_a_d == 'h':
                        list_im_h.append(data_home + current_experiment + "post_processing/alt_ref_taps/"+file)
                    elif h_a_d == 'a':
                        list_im_a.append(data_home + current_experiment + "post_processing/alt_ref_taps/"+file)
                    elif h_a_d == 'd':
                        list_im_d.append(data_home + current_experiment + "post_processing/alt_ref_taps/"+file)
        list_im_h.sort(key=extract_val)
        list_im_a.sort(key=extract_val)
        list_im_d.sort(key=extract_val)

        # print(list_im_h)
        # print(list_im_a)
        # print(list_im_d)

        concatenated = Image.fromarray(
            np.concatenate(
                # [np.array(Image.open(x).resize((49,37)) ) for x in list_im],
                [np.array(Image.open(x)) for x in list_im_h],
                axis=1,
            )
        )
        concatenated.save(data_home + current_experiment + "post_processing/alt_ref_taps/" + "summary_h.png")
        # concatenated.save(data_home + current_experiment + "post_processing/alt_ref_taps/" + "summary_h.jpg")
        # concatenated.save(data_home + current_experiment + "post_processing/alt_ref_taps/" + "summary_h.svg")

        concatenated = Image.fromarray(
            np.concatenate(
                # [np.array(Image.open(x).resize((49,37)) ) for x in list_im],
                [np.array(Image.open(x)) for x in list_im_a],
                axis=1,
            )
        )
        concatenated.save(data_home + current_experiment + "post_processing/alt_ref_taps/" + "summary_a.png")

        concatenated = Image.fromarray(
            np.concatenate(
                # [np.array(Image.open(x).resize((49,37)) ) for x in list_im],
                [np.array(Image.open(x)) for x in list_im_d],
                axis=1,
            )
        )
        concatenated.save(data_home + current_experiment + "post_processing/alt_ref_taps/" + "summary_d.png")


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

    # main(state.ex, state.meta,train_or_test="train", train_folder="model_two_cross/")
    # main(state.ex, state.meta,train_or_test="test_line_angles",train_folder="model_two_cross/")
    # main(state.ex, state.meta,train_or_test="test_single_taps", train_folder="model_two_cross/")

    # main(state.ex, state.meta,train_or_test="train", train_folder="model_one_grid/")
    # main(state.ex, state.meta,train_or_test="test_line_angles",train_folder="model_one_grid/")
    # main(state.ex, state.meta,train_or_test="test_single_taps", train_folder="model_one_grid/")

    # main(state.ex, state.meta,train_or_test="train", train_folder="model_two_grid/")
    # main(state.ex, state.meta,train_or_test="test_line_angles",train_folder="model_two_grid/")
    main(state.ex, state.meta, make_graphs=True, combine_graphs=True)

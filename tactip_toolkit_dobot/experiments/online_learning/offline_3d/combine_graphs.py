import numpy as np
import PIL
from PIL import Image

# import numpy as np
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



def main(ex, meta):

    # load data
    path = data_home + current_experiment

    nums = np.array(meta["angle_range"])
    list_im = [None]*len(nums)

    for  i, thing in enumerate(nums):
        list_im[i] = path +  "grid_0" + str(thing) + ".png"


    concatenated = Image.fromarray(
      np.concatenate(
        # [np.array(Image.open(x).resize((49,37)) ) for x in list_im],
        [np.array(Image.open(x) ) for x in list_im],
        axis=1
      )
    )
    concatenated.save( path + 'Trifecta.png' )

    # # list_im = ['Test1.jpg', 'Test2.jpg', 'Test3.jpg']
    # imgs    = [ PIL.Image.open(i) for i in list_im ]
    # # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    # min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    # imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    #
    # # save that beautiful picture
    # imgs_comb = PIL.Image.fromarray( imgs_comb)
    # imgs_comb.save( path + 'Trifecta.png' )


    #
    # neutral_tap = np.array(common.load_data(path + "neutral_tap.json"))
    # ref_tap = np.array(common.load_data(path + "ref_tap.json"))
    #
    # locations = np.array(common.load_data(path + "post_processing/all_locations.json"))
    # lines = np.array(common.load_data(path + "post_processing/all_lines.json"))
    # dissims = np.array(common.load_data(path + "post_processing/dissims.json"))
    #
    # optm_disps = np.array(
    #     common.load_data(path + "post_processing/corrected_disps_basic.json")
    # )
    #
    # heights = meta["height_range"]
    # num_heights = len(heights)
    # angles = meta["angle_range"]
    # num_angles = len(angles)
    # real_disp = meta["line_range"]
    # num_disps = len(real_disp)
    #
    # # Find location of disp minima
    # training_local_1 = [0, 5]  # [height(in mm), angle(in deg)]
    #
    #
    # for local in np.concatenate((np.zeros((num_angles,1),), np.array([np.arange(angles[0],angles[-1]+1,5,dtype=int)]).T), axis=1).tolist():
    #     # new_taps = extract_line_at(training_local_1, lines, meta).y
    #
    #     # ready_plane = get_calibrated_plane(
    #     #     local, meta, lines, optm_disps, ref_tap, num_disps
    #     # )
    #
    #     ready_plane = at_plane_extract(
    #         local,
    #         lines,
    #         meta,
    #         method="full",
    #         cross_length=2,
    #         cross_disp=0,
    #         dissims=dissims
    #     )
    #
    #     print(f"calibrated plane is: {ready_plane.__dict__}")
    #
    #
    #     ready_plane.make_all_phis(1)
    #     ready_plane.make_x()
    #
    #     plot_dissim_grid(ready_plane, meta, step_num_str="0"+str(int(local[1])), show_fig=False, filled=True)
    #
    #     plt.clf()

if __name__ == "__main__":

    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
    )
    # current_experiment = "collect_dataset_3d_21y-03m-03d_15h18m06s/"
    # current_experiment = "collect_dataset_3d_21y-11m-19d_12h24m42s/"
    current_experiment = "collect_dataset_3d_21y-11m-22d_16h10m54s/"    

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

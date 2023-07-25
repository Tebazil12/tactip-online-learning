import scipy.io
import scipy.spatial
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

# import gplvm
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.gp as gp
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp


def load_data():
    n_angles = 19
    all_data = [None] * n_angles  # all_data[angle][disp][frame][pin][xory]

    # folder_path = "C:\\Users\\ea-stone\\Documents\\data\\singleRadius2019-01-16_1651\\"
    folder_path = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/singleRadius2019-01-16_1651/"
    )

    for i in range(0, n_angles):
        if i < 9:
            file_name = "c0" + str(i + 1) + "_01.mat"
        else:
            file_name = "c" + str(i + 1) + "_01.mat"

        full_load_name = folder_path + file_name
        mat = scipy.io.loadmat(full_load_name)
        all_data[i] = mat["data"][0]
        # all_data[i].round(2)

    return all_data


def best_frame(all_frames):
    """
    For the given tap, select frame with the highest average pin displacement
    from the first frame. Takes in one tap, in form (16ish x 126 ish x 2)
    np.array, returns (1 x 126ish x 2) np.array
    """
    # make into pin displacement instead of absolute position
    all_frames_disp = all_frames - all_frames[0]

    # Find frame where largest pin displacement takes place (on average)
    # #TODO this method is not the same as in MATLAB!
    # Average disps per frame (over all pins)
    mean_disp_per_frame = np.mean(np.abs(all_frames_disp), axis=1)

    # Find euclidean distance per frame
    distances_all_disps = np.linalg.norm(mean_disp_per_frame, axis=1)

    # Find frame with max euclidean distance
    result = np.where(distances_all_disps == np.amax(distances_all_disps))
    max_frame_i = result[0][0]

    tap = all_frames_disp[max_frame_i]

    return tap.reshape(tap.shape[0] * tap.shape[1])


def get_processed_data(all_data, ref_tap, indexes=None):
    """
    Return two lists both the same length as number of training angles. Takes
    all_data as lists of np.arrays)
    """

    y_processed = []  # TODO THIS METHOD OF ARRAY BUILDING IS NOT EFFICIENT
    dissim_processed = []

    if indexes is None:
        i_angles = range(0, len(all_data))
    else:
        i_angles = range(0, len(indexes))

    for i_angle in i_angles:
        if indexes is not None:
            y_train_line = extract_ytrain(all_data[indexes[i_angle]])
        else:
            y_train_line = extract_ytrain(all_data[i_angle])

        dissim_line = calc_dissims(y_train_line, ref_tap)

        y_processed.append(y_train_line)
        dissim_processed.append(dissim_line)

    return [y_processed, dissim_processed]


def extract_ytrain(radii_data):
    """ Extract ytrain given radii_data[disp][frame][pin][xory] """

    # shape for the radius data to be returned (note this is 2d, not 3d)
    data_shape = (
        radii_data.shape[0],
        (radii_data[0][0].shape[0] * radii_data[0][0].shape[1]),
    )
    y_train = np.zeros(shape=data_shape)

    for disp_num in range(0, len(radii_data)):  # for each tap on radius
        tap = best_frame(radii_data[disp_num])
        y_train[disp_num] = tap

    return y_train


def calc_dissims(y_train, ref_tap):
    # print("calc_dissim")

    diffs = -y_train + ref_tap
    # print(diffs.shape)

    # reshape to 21 by 126 by 2?
    diffs_3d = diffs.reshape(diffs.shape[0], int(diffs.shape[1] / 2), 2)
    # print(diffs_3d.shape)

    y_train_2d = y_train.reshape(y_train.shape[0], int(y_train.shape[1] / 2), 2)
    ref_tap_2d = ref_tap.reshape(int(ref_tap.shape[0] / 2), 2)

    sum_diffs = diffs_3d.sum(1)  # sum in x and y
    # print(sum_diffs)
    # print(sum_diffs.shape)

    sum_ys = y_train_2d.sum(1)
    sum_ref = ref_tap_2d.sum(0)
    # print("ys and ref")
    # print(sum_ys)

    # print(ref_tap.shape)
    # print(y_train.shape)
    # print(sum_ref)

    # TODO recreate matlab ordering of array(to see if this is causing the disparity):

    # Calculate Euclidean distance as dissimilarity measure
    # dissim = np.linalg.norm(sum_diffs,axis=1)
    # print("original dissim")
    # print(dissim)

    # trying to recreate matlab - ignore, its the same results as the working python one, but hard to impolement
    # across rows properly in pythoon
    # dissim = np.linalg.norm(diffs_3d[1])
    # print("original dissim")
    # print(dissim)

    # dissim = scipy.spatial.distance.cdist(np.array([[0,0]]), sum_diffs, 'euclidean') #same as above method
    # print("dissim")
    # print(dissim)
    # print(dissim.shape)

    # dissim = scipy.spatial.distance.cdist([sum_ref], sum_ys, 'euclidean') #same as above 2 methods
    # print("dissim sums")
    # print(dissim[0])
    # print(dissim.shape)

    # todo this one works well
    dissim = scipy.spatial.distance.cdist(
        [ref_tap], y_train, "euclidean"
    )  # NOTsame as above 2 methods
    # print(diffs.shape)

    # trying to replicate matlabs worse values
    # dissim = scipy.spatial.distance.cdist(np.empty(diffs.shape), diffs, 'euclidean')
    # print("dissim sums")
    # print(dissim)
    # print(dissim.shape)

    # dissim = scipy.spatial.distance.cdist([ref_tap], y_train, 'cosine')

    # dissim = scipy.spatial.distance.cdist([sum_ref], sum_ys, 'cosine')
    # dissim = np.rad2deg(dissim)
    # print("dissim sums cosine")
    # print(dissim)
    # print(dissim.shape)
    # print(dissim_degs)

    return dissim[0]  # so that not array within array...


def show_dissim_profile(disp, dissim):
    """ dissim needs to be a list, with each entry a line of taps corresponding with disp"""
    # print(dissim)
    for i in range(0, len(dissim)):
        # print(len(disp))
        if len(disp) == 1:
            plt.plot(disp[0], dissim[i], marker="x")
        elif len(disp) == len(dissim):
            plt.plot(disp[i], dissim[i], marker="x")
        else:
            raise NameError("disp not correct length for plotting")
    plt.xticks(np.arange(np.amin(disp[0]), np.amax(disp[0]) + 2, 2))
    ax = plt.gca()
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    plt.ylabel("dissim")
    plt.xlabel("disp")
    # plt.ioff()
    # plt.show(block=False)
    # plt.show()


def align_all_xs_via_dissim(disp, dissim):
    # todo for loop
    corrected_disps = [None] * len(dissim)
    for i in range(0, len(dissim)):
        corrected_disps[i] = align_radius(disp[0], dissim[i])

    return corrected_disps


def align_radius(disp, dissim, gp_extrap=False):
    if gp_extrap:
        sigma_n_diss = 5
        start_params = [15.0, 15.0]  # sigma_f and L respectively
        data = [disp, dissim, sigma_n_diss]
        # minimizer_kwargs = {"args": data}
        result = scipy.optimize.minimize(
            gp.max_log_like, start_params, args=data, method="BFGS"
        )
        # print(result)

        [sigma_f, L] = result.x

        disp_stars, dissim_stars = gp.interpolate(
            disp, dissim, sigma_f, L, sigma_n_diss
        )

        show_dissim_profile([disp_stars], [dissim_stars])

        # return shift
    else:
        result = np.where(dissim == np.amin(dissim))
        # print(result[0][0])
        disp_offset = disp[result[0][0]]
        # print(disp_offset)

    corrected_disp = (
        disp - disp_offset
    )  # TODO data is backwards, this maybe need to be +ve for online stuff

    # print(corrected_disp)

    return corrected_disp

def test_all():
    all_data = load_data()


    ref_tap = dp.best_frame(all_data[10 - 1][11 - 1])
    ref_tap2 = best_frame(all_data[10 - 1][11 - 1])

    assert (ref_tap == ref_tap2).all()

    i_training_angles = [10 - 1, 15 - 1, 19 - 1, 5 - 1, 1 - 1]

    [y_train, dissim_train] = dp.get_processed_data(
        all_data, ref_tap, indexes=i_training_angles
    )

    [y_train2, dissim_train2] = get_processed_data(
        all_data, ref_tap, indexes=i_training_angles
    )

    assert type(y_train2) == type(y_train)
    assert len(y_train2[0]) == len(y_train[0])
    assert len(y_train2[0][0]) == len(y_train[0][0])

    assert(np.array(y_train) == np.array(y_train2)).all()

    assert (np.array(dissim_train) == np.array(dissim_train2)).all()


    disp_real = [-np.arange(-10, 11, dtype=np.float)]
    disp_train = align_all_xs_via_dissim(disp_real, dissim_train)
    disp_train2 = dp.align_all_xs_via_dissim(disp_real, dissim_train)

    assert type(disp_train) == type(disp_train2)
    assert np.shape(disp_train) == np.shape(dissim_train2)
    # print(np.shape(disp_train))
    assert (np.array(disp_train2) == np.array(disp_train)).all()

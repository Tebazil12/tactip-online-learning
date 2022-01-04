import numpy as np
# import scipy
import scipy.optimize
import matplotlib.pyplot as plt
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.gp as gp


def add_line_mu(disps, mu_for_line):
    """

    :param disps: should be np.array of shape (n_taps_on_line, 1)
    :param mu_for_line: single number to be added uniformly as mu for the given disps
    :return: x as a np.array of shape (n_taps_on_line, 2)
    """
    # print(f"disps for line adding: {disps} of type {type(disps)} and shape {np.shape(disps)}")



    if np.shape(disps)[1]>2: # in case of 3d using planes, need to replace phi
        disps[:,-1] = mu_for_line
        result = disps
    else:
        mus = np.array([[mu_for_line]] * np.shape(disps)[0])
        result = np.concatenate((disps, mus), axis=1)

    return np.array(result, dtype=np.float64) # normal int etc can't be mathed


def add_mus(disps, mu_limits=[-1, 1], line_ordering=None):
    """
    Disps must be a list, with each entry containing a np.array of
    displacement values (i.e. an entry per line)
    """
    # print(disps)

    n_disp_lines = len(disps)

    # [np.empty(disps[0].shape)] * len(disps)
    x = np.empty((n_disp_lines, disps[0].shape[0], 2))

    mu_for_line = np.linspace(mu_limits[0], mu_limits[1], n_disp_lines)
    # print(mu_for_line)

    for i, disp in enumerate(disps):
        if line_ordering is not None:
            if len(line_ordering) != n_disp_lines:
                raise NameError("line_ordering has different length to disps")
            # print(line_ordering[i])
            mus = np.full(disp.shape, mu_for_line[line_ordering[i]])
        else:
            mus = np.full(disp.shape, mu_for_line[i])
        # print(mus)
        # print(type(mus))

        an_x = np.vstack((disp, mus))

        # print(an_x)

        x[i] = an_x.T  # todo is this ok, or should np.copy be used for assignment?

    # print(x)
    # print(x.shape)
    return x


def best_frame(all_frames, neutral_tap=None, selection_criteria="Max"):
    """
    For the given tap, select frame with the highest average pin displacement
    from the first frame. Takes in one tap, in form (16ish x 126 ish x 2)
    np.array, returns (1 x 126ish x 2) np.array

    :param neutral_tap: needs to be flattened (will be expanded)
    """
    # make into pin displacement instead of absolute position
    if neutral_tap is None:
        all_frames_disp = all_frames - all_frames[0]
    else:
        # todo safety check lengths of neutral and all frames...
        neutral_tap_2d = neutral_tap.reshape(int(neutral_tap.shape[0] / 2), 2)
        all_frames_disp = all_frames - neutral_tap_2d

    # Find frame where largest pin displacement takes place (on average)
    # #TODO this method is not the same as in MATLAB!
    # Average disps per frame (over all pins)
    mean_disp_per_frame = np.mean(np.abs(all_frames_disp), axis=1)

    # Find euclidean distance per frame
    distances_all_disps = np.linalg.norm(mean_disp_per_frame, axis=1)

    if selection_criteria == "Max":  # tap will be pin DISLACEMENTS (relative)
        # Find frame with max euclidean distance
        result = np.where(distances_all_disps == np.amax(distances_all_disps))
        max_frame_i = result[0][0]

        tap = all_frames_disp[max_frame_i]

    elif selection_criteria == "Mean":  # tap will be pin POSITIONS (absolute)
        # this way does not select a frame, but creates a new frame which is the average
        tap = np.mean(all_frames, 0)
    else:
        print(selection_criteria)
        raise NameError("Selection criteria for frames is not recognised")

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


def calc_dissims(y_train, ref_tap, method="euclidean"):
    # print("calc_dissim")
    print(f"ytrain shape {y_train.shape}")

    if type(y_train) is not np.ndarray:
        raise NameError(f"y_train is not a np.array: {type(y_train)}")

    if type(ref_tap) is not np.ndarray:
        raise NameError(f"ref tap is not a np.array: {type(ref_tap)}")

    diffs = -y_train + ref_tap
    print(diffs.shape)

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
        [ref_tap], y_train, method
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
    fig = plt.figure()
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
    plt.show()


def align_all_xs_via_dissim(disp, dissim):
    # todo for loop
    corrected_disps = [None] * len(dissim)
    for i in range(0, len(dissim)):
        [corrected_disps[i],_] = align_radius(disp[0], dissim[i])

    return corrected_disps


def align_radius(disp, dissim, gp_extrap=False, show_graph=False, start_hypers=None):

    # print(show_graph)
    if type(disp) is not np.ndarray:
        raise NameError(f"disp should be np.array not {type(disp)}")

    if type(dissim) is not np.ndarray:
        raise NameError(f"disp should be np.array not {type(dissim)}")

    if len(disp) != len(dissim):
        raise NameError(f"disp is len({len(disp)}) but dissim is len({len(dissim)})")

    if gp_extrap:
        if  start_hypers is None:
            sigma_n_diss = .01 # this should really be calculated from sigma_n_y/same data
            # start_params = [100, 9]  # sigma_f and L respectively
            start_params = np.array([5,5])  # sigma_f and L respectively
        else:
            sigma_n_diss = start_hypers[0]
            start_params = start_hypers[1:] # sigma_f and L respectively

        data = [disp, np.array([dissim]).T, sigma_n_diss]
        # minimizer_kwargs = {"args": data}
        result = scipy.optimize.minimize(
            gp.max_log_like, start_params, args=data,method="BFGS",
            options={"gtol": 0.0000001, "maxiter": 30000})#, method="BFGS",tol=0.000001,
            # options={"gtol": 0.00001, "maxiter": 300000},
        # )
        # print(result)

        [sigma_f, L] = result.x
        print(f"sigmaf={sigma_f} L={L}")

        disp_stars, dissim_stars = gp.interpolate(
            disp, dissim, sigma_f, L, sigma_n_diss
        )
        print(f"disp_stars = {disp_stars}")
        print(f"dissim_stars = {dissim_stars}")
        if show_graph is True:
            show_dissim_profile([disp_stars, disp], [dissim_stars, dissim])
        # show_dissim_profile(np.vstack(([disp_stars],disp)), np.vstack(([dissim_stars],dissim)))

        result = np.where(dissim_stars == np.amin(dissim_stars))
        # print(result[0][0])
        disp_offset = disp_stars[result[0][0]]

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

    return corrected_disp, disp_offset

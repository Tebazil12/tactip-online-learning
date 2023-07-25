import scipy.io
import scipy.spatial
import scipy.optimize
import numpy as np

import tactip_toolkit_dobot.experiments.online_learning.offline_setup.gplvm as gplvm

import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp


def load_data():
    n_angles = 19
    all_data = [None] * n_angles  # all_data[angle][disp][frame][pin][xory]

    # folder_path = "C:\\Users\\ea-stone\\Documents\\data\\singleRadius2019-01-16_1651\\"
    folder_path = "/home/lizzie/git/tactip_toolkit_dobot/data/singleRadius2019-01-16_1651/"

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


if __name__ == "__main__":
    all_data = load_data()
    # print(len(all_data[0]))

    ref_tap = dp.best_frame(
        all_data[10 - 1][11 - 1]
    )  # -1 to translate properly from matlab
    # print(ref_tap)
    # print((ref_tap.shape))

    # indexes copied from MATLAB, hence -1 for easy comparison
    i_training_angles = [10 - 1, 15 - 1, 19 - 1, 5 - 1, 1 - 1]
    [y_train, dissim_train] = dp.get_processed_data(
        all_data, ref_tap, indexes=i_training_angles
    )

    # with np.printoptions(precision=3, suppress=True):
    #     print(dissim_train)
    # print(len(y_train))
    # print(y_train[0].shape)
    # print(len(dissim))
    # print(dissim[0].shape)
    disp_real = [-np.arange(-10, 11, dtype=np.float)]
    dp.show_dissim_profile(disp_real, dissim_train)

    # calculate line shifts based off dissimilarity (EXCEPT HERE WE ARENT
    # USING GP TO PREDICT SMOOTH PROFILE, GP IS PROBABLY NOT IMPLEMENTED
    # CORRECTLY! Use flag to use gp smoothing)
    disp_train = dp.align_all_xs_via_dissim(disp_real, dissim_train)
    # print(len(disp_train))
    dp.show_dissim_profile(disp_train, dissim_train)
    # gplvm.gp_lvm_max_lik()

    # plt.show()  # needs to be at end or graphs will block everything/disappear

    # TODO optimize gp-lvm hyperparams
    print(np.argsort(i_training_angles))

    mu_range = [-2, 2]

    # need double arg sort to get index of sorted order
    x_train = dp.add_mus(
        disp_train,
        line_ordering=np.argsort(np.argsort(i_training_angles)),
        mu_limits=mu_range,  # to match offline in matlab
    )
    x_train = x_train.reshape(x_train.shape[0] * x_train.shape[1], x_train.shape[2])

    # reshape to be in correct format for GPLVM calcs
    y_train = np.array(y_train)
    y_train = y_train.reshape(y_train.shape[0] * y_train.shape[1], y_train.shape[2])

    print("inting model")
    # model = gplvm.GPLVM(x_train, y_train)  # init includes hyperpar optm.
    model = gplvm.GPLVM(x_train, y_train, sigma_f=5.57810908398668, ls=[-5.139036963224281, -3.8544548444550086])  # init includes hyperpar optm.
    print(vars(model))  # matrices print rather long

    print(f"done initing, sigma_f = {model.sigma_f}, ls = {model.ls}")

    # here matlab attempts to show pattern of taps using GP, to see if
    # predictions are sensible, but that is probably too much effort for
    # now (and the results weren't intelligible in matlab)

    # Test the GP-LVM optimisation
    [y_test, dissim_test] = dp.get_processed_data(all_data, ref_tap)
    # y_test = np.array(y_test)
    # y_test = y_test.reshape(y_test.shape[0] * y_test.shape[1], y_test.shape[2])

    # calculate line shifts based off dissimilarity (EXCEPT HERE WE ARENT
    # USING GP TO PREDICT SMOOTH PROFILE, GP IS PROBABLY NOT IMPLEMENTED
    # CORRECTLY! Use flag to use gp smoothing)
    disp_test = dp.align_all_xs_via_dissim(disp_real, dissim_test)

    print(f"shapedisp = {np.shape(disp_test)}")
    print("testing mus ")
    mus_test = model.optim_many_mu(disp_test, y_test)

    print(f"in main, final mus {mus_test}")
    mu_real = np.linspace(mu_range[0], mu_range[1], len(disp_test))
    print(f"in main, real mus {mu_real}")

    mu_error = mus_test - mu_real

    with np.printoptions(precision=3, suppress=True):
        print(f"error in mu predictions {mu_error}")

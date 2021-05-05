import numpy as np


def max_log_like(hyper_pars, data):
    [sigma_f, L] = hyper_pars
    [y, x, sigma_n] = data

    k_cap = calc_K(x, sigma_f, L, sigma_n)

    # with np.printoptions(precision=3, suppress=True):
    # print(k_cap)
    # print("########################################")

    r_cap = np.linalg.cholesky(k_cap)

    sign, logdet_K = np.linalg.slogdet(r_cap)

    alpha = np.linalg.solve(r_cap, np.linalg.solve(r_cap.conj().T, y))

    # todo check its doing matrix mult. not element wise mult
    mll = (
        (0.5 * (y.conj().T @ alpha))
        + 0.5 * logdet_K
        + 0.5 * k_cap.shape[0] * np.log(2 * np.pi)
    )

    # print(mll)
    # raise(stop)

    # print(".", end='')

    return mll


def calc_K(x, sigma_f, L, sigma_n):
    n_xs = len(x)

    k_cap = np.empty([n_xs, n_xs])

    for i in range(0, n_xs):
        for j in range(0, n_xs):
            k_cap[i, j] = calc_covariance(x[i], x[j], sigma_f, L)

    # print(k_cap)
    # print(k_cap.shape)

    k_cap = k_cap + (np.identity(n_xs) * sigma_n)

    return k_cap


def cal_K_star(x, x_star, sigma_f, L, sigma_n):
    # print(len(x))
    # print(x)
    # print(x_star)
    # print(sigma_f)
    # print(L)
    # print(x_star.shape)

    k_cap_star = np.empty(x.shape)

    for i in range(0, len(k_cap_star)):
        k_cap_star[i] = calc_covariance(x_star, x[i], sigma_f, L)

    if np.isnan(k_cap_star).any():
        raise NameError("Entry in k_cap_star is nan")

    # with np.printoptions(precision=3, suppress=True):
    #     print(k_cap_star)
    return k_cap_star


def calc_covariance(x, x_prime, sigma_f, L):
    if not np.isscalar(sigma_f):
        raise NameError(f"sigma_f must be a scalar, not an array: {sigma_f} {type(sigma_f)}")


    if np.isscalar(x):
        if not np.isscalar(L):
            raise NameError(f"x is scalar ({type(x)}) but L is {type(L)}")

    # if type(L) is not np.ndarray:
    #     raise NameError(f"L must be np.array but is: {type(L)}")
    # todo check shape of x and x_prime
    else:
        if np.isscalar(L):
            raise NameError(f"x is non-scalar ({type(x)}) but L is scalar ({type(L)})")
        if len(x) != len(L):
            raise NameError(f"Dimensions of x do not match number of Ls")

    # print("start of cal cov")
    # print(x,x_prime,sigma_f,L)

    x_diff = x - x_prime
    # print(x_diff)
    x_diff_sqr = np.asarray(x_diff ** 2)  # asarray as can be both scalar or array

    x_sqr_l_sqr = np.asarray(np.nan_to_num(x_diff_sqr / (2 * (L ** 2))))
    x_sqr_l_sqr[x_sqr_l_sqr == np.inf] = 0  # remove infs

    sum_of_sqrs = np.sum(x_sqr_l_sqr)

    exp_sum_sqr = np.exp(-sum_of_sqrs)

    k = exp_sum_sqr * (sigma_f ** 2)

    if np.isnan(k) or np.isinf(k):
        raise NameError("k contains not a number (has nan or inf)")

    if np.isscalar(k) == False:
        raise NameError("k is not scalar - must be a scalar")
    # print(k)
    return k


def interpolate(x, y, sigma_f, L, sigma_n, x_limits=None, x_step=0.1):
    """
    Given example data, x and y, and (optimised) hyperparameters, use a gp
    to interpolate y at a range of x's between the x_limits (both inclusive).
    If x_limits are not set, use the max and min of x as limits
    """
    y = np.array([y])  # make 2d, so that transpose works...

    k_cap = calc_K(x, sigma_f, L, sigma_n)
    # k_cap = np.array([k_cap])  # so that transpose works...

    if x_limits is None:
        x_stars = np.arange(np.amin(x), np.amax(x) + x_step, x_step)
    else:
        x_stars = np.arange(x_limits[0], x_limits[1] + x_step, x_step)

    x_stars = np.around(x_stars, 4)  # get rid of tiny errors

    y_stars = np.empty(x_stars.shape)

    for i, x_star in enumerate(x_stars):
        # print(i, x_star)

        k_cap_star = cal_K_star(x, x_star, sigma_f, L, sigma_n)
        k_cap_star = np.array([k_cap_star])

        # print("k_cap_star")
        # print(k_cap_star.shape)
        # print(k_cap_star)

        # print("inv_k")
        inv_K = np.linalg.inv(k_cap)
        # print(inv_K.shape)

        # print("y and y apostrophe")
        # print(y.shape)
        y_con_t = y.conj().T
        # print(y_con_t.shape)

        # print("K* x inv(K) x y_appostrophe ")
        ks_by_yt = k_cap_star @ inv_K @ y_con_t  # the @ means matrix mult.
        # print(ks_by_yt.shape)

        y_stars[i] = ks_by_yt
        # raise(stop)

        # k_star_star = calc_covariance(x_star, x_star, sigma_f, L)
    # with np.printoptions(precision=3, suppress=True):
    #     print(y_stars)
    return x_stars, y_stars

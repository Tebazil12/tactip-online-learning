import scipy.optimize
import numpy as np
import gp
import data_processing as dp


class GPLVM:
    sigma_n_y = 1.14  # TODO this is for the normal tactip, needs setting for others!

    def __init__(self, x, y, sigma_f=None, ls=None):
        """
        Take in x and y as np.arrays of the correct size and shape to be used
        """
        self.x = x
        self.y = y

        # print(y.shape)
        # print(x.shape)
        if sigma_f is None or ls is None:
            # optmise
            self.optim_hyperpars()
        else:
            # assuming hyperpars already optimised
            self.sigma_f = sigma_f
            self.ls = ls

    def max_ll_optim_hyperpars(self, to_optimise, set_vals):
        sigma_f, l_disp, l_mu = to_optimise
        x, y = set_vals
        return self.max_log_like(sigma_f, l_disp, l_mu, x, y)


    def max_ll_optim_mu(self, to_optimise, set_vals):
        mus = to_optimise
        disp, y = set_vals

        # make x from disp and optimising mu
        x = dp.add_mus(
            [disp],
            mu_limits=[mus, mus],  # same as only one line being passed
        )

        # print(y)
        # y = np.array([y])
        # print(y)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        # y = y.reshape(y.shape[0] * y.shape[1], y.shape[2])

        #todo add in self.x and self.y otherwise your not using the right model!

        all_xs = np.vstack((self.x, x))
        all_ys = np.vstack((self.y, y))

        return self.max_log_like(self.sigma_f, self.ls[0], self.ls[1], all_xs, all_ys)


    def max_log_like(self, sigma_f, l_disp, l_mu, x, y):

        # print(y.shape)
        d_cap, n_cap = y.shape

        s_cap = (1 / d_cap) * (y @ y.conj().T)

        # if not np.isscalar(s_cap):
        #     raise NameError("s_cap is not scalar!")

        ls = np.array([l_disp, l_mu])  # TODO confirm 2 ls works with calc_K

        k_cap = gp.calc_K(x, sigma_f, ls, self.sigma_n_y)

        r_cap = np.linalg.cholesky(k_cap)
        sign, logdet_K = np.linalg.slogdet(r_cap)

        part_1 = -(d_cap * n_cap) * 0.5 * np.log(2 * np.pi)
        part_2 = -d_cap * 0.5 * logdet_K

        # print("Here")
        # print(d_cap)
        # print(s_cap)
        # print(np.trace(np.linalg.inv(k_cap)))
        part_3 = -d_cap * 0.5 * np.trace(np.linalg.inv(k_cap) @ s_cap)

        neg_val = part_1 + part_2 + part_3
        # print(neg_val)
        return -neg_val  # because trying to find max with a min search

    def optim_hyperpars(self, x=None, y=None, start_hyperpars=None, update_data=False):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        if start_hyperpars is None:
            start_hyperpars = np.array(
                [1, 300, 5]
            )  # sigma_f , L_disp, L_mu respectively

        data = [x, y]
        # minimizer_kwargs = {"args": data}
        result = scipy.optimize.minimize(
            self.max_ll_optim_hyperpars,
            start_hyperpars,
            args=data,
            method="BFGS",
            options={"gtol": 0.01, "maxiter": 300},  # is this the best number?
        )
        # print(result)

        [sigma_f, l_disp, l_mu] = result.x
        self.sigma_f = sigma_f
        self.ls = [l_disp, l_mu]
        # print(result)

    def optim_mu(self, disps, y):
        #TODO test that hyperpars have been optimised as can't continue without

        mus_test = np.empty((len(disps)))

        # for loop is used so that only one line is trained at a time, otherwise
        # this isn't representative of online learning
        for i, disp in enumerate(disps):
            start_mu = 0 # only one value, which is mu for a line
            data = [disp, y[i]]
            # minimizer_kwargs = {"args": data}
            result = scipy.optimize.minimize(
                self.max_ll_optim_mu,
                start_mu,
                args=data,
                method="BFGS",
                options={"gtol": 0.01, "maxiter": 300},  #TODO is this the best number?
            )
            # print(result)

            [mus_test[i]] = result.x #TODO figure out return type
        print(f'The final mus list: {mus_test}')

        return mus_test
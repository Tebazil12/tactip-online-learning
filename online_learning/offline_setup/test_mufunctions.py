import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp
import numpy as np

a = np.array([[-10], [ -8], [ -6], [ -4], [ -2], [  0], [  2], [  4], [  6], [  8], [ 10]])

# a = [[2]]

# d = dp.add_line_mu(a, 4)

# print(d)

mu=2

x2 = dp.add_line_mu(a, mu)

x1 = dp.add_mus([a], mu_limits=[mu, mu]) # todo, this is actually broken (tries
# to output (1,22) into x which was inited as (11,2)... why did this not error
# in offline stuff?



print(f"x1={x1}, x2={x2}")

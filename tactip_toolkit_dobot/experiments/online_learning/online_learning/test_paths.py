import os
import time

# print( os.getenv('PWD'))

print(os.getcwd())

print(__file__)

full_path = os.path.join(os.getcwd(), __file__)
print(os.path.join(os.getcwd(), __file__))

trnced, enf = os.path.split(full_path)

natahns_filename = os.path.join(
    "data",
    os.path.basename(__file__)[:-3] + "_" + time.strftime("%m-%d_%H%Mh"),
    "meta.json",
)

print(natahns_filename)

print(os.path.join(trnced, natahns_filename))

# print('basename:    ', os.path.basename(__file__))
# print('dirname:     ', os.path.dirname(__file__))

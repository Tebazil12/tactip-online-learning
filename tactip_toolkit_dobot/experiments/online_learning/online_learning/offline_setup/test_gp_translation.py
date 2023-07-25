import gp
import matlab.engine
import os, io

"""This will only work on windows (due to bad core handling of paths)"""
eng = matlab.engine.start_matlab()

path = os.path.join(os.environ["PYTHONPATH"], "core", "utils")
eng.addpath(eng.genpath(path))  # presumably this adds the tactile core code


def test_max_log_like():
    pass

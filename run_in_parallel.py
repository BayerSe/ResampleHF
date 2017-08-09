#!/usr/bin/env python

import os
from joblib import Parallel, delayed

from process_data import process_data


def f(asset):
    process_data(asset=asset, avg_dur=60, path="data")


if __name__ == "__main__":
    Parallel(n_jobs=15)(delayed(f)(asset) for asset in os.listdir("data"))

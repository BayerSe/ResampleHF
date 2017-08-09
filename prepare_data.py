#!/usr/bin/env python

from glob import glob
import os

import numpy as np
import pandas as pd
from natsort import natsorted


def prepare_forex(asset, path):
    if not os.path.exists(path + asset + "/h5"):
        os.makedirs(path + asset + "/h5")

    def dateparse(date, time):
        return pd.to_datetime(date + time, format='%Y%m%d%H%M%S%f')

    def process_data(file):
        data = pd.read_csv(file, header=None, names=["Date", "Time", "Bid", "Ask"], index_col="datetime",
                           parse_dates={'datetime': ['Date', 'Time']}, date_parser=dateparse)

        # Add the midquote
        data["Midquote"] = (data["Bid"] + data["Ask"]) / 2
        data.drop(["Bid", "Ask"], axis=1, inplace=True)
        data = data.iloc[:, 0]

        # Shift the index such that trading time is from 0-24h
        idx_1 = data[:'2014-08-02'].index + pd.Timedelta('8h')
        idx_2 = data['2014-08-03':].index + pd.Timedelta('6h')
        data.index = idx_1.union(idx_2)

        # Change the first and the last timestamp
        def change_timestamp(x):
            if len(x) > 0:
                x[0] = x[0].replace(hour=0, minute=0, second=0, microsecond=0)
                x[-1] = x[-1].replace(hour=23, minute=59, second=59, microsecond=999999)
                return x

        new_idx = data.index.to_series().groupby(pd.TimeGrouper("1d")).apply(change_timestamp)
        data.index = new_idx

        # Save the data to the disk
        for day, data_day in data.groupby(pd.TimeGrouper("1d")):
            if data_day.size > 0:
                file = path + asset + "/h5/" + day.strftime("%Y-%m-%d") + ".h5"
                data_day.to_hdf(file, "table")

    # List all files and loop over them
    file_list = natsorted(glob(path + asset + "/raw/*"))
    for file in file_list:
        process_data(file)


def prepare_nyse(asset, path):
    if not os.path.exists(path + asset + "/h5"):
        os.makedirs(path + asset + "/h5")

    def dateparse(date, time):
        time = float(time)
        time_s = int(time)
        f = time - time_s
        m, s = divmod(time_s, 60)
        h, m = divmod(m, 60)
        return pd.to_datetime(date + str(h).zfill(2) + str(m).zfill(2) + str(s).zfill(2) + ("%.6f" % f)[2:],
                              format='%Y%m%d%H%M%S%f')  # TODO: improve rounding of the fractional part

    def process_data(file):
        data = pd.read_csv(file, usecols=["Date", "Time", "Bid", "Ask"], index_col="datetime",
                           parse_dates={'datetime': ['Date', 'Time']}, date_parser=dateparse)

        # Add the midquote
        data["Midquote"] = (data["Bid"] + data["Ask"]) / 2
        data.drop(["Bid", "Ask"], axis=1, inplace=True)
        data = data.iloc[:, 0]

        # Shift the index such that trading time is from 0-6:30h
        data.index -= pd.to_timedelta("9.5h")

        # Change the first and the last timestamp
        def change_timestamp(x):
            if len(x) > 0:
                x[0] = x[0].replace(hour=0, minute=0, second=0, microsecond=0)
                x[-1] = x[-1].replace(hour=6, minute=30, second=0, microsecond=0)
                return x

        new_idx = data.index.to_series().groupby(pd.TimeGrouper("1d")).apply(change_timestamp)
        data.index = new_idx

        # Save the data to the disk
        for day, data_day in data.groupby(pd.TimeGrouper("1d")):
            if data_day.size > 0:
                file = path + asset + "/h5/" + day.strftime("%Y-%m-%d") + ".h5"
                data_day.to_hdf(file, "table")

    # List all loop over them
    file_list = natsorted(glob(path + asset + "/raw/**/" + asset + "_quotes*.txt", recursive=True))
    for file in file_list:
        process_data(file)


def remove_bad_days_forex(asset, path):
    file_list = glob(path + asset + "/" + "h5" + "/*")
    for file in file_list:
        print(file)
        year, month, day = os.path.basename(file).replace(".h5", "").split("-")

        # Kick Jan 1, Dec 25
        if ((day == "01") and (month == "01")) or ((day == "25") and (month == "12")):
            os.remove(file)
            continue

        # Check whether the trading time is less than 22:30h
        data = pd.read_hdf(file, "table")
        check = data.index[-2] - data.index[1] < pd.to_timedelta("22.5h")

        if check:
            os.remove(file)


def remove_bad_days_nyse(asset, path):

    file_list = pd.Series(glob(path + asset + "/" + "h5" + "/*"))
    ff = np.array([int(os.path.basename(f).replace(".h5", "").replace("-", "")) for f in file_list])
    bad_days = np.array([20010608, 20010703, 20011123, 20011224,
                         20020705, 20020911, 20021129, 20021224,
                         20030703, 20031128, 20031224, 20031226,
                         20041126, 20051125, 20060703, 20061124,
                         20070703, 20071123, 20071224,
                         20080703, 20081128, 20081224,
                         20091127, 20091224,
                         20101126, 20111125,
                         20120703, 20121123, 20121224,
                         20130606, 20130703, 20131129, 20131224,
                         20140703, 20141128, 20141224,
                         20151127, 20151224,
                         20161125,
                         20170703, 20171124,
                         20180703, 20181123, 20181224,
                         20150128
                         ])
    flag = np.isin(ff, bad_days)
    for file in file_list[flag]:
        os.remove(file)


def create_index(asset_list, path, index_name):
    if not os.path.exists(path + index_name + "/h5"):
        os.makedirs(path + index_name + "/h5")
    file_list = [pd.Series(glob(path + asset + "/" + "h5" + "/*")) for asset in asset_list]
    file_list = pd.concat(file_list).reset_index(drop=True)
    counts = file_list.apply(lambda x: os.path.basename(x).replace(".h5", "")).value_counts()
    common_days = counts[counts == len(asset_list)].index.unique().sort_values()
    for day in common_days:
        print(day)
        index = pd.concat([pd.read_hdf(file, "table") for file in file_list[file_list.str.contains(day)]], axis=1).ffill().mean(axis=1)
        index.to_hdf(path + index_name + "/h5/" + day + ".h5", "table")


if __name__ == "__main__":
    path = "."

    # Prepare forex data
    for asset in ["EURGBP", "EURUSD"]:
        print(asset)
        prepare_forex(asset, path)
        remove_bad_days_forex(asset, path)

    # Prepare stock data
    asset_list = ['AA', 'AXP', 'BA', 'BAC', 'C', 'CAT', 'DD', 'DIS', 'GE', 'GS', 'HD', 'HON', 'HPQ', 'IBM', 'IP', 'JNJ',
                  'JPM', 'KO', 'MCD', 'MMM', 'MO', 'MRK', 'NKE', 'PFE', 'PG', 'UTX', 'VZ', 'WMT', 'XOM', "IDX"]

    create_index(asset_list=asset_list, path=path, index_name="IDX")
    asset_list += ["IDX"]

    for asset in asset_list:
        print(asset)
        prepare_nyse(asset, path)
        remove_bad_days_nyse(asset, path)




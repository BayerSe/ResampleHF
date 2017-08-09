#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import sys
from glob import glob
from natsort import natsorted
from scipy.signal import savgol_filter
from scipy.special import gamma


def tripower_volatility(x):
    """
    Realized tripower volatility (e.g. Barndorff-Nielsen, Shephard, and Winkel (2006))
    """
    x = pd.Series(x)
    xi = 0.5 * (gamma(5 / 6) / gamma(1 / 2)) ** -3
    z = (x.abs() ** (2 / 3) * x.shift(1).abs() ** (2 / 3) * x.shift(-1).abs() ** (2 / 3)).bfill().ffill()
    return xi * z.sum()


def shortest_half(x):
    """
    Shortest-half scale estimator (Rousseeuw and Leroy, 1998)
    """
    xs = np.sort(x)
    l = x.size
    h = int(np.floor(l / 2) + 1)
    if l % 2 == 0:
        sh = 0.7413 * np.min(xs[h - 1:] - xs[:h - 1])
    else:
        sh = 0.7413 * np.min(xs[h - 1:] - xs[:h])
    return sh


def time_to_ssm(x):
    """
    Transforms a datetime index into the numerical date (YYYMMDD) and the seconds since midnight.
    """
    x = pd.DataFrame(x)
    date = x.index.map(lambda d: d.year * 10000 + d.month * 100 + d.day).values
    ssm = x.index.map(lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6).values
    x.insert(0, "Date", date)
    x.insert(1, "SSM", ssm)
    x.reset_index(drop=True)
    return x


def resample_prices(intensity, data, n_trades):
    T = intensity.size                   # Trading seconds per day
    eps = 0.000001 if T == 86400 else 0  # Ensure that the days do not overlap
    if intensity.isnull().any():
        intensity.interpolate(method="pchip", inplace=True, limit_direction="both", limit=T)
        intensity[intensity < 0] = 0  # interpolated values could be negative
    Q = intensity.cumsum() / intensity.sum() * T
    Q_inv = pd.Series(np.concatenate((np.array([0]),
                                      np.interp(np.arange(1, T), xp=Q.values, fp=Q.index),
                                      np.array([T-eps]))), index=range(T+1))
    idx = data.index[0] + pd.to_timedelta(Q_inv, unit="s")
    reindexed_data = data.reindex(idx, method="ffill")
    resampled_data = reindexed_data.iloc[np.linspace(0, reindexed_data.size - 1, num=n_trades).round()]

    return Q, resampled_data


def process_data(asset, avg_dur, path):
    # region Set some variables
    T = 86400 if asset in ["EURGBP", "EURUSD"] else 23400
    n_trades = int(np.ceil(T / avg_dur)) + 1
    file_list = natsorted(glob(path + asset + "/" + "h5" + "/**"))
    dt = pd.to_datetime([os.path.basename(f).replace(".h5", "") for f in file_list])
    total_trades_per_second = pd.Series(0, index=range(1, T + 1))

    # Empty pd.Series for the resampled prices
    cts = pd.Series()
    tts = pd.Series()
    trts = pd.Series()
    da = pd.Series()
    tt = pd.Series()
    bts = pd.Series()
    sbts = pd.Series()
    wsd = pd.Series()

    # Time slices for the duration adjustment
    if T == 23400:
        slices = pd.DataFrame(columns=["start", "end"], index=range(14))
        slices.iloc[0, :2] = 0, 15 * 60
        for i in range(1, 13):
            slices.iloc[i, :2] = slices.iloc[i - 1, 1], slices.iloc[i - 1, 1] + 30 * 60
        slices.iloc[-1, :2] = 22500, 23400
    elif T == 86400:
        slices = pd.DataFrame(columns=["start", "end"], index=range(48))
        slices.iloc[0, :2] = 0, 30 * 60
        for i in range(1, 48):
            slices.iloc[i, :2] = slices.iloc[i - 1, 1], slices.iloc[i - 1, 1] + 30 * 60

    # For the weighted standard deviation
    standardized_returns = pd.DataFrame(np.nan, index=dt, columns=range(int(np.ceil(T / 300))))

    # Store all estimated intentsity functions
    # TODO: replace with xarray or list
    all_Q = pd.Panel(np.nan, items=["CTS", "TrTS", "TTS", "TT", "DA", "BTS", "sBTS", "WSD"],
                     major_axis=dt,
                     minor_axis=range(1, T + 1))
    # endregion

    # Loop over the file list
    for row, file in enumerate(file_list):
        print(asset + ": "+ dt[row].strftime("%Y-%m-%d"))

        # region Load the h5 file
        data = pd.read_hdf(file, "table")

        # Merge duplicate timestamps
        data_unique = data.groupby(data.index).median()
        # endregion

        # region Store the number of trades per second
        trades_per_second = data.groupby(
            data.index.map(lambda t: t.hour * 3600 + t.minute * 60 + t.second)).count(). \
            reindex(range(0, T)).fillna(0).astype(int)
        trades_per_second.index += 1
        total_trades_per_second += trades_per_second
        idx = data.diff() != 0
        nz_trades_per_second = data[idx].groupby(
            data[idx].index.map(lambda t: t.hour * 3600 + t.minute * 60 + t.second)).count(). \
            reindex(range(0, T)).fillna(0).astype(int)
        # endregion

        # region Calendar Time Sampling
        intensity = pd.Series(1, index=range(1, T + 1))
        Q, rs = resample_prices(intensity=intensity, data=data_unique, n_trades=n_trades)
        all_Q["CTS"].iloc[row] = Q.values
        cts = cts.append(rs)
        # endregion

        # region Transaction Time Sampling
        Q, rs = resample_prices(intensity=trades_per_second, data=data_unique, n_trades=n_trades)
        all_Q["TrTS"].iloc[row] = Q.values
        trts = trts.append(rs)
        # endregion

        # region Tick Time Sampling
        Q, rs = resample_prices(intensity=nz_trades_per_second, data=data_unique, n_trades=n_trades)
        all_Q["TTS"].iloc[row] = Q.values
        tts = tts.append(rs)
        # endregion

        # region Time transformation
        """
        Computes the Q function using the Time Transformation approach of Wu (2012), also see Tse & Dong (2014).
        See Appendix A.1 of Tse & Dong (2014) for the implementation details.
        """
        Q, rs = resample_prices(intensity=total_trades_per_second, data=data_unique, n_trades=n_trades)
        all_Q["TT"].iloc[row] = Q
        tt = tt.append(rs)
        # endregion

        # region Duration adjustment
        """
        This is the approach of Bauwens and Giot (2000).
        For the specific implementation details, see Tse & Dong (2014).

        For the Forex, we split the data into consecutive 30 min intervals.
        """
        trades_per_slice = pd.Series(np.nan, index=range(1, T + 1))
        for i in range(slices.shape[0]):
            start = slices.iloc[i]["start"]
            end = slices.iloc[i]["end"]
            midpoint = int(start + (end - start) / 2)
            trades_per_slice.iloc[midpoint] = total_trades_per_second.iloc[start:end].sum() / (end - start)

        # Resample the prices
        Q, res = resample_prices(intensity=trades_per_slice, data=data_unique, n_trades=n_trades)
        all_Q["DA"].iloc[row] = Q.values
        da = da.append(rs)
        # endregion

        # region Business Time Sampling
        rp = data_unique.resample("60s", closed="right", label="right").last()
        if T == 86400:
            rp = pd.concat((rp.iloc[:-1], data_unique.iloc[-1:]))
        cts_returns = np.log(rp.ffill()).diff().iloc[1:]

        # Subsampled tripower volatility over a 10 / 30 min grid
        window = 30 if T == 86400 else 10
        zz = pd.Series(np.nan, index=np.arange(int(window / 2 * 60), T, int(window * 60)))
        for ii in zz.index:
            iii = int(ii / 60)  # in minutes, since cts returns are minute-based
            uu = [str(iii - b) + ":" + str(iii + 10 - b) for b in np.arange(window, 0, -1) if
                  iii - b >= 0 and iii + b <= T / 60]  # attention: 0-indexing!
            zz[ii] = np.array(
                [tripower_volatility(cts_returns.iloc[int(uu[i].split(":")[0]):int(uu[i].split(":")[1])]) for i in
                 range(len(uu))]).mean()
        zz = zz.reindex(range(1, T + 1))

        # Resample the prices
        Q, rs = resample_prices(intensity=zz, data=data_unique, n_trades=n_trades)
        all_Q["BTS"].iloc[row] = Q.values
        bts = bts.append(rs)
        # endregion

        # region Smoothed Business Time Sampling
        # Tripower volatility over a rolling 10 / 30 min grid
        window = 30 if T == 86400 else 10
        zz = cts_returns.rolling(window=window).apply(tripower_volatility).dropna()

        # Shift index to the middle of the interval
        zz.index = np.arange(int(window/2 * 60), T - int(window/2 - 1) * 60, 60)
        zz = zz.reindex(range(1, T + 1))

        # Smooth the series
        zz[zz < 1e-10] = 1e-10
        log_zz = np.log(zz.dropna())
        log_zz = pd.Series(savgol_filter(log_zz, window_length=int(np.ceil(log_zz.size * 0.2) // 2 * 2 + 1),
                                         polyorder=3), index=log_zz.index)
        zz[log_zz.index] = pd.Series(np.exp(log_zz))

        # Resample the prices
        Q, rs = resample_prices(intensity=zz, data=data_unique, n_trades=n_trades)
        all_Q["sBTS"].iloc[row] = Q.values
        sbts = sbts.append(rs)
        # endregion

        # region Weighted Standard Deviation
        """
        Boudt, Croux and Laurent (2011)
        """
        M = 300
        rp = data_unique.resample(str(M) + "s", closed="right", label="right").last()
        if T == 86400:
            rp = pd.concat((rp.iloc[:-1], data_unique.iloc[-1:]))
        cts_returns = np.log(rp.ffill()).diff().iloc[1:]

        # Realized bipower variation
        rbv = np.sqrt(np.pi / 2 * 1 / (T / M - 1) * (cts_returns.abs() * cts_returns.shift().abs()).sum())  # (2.6)

        # Standardized returns
        standardized_returns.iloc[row] = (cts_returns / rbv).values  # (2.7)

        # Current set of returns
        rr = standardized_returns.iloc[:row + 1, :]

        # Shortest half scale
        shorth = rr.apply(shortest_half)  # (2.9)
        f_shorth = shorth / np.sqrt(M / T * shorth.pow(2).sum())  # (2.10

        # Weights (below 2.12)
        w = (rr / f_shorth).applymap(lambda x: 1 if x ** 2 <= 6.635 else 0)

        # WSD (2.12)
        z = np.sqrt(1.081 * (rr.pow(2) * w).sum() / w.sum())
        f_wsd = z / np.sqrt(M / T * z.pow(2).sum())

        # Shift index to the middle of the interval
        f_wsd.index = np.arange(int(2.5 * 60), T, 5 * 60)
        f_wsd = f_wsd.reindex(range(1, T + 1))

        # CTS for the first day
        if row == 0:
            f_wsd.fillna(1, inplace=True)

        # Resample the prices
        Q, rs = resample_prices(intensity=f_wsd, data=data_unique, n_trades=n_trades)
        all_Q["WSD"].iloc[row] = Q.values
        wsd = wsd.append(rs)
        # endregion

    # region Save the results
    results_path = path + asset + "/resampled_prices/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for approach in ["cts", "tts", "trts", "da", "tt", "bts", "sbts", "wsd"]:
        x = eval(approach)
        x.to_hdf(results_path + approach + ".h5", "table")
        time_to_ssm(x).to_csv(results_path + approach + ".csv", index=None, header=False, float_format='%.6f')
    all_Q.to_hdf(results_path + "Q.h5", "table", complevel=5, complib="zlib")
    # endregion


if __name__ == "__main__":

    asset = sys.argv[1]
    process_data(asset=asset, avg_dur=60, path=".")

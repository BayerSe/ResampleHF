import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.ioff()

if __name__ == "__main__":

    sns.set(style="whitegrid",
            rc={'font.family': 'serif',
                'axes.labelsize': 7, 'axes.titlesize': 7, 'font.size': 7, 'legend.fontsize': 7,
                'xtick.labelsize': 7, 'ytick.labelsize': 7})

    data_path = "data"
    results_path = "plots"

    for asset in os.listdir(data_path):

        path = data_path + asset + "/resampled_prices/"
        Q = pd.read_hdf(path + "Q.h5", "table")

        QQ = Q.iloc[:, -4, :]
        QQQ = QQ.iloc[:, ~QQ.columns.isin(["TTS", "TrTS"])]

        ax = QQ.plot(figsize=(6, 4))
        sns.despine()
        ax.set_ylabel("Time Transformation")
        ax.set_xlabel("Time of the Day")
        plt.tight_layout(pad=0.1)
        plt.savefig(results_path + asset + "_Q.pdf")

        QQQ.diff().plot(figsize=(6, 4))
        ax.set_ylabel("Intensity")
        ax.set_xlabel("Time of the Day")
        sns.despine()
        plt.tight_layout(pad=0.1)
        plt.savefig(results_path + asset + "_Q_diff.pdf")

        if asset in ["EURUSD", "EURGBP", "IBM", "IDX"]:
            for method in ["BTS", "sBTS"]:
                QQQQ = Q.iloc[:, -10:, :].loc[method].T
                ax = QQQQ.diff().plot(figsize=(6, 4))
                ax.set_ylabel("Intensity")
                ax.set_xlabel("Time of the Day")
                sns.despine()
                plt.tight_layout(pad=0.1)
                plt.savefig(results_path + asset + "_" + method + "_time.pdf")

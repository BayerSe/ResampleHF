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

    data_path = "D:/HF_Data/"
    results_path = "D:/Q_plots/"

    for asset in os.listdir(data_path):
        print(asset)

        # Load the Q function
        Q = pd.read_hdf(data_path + asset + "/resampled_prices/Q.h5", "table")

        # Average over the intensity functions
        Q_avg = pd.concat((Q.loc["CTS"].iloc[0].rename("CTS"),
                           Q.loc["TrTS"].mean().rename("TickTS"),
                           Q.loc["TTS"].mean().rename("nzTickTS"),
                           Q.loc["TT"].iloc[-1].rename("TTS"),
                           Q.loc["DA"].iloc[-1].rename("DAS"),
                           Q.loc["BTS"].mean().rename("BTS"),
                           Q.loc["sBTS"].mean().rename("sBTS"),
                           Q.loc["WSD"].iloc[-1].rename("WSDS")
                           ), axis=1)

        # Plots
        ax = Q_avg.plot(figsize=(6, 4))
        ax.set_ylabel("Transformed Time")
        ax.set_xlabel("Time of the Day")
        sns.despine()
        plt.tight_layout()
        plt.savefig(results_path + asset + "_Q.pdf")
        plt.close("all")

        ax = Q_avg.diff().plot(figsize=(6, 4))
        ax.set_ylabel("Intensity")
        ax.set_xlabel("Time of the Day")
        sns.despine()
        plt.tight_layout()
        plt.savefig(results_path + asset + "_Q_diff.pdf")
        plt.close("all")

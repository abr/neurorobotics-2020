import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from abr_analyze import DataHandler
from abr_analyze.paths import figures_dir


def plot_latency(ax=None):
    test_group = "weighted_tests"
    main_db = "dewolf2018weight-tests"
    test_list = ["pd", "pid", "nengo_loihi1k", "nengo_cpu1k", "nengo_gpu1k"]
    colors = ["k", "tab:brown", "b", "g", "r"]

    dat = DataHandler(main_db)
    times = []
    for ii, test in enumerate(test_list):
        data = dat.load(
            parameters=["time"],
            save_location="weighted_tests/%s/session000/run000" % test,
        )
        time = data["time"]
        times.append(np.array(time))
    times = np.array(times)

    if ax is None:
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Control loop latency")
    ax.set_ylabel("Time (ms)",)
    ax.grid()

    for ii, time in enumerate(times):
        violin_parts = ax.violinplot(
            time, [ii], showmeans=True, showextrema=False, points=1000
        )
        violin_parts["cmeans"].set_edgecolor("black")
        violin_parts["cmeans"].set_linewidth(3)
        violin_parts["bodies"][0].set_alpha(0.75)
        violin_parts["bodies"][0].set_facecolor(colors[ii])
        print("%s mean: %.5f" % (test_list[ii], np.mean(time)))

    ax.set_ylim([0.0025, 0.005])
    yticks = np.arange(0.0025, 0.005, 0.0005)
    ylabels = np.arange(2.5, 6, 0.5)
    plt.yticks(yticks, ylabels)

    names = ["PD", "PID", "Loihi", "CPU", "GPU"]
    plt.xticks(range(len(names)), names)

    return violin_parts


if __name__ == "__main__":
    violin_parts = plot_latency()

    plt.tight_layout()
    loc = "figures/latency.pdf"
    print("Figure saved to %s" % loc)
    plt.savefig(loc)
    plt.show()

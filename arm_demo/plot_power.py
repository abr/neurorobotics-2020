import numpy as np
import matplotlib.pyplot as plt
import csv


def plot_power(ax=None, plot_full_signal=False):
    # read in CPU power numbers for PD controller
    pd_data = []
    with open("data/PD_CPU_power.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for ii, row in enumerate(reader):
            if ii > 0:
                pd_data.append(row[-2].split(" ")[0])
    pd_data = np.array([float(val) for val in pd_data])
    pd_inuse = np.mean(pd_data[28:45])

    # read in CPU power numbers for Nengo CPU controller
    cpu_data = []
    with open("data/Nengo_cpu_CPU_power.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for ii, row in enumerate(reader):
            if ii > 0:
                cpu_data.append(row[-2].split(" ")[0])
    cpu_data = np.array([float(val) for val in cpu_data])
    cpu_inuse = np.mean(cpu_data[26:45])

    # read in GPU power numbers for Nengo GPU controller
    gpu_data = []
    with open("data/Nengo_gpu_GPU_power.log", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for ii, row in enumerate(reader):
            if len(row) > 0 and "Draw" in row[0]:
                row = row[0]
                temp = row.split(": ")[1]
                temp = float(temp.split(" ")[0])
                gpu_data.append(temp)
    gpu_data = np.array(gpu_data)
    gpu_inuse = np.mean(gpu_data[100:150])
    gpu_baseline = np.mean(gpu_data[0:50])

    loihi_data = []
    with open("data/telemetry.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")

        for ii, row in enumerate(reader):
            if ii == 0:
                index0 = row.index("READ_VOUT-U0_RUN_VDD")
                index1 = row.index("READ_VOUT-U0_RUN_VDDM")
                index2 = row.index("READ_IOUT-U0_RUN_VDD")
                index3 = row.index("READ_IOUT-U0_RUN_VDDM")
            else:
                # power = VDD*i_VDD + VDDM*i_VDDM
                loihi_data.append(
                    float(row[index0]) * float(row[index2])
                    + float(row[index1]) * float(row[index3])
                )
    # count entire cost of powering Loihi in calculation, not just dynamic power cost
    loihi_chip_inuse = np.mean(loihi_data[120:170])

    if plot_full_signal:
        fig, axs = plt.subplots(4, 1)
        axs[0].plot(pd_data)
        axs[0].set_title("Non-adaptive CPU")
        axs[1].plot(loihi_data)
        axs[1].set_title("Adaptive Loihi")
        axs[2].plot(cpu_data)
        axs[2].set_title("Adaptive CPU")
        axs[3].plot(gpu_data)
        axs[3].set_title("Adaptive GPU")
        for ax in axs:
            ax.set_ylabel("Power (W)")
            ax.set_xlabel("Time (ms)")
        plt.tight_layout()

    # calculate power for each relative to loihi power consumption
    cpu_scaled = (cpu_inuse - pd_inuse) / loihi_chip_inuse
    loihi_scaled = loihi_chip_inuse / loihi_chip_inuse
    gpu_scaled = (gpu_inuse - gpu_baseline) / loihi_chip_inuse
    names = ["Loihi", "CPU", "GPU"]
    print("[Loihi, CPU, GPU]: ", [loihi_scaled, cpu_scaled, gpu_scaled])

    if ax is None:
        fig = plt.figure(figsize=(6, 3))
        ax = fig.gca()

    # ax.bar(range(3), [loihi_scaled, cpu_scaled, gpu_scaled])
    ax.bar([0], [loihi_scaled], color="b", alpha=0.75)
    ax.bar([1], [cpu_scaled], color="g", alpha=0.75)
    ax.bar([2], [gpu_scaled], color="r", alpha=0.75)
    rects = ax.patches
    labels = ["%.1fx" % val for val in [loihi_scaled, cpu_scaled, gpu_scaled]]

    for rect, label in zip(rects, labels):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            1.2,
            label,
            ha="center",
            va="bottom",
            fontdict={"weight": "bold", "size": "16"},
        )

    plt.xticks(range(3), names)
    ax.grid()
    ax.set_yscale("log")
    ax.set_ylabel("Power (W)")
    ax.set_title("Power cost of adaptation")


if __name__ == "__main__":
    plot_power()

    plt.tight_layout()
    loc = "figures/power.pdf"
    print("Figure saved to %s" % loc)
    plt.savefig(loc)
    plt.show()

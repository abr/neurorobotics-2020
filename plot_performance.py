from abr_analyze import DataHandler
import numpy as np
import matplotlib.pyplot as plt


def plot_performance(ax=None):
    test_group = "weighted_tests"
    main_db = "dewolf2018weight-tests"
    test_list = [
        ["pd_no_weight", "", "k", main_db, test_group, "-"],
        ["pd", "", "tab:grey", main_db, test_group, "-"],
        ["pid", "PID", "tab:brown", main_db, test_group, "-"],
        ["nengo_cpu1k", "Adaptive CPU", "g", main_db, test_group, "-"],
        ["nengo_gpu1k", "Adaptive GPU", "r", main_db, test_group, "-"],
        ["nengo_loihi1k", "Adaptive Loihi", "b", main_db, test_group, "-"],
    ]
    pd_dat = DataHandler(main_db)

    # load in PD mean performance with and without weight for normalizing
    pd_no_weight_mean = pd_dat.load(
        parameters=["mean"],
        save_location="%s/pd_no_weight/statistical_error_0" % (test_group),
    )["mean"]
    pd_weight_mean = pd_dat.load(
        parameters=["mean"], save_location="%s/pd/statistical_error_0/" % (test_group)
    )["mean"]

    n_sessions = 5
    n_runs = 50
    interpolated_samples = 400

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    for ii, test in enumerate(test_list):
        print("Processing test %i/%i: %s" % (ii + 1, len(test_list), test[0]))
        save_location = "%s/%s" % (test[4], test[0])
        errors = []
        for session in range(n_sessions):
            session_error = []
            for run in range(n_runs):
                print(
                    "%.3f processing complete..."
                    % (100 * ((run + 1) + (session * n_runs)) / (n_sessions * n_runs)),
                    end="\r",
                )
                loc = "%s/session%03d/run%03d" % (save_location, session, run)
                session_error.append(
                    np.mean(
                        pd_dat.load(parameters=["error"], save_location=loc)["error"]
                    )
                )
            errors.append(session_error)

        n = 3000  # use 3000 sample points
        p = 0.95  # calculate 95% confidence interval
        sample = []
        upper_bound = []
        lower_bound = []
        sets = np.array(errors).shape[0]
        data_pts = np.array(errors).shape[1]
        print(
            "Mean and CI calculation found %i sets of %i data points" % (sets, data_pts)
        )
        errors = np.array(errors)
        for i in range(data_pts):
            data = errors[:, i]
            index = int(n * (1 - p) / 2)
            samples = np.random.choice(data, size=(n, len(data)))
            r = [np.mean(s) for s in samples]
            r.sort()
            ci = r[index], r[-index]
            sample.append(np.mean(data))
            lower_bound.append(ci[0])
            upper_bound.append(ci[1])

        ci_errors = {
            "mean": sample,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
        ci_errors["time_derivative"] = 0

        data = pd_dat.load(
            parameters=["mean", "upper_bound", "lower_bound"],
            save_location="%s/statistical_error_0" % save_location,
        )

        # subtract the PD performance with no weight
        data["mean"] = data["mean"] - pd_no_weight_mean
        data["upper_bound"] = data["upper_bound"] - pd_no_weight_mean
        data["lower_bound"] = data["lower_bound"] - pd_no_weight_mean

        # normalize by the PD performance with a weight
        data["mean"] = data["mean"] / (pd_weight_mean - pd_no_weight_mean)
        data["upper_bound"] = data["upper_bound"] / (pd_weight_mean - pd_no_weight_mean)
        data["lower_bound"] = data["lower_bound"] / (pd_weight_mean - pd_no_weight_mean)

        ax.fill_between(
            range(np.array(data["mean"]).shape[0]),
            data["upper_bound"],
            data["lower_bound"],
            color=test[2],
            alpha=0.5,
        )
        ax.plot(data["mean"], color=test[2], label=test[1], linestyle=test[5])
        ax.set_title("Performance")

    ax.set_ylabel("Percent error")
    ax.set_xlabel("Trial")
    ax.legend()
    ax.grid()


if __name__ == "__main__":
    plot_performance()

    plt.tight_layout()
    loc = "Figures/performance.pdf"
    print("Figure saved to %s" % loc)
    plt.savefig(loc)
    plt.show()

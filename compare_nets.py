import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from abr_analyze import DataHandler

dat = DataHandler('rover_vis_comparison')
# tests = ['combined_net', 'split_net']
tests = [
        # 'combined_net_0004',
        # 'split_net_0004',
        'combined_net_1conv_layer',
        'combined_net_1conv_layer_spiking_with_synapses',
        # 'combined_net_loihi_0000',

        ]
params = ['target', 'motor', 'prediction']
steps = [1, 100, 100]

fig = plt.figure()
a = []
ls = ['-', '--']
cols = ['k', 'r']
col_cycle = [False, True]
for ii in range(len(params)):
    a.append(fig.add_subplot(len(params), 1, ii+1))

for jj, test in enumerate(tests):
    print(test)
    data = dat.load(params, test)

    for ii, param in enumerate(params):
        print(param)
        print(data[param].shape)
        a[ii].set_title(param)
        a[ii].plot(data[param], linestyle=ls[jj])#, color=cols[jj])
        # a[ii].set_prop_cycle(None)

custom_lines = [Line2D([0], [0], color=cols[0], linestyle=ls[0]),
                Line2D([0], [0], color=cols[0], linestyle=ls[1])]
a[0].legend(custom_lines, tests)
plt.show()

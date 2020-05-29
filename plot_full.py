import matplotlib.pyplot as plt

from plot_latency import plot_latency
from plot_performance import plot_performance
from plot_power import plot_power

fig = plt.figure(figsize=(8, 4))
ax = plt.subplot(1, 2, 1)
plot_performance(ax)

ax = plt.subplot(2, 2, 2)
plot_power(ax)

ax = plt.subplot(2, 2, 4)
plot_latency(ax)

plt.tight_layout()
loc = "Figures/adaptive_arm_results.pdf"
print("Figure saved to %s" % loc)
plt.savefig(loc)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

from abr_analyze import DataHandler

d1 = np.load('/home/tdewolf/Downloads/rover_training_0004_training_images_processed.npz')
d2 = np.load('/home/tdewolf/Downloads/abr_analyze_training_images_processed.npz')

n1 = len(d1['images'])
n2 = len(d2['images'])
n = min(n1, n2)

print(d1['images'].shape)

d1_targets_rounded = np.around(d1['targets'][:n], decimals=1)
d2_targets_rounded = np.around(d2['targets'][:n], decimals=1)

d1_indices = np.lexsort((d1_targets_rounded[:, 1], d1_targets_rounded[:, 0]))
d2_indices = np.lexsort((d2_targets_rounded[:, 1], d2_targets_rounded[:, 0]))

d1_targets = d1['targets'][d1_indices]
d2_targets = d2['targets'][d2_indices]

print(d1_targets)

# plot all the targets
plt.plot(d2['targets'][:, 0], d2['targets'][:, 1], 'rx')
plt.plot(d1['targets'][:, 0], d1['targets'][:, 1], 'bx', alpha=.5)
plt.gca().set_aspect('equal')
plt.savefig('plots/targets.png')
plt.clf()

print(d1_indices)

# plot all the images
res = [32, 128]
for ii, (d1_i, d2_i) in enumerate(zip(d1_indices, d2_indices)):

    # only plot if they're relatively close to each other
    dist1 = np.linalg.norm(d1_targets[ii])
    dist2 = np.linalg.norm(d2_targets[ii])
    if dist1 > 3.0 or dist2 > 3.0:  # arbitrary threshold

        a = plt.subplot(2, 1, 1)
        a.imshow(d1['images'][d1_i].reshape((res[0], res[1], 3)) / 255)
        plt.title('d1 target %i: %.3f, %.3f' % (ii, d1_targets[ii][0], d1_targets[ii][1]))

        b = plt.subplot(2, 1, 2)
        b.imshow(d2['images'][d2_i].reshape((res[0], res[1], 3)) / 255)
        plt.title('d2 target %i: %.3f, %.3f' % (ii, d2_targets[ii][0], d2_targets[ii][1]))

        plt.suptitle('dist = %.3f' % max(dist1, dist2))

        plt.savefig('plots/%i.png' % ii)
        plt.clf()

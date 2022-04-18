import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import psutil

from game_theoretic_clusterization import GameTheoreticClusterization


clust = GameTheoreticClusterization(image_path='./test_objects.png', rep_dyn_t_max=20, sigma=12)
clust.clusterization()
used_ram = psutil.Process().memory_info().rss / (1024 * 1024)
print(f'Memory used: {used_ram:.1f} MB')

matplotlib.use('TkAgg')

fig, ax = plt.subplots(1, 2)

plt.setp(ax, xticks=[], yticks=[])
ax[0].imshow(clust.image, cmap='gray')
ax[0].set_title(f'Original (height: {clust.image.shape[0]}px, width: {clust.image.shape[1]}px)')
ax[1].imshow(clust.final_seg, cmap='tab20b')
ax[1].set_title(f'Segmentation ({int(clust.final_seg.max())} clusters)')

plt.show()

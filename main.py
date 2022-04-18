import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from game_theoretic_clusterization import GameTheoreticClusterization


clust = GameTheoreticClusterization(image_path='./test_complex3.png', rep_dyn_t_max=3)
clust.clusterization()

matplotlib.use('TkAgg')

fig, ax = plt.subplots(1, 2)

plt.setp(ax, xticks=[], yticks=[])
ax[0].imshow(clust.image, cmap='gray')
ax[0].set_title(f'Original (height: {clust.image.shape[0]}px, width: {clust.image.shape[1]}px)')
ax[1].imshow(clust.final_seg, cmap='gray')
ax[1].set_title('Segmentation')

plt.show()

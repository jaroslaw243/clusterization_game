from matplotlib import pyplot as plt
import matplotlib
import time
import cv2

from game_theoretic_clusterization import GameTheoreticClusterization


clust = GameTheoreticClusterization(image_path='./test_objects512.png', rep_dyn_t_max=20, sigma=12, sigma_dist=2000,
                                    cluster_size_thresh_perc=0.0025, use_measure_memory_usage=True, max_iter=100)

scale = 100/clust.image.shape[1]
new_dim = (int(scale*clust.image.shape[1]), int(scale*clust.image.shape[0]))
clust.image = cv2.resize(clust.image, new_dim)

start_time = time.time()
clust.clusterization()
final_time = time.time() - start_time

if clust.use_measure_memory_usage:
    sparse_dense_perc = (clust.sim_matrix_size_in_memory / clust.sim_matrix_size_in_memory_if_dense) * 100

    print(f'Memory used: {clust.all_memory_used:.1f} MB')
    print(f'Similarity matrix size: {clust.sim_matrix_size_in_memory:.1f} MB')
    print(f'Similarity matrix size if it was dense: {clust.sim_matrix_size_in_memory_if_dense:.1f} MB')
    print(f'Sparse matrix size in memory is {sparse_dense_perc:.1f} % of the equivalent dense matrix')

print(f'Elapsed time: {final_time:.2f} s')

matplotlib.use('TkAgg')

fig, ax = plt.subplots(1, 2)

plt.setp(ax, xticks=[], yticks=[])
ax[0].imshow(clust.image, cmap='gray')
ax[0].set_title(f'Original (height: {clust.image.shape[0]}px, width: {clust.image.shape[1]}px)')
ax[1].imshow(clust.final_seg, cmap='tab20b')
title_vals = (clust.final_cluster_count, clust.large_cluster_elements_thresh, clust.sigma, clust.sigma_dist)
ax[1].set_title(r'Segmentation (%d clusters, large cluster$ \geq %d \mathrm{ \ el},'
                r' \sigma_{int} = %d, \sigma_{dist} = %d$)' % title_vals)

plt.show()

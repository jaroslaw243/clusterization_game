from matplotlib import pyplot as plt
import matplotlib
import time
import cv2

from game_theoretic_clusterization import GameTheoreticClusterization


clust = GameTheoreticClusterization(image_path='./cameraman.tif', rep_dyn_t_max=20, sigma=12000, sigma_dist=50000,
                                    cluster_size_thresh_perc=0.01, use_measure_memory_usage=True, max_iter=100)

scale = 75/clust.image.shape[1]
new_dim = (int(scale*clust.image.shape[1]), int(scale*clust.image.shape[0]))
clust.image = cv2.resize(clust.image, new_dim)

start_time = time.time()
clust.clusterization()
final_time = time.time() - start_time

sparse_dense_perc = (clust.sim_matrix_size_in_memory / clust.sim_matrix_size_in_memory_if_dense) * 100

if clust.use_measure_memory_usage:
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
ax[1].set_title(r'Segmentation (%d clusters, $\sigma_{int} = %d, \sigma_{dist} = %d$)' % (clust.final_cluster_count,
                                                                                          clust.sigma, clust.sigma_dist))

plt.show()

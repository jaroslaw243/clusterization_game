from matplotlib import pyplot as plt
import time
import argparse
import yaml
from game_theoretic_clusterization import GameTheoreticClusterization

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="yaml file with GameTheoreticClusterization object", required=True)
parser.add_argument("-fn", "--figures_name", help="name used for output image with figures",
                    required=True)
args = parser.parse_args()


with open(args.input) as file:
    clust = yaml.load(file, Loader=yaml.Loader)

clust.load_image()

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

dpi_for_fig = 150

fig, ax = plt.subplots(1, 2, figsize=(round(1920/dpi_for_fig), round(1080/dpi_for_fig)), dpi=dpi_for_fig)

plt.setp(ax, xticks=[], yticks=[])
ax[0].imshow(clust.image, cmap='gray')
ax[0].set_title(f'Original (height: {clust.image.shape[0]}px, width: {clust.image.shape[1]}px)')
ax[1].imshow(clust.final_seg, cmap='tab20b')
ax[1].set_title(r'Segmentation (%d clusters, $\sigma_{int} = %d, \sigma_{dist} = %d$)' % (clust.final_cluster_count,
                                                                                          clust.sigma, clust.sigma_dist))

plt.savefig(args.figures_name, bbox_inches='tight', dpi=dpi_for_fig)

import yaml
from game_theoretic_clusterization import GameTheoreticClusterization

clust = GameTheoreticClusterization(image_path='./test_objects.png', rep_dyn_t_max=20, sigma=12, sigma_dist=2000,
                                    cluster_size_thresh_perc=0.005, use_measure_memory_usage=True, max_iter=100,
                                    load_image_at_start=False)

with open(r'.\clust.yaml', 'w') as file:
    documents = yaml.dump(clust, file)

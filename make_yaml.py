import yaml
from game_theoretic_clusterization import GameTheoreticClusterization

clust = GameTheoreticClusterization(image_path='./test_objects.png', rep_dyn_t_max=20, sigma=12,
                                    use_measure_memory_usage=True, load_image_at_start=False)

with open(r'.\clust.yaml', 'w') as file:
    documents = yaml.dump(clust, file)

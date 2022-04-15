import cv2
import numpy as np

from game_theoretic_clusterization import GameTheoreticClusterization


clust = GameTheoreticClusterization(image_path='./test_complex3.png', rep_dyn_t_max=50)
clust.generate_similarity_matrix()

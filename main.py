import cv2
import numpy as np

from game_theoretic_clusterization import GameTheoreticClusterization


clust = GameTheoreticClusterization(image_path='./test_complex3.png')
clust.generate_similarity_matrix()

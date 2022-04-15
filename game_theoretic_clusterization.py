import numpy as np
from scipy.sparse import lil_array
import cv2


class GameTheoreticClusterization:
    def __init__(self, image_path, rep_dyn_t_max, sigma=1, load_image_at_start=True):
        self.image_path = image_path
        self.image = None
        self.sigma = np.float64(sigma)
        self.sim_matrix = None
        self.rep_dyn_t_max = rep_dyn_t_max

        if load_image_at_start:
            self.load_image()

    def load_image(self):
        self.image = np.array(cv2.imread(self.image_path, 0), dtype=np.float64)

    def generate_similarity_matrix(self):
        sim_matrix = lil_array((self.image.shape[0] * self.image.shape[1], self.image.shape[0] * self.image.shape[1]),
                               dtype=np.float64)

        k = 0
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                result = self.image[i, j] - self.image
                result = np.exp((np.square(result) * (-1))/(np.square(self.sigma)))
                result = np.reshape(result, (1, self.image.shape[0] * self.image.shape[1]))
                result = lil_array(result)
                sim_matrix[k, :] = result
                k += 1

        self.sim_matrix = sim_matrix.transpose()

    def discrete_replicator_dynamics(self):
        pass

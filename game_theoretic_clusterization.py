import numpy as np
from scipy.sparse import lil_array
import cv2
from utils import actualsize
import psutil


class GameTheoreticClusterization:
    def __init__(self, image_path, rep_dyn_t_max, remove_small_clust_att=10, cluster_size_thresh_perc=0.01, sigma=1,
                 sigma_dist=1, max_iter=100, use_measure_memory_usage=False, load_image_at_start=True):
        self.image_path = image_path
        self.image = None
        self.sigma = np.float64(sigma)
        self.sigma_dist = np.float64(sigma_dist)
        self.sim_matrix = None
        self.rep_dyn_t_max = rep_dyn_t_max
        self.indices_vec = None
        self.prob_in_time = None
        self.max_iter = max_iter
        self.final_seg = None
        self.org_image_dtype = None
        self.cluster_size_thresh_perc = cluster_size_thresh_perc
        self.remove_small_clust_att = remove_small_clust_att
        self.sim_matrix_size_in_memory = None
        self.sim_matrix_size_in_memory_if_dense = None
        self.all_memory_used = None
        self.use_measure_memory_usage = use_measure_memory_usage

        if load_image_at_start:
            self.load_image()

    def measure_memory_usage(self):
        div_val = 1024 * 1024
        self.sim_matrix_size_in_memory = actualsize(self.sim_matrix) / div_val
        self.sim_matrix_size_in_memory_if_dense = (((self.image.shape[0] * self.image.shape[1]) ** 2) * 8) / div_val
        self.all_memory_used = psutil.Process().memory_info().rss / div_val

    def load_image(self):
        image = cv2.imread(self.image_path, 0)
        self.org_image_dtype = image.dtype
        self.image = np.array(image, dtype=np.float64)

    def generate_similarity_matrix(self):
        sim_matrix = lil_array((self.image.shape[0] * self.image.shape[1], self.image.shape[0] * self.image.shape[1]),
                               dtype=np.float64)

        ind_mat = np.indices(self.image.shape, dtype=np.float64)
        max_image_val = np.float64(np.iinfo(self.org_image_dtype).max)
        max_dist = np.float64(np.sqrt(np.square(self.image.shape[0] - 1) + np.square(self.image.shape[1] - 1)))
        k = 0
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                intensity_term = self.image[i, j] - self.image
                intensity_term = np.exp((np.square(intensity_term) * (-1)) / (np.square(self.sigma)))

                distance_term = np.sqrt(np.square(i - ind_mat[0]) + np.square(j - ind_mat[1]))
                distance_term = (distance_term / max_dist) * max_image_val
                distance_term = np.exp((np.square(distance_term) * (-1)) / (np.square(self.sigma_dist)))

                combined_term = intensity_term * distance_term

                combined_term = np.reshape(combined_term, (1, self.image.shape[0] * self.image.shape[1]))
                combined_term = lil_array(combined_term)
                sim_matrix[k, :] = combined_term
                k += 1

        self.sim_matrix = sim_matrix.transpose()

    def discrete_replicator_dynamics(self, first_prob_vec):
        prob_in_time = lil_array((self.rep_dyn_t_max, first_prob_vec.shape[1]), dtype=np.float64)
        prob_in_time[0, :] = lil_array(first_prob_vec)

        for i in range(self.rep_dyn_t_max - 1):
            a = self.sim_matrix @ prob_in_time[[i], :].transpose()
            b = prob_in_time[[i], :] @ a
            temp_div = a / b.toarray()
            prob_in_time[[i + 1], :] = prob_in_time[[i], :] * temp_div.transpose()

        self.prob_in_time = prob_in_time

    def clusterization(self):
        q = self.image.shape[0] * self.image.shape[1]
        image_vec = np.reshape(np.transpose(self.image), (1, q))
        indices_vec = np.arange(stop=q, dtype=int)

        self.generate_similarity_matrix()

        if self.use_measure_memory_usage:
            self.measure_memory_usage()

        seg = np.zeros((1, q), dtype=np.uint8)
        iter_n = 1
        curr_label = 1
        all_pixels_labeled = False
        while not all_pixels_labeled and iter_n <= self.max_iter:
            q = image_vec.shape[0] * image_vec.shape[1]
            a_init = np.random.uniform(size=(1, q))
            a_init = a_init / np.sum(a_init)

            self.discrete_replicator_dynamics(a_init)

            prob_increase_ind = self.prob_in_time[[1], :] < self.prob_in_time[[-1], :]
            prob_increase_ind = prob_increase_ind.toarray()
            prob_increase_ind.shape = q
            prob_increase_ind_for_rm = np.logical_not(prob_increase_ind)
            prob_increase_ind_for_rm.shape = q

            seg[0, indices_vec[prob_increase_ind]] = curr_label

            indices_vec = indices_vec[prob_increase_ind_for_rm]
            image_vec = image_vec[0, prob_increase_ind_for_rm]
            image_vec = np.reshape(image_vec, (1, -1))

            self.sim_matrix = self.sim_matrix[:, prob_increase_ind_for_rm]
            self.sim_matrix = self.sim_matrix[prob_increase_ind_for_rm, :]

            if image_vec.size == 0:
                all_pixels_labeled = True

            iter_n += 1
            curr_label += 1

        self.final_seg = np.reshape(seg, self.image.shape)
        self.merge_small_clusters()

    def merge_small_clusters(self):
        cluster_kinds, cluster_sizes = np.unique(self.final_seg, return_counts=True)
        cluster_size_thresh = self.cluster_size_thresh_perc * self.final_seg.size
        small_clusters = cluster_kinds[cluster_sizes < cluster_size_thresh]
        large_clusters = cluster_kinds[cluster_sizes >= cluster_size_thresh]
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

        avg_colours = []
        for l_cluster in large_clusters:
            avg_colours.append(np.mean(self.image[self.final_seg == l_cluster], dtype=np.float64))
        avg_colours = np.array(avg_colours, dtype=np.float64)

        for s_cluster in small_clusters:
            cluster_seg = np.array(self.final_seg == s_cluster, dtype=np.uint8)
            cluster_seg_dil = cv2.dilate(cluster_seg, kernel, iterations=1)
            cluster_neighborhood = cluster_seg_dil
            cluster_neighborhood[cluster_seg == cluster_neighborhood] = 0
            neighbors = np.unique(self.final_seg[cluster_neighborhood])
            large_neighbors, large_neighbors_inds1, large_neighbors_inds2 = np.intersect1d(neighbors, large_clusters,
                                                                                           return_indices=True)

            mean_colour = np.mean(self.image[self.final_seg == s_cluster], dtype=np.float64)
            mean_colour_diff = np.abs(avg_colours[large_neighbors_inds2] - mean_colour)
            closest_cluster_ind = np.argmin(mean_colour_diff)
            closest_cluster = large_neighbors[closest_cluster_ind]
            self.final_seg[self.final_seg == s_cluster] = closest_cluster

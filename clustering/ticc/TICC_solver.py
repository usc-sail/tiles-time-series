import numpy as np
import math, time, collections, os, errno, sys, code, random
import matplotlib

import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.cluster import KMeans
import copy
import pandas as pd
from multiprocessing import Pool

from src.TICC_helper import *
from src.admm_solver import ADMMSolver


class TICC:
    
    def __init__(self, data_config, times_series_arr=None, window_size=10, number_of_clusters=5,
                 lambda_parameter=11e-2, beta=400, maxIters=1000, threshold=2e-5,
                 write_out_file=False, prefix_string="", num_proc=1,
                 compute_BIC=False, cluster_reassignment=20, participant_id=None):
        """
        Parameters:
            - window_size: size of the sliding window
            - number_of_clusters: number of clusters
            - lambda_parameter: sparsity parameter
            - switch_penalty: temporal consistency parameter
            - maxIters: number of iterations
            - threshold: convergence threshold
            - write_out_file: (bool) if true, prefix_string is output file dir
            - prefix_string: output directory if necessary
            - cluster_reassignment: number of points to reassign to a 0 cluster
        """

        self.times_series_arr = np.array(times_series_arr)
        self.window_size = window_size
        self.number_of_clusters = number_of_clusters
        self.lambda_parameter = lambda_parameter
        self.switch_penalty = beta
        self.maxIters = maxIters
        self.threshold = threshold
        self.write_out_file = write_out_file
        self.prefix_string = prefix_string
        self.num_proc = num_proc
        self.compute_BIC = compute_BIC
        self.cluster_reassignment = cluster_reassignment
        self.num_blocks = self.window_size + 1
        self.data_config = data_config
        self.participant_id = participant_id
        
        self.trained_model = {}

        pd.set_option('display.max_columns', 500)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        np.random.seed(102)

        ###########################################################
        # Save model paths
        ###########################################################
        self.save_TICC_model_path = os.path.join(self.data_config.fitbit_sensor_dict['clustering_path'], 'model')
        self.create_folder(self.save_TICC_model_path)
        self.save_TICC_model_path = os.path.join(self.save_TICC_model_path, participant_id)
        self.create_folder(self.save_TICC_model_path)

    def fit(self, data_df=None, mean=None, std=None):
        """
        Main method for ticc solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()
        
        self.complete_D_train = None
        last_end_index = 0
        
        if len(data_df) == 0:
            return
        
        data_df = data_df.fillna(data_df.mean())
        self.time_index = list(data_df.index)
        self.data_df = data_df
        
        ###########################################################
        # 1.1 Find the row and col size of the input
        ###########################################################
        times_series_arr = np.array((data_df - mean) / std)
        time_series_rows_size = times_series_arr.shape[0]
        time_series_col_size = times_series_arr.shape[1]

        ###########################################################
        # 1.2 Train test split
        ###########################################################
        training_indices = getTrainTestSplit(time_series_rows_size, self.num_blocks, self.window_size)  # indices of the training samples
        num_train_points = len(training_indices)

        ###########################################################
        # 1.3 Stack the training data
        ###########################################################
        complete_D_train = self.stack_training_data(times_series_arr, time_series_col_size, num_train_points, training_indices)
        
        if self.complete_D_train is None:
            self.complete_D_train = complete_D_train
        else:
            self.complete_D_train = np.append(self.complete_D_train, complete_D_train, axis=0)

        ###########################################################
        # 1.4 Initiate using Gaussian Mixture
        ###########################################################
        gmm = mixture.GaussianMixture(n_components=self.number_of_clusters, covariance_type="full", max_iter=300)
        gmm.fit(self.complete_D_train)
        clustered_points = gmm.predict(self.complete_D_train)

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance, empirical_covariances = {}, {}
        cluster_mean_info, cluster_mean_stacked_info = {}, {}
        old_clustered_points = None  # points from last iteration

        ###########################################################
        # 2. PERFORM TRAINING ITERATIONS
        ###########################################################
        pool = Pool(processes=self.num_proc)  # multi-threading

        save_cluster_path = os.path.join(self.data_config.fitbit_sensor_dict['clustering_path'], self.participant_id + '.csv.gz')

        for iters in range(self.maxIters):
            print("\n\n\nITERATION ###", iters)
    
            ###########################################################
            # 2.1 Get the train and test points
            ###########################################################
            train_clusters_arr = collections.defaultdict(list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)
    
            len_train_clusters = {k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}
    
            ###########################################################
            # Train_clusters holds the indices in complete_D_train
            # for each of the clusters
            ###########################################################
            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, self.complete_D_train,
                                          empirical_covariances, len_train_clusters, time_series_col_size,
                                          pool, train_clusters_arr)
    
            ###########################################################
            # 2.2 Optimize using ADMM
            ###########################################################
            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res, train_cluster_inverse)
            self.train_cluster_inverse = train_cluster_inverse

            ###########################################################
            # 2.3 Update old computed covariance
            ###########################################################
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': self.complete_D_train,
                                  'time_series_col_size': time_series_col_size}

            clustered_points = self.predict_clusters()

            ###########################################################
            # 2.4 Recalculate lengths
            ###########################################################
            new_train_clusters = collections.defaultdict(list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            ###########################################################
            # 2.5 Handle cluster with 0 number
            ###########################################################
            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
    
                # clusters that are not 0 as sorted by norm
                valid_clusters = [cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]
    
                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
    
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        cluster_selected = valid_clusters[counter]  # a cluster that is not len 0
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num, "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = self.complete_D_train[point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] = self.complete_D_train[point_to_move, :][(self.window_size - 1) * time_series_col_size:self.window_size * time_series_col_size]
    
                for cluster_num in range(self.number_of_clusters):
                    print("length of cluster #", cluster_num, "-------->", sum([x == cluster_num for x in clustered_points]))
    
                str_NULL = 'output'
                if os.path.exists(str_NULL) is False:
                    os.mkdir(str_NULL)

            print("\n\n\n")

            ###########################################################
            # 2.6 Save model parameters
            ###########################################################
            self.save_model_parameters()

            ###########################################################
            # 2.7 If converge
            ###########################################################
            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                self.data_df.to_csv(save_cluster_path, compression='gzip')
                break
            old_clustered_points = before_empty_cluster_assign

        if pool is not None:
            pool.close()
            pool.join()

        if self.compute_BIC:
            bic = computeBIC(self.number_of_clusters, self.complete_D_train.shape[0], clustered_points, train_cluster_inverse, empirical_covariances)
            return clustered_points, train_cluster_inverse, bic

        self.data_df.to_csv(save_cluster_path, compression='gzip')

        return clustered_points, train_cluster_inverse

    def write_plot(self, clustered_points, str_NULL, training_indices):
        # Save a figure of segmentation
        plt.figure()
        plt.plot(training_indices[0:len(clustered_points)], clustered_points, color="r")  # ,marker = ".",s =100)
        plt.ylim((-0.5, self.number_of_clusters + 0.5))
        if self.write_out_file: plt.savefig(str_NULL + "TRAINING_EM_lam_sparse=" + str(self.lambda_parameter) + "switch_penalty = " + str(self.switch_penalty) + ".jpg")
        plt.close("all")
        print("Done writing the figure")

    def optimize_clusters(self, computed_covariance, len_train_clusters, log_det_values, optRes, train_cluster_inverse):
        for cluster in range(self.number_of_clusters):
            if optRes[cluster] == None:
                continue

            val = optRes[cluster].get()
            print("OPTIMIZATION for Cluster #", cluster, "DONE!!!")

            # THIS IS THE SOLUTION
            S_est = upperToFull(val, 0)
            # This is the inverse covariance matrix
            X2 = S_est
            u, _ = np.linalg.eig(S_est)
            cov_out = np.linalg.inv(X2)
            
            # Store the log-det, covariance, inverse-covariance, cluster means, stacked means
            log_det_values[self.number_of_clusters, cluster] = np.log(np.linalg.det(cov_out))
            computed_covariance[self.number_of_clusters, cluster] = cov_out
            train_cluster_inverse[cluster] = X2
        
        for cluster in range(self.number_of_clusters):
            print("length of the cluster ", cluster, "------>", len_train_clusters[cluster])
        
    def train_clusters(self, cluster_mean_info, cluster_mean_stacked_info, complete_D_train, empirical_covariances, len_train_clusters, n, pool, train_clusters_arr):
    
        optRes = [None for i in range(self.number_of_clusters)]

        for cluster in range(self.number_of_clusters):
            cluster_length = len_train_clusters[cluster]
            
            if cluster_length != 0:
                size_blocks = n
                indices = train_clusters_arr[cluster]

                D_train = np.zeros([cluster_length, self.window_size * n])
                for i in range(cluster_length):
                    point = indices[i]
                    D_train[i, :] = complete_D_train[point, :]

                cluster_mean_info[self.number_of_clusters, cluster] = np.mean(D_train, axis=0)[(self.window_size - 1) * n:self.window_size * n].reshape([1, n])
                cluster_mean_stacked_info[self.number_of_clusters, cluster] = np.mean(D_train, axis=0)

                ##Fit a models - OPTIMIZATION
                probSize = self.window_size * size_blocks
                lamb = np.zeros((probSize, probSize)) + self.lambda_parameter
                S = np.cov(np.transpose(D_train))

                empirical_covariances[cluster] = S
                rho = 1
                
                solver = ADMMSolver(lamb, self.window_size, size_blocks, 1, S)
                # apply to process pool
                optRes[cluster] = pool.apply_async(solver, (1000, 1e-6, 1e-6, False,))
                
        return optRes
    
    def stack_training_data(self, Data, n, num_train_points, training_indices):
        complete_D_train = np.zeros([num_train_points, self.window_size * n])
        for i in range(num_train_points):
            for k in range(self.window_size):
                if i + k < num_train_points:
                    idx_k = training_indices[i + k]
                    complete_D_train[i][k * n:(k + 1) * n] = Data[idx_k][0:n]
        return complete_D_train
    
    def log_parameters(self):
        print("lam_sparse", self.lambda_parameter)
        print("switch_penalty", self.switch_penalty)
        print("num_cluster", self.number_of_clusters)
        print("num stacked", self.window_size)

    def smoothen_clusters(self, cluster_mean_info, computed_covariance,
                          cluster_mean_stacked_info, complete_D_train, n):
        clustered_points_len = len(complete_D_train)
        inv_cov_dict = {}  # cluster to inv_cov
        log_det_dict = {}  # cluster to log_det
        
        for cluster in range(self.number_of_clusters):
            cov_matrix = computed_covariance[self.number_of_clusters, cluster][0:(self.num_blocks - 1) * n, 0:(self.num_blocks - 1) * n]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov

        # For each point compute the LLE
        # print("beginning the smoothening ALGORITHM")
        LLE_all_points_clusters = np.zeros([clustered_points_len, self.number_of_clusters])
        for point in range(clustered_points_len):
            if point + self.window_size - 1 < complete_D_train.shape[0]:
                for cluster in range(self.number_of_clusters):
                    cluster_mean_stacked = cluster_mean_stacked_info[self.number_of_clusters, cluster]
                    x = complete_D_train[point, :] - cluster_mean_stacked[0:(self.num_blocks - 1) * n]
                    inv_cov_matrix = inv_cov_dict[cluster]
                    log_det_cov = log_det_dict[cluster]
                    lle = np.dot(x.reshape([1, (self.num_blocks - 1) * n]), np.dot(inv_cov_matrix, x.reshape([n * (self.num_blocks - 1), 1]))) + log_det_cov
                    LLE_all_points_clusters[point, cluster] = lle

        return LLE_all_points_clusters
        
    def predict_clusters(self, test_data=None):
        '''
        Given the current trained models, predict clusters.  If the cluster segmentation has not been optimized yet,
        than this will be part of the interative process.

        Args:
            numpy array of data for which to predict clusters.  Columns are dimensions of the data, each row is
            a different timestamp

        Returns:
            vector of predicted cluster for the points
        '''
        if test_data is not None:
            if not isinstance(test_data, np.ndarray):
                raise TypeError("input must be a numpy array!")
        else:
            test_data = self.trained_model['complete_D_train']

        # SMOOTHENING
        clustered_points = None

        save_cluster_path = os.path.join(self.data_config.fitbit_sensor_dict['clustering_path'], self.participant_id + '.csv.gz')
        
        data_per_segment = test_data.copy()
        lle_all_points_clusters = self.smoothen_clusters(self.trained_model['cluster_mean_info'],
                                                         self.trained_model['computed_covariance'],
                                                         self.trained_model['cluster_mean_stacked_info'],
                                                         data_per_segment, self.trained_model['time_series_col_size'])

        # Update cluster points - using NEW smoothening
        if clustered_points is None:
            clustered_points = updateClusters(lle_all_points_clusters, switch_penalty=self.switch_penalty)
    
            clustered_points_df = pd.DataFrame(clustered_points, index=self.time_index[:len(clustered_points)])
            # clustered_points_df.to_csv(save_cluster_path, compression='gzip')
            self.data_df.loc[self.time_index[:len(clustered_points)], 'cluster'] = clustered_points
            # self.data_df.to_csv(save_cluster_path, compression='gzip')
        else:
            clustered_points_per_segments = updateClusters(lle_all_points_clusters, switch_penalty=self.switch_penalty)
            clustered_points = np.append(clustered_points, clustered_points_per_segments)
    
            clustered_points_df = pd.DataFrame(clustered_points_per_segments)
            clustered_points_df.to_csv(save_cluster_path, compression='gzip')
            self.data_df.loc[self.time_index[:len(clustered_points)], 'cluster'] = clustered_points
            # self.data_df.to_csv(save_cluster_path, compression='gzip')
            
        return (clustered_points)
        
    def save_model_parameters(self):
        ###########################################################
        # Save model parameters
        ###########################################################
        np.save(os.path.join(self.save_TICC_model_path, 'cluster_mean_info.npy'), self.trained_model['cluster_mean_info'])
        np.save(os.path.join(self.save_TICC_model_path, 'computed_covariance.npy'), self.trained_model['computed_covariance'])
        np.save(os.path.join(self.save_TICC_model_path, 'cluster_mean_stacked_info.npy'), self.trained_model['cluster_mean_stacked_info'])
        np.save(os.path.join(self.save_TICC_model_path, 'train_cluster_inverse.npy'), self.train_cluster_inverse)

    def load_model_parameters(self):
        ###########################################################
        # Load model parameters
        ###########################################################
        self.trained_model['cluster_mean_info'] = np.load(os.path.join(self.save_TICC_model_path, 'cluster_mean_info.npy')).item()
        self.trained_model['computed_covariance'] = np.load(os.path.join(self.save_TICC_model_path, 'computed_covariance.npy')).item()
        self.trained_model['cluster_mean_stacked_info'] = np.load(os.path.join(self.save_TICC_model_path, 'cluster_mean_stacked_info.npy')).item()
        self.train_cluster_inverse = np.load(os.path.join(self.save_TICC_model_path, 'train_cluster_inverse.npy')).item()
        
        print('Successfully load the model!')

    def create_folder(self, folder):
        if os.path.exists(folder) is False:
            os.mkdir(folder)
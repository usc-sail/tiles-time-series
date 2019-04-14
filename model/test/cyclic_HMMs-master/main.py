import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import random
from constants_and_util import *
from test_on_simulated_data import *
import cyclic_HMM

np.random.seed(42)
random.seed(42)
confirm_results_do_not_change = True # make sure that tinkering with the model didn't change any fitting parameters: useful for debugging and unit tests.

if __name__ == '__main__':
    all_param_settings = generate_params_for_all_tests()
    param_setting = all_param_settings[0]
    print('The simulated data parameter setting is', param_setting.__dict__)
    data_generator = simulatedDataGenerator(param_setting)
    data_generator.generate_simulated_data()
    samples = [a['sample'] for a in data_generator.all_individual_data]

    
    model = cyclic_HMM.fit_cyhmm_model(n_states=4,
                                       samples=samples,
                                       symptom_names=data_generator.symptoms,
                                       max_iterations=100,
                                       duration_distribution_name='poisson',
                                       emission_distribution_name='normal_with_missing_data',
                                       hypothesized_duration=20,
                                       verbose=True,
                                       n_processes=1,
                                       min_iterations=10)
    
    
    cluster_model = cyclic_HMM.fit_clustering_model(n_states=4,
                                                    n_clusters=2,
                                                    samples=samples,
                                                    symptom_names=data_generator.symptoms,
                                                    duration_distribution_name='poisson',
                                                    emission_distribution_name='normal_with_missing_data',
                                                    hypothesized_duration=20,
                                                    n_samples_to_use_in_clustering=50,
                                                    max_clustering_iterations=20)










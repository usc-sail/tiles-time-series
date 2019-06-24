import os


def create_folder(folder_dir):
	if os.path.exists(folder_dir) is False:
		os.mkdir(folder_dir)


def load_preprocess_path(data_config, process_data_path, data_name='preprocess'):
	""" Load the preprocess path to data_config

	Params:
	data_config - config setting
	process_data_path - the path contains all processed stream
	data_name - data name

	Returns:

	"""
	tmp_path = os.path.join(process_data_path, data_name)
	create_folder(tmp_path)

	# om signal
	tmp_path = os.path.join(tmp_path, data_config.omsignal_sensor_dict['name'])
	create_folder(tmp_path)
	preprocess_str = data_config.omsignal_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.omsignal_sensor_dict['preprocess_setting']
	preprocess_str = preprocess_str + '_' + data_config.omsignal_sensor_dict['preprocess_cols']
	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.omsignal_sensor_dict['preprocess_path'] = tmp_path

	# Fitbit
	tmp_path = os.path.join(process_data_path, data_name, data_config.fitbit_sensor_dict['name'])
	create_folder(tmp_path)

	if data_config.fitbit_sensor_dict['imputation'] != '':
		preprocess_str = 'impute_' + data_config.fitbit_sensor_dict['imputation'] + '_' + str(data_config.fitbit_sensor_dict['imputation_threshold'])
	else:
		preprocess_str = data_config.fitbit_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_setting']
	preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_cols']
	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.fitbit_sensor_dict['preprocess_path'] = tmp_path

	# owl_in_one
	tmp_path = os.path.join(process_data_path, data_name, data_config.owl_in_one_sensor_dict['name'])
	create_folder(tmp_path)

	preprocess_str = data_config.owl_in_one_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.owl_in_one_sensor_dict['preprocess_setting']

	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.owl_in_one_sensor_dict['preprocess_path'] = tmp_path

	# realizd
	tmp_path = os.path.join(process_data_path, data_name, data_config.realizd_sensor_dict['name'])
	create_folder(tmp_path)
	preprocess_str = data_config.realizd_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.realizd_sensor_dict['preprocess_setting']
	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.realizd_sensor_dict['preprocess_path'] = tmp_path

	# audio
	tmp_path = os.path.join(process_data_path, data_name, data_config.audio_sensor_dict['name'])
	create_folder(tmp_path)
	preprocess_str = data_config.audio_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.audio_sensor_dict['preprocess_setting']
	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.audio_sensor_dict['preprocess_path'] = tmp_path


def load_plot_path(data_config, process_data_path, data_name='plot'):
	""" Load the plot path to data_config

	Params:
	data_config - config setting
	process_data_path - the path contains all processed stream
	data_name - data name

	Returns:

	"""
	tmp_path = os.path.join(process_data_path, data_name)
	create_folder(tmp_path)

	# om signal
	tmp_path = os.path.join(tmp_path, data_config.omsignal_sensor_dict['name'])
	create_folder(tmp_path)
	preprocess_str = data_config.omsignal_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.omsignal_sensor_dict['preprocess_setting']
	preprocess_str = preprocess_str + '_' + data_config.omsignal_sensor_dict['preprocess_cols']
	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.omsignal_sensor_dict['plot_path'] = tmp_path

	# Fitbit
	tmp_path = os.path.join(process_data_path, data_name, data_config.fitbit_sensor_dict['name'])
	create_folder(tmp_path)

	if data_config.fitbit_sensor_dict['imputation'] != None:
		preprocess_str = 'impute_' + data_config.fitbit_sensor_dict['imputation'] + '_' + str(data_config.fitbit_sensor_dict['imputation_threshold'])
	else:
		preprocess_str = data_config.fitbit_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_setting']
	preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_cols']
	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.fitbit_sensor_dict['plot_path'] = tmp_path

	# owl_in_one
	tmp_path = os.path.join(process_data_path, data_name, data_config.owl_in_one_sensor_dict['name'])
	create_folder(tmp_path)

	preprocess_str = data_config.owl_in_one_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.owl_in_one_sensor_dict['preprocess_setting']

	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.owl_in_one_sensor_dict['plot_path'] = tmp_path

	# realizd
	tmp_path = os.path.join(process_data_path, data_name, data_config.realizd_sensor_dict['name'])
	create_folder(tmp_path)
	preprocess_str = data_config.realizd_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.realizd_sensor_dict['preprocess_setting']
	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.realizd_sensor_dict['plot_path'] = tmp_path

	# audio
	tmp_path = os.path.join(process_data_path, data_name, data_config.audio_sensor_dict['name'])
	create_folder(tmp_path)
	preprocess_str = data_config.audio_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.audio_sensor_dict['preprocess_setting']
	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.audio_sensor_dict['plot_path'] = tmp_path

def load_filter_path(data_config, process_data_path, data_name='filter_data'):
	""" Load the filter data path to data_config.

	Params:
	data_config - config setting
	process_data_path - the path contains all processed stream
	data_name - data name

	Returns:

	"""
	tmp_path = os.path.join(process_data_path, data_name)
	create_folder(tmp_path)
	tmp_path = os.path.join(tmp_path, data_config.filter_method)
	create_folder(tmp_path)

	# om signal
	tmp_path = os.path.join(tmp_path, data_config.omsignal_sensor_dict['name'])
	create_folder(tmp_path)
	preprocess_str = data_config.omsignal_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.omsignal_sensor_dict['preprocess_setting']
	preprocess_str = preprocess_str + '_' + data_config.omsignal_sensor_dict['preprocess_cols']
	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.omsignal_sensor_dict['filter_path'] = tmp_path

	# Fitbit
	tmp_path = os.path.join(process_data_path, data_name, data_config.filter_method, data_config.fitbit_sensor_dict['name'])
	create_folder(tmp_path)

	if data_config.fitbit_sensor_dict['imputation'] != None:
		preprocess_str = 'impute_' + data_config.fitbit_sensor_dict['imputation'] + '_' + str(data_config.fitbit_sensor_dict['imputation_threshold'])
	else:
		preprocess_str = data_config.fitbit_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_setting']
	preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_cols']
	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.fitbit_sensor_dict['filter_path'] = tmp_path

	# owl_in_one
	tmp_path = os.path.join(process_data_path, data_name, data_config.filter_method, data_config.owl_in_one_sensor_dict['name'])
	create_folder(tmp_path)

	preprocess_str = data_config.owl_in_one_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.owl_in_one_sensor_dict['preprocess_setting']

	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.owl_in_one_sensor_dict['filter_path'] = tmp_path

	# realizd
	tmp_path = os.path.join(process_data_path, data_name, data_config.filter_method, data_config.realizd_sensor_dict['name'])
	create_folder(tmp_path)
	preprocess_str = data_config.realizd_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.realizd_sensor_dict['preprocess_setting']
	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.realizd_sensor_dict['filter_path'] = tmp_path

	# audio
	tmp_path = os.path.join(process_data_path, data_name, data_config.filter_method, data_config.audio_sensor_dict['name'])
	create_folder(tmp_path)
	preprocess_str = data_config.audio_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.audio_sensor_dict['preprocess_setting']
	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.audio_sensor_dict['filter_path'] = tmp_path


def load_segmentation_path(data_config, process_data_path, data_name='segmentation', filter_data=False):
	""" Load the segmentation data path to data_config.

	Params:
	data_config - config setting
	process_data_path - the path contains all processed stream
	data_name - data name

	Returns:

	"""
	tmp_path = os.path.join(process_data_path, data_config.experiement)
	create_folder(tmp_path)

	if filter_data == True:
		tmp_path = os.path.join(tmp_path, 'filter_data_' + data_config.filter_method)
	else:
		tmp_path = os.path.join(tmp_path, 'preprocess')
	create_folder(tmp_path)

	tmp_path = os.path.join(tmp_path, data_name)
	create_folder(tmp_path)

	# Fitbit segmentation
	tmp_path = os.path.join(tmp_path, data_config.fitbit_sensor_dict['name'])
	create_folder(tmp_path)

	if data_config.fitbit_sensor_dict['segmentation_method'] == 'gaussian':
		preprocess_str = 'ggs_' + str(data_config.fitbit_sensor_dict['segmentation_lamb'])
	else:
		return

	if data_config.fitbit_sensor_dict['imputation'] != None:
		preprocess_str = preprocess_str + '_impute_' + data_config.fitbit_sensor_dict['imputation'] + '_' + str(data_config.fitbit_sensor_dict['imputation_threshold'])
	else:
		preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_setting']
	preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_cols']
	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)

	data_config.fitbit_sensor_dict['segmentation_path'] = tmp_path


def load_fitbit_summary_path(tiles_data_path, data_name='3_preprocessed_data'):
	return os.path.join(tiles_data_path, data_name, 'fitbit')


def load_clustering_path(data_config, process_data_path, data_name='clustering', filter_data=False):
	""" Load the clustering data path to data_config.

	Params:
	data_config - config setting
	process_data_path - the path contains all processed stream
	data_name - data name

	Returns:

	"""
	tmp_path = os.path.join(process_data_path, data_config.experiement)
	create_folder(tmp_path)

	if filter_data == True:
		tmp_path = os.path.join(tmp_path, 'filter_data_' + data_config.filter_method)
	else:
		tmp_path = os.path.join(tmp_path, 'preprocess')
	create_folder(tmp_path)

	tmp_path = os.path.join(tmp_path, data_name)
	create_folder(tmp_path)

	# Fitbit clustering
	tmp_path = os.path.join(tmp_path, data_config.fitbit_sensor_dict['name'])
	create_folder(tmp_path)

	preprocess_str = data_config.fitbit_sensor_dict['cluster_method'] + '_num_cluster_' + str(data_config.fitbit_sensor_dict['num_cluster'])
	if data_config.fitbit_sensor_dict['cluster_method'] == 'ticc':
		preprocess_str = preprocess_str + '_window_' + str(data_config.fitbit_sensor_dict['ticc_window'])
		preprocess_str = preprocess_str + '_penalty_' + str(data_config.fitbit_sensor_dict['ticc_switch_penalty'])
		preprocess_str = preprocess_str + '_sparsity_' + str(data_config.fitbit_sensor_dict['ticc_sparsity'])
	if filter_data == True:
		preprocess_str = preprocess_str + '_cluster_days_' + str(data_config.fitbit_sensor_dict['ticc_cluster_days'])

	if data_config.fitbit_sensor_dict['segmentation_method'] == 'gaussian':
		preprocess_str = preprocess_str + '_ggs_' + str(data_config.fitbit_sensor_dict['segmentation_lamb'])

	if data_config.fitbit_sensor_dict['imputation'] != None:
		preprocess_str = preprocess_str + '_impute_' + data_config.fitbit_sensor_dict['imputation'] + '_' + str(data_config.fitbit_sensor_dict['imputation_threshold'])
	else:
		preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_setting']
	preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_cols']
	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)
	data_config.fitbit_sensor_dict['clustering_path'] = tmp_path

	# Raw audio clustering
	tmp_path = os.path.join(process_data_path, 'clustering')
	create_folder(tmp_path)
	if data_config.audio_sensor_dict['cluster_data'] == 'snippet':
		tp_path = os.path.join(tmp_path, 'tp_audio_' + data_config.audio_sensor_dict['audio_feature'])
		tmp_path = os.path.join(tmp_path, 'audio_' + data_config.audio_sensor_dict['audio_feature'])
	else:
		tp_path =  os.path.join(tmp_path, 'tp_audio_pause_threshold_' + str(data_config.audio_sensor_dict['pause_threshold']) + '_' + data_config.audio_sensor_dict['audio_feature'])
		tmp_path = os.path.join(tmp_path, 'audio_pause_threshold_' + str(data_config.audio_sensor_dict['pause_threshold']) + '_' + data_config.audio_sensor_dict['audio_feature'])
	create_folder(tmp_path)
	create_folder(tp_path)
	tmp_path = os.path.join(tmp_path, data_config.audio_sensor_dict['cluster_data'])
	tp_path = os.path.join(tp_path, data_config.audio_sensor_dict['cluster_data'])
	create_folder(tmp_path)
	create_folder(tp_path)
	
	if 'bgm' in data_config.audio_sensor_dict['cluster_method'] or 'kmeans' in data_config.audio_sensor_dict['cluster_method']:
		cluster_str = 'method_' + data_config.audio_sensor_dict['cluster_method']
	else:
		cluster_str = 'alpha_' + data_config.audio_sensor_dict['cluster_alpha'] + '_method_' + data_config.audio_sensor_dict['cluster_method']
	tmp_path = os.path.join(tmp_path, cluster_str)
	tp_path = os.path.join(tp_path, cluster_str)
	create_folder(tmp_path)
	create_folder(tp_path)

	data_config.audio_sensor_dict['clustering_path'] = tmp_path
	data_config.audio_sensor_dict['tp_path'] = tp_path

	# process cluster name
	if data_config.audio_sensor_dict['cluster_data'] == 'utterance':
		cluster_name = 'utterance_cluster'
	elif data_config.audio_sensor_dict['cluster_data'] == 'minute':
		cluster_name = 'minute_cluster'
	elif data_config.audio_sensor_dict['cluster_data'] == 'snippet':
		cluster_name = 'snippet_cluster'
	elif data_config.audio_sensor_dict['cluster_data'] == 'raw_audio':
		cluster_name = 'raw_audio_cluster'
	else:
		cluster_name = 'snippet_cluster'
	data_config.audio_sensor_dict['lda_clustering_path'] = 'lda_' + str(data_config.audio_sensor_dict['lda_num']) + '_' + cluster_name + '.csv.gz'

	data_config.audio_sensor_dict['lda_path'] = 'lda_' + str(data_config.audio_sensor_dict['lda_num']) + '.csv.gz'

	data_config.audio_sensor_dict['final_save_prefix'] = 'offset_' + str(data_config.audio_sensor_dict['cluster_offset']) + \
														 '_lda_' + str(data_config.audio_sensor_dict['lda_num']) + '_' + \
														 cluster_name + '_' + data_config.audio_sensor_dict['topic_method'] + '_' + \
														 data_config.audio_sensor_dict['topic_num']


def load_days_at_work_path(data_config, process_data_path):
	tmp_path = os.path.join(process_data_path, 'preprocess')
	create_folder(tmp_path)
	
	tmp_path = os.path.join(tmp_path, 'days_at_work')
	create_folder(tmp_path)
	
	data_config.days_at_work_path = tmp_path


def load_sleep_path(data_config, process_data_path, data_name='preprocess'):
	tmp_path = os.path.join(process_data_path, data_name, 'sleep')
	create_folder(tmp_path)
	
	if data_config.fitbit_sensor_dict['imputation'] != '':
		preprocess_str = 'impute_' + data_config.fitbit_sensor_dict['imputation'] + '_' + str(data_config.fitbit_sensor_dict['imputation_threshold'])
	else:
		preprocess_str = data_config.fitbit_sensor_dict['feature']
	preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_setting']
	preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_cols']
	tmp_path = os.path.join(tmp_path, preprocess_str)
	create_folder(tmp_path)
	
	data_config.sleep_path = tmp_path


def load_phone_usage_path(data_config, process_data_path, data_name='preprocess'):
	tmp_path = os.path.join(process_data_path, data_name, 'phone_usage')
	create_folder(tmp_path)
	
	data_config.phone_usage_path = tmp_path
	

def load_chi_preprocess_path(chi_data_config, process_data_path, experiment='chi'):
	if chi_data_config.fitbit_sensor_dict['imputation'] != '':
		preprocess_str = 'impute_' + chi_data_config.fitbit_sensor_dict['imputation'] + '_' + str(chi_data_config.fitbit_sensor_dict['imputation_threshold'])
	else:
		preprocess_str = chi_data_config.fitbit_sensor_dict['feature']
	
	tmp_path = os.path.join(process_data_path, experiment)
	create_folder(tmp_path)
	
	tmp_path = os.path.join(process_data_path, experiment, preprocess_str)
	create_folder(tmp_path)
	
	chi_config_str = 'num_of_point_' + str(chi_data_config.num_point_per_day) + '_window_' + str(chi_data_config.window)
	tmp_path = os.path.join(process_data_path, experiment, preprocess_str, chi_config_str)
	create_folder(tmp_path)
	
	chi_data_config.save_path =os.path.join(process_data_path, experiment, preprocess_str, chi_config_str)


def load_all_available_path(data_config, process_data_path, filter_data=False,
							preprocess_data_identifier='preprocess',
							segmentation_data_identifier='segmentation',
							filter_data_identifier='filter_data',
							plot_identifier='plot',
							clustering_data_identifier='clustering'):
	""" Load all available data path to data_config.

	Params:
	data_config - config setting
	process_data_path - the path contains all processed stream
	identifier - root path for different process scheme

	Returns:

	"""
	# Load preprocess folder
	load_preprocess_path(data_config, process_data_path, data_name=preprocess_data_identifier)

	# Load segmentation folder
	load_segmentation_path(data_config, process_data_path, data_name=segmentation_data_identifier, filter_data=filter_data)

	# Load filter folder
	load_filter_path(data_config, process_data_path, data_name=filter_data_identifier)

	# Load clustering folder
	load_clustering_path(data_config, process_data_path, data_name=clustering_data_identifier, filter_data=filter_data)

	# Load plot folder
	load_plot_path(data_config, process_data_path, data_name=plot_identifier)
	
	# Load days at work folder
	load_days_at_work_path(data_config, process_data_path)
	
	# Load sleep folder
	load_sleep_path(data_config, process_data_path)
	
	# Load daily phone usage folder
	load_phone_usage_path(data_config, process_data_path)


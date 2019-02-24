import os


def create_folder(folder_dir):
    if os.path.exists(folder_dir) is False:
        os.mkdir(folder_dir)


def load_preprocess_path(data_config, process_data_path, data_name='preprocess_data'):
    """ Produces a fixed-length summary vector of the sequence for each segment.

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
    
    if data_config.fitbit_sensor_dict['imputation'] != None:
        preprocess_str = 'impute_' + data_config.fitbit_sensor_dict['imputation']
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


def load_segmentation_path(data_config, process_data_path, data_name='segmentation'):
    """ Produces a fixed-length summary vector of the sequence for each segment.

    Params:
    data_config - config setting
    process_data_path - the path contains all processed stream
    data_name - data name

    Returns:

    """
    tmp_path = os.path.join(process_data_path, data_config.experiement)
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
        preprocess_str = preprocess_str + '_impute_' + data_config.fitbit_sensor_dict['imputation']
    else:
        preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['feature']
    preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_setting']
    preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_cols']
    tmp_path = os.path.join(tmp_path, preprocess_str)
    create_folder(tmp_path)
    
    data_config.fitbit_sensor_dict['segmentation_path'] = tmp_path
    

def load_fitbit_summary_path(tiles_data_path, data_name='3_preprocessed_data'):
    return os.path.join(tiles_data_path, data_name, 'fitbit')


def load_clustering_path(data_config, process_data_path, data_name='clustering'):
    """ Produces a fixed-length summary vector of the sequence for each segment.

    Params:
    data_config - config setting
    process_data_path - the path contains all processed stream
    data_name - data name

    Returns:

    """
    tmp_path = os.path.join(process_data_path, data_config.experiement)
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
    
    if data_config.fitbit_sensor_dict['segmentation_method'] == 'gaussian':
        preprocess_str = preprocess_str + '_ggs_' + str(data_config.fitbit_sensor_dict['segmentation_lamb'])

    if data_config.fitbit_sensor_dict['imputation'] != None:
        preprocess_str = preprocess_str + '_impute_' + data_config.fitbit_sensor_dict['imputation']
    else:
        preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['feature']
    preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_setting']
    preprocess_str = preprocess_str + '_' + data_config.fitbit_sensor_dict['preprocess_cols']
    tmp_path = os.path.join(tmp_path, preprocess_str)
    create_folder(tmp_path)
    
    data_config.fitbit_sensor_dict['clustering_path'] = tmp_path
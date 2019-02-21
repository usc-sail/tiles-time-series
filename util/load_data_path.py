import os
import sys


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
    tmp_path = os.path.join(tmp_path, data_config.omsignal_sensor_dict['feature'])
    create_folder(tmp_path)
    tmp_path = os.path.join(tmp_path, data_config.omsignal_sensor_dict['preprocess_setting'])
    create_folder(tmp_path)
    tmp_path = os.path.join(tmp_path, data_config.omsignal_sensor_dict['preprocess_cols'])
    create_folder(tmp_path)
    
    data_config.omsignal_sensor_dict['preprocess_path'] = tmp_path

    # Fitbit
    tmp_path = os.path.join(process_data_path, data_name, data_config.fitbit_sensor_dict['name'])
    create_folder(tmp_path)
    
    if data_config.fitbit_sensor_dict['imputation'] != None:
        tmp_path = os.path.join(tmp_path, 'impute_' + data_config.fitbit_sensor_dict['imputation'])
    else:
        tmp_path = os.path.join(tmp_path, data_config.omsignal_sensor_dict['feature'])
    
    create_folder(tmp_path)
    tmp_path = os.path.join(tmp_path, data_config.fitbit_sensor_dict['preprocess_setting'])
    create_folder(tmp_path)
    tmp_path = os.path.join(tmp_path, data_config.fitbit_sensor_dict['preprocess_cols'])
    create_folder(tmp_path)

    data_config.fitbit_sensor_dict['preprocess_path'] = tmp_path

    # owl_in_one
    tmp_path = os.path.join(process_data_path, data_name, data_config.owl_in_one_sensor_dict['name'])
    create_folder(tmp_path)
    tmp_path = os.path.join(tmp_path, data_config.owl_in_one_sensor_dict['feature'])
    create_folder(tmp_path)
    tmp_path = os.path.join(tmp_path, data_config.owl_in_one_sensor_dict['preprocess_setting'])
    create_folder(tmp_path)
    
    data_config.owl_in_one_sensor_dict['preprocess_path'] = tmp_path
    
    # owl_in_one
    tmp_path = os.path.join(process_data_path, data_name, data_config.realizd_sensor_dict['name'])
    create_folder(tmp_path)
    tmp_path = os.path.join(tmp_path, data_config.realizd_sensor_dict['feature'])
    create_folder(tmp_path)
    tmp_path = os.path.join(tmp_path, data_config.realizd_sensor_dict['preprocess_setting'])
    create_folder(tmp_path)

    data_config.realizd_sensor_dict['preprocess_path'] = tmp_path


def load_segmentation_path(data_config, process_data_path, data_name='preprocess_data'):
    """ Produces a fixed-length summary vector of the sequence for each segment.

    Params:
    data_config - config setting
    process_data_path - the path contains all processed stream
    data_name - data name

    Returns:

    """
    tmp_path = os.path.join(process_data_path, data_name)
    create_folder(tmp_path)
    
    # Fitbit segmentation
    tmp_path = os.path.join(process_data_path, data_name, data_config.fitbit_sensor_dict['name'])
    create_folder(tmp_path)
    
    if data_config.fitbit_sensor_dict['imputation'] != None:
        tmp_path = os.path.join(tmp_path, 'impute_' + data_config.fitbit_sensor_dict['imputation'])
    else:
        tmp_path = os.path.join(tmp_path, data_config.omsignal_sensor_dict['feature'])

    if data_config.fitbit_sensor_dict['segmentation_method'] == 'gaussian':
        tmp_path = os.path.join(tmp_path, 'ggs_' + str(data_config.fitbit_sensor_dict['segmentation_lamb']))
    else:
        tmp_path = os.path.join(tmp_path, data_config.fitbit_sensor_dict['segmentation_method'])
    
    create_folder(tmp_path)
    tmp_path = os.path.join(tmp_path, data_config.fitbit_sensor_dict['preprocess_setting'])
    create_folder(tmp_path)
    tmp_path = os.path.join(tmp_path, data_config.fitbit_sensor_dict['preprocess_cols'])
    create_folder(tmp_path)
    
    data_config.fitbit_sensor_dict['segmentation_path'] = tmp_path
    

def load_fitbit_summary_path(tiles_data_path, data_name='3_preprocessed_data'):
    return os.path.join(tiles_data_path, data_name, 'fitbit')


def load_clustering_path(data_config, process_data_path, data_name='preprocess_data'):
    """ Produces a fixed-length summary vector of the sequence for each segment.

    Params:
    data_config - config setting
    process_data_path - the path contains all processed stream
    data_name - data name

    Returns:

    """
    tmp_path = os.path.join(process_data_path, data_name)
    create_folder(tmp_path)
    
    # Fitbit segmentation
    tmp_path = os.path.join(process_data_path, data_name, data_config.fitbit_sensor_dict['name'])
    create_folder(tmp_path)
    
    if data_config.fitbit_sensor_dict['imputation'] != None:
        tmp_path = os.path.join(tmp_path, 'impute_' + data_config.fitbit_sensor_dict['imputation'])
    else:
        tmp_path = os.path.join(tmp_path, data_config.omsignal_sensor_dict['feature'])
    create_folder(tmp_path)
    
    if data_config.fitbit_sensor_dict['segmentation_method'] == 'gaussian':
        tmp_path = os.path.join(tmp_path, 'ggs_' + str(data_config.fitbit_sensor_dict['segmentation_lamb']))
    else:
        tmp_path = os.path.join(tmp_path, data_config.fitbit_sensor_dict['segmentation_method'])
    create_folder(tmp_path)

    tmp_path = os.path.join(tmp_path, data_config.fitbit_sensor_dict['cluster_method'] + '_' + str(data_config.fitbit_sensor_dict['num_cluster']))
    create_folder(tmp_path)
    tmp_path = os.path.join(tmp_path, data_config.fitbit_sensor_dict['preprocess_setting'])
    create_folder(tmp_path)
    tmp_path = os.path.join(tmp_path, data_config.fitbit_sensor_dict['preprocess_cols'])
    create_folder(tmp_path)
    
    data_config.fitbit_sensor_dict['clustering_path'] = tmp_path
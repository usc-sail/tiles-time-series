"""
Config Files
"""
from __future__ import print_function

###########################################################
# Import system library
###########################################################
import sys
import csv
import os
import copy
from configparser import ConfigParser

__all__ = ['Config']


class Config(object):
    _process_hype = {'data_type': 'preprocess_data', 'imputation': None,
                     'segmentation': None, 'segmentation_lamb': 1e0, 'sub_segmentation_lamb': None,
                     'offset': 30, 'overlap': 0}
    
    _default_signal = {'raw_cols': ['BreathingDepth', 'BreathingRate', 'Cadence', 'HeartRate', 'Intensity', 'Steps'],
                       'MinPeakDistance': 100, 'MinPeakHeight': 0.04}
    
    def __init__(self):
        ###########################################################
        # Initiate config
        ###########################################################
        self.config = ConfigParser()
        
    def saveConfig(self, om_process_param, fitbit_process_param, owl_in_one_param, realizd_param,
                   segmentation_param, cluster_param, global_param):
    
        ###########################################################
        # Initiate OMSignal
        ###########################################################
        self.config.add_section('omsignal')
        self.config.set('omsignal', 'preprocess_setting', 'offset_' + str(om_process_param['offset']) + '_overlap_' + str(om_process_param['overlap']))
        self.config.set('omsignal', 'offset', str(om_process_param['offset']))
        self.config.set('omsignal', 'feature', om_process_param['feature'])
        self.config.set('omsignal', 'imputation', str(om_process_param['imputation']))

        process_col_array = list(om_process_param['preprocess_cols'])
        process_col_array.sort()
        self.config.set('omsignal', 'preprocess_cols', '-'.join(process_col_array))

        ###########################################################
        # Initiate Fitbit
        ###########################################################
        self.config.add_section('fitbit')
        self.config.set('fitbit', 'preprocess_setting', 'offset_' + str(fitbit_process_param['offset']) + '_overlap_' + str(fitbit_process_param['overlap']))
        self.config.set('fitbit', 'offset', str(fitbit_process_param['offset']))
        self.config.set('fitbit', 'feature', fitbit_process_param['feature'])
        self.config.set('fitbit', 'imputation', fitbit_process_param['imputation'])

        process_col_array = list(fitbit_process_param['preprocess_cols'])
        process_col_array.sort()
        self.config.set('fitbit', 'preprocess_cols', '-'.join(process_col_array))

        ###########################################################
        # Initiate owl_in_one
        ###########################################################
        self.config.add_section('owl_in_one')
        self.config.set('owl_in_one', 'preprocess_setting', 'offset_' + str(owl_in_one_param['offset']))
        self.config.set('owl_in_one', 'offset', str(owl_in_one_param['offset']))

        ###########################################################
        # Initiate realizd
        ###########################################################
        self.config.add_section('realizd')
        self.config.set('realizd', 'preprocess_setting', 'offset_' + str(realizd_param['offset']))
        self.config.set('realizd', 'offset', str(realizd_param['offset']))
        
        ###########################################################
        # Initiate segmentation and cluster parameters
        ###########################################################
        self.config.add_section('segmentation')
        self.config.set('segmentation', 'method', segmentation_param['method'])
        self.config.set('segmentation', 'lamb', str(segmentation_param['segmentation_lamb']))

        self.config.add_section('clustering')
        self.config.set('clustering', 'method', cluster_param['method'])
        self.config.set('clustering', 'num_cluster', str(cluster_param['num_cluster']))

        ###########################################################
        # Initiate global parameters
        ###########################################################
        self.config.add_section('global')
        self.config.set('global', 'plot', str(global_param['enable_plot']))

    def createConfigFile(self):
        ###########################################################
        # Add folder information
        ###########################################################
        configFilePath = 'settings.ini'
        with open(configFilePath, 'w') as config_file:
            self.config.write(config_file)
        
    def readConfigFile(self):
    
        ###########################################################
        # Config folder information
        ###########################################################
        configFilePath = 'settings.ini'
    
        if os.path.exists(configFilePath) is False:
            print('Config file not exist! Please Check!')

        self.config.read(configFilePath)
        
        ###########################################################
        # Read OMSignal
        ###########################################################
        self.omsignal_sensor_dict = {}
        self.omsignal_sensor_dict['name'] = 'omsignal'
        self.omsignal_sensor_dict['preprocess_setting'] = self.getSetting('omsignal', 'preprocess_setting')
        self.omsignal_sensor_dict['offset'] = int(self.getSetting('omsignal', 'offset'))
        self.omsignal_sensor_dict['feature'] = self.getSetting('omsignal', 'feature')
        self.omsignal_sensor_dict['imputation'] = self.getSetting('omsignal', 'imputation')
        self.omsignal_sensor_dict['preprocess_cols'] = self.getSetting('omsignal', 'preprocess_cols')
    
        ###########################################################
        # Read Fitbit
        ###########################################################
        self.fitbit_sensor_dict = {}
        self.fitbit_sensor_dict['name'] = 'fitbit'
        self.fitbit_sensor_dict['preprocess_setting'] = self.getSetting('fitbit', 'preprocess_setting')
        self.fitbit_sensor_dict['offset'] = int(self.getSetting('fitbit', 'offset'))
        self.fitbit_sensor_dict['feature'] = self.getSetting('fitbit', 'feature')
        self.fitbit_sensor_dict['imputation'] = self.getSetting('fitbit', 'imputation')
        self.fitbit_sensor_dict['preprocess_cols'] = self.getSetting('fitbit', 'preprocess_cols')
    
        ###########################################################
        # Read owl_in_one
        ###########################################################
        self.owl_in_one_sensor_dict = {}
        self.owl_in_one_sensor_dict['name'] = 'owl_in_one'
        self.owl_in_one_sensor_dict['preprocess_setting'] = self.getSetting('owl_in_one', 'preprocess_setting')
        self.owl_in_one_sensor_dict['offset'] = int(self.getSetting('owl_in_one', 'offset'))
    
        ###########################################################
        # Read Fitbit
        ###########################################################
        self.owl_in_one_sensor_dict = {}
        self.owl_in_one_sensor_dict['name'] = 'realizd'
        self.owl_in_one_sensor_dict['preprocess_setting'] = self.getSetting('realizd', 'preprocess_setting')
        self.owl_in_one_sensor_dict['offset'] = int(self.getSetting('realizd', 'offset'))
        
        ###########################################################
        # Read global parameters
        ###########################################################
        self.enable_plot = bool(self.getSetting('global', 'plot'))
        
        self.segmentation_method = self.getSetting('segmentation', 'method')
        self.segmentation_lamb = float(self.getSetting('segmentation', 'lamb'))

        self.cluster_method = self.getSetting('clustering', 'method')
        self.num_cluster = int(self.getSetting('clustering', 'num_cluster'))
        
    def createFolder(self, folder):
        if os.path.exists(folder) is False: os.mkdir(folder)

    def updateFolderName(self):
        ###########################################################
        # Create main folders
        ###########################################################
        self.createFolder(self.main_folder)
        self.main_folder = os.path.join(self.main_folder, self.data_type)
        self.createFolder(self.main_folder)

        ###########################################################
        # Create Signal folders
        ###########################################################
        self.signal_type_folder = os.path.join(self.main_folder, self.sensor)
        self.createFolder(self.signal_type_folder)
        
        if self.return_full_feature == True and self.sensor is not 'realizd':
            self.signal_type_folder = os.path.join(self.signal_type_folder, 'feat')
        elif self.enable_impute == True and self.sensor is not 'realizd':
            self.signal_type_folder = os.path.join(self.signal_type_folder, 'impute_' + self.imputation_method)
        else:
            self.signal_type_folder = os.path.join(self.signal_type_folder, 'original')
        self.createFolder(self.signal_type_folder)
        
        if self.enable_segmentation == True:
            if self.sub_segmentation_lamb is not None:
                self.signal_type_folder = os.path.join(self.signal_type_folder,
                                                       self.segmentation + '_' + str(self.segmentation_lamb) +
                                                       '_' + str(self.sub_segmentation_lamb))
            else:
                self.signal_type_folder = os.path.join(self.signal_type_folder,
                                                       self.segmentation + '_' + str(self.segmentation_lamb))
        self.createFolder(self.signal_type_folder)

        ###########################################################
        # Create parameter folder
        ###########################################################
        if self.sensor != 'realizd' and self.sensor != 'owl_in_one':
            self.process_folder = os.path.join(self.signal_type_folder, self.process_basic_str)
            self.createFolder(self.process_folder)
            
            self.process_folder = os.path.join(self.process_folder, self.process_col_str)
            self.createFolder(self.process_folder)
        else:
            self.process_folder = os.path.join(self.signal_type_folder, self.process_basic_str)
            self.createFolder(self.process_folder)
    
    def getSetting(self, section, setting):
        ###########################################################
        # Return value of a setting
        ###########################################################
        return self.config.get(section, setting)
    
    def printConfig(self):
        ###########################################################
        # Print Configs
        ###########################################################
        print('----------------------------------------------------')
        print('Config Class, printConfig')
        if self.data_type != '3_preprocessed_data':
            print('process_basic_folder: ' + self.process_folder)
        print('sensor: ' + self.sensor)
        print('----------------------------------------------------')
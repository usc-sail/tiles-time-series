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
    
    def __init__(self):
        ###########################################################
        # Initiate config
        ###########################################################
        self.config = ConfigParser()
        
    def saveConfig(self, om_process_param, fitbit_process_param, owl_in_one_param, realizd_param, audio_param,
                   segmentation_param, cluster_param, feature_engineering_param, global_param, experiement):
    
        ###########################################################
        # Initiate OMSignal
        ###########################################################
        self.config.add_section('om_signal')
        self.config.set('om_signal', 'preprocess_setting', 'offset_' + str(om_process_param['offset']) + '_overlap_' + str(om_process_param['overlap']))
        self.config.set('om_signal', 'offset', str(om_process_param['offset']))
        self.config.set('om_signal', 'overlap', str(om_process_param['overlap']))
        self.config.set('om_signal', 'feature', om_process_param['feature'])
        self.config.set('om_signal', 'imputation', str(om_process_param['imputation']))

        process_col_array = list(om_process_param['preprocess_cols'])
        process_col_array.sort()
        self.config.set('om_signal', 'preprocess_cols', '-'.join(process_col_array))

        ###########################################################
        # Initiate Fitbit
        ###########################################################
        self.config.add_section('fitbit')
        self.config.set('fitbit', 'preprocess_setting', 'offset_' + str(fitbit_process_param['offset']) + '_overlap_' + str(fitbit_process_param['overlap']))
        self.config.set('fitbit', 'offset', str(fitbit_process_param['offset']))
        self.config.set('fitbit', 'overlap', str(fitbit_process_param['overlap']))
        self.config.set('fitbit', 'feature', fitbit_process_param['feature'])
        self.config.set('fitbit', 'imputation', fitbit_process_param['imputation'])
        self.config.set('fitbit', 'segmentation_method', str(fitbit_process_param['segmentation_method']))
        self.config.set('fitbit', 'segmentation_lamb', str(fitbit_process_param['segmentation_lamb']))
        self.config.set('fitbit', 'cluster_method', fitbit_process_param['cluster_method'])
        self.config.set('fitbit', 'num_cluster', str(fitbit_process_param['num_cluster']))
        if fitbit_process_param['cluster_method'] == 'ticc':
            self.config.set('fitbit', 'ticc_window', str(fitbit_process_param['ticc_window']))
            self.config.set('fitbit', 'ticc_switch_penalty', str(fitbit_process_param['ticc_switch_penalty']))
            self.config.set('fitbit', 'ticc_sparsity', str(fitbit_process_param['ticc_sparsity']))
            self.config.set('fitbit', 'ticc_cluster_days', str(fitbit_process_param['ticc_cluster_days']))

        process_col_array = list(fitbit_process_param['preprocess_cols'])
        process_col_array.sort()
        self.config.set('fitbit', 'preprocess_cols', '-'.join(process_col_array))

        ###########################################################
        # Initiate owl_in_one
        ###########################################################
        self.config.add_section('owl_in_one')
        self.config.set('owl_in_one', 'feature', owl_in_one_param['feature'])
        self.config.set('owl_in_one', 'preprocess_setting', 'offset_' + str(owl_in_one_param['offset']))
        self.config.set('owl_in_one', 'offset', str(owl_in_one_param['offset']))

        ###########################################################
        # Initiate realizd
        ###########################################################
        self.config.add_section('realizd')
        self.config.set('realizd', 'feature', realizd_param['feature'])
        self.config.set('realizd', 'preprocess_setting', 'offset_' + str(realizd_param['offset']))
        self.config.set('realizd', 'offset', str(realizd_param['offset']))

        ###########################################################
        # Initiate audio
        ###########################################################
        self.config.add_section('audio')
        self.config.set('audio', 'feature', audio_param['feature'])
        self.config.set('audio', 'preprocess_setting', 'offset_%d'%(int(audio_param['offset'])))
        self.config.set('audio', 'offset', str(audio_param['offset']))

        ###########################################################
        # Initiate feature engineering parameters
        ###########################################################
        self.config.add_section('feature_engineering')
        self.config.set('feature_engineering', 'features', str(feature_engineering_param['features']))
        
        ###########################################################
        # Initiate global parameters
        ###########################################################
        self.config.add_section('global')
        self.config.set('global', 'filter_method', str(global_param['filter_method']))
        self.config.set('global', 'plot', str(global_param['enable_plot']))

    def createConfigFile(self, dataDir, experiement):
        ###########################################################
        # Add folder information
        ###########################################################
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)
        configFilePath = os.path.join(dataDir, experiement + '.cfg')
        with open(configFilePath, 'w') as config_file:
            self.config.write(config_file)
        
    def readConfigFile(self, dataDir, experiement):
    
        ###########################################################
        # Config folder information
        ###########################################################
        configFilePath = os.path.join(dataDir, experiement + '.cfg')
    
        if os.path.exists(configFilePath) is False:
            print('Config file not exist! Please Check!')

        self.config.read(configFilePath)
        
        self.experiement = experiement
        
        ###########################################################
        # Read OMSignal
        ###########################################################
        self.omsignal_sensor_dict = {}
        self.omsignal_sensor_dict['name'] = 'om_signal'
        self.omsignal_sensor_dict['preprocess_setting'] = self.getSetting('om_signal', 'preprocess_setting')
        self.omsignal_sensor_dict['offset'] = int(self.getSetting('om_signal', 'offset'))
        self.omsignal_sensor_dict['overlap'] = int(self.getSetting('om_signal', 'overlap'))
        self.omsignal_sensor_dict['feature'] = self.getSetting('om_signal', 'feature')
        self.omsignal_sensor_dict['imputation'] = self.getSetting('om_signal', 'imputation')
        self.omsignal_sensor_dict['preprocess_cols'] = self.getSetting('om_signal', 'preprocess_cols')
    
        ###########################################################
        # Read Fitbit
        ###########################################################
        self.fitbit_sensor_dict = {}
        self.fitbit_sensor_dict['name'] = 'fitbit'
        self.fitbit_sensor_dict['preprocess_setting'] = self.getSetting('fitbit', 'preprocess_setting')
        self.fitbit_sensor_dict['offset'] = int(self.getSetting('fitbit', 'offset'))
        self.fitbit_sensor_dict['overlap'] = int(self.getSetting('fitbit', 'overlap'))
        self.fitbit_sensor_dict['feature'] = self.getSetting('fitbit', 'feature')
        self.fitbit_sensor_dict['imputation'] = self.getSetting('fitbit', 'imputation')
        self.fitbit_sensor_dict['imputation_threshold'] = int(self.getSetting('fitbit', 'imputation_threshold'))
        self.fitbit_sensor_dict['preprocess_cols'] = self.getSetting('fitbit', 'preprocess_cols')
        self.fitbit_sensor_dict['segmentation_method'] = self.getSetting('fitbit', 'segmentation_method')
        self.fitbit_sensor_dict['segmentation_lamb'] = float(self.getSetting('fitbit', 'segmentation_lamb'))
        self.fitbit_sensor_dict['cluster_method'] = self.getSetting('fitbit', 'cluster_method')
        self.fitbit_sensor_dict['num_cluster'] = int(self.getSetting('fitbit', 'num_cluster'))
        
        if self.fitbit_sensor_dict['cluster_method'] == 'ticc':
            self.fitbit_sensor_dict['ticc_window'] = int(self.getSetting('fitbit', 'ticc_window'))
            self.fitbit_sensor_dict['ticc_switch_penalty'] = float(self.getSetting('fitbit', 'ticc_switch_penalty'))
            self.fitbit_sensor_dict['ticc_sparsity'] = float(self.getSetting('fitbit', 'ticc_sparsity'))
            self.fitbit_sensor_dict['ticc_cluster_days'] = int(self.getSetting('fitbit', 'ticc_cluster_days'))
            
        ###########################################################
        # Read owl_in_one
        ###########################################################
        self.owl_in_one_sensor_dict = {}
        self.owl_in_one_sensor_dict['name'] = 'owl_in_one'
        self.owl_in_one_sensor_dict['preprocess_setting'] = self.getSetting('owl_in_one', 'preprocess_setting')
        self.owl_in_one_sensor_dict['feature'] = self.getSetting('owl_in_one', 'feature')
        self.owl_in_one_sensor_dict['offset'] = int(self.getSetting('owl_in_one', 'offset'))
    
        ###########################################################
        # Read realizd
        ###########################################################
        self.realizd_sensor_dict = {}
        self.realizd_sensor_dict['name'] = 'realizd'
        self.realizd_sensor_dict['feature'] = self.getSetting('realizd', 'feature')
        self.realizd_sensor_dict['preprocess_setting'] = self.getSetting('realizd', 'preprocess_setting')
        self.realizd_sensor_dict['offset'] = int(self.getSetting('realizd', 'offset'))

        ###########################################################
        # Read audio 
        ###########################################################
        self.audio_sensor_dict = {}
        self.audio_sensor_dict['name'] = 'audio'
        self.audio_sensor_dict['feature'] = self.getSetting('audio', 'feature')
        self.audio_sensor_dict['preprocess_setting'] = self.getSetting('audio', 'preprocess_setting')
        self.audio_sensor_dict['offset'] = self.getSetting('audio', 'offset')
        self.audio_sensor_dict['cluster_alpha'] = self.getSetting('audio', 'cluster_alpha')
        self.audio_sensor_dict['cluster_method'] = self.getSetting('audio', 'cluster_method')
        self.audio_sensor_dict['cluster_data'] = self.getSetting('audio', 'cluster_data')
        self.audio_sensor_dict['lda_num'] = self.getSetting('audio', 'lda_num')
        self.audio_sensor_dict['audio_feature'] = self.getSetting('audio', 'audio_feature')
        self.audio_sensor_dict['pause_threshold'] = self.getSetting('audio', 'pause_threshold')

        ###########################################################
        # Read feature engineering parameters
        ###########################################################
        self.feature_engineering_dict = {}
        self.feature_engineering_dict['features'] = self.getSetting('feature_engineering', 'features')
        
        ###########################################################
        # Read global parameters
        ###########################################################
        self.enable_plot = bool(self.getSetting('global', 'plot'))
        self.filter_method = self.getSetting('global', 'filter_method')
        
    def createFolder(self, folder):
        if os.path.exists(folder) is False: os.mkdir(folder)
    
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
        print('----------------------------------------------------')

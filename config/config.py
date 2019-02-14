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
                     'method': 'ma', 'offset': 30, 'overlap': 0}
    
    _default_signal = {'raw_cols': ['BreathingDepth', 'BreathingRate', 'Cadence', 'HeartRate', 'Intensity', 'Steps'],
                       'MinPeakDistance': 100, 'MinPeakHeight': 0.04}
    
    def __init__(self, data_type='preprocess_data', sensor='fitbit', read_folder=os.path.curdir,
                 return_full_feature=False, process_hyper=_process_hype, signal_hyper=_default_signal):
        ###########################################################
        # Initiate parameters
        ###########################################################
        self.sensor = sensor
        self.data_type = data_type
        self.main_folder = read_folder
        
        ###########################################################
        # 1. Update hyper parameters for signal and preprocess method
        ###########################################################
        self.process_hyper = copy.deepcopy(self._process_hype)
        self.process_hyper.update(process_hyper)

        self.signal_hypers = copy.deepcopy(self._default_signal)
        self.signal_hypers.update(signal_hyper)

        ###########################################################
        # 2. save data folder
        ###########################################################
        if self.sensor != 'realizd' and self.sensor != 'owl_in_one':
            self.process_basic_str = 'method_' + self.process_hyper['method'] + '_offset_' + str(self.process_hyper['offset']) + '_overlap_' + str(self.process_hyper['overlap'])
            process_col_array = list(self.process_hyper['preprocess_cols'])
            process_col_array.sort()
            self.process_col_str = '-'.join(process_col_array)
        else:
            self.process_basic_str = 'offset_' + str(self.process_hyper['offset'])
            
        self.offset = self.process_hyper['offset']
        self.segmentation_lamb = self.process_hyper['segmentation_lamb']
        self.sub_segmentation_lamb = self.process_hyper['sub_segmentation_lamb']
    
        # self.participant_id = participant_id
        self.imputation_method = self.process_hyper['imputation']
        self.segmentation = self.process_hyper['segmentation']

        ###########################################################
        # 3. Create Sub folder
        ###########################################################
        self.return_full_feature = return_full_feature
        self.enable_impute = True if self.imputation_method is not None else False
        self.enable_segmentation = True if self.segmentation is not None else False
        
        if self.data_type != '3_preprocessed_data':
            self.updateFolderName()
        else:
            self.main_folder = os.path.join(self.main_folder, self.data_type)
            self.signal_type_folder = os.path.join(self.main_folder, self.sensor)
        
        self.config = ConfigParser()
        self.initConfig()
        
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
        
    def initConfig(self):
        ###########################################################
        # Add folder information
        ###########################################################
        self.config.add_section('folder')
        self.config.set('folder', 'save_main_folder', str(self.main_folder))
        self.config.set('folder', 'signal_type_folder', str(self.signal_type_folder))

        if self.data_type != '3_preprocessed_data':
            self.config.set('folder', 'read_process_folder', str(self.process_folder))
        
        ###########################################################
        # Add aggregation information
        ###########################################################
        self.config.add_section('sensor')
        self.config.set('sensor', 'sensor', str(self.sensor))
    
    def createConfigFile(self, configDir):
        ###########################################################
        # Add folder information
        ###########################################################
        configFilePath = os.path.join(configDir, 'settings.ini')
        with open(configFilePath, 'w') as config_file:
            self.config.write(config_file)
    
    def readConfigFile(self, configDir):
        ###########################################################
        # Config folder information
        ###########################################################
        configFilePath = os.path.join(configDir, 'settings.ini')
        
        if os.path.exists(configFilePath) is False:
            print('Config file not exist! Please Check!')
        
        self.config.read(configFilePath)
        self.sensor = self.getSetting('sensor', 'sensor')
        
        ###########################################################
        # Print parameters
        ###########################################################
        self.printConfig()
    
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
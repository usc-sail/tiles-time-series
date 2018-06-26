"""
This is script is modified based on Karel Mundnich's script: days_at_work.py
Script is modified by Tiantian Feng
"""

import os, errno
import glob
import argparse
import numpy as np
import pandas as pd
from dateutil import rrule
from datetime import datetime, timedelta

is_use_hospital_information = False

def getParticipantIDsFromFiles(files):
    slash_index = files[0].rfind('/')
    underscore_index = files[0].rfind('_')
    
    participantIDs = []
    
    for file in files:
        participantIDs.append(file[slash_index + 1:underscore_index])
    
    return participantIDs


def getAllParticipantIDs(owl_files, om_files):
    owlParticipantIDs = getParticipantIDsFromFiles(owl_files)
    omParticipantIDs = getParticipantIDsFromFiles(om_files)
    
    participantIDs = mergeParticipantLists(owlParticipantIDs, omParticipantIDs)
    
    return sorted(participantIDs)


def mergeParticipantLists(listA, listB):
    inA = set(listA)
    inB = set(listB)
    
    return list(inA) + list(inB - inA)


def getParticipantIDFromFileName(file):
    slash_index = file.rfind('/')
    underscore_index = file.rfind('_')
    
    return file[slash_index + 1:underscore_index]


def getDatesFromFiles(files):
    start_dates = []
    end_dates = []
    for file in files:
        print('--------- getDatesFromFiles ---------')
        print('Read:' + file)
        
        data = getDataFrame(file)

        # If there is something with timestamp
        for i in range(100):
            if data.index[i].year > 2017:
                start_date = data.index[i]
                break
        
        start_dates.append(start_date)
        end_dates.append(data.index[-1])
    
    return min(start_dates), max(end_dates)


def getDataFrame(file):
    # Read and prepare owl data per participant
    data = pd.read_csv(file, index_col=0)
    data.index = pd.to_datetime(data.index)
    
    return data


def getDaysAtWorkDataFrame(file):
    return pd.read_csv(file, index_col=0)


def isKeck(row, rounding_error=3):
    # Keck is at (lat, lon) = (34.061636, -118.201321)
    return round(row['gps_lat'], rounding_error) == 34.062 and round(row['gps_lon'], rounding_error) == -118.201


def isAnotherHospital(data_index, data_row, participant, hospitals=None, rounding_error=3):
    if hospitals is None:
        try:
            hospitals_file = './hospitals.csv'
            hospitals = pd.read_csv(hospitals_file)
        except FileNotFoundError:
            print('Include hospitals.csv in current directory.')
    
    for hospital_index, hospital_row in hospitals.iterrows():
        try:
            latitudes = round(hospital_row['lat'], rounding_error) == round(data_row['gps_lat'], rounding_error)
            longitudes = round(hospital_row['lon'], rounding_error) == round(data_row['gps_lon'], rounding_error)
            Keck = hospital_row['Facility Name'] == 'Keck Hospital of USC'
            Norris = hospital_row['Facility Name'] == 'USC Kenneth Norris Jr. Cancer Hospital'
            if latitudes and longitudes and not Keck and not Norris:
                print(data_index, participant, hospital_row['Facility Name'])
                return True
        except ValueError:
            pass
    
    return False


def sideBySide(series1, series2):
    return pd.DataFrame(dict(s1=series1, s2=series2))


def main(data_directory, output_directory):
    stream = os.path.basename(data_directory)
    streams = ['omsignal', 'owl_in_one', 'phone_events', 'ground_truth']
    assert (stream in streams), "Stream " + stream + " not found in " + str(streams) + ". Please check data directory."
    
    csv_file = os.path.join(output_directory, stream + '_days_at_work.csv')
    
    if not os.path.exists(csv_file):
        # Obtain participant IDs from file names
        files = glob.glob(os.path.join(data_directory, '*.csv'))
        if not files:
            raise FileNotFoundError('Passed data directory is empty')
        
        if stream in ['omsignal', 'owl_in_one', 'phone_events']:
            participantIDs = getParticipantIDsFromFiles(files)
            
            # Create a time range for the data to be used. We read through all the
            # files and obtain the earliest and latest dates. This is the time range
            # used to produced the data to be saved in 'preprocessed/'
            print('Finding all dates, this may take a while...', end='')
            start_date, end_date = getDatesFromFiles(files)
            dates_range = pd.date_range(start=start_date, end=end_date, normalize=True)
            print('done.')
            
            # Initialize the time frame to store days at Keck
            days_at_work = pd.DataFrame(np.nan, index=dates_range, columns=participantIDs)
            
            for participant in participantIDs:
                if stream == 'owl_in_one':
                    file = os.path.join(data_directory, participant + '_bleProximity.csv')
                elif stream == 'omsignal':
                    file = os.path.join(data_directory, participant + '_omsignal.csv')
                elif stream == 'phone_events':
                    file = os.path.join(data_directory, participant + '_phoneEvents.csv')
                
                data = getDataFrame(file)
                dates_worked = list(set([date.date() for date in data.index]))
                
                if stream == 'owl_in_one' or stream == 'omsignal':
                    for date in dates_range:
                        if date.date() in dates_worked:
                            days_at_work.loc[date.date(), participant] = 1
                
                
                elif stream == 'phone_events':
                    if is_use_hospital_information is True:
                        hospitals_file = './hospitals.csv'
                        hospitals = pd.read_csv(hospitals_file)
                        
                        for index, row in data.iterrows():
                            # Check if the participant was at Keck
                            if isKeck(row):
                                # Take only the date in 'Timestamp' and use it as index
                                days_at_work.loc[index.date(), participant] = 1.0
                            # Check if the participant was at another hospital, will assume that they were working
                            # TODO: Check if the participant has visited another hospital multiple times
                            elif isAnotherHospital(index, row, participant):
                                days_at_work.loc[index.date(), participant] = 1.0
        
        elif stream == 'ground_truth':
            MGT = pd.read_csv(os.path.join(data_directory, 'MGT.csv'), index_col=2)
            IDs = pd.read_csv(os.path.join(data_directory, 'IDs.csv'), index_col=1)
            IDs.columns = ['Evidation_id']
            IDs.index.names = ['MITRE_id']
            
            MGT.index = pd.to_datetime(MGT.index)
            
            start_date = min(MGT.index).date()
            end_date = max(MGT.index).date()
            
            dates_range = pd.date_range(start=start_date, end=end_date)
            participantIDs = sorted(list(IDs['Evidation_id'].unique()))
            
            days_at_work = pd.DataFrame(np.nan, index=dates_range, columns=participantIDs)
            
            for index, row in MGT.iterrows():
                try:
                    participant = IDs.loc[row['uid']]['Evidation_id']
                    if row['location_mgt'] == 2.0:  # At work when answering the survey according to MGT, question 1
                        days_at_work.loc[index.date()][participant] = 1.0
                except KeyError:
                    print('Participant ' + row['uid'] + ' is not in participant list from IDs.csv.')
        
        if not os.path.exists(output_directory):
            try:
                print('Creating directory ' + output_directory)
                os.makedirs(output_directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        
        days_at_work.to_csv(os.path.join(output_directory, stream + '_days_at_work.csv'), index_label='Timestamp')
    
    else:
        print('File ' + csv_file + ' already exists. Exiting.')


if __name__ == "__main__":
    
    """
        Parse the args:
        1. main_data_directory: directory to store keck data
        2. days_at_work_directory: directory to store days at work data using different modalities
    """
    parser = argparse.ArgumentParser(description='Create a dataframe of worked days.')
    parser.add_argument('-i', '--main_data_directory', type=str, required=True,
                        help='Directory for data.')
    parser.add_argument('-d', '--days_at_work_directory', type=str, required=True,
                        help='Directory with processed data.')
    args = parser.parse_args()
    
    main_data_directory = os.path.expanduser(os.path.normpath(args.main_data_directory))
    days_at_work_directory = os.path.expanduser(os.path.normpath(args.days_at_work_directory))
    
    # if path not exist, create the path
    if os.path.exists(days_at_work_directory): os.mkdir(days_at_work_directory)

    # Not using phone_events, since I don't have hospital.csv
    # stream_types = ['omsignal', 'owl_in_one', 'phone_events', 'ground_truth']
    stream_types = ['omsignal', 'owl_in_one', 'ground_truth']
    
    for steam in stream_types:
        data_directory = os.path.join(main_data_directory, steam)
        main(data_directory, days_at_work_directory)

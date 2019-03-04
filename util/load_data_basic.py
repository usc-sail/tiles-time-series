import os, errno
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import load_sensor_data

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

'''
Inpatient Rehabilitation Unit (3 North)
Cardiovascular Thoracic ICU (4 South),
Pulmonary Medical ICU (5 ICU), Stepdown Telemetry Unit (5 South),
Cardiothoracic Surgery ICU (5 West), Cardiothoracic Surgery Telemetry (5 East)
Medical/Surgical Telemetry Abdominal Organ Transplant (6 South),
Surgical ICU (7 West), Neurosciences ICU (7 South),
Abdominal Organ Transplant ICU (7 East), Surgical Telemetry (7 North)
Surgical Oncology ICU (8 West), Pulmonary Medical Telemetry (8 East)
Surgical Oncology Telemetry (9 East)
'''

nurse_unit_types = {'keck:floor4:south': 'ICU',
                    'keck:floor5:north': 'ICU', 'keck:floor5:southICU': 'ICU', 'keck:floor5:west': 'ICU',
                    'keck:floor7:east': 'ICU', 'keck:floor7:south': 'ICU', 'keck:floor7:west': 'ICU',
                    'keck:floor8:west': 'ICU',
                    'keck:floor5:east': 'Normal', 'keck:floor5:south': 'Normal',
                    'keck:floor6:north': 'Normal', 'keck:floor6:south': 'Normal', 'keck:floor6:west': 'Normal',
                    'keck:floor7:north': 'Normal',
                    'keck:floor8:east': 'Normal',
                    'keck:floor9:east': 'Normal'}


def getParticipantIDJobShift(main_data_directory):
    participant_id_job_shift_df = []
    
    # job shift
    job_shift_df = pd.read_csv(os.path.join(main_data_directory, 'job shift/Job_Shift.csv'))
    
    # read id
    id_data_df = pd.read_csv(os.path.join(main_data_directory, 'ground_truth/IDs.csv'))
    
    for index, id_data in id_data_df.iterrows():
        # get job shift and participant id
        job_shift = job_shift_df.loc[job_shift_df['uid'] == id_data['user_id']]['job_shift'].values[0]
        participant_id = id_data['user_id']
        
        frame_df = pd.DataFrame(job_shift, index=['job_shift'], columns=[participant_id]).transpose()
        
        participant_id_job_shift_df = frame_df if len(
            participant_id_job_shift_df) == 0 else participant_id_job_shift_df.append(frame_df)
    
    return participant_id_job_shift_df


def getParticipantInfo(main_data_directory, index=1):
    
    IDs = pd.read_csv(os.path.join(main_data_directory, 'id-mapping', 'mitreids.csv'))
    participant_info = pd.read_csv(os.path.join(main_data_directory, 'participant_info', 'participant_info.csv'))
    participant_info = participant_info.fillna("")
    
    for index, row in participant_info.iterrows():
        
        participant_id = row['ParticipantID']
        mitre_id = row['MitreID']
        
        participant_info.loc[index, 'MitreID'] = IDs.loc[IDs['participant_id'] == participant_id]['mitre_id'].values[0]

    # IDs.index.names = ['MitreID']
    # participant_info = participant_info.set_index('MitreID')
    
    return participant_info

# start date, end date for wave 1 and pilot
def getParticipantStartTime():
    return datetime(year=2018, month=2, day=20)


def getParticipantEndTime():
    return datetime(year=2018, month=6, day=10)


# Load mgt data
def read_MGT(main_data_directory):
    MGT = pd.read_csv(os.path.join(main_data_directory, 'ground_truth/MGT', 'MGT.csv.gz'))
    
    timestamp_str_list = []
    
    for timestamp in MGT.timestamp:
        timestamp_str_list.append(pd.to_datetime(timestamp).strftime(date_time_format)[:-3])

    MGT.index = timestamp_str_list
    
    return MGT


# Load mgt data
def read_app_survey(main_data_directory):
    app_survey_df = pd.read_csv(os.path.join(main_data_directory, '2_raw_csv_data/app_surveys', 'app_surveys_processed.csv.gz'), index_col=0)
    
    return app_survey_df

# Load pre study data
def read_pre_study_info(main_data_directory):
    PreStudyInfo = pd.read_csv(os.path.join(main_data_directory, 'participant_info', 'Pre-Study Data 11-13-18.csv'), index_col=3)
    PreStudyInfo.index = pd.to_datetime(PreStudyInfo.index)
    
    return PreStudyInfo


# Load IGTB data
def read_IGTB(main_data_directory):
    IGTB = pd.read_csv(os.path.join(main_data_directory, 'ground_truth/IGTB', 'IGTB.csv.gz'), index_col=1)
    IGTB.index = pd.to_datetime(IGTB.index)
    
    return IGTB


# Load IGTB data
def read_Demographic(main_data_directory):
    DemoGraphic = pd.read_csv(os.path.join(main_data_directory, 'participant_info', 'Demographic.csv'))
    DemoGraphic.index = pd.to_datetime(DemoGraphic.index)
    
    return DemoGraphic


# Load IGTB data
def read_IGTB_Raw(main_data_directory):
    IGTB = pd.read_csv(os.path.join(main_data_directory, 'ground_truth', 'IGTB_R.csv'), index_col=False)
    IGTB.index = pd.to_datetime(IGTB.index)
    
    return IGTB


# Load all basic data
def read_AllBasic(main_data_directory):
    
    # Read participant information
    participant_info = getParticipantInfo(os.path.join(main_data_directory))
    
    # Read MGT
    # MGT = read_MGT(os.path.join(main_data_directory, 'keck_wave_all'))

    # Read Pre-Study info
    PreStudyInfo = read_pre_study_info(os.path.join(main_data_directory))

    # Read IGTB info
    IGTB = read_IGTB(os.path.join(main_data_directory))

    # Demographic
    Demographic = read_Demographic(os.path.join(main_data_directory))

    UserInfo = pd.merge(IGTB, PreStudyInfo, left_on='uid', right_on='ID', how='outer')
    UserInfo = pd.merge(UserInfo, participant_info, left_on='uid', right_on='MitreID', how='outer')
    UserInfo = pd.merge(UserInfo, Demographic, left_on='uid', right_on='uid', how='outer')
    # UserInfo = UserInfo.loc[UserInfo['Wave'] != 3]

    UserInfo = UserInfo.set_index('uid')
    
    return UserInfo


# Load IGTB data per unit
def read_IGTB_per_unit(user_df, participant_unit_dict):
    
    igtb_cols = [col for col in user_df.columns if 'igtb' in col]
    
    for participant in participant_unit_dict:
        if participant_unit_dict[participant] in nurse_unit_types:
            uid = user_df.loc[user_df['ParticipantID'] == participant].index[0]
            user_df.loc[uid, 'primary_unit'] = participant_unit_dict[participant]
            user_df.loc[uid, 'unit_type'] = nurse_unit_types[participant_unit_dict[participant]]

    ###########################################################
    # 1. Read IGTB per unit base
    ###########################################################
    unique_unit_list = list(set(participant_unit_dict.values()))
    
    return_unit_df = pd.DataFrame()
    
    for unique_unit in unique_unit_list:
        
        if unique_unit in nurse_unit_types.keys():
            unit_df = user_df.loc[user_df['primary_unit'] == unique_unit]
            unit_df = unit_df[igtb_cols]
            
            tmp_array, tmp_col_array = [], []
            
            for name, value in unit_df.mean().iteritems():
                tmp_array.append(value)
                tmp_col_array.append('avg_' + name)
                
            for name, value in unit_df.std().iteritems():
                tmp_array.append(value)
                tmp_col_array.append('std_' + name)
                
            # index=[unique_unit + '_' + nurse_unit_types[unique_unit]],
            unit_igtb_df = pd.DataFrame(np.array(tmp_array).reshape([1, len(tmp_array)]),
                                        index=[unique_unit], columns=tmp_col_array)
    
            return_unit_df = return_unit_df.append(unit_igtb_df)

    ###########################################################
    # 2. Read IGTB per unit type base
    ###########################################################
    return_unit_type_df = pd.DataFrame()
    unit_type_list = list(set(nurse_unit_types.values()))
    for unit_type in unit_type_list:
        unit_type_df = user_df.loc[user_df['unit_type'] == unit_type]
        unit_type_df = unit_type_df[igtb_cols]
    
        tmp_array, tmp_col_array = [], []
    
        for name, value in unit_type_df.mean().iteritems():
            tmp_array.append(value)
            tmp_col_array.append('avg_' + name)
    
        for name, value in unit_type_df.std().iteritems():
            tmp_array.append(value)
            tmp_col_array.append('std_' + name)
    
        unit_type_igtb_df = pd.DataFrame(np.array(tmp_array).reshape([1, len(tmp_array)]),
                                         index=[unit_type], columns=tmp_col_array)

        return_unit_type_df = return_unit_type_df.append(unit_type_igtb_df)
        
    return return_unit_df, return_unit_type_df


def return_participant(tiles_data_path):
    """
    Return all participants that with data

    Params:
    tiles_data_path - tiles data folder

    Returns:
    list(fitbit_file_dict_list.keys()) - participant list
    """
    ###########################################################
    fitbit_folder = os.path.join(tiles_data_path, '3_preprocessed_data/fitbit/')
    fitbit_file_list = os.listdir(fitbit_folder)
    fitbit_file_dict_list = {}
    
    for fitbit_file in fitbit_file_list:
        
        if '.DS' in fitbit_file:
            continue
        
        participant_id = fitbit_file.split('_')[0]
        if participant_id not in list(fitbit_file_dict_list.keys()):
            fitbit_file_dict_list[participant_id] = {}
            
    participant_id_list = list(fitbit_file_dict_list.keys())
    participant_id_list.sort()
    return participant_id_list


def return_top_k_participant(path, tiles_data_path, k=None, data_config=None):
    """
    Return all participants that with data

    Params:
    tiles_data_path - tiles data folder

    Returns:
    fitbit_len_df - top participant df with fitbit data
    """
    # If we have the data, read it
    if os.path.exists(path) is True:
        top_participant_id_df = pd.read_csv(path, index_col=0, compression='gzip')
        if k is not None: top_participant_id_df = top_participant_id_df[:k]
    else:
        # Get all participant id
        participant_id_list = return_participant(tiles_data_path)
        participant_id_list.sort()

        # Iterate to get top k participant
        fitbit_len_list = []
        for idx, participant_id in enumerate(participant_id_list):
            print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(participant_id_list)))
    
            fitbit_df = load_sensor_data.read_preprocessed_fitbit(data_config.fitbit_sensor_dict['preprocess_path'], participant_id)
    
            if fitbit_df is not None:
                fitbit_len_list.append(len(fitbit_df))
            else:
                fitbit_len_list.append(0)

        if k is not None:
            top_participant_list = [participant_id_list[i] for i in np.argsort(fitbit_len_list)[::-1][:k]]
            fitbit_len_sort = np.sort(fitbit_len_list)[::-1][:k]
        else:
            top_participant_list = [participant_id_list[i] for i in np.argsort(fitbit_len_list)[::-1]]
            fitbit_len_sort = np.sort(fitbit_len_list)[::-1]

        top_participant_id_df = pd.DataFrame(fitbit_len_sort, index=top_participant_list)

        top_participant_id_df.to_csv(path, compression='gzip')

    return top_participant_id_df



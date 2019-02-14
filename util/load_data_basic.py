import os, errno
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

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
    MGT = pd.read_csv(os.path.join(main_data_directory, 'ground_truth/MGT', 'MGT.csv'))
    
    timestamp_str_list = []
    
    for timestamp in MGT.timestamp:
        timestamp_str_list.append(pd.to_datetime(timestamp).strftime(date_time_format)[:-3])

    MGT.index = timestamp_str_list
    
    return MGT


# Load pre study data
def read_pre_study_info(main_data_directory):
    PreStudyInfo = pd.read_csv(os.path.join(main_data_directory, 'participant_info', 'prestudy_info.csv'), index_col=3)
    PreStudyInfo.index = pd.to_datetime(PreStudyInfo.index)
    
    return PreStudyInfo


# Load IGTB data
def read_IGTB(main_data_directory):
    IGTB = pd.read_csv(os.path.join(main_data_directory, 'ground_truth/IGTB', 'IGTB.csv'), index_col=1)
    IGTB.index = pd.to_datetime(IGTB.index)
    
    return IGTB


# Load IGTB data
def read_Demographic(main_data_directory):
    DemoGraphic = pd.read_csv(os.path.join(main_data_directory, 'demographic', 'Demographic.csv'))
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

    UserInfo = pd.merge(IGTB, PreStudyInfo, left_on='uid', right_on='redcap_survey_identifier', how='outer')
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

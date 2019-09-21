"""
Filter the data
"""
from __future__ import print_function

import os
import sys

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser
import pandas as pd
import numpy as np
from datetime import timedelta
import pickle
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']


def main(tiles_data_path, config_path, experiment):

	# Create Config
	process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, os.pardir, 'data'))

	data_config = config.Config()
	data_config.readConfigFile(config_path, experiment)

	# Load all data path according to config file
	load_data_path.load_all_available_path(data_config, process_data_path,
										   preprocess_data_identifier='preprocess',
										   segmentation_data_identifier='segmentation',
										   filter_data_identifier='filter_data',
										   clustering_data_identifier='clustering')

	# Read ground truth data
	igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
	igtb_df = igtb_df.drop_duplicates(keep='first')
	mgt_df = load_data_basic.read_MGT(tiles_data_path)
	igtb_raw = load_data_basic.read_IGTB_Raw(tiles_data_path)

	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()

	# feat_list = ['F0_sma', 'fft', 'pcm_intensity_sma'] # pcm_loudness_sma, pcm_intensity_sma, pcm_RMSenergy_sma, pcm_zcr_sma
	num_of_sec = 6
	all_df = pd.DataFrame()

	for idx, participant_id in enumerate(top_participant_id_list[:]):

		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# Read id
		# nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
		# supervise = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].supervise[0]
		# language = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].language[0]
		# primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
		# shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
		# gender_str = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Sex[0]
		# uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]

		nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
		primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
		shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
		uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
		language = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].language[0]

		job_str = 'nurse' if nurse == 1 else 'non_nurse'
		shift_str = 'day' if shift == 'Day shift' else 'night'
		language_str = 'English' if language == 1 else 'Non-English'

		row_df = pd.DataFrame(index=[participant_id])
		if os.path.exists(os.path.join('arousal', 'global', 'num_of_sec_' + str(num_of_sec), participant_id + '.pkl')) is False:
			continue

		row_df['job'] = job_str
		row_df['shift'] = shift_str
		row_df['english'] = language_str

		for col in list(igtb_df.columns):
			row_df[col] = igtb_df.loc[uid, col]

		for col in ['nurseyears']:
			if len(str(igtb_df.loc[uid, col])) != 3 and str(igtb_df.loc[uid, col]) != ' ':
				row_df[col] = float(igtb_df.loc[uid, col])
			else:
				row_df[col] = np.nan

		for col in ['age', 'supervise']:
			if len(str(igtb_raw.loc[uid, col])) != 3:
				row_df.loc[col] = int(igtb_raw.loc[uid, col])
			else:
				row_df.loc[col] = np.nan
		all_df = all_df.append(row_df)

	nurse_df = all_df.loc[all_df['job'] == 'nurse']
	print(nurse_df.shape)

	print('Number of participant who take nurseyear survey %d' % (len(nurse_df.dropna(subset=['nurseyears']))))
	print('Average nurse year of participant %.3f std: %.3f' % (np.mean(nurse_df['nurseyears'].dropna()), np.std(nurse_df['nurseyears'].dropna())))
	print('Nurse year range of participants %d - %d' % (np.min(nurse_df['nurseyears']), np.max(nurse_df['nurseyears'])))

	print('\n')

	print('Number of participant who take age survey %d' % (len(nurse_df.dropna(subset=['age']))))
	print('Average age of participant %.3f std: %.3f' % (np.mean(nurse_df['age'].dropna()), np.std(nurse_df['age'])))
	print('Age range of participants %d - %d' % (np.min(nurse_df['age']), np.max(nurse_df['age'])))

	print('\n')

	# 20 - 29
	print('Number of participants in range between 20 - 29: %d, percentage: %.3f' % (len(nurse_df[(nurse_df['age'] >= 20) & (nurse_df['age'] < 30)]), len(nurse_df[(nurse_df['age'] >= 20) & (nurse_df['age'] < 30)]) / len(nurse_df.dropna(subset=['age'])) * 100))
	# 30 - 39
	print('Number of participants in range between 30 - 39: %d, percentage: %.3f' % (len(nurse_df[(nurse_df['age'] >= 30) & (nurse_df['age'] < 40)]), len(nurse_df[(nurse_df['age'] >= 30) & (nurse_df['age'] < 40)]) / len(nurse_df.dropna(subset=['age'])) * 100))

	# Above 40
	print('Number of participants in range above 40: %d, %.3f' % (len(nurse_df[(nurse_df['age'] >= 40) & (nurse_df['age'] < 80)]), len(nurse_df[(nurse_df['age'] >= 40) & (nurse_df['age'] < 80)]) / len(nurse_df.dropna(subset=['age']))))

	# Nurse specific
	print('Number of nurses who take Shift survey %d' % (len(nurse_df.dropna(subset=['Shift']))))
	print('Number of day shift nurse %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['Shift'] == 'Day shift']), len(nurse_df.loc[nurse_df['Shift'] == 'Day shift']) / len(nurse_df) * 100))
	print('Number of night shift nurse %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['Shift'] == 'Night shift']), len(nurse_df.loc[nurse_df['Shift'] == 'Night shift']) / len(nurse_df) * 100))

	print('\n')
	print('Number of nurses who take PrimaryUnit survey %d' % (len(nurse_df.dropna(subset=['PrimaryUnit']))))
	print('Number of ICU nurse %d' % (len(nurse_df.loc[nurse_df['PrimaryUnit'].str.contains('ICU') == True])))
	print('Number of Non-ICU shift nurse %d' % (len(nurse_df.loc[nurse_df['PrimaryUnit'].str.contains('ICU') == False])))

	print('Number of participant who take gender survey %d' % (len(nurse_df.dropna(subset=['Sex']))))
	print('Number of male participant %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['Sex'] == 'Male']), len(nurse_df.loc[nurse_df['Sex'] == 'Male']) / len(nurse_df.dropna(subset=['Sex'])) * 100))
	print('Number of female participant %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['Sex'] == 'Female']), len(nurse_df.loc[nurse_df['Sex'] == 'Female']) / len(nurse_df.dropna(subset=['Sex'])) * 100))

	print('Number of participant who take education survey %d' % (len(nurse_df.dropna(subset=['education']))))
	print('Number of participant who attend college %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['education'] > 2]), len(nurse_df.loc[nurse_df['education'] > 2]) / len(nurse_df.dropna(subset=['education'])) * 100))

	print('Number of participant who attend gradute school %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['education'] > 4]), len(nurse_df.loc[nurse_df['education'] > 4]) / len(nurse_df.dropna(subset=['education'])) * 100))
	print('Number of participant who attend gradute school %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['education'] > 6]), len(nurse_df.loc[nurse_df['education'] > 6]) / len(nurse_df.dropna(subset=['education'])) * 100))

	# Supervise
	print('Number of participant who take supervise survey %d' % (len(nurse_df.dropna(subset=['supervise']))))
	print('Number of participant who supervise others %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['supervise'] == 1]), len(nurse_df.loc[nurse_df['supervise'] == 1]) / len(nurse_df.dropna(subset=['supervise'])) * 100))

	# English
	print('Number of participant who take english survey %d' % (len(nurse_df.dropna(subset=['english']))))
	print('Number of participant who speak english %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['english'] == 'English']), len(nurse_df.loc[nurse_df['english'] == 'English']) / len(nurse_df.dropna(subset=['english'])) * 100))
	print('Number of participant who speak don\'t english %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['english'] == 'Non-English']), len(nurse_df.loc[nurse_df['english'] == 'Non-English']) / len(nurse_df.dropna(subset=['english'])) * 100))


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'audio_location' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
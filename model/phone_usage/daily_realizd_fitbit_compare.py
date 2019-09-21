"""
Filter the data
"""
from __future__ import print_function

import os
import sys
import matplotlib.pyplot as plt

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser
import numpy as np
import pandas as pd
import pickle
import preprocess
from scipy import stats
from datetime import timedelta
import collections

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']


def read_basic_df(row_df, igtb_df, participant_id):
	# Read id
	nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
	supervise = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].supervise[0]
	language = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].language[0]
	primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
	shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
	gender_str = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Sex[0]
	
	job_str = 'nurse' if nurse == 1 or nurse == 2 else 'non_nurse'
	shift_str = 'day' if shift == 'Day shift' else 'night'
	supervise_str = 'Supervise' if supervise == 1 else 'Non-Supervise'
	language_str = 'English' if language == 1 else 'Non-English'
	
	icu_str = 'non_icu'
	for unit in icu_list:
		if unit in primary_unit:
			icu_str = 'icu'
	
	if 'ICU' in primary_unit:
		icu_str = 'icu'
	
	row_df['shift'] = shift_str
	row_df['job'] = job_str
	row_df['supervise'] = supervise_str
	row_df['language'] = language_str
	row_df['icu'] = icu_str
	row_df['gender'] = gender_str


def main(tiles_data_path, config_path, experiment):
	# Create Config
	process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))

	data_config = config.Config()
	data_config.readConfigFile(config_path, experiment)

	chi_data_config = config.Config()
	chi_data_config.readChiConfigFile(config_path)

	# Load all data path according to config file
	load_data_path.load_all_available_path(data_config, process_data_path,
										   preprocess_data_identifier='preprocess',
										   segmentation_data_identifier='segmentation',
										   filter_data_identifier='filter_data',
										   clustering_data_identifier='clustering')

	load_data_path.load_chi_preprocess_path(chi_data_config, process_data_path)

	# Read ground truth data
	igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
	igtb_df = igtb_df.drop_duplicates(keep='first')
	igtb_cols = [col for col in list(igtb_df.columns) if 'igtb' in col]
	psqi_raw_igtb = load_data_basic.read_PSQI_Raw(tiles_data_path)

	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()

	realizd_col_list = ['total_time', 'mean_time', 'mean_inter', 'frequency', 'less_than_1min', 'above_1min']
	'''
	fitbit_col_list = ['heart_rate_mean', 'heart_rate_std', 'number_of_step', 'step_count_mean', 'step_count_std',
					   'resting_heart_rate', 'sleep_asleep_minute', 'sleep_in_bed_minute', 'sleep_per_day',
					   'peak_ratio', 'out_of_range_ratio', 'fat_burn_ratio', 'cardio_ratio']
	'''
	
	fitbit_col_list = ['step_count_mean', 'step_count_sum', 'sleep_in_bed_minute', 'sleep_per_day',
					   'peak_ratio', 'out_of_range_ratio', 'fat_burn_ratio', 'cardio_ratio']
	
	add_cols = ['Emotional_Wellbeing', 'Pain', 'LifeSatisfaction', 'General_Health',
				'Flexbility', 'Inflexbility', 'Perceivedstress',
				'energy_fatigue', 'energy', 'fatigue', 'Engage']
	
	if os.path.exists(os.path.join('daily_realizd_fitbit.csv.gz')) is True:
		final_df = pd.read_csv(os.path.join('daily_realizd_fitbit.csv.gz'), index_col=0)
	else:
		final_df = pd.DataFrame()
		for idx, participant_id in enumerate(top_participant_id_list[:]):
			print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
			
			process_data_path = os.path.join(data_config.phone_usage_path, participant_id + '_fitbit_phone.csv.gz')
			if os.path.exists(process_data_path) is False:
				continue
			
			# Read process data
			process_df = pd.read_csv(process_data_path, index_col=0)
			tmp_list = list(process_df.loc[process_df['num_of_minutes'] < 720].index)
			process_df.loc[tmp_list, 'num_of_minutes'] = np.nan
			tmp_list = list(process_df.loc[process_df['sleep_in_bed_minute'] == 0].index)
			process_df.loc[tmp_list, 'sleep_in_bed_minute'] = np.nan
			
			process_df = process_df[realizd_col_list + fitbit_col_list+['work']]
			
			# Read work and off data
			work_df = process_df.loc[process_df['work'] == 1]
			off_df = process_df.loc[process_df['work'] == 0]
		
			if len(process_df) < 10:
				continue
			
			row_df = pd.DataFrame(index=[participant_id])
			
			# Process features
			for col in realizd_col_list+fitbit_col_list:
				row_df[col + '_mean'] = np.nanmean(np.array(process_df[col]))
				row_df[col + '_std'] = np.nanstd(np.array(process_df[col]))
				row_df[col + '_work_mean'] = np.nanmean(np.array(work_df[col]))
				row_df[col + '_work_std'] = np.nanstd(np.array(work_df[col]))
				row_df[col + '_off_mean'] = np.nanmean(np.array(off_df[col]))
				row_df[col + '_off_std'] = np.nanstd(np.array(off_df[col]))
			
			read_basic_df(row_df, igtb_df, participant_id)
			
			# Read IGTB
			for col in igtb_cols:
				row_df[col] = igtb_df.loc[igtb_df['ParticipantID'] == participant_id][col][0]
				
			# Read additional feature
			for col in add_cols:
				data_str = str(igtb_df.loc[igtb_df['ParticipantID'] == participant_id][col][0])
				if len(data_str) == 0:
					row_df[col] = np.nan
					continue
	
				if 'a' in data_str or ' ' in data_str:
					row_df[col] = np.nan
				else:
					row_df[col] = float(data_str)
			
			final_df = final_df.append(row_df)
		
		# Save the data
		final_df.to_csv('daily_realizd_fitbit.csv.gz', compression='gzip')
	
	analysis_col_list = []
	fitbit_col_list = ['step_count_mean', 'sleep_in_bed_minute',
					   'peak_ratio', 'out_of_range_ratio', 'fat_burn_ratio', 'cardio_ratio']
	realizd_col_list = ['total_time', 'mean_time', 'mean_inter', 'less_than_1min', 'above_1min']
	
	for col in fitbit_col_list+realizd_col_list:
		analysis_col_list.append(col + '_mean')
		analysis_col_list.append(col + '_work_mean')
		analysis_col_list.append(col + '_off_mean')
	for col in list(final_df.columns):
		if 'mean' not in col and 'std' not in col:
			analysis_col_list.append(col)
	
	nurse_df = final_df.loc[final_df['job'] == 'nurse']
	compare_method_list = ['shift', 'language', 'supervise', 'gender', 'icu', 'job']
	p_df = pd.DataFrame()
	
	for compare_method in compare_method_list:
		if compare_method == 'shift':
			first_data_df = nurse_df.loc[nurse_df['shift'] == 'day']
			second_data_df = nurse_df.loc[nurse_df['shift'] == 'night']
		elif compare_method == 'supervise':
			first_data_df = nurse_df.loc[nurse_df['supervise'] == 'Supervise']
			second_data_df = nurse_df.loc[nurse_df['supervise'] == 'Non-Supervise']
		elif compare_method == 'language':
			first_data_df = nurse_df.loc[nurse_df['language'] == 'English']
			second_data_df = nurse_df.loc[nurse_df['language'] == 'Non-English']
		elif compare_method == 'gender':
			first_data_df = nurse_df.loc[nurse_df['gender'] == 'Male']
			second_data_df = nurse_df.loc[nurse_df['gender'] == 'Female']
		elif compare_method == 'job':
			first_data_df = final_df.loc[final_df['job'] == 'nurse']
			second_data_df = final_df.loc[final_df['job'] == 'non_nurse']
		else:
			first_data_df = nurse_df.loc[nurse_df['icu'] == 'icu']
			second_data_df = nurse_df.loc[nurse_df['icu'] == 'non_icu']
		
		row_df = pd.DataFrame(index=[compare_method])
		for analysis_col in analysis_col_list:
			stat_value, p_value = stats.ks_2samp(first_data_df[analysis_col].dropna(), second_data_df[analysis_col].dropna())
			row_df[analysis_col] = p_value
		p_df = p_df.append(row_df)
		
	print()
	

if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
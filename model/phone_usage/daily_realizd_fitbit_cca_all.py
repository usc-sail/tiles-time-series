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
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import rcca

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']


def read_basic_df(process_df, igtb_df, participant_id):
	# Read id
	nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
	supervise = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].supervise[0]
	language = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].language[0]
	primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
	shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
	gender_str = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Sex[0]
	children = str(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].children[0])
	age = str(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].age[0])
	
	# job_str = 'nurse' if nurse == 1 or nurse == 2 else 'non_nurse'
	job_str = 'nurse' if nurse == 1 else 'non_nurse'
	shift_str = 'day' if shift == 'Day shift' else 'night'
	supervise_str = 'Supervise' if supervise == 1 else 'Non-Supervise'
	language_str = 'English' if language == 1 else 'Non-English'
	children_data = int(children) if len(children) == 1 and children != ' ' else np.nan
	age = float(age) if age != 'nan' else np.nan
	
	icu_str = 'non_icu'
	for unit in icu_list:
		if unit in primary_unit:
			icu_str = 'icu'
	
	if 'ICU' in primary_unit:
		icu_str = 'icu'
	
	process_df['shift'] = shift_str
	process_df['job'] = job_str
	process_df['supervise'] = supervise_str
	process_df['language'] = language_str
	process_df['icu'] = icu_str
	process_df['gender'] = gender_str
	process_df['children'] = children_data
	process_df['employer_duration'] = int(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].employer_duration[0])
	process_df['age'] = age
	

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
	
	fitbit_col_list = ['step_count_mean', 'step_count_sum', 'sleep_in_bed_minute', 'sleep_per_day',
					   'peak_ratio', 'out_of_range_ratio', 'fat_burn_ratio', 'cardio_ratio']
	
	if os.path.exists(os.path.join('daily_realizd_fitbit_all.csv.gz')) is True:
		final_df = pd.read_csv(os.path.join('daily_realizd_fitbit_all.csv.gz'), index_col=0)
	else:
		final_df = pd.DataFrame()
		final_work_df = pd.DataFrame()
		final_off_df = pd.DataFrame()
		
		for idx, participant_id in enumerate(top_participant_id_list[:]):
			print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
			
			process_data_path = os.path.join(data_config.phone_usage_path, participant_id + '_fitbit_phone.csv.gz')
			if os.path.exists(process_data_path) is False:
				continue
			
			# Read process data
			process_df = pd.read_csv(process_data_path, index_col=0)
			process_df = process_df.loc[process_df['num_of_minutes'] > 1000]
			
			tmp_list = list(process_df.loc[process_df['num_of_minutes'] < 720].index)
			process_df.loc[tmp_list, 'num_of_minutes'] = np.nan
			tmp_list = list(process_df.loc[process_df['sleep_in_bed_minute'] == 0].index)
			process_df.loc[tmp_list, 'sleep_in_bed_minute'] = np.nan
			
			process_df = process_df[realizd_col_list + fitbit_col_list+['work']]
			
			if len(process_df) < 10:
				continue
			
			# tmp_df = process_df.loc[:, realizd_col_list + fitbit_col_list]
			# tmp_df = (tmp_df - tmp_df.mean()) / tmp_df.std()
			# process_df.loc[:, realizd_col_list + fitbit_col_list] = tmp_df.loc[:, realizd_col_list + fitbit_col_list]
			
			read_basic_df(process_df, igtb_df, participant_id)
			
			# Read work and off data
			work_df = process_df.loc[process_df['work'] == 1]
			off_df = process_df.loc[process_df['work'] == 0]
			
			final_df = final_df.append(process_df)
			final_work_df = final_work_df.append(work_df)
			final_off_df = final_off_df.append(off_df)
		
		# Save the data
		final_df.to_csv('daily_realizd_fitbit_all.csv.gz', compression='gzip')
	
	nurse_df = final_df.loc[final_df['job'] == 'nurse']
	compare_method_list = ['shift', 'language', 'supervise', 'gender', 'icu', 'job', 'children', 'age', 'employer_duration']
	p_df = pd.DataFrame()
	
	realizd_col_list = ['total_time', 'mean_time', 'mean_inter',
						'less_than_1min', 'above_1min']
	fitbit_col_list = ['step_count_mean', 'peak_ratio', 'out_of_range_ratio', 'fat_burn_ratio', 'cardio_ratio']

	for compare_method in compare_method_list:
		if compare_method == 'shift':
			first_data_df = nurse_df.loc[nurse_df['shift'] == 'day']
			second_data_df = nurse_df.loc[nurse_df['shift'] == 'night']
			cond_list = ['day', 'night']
		elif compare_method == 'supervise':
			first_data_df = nurse_df.loc[nurse_df['supervise'] == 'Supervise']
			second_data_df = nurse_df.loc[nurse_df['supervise'] == 'Non-Supervise']
			cond_list = ['Supervise', 'Non-Supervise']
		elif compare_method == 'language':
			first_data_df = nurse_df.loc[nurse_df['language'] == 'English']
			second_data_df = nurse_df.loc[nurse_df['language'] == 'Non-English']
			cond_list = ['English', 'Non-English']
		elif compare_method == 'gender':
			first_data_df = nurse_df.loc[nurse_df['gender'] == 'Male']
			second_data_df = nurse_df.loc[nurse_df['gender'] == 'Female']
			cond_list = ['Male', 'Female']
		elif compare_method == 'job':
			first_data_df = final_df.loc[final_df['job'] == 'nurse']
			second_data_df = final_df.loc[final_df['job'] == 'non_nurse']
			cond_list = ['nurse', 'non_nurse']
		elif compare_method == 'children':
			first_data_df = nurse_df.loc[nurse_df['children'] > 0]
			second_data_df = nurse_df.loc[nurse_df['children'] == 0]
			cond_list = ['have_child', 'no_child']
		elif compare_method == 'age':
			first_data_df = nurse_df.loc[nurse_df['age'] > np.nanmedian(nurse_df['age'])]
			second_data_df = nurse_df.loc[nurse_df['age'] <= np.nanmedian(nurse_df['age'])]
			cond_list = ['older', 'younger']
		elif compare_method == 'employer_duration':
			first_data_df = nurse_df.loc[nurse_df['employer_duration'] > np.nanmedian(nurse_df['employer_duration'])]
			second_data_df = nurse_df.loc[nurse_df['employer_duration'] <= np.nanmedian(nurse_df['employer_duration'])]
			cond_list = ['longer_employ', 'shorter_employ']
		else:
			first_data_df = nurse_df.loc[nurse_df['icu'] == 'icu']
			second_data_df = nurse_df.loc[nurse_df['icu'] == 'non_icu']
			cond_list = ['icu', 'non_icu']
		
		all_col_list = realizd_col_list + fitbit_col_list
		
		# first_data_df = first_data_df[all_col_list].dropna()
		# second_data_df = second_data_df[all_col_list].dropna()
		first_data_df = first_data_df[all_col_list + ['work']]
		second_data_df = second_data_df[all_col_list + ['work']]
		
		# first_data_df = (first_data_df - first_data_df.mean()) / first_data_df.std()
		# second_data_df = (second_data_df - second_data_df.mean()) / second_data_df.std()
		# n_components = len(realizd_col_list)
		n_components = 3
		
		col_list = [[realizd_col_list, fitbit_col_list, 'all'],
					[realizd_col_list, fitbit_col_list, 'work'],
					[realizd_col_list, fitbit_col_list, 'off']]
		row_df = pd.DataFrame(index=[cond_list[0]])
		for col in col_list:
			if col[2] == 'work':
				tmp_df = first_data_df.loc[first_data_df['work'] == 1]
				tmp_df = (tmp_df - tmp_df.mean()) / tmp_df.std()
			elif col[2] == 'off':
				tmp_df = first_data_df.loc[first_data_df['work'] == 0]
				tmp_df = (tmp_df - tmp_df.mean()) / tmp_df.std()
			else:
				tmp_df = (first_data_df - first_data_df.mean()) / first_data_df.std()
			
			tmp_df = tmp_df[col[0] + col[1]].dropna()
			
			# Create a cca object as an instantiation of the CCA object class.
			# cca = rcca.CCA(kernelcca=False, reg=1, numCC=2) # linear
			# cca = rcca.CCA(kernelcca=True, ktype='linear', reg=0.1, numCC=2)
			
			# Use the train() method to find a CCA mapping between the two training sets.
			# cca.train([np.array(tmp_df[col[0]]), np.array(tmp_df[col[1]])])
			# cca.validate([np.array(tmp_df[col[0]]), np.array(tmp_df[col[1]])])
			
			first_cca = CCA(n_components=n_components)
			# first_cca.fit(array1, array2)
			first_cca.fit(np.array(tmp_df[col[0]]), np.array(tmp_df[col[1]]))
			score = np.diag(np.corrcoef(first_cca.x_scores_, first_cca.y_scores_, rowvar=False)[:n_components, n_components:])
			for i in range(len(score)):
				row_df[col[2] + '_' + str(i)] = score[i]
			row_df[col[2] + '_' +'num'] = len(tmp_df)
		p_df = p_df.append(row_df)
		
		row_df = pd.DataFrame(index=[cond_list[1]])
		for col in col_list:
			if col[2] == 'work':
				tmp_df = second_data_df.loc[second_data_df['work'] == 1]
				tmp_df = (tmp_df - tmp_df.mean()) / tmp_df.std()
			elif col[2] == 'off':
				tmp_df = second_data_df.loc[second_data_df['work'] == 0]
				tmp_df = (tmp_df - tmp_df.mean()) / tmp_df.std()
			else:
				tmp_df = (second_data_df - second_data_df.mean()) / second_data_df.std()
			
			tmp_df = tmp_df[col[0] + col[1]].dropna()
			second_cca = CCA(n_components=n_components)
			second_cca.fit(np.array(tmp_df[col[0]]), np.array(tmp_df[col[1]]))
			score = np.diag(np.corrcoef(second_cca.x_scores_, second_cca.y_scores_, rowvar=False)[:n_components, n_components:])
			for i in range(len(score)):
				row_df[col[2] + '_' + str(i)] = score[i]
			row_df[col[2] + '_' +'num'] = len(tmp_df)
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
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
import statsmodels.api as sm

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']


def read_basic_df(row_df, process_df, igtb_df, participant_id):
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
	
	if job_str == 'non_nurse':
		if 'lab' in primary_unit or 'Lab' in primary_unit or 'pool' in primary_unit or 'tomist' in primary_unit or nurse == 3:
		#	job_str = 'lab'
		# if nurse == 4 or nurse == 5 or nurse == 6 or nurse == 7:
			job_str = 'lab'
	
	data_str = str(igtb_df.loc[igtb_df['ParticipantID'] == participant_id]['nurseyears'][0])
	if len(data_str) == 0:
		row_df['employer_duration'] = np.nan
	else:
		if 'a' in data_str or ' ' in data_str:
			row_df['employer_duration'] = np.nan
		else:
			row_df['employer_duration'] = float(data_str)
			
	row_df['shift'] = shift_str
	row_df['job'] = job_str
	row_df['supervise'] = supervise_str
	row_df['language'] = language_str
	row_df['icu'] = icu_str
	row_df['gender'] = gender_str
	row_df['children'] = children_data
	# row_df['employer_duration'] = int(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].employer_duration[0])
	row_df['age'] = age
	

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
	
	add_cols = ['Emotional_Wellbeing', 'Pain', 'LifeSatisfaction', 'General_Health',
				'Flexbility', 'Inflexbility', 'Perceivedstress', 'Social_functioning',
				'challengestressors', 'Hindrancestressors', 'WAAQ',
				'energy_fatigue', 'energy', 'fatigue', 'Engage']
	
	if os.path.exists(os.path.join('daily_realizd_fitbit_mr.csv.gz')) is True:
		final_df = pd.read_csv(os.path.join('daily_realizd_fitbit_mr.csv.gz'), index_col=0)
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
			
			read_basic_df(row_df, process_df, igtb_df, participant_id)
			
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
		final_df.to_csv('daily_realizd_fitbit_mr.csv.gz', compression='gzip')
	
	realizd_analysis_col_list = []
	realizd_col_list = ['total_time', 'less_than_1min', 'above_1min', 'mean_inter', 'step_count_mean', 'sleep_in_bed_minute'] # 'less_than_1min', 'above_1min' mean_time]
	
	for col in realizd_col_list:
		realizd_analysis_col_list.append(col + '_mean')
		
	nurse_df = final_df.loc[final_df['job'] == 'nurse']
	compare_method_list = ['supervise', 'icu', 'children', 'age']
	p_df = pd.DataFrame()
	
	for compare_method in compare_method_list:
		if compare_method == 'supervise':
			first_data_df = nurse_df.loc[nurse_df['supervise'] == 'Supervise']
			second_data_df = nurse_df.loc[nurse_df['supervise'] == 'Non-Supervise']
			mr_df = nurse_df
			cond_list = ['Supervise', 'Non-Supervise']
		elif compare_method == 'children':
			first_data_df = nurse_df.loc[final_df['children'] > 0]
			second_data_df = nurse_df.loc[final_df['children'] == 0]
			mr_df = nurse_df
			cond_list = ['have_child', 'no_child']
		elif compare_method == 'age':
			first_data_df = nurse_df.loc[nurse_df['age'] > np.nanmedian(nurse_df['age'])]
			second_data_df = nurse_df.loc[nurse_df['age'] <= np.nanmedian(nurse_df['age'])]
			mr_df = nurse_df
			cond_list = ['older', 'younger']
		else:
			first_data_df = nurse_df.loc[nurse_df['icu'] == 'icu']
			second_data_df = nurse_df.loc[nurse_df['icu'] == 'non_icu']
			mr_df = nurse_df
			cond_list = ['icu', 'non_icu']
		
		compare_cols = ['itp_igtb', 'irb_igtb', 'ocb_igtb', 'stai_igtb', 'pos_af_igtb', 'neg_af_igtb', 'Pain', 'energy_fatigue']
		
		all_col_list = realizd_analysis_col_list + compare_cols
		mr_df = mr_df[all_col_list]
		
		first_data_df = first_data_df[all_col_list]
		second_data_df = second_data_df[all_col_list]
		
		first_data_df = (first_data_df - first_data_df.mean()) / first_data_df.std()
		second_data_df = (second_data_df - second_data_df.mean()) / second_data_df.std()
		
		row_df = pd.DataFrame(index=[cond_list[0]])
		row_df['num'] = len(first_data_df[realizd_analysis_col_list].dropna())
		for col in compare_cols:
			tmp_df = first_data_df[[col] + realizd_analysis_col_list].dropna()
			x = tmp_df[realizd_analysis_col_list]
			y = tmp_df[col]
			x = sm.add_constant(x)
			model = sm.OLS(y, x).fit()
			row_df[col + '_adj'] = model.rsquared_adj
			# row_df[col + '_resid'] = model.mse_resid
			row_df[col + '_f_pvalue'] = model.f_pvalue
			
			for index in list(model.pvalues.index):
				if 'const' in index:
					continue
				row_df[col+'_'+index] = model.pvalues[index]
			
			print(model.summary())
		
		p_df = p_df.append(row_df)
	
		row_df = pd.DataFrame(index=[cond_list[1]])
		row_df['num'] = len(second_data_df[realizd_analysis_col_list].dropna())
		for col in compare_cols:
			tmp_df = second_data_df[[col] + realizd_analysis_col_list].dropna()
			x = tmp_df[realizd_analysis_col_list]
			y = tmp_df[col]
			x = sm.add_constant(x)
			model = sm.OLS(y, x).fit()
			row_df[col + '_adj'] = model.rsquared_adj
			# row_df[col + '_resid'] = model.mse_resid
			row_df[col + '_f_pvalue'] = model.f_pvalue
			
			for index in list(model.pvalues.index):
				if 'const' in index:
					continue
				row_df[col+'_'+index] = model.pvalues[index]
			
			print(model.summary())
		
		p_df = p_df.append(row_df)
		
		# first_data_df = first_data_df[all_col_list].dropna()
		# second_data_df = second_data_df[all_col_list].dropna()
		# first_data_df = first_data_df[all_col_list]
		# second_data_df = second_data_df[all_col_list]
		
		# first_data_df = (first_data_df - first_data_df.mean()) / first_data_df.std()
		# second_data_df = (second_data_df - second_data_df.mean()) / second_data_df.std()
		
		# p_df = p_df.append(row_df)
		
	print()
	

if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
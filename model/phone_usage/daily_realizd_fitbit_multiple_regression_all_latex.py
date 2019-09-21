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
	realizd_col_list = ['total_time', 'mean_time', 'mean_inter', 'less_than_1min', 'above_1min',
						'step_count_mean', 'sleep_in_bed_minute'] # 'less_than_1min', 'above_1min' ]
	
	for col in realizd_col_list:
		realizd_analysis_col_list.append(col + '_mean')
		
	nurse_df = final_df.loc[final_df['job'] == 'nurse']
	compare_method_list = ['supervise', 'age']
	
	p_df = pd.DataFrame()
	mr_df = nurse_df.copy()
	
	for compare_method in compare_method_list:
		if compare_method == 'supervise':
			first_data_df = nurse_df.loc[nurse_df['supervise'] == 'Supervise']
			second_data_df = nurse_df.loc[nurse_df['supervise'] == 'Non-Supervise']
		else:
			first_data_df = nurse_df.loc[nurse_df['age'] > np.nanmedian(nurse_df['age'])]
			second_data_df = nurse_df.loc[nurse_df['age'] <= np.nanmedian(nurse_df['age'])]
			
		mr_df[compare_method] = np.nan
		mr_df.loc[list(first_data_df.index), compare_method] = 1
		mr_df.loc[list(second_data_df.index), compare_method] = -1
	
	compare_cols = ['itp_igtb', 'irb_igtb', 'ocb_igtb', 'stai_igtb', 'pos_af_igtb', 'neg_af_igtb', 'Pain', 'energy_fatigue']
	
	add_cols = ['Emotional_Wellbeing', 'Pain', 'LifeSatisfaction', 'General_Health',
				'Flexbility', 'Inflexbility', 'Perceivedstress', 'Social_functioning',
				'challengestressors', 'Hindrancestressors', 'WAAQ',
				'energy_fatigue', 'energy', 'fatigue', 'Engage']
	
	# compare_cols = igtb_cols + add_cols
	
	all_col_list = realizd_analysis_col_list + compare_cols + compare_method_list
	norm_mr_df = mr_df[all_col_list]
	norm_mr_df = (norm_mr_df - norm_mr_df.mean()) / norm_mr_df.std()
	
	'''
	for compare_method in compare_method_list:
		norm_mr_df.loc[list(mr_df.index), compare_method] = mr_df.loc[list(mr_df.index), compare_method]
	'''
	
	for col in compare_cols:
		
		row_df = pd.DataFrame(index=[col])
		
		tmp_df = norm_mr_df[[col] + realizd_analysis_col_list + compare_method_list].dropna()
		x = tmp_df[realizd_analysis_col_list + compare_method_list]
		y = tmp_df[col]
		x = sm.add_constant(x)
		model = sm.OLS(y, x).fit()
		row_df['adj'] = model.rsquared_adj
		# row_df[col + '_resid'] = model.mse_resid
		row_df['f_pvalue'] = model.f_pvalue
		
		for index in list(model.pvalues.index):
			if 'const' in index:
				continue
			row_df[index + '_p'] = model.pvalues[index]
		
		for index in list(model.params.index):
			if 'const' in index:
				continue
			row_df[index] = model.params[index]
			
		
		print(model.summary())
		p_df = p_df.append(row_df)
	
	p_df = p_df.transpose()
	
	'''
	compare_dict = {'itp_igtb': 'ITP', 'irb_igtb': 'IRB', 'ocb_igtb': 'OCB', 'stai_igtb': 'Anxiety',
					'pos_af_igtb', 'neg_af_igtb', 'Pain',
					'energy_fatigue'}
	'''
	
	row_dict = {'adj': 'Adj. R', 'f_pvalue': 'p-value',
				'total_time_mean': 'Total Usage Time',
				'less_than_1min_mean': 'Session Freq. (<1min)',
				'above_1min_mean': 'Session Freq. (>1min)',
				'mean_inter_mean': 'Average Inter-session Time',
	 			'mean_time_mean': 'Average Session Time',
				'step_count_mean_mean': 'Average Step per Min.',
			    'sleep_in_bed_minute_mean': 'Total Sleep Time',
			    'supervise': 'Management',
			    'age': 'Age'}
	
	for row_index in list(p_df.index):
		
		if row_index == 'f_pvalue' or '_p' in row_index:
			continue
		
		print()
		print()
		print('\multicolumn{1}{l}{' + row_dict[row_index] + '} &')
		
		for i, col_index in enumerate(list(p_df.columns)):
			val = p_df.loc[row_index, col_index]
			
			if row_index == 'adj':
				p = p_df.loc['f_pvalue', col_index]
				
				if i == len(p_df.columns) - 1:
					if p < 0.01:
						print('\multicolumn{1}{|c}{$%.3f^{**}$} \\rule{0pt}{2.25ex} \\\\' % (val))
					elif p < 0.05:
						print('\multicolumn{1}{|c}{$%.3f^{*}$} \\rule{0pt}{2.25ex} \\\\' % (val))
					else:
						print('\multicolumn{1}{|c}{$%.3f$} \\rule{0pt}{2.25ex} \\\\' % (val))
				else:
					if p < 0.01:
						print('\multicolumn{1}{|c}{$%.3f^{**}$} &' % (val))
					elif p < 0.05:
						print('\multicolumn{1}{|c}{$%.3f^{*}$} &' % (val))
					else:
						print('\multicolumn{1}{|c}{$%.3f$} &' % (val))
			else:
				p = p_df.loc[row_index + '_p', col_index]
				if i == len(p_df.columns) - 1:
					if p < 0.01:
						print('\multicolumn{1}{|c}{${%.3f}^{**}$} \\rule{0pt}{2.25ex} \\\\' % (val))
					elif p < 0.05:
						print('\multicolumn{1}{|c}{${%.3f}^{*}$} \\rule{0pt}{2.25ex} \\\\' % (val))
					elif p < 0.1:
						print('\multicolumn{1}{|c}{${%.3f}$} \\rule{0pt}{2.25ex} \\\\' % (val))
					else:
						print('\multicolumn{1}{|c}{} \\rule{0pt}{2.25ex} \\\\')
				else:
					if p < 0.01:
						print('\multicolumn{1}{|c}{$\mathbf{%.3f}^{**}$} &' % (val))
					elif p < 0.05:
						print('\multicolumn{1}{|c}{$\mathbf{%.3f}^{*}$} &' % (val))
					elif p < 0.1:
						print('\multicolumn{1}{|c}{$%.3f$)} &' % (val))
					else:
						print('\multicolumn{1}{|c}{} &')
				
				'''
				if i == len(p_df.columns) - 1:
					if val < 0.001:
						print('\multicolumn{1}{|c}{\mathbf{<0.001}} \\rule{0pt}{2.25ex} \\\\')
					elif val < 0.05:
						print('\multicolumn{1}{|c}{$\mathbf{%.3f}$} \\rule{0pt}{2.25ex} \\\\' % (val))
					elif val < 0.15:
						print('\multicolumn{1}{|c}{${%.3f}$} \\rule{0pt}{2.25ex} \\\\' % (val))
					else:
						print('\multicolumn{1}{|c}{} \\rule{0pt}{2.25ex} \\\\')
				else:
					if val < 0.001:
						print('\multicolumn{1}{|c}{$\mathbf{<0.001}$} &')
					elif val < 0.05:
						print('\multicolumn{1}{|c}{$\mathbf{%.3f}$} &' % (val))
					elif val < 0.15:
						print('\multicolumn{1}{|c}{$%.3f$)} &' % (val))
					else:
						print('\multicolumn{1}{|c}{} &')
				'''

	print()
	

if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
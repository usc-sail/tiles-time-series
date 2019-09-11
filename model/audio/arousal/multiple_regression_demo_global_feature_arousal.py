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
from matplotlib import rc, font_manager
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")


icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']
ana_igtb_cols_list = ['itp_igtb', 'irb_igtb', 'iod_id_igtb', 'iod_od_igtb', 'ocb_igtb', 'pos_af_igtb', 'neg_af_igtb', 'stai_igtb']


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
	igtb_cols = [col for col in list(igtb_df.columns) if 'igtb' in col]
	mgt_df = load_data_basic.read_MGT(tiles_data_path)

	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()

	# feat_list = ['F0_sma', 'fft', 'pcm_intensity_sma'] # pcm_loudness_sma, pcm_intensity_sma, pcm_RMSenergy_sma, pcm_zcr_sma
	num_of_sec = 4
	# compare_list = ['shift', 'icu', 'gender', 'supervise', 'language']
	compare_list = ['shift', 'icu', 'supervise']
	feat_type = 'ratio'

	data_df = pd.DataFrame()


	feat_cols = ['pitch_arousal_ratio_mean', 'intensity_arousal_ratio_mean', 'speak_rate_mean',
	             'pitch_arousal_ratio_std', 'intensity_arousal_ratio_std', 'speak_rate_std']

	# feat_cols = ['arousal_ratio_mean', 'speak_rate_mean', 'arousal_ratio_std', 'speak_rate_std']

	# feat_cols = []
	for i in range(num_of_sec):
		feat_cols.append('pitch_arousal_ratio_' + str(i))
		feat_cols.append('intensity_arousal_ratio_' + str(i))
		# feat_cols.append('arousal_ratio_' + str(i))
		feat_cols.append('speak_rate_' + str(i))


	for idx, participant_id in enumerate(top_participant_id_list[:]):

		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# Read id
		nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
		supervise = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].supervise[0]
		language = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].language[0]
		primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
		shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
		gender_str = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Sex[0]

		job_str = 'nurse' if nurse == 1 else 'non_nurse'
		shift_str = 'Day Shift' if shift == 'Day shift' else 'Night Shift'
		supervise_str = 'Nurse Manager' if supervise == 1 else 'Non-nurse Manager'
		language_str = 'English' if language == 1 else 'Not English'

		if job_str == 'non_nurse':
			continue

		icu_str = 'Non-ICU'
		for unit in icu_list:
			if unit in primary_unit:
				icu_str = 'ICU'

		if 'ICU' in primary_unit:
			icu_str = 'ICU'

		if os.path.exists(os.path.join('arousal', 'global', 'num_of_sec_' + str(num_of_sec), participant_id + '.pkl')) is False:
			continue

		arousal_loc_dict = np.load(os.path.join('arousal', 'global', 'num_of_sec_' + str(num_of_sec), participant_id + '.pkl'), allow_pickle=True)
		daily_dict, loc_dict, global_dict = arousal_loc_dict['daily'], arousal_loc_dict['loc'], arousal_loc_dict['global']

		row_df = pd.DataFrame(index=[participant_id])
		row_df['job'] = job_str
		row_df['icu'] = icu_str
		row_df['shift'] = shift_str
		row_df['supervise'] = supervise_str
		row_df['language'] = language_str
		row_df['gender'] = gender_str

		for i in range(num_of_sec):
			row_df['pitch_arousal_ratio_' + str(i)] = global_dict[i]['pitch_arousal_ratio']
			row_df['intensity_arousal_ratio_' + str(i)] = global_dict[i]['intensity_arousal_ratio']
			row_df['arousal_ratio_' + str(i)] = global_dict[i]['arousal_ratio']
			row_df['speak_rate_' + str(i)] = global_dict[i]['speak_rate']

		pitch_array, intensity_array, speak_rate_array = np.zeros([1, num_of_sec]), np.zeros([1, num_of_sec]) , np.zeros([1, num_of_sec])
		arousal_array = np.zeros([1, num_of_sec])

		for i in range(num_of_sec):
			pitch_array[0, i] = global_dict[i]['pitch_arousal_ratio']
			intensity_array[0, i] = global_dict[i]['intensity_arousal_ratio']
			arousal_array[0, i] = global_dict[i]['arousal_ratio']
			speak_rate_array[0, i] = global_dict[i]['speak_rate']

		row_df['pitch_arousal_ratio_mean'] = np.nanmean(pitch_array)
		row_df['intensity_arousal_ratio_mean'] = np.nanmean(intensity_array)
		row_df['arousal_ratio_mean'] = np.nanmean(arousal_array)
		row_df['speak_rate_mean'] = np.nanmean(speak_rate_array)

		row_df['pitch_arousal_ratio_std'] = np.nanstd(pitch_array)
		row_df['intensity_arousal_ratio_std'] = np.nanstd(intensity_array)
		row_df['arousal_ratio_std'] = np.nanstd(arousal_array)
		if np.nanstd(speak_rate_array) == 0:
			row_df['speak_rate_std'] = np.nan
		else:
			row_df['speak_rate_std'] = np.nanstd(speak_rate_array)

		for col in ana_igtb_cols_list:
			row_df[col] = igtb_df.loc[igtb_df['ParticipantID'] == participant_id][col][0]
		data_df = data_df.append(row_df)

	data_df = data_df.fillna(data_df.mean())

	rsquared_df = pd.DataFrame()
	p_df = pd.DataFrame()

	nurse_df = data_df.loc[data_df['job'] == 'nurse']
	for compare_method in compare_list:

		if compare_method == 'shift':
			first_data_df = nurse_df.loc[nurse_df['shift'] == 'Day Shift']
			second_data_df = nurse_df.loc[nurse_df['shift'] == 'Night Shift']
			nurse_df.loc[list(first_data_df.index), compare_method] = 1
			nurse_df.loc[list(second_data_df.index), compare_method] = -1
			cond_list = ['day', 'night']
			title_list = ['Day Shift Nurse', 'Night Shift Nurse']
		elif compare_method == 'supervise':
			first_data_df = nurse_df.loc[nurse_df['supervise'] == 'Nurse Manager']
			second_data_df = nurse_df.loc[nurse_df['supervise'] == 'Non-nurse Manager']
			nurse_df.loc[list(first_data_df.index), compare_method] = 1
			nurse_df.loc[list(second_data_df.index), compare_method] = -1
			cond_list = ['Supervise', 'Non-Supervise']
			title_list = ['Nurse Manager', 'Non-Nurse Manager']
		elif compare_method == 'language':
			first_data_df = nurse_df.loc[nurse_df['language'] == 'English']
			second_data_df = nurse_df.loc[nurse_df['language'] == 'Not English']
			nurse_df.loc[list(first_data_df.index), compare_method] = 1
			nurse_df.loc[list(second_data_df.index), compare_method] = -1
			cond_list = ['English', 'Non-English']
			title_list = ['Native Language: English', 'Native Language: Non-English']
		elif compare_method == 'gender':
			first_data_df = nurse_df.loc[nurse_df['gender'] == 'Male']
			second_data_df = nurse_df.loc[nurse_df['gender'] == 'Female']
			nurse_df.loc[list(first_data_df.index), compare_method] = 1
			nurse_df.loc[list(second_data_df.index), compare_method] = -1
			cond_list = ['Male', 'Female']
			title_list = ['Male Nurse', 'Female Nurse']
		else:
			first_data_df = nurse_df.loc[nurse_df['icu'] == 'ICU']
			second_data_df = nurse_df.loc[nurse_df['icu'] == 'Non-ICU']
			nurse_df.loc[list(first_data_df.index), compare_method] = 1
			nurse_df.loc[list(second_data_df.index), compare_method] = -1
			cond_list = ['icu', 'non_icu']
			title_list = ['ICU Nurse', 'Non-ICU Nurse']

	remain_cols = compare_list + ana_igtb_cols_list

	result_dict = {}
	unique_data_cols = []
	for i in range(4):
		if i == 4:
			print()

		else:
			data_cols = []
			final_df = pd.DataFrame()
			for col in list(data_df.columns):
				if str(i) in col:
					data_cols.append(col)
					unique_data_cols.append(col)

			norm_mr_df = nurse_df[data_cols + remain_cols].dropna()
			norm_mr_df = (norm_mr_df - norm_mr_df.mean()) / norm_mr_df.std()

			for col in ana_igtb_cols_list:

				row_df = pd.DataFrame(index=[col])

				mr_df = norm_mr_df[[col] + compare_list + data_cols].dropna()
				y = mr_df[col]
				x = mr_df[data_cols+compare_list]
				x = sm.add_constant(x)
				model = sm.OLS(y, x).fit()
				row_df['adj'] = model.rsquared_adj
				row_df['f_value'] = model.f_pvalue

				for index in list(model.pvalues.index):
					if 'const' in index:
						continue
					row_df[index+'_p'] = model.pvalues[index]

				for index in list(model.params.index):
					if 'const' in index:
						continue
					row_df[index+'_param'] = model.params[index]

				final_df = final_df.append(row_df)

			result_dict[i] = final_df

	norm_mr_df = nurse_df[unique_data_cols + remain_cols].dropna()
	norm_mr_df = (norm_mr_df - norm_mr_df.mean()) / norm_mr_df.std()

	final_df = pd.DataFrame()
	for col in ana_igtb_cols_list:

		row_df = pd.DataFrame(index=[col])

		mr_df = norm_mr_df[[col] + compare_list + unique_data_cols].dropna()
		y = mr_df[col]
		x = mr_df[unique_data_cols + compare_list]
		x = sm.add_constant(x)
		model = sm.OLS(y, x).fit()
		row_df['adj'] = model.rsquared_adj
		row_df['f_value'] = model.f_pvalue

		for index in list(model.pvalues.index):
			if 'const' in index:
				continue
			row_df[index + '_p'] = model.pvalues[index]

		for index in list(model.params.index):
			if 'const' in index:
				continue
			row_df[index + '_param'] = model.params[index]

		final_df = final_df.append(row_df)

	result_dict[4] = final_df

	'''
	Based on feature
	'''
	# result_dict = {}
	feature_list = ['pitch', 'intensity', 'arousal_ratio', 'speak_rate']
	for feature_col in feature_list:
		data_cols = []
		final_df = pd.DataFrame()
		for col in list(data_df.columns):
			if col in remain_cols or col in compare_list:
				continue

			if feature_col == 'pitch' or feature_col == 'intensity':
				if feature_col in col:
					data_cols.append(col)
			elif feature_col in col and 'pitch' not in col and 'intensity' not in col:
				data_cols.append(col)

		norm_mr_df = nurse_df[data_cols + remain_cols].dropna()
		norm_mr_df = (norm_mr_df - norm_mr_df.mean()) / norm_mr_df.std()

		for col in ana_igtb_cols_list:

			row_df = pd.DataFrame(index=[col])

			mr_df = norm_mr_df[[col] + compare_list + data_cols].dropna()
			y = mr_df[col]
			x = mr_df[data_cols + compare_list]
			x = sm.add_constant(x)
			model = sm.OLS(y, x).fit()
			row_df['adj'] = model.rsquared_adj
			row_df['f_value'] = model.f_pvalue

			for index in list(model.pvalues.index):
				if 'const' in index:
					continue
				row_df[index + '_p'] = model.pvalues[index]

			for index in list(model.params.index):
				if 'const' in index:
					continue
				row_df[index + '_param'] = model.params[index]

			final_df = final_df.append(row_df)

		result_dict[feature_col] = final_df

	# 'speak_rate_0', 'arousal_ratio_1', 'intensity_ratio_1', 'speak_ratio_1', 'speak_ratio_2', 'arousal_ratio_2'

	data_cols = ['speak_rate_0', 'arousal_ratio_1', 'speak_rate_2']
	norm_mr_df = nurse_df[data_cols + remain_cols].dropna()
	norm_mr_df = (norm_mr_df - norm_mr_df.mean()) / norm_mr_df.std()

	final_df = pd.DataFrame()
	for col in ana_igtb_cols_list:

		row_df = pd.DataFrame(index=[col])

		mr_df = norm_mr_df[[col] + compare_list + data_cols].dropna()
		y = mr_df[col]
		x = mr_df[data_cols + compare_list]
		x = sm.add_constant(x)
		model = sm.OLS(y, x).fit()
		row_df['adj'] = model.rsquared_adj
		row_df['f_value'] = model.f_pvalue

		for index in list(model.pvalues.index):
			if 'const' in index:
				continue
			row_df[index + '_p'] = model.pvalues[index]

		for index in list(model.params.index):
			if 'const' in index:
				continue
			row_df[index + '_param'] = model.params[index]

		final_df = final_df.append(row_df)

	result_dict['selected'] = final_df

	rsquared_df.to_csv(os.path.join('global.csv.gz'), compression='gzip')
	print()


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'audio_location' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
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
import numpy as np
import pandas as pd

import statsmodels.api as sm


row_cols = ['Total Usage Time', 'Mean Session Length', 'Mean Inter-session Time',
			'Session Frequency', 'Session Frequency (<1min)', 'Session Frequency (>1min)']

col_cols = ['Shipley Abs.', 'Shipley Voc.',
			'Anxiety', 'Pos. Affect', 'Neg. Affect',
			'Neuroticism', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Openness', 'Total PSQI']


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

	if os.path.exists(os.path.join(data_config.phone_usage_path, 'daily_summary.csv.gz')) is True:
		all_df = pd.read_csv(os.path.join(data_config.phone_usage_path, 'daily_summary.csv.gz'), index_col=0)

		for participant_id in list(all_df.index):
			nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
			primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
			shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
			job_str = 'nurse' if nurse == 1 else 'non_nurse'
			shift_str = 'day' if shift == 'Day shift' else 'night'

			all_df.loc[participant_id, 'job'] = job_str
			all_df.loc[participant_id, 'shift'] = shift_str

		nurse_df = all_df.loc[all_df['job'] == 'nurse']
		day_nurse_df = nurse_df.loc[nurse_df['shift'] == 'day']
		night_nurse_df = nurse_df.loc[nurse_df['shift'] == 'night']

		final_cols = []
		for col in list(nurse_df.columns):
			if 'work' not in col and 'off' not in col:
				final_cols.append(col)

		ana_igtb_cols = ['shipley_abs_igtb', 'shipley_voc_igtb', 'stai_igtb', 'pos_af_igtb', 'neg_af_igtb',
						 'neu_igtb', 'con_igtb', 'ext_igtb', 'agr_igtb', 'ope_igtb', 'psqi_igtb']

		ana_igtb_dict = {'shipley_abs_igtb': 'Shipley Abs.', 'shipley_voc_igtb': 'Shipley Voc.',
						 'stai_igtb': 'Anxiety', 'psqi_igtb': 'Total PSQI',
						 'pos_af_igtb': 'Pos. Affect', 'neg_af_igtb': 'Neg. Affect',
						 'neu_igtb': 'Neuroticism', 'con_igtb': 'Conscientiousness',
						 'ext_igtb': 'Extraversion', 'agr_igtb': 'Agreeableness', 'ope_igtb': 'Openness'}

		plot_df = pd.DataFrame()
		for i in range(len(nurse_df)):
			participant_df = nurse_df.iloc[i, :]
			participant_id = list(nurse_df.index)[i]
			shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
			row_df = pd.DataFrame(index=[participant_id])
			row_df['Total Usage Time'] = participant_df['total_time_mean']
			row_df['Mean Session Length'] = participant_df['mean_time_mean']
			row_df['Session Frequency'] = participant_df['frequency_mean']
			row_df['Mean Inter-session Time'] = participant_df['mean_inter_mean']
			row_df['Session Frequency (<1min)'] = participant_df['less_than_1min_mean']
			row_df['Session Frequency (>1min)'] = participant_df['above_1min_mean']
			row_df['Data Type'] = 'Combined'
			row_df['Shift Type'] = shift

			for col in ana_igtb_cols:
				row_df[ana_igtb_dict[col]] = participant_df[col]

			plot_df = plot_df.append(row_df)

			row_df = pd.DataFrame(index=[participant_id + 'work'])
			row_df['Total Usage Time'] = participant_df['work_total_time_mean']
			row_df['Mean Session Length'] = participant_df['work_mean_time_mean']
			row_df['Session Frequency'] = participant_df['work_frequency_mean']
			row_df['Mean Inter-session Time'] = participant_df['work_mean_inter_mean']
			row_df['Session Frequency (<1min)'] = participant_df['work_less_than_1min_mean']
			row_df['Session Frequency (>1min)'] = participant_df['work_above_1min_mean']
			row_df['Data Type'] = 'Workday'
			row_df['Shift Type'] = shift

			for col in ana_igtb_cols:
				row_df[ana_igtb_dict[col]] = participant_df[col]

			plot_df = plot_df.append(row_df)

			row_df = pd.DataFrame(index=[participant_id + 'off'])
			row_df['Total Usage Time'] = participant_df['off_total_time_mean']
			row_df['Mean Session Length'] = participant_df['off_mean_time_mean']
			row_df['Session Frequency'] = participant_df['off_frequency_mean']
			row_df['Mean Inter-session Time'] = participant_df['off_mean_inter_mean']
			row_df['Session Frequency (<1min)'] = participant_df['off_less_than_1min_mean']
			row_df['Session Frequency (>1min)'] = participant_df['off_above_1min_mean']
			row_df['Data Type'] = 'Off-day'
			row_df['Shift Type'] = shift

			for col in ana_igtb_cols:
				row_df[ana_igtb_dict[col]] = participant_df[col]

			plot_df = plot_df.append(row_df)

		day_df = plot_df.loc[plot_df['Shift Type'] == 'Day shift']
		night_df = plot_df.loc[plot_df['Shift Type'] == 'Night shift']

		work_day_df, work_night_df = day_df.loc[day_df['Data Type'] == 'Workday'], night_df.loc[night_df['Data Type'] == 'Workday']
		off_day_df, off_night_df = day_df.loc[day_df['Data Type'] == 'Off-day'], night_df.loc[night_df['Data Type'] == 'Off-day']

		work_df, off_df = plot_df.loc[plot_df['Data Type'] == 'Workday'], plot_df.loc[plot_df['Data Type'] == 'Off-day']

		feat_cols = ['Total Usage Time', 'Mean Session Length', 'Session Frequency', 'Mean Inter-session Time', 'Session Frequency (<1min)', 'Session Frequency (>1min)']
		feat_cols = ['Total Usage Time', 'Mean Session Length', 'Session Frequency (<1min)', 'Session Frequency (>1min)']
		
		# 'Session Frequency', 'Mean Inter-session Time',
		
		rsquared_df = pd.DataFrame()
		day_pvalue_df, night_pvalue_df, overall_pvalue_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
		
		norm = 'z_norm'
		
		for col in ana_igtb_cols:
			
			row_df = pd.DataFrame(index=[col])
			day_row_df = pd.DataFrame(index=[col])
			night_row_df = pd.DataFrame(index=[col])
			overall_row_df = pd.DataFrame(index=[col])
			
			# day shift nurse
			# work day
			data_df_work = work_day_df[[ana_igtb_dict[col]] + feat_cols].dropna()
			# off day
			data_df_off = off_day_df[[ana_igtb_dict[col]] + feat_cols].dropna()
			
			participant_list = [index[:-3] for index in list(data_df_off.index)][:]
			print(len(participant_list))
			
			final_df = pd.DataFrame(index=participant_list)
			feat_final_cols = []
			for participant_id in participant_list:
				final_df.loc[participant_id, ana_igtb_dict[col]] = data_df_work.loc[participant_id + 'work', ana_igtb_dict[col]]
				for feat_col in feat_cols:
					feat_final_cols.append(feat_col + '_work')
					feat_final_cols.append(feat_col + '_off')
					
					# if 'Total' in feat_col:
					#	final_df.loc[participant_id, feat_col + '_work'] = np.log(data_df_work.loc[participant_id + 'work', feat_col])
					#	final_df.loc[participant_id, feat_col + '_off'] = np.log(data_df_off.loc[participant_id + 'off', feat_col])
					# else:
					final_df.loc[participant_id, feat_col + '_work'] = data_df_work.loc[participant_id+'work', feat_col]
					final_df.loc[participant_id, feat_col + '_off'] = data_df_off.loc[participant_id+'off', feat_col]
			
			feat_final_cols = list(set(feat_final_cols))
			y = final_df[[ana_igtb_dict[col]]]
			x = final_df[feat_final_cols].copy()
			if norm == 'z_norm':
				x = (x - x.mean()) / x.std()
			elif norm == 'min_max':
				x = (x - x.min()) / (x.max() - x.min())
			x = sm.add_constant(x)  # adding a constant
			model = sm.OLS(y, x).fit()
			day_row_df['rsquared_day'] = model.rsquared_adj
			day_row_df['mse_resid_day'] = model.mse_resid
			
			for p_col in list(model.pvalues.index):
				day_row_df[p_col] = model.pvalues[p_col]
			day_pvalue_df = day_pvalue_df.append(day_row_df)
			print(model.summary())
			
			# night shift nurse
			# work day
			data_df_work = work_night_df[[ana_igtb_dict[col]] + feat_cols].dropna()
			# off day
			data_df_off = off_night_df[[ana_igtb_dict[col]] + feat_cols].dropna()
			
			participant_list = [index[:-3] for index in list(data_df_off.index)]
			final_df = pd.DataFrame(index=participant_list)
			for participant_id in participant_list:
				final_df.loc[participant_id, ana_igtb_dict[col]] = data_df_work.loc[
					participant_id + 'work', ana_igtb_dict[col]]
				for feat_col in feat_cols:
					final_df.loc[participant_id, feat_col + '_work'] = data_df_work.loc[
						participant_id + 'work', feat_col]
					final_df.loc[participant_id, feat_col + '_off'] = data_df_off.loc[participant_id + 'off', feat_col]
			
			y = final_df[[ana_igtb_dict[col]]]
			x = final_df[feat_final_cols].copy()
			if norm == 'z_norm':
				x = (x - x.mean()) / x.std()
			elif norm == 'min_max':
				x = (x - x.min()) / (x.max() - x.min())
			x = sm.add_constant(x)  # adding a constant
			model = sm.OLS(y, x).fit()
			night_row_df['rsquared_night'] = model.rsquared_adj
			night_row_df['mse_resid_night'] = model.mse_resid
			for p_col in list(model.pvalues.index):
				night_row_df[p_col] = model.pvalues[p_col]
			night_pvalue_df = night_pvalue_df.append(night_row_df)
			print(model.summary())
			
			# overall shift nurse
			# work day
			data_df_work = work_df[[ana_igtb_dict[col]] + feat_cols].dropna()
			# off day
			data_df_off = off_df[[ana_igtb_dict[col]] + feat_cols].dropna()
			
			participant_list = [index[:-3] for index in list(data_df_off.index)]
			final_df = pd.DataFrame(index=participant_list)
			for participant_id in participant_list:
				final_df.loc[participant_id, ana_igtb_dict[col]] = data_df_work.loc[participant_id + 'work', ana_igtb_dict[col]]
				for feat_col in feat_cols:
					final_df.loc[participant_id, feat_col + '_work'] = data_df_work.loc[participant_id + 'work', feat_col]
					final_df.loc[participant_id, feat_col + '_off'] = data_df_off.loc[participant_id + 'off', feat_col]
			
			y = final_df[[ana_igtb_dict[col]]]
			x = final_df[feat_final_cols].copy()
			if norm == 'z_norm':
				x = (x - x.mean()) / x.std()
			elif norm == 'min_max':
				x = (x - x.min()) / (x.max() - x.min())
			x = sm.add_constant(x)  # adding a constant
			model = sm.OLS(y, x).fit()
			overall_row_df['rsquared_overall'] = model.rsquared_adj
			overall_row_df['mse_resid_overall'] = model.mse_resid
			
			for p_col in list(model.pvalues.index):
				overall_row_df[p_col] = model.pvalues[p_col]
			overall_pvalue_df = overall_pvalue_df.append(overall_row_df)
			print(model.summary())
			# rsquared_df = rsquared_df.append(row_df)
		
		# rsquared_df.to_csv(os.path.join('rsquared_df_all.csv'))
		day_pvalue_df.to_csv(os.path.join('day_p_value.csv'))
		night_pvalue_df.to_csv(os.path.join('night_p_value.csv'))
		overall_pvalue_df.to_csv(os.path.join('overall_p_value.csv'))
		
		rsquared_df = pd.DataFrame()
		for col in ana_igtb_cols:
			row_df = pd.DataFrame(index=[col])

			# predictions = model.predict(X)
			# model.summary()
			data_df = work_day_df[[ana_igtb_dict[col]]+feat_cols].dropna()
			y = data_df[[ana_igtb_dict[col]]]
			x = data_df[feat_cols].copy()
			if norm == 'z_norm':
				x = (x - x.mean()) / x.std()
			elif norm == 'min_max':
				x = (x - x.min()) / (x.max() - x.min())
			x = sm.add_constant(x)  # adding a constant
			model = sm.OLS(y, x).fit()
			row_df['rsquared_day_work'] = model.rsquared_adj
			print(model.summary())

			# work night
			data_df = work_night_df[[ana_igtb_dict[col]] + feat_cols].dropna()
			y = data_df[[ana_igtb_dict[col]]]
			x = data_df[feat_cols].copy()
			if norm == 'z_norm':
				x = (x - x.mean()) / x.std()
			elif norm == 'min_max':
				x = (x - x.min()) / (x.max() - x.min())
			x = sm.add_constant(x)  # adding a constant
			model = sm.OLS(y, x).fit()
			row_df['rsquared_night_work'] = model.rsquared_adj

			# off day
			data_df = off_day_df[[ana_igtb_dict[col]] + feat_cols].dropna()
			y = data_df[[ana_igtb_dict[col]]]
			x = data_df[feat_cols].copy()
			if norm == 'z_norm':
				x = (x - x.mean()) / x.std()
			elif norm == 'min_max':
				x = (x - x.min()) / (x.max() - x.min())
			x = sm.add_constant(x)  # adding a constant
			model = sm.OLS(y, x).fit()
			row_df['rsquared_day_off'] = model.rsquared_adj

			# off night
			data_df = off_night_df[[ana_igtb_dict[col]] + feat_cols].dropna()
			y = data_df[[ana_igtb_dict[col]]]
			x = data_df[feat_cols].copy()
			if norm == 'z_norm':
				x = (x - x.mean()) / x.std()
			elif norm == 'min_max':
				x = (x - x.min()) / (x.max() - x.min())
			x = sm.add_constant(x)  # adding a constant
			model = sm.OLS(y, x).fit()
			row_df['rsquared_night_off'] = model.rsquared_adj

			# work
			data_df = work_df[[ana_igtb_dict[col]] + feat_cols].dropna()
			y = data_df[[ana_igtb_dict[col]]]
			x = data_df[feat_cols].copy()
			if norm == 'z_norm':
				x = (x - x.mean()) / x.std()
			elif norm == 'min_max':
				x = (x - x.min()) / (x.max() - x.min())
			x = sm.add_constant(x)  # adding a constant
			model = sm.OLS(y, x).fit()
			row_df['rsquared_work'] = model.rsquared_adj
			
			# off
			data_df = off_df[[ana_igtb_dict[col]] + feat_cols].dropna()
			y = data_df[[ana_igtb_dict[col]]]
			x = data_df[feat_cols].copy()
			if norm == 'z_norm':
				x = (x - x.mean()) / x.std()
			elif norm == 'min_max':
				x = (x - x.min()) / (x.max() - x.min())
			x = sm.add_constant(x)  # adding a constant
			model = sm.OLS(y, x).fit()
			row_df['rsquared_off'] = model.rsquared_adj

			# overall
			data_df = plot_df[[ana_igtb_dict[col]] + feat_cols].dropna()
			y = data_df[[ana_igtb_dict[col]]]
			x = data_df[feat_cols].copy()
			if norm == 'z_norm':
				x = (x - x.mean()) / x.std()
			elif norm == 'min_max':
				x = (x - x.min()) / (x.max() - x.min())
			x = sm.add_constant(x)  # adding a constant
			model = sm.OLS(y, x).fit()
			row_df['rsquared_all'] = model.rsquared_adj

			rsquared_df = rsquared_df.append(row_df)

		rsquared_df.to_csv(os.path.join('rsquared_df.csv'))
		print()


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
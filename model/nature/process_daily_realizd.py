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


def process_daily_realizd(data_config, data_df, days_at_work_df, igtb_df, participant_id):

	process_df = pd.DataFrame()

	if len(data_df) < 700 or len(days_at_work_df) < 5:
		return None

	days = (pd.to_datetime(data_df.index[-1]).date() - pd.to_datetime(data_df.index[0]).date()).days

	start_str = pd.to_datetime(data_df.index[0]).date()
	end_str = pd.to_datetime(data_df.index[-1]).date()
	for i in range(days):

		date_start_str = (pd.to_datetime(start_str) + timedelta(days=i)).strftime(load_data_basic.date_time_format)[:-3]
		date_end_str = (pd.to_datetime(start_str) + timedelta(days=i+1)).strftime(load_data_basic.date_time_format)[:-3]
		row_df = pd.DataFrame(index=[date_start_str])
		
		raw_df = data_df[date_start_str:date_end_str]
		
		if len(raw_df) == 0:
			continue
		
		if len(raw_df) > 0:
			row_df['frequency'] = len(raw_df)
			row_df['total_time'] = np.sum(np.array(raw_df))
			row_df['mean_time'] = np.mean(np.array(raw_df))
			# row_df['std_time'] = np.std(np.array(raw_df))
			row_df['less_than_1min'] = len(np.where(np.array(raw_df) <= 60)[0])
			row_df['1min_5min'] = len(np.where((np.array(raw_df) > 60) & (np.array(raw_df) <= 300))[0])
			row_df['above_1min'] = len(np.where((np.array(raw_df) > 60))[0])
			row_df['above_5min'] = len(np.where(np.array(raw_df) >= 300)[0])
		else:
			row_df['frequency'] = 0
			row_df['total_time'] = 0
			row_df['mean_time'] = 0
			row_df['less_than_1min'] = 0
			row_df['1min_5min'] = 0
			row_df['above_1min'] = 0
			row_df['above_5min'] = 0
		
		inter_df = pd.DataFrame()
		for j in range(len(raw_df)):
			time_df = raw_df.iloc[j, :]
			time_row_df = pd.DataFrame(index=[list(raw_df.index)[j]])
			time_row_df['start'] = list(raw_df.index)[j]
			time_row_df['end'] = (pd.to_datetime(list(raw_df.index)[j]) + timedelta(seconds=int(time_df['SecondsOnPhone']))).strftime(load_data_basic.date_time_format)[:-3]
			inter_df = inter_df.append(time_row_df)
		
		inter_duration_list = []
		start_list = list(pd.to_datetime(inter_df['start']))
		end_list = list(pd.to_datetime(inter_df['end']))
		for j in range(len(raw_df)-1):
			inter_time = (start_list[j+1] - end_list[j]).total_seconds()
			# if inter time is larger than 4 hours, we assume it is sleep
			if inter_time > 3600 * 4:
				continue
			inter_duration_list.append(inter_time)
		row_df['mean_inter'] = np.mean(inter_duration_list)

		if days_at_work_df.loc[date_start_str, 'work'] == 1:
			row_df['work'] = 1
		else:
			row_df['work'] = 0

		process_df = process_df.append(row_df)

	work_df = process_df.loc[process_df['work'] == 1]
	off_df = process_df.loc[process_df['work'] == 0]

	process_df.to_csv(os.path.join(data_config.phone_usage_path, participant_id + '.csv.gz'), compression='gzip')
	participant_df = pd.DataFrame(index=[participant_id])

	for col in list(process_df.columns):
		if 'work' not in col:
			participant_df[col + '_mean'] = np.nanmean(np.array(process_df[col]))
			participant_df['work_' + col + '_mean'] = np.nanmean(np.array(work_df[col]))
			participant_df['off_' + col + '_mean'] = np.nanmean(np.array(off_df[col]))
			
	igtb_cols = [col for col in list(igtb_df.columns) if 'igtb' in col]
	for col in igtb_cols:
		participant_df[col] = igtb_df.loc[igtb_df['ParticipantID'] == participant_id][col][0]

	return participant_df


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
			row_df['Session Frequency (1-5min)'] = participant_df['1min_5min_mean']
			row_df['Session Frequency (>5min)'] = participant_df['above_5min_mean']
			
			row_df['Data Type'] = 'Combined'
			row_df['Shift Type'] = shift
			
			plot_df = plot_df.append(row_df)
			
			row_df = pd.DataFrame(index=[participant_id+'work'])
			row_df['Total Usage Time'] = participant_df['work_total_time_mean']
			row_df['Mean Session Length'] = participant_df['work_mean_time_mean']
			row_df['Session Frequency'] = participant_df['work_frequency_mean']
			row_df['Mean Inter-session Time'] = participant_df['work_mean_inter_mean']
			row_df['Session Frequency (<1min)'] = participant_df['work_less_than_1min_mean']
			row_df['Session Frequency (>1min)'] = participant_df['work_above_1min_mean']
			row_df['Session Frequency (1-5min)'] = participant_df['work_1min_5min_mean']
			row_df['Session Frequency (>5min)'] = participant_df['work_above_5min_mean']
			row_df['Data Type'] = 'Workday'
			row_df['Shift Type'] = shift
			
			plot_df = plot_df.append(row_df)
			
			row_df = pd.DataFrame(index=[participant_id+'off'])
			row_df['Total Usage Time'] = participant_df['off_total_time_mean']
			row_df['Mean Session Length'] = participant_df['off_mean_time_mean']
			row_df['Session Frequency'] = participant_df['off_frequency_mean']
			row_df['Mean Inter-session Time'] = participant_df['off_mean_inter_mean']
			row_df['Session Frequency (<1min)'] = participant_df['off_less_than_1min_mean']
			row_df['Session Frequency (>1min)'] = participant_df['off_above_1min_mean']
			row_df['Session Frequency (1-5min)'] = participant_df['off_1min_5min_mean']
			row_df['Session Frequency (>5min)'] = participant_df['off_above_5min_mean']
			row_df['Data Type'] = 'Non-Workday'
			row_df['Shift Type'] = shift
			
			plot_df = plot_df.append(row_df)
			
		import seaborn as sns
		sns.set(style="whitegrid")
		
		# import ssl
		# ssl._create_default_https_context = ssl._create_unverified_context
		# tips = sns.load_dataset("tips")
		# ax = sns.boxplot(x=tips["total_bill"])
		'''
		fig, ax = plt.subplots(1, 2, sharey=True)
		for i in range(len())
		for i, grp in enumerate(df.filter(regex="a").groupby(by=df.b)):
			sns.boxplot(grp[1], ax=ax[i])
		'''
		'''
		fig, ax = plt.subplots(1, 3, sharey=True)
		for i in range(3):
			tmp_df = plot_df.loc[plot_df['work'] == 'Off-day']
			sns.boxplot(data=tmp_df, y="Session Frequency", x="shift", ax=ax[i], palette="colorblind")
		plt.show()
		'''
		# ax.get_xaxis().set_visible(False)
		# plt.xlabel
		# ax.set_title('')
		
		if os.path.exists(os.path.join('plot')) is False:
			os.mkdir(os.path.join('plot'))
		if os.path.exists(os.path.join('plot', 'daily')) is False:
			os.mkdir(os.path.join('plot', 'daily'))
		
		plt.rcParams["font.weight"] = "bold"
		plt.rcParams["font.size"] = 16
		
		fig = plt.figure(figsize=(16, 6.25))
		axes = fig.subplots(nrows=2, ncols=3)
		
		for col in list(plot_df.columns):
			if 'Type' not in col:
				if 'Total Usage Time' in col:
					data_col = 'total_time_mean'
					max_y = 50000
					idx = 0
					y_lable_text = 'Seconds'
				elif 'Mean Session Length' in col:
					data_col = 'mean_time_mean'
					max_y = 1250
					idx = 1
					y_lable_text = 'Seconds'
				elif 'Session Frequency' in col and '(' not in col:
					data_col = 'frequency_mean'
					max_y = 200
					idx = 3
					y_lable_text = 'Number of Sessions'
				elif 'Mean Inter-session Time' in col:
					data_col = 'mean_inter_mean'
					max_y = 3000
					idx = 2
					y_lable_text = 'Seconds'
				elif '(<1min)' in col:
					data_col = 'less_than_1min_mean'
					max_y = 120
					idx = 4
					y_lable_text = 'Number of Sessions'
				elif '(>1min)' in col:
					data_col = 'above_1min_mean'
					max_y = 100
					idx = 5
					y_lable_text = 'Number of Sessions'
				elif '(1-5min)' in col:
					data_col = '1min_5min_mean'
					max_y = 60
					idx = 999
					y_lable_text = 'Number of Sessions'
				else:
					data_col = 'above_5min_mean'
					max_y = 30
					idx = 999
					y_lable_text = 'Number of Sessions'
					
				if idx is 999:
					continue
				
				col_idx = int(idx % 3)
				row_idx = int(idx / 3)
				# ax = sns.boxplot(x="Data Type", y=col, hue="Shift Type", data=plot_df, palette="colorblind", ax=axes[row_idx][col_idx])
				ax = sns.boxplot(x="Data Type", y=col, hue="Shift Type", data=plot_df, palette="seismic", ax=axes[row_idx][col_idx])
				lgd = ax.legend()
				lgd.set_title('')
				ax.set_xlabel('')
				ax.set_ylabel(y_lable_text, fontweight='bold', fontsize=14)
				ax.set_title(col, fontweight='bold', fontsize=14)
				
				label_list = []
				
				print('\n\n')
				print('K-S test for %s day shift' % data_col)
				stat, p = stats.ks_2samp(day_nurse_df['work_' + data_col].dropna(), day_nurse_df['off_' + data_col].dropna())
				print('Statistics = %.3f, p = %.3f' % (stat, p))
				print('Day shift work: mean = %.2f, std = %.2f' % (np.mean(day_nurse_df['work_' + data_col].dropna()), np.std(day_nurse_df['work_' + data_col].dropna())))
				print('Day shift off: mean = %.2f, std = %.2f' % (np.mean(day_nurse_df['off_' + data_col].dropna()), np.std(day_nurse_df['off_' + data_col].dropna())))
				
				print('\n\n')
				print('K-S test for %s night shift' % data_col)
				stat, p = stats.ks_2samp(night_nurse_df['work_' + data_col].dropna(), night_nurse_df['off_' + data_col].dropna())
				
				print('Statistics = %.3f, p = %.3f' % (stat, p))
				print('Night shift work: mean = %.2f, std = %.2f' % (np.mean(night_nurse_df['work_' + data_col].dropna()), np.std(night_nurse_df['work_' + data_col].dropna())))
				print('Night shift off: mean = %.2f, std = %.2f' % (np.mean(night_nurse_df['off_' + data_col].dropna()), np.std(night_nurse_df['off_' + data_col].dropna())))
				
				for ticklabel in ax.get_xticklabels():
					if ticklabel.get_text() == 'Non-Workday':
						final_data_col = 'off_' + data_col
					elif ticklabel.get_text() == 'Workday':
						final_data_col = 'work_' + data_col
					else:
						final_data_col = data_col
					
					stat, p = stats.ks_2samp(day_nurse_df[final_data_col].dropna(), night_nurse_df[final_data_col].dropna())
					
					'''
					print('\n\n')
					print('K-S test for %s' % final_data_col)
					print('Statistics = %.3f, p = %.3f' % (stat, p))
					print('Day shift: mean = %.2f, std = %.2f' % (np.mean(day_nurse_df[final_data_col].dropna()), np.std(day_nurse_df[final_data_col].dropna())))
					print('Night shift: mean = %.2f, std = %.2f' % (np.mean(night_nurse_df[final_data_col].dropna()), np.std(night_nurse_df[final_data_col].dropna())))
					'''
					if p < 0.001:
						label_list.append(ticklabel.get_text() + '\n(p<0.001)')
					else:
						label_list.append(ticklabel.get_text() + '\n(p=' + str(p)[:5] + ')')
					
				ax.set_xticklabels(labels=label_list, fontweight='bold', fontsize=14)
				ax.set_ylim([0, max_y])
				ax.legend(loc='center right', bbox_to_anchor=(0.02, 0.80, 0.99, .1), prop={'size':11})
				# ax.legend(loc='center left', bbox_to_anchor=(0.04, 0.82, 1., .102))
		
		for i in range(2):
			for j in range(3):
				axes[i][j].yaxis.set_tick_params(size=1)
				axes[i][j].tick_params(axis="y", labelsize=15)
				axes[i][j].grid(linestyle='--')
				axes[i][j].grid(False, axis='x')
				axes[i][j].tick_params(axis="y", labelsize=15)
				
		plt.tight_layout()
		plt.savefig(os.path.join('plot', 'daily', 'daily'), dpi=300)
		plt.close()
		
		'''
		for col in list(nurse_df.columns):
			
			if 'shift' not in col or 'job' not in col or 'igtb' not in col:
				# K-S test
				stat, p = stats.ks_2samp(day_nurse_df[col].dropna(), night_nurse_df[col].dropna())
				print('K-S test for %s' % col)
				print('Statistics = %.3f, p = %.3f\n\n' % (stat, p))
		'''
	all_df = pd.DataFrame()

	valid_data = 0
	for idx, participant_id in enumerate(top_participant_id_list[:]):
		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		days_at_work_df = load_sensor_data.read_preprocessed_days_at_work(data_config.days_at_work_path, participant_id)
		realizd_raw_df = load_sensor_data.read_realizd(os.path.join(tiles_data_path, '2_raw_csv_data/realizd/'), participant_id)

		nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
		primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
		shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
		job_str = 'nurse' if nurse == 1 else 'non_nurse'
		shift_str = 'day' if shift == 'Day shift' else 'night'

		if job_str == 'nurse' and len(realizd_raw_df) > 700:
			valid_data = valid_data + 1

		if realizd_raw_df is None or days_at_work_df is None:
			continue

		if len(realizd_raw_df) < 700 or len(days_at_work_df) < 5:
			continue

		participant_df = process_daily_realizd(data_config, realizd_raw_df, days_at_work_df, igtb_df, participant_id)

		if participant_df is not None:
			all_df = all_df.append(participant_df)
			all_df = all_df.loc[:, list(participant_df.columns)]

	print(valid_data)
	all_df.to_csv(os.path.join(data_config.phone_usage_path, 'daily_summary.csv.gz'), compression='gzip')


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
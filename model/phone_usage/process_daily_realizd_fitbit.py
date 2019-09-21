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


def process_daily_realizd(data_config, realizd_df, fitbit_df, summary_df, days_at_work_df, participant_id):

	process_df = pd.DataFrame()

	if len(realizd_df) < 700 or len(days_at_work_df) < 5:
		return None

	days = (pd.to_datetime(realizd_df.index[-1]).date() - pd.to_datetime(realizd_df.index[0]).date()).days

	start_str = pd.to_datetime(realizd_df.index[0]).date()
	end_str = pd.to_datetime(realizd_df.index[-1]).date()
	for i in range(days):

		date_start_str = (pd.to_datetime(start_str) + timedelta(days=i)).strftime(load_data_basic.date_time_format)[:-3]
		date_end_str = (pd.to_datetime(start_str) + timedelta(days=i+1)).strftime(load_data_basic.date_time_format)[:-3]
		row_df = pd.DataFrame(index=[date_start_str])
		
		date_str = (pd.to_datetime(start_str) + timedelta(days=i)).strftime(load_data_basic.date_only_date_time_format)
		raw_df = realizd_df[date_start_str:date_end_str]
		
		if len(raw_df) == 0:
			continue
		
		if len(raw_df) > 0:
			row_df['frequency'] = len(raw_df)
			row_df['total_time'] = np.sum(np.array(raw_df))
			row_df['mean_time'] = np.mean(np.array(raw_df))
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

		# Fitbit
		fitbit_daily_df = fitbit_df[date_start_str:date_end_str]
		if len(fitbit_daily_df) > 0:
			if len(fitbit_daily_df.dropna()) > 720:
				row_df['heart_rate_mean'] = np.nanmean(np.array(fitbit_daily_df['HeartRatePPG']))
				row_df['heart_rate_std'] = np.nanstd(np.array(fitbit_daily_df['HeartRatePPG']))
				row_df['heart_rate_range'] = np.nanmax(np.array(fitbit_daily_df['HeartRatePPG'])) - np.nanmin(np.array(fitbit_daily_df['HeartRatePPG']))
				
				row_df['step_count_mean'] = np.nanmean(np.array(fitbit_daily_df['StepCount']))
				row_df['step_count_std'] = np.nanstd(np.array(fitbit_daily_df['StepCount']))
				row_df['step_count_sum'] = np.nansum(np.array(fitbit_daily_df['StepCount']))
				row_df['num_of_minutes'] = len(fitbit_daily_df.dropna())
				
		# Fitbit Summary df
		if date_str in list(summary_df.index):
			daily_summary_df = summary_df.loc[date_str, :]
			row_df['resting_heart_rate'] = daily_summary_df['RestingHeartRate']
			row_df['sleep_per_day'] = daily_summary_df['SleepPerDay']
			row_df['sleep_in_bed_minute'] = daily_summary_df['SleepMinutesInBed']
			row_df['sleep_asleep_minute'] = daily_summary_df['SleepMinutesAsleep']
			row_df['number_of_step'] = daily_summary_df['NumberSteps']
			
			minutes_array = np.zeros([1, 4])
			row_df['cardio_minutes'] = daily_summary_df['Cardio_minutes']
			row_df['out_of_range_minutes'] = daily_summary_df['Out of Range_minutes']
			row_df['fat_burn_minutes'] = daily_summary_df['Fat Burn_minutes']
			row_df['peak_minutes'] = daily_summary_df['Peak_minutes']
			
			minutes_array[0, 0] = daily_summary_df['Cardio_minutes']
			minutes_array[0, 1] = daily_summary_df['Out of Range_minutes']
			minutes_array[0, 2] = daily_summary_df['Fat Burn_minutes']
			minutes_array[0, 3] = daily_summary_df['Peak_minutes']
			
			all_minutes = np.nansum(minutes_array)
			
			if len(fitbit_daily_df.dropna()) > 0:
				row_df['cardio_ratio'] = daily_summary_df['Cardio_minutes'] / all_minutes
				row_df['out_of_range_ratio'] = daily_summary_df['Out of Range_minutes'] / all_minutes
				row_df['fat_burn_ratio'] = daily_summary_df['Fat Burn_minutes'] / all_minutes
				row_df['peak_ratio'] = daily_summary_df['Peak_minutes'] / all_minutes

		process_df = process_df.append(row_df)

	participant_df = pd.DataFrame(index=[participant_id])
	process_df.to_csv(os.path.join(data_config.phone_usage_path, participant_id + '_fitbit_phone.csv.gz'), compression='gzip')

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

	if os.path.exists(os.path.join(data_config.phone_usage_path, 'daily_summary_phone_fitbit.csv.gz')) is True:
		all_df = pd.read_csv(os.path.join(data_config.phone_usage_path, 'daily_summary_phone_fitbit.csv.gz'), index_col=0)

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
		
	all_df = pd.DataFrame()

	valid_data = 0
	for idx, participant_id in enumerate(top_participant_id_list[180:]):
		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		days_at_work_df = load_sensor_data.read_preprocessed_days_at_work(data_config.days_at_work_path, participant_id)
		realizd_raw_df = load_sensor_data.read_realizd(os.path.join(tiles_data_path, '2_raw_csv_data/realizd/'), participant_id)
		fitbit_df = load_sensor_data.read_preprocessed_fitbit(data_config.fitbit_sensor_dict['preprocess_path'], participant_id)
		fitbit_data_dict = load_sensor_data.read_fitbit(os.path.join(tiles_data_path, '3_preprocessed_data/fitbit/'), participant_id)
		summary_df = fitbit_data_dict['summary']
		
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

		participant_df = process_daily_realizd(data_config, realizd_raw_df, fitbit_df, summary_df, days_at_work_df, participant_id)

		if participant_df is not None:
			all_df = all_df.append(participant_df)
			all_df = all_df.loc[:, list(participant_df.columns)]

	print(valid_data)
	all_df.to_csv(os.path.join(data_config.phone_usage_path, 'daily_summary_phone_fitbit.csv.gz'), compression='gzip')


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
"""
Top level classes for the preprocess model.
"""
from __future__ import print_function

import copy
import pdb
from om_signal.helper import *
from fitbit.helper import *
from realizd.helper import *
from owl_in_one.helper import *
import pandas as pd
import pickle

date_time_format = '%Y-%m-%dT%H:%M:%S.%f'


__all__ = ['Preprocess']


class Preprocess(object):
	"""
	Preprocess script for all signal
	"""

	def __init__(self, data_config=None, participant_id=None):
		"""
		Initialization method
		"""

		###########################################################
		# Assert if these parameters are not parsed
		###########################################################
		assert data_config is not None
		self.data_config = data_config

		###########################################################
		# Initialize data array within class
		###########################################################
		self.sliced_data_array = []
		self.processed_sliced_data_array = []

		self.participant_id = participant_id

	def process_fitbit(self, ppg_df, step_df, valid_slice_in_min=60, imputation=''):

		print('---------------------------------------------------------------------')
		print('Function: process_fitbit')
		print('---------------------------------------------------------------------')
		if len(imputation) == 0:
			preprocess_data_all_df = fitbit_process_data(ppg_df, step_df, participant=self.participant_id, check_saved=True, data_config=self.data_config)
			preprocess_data_all_df.to_csv(os.path.join(self.data_config.fitbit_sensor_dict['preprocess_path'], self.participant_id + '.csv.gz'), compression='gzip')
		else:
			###########################################################
			# Get start and end time of a chunk
			###########################################################
			"""
			Slice the data based on shift
			"""
			start_time_array, end_time_array = fitbit_sliced_data_start_end_array(ppg_df, threshold=timedelta(seconds=60*self.data_config.fitbit_sensor_dict['imputation_threshold']))
			preprocess_data_all_df = pd.DataFrame()

			check_saved = True

			if os.path.exists(os.path.join(self.data_config.fitbit_sensor_dict['preprocess_path'], self.participant_id + '.csv.gz')) is True and check_saved:
				return

			###########################################################
			# Slice the data
			###########################################################
			for i in range(len(start_time_array)):

				print('Complete process for %s: %.2f' % (self.participant_id, 100 * i / len(start_time_array)))

				start_time, end_time = start_time_array[i], end_time_array[i]
				tmp_ppg_data_df = ppg_df[start_time:end_time]
				tmp_step_data_df = step_df[start_time:end_time]

				if (pd.to_datetime(end_time) - pd.to_datetime(start_time)).seconds > 60 * valid_slice_in_min:
					###########################################################
					# Process sliced data
					###########################################################
					preprocess_data_df = fitbit_process_sliced_data(tmp_ppg_data_df, tmp_step_data_df,
																	participant=self.participant_id, check_saved=True,
																	data_config=self.data_config)

					if len(preprocess_data_df) > 0:
						preprocess_data_all_df = preprocess_data_df if len(preprocess_data_all_df) == 0 else preprocess_data_all_df.append(preprocess_data_df)
						if os.path.exists(os.path.join(self.data_config.fitbit_sensor_dict['preprocess_path'], self.participant_id)) is False:
							os.mkdir(os.path.join(self.data_config.fitbit_sensor_dict['preprocess_path'], self.participant_id))

						preprocess_data_df.to_csv(os.path.join(self.data_config.fitbit_sensor_dict['preprocess_path'], self.participant_id, start_time + '.csv.gz'), compression='gzip')
			preprocess_data_all_df.to_csv(os.path.join(self.data_config.fitbit_sensor_dict['preprocess_path'], self.participant_id + '.csv.gz'), compression='gzip')

	def preprocess_realizd(self, data_df):
		"""
		Process realizd data based on shift
		"""
		print('---------------------------------------------------------------------')
		print('Function: process_realizd')
		print('---------------------------------------------------------------------')
		self.preprocess_data_all_df = pd.DataFrame()

		if os.path.exists(os.path.join(self.data_config.realizd_sensor_dict['preprocess_path'], self.participant_id + '.csv.gz')) is True:
			return

		###########################################################
		# Get start and end time of a shift
		###########################################################
		if len(data_df) > 300:
			self.preprocess_data_all_df = realizd_process_data(data_df, offset=self.data_config.realizd_sensor_dict['offset'])
			self.preprocess_data_all_df.to_csv(os.path.join(self.data_config.realizd_sensor_dict['preprocess_path'], self.participant_id + '.csv.gz'), compression='gzip')

	def preprocess_owl_in_one(self, data_df):
		"""
		Process owl_in_one data based on shift
		"""
		print('---------------------------------------------------------------------')
		print('Function: process_owl_in_one')
		print('---------------------------------------------------------------------')
		self.preprocess_data_all_df = pd.DataFrame()

		###########################################################
		# Get start and end time of a shift
		###########################################################
		if len(data_df) > 300:

			# if os.path.exists(os.path.join(self.data_config.owl_in_one_sensor_dict['preprocess_path'], self.participant_id + '.csv.gz')) is False:
			self.preprocess_data_all_df = process_owl_in_one_data(data_df, offset=self.data_config.owl_in_one_sensor_dict['offset'])
			self.preprocess_data_all_df.to_csv(os.path.join(self.data_config.owl_in_one_sensor_dict['preprocess_path'], self.participant_id + '.csv.gz'), compression='gzip')

	def preprocess_audio(self, data_df):
		"""
		Process audio data
		"""
		print('---------------------------------------------------------------------')
		print('Function: preprocess_audio')
		print('---------------------------------------------------------------------')
		sample_rate = 0.01 # The sampling rate of the audio data in seconds (ignoring missing data)
		offset = int(self.data_config.audio_sensor_dict['offset'])

		out_file_path = os.path.join(self.data_config.audio_sensor_dict['preprocess_path'], self.participant_id + '.csv.gz')
		if os.path.exists(out_file_path): # Skip files that have already been written
			return

		data_df.index = pd.to_datetime(data_df.index)
		unix_times = pd.DatetimeIndex([data_df.index[0], data_df.index[data_df.shape[0]-1]]).astype(np.int64)
		start_unix_time_minute_clamp = int(unix_times[0] - (unix_times[0]% int(offset*1e9)))
		end_unix_time_minute_clamp = int(unix_times[1] - (unix_times[1] % int(offset*1e9)))
		time_index = pd.to_datetime(range(start_unix_time_minute_clamp, end_unix_time_minute_clamp+int(offset*1e9), int(offset*1e9))).strftime(date_time_format)[:-3]
		foreground_data = np.zeros(len(time_index))

		idx = 0
		last_time_index = 0
		while idx < len(time_index):
			if (idx%(len(time_index)/20)) == 0:
				print("Percent complete: %f"%(float(idx)/len(time_index)))
			cur_time_index = np.searchsorted(data_df.index, time_index[idx])
			num_foreground_samples = max(cur_time_index-last_time_index-1, 0)
			percentage_foreground = (num_foreground_samples*sample_rate)/float(offset)
			foreground_data[idx] = percentage_foreground
			last_time_index = cur_time_index
			idx += 1

		self.preprocess_data_all_df = pd.DataFrame(data={'foreground': foreground_data}, index=time_index)
		self.preprocess_data_all_df.to_csv(out_file_path, index=True, index_label='Timestamp', compression='gzip')


	def slice_raw_data(self, method=None, valid_slice_in_min=180, data_df=None):
		"""
		Slice the data based on block (chunk)
		"""

		# Assert if method parameter is not parsed
		assert method is not None

		print('---------------------------------------------------------------------')
		print('Function: slice_raw_data')
		print('---------------------------------------------------------------------')

		###########################################################
		# Get start and end time of a shift
		###########################################################
		start_time_array, end_time_array = om_signal_sliced_data_start_end_array(data_df, threshold=timedelta(seconds=60 * 2))

		###########################################################
		# Slice the data
		###########################################################
		for i in range(len(start_time_array)):

			start_time, end_time = start_time_array[i], end_time_array[i]

			shift_data_df = data_df[start_time:end_time]

			# At least 3 hours of data for a valid shift
			if len(shift_data_df) / 60 > valid_slice_in_min:
				self.sliced_data_array.append(shift_data_df)

		return self.sliced_data_array

	def preprocess_slice_raw_data_full_feature(self, check_saved=False):

		for index, sliced_data_df in enumerate(self.sliced_data_array):
			print('---------------------------------------------------------------------')
			print('Function: preprocess_slice_raw_data')
			print('Process data with start index: %s' % (sliced_data_df.index[0]))
			print('Process data process %.2f' % (index / len(self.sliced_data_array) * 100))
			print('---------------------------------------------------------------------')

			###########################################################
			# 2. If we have weird data, skip
			###########################################################
			if len(np.unique(np.array(sliced_data_df)[:, 0])) < 5:
				continue

			preprocess_data_df = om_signal_process_sliced_data_full_feature(sliced_data_df, self.data_config, self.participant_id)

			if preprocess_data_df is not None:
				self.processed_sliced_data_array.append(preprocess_data_df)

	def preprocess_days_at_work(self, mgt_df, owl_in_one_df, omsignal_df, fitbit_summary_df, shift='day', job='nurse'):

		from datetime import datetime
		# Define start and end date
		start_date = datetime(year=2018, month=2, day=15)
		end_date = datetime(year=2018, month=9, day=10)

		# Create a time range for the data to be used. We read through all the
		# files and obtain the earliest and latest dates. This is the time range
		# used to produced the data to be saved in 'preprocessed/'
		dates_range = pd.date_range(start=start_date, end=end_date, normalize=True)
		dates_str_list = [date.strftime(date_time_format)[:-3] for date in dates_range]
		work_df = pd.DataFrame(index=dates_str_list, columns=['work_start', 'work_end'])
		days_at_work_df = pd.DataFrame(index=dates_str_list, columns=['work'])

		sleep_list = list(fitbit_summary_df['Sleep1BeginTimestamp'].dropna())
		sleep_list += list(fitbit_summary_df['Sleep2BeginTimestamp'].dropna())
		sleep_list += list(fitbit_summary_df['Sleep3BeginTimestamp'].dropna())
		sleep_list = list(set(sleep_list))
		sleep_list.sort()
		sleep_df = pd.DataFrame(sleep_list, index=sleep_list)

		data_list = [owl_in_one_df, omsignal_df]

		# First part
		for i, data_df in enumerate(data_list):

			if data_df is None:
				continue
			dates_worked = list(set([pd.to_datetime(date).date().strftime(date_time_format)[:-3] for date in data_df.index]))

			for date in dates_range:
				if date.date().strftime(date_time_format)[:-3] in dates_worked:
					days_at_work_df.loc[pd.to_datetime(date).date().strftime(date_time_format)[:-3], 'work'] = 1
		for i in range(len(mgt_df)):
			work = mgt_df.iloc[i, :].location_mgt
			if work == 2:
				time_stamp = mgt_df.iloc[i, :].timestamp
				time_of_survey = pd.to_datetime(time_stamp)
				days_at_work_df.loc[time_of_survey.date().strftime(date_time_format)[:-3], 'work'] = 1
		days_at_work_df.to_csv(os.path.join(self.data_config.days_at_work_path, self.participant_id + '.csv.gz'), compression='gzip')

		if job == 'nurse':
			# Second part, detailed, nurse only, since we know their work start and end time
			for i in range(len(mgt_df)):
				work = mgt_df.iloc[i, :].location_mgt
				time_stamp = mgt_df.iloc[i, :].timestamp
				time_of_survey = pd.to_datetime(time_stamp)

				if work != 2:
					continue

				start_time_str = None
				if job == 'nurse':
					if shift == 'day':
						if 4 < pd.to_datetime(time_of_survey).hour <= 24:
							start_time_str = time_of_survey.replace(hour=7, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
							end_time_str = time_of_survey.replace(hour=19, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
					else:
						if 16 <= pd.to_datetime(time_of_survey).hour <= 24:
							start_time_str = time_of_survey.replace(hour=19, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
							end_time_str = (time_of_survey + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
						elif 0 <= pd.to_datetime(time_of_survey).hour <= 12:
							start_time_str = (time_of_survey - timedelta(days=1)).replace(hour=19, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
							end_time_str = time_of_survey.replace(hour=7, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]

				if start_time_str is None:
					continue

				if len(sleep_df[start_time_str:end_time_str]) > 0:
					continue
				dates_str = pd.to_datetime(start_time_str).replace(hour=0, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
				work_df.loc[dates_str, 'work_start'] = start_time_str
				work_df.loc[dates_str, 'work_end'] = end_time_str
				work_df.loc[dates_str, 'mgt_str'] = pd.to_datetime(time_stamp).strftime(date_time_format)[:-3]
				work_df.loc[dates_str, 'owl_in_one_str'] = 'None'
				work_df.loc[dates_str, 'om_str'] = 'None'

			for i, data_df in enumerate(data_list):

				if data_df is None:
					continue

				time_diff = list((pd.to_datetime(list(data_df.index[1:])) - pd.to_datetime(list(data_df.index[:-1]))).total_seconds())
				change_point_start_list = [0]
				change_point_end_list = list(np.where(np.array(time_diff) > 3600 * 8)[0])

				if len(change_point_end_list) > 2:
					[change_point_start_list.append(change_point_end + 1) for change_point_end in change_point_end_list]
					for j, change_point_end in enumerate(change_point_end_list):
						time_start = pd.to_datetime(list(data_df.index)[change_point_start_list[j]])
						dates_str = time_start.replace(hour=0, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
						start_time_str = None
						if job == 'nurse':
							if shift == 'day':
								if 4 < pd.to_datetime(time_start).hour < 22:
									start_time_str = time_start.replace(hour=7, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
									end_time_str = time_start.replace(hour=19, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
							else:
								if 16 <= pd.to_datetime(time_start).hour <= 24:
									start_time_str = time_start.replace(hour=19, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
									end_time_str = (time_start + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
								elif 0 <= pd.to_datetime(time_start).hour <= 12:
									start_time_str = (time_start - timedelta(days=1)).replace(hour=19, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
									end_time_str = time_start.replace(hour=7, minute=0, second=0,  microsecond=0).strftime(date_time_format)[:-3]

						if start_time_str is None:
							continue

						if len(sleep_df[start_time_str:end_time_str]) > 0:
							continue

						dates_str = pd.to_datetime(start_time_str).replace(hour=0, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
						work_df.loc[dates_str, 'work_start'] = start_time_str
						work_df.loc[dates_str, 'work_end'] = end_time_str

						if i == 0:
							work_df.loc[dates_str, 'mgt_str'] = 'None'
							work_df.loc[dates_str, 'om_str'] = 'None'
							work_df.loc[dates_str, 'owl_in_one_str'] = pd.to_datetime(list(data_df.index)[change_point_start_list[j]]).strftime(date_time_format)[:-3]
						else:
							work_df.loc[dates_str, 'mgt_str'] = 'None'
							work_df.loc[dates_str, 'owl_in_one_str'] = 'None'
							work_df.loc[dates_str, 'om_str'] = pd.to_datetime(list(data_df.index)[change_point_start_list[j]]).strftime(date_time_format)[:-3]
			work_df = work_df.dropna(subset=['work_start'])
			work_df = work_df.sort_index()
			work_df.to_csv(os.path.join(self.data_config.days_at_work_path, self.participant_id + '_detailed.csv.gz'), compression='gzip')

	def preprocess_sleep_data(self, fitbit_summary_df, fitbit_df, days_at_work_df):

		if fitbit_summary_df is None or fitbit_df is None or days_at_work_df is None:
			return

		sleep_col_list = []
		for i in range(4):
			sleep_col_list.append([col for col in list(fitbit_summary_df.columns) if 'Sleep' + str(i) in col])

		sleep_dict = {}
		sleep_dict['summary'] = pd.DataFrame()
		sleep_dict['data'] = {}

		for day in range(len(fitbit_summary_df)):
			row = fitbit_summary_df.iloc[day, :]
			for i in range(4):

				if len(sleep_col_list[i]) > 0 and len(str(row['Sleep'+str(i)+'BeginTimestamp'])) > 5:
					sleep_row_df = pd.DataFrame(index=[row['Sleep'+str(i)+'BeginTimestamp']])
					for col in sleep_col_list[i]:
						sleep_row_df[col.replace(str(i), '')] = row[col]

					sleep_row_df['sleep_before_work'] = 0
					sleep_row_df['sleep_after_work'] = 0
					sleep_row_df['sleep_before_work_nearest'] = 0
					sleep_row_df['RestingHeartRate'] = row['RestingHeartRate']

					sleep_start = pd.to_datetime(sleep_row_df['SleepBeginTimestamp'][0])
					sleep_end = pd.to_datetime(sleep_row_df['SleepEndTimestamp'][0])

					twelve_hour_before_sleep = sleep_start - timedelta(hours=12)
					twelve_hour_after_sleep = sleep_end + timedelta(hours=12)

					# Before sleep process
					tmp_data_df = fitbit_df[twelve_hour_before_sleep.strftime(date_time_format)[:-3]:sleep_start.strftime(date_time_format)[:-3]]

					if len(tmp_data_df) > 360:
						sleep_row_df['step_before_sleep'] = np.nansum(np.array(tmp_data_df['StepCount']))
						max_hr = np.nanmax(np.array(tmp_data_df['HeartRatePPG']))
						max_step_count = np.nanmax(np.array(tmp_data_df['StepCount']))
						sleep_row_df['max_hr_before_sleep'] = max_hr
						sleep_row_df['max_stepcount_before_sleep'] = max_step_count

						sleep_row_df['peak_min_before_sleep'] = len(np.where(np.array(tmp_data_df['HeartRatePPG']) >= 0.85 * max_hr)[0]) / len(tmp_data_df)

						cond1 = np.array(tmp_data_df['HeartRatePPG']) < 0.85 * max_hr
						cond2 = np.array(tmp_data_df['HeartRatePPG']) >= 0.7 * max_hr
						sleep_row_df['cardio_min_before_sleep'] = len(np.where(cond1 & cond2)[0]) / len(tmp_data_df)

						cond1 = np.array(tmp_data_df['HeartRatePPG']) < 0.70 * max_hr
						cond2 = np.array(tmp_data_df['HeartRatePPG']) >= 0.50 * max_hr
						sleep_row_df['fatburn_min_before_sleep'] = len(np.where(cond1 & cond2)[0]) / len(tmp_data_df)

						cond1 = np.array(tmp_data_df['HeartRatePPG']) < 0.50 * max_hr
						sleep_row_df['out_range_min_before_sleep'] = len(np.where(cond1)[0]) / len(tmp_data_df)

					# After sleep process
					tmp_data_df = fitbit_df[sleep_end.strftime(date_time_format)[:-3]:twelve_hour_after_sleep.strftime(date_time_format)[:-3]]

					if len(tmp_data_df) > 360:
						sleep_row_df['step_after_sleep'] = np.nansum(np.array(tmp_data_df['StepCount']))

						max_hr = np.nanmax(np.array(tmp_data_df['HeartRatePPG']))
						max_step_count = np.nanmax(np.array(tmp_data_df['StepCount']))
						sleep_row_df['max_hr_after_sleep'] = max_hr
						sleep_row_df['max_stepcount_after_sleep'] = max_step_count

						sleep_row_df['peak_min_after_sleep'] = len(np.where(np.array(tmp_data_df['HeartRatePPG']) >= 0.85 * max_hr)[0]) / len(tmp_data_df)

						cond1 = np.array(tmp_data_df['HeartRatePPG']) < 0.85 * max_hr
						cond2 = np.array(tmp_data_df['HeartRatePPG']) >= 0.7 * max_hr
						sleep_row_df['cardio_min_after_sleep'] = len(np.where(cond1 & cond2)[0]) / len(tmp_data_df)

						cond1 = np.array(tmp_data_df['HeartRatePPG']) < 0.70 * max_hr
						cond2 = np.array(tmp_data_df['HeartRatePPG']) >= 0.50 * max_hr
						sleep_row_df['fatburn_min_after_sleep'] = len(np.where(cond1 & cond2)[0]) / len(tmp_data_df)

						cond1 = np.array(tmp_data_df['HeartRatePPG']) < 0.50 * max_hr
						sleep_row_df['out_range_min_after_sleep'] = len(np.where(cond1)[0]) / len(tmp_data_df)

					for j in range(len(days_at_work_df)):

						work_start = pd.to_datetime(days_at_work_df.iloc[j]['work_start'])
						work_end = pd.to_datetime(days_at_work_df.iloc[j]['work_end'])
						if 0 < (work_start - sleep_end).total_seconds() / 3600 < 8:
							sleep_row_df['sleep_before_work_nearest'] = 1
						if 0 < (work_start - sleep_end).total_seconds() / 3600 < 16:
							sleep_row_df['sleep_before_work'] = 1
						if 0 < (sleep_start - work_end).total_seconds() / 3600 < 8:
							sleep_row_df['sleep_after_work'] = 1

					sleep_fitbit_df = fitbit_df[sleep_row_df['SleepBeginTimestamp'][0]:sleep_row_df['SleepEndTimestamp'][0]]
					sleep_row_df['duration'] = (pd.to_datetime(sleep_row_df['SleepEndTimestamp'][0]) - pd.to_datetime(sleep_row_df['SleepBeginTimestamp'][0])).total_seconds() / 3600
					sleep_row_df['duration_in_minute'] = (pd.to_datetime(sleep_row_df['SleepEndTimestamp'][0]) - pd.to_datetime(sleep_row_df['SleepBeginTimestamp'][0])).total_seconds() / 60
					sleep_row_df['min_of_step_above_zero'] = len(np.where(np.array(sleep_fitbit_df['StepCount']) > 0)[0])

					sleep_dict['data'][row['Sleep'+str(i)+'BeginTimestamp']] = sleep_fitbit_df
					sleep_dict['summary'] = sleep_dict['summary'].append(sleep_row_df)

		sleep_dict['summary'] = sleep_dict['summary'].sort_index()
		output = open(os.path.join(self.data_config.sleep_path, self.participant_id + '.pkl'), 'wb')
		pickle.dump(sleep_dict, output)


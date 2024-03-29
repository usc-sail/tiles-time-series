from datetime import timedelta

import os
import numpy as np
import pandas as pd
import scipy.signal

from fancyimpute import (
	IterativeImputer,
	SoftImpute,
	KNN
)

from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'


def fitbit_sliced_data_start_end_array(raw_data_df, threshold=timedelta(seconds=1)):
	###########################################################
	# Trick: cal time offset of consecutive rows
	###########################################################
	raw_data_time_array = pd.to_datetime(raw_data_df.index)

	# Get consecutive time offset
	raw_data_time_offset_array = raw_data_time_array[1:] - raw_data_time_array[:-1]
	offset_mask_true_index = np.where((raw_data_time_offset_array > threshold))[0]

	###########################################################
	# Get start and end time index
	###########################################################
	start_time_array, end_time_array = [raw_data_df.index[0]], []

	for i, true_idx, in enumerate(offset_mask_true_index):
		end_time_array.append(raw_data_df.index[true_idx])
		start_time_array.append(raw_data_df.index[true_idx + 1])

	end_time_array.append(raw_data_df.index[-1])

	return start_time_array, end_time_array

def fitbit_process_sliced_data(ppg_data_df, step_data_df, participant=None, data_config=None, check_saved=True):

	###########################################################
	# Initialization
	###########################################################
	offset = data_config.fitbit_sensor_dict['offset']
	overlap = data_config.fitbit_sensor_dict['overlap']
	preprocess_cols = data_config.fitbit_sensor_dict['preprocess_cols']

	threshold = 2
	interval = int(offset / 60)

	# Returned data
	preprocess_data_df = pd.DataFrame()

	###########################################################
	# Process function, only for heart rate, cadence, intensity
	###########################################################
	process_func = np.nanmean

	###########################################################
	# Start iterate data with given parameters
	###########################################################
	replace_minute = int(pd.to_datetime(ppg_data_df.index[0]).minute * 60 / offset) * interval
	start_str = pd.to_datetime(ppg_data_df.index[0]).replace(minute=replace_minute, second=0, microsecond=0).strftime(date_time_format)[:-3]

	replace_minute = int(pd.to_datetime(ppg_data_df.index[-1]).minute * 60 / offset) * interval
	end_str = (pd.to_datetime(ppg_data_df.index[-1]).replace(minute=replace_minute, second=0, microsecond=0) + timedelta(minutes=interval)).strftime(date_time_format)[:-3]

	time_length = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).total_seconds()
	start_off_dt = pd.to_datetime(start_str)

	for i in range(int(time_length / offset)):

		###########################################################
		# For normal data, calculate each time step time range
		###########################################################
		start_off_str = (start_off_dt + timedelta(seconds=offset*i-1)).strftime(date_time_format)[:-3]
		end_off_str = (start_off_dt + timedelta(seconds=offset*(i+1)+overlap-1)).strftime(date_time_format)[:-3]

		save_str = (start_off_dt + timedelta(seconds=offset*i)).strftime(date_time_format)[:-3]

		###########################################################
		# Initialize save data array
		###########################################################
		process_row_df = pd.DataFrame(index=[save_str])

		###########################################################
		# Filter data in the time range
		###########################################################
		ppg_data_row_df = ppg_data_df[start_off_str:end_off_str]
		step_data_row_df = step_data_df[start_off_str:end_off_str]

		###########################################################
		# Process the data
		###########################################################
		process_row_df['HeartRatePPG'] = process_func(ppg_data_row_df) if len(ppg_data_row_df) > threshold else np.nan
		process_row_df['StepCount'] = np.nansum(step_data_row_df) if len(step_data_row_df) > 0 else np.nan
		preprocess_data_df = preprocess_data_df.append(process_row_df)

		###########################################################
		# If check saved or not
		###########################################################
		if check_saved is True:
			if len(preprocess_data_df) > 0 and os.path.join(data_config.fitbit_sensor_dict['preprocess_path'], participant, preprocess_data_df.index[0] + '.csv.gz') is True:
				return None

	###########################################################
	# If we add imputation or not
	###########################################################
	if data_config.fitbit_sensor_dict['imputation'] != '':

		len_seq = len(preprocess_data_df)
		iteration = int(len_seq / 30)

		if data_config.fitbit_sensor_dict['imputation'] == 'knn':
			model = KNN(k=5)
		else:
			model = IterativeImputer()

		if len(preprocess_data_df.dropna()) / len(preprocess_data_df) > 0.75:
			if data_config.fitbit_sensor_dict['imputation'] == 'arima' or data_config.fitbit_sensor_dict['imputation'] == 'auto_arima':
				nan_index = np.where((np.array(preprocess_data_df) >= -1) == False)
				impute_array = np.array(preprocess_data_df)
				if len(np.where(nan_index[0] < 50)[0]) > 25:
					impute_array = impute_array[50:, :]
					preprocess_data_df = preprocess_data_df.loc[list(preprocess_data_df.index)[50:], :]

				knn_imputed_array = KNN(k=5).fit_transform(impute_array)
				nan_index = np.where((impute_array >= -1) == False)

				for i in range(len(nan_index[0])):

					if nan_index[1][i] == 1:
						if nan_index[0][i] <= 5:
							impute_array[nan_index[0][i], nan_index[1][i]] = np.nanmean(impute_array[:, nan_index[1][i]])
						else:
							impute_array[nan_index[0][i], nan_index[1][i]] = np.nanmean(impute_array[nan_index[0][i]-3:nan_index[0][i]+3, nan_index[1][i]])
					else:
						if nan_index[0][i] <= 5:
							impute_array[nan_index[0][i], nan_index[1][i]] = np.nanmean(impute_array[:, nan_index[1][i]])
						elif 5 < nan_index[0][i] < 50:
							impute_array[nan_index[0][i], nan_index[1][i]] = knn_imputed_array[nan_index[0][i], nan_index[1][i]]
						else:
							start_index = nan_index[0][i] - 200 if nan_index[0][i] > 200 else 0

							if len(np.unique(np.array(impute_array)[start_index:nan_index[0][i], nan_index[1][i]])) < 25:
								impute_array[nan_index[0][i], nan_index[1][i]] = knn_imputed_array[nan_index[0][i], nan_index[1][i]]
							else:
								if data_config.fitbit_sensor_dict['imputation'] == 'arima':
									model = ARIMA(np.array(impute_array)[start_index:nan_index[0][i], nan_index[1][i]], order=(3, 1, 0))
									model_fit = model.fit(disp=0)
									impute_array[nan_index[0][i], nan_index[1][i]] = model_fit.forecast()[0]
								else:
									model = auto_arima(np.array(impute_array)[start_index:nan_index[0][i], nan_index[1][i]],
													   start_p=1, start_q=1, start_P=1, start_Q=1,
													   max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
													   stepwise=True, suppress_warnings=True, D=10, max_D=10,
													   error_action='ignore')

									preds = model.predict(n_periods=1, return_conf_int=False)
									impute_array[nan_index[0][i], nan_index[1][i]] = preds
			else:
				impute_array = model.fit_transform(np.array(preprocess_data_df))

			hr_array = impute_array[:, 0]
			hr_array = scipy.signal.savgol_filter(hr_array, 5, 3)

			preprocess_data_df.loc[:, 'HeartRatePPG'] = hr_array
			preprocess_data_df.loc[:, 'StepCount'] = impute_array[:, 1]

		'''
		last_un_imputed_idx = -1
		for iter in range(iteration):
			data_iter_df = preprocess_data_df[iter*30:(iter+1)*30+30]
			if len(data_iter_df.dropna()) > 10 and len(data_iter_df.dropna()) / len(data_iter_df) > 0.75:
				impute_array = model.fit_transform(np.array(data_iter_df))
					
				preprocess_data_df.loc[data_iter_df.index, 'HeartRatePPG'] = impute_array[:, 0]
				preprocess_data_df.loc[data_iter_df.index, 'StepCount'] = impute_array[:, 1]
			
			else:
				filter_df = preprocess_data_df[(last_un_imputed_idx + 1) * 30:iter * 30]
				if len(filter_df.dropna()) == len(filter_df) and len(filter_df.dropna()) > 20:
					filter_array = np.array(filter_df)[:, 0]
					filter_array = scipy.signal.savgol_filter(filter_array, 5, 3)
		
					preprocess_data_df.loc[filter_df.index, 'HeartRatePPG'] = filter_array
					last_un_imputed_idx = iter
		
		if len(preprocess_data_df[(last_un_imputed_idx + 1) * 30:]) > 60:
			filter_df = preprocess_data_df[(last_un_imputed_idx + 1) * 30:]
			filter_array = np.array(filter_df)[:, 0]
			filter_array = scipy.signal.savgol_filter(filter_array, 5, 3)
	
			preprocess_data_df.loc[filter_df.index, 'HeartRatePPG'] = filter_array
		'''
	return preprocess_data_df


def fitbit_process_data(ppg_data_df, step_data_df, participant=None, data_config=None, check_saved=True):

	###########################################################
	# Initialization
	###########################################################
	offset = data_config.fitbit_sensor_dict['offset']
	overlap = data_config.fitbit_sensor_dict['overlap']
	preprocess_cols = data_config.fitbit_sensor_dict['preprocess_cols']
	interval = int(offset / 60)

	# Returned data
	preprocess_data_df = pd.DataFrame()

	###########################################################
	# Process function, only for heart rate, cadence, intensity
	###########################################################
	process_func = np.nanmean

	###########################################################
	# Start iterate data with given parameters
	###########################################################
	start_str = pd.to_datetime(ppg_data_df.index[0]).replace(minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
	end_str = (pd.to_datetime(ppg_data_df.index[-1]).replace(minute=0, second=0, microsecond=0) + timedelta(days=1)).strftime(date_time_format)[:-3]
	num_days = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days

	# Process on daily basis
	for i in range(num_days):
		###########################################################
		# For normal data, calculate each time step time range
		###########################################################
		print('Process data at day: %d, total days: %d' % (i, num_days))

		start_date_str = (pd.to_datetime(start_str) + timedelta(days=i)).strftime(date_time_format)[:-3]
		end_date_str = (pd.to_datetime(start_str) + timedelta(days=i+1)).strftime(date_time_format)[:-3]

		day_ppg_df = ppg_data_df[start_date_str:end_date_str]
		day_step_df = step_data_df[start_date_str:end_date_str]

		# The preprocess data for the day
		day_df = pd.DataFrame()

		for j in range(int(1440 / interval)):
			start_off_str = (pd.to_datetime(start_date_str) + timedelta(seconds=offset*j-int(offset/2))).strftime(date_time_format)[:-3]
			end_off_str = (pd.to_datetime(start_date_str) + timedelta(seconds=offset*j+int(offset/2))).strftime(date_time_format)[:-3]

			save_str = (pd.to_datetime(start_date_str) + timedelta(seconds=offset*j)).strftime(date_time_format)[:-3]

			###########################################################
			# Initialize save data array
			###########################################################
			row_df = pd.DataFrame(index=[save_str])

			###########################################################
			# Filter data in the time range
			###########################################################
			ppg_data_row_df = day_ppg_df[start_off_str:end_off_str]
			step_data_row_df = day_step_df[start_off_str:end_off_str]

			###########################################################
			# Process the data
			###########################################################
			row_df['HeartRatePPG'] = process_func(ppg_data_row_df) if len(ppg_data_row_df) > 1 else np.nan
			row_df['StepCount'] = np.nansum(step_data_row_df) if len(step_data_row_df) > 0 else np.nan
			day_df = day_df.append(row_df)

		preprocess_data_df = preprocess_data_df.append(day_df)
	###########################################################
	# If check saved or not
	###########################################################
	if check_saved is True:
		if len(preprocess_data_df) > 0 and os.path.join(data_config.fitbit_sensor_dict['preprocess_path'], participant, preprocess_data_df.index[0] + '.csv.gz') is True:
			return None
	return preprocess_data_df
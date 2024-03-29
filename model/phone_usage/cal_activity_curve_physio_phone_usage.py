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

from statsmodels.tsa.stattools import grangercausalitytests
import scipy.stats
from scipy import stats

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']


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
	agg, sliding = 10, 5

	load_data_path.load_chi_preprocess_path(chi_data_config, process_data_path)
	load_data_path.load_chi_activity_curve_path(chi_data_config, process_data_path, agg=agg, sliding=sliding)

	# Read ground truth data
	igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
	igtb_df = igtb_df.drop_duplicates(keep='first')
	igtb_cols = [col for col in list(igtb_df.columns) if 'igtb' in col]

	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()

	final_df = pd.DataFrame()
	for idx, participant_id in enumerate(top_participant_id_list[:]):

		nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
		primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
		shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
		job_str = 'nurse' if nurse == 1 else 'non_nurse'
		shift_str = 'day' if shift == 'Day shift' else 'night'

		icu_str = 'non_icu'

		for unit in icu_list:
			if unit in primary_unit:
				icu_str = 'icu'

		if 'ICU' in primary_unit:
			icu_str = 'icu'

		if job_str == 'non_nurse':
			if 'lab' in primary_unit or 'Lab' in primary_unit:
				icu_str = 'lab'

		row_df = pd.DataFrame(index=[participant_id])
		row_df['job'] = job_str
		row_df['icu'] = icu_str
		row_df['shift'] = shift_str

		# if job_str == 'nurse':
		# 	continue

		# if icu_str == 'non_icu':
		#    continue

		# print('job shift: %s, job type: %s' % (shift_str, job_str))

		# print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		if os.path.exists(os.path.join(chi_data_config.activity_curve_path, participant_id + '_combine.pkl')):
			data_dict = np.load(os.path.join(chi_data_config.activity_curve_path, participant_id + '_combine.pkl'), allow_pickle=True)

			regular_dict = data_dict['regular']
			shuffle_dict = data_dict['shuffle']

			dist_array = np.zeros([len(regular_dict.keys()), 2])

			for dates_idx, dates in enumerate(list(regular_dict.keys())):
				physio_dist = np.sum(regular_dict[dates]['physio']['dist'])
				realizd_dist = np.sum(regular_dict[dates]['realizd']['dist'])

				physio_shuffle_dist = shuffle_dict[dates]
				realizd_shuffle_dist = shuffle_dict[dates]

				physio_change, realizd_change = 0, 0

				for shuffle_idx in list(physio_shuffle_dist.keys()):
					if np.sum(physio_shuffle_dist[shuffle_idx]['physio']['dist']) > physio_dist * 1.1:
						physio_change += 1

					if np.sum(realizd_shuffle_dist[shuffle_idx]['realizd']['dist']) > realizd_dist * 1.1:
						realizd_change += 1

				dist_array[dates_idx, 0] = physio_change / 100
				dist_array[dates_idx, 1] = realizd_change / 100

			# dist_array = dist_array[:-2, :]
			# dist_array = dist_array[:, :]
			# print(dist_array)
			print(np.corrcoef(dist_array[:, 0], dist_array[:, 1])[0, 1])

			'''
			dist_array_inv = dist_array
			dist_array_inv[:, 1] = dist_array[:, 0]
			dist_array_inv[:, 0] = dist_array[:, 1]
			'''
			# row_df['p'] = np.abs(np.corrcoef(dist_array[:, 0], dist_array[:, 1])[0, 1])
			# row_df['p'] = np.corrcoef(dist_array[:, 0], dist_array[:, 1])[0, 1]
			# row_df['p'] = scipy.stats.spearmanr(dist_array[:, 0], dist_array[:, 1])[0]
			# row_df['p'] = np.abs(scipy.stats.spearmanr(dist_array[:, 0], dist_array[:, 1])[0])
			row_df['p'] = scipy.stats.spearmanr(dist_array[:, 0], dist_array[:, 1])[0]
			# print(scipy.stats.spearmanr(dist_array[:, 0], dist_array[:, 1]))
			for col in igtb_cols:
				row_df[col] = igtb_df.loc[igtb_df['ParticipantID'] == participant_id][col][0]
			# grangercausalitytests(dist_array, maxlag=3)
			final_df = final_df.append(row_df)

	final_df = final_df.dropna()
	non_nurse_df = final_df.loc[final_df['job'] == 'non_nurse']
	lab_df = non_nurse_df.loc[non_nurse_df['icu'] == 'lab']

	nurse_df = final_df.loc[final_df['job'] == 'nurse']
	day_nurse_df = nurse_df.loc[nurse_df['shift'] == 'day']
	night_nurse_df = nurse_df.loc[nurse_df['shift'] == 'night']
	icu_nurse_df = nurse_df.loc[nurse_df['icu'] == 'icu']
	non_icu_nurse_df = nurse_df.loc[nurse_df['icu'] == 'non_icu']

	# len1 = len(non_nurse_df.loc[(non_nurse_df['p'] > 0.25) | (non_nurse_df['p'] < -0.25)])
	# len2 = len(nurse_df.loc[(nurse_df['p'] > 0.25) | (nurse_df['p'] < -0.25)])
	len0 = len(non_nurse_df.loc[(non_nurse_df['p'] > 0.3)])
	len1 = len(lab_df.loc[(lab_df['p'] > 0.3)])
	len2 = len(nurse_df.loc[(nurse_df['p'] > 0.3)])
	
	'''
	len3 = len(day_nurse_df.loc[(day_nurse_df['p'] > 0.3) | (day_nurse_df['p'] < -0.3)])
	len4 = len(night_nurse_df.loc[(night_nurse_df['p'] > 0.3) | (night_nurse_df['p'] < -0.3)])
	len5 = len(icu_nurse_df.loc[(icu_nurse_df['p'] > 0.3) | (icu_nurse_df['p'] < -0.3)])
	len6 = len(non_icu_nurse_df.loc[(non_icu_nurse_df['p'] > 0.3) | (non_icu_nurse_df['p'] < -0.3)])
	'''
	
	len3 = len(day_nurse_df.loc[(day_nurse_df['p'] > 0.3)])
	len4 = len(night_nurse_df.loc[(night_nurse_df['p'] > 0.3)])
	len5 = len(icu_nurse_df.loc[(icu_nurse_df['p'] > 0.3)])
	len6 = len(non_icu_nurse_df.loc[(non_icu_nurse_df['p'] > 0.3)])
	
	print('\nnon_nurse (%d): %.2f' % (len(non_nurse_df), len0 / len(non_nurse_df) * 100))
	print('lab (%d): %.2f\n' % (len(lab_df), len1 / len(lab_df) * 100))

	print('nurse (%d): %.2f\n' % (len(nurse_df), len2 / len(nurse_df) * 100))
	print('day nurse (%d): %.2f' % (len(day_nurse_df), len3/ len(day_nurse_df) * 100))
	print('night nurse (%d): %.2f' % (len(night_nurse_df), len4 / len(night_nurse_df) * 100))
	print('icu nurse (%d): %.2f' % (len(icu_nurse_df), len5 / len(icu_nurse_df) * 100))
	print('non_icu nurse (%d): %.2f' % (len(non_icu_nurse_df), len6 / len(non_icu_nurse_df) * 100))

	data1_df = final_df.loc[(final_df['p'] > 0.3)]
	data2_df = final_df.loc[(final_df['p'] <= 0.3)]
	for col in igtb_cols:
		print(col)
		stat, p = stats.ks_2samp(data1_df[col].dropna(), data2_df[col].dropna())
		print('High sync: mean = %.2f, std = %.2f' % (np.mean(data1_df[col]), np.std(data1_df[col])))
		print('Low sync: mean = %.2f, std = %.2f' % (np.mean(data2_df[col]), np.std(data2_df[col])))
		print('K-S test for %s' % col)
		print('Statistics = %.3f, p = %.3f\n\n' % (stat, p))


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
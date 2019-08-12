"""
Compare IGTB data
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
from scipy import stats
from scipy.stats import skew

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']

def print_psqi_igtb(day_df, night_df, cols):

	print('\multicolumn{1}{l}{\\textbf{Total PSQI score}} &')
	print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(day_df['psqi_igtb'])))
	print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(day_df['psqi_igtb'])))
	print('\multicolumn{1}{c}{$%.2f$} &' % (skew(day_df['psqi_igtb'].dropna())))
	print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(night_df['psqi_igtb'])))
	print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(night_df['psqi_igtb'])))
	print('\multicolumn{1}{c}{$%.2f$} &' % (skew(night_df['psqi_igtb'].dropna())))

	stat, p = stats.ks_2samp(day_df['psqi_igtb'].dropna(), night_df['psqi_igtb'].dropna())

	if p < 0.001:
		print('\multicolumn{1}{c}{\mathbf{<0.001}} \\rule{0pt}{3ex} \\\\')
	elif p < 0.05:
		print('\multicolumn{1}{c}{$\mathbf{%.3f}$} \\rule{0pt}{3ex} \\\\' % (p))
	else:
		print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{3ex} \\\\' % (p))
	print('\n')

	for col in cols:

		if 'dysfunction' in col:
			print_col = 'Daytime dysfunction'
		elif 'med' in col:
			print_col = 'Sleep medication'
		elif 'sub' in col:
			print_col = 'Subjective sleep quality'
		elif 'tency' in col:
			print_col = 'Sleep latency'
		elif 'fficiency' in col:
			print_col = 'Sleep efficiency'
		elif 'disturbance' in col:
			print_col = 'Sleep disturbance'
		else:
			print_col = 'Sleep duration'

		print('\multicolumn{1}{l}{\hspace{0.5cm}' + print_col + '} &')
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(day_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(day_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (skew(day_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(night_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(night_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (skew(night_df[col])))

		stat, p = stats.ks_2samp(day_df[col].dropna(), night_df[col].dropna())

		if p < 0.001:
			print('\multicolumn{1}{c}{\mathbf{<0.001}} \\rule{0pt}{3ex} \\\\')
		elif p < 0.05:
			print('\multicolumn{1}{c}{$\mathbf{%.3f}$} \\rule{0pt}{3ex} \\\\' % (p))
		else:
			print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{3ex} \\\\' % (p))
		print('\n')


def print_personality_igtb(day_df, night_df, cols):

	print('\multicolumn{1}{l}{\\textbf{Personality}} & & & & & & \\rule{0pt}{3ex} \\\\')
	print('\n')

	for col in cols:

		if 'con_igtb' in col:
			print_col = 'Conscientiousness'
		elif 'ext_igtb' in col:
			print_col = 'Extraversion'
		elif 'agr_igtb' in col:
			print_col = 'Agreeableness'
		elif 'ope_igtb' in col:
			print_col = 'Openness to Experience'
		else:
			print_col = 'Neuroticism'

		print('\multicolumn{1}{l}{\hspace{0.5cm}' + print_col + '} &')
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(day_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(day_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (skew(day_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(night_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(night_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (skew(night_df[col])))

		stat, p = stats.ks_2samp(day_df[col].dropna(), night_df[col].dropna())

		if p < 0.001:
			print('\multicolumn{1}{c}{\mathbf{<0.001}} \\rule{0pt}{3ex} \\\\')
		elif p < 0.05:
			print('\multicolumn{1}{c}{$\mathbf{%.3f}$} \\rule{0pt}{3ex} \\\\' % (p))
		else:
			print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{3ex} \\\\' % (p))

		print()


def print_affect_igtb(day_df, night_df, cols):
	print('\multicolumn{1}{l}{\\textbf{Affect}} & & & & & & \\rule{0pt}{3ex} \\\\')
	print('\n')

	for col in cols:

		if 'neg_af_igtb' in col:
			print_col = 'Negative Affect'
		else:
			print_col = 'Positive Affect'

		print('\multicolumn{1}{l}{\hspace{0.5cm}' + print_col + '} &')
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(day_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(day_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (skew(day_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(night_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(night_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (skew(night_df[col])))

		stat, p = stats.ks_2samp(day_df[col].dropna(), night_df[col].dropna())

		if p < 0.001:
			print('\multicolumn{1}{c}{\mathbf{<0.001}} \\rule{0pt}{3ex} \\\\')
		elif p < 0.05:
			print('\multicolumn{1}{c}{$\mathbf{%.3f}$} \\rule{0pt}{3ex} \\\\' % (p))
		else:
			print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{3ex} \\\\' % (p))

		print()


def print_anxiety_igtb(day_df, night_df):
	print('\multicolumn{1}{l}{\\textbf{Anxiety}} &')
	print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(day_df['stai_igtb'])))
	print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(day_df['stai_igtb'])))
	print('\multicolumn{1}{c}{$%.2f$} &' % (skew(day_df['stai_igtb'].dropna())))
	print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(night_df['stai_igtb'])))
	print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(night_df['stai_igtb'])))
	print('\multicolumn{1}{c}{$%.2f$} &' % (skew(night_df['stai_igtb'].dropna())))

	stat, p = stats.ks_2samp(day_df['stai_igtb'].dropna(), night_df['stai_igtb'].dropna())

	if p < 0.001:
		print('\multicolumn{1}{c}{\mathbf{<0.001}} \\rule{0pt}{3ex} \\\\')
	elif p < 0.05:
		print('\multicolumn{1}{c}{$\mathbf{%.3f}$} \\rule{0pt}{3ex} \\\\' % (p))
	else:
		print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{3ex} \\\\' % (p))
	print()


def print_audit_igtb(day_df, night_df):
	print('\multicolumn{1}{l}{\\textbf{AUDIT score}} &')
	print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(day_df['audit_igtb'])))
	print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(day_df['audit_igtb'])))
	print('\multicolumn{1}{c}{$%.2f$} &' % (skew(day_df['audit_igtb'].dropna())))
	print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(night_df['audit_igtb'])))
	print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(night_df['audit_igtb'])))
	print('\multicolumn{1}{c}{$%.2f$} &' % (skew(night_df['audit_igtb'].dropna())))

	stat, p = stats.ks_2samp(day_df['audit_igtb'].dropna(), night_df['audit_igtb'].dropna())

	if p < 0.001:
		print('\multicolumn{1}{c}{\mathbf{<0.001}} \\rule{0pt}{3ex} \\\\')
	elif p < 0.05:
		print('\multicolumn{1}{c}{$\mathbf{%.3f}$} \\rule{0pt}{3ex} \\\\' % (p))
	else:
		print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{3ex} \\\\' % (p))
	print()


def print_cognition_igtb(day_df, night_df, cols):
	print('\multicolumn{1}{l}{\\textbf{Cognition}} & & & & & & \\rule{0pt}{3ex} \\\\')
	print('\n')

	for col in cols:

		if 'shipley_abs_igtb' in col:
			print_col = 'Shipley Abstraction'
		else:
			print_col = 'Shipley Vocabulary'

		print('\multicolumn{1}{l}{\hspace{0.5cm}' + print_col + '} &')
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(day_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(day_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (skew(day_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(night_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(night_df[col])))
		print('\multicolumn{1}{c}{$%.2f$} &' % (skew(night_df[col])))

		stat, p = stats.ks_2samp(day_df[col].dropna(), night_df[col].dropna())

		if p < 0.001:
			print('\multicolumn{1}{c}{\mathbf{<0.001}} \\rule{0pt}{3ex} \\\\')
		elif p < 0.05:
			print('\multicolumn{1}{c}{$\mathbf{%.3f}$} \\rule{0pt}{3ex} \\\\' % (p))
		else:
			print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{3ex} \\\\' % (p))

		print()


def main(tiles_data_path, config_path, experiment):
	# Create Config
	process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, os.pardir, 'data'))

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
	igtb_raw = load_data_basic.read_IGTB_Raw(tiles_data_path)

	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()

	num_of_sec = 6
	all_df = pd.DataFrame()

	compare_method = 'english'

	if compare_method == 'shift_str':
		first_str = 'day'
		second_str = 'night'
	elif compare_method == 'english':
		first_str = 'English'
		second_str = 'Non-English'
	elif compare_method == 'icu':
		first_str = 'icu'
		second_str = 'non_icu'
	elif compare_method == 'supervise_str':
		first_str = 'supervise'
		second_str = 'non-supervise'
	elif compare_method == 'Sex':
		first_str = 'Male'
		second_str = 'Female'

	for idx, participant_id in enumerate(top_participant_id_list[:]):

		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
		if os.path.exists(os.path.join('arousal', 'global', 'num_of_sec_' + str(num_of_sec), participant_id + '.pkl')) is False:
			continue

		nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
		primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
		shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
		uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
		language = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].language[0]

		job_str = 'nurse' if nurse == 1 else 'non_nurse'
		shift_str = 'day' if shift == 'Day shift' else 'night'
		language_str = 'English' if language == 1 else 'Non-English'

		icu_str = 'non_icu'
		for unit in icu_list:
			if unit in primary_unit:
				icu_str = 'icu'

		if 'ICU' in primary_unit:
			icu_str = 'icu'

		row_df = pd.DataFrame(index=[participant_id])
		row_df['job'] = job_str
		row_df['shift_str'] = shift_str
		row_df['english'] = language_str
		row_df['icu'] = icu_str

		for col in list(igtb_df.columns):
			row_df[col] = igtb_df.loc[uid, col]

		for col in list(psqi_raw_igtb.columns):
			row_df[col] = psqi_raw_igtb.loc[uid, col]

		for col in ['supervise']:
			if len(str(igtb_raw.loc[uid, col])) != 3:
				row_df['supervise_str'] = 'supervise' if int(igtb_raw.loc[uid, col]) == 1 else 'non-supervise'
			else:
				row_df[col] = np.nan

		all_df = all_df.append(row_df)

	# shift_pre-study
	nurse_df = all_df.loc[all_df['job'] == 'nurse']
	first_df = nurse_df.loc[nurse_df[compare_method] == first_str]
	second_df = nurse_df.loc[nurse_df[compare_method] == second_str]

	big5_col = ['neu_igtb', 'con_igtb', 'ext_igtb', 'agr_igtb', 'ope_igtb']
	for col in big5_col:
		print(col)
		print('Number of valid participant: %s: %i; %s: %i\n' % (first_str, len(first_df), second_str, len(second_df)))

		# Print
		print('Total: mean = %.2f, std = %.2f, range is %.3f - %.3f' % (np.mean(first_df[col]), np.std(first_df[col]), np.min(first_df[col]), np.max(first_df[col])))
		print('%s: mean = %.2f, std = %.2f' % (first_str, np.mean(first_df[col]), np.std(first_df[col])))
		print('%s: range is %.3f - %.3f' % (first_str, np.min(first_df[col]), np.max(first_df[col])))
		print('%s: skew = %.3f' % (first_str, skew(first_df[col])))

		print('%s: mean = %.2f, std = %.2f' % (second_str, np.mean(second_df[col]), np.std(second_df[col])))
		print('%s: range is %.3f - %.3f' % (second_str, np.min(second_df[col]), np.max(second_df[col])))
		print('%s: skew = %.3f' % (second_str, skew(second_df[col])))
		# K-S test
		stat, p = stats.ks_2samp(first_df[col].dropna(), second_df[col].dropna())
		print('K-S test for %s' % col)
		print('Statistics = %.3f, p = %.3f\n\n' % (stat, p))

	affect_col = ['stai_igtb', 'pos_af_igtb', 'neg_af_igtb']
	for col in affect_col:
		print(col)
		print('Number of valid participant: %s: %i; %s: %i\n' % (first_str, len(first_df), second_str, len(second_df)))

		print('Total: mean = %.2f, std = %.2f, range is %.3f - %.3f' % (np.mean(first_df[col]), np.std(first_df[col]), np.min(first_df[col]), np.max(first_df[col])))
		print('%s: mean = %.2f, std = %.2f' % (first_str, np.mean(first_df[col]), np.std(first_df[col])))
		print('%s: range is %.3f - %.3f' % (first_str, np.min(first_df[col]), np.max(first_df[col])))
		print('%s: skew = %.3f' % (first_str, skew(first_df[col])))

		print('%s: mean = %.2f, std = %.2f' % (second_str, np.mean(second_df[col]), np.std(second_df[col])))
		print('%s: range is %.3f - %.3f' % (second_str, np.min(second_df[col]), np.max(second_df[col])))
		print('%s: skew = %.3f' % (second_str, skew(second_df[col])))
		# K-S test
		stat, p = stats.ks_2samp(first_df[col].dropna(), second_df[col].dropna())
		print('K-S test for %s' % col)
		print('Statistics = %.3f, p = %.3f\n\n' % (stat, p))

	work_col = ['itp_igtb', 'irb_igtb', 'iod_id_igtb', 'iod_od_igtb', 'ocb_igtb', 'shipley_abs_igtb', 'shipley_voc_igtb']
	for col in work_col:
		print(col)
		print('Number of valid participant: %s: %i; %s: %i\n' % (first_str, len(first_df), second_str, len(second_df)))

		print('Total: mean = %.2f, std = %.2f, range is %.3f - %.3f' % (
		np.mean(first_df[col]), np.std(first_df[col]), np.min(first_df[col]), np.max(first_df[col])))
		print('%s: mean = %.2f, std = %.2f' % (first_str, np.mean(first_df[col]), np.std(first_df[col])))
		print('%s: range is %.3f - %.3f' % (first_str, np.min(first_df[col]), np.max(first_df[col])))
		print('%s: skew = %.3f' % (first_str, skew(first_df[col])))

		print('%s: mean = %.2f, std = %.2f' % (second_str, np.mean(second_df[col]), np.std(second_df[col])))
		print('%s: range is %.3f - %.3f' % (second_str, np.min(second_df[col]), np.max(second_df[col])))
		print('%s: skew = %.3f' % (second_str, skew(second_df[col])))
		# K-S test
		stat, p = stats.ks_2samp(first_df[col].dropna(), second_df[col].dropna())
		print('K-S test for %s' % col)
		print('Statistics = %.3f, p = %.3f\n\n' % (stat, p))


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'audio_location' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
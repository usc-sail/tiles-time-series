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
from scipy import stats
from scipy.stats import skew
import seaborn as sns
import matplotlib.ticker as mtick

sns.set(style="whitegrid")

compare_cols = ['itp_igtb', 'irb_igtb', 'ocb_igtb', 'stai_igtb', 'pos_af_igtb', 'neg_af_igtb', 'Pain', 'energy_fatigue']


def plot_comparison(first_df, second_df, plot_df, data_col, axes, cond, hue_str):
	plt.rcParams["font.weight"] = "bold"
	if 'Total Usage Time' in data_col:
		raw_col = 'total_time'
		max_y = 50000
		idx = 0
		y_lable_text = 'Seconds'
	elif 'Average Session Length' in data_col:
		raw_col = 'mean_time'
		max_y = 1000
		idx = 1
		y_lable_text = 'Seconds'
	elif 'Average Inter-session Time' in data_col:
		raw_col = 'mean_inter'
		max_y = 4000
		idx = 2
		y_lable_text = 'Seconds'
	elif '(<1min)' in data_col:
		raw_col = 'less_than_1min'
		max_y = 150
		idx = 4
		y_lable_text = 'Number of Sessions'
	else:
		raw_col = 'above_1min'
		max_y = 100
		idx = 5
		y_lable_text = 'Number of Sessions'
	
	sns.boxplot(x="Data Type", y=data_col, hue=cond, data=plot_df, palette=hue_str, ax=axes)
	lgd = axes.legend()
	lgd.set_title('')
	axes.set_xlabel('')
	if cond == 'Management':
		axes.set_ylabel(y_lable_text, fontweight='bold', fontsize=15)
		if 'Total Usage Time' in data_col:
			axes.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
	else:
		axes.set_ylabel('')
		axes.set_yticklabels(labels='')
		
	axes.set_title(cond, fontweight='bold', fontsize=15)
	
	label_list = []
	for ticklabel in axes.get_xticklabels():
		if ticklabel.get_text() == 'Non-workday':
			final_data_col = raw_col + '_off_mean'
		elif ticklabel.get_text() == 'Workday':
			final_data_col = raw_col + '_work_mean'
		else:
			final_data_col = raw_col + '_mean'

		stat, p = stats.ks_2samp(first_df[final_data_col].dropna(), second_df[final_data_col].dropna())
		
		if p < 0.001:
			label_list.append(ticklabel.get_text() + '\n(p<0.001)')
		else:
			label_list.append(ticklabel.get_text() + '\n(p=' + str(p)[:5] + ')')
	
	axes.set_xticklabels(labels=label_list, fontweight='bold', fontsize=15)
	axes.set_ylim([0, max_y])
	axes.legend(loc='center right', bbox_to_anchor=(0.02, 0.80, 0.99, .1), prop={'size': 12})
	
	axes.yaxis.set_tick_params(size=1)
	axes.tick_params(axis="y", labelsize=15)
	axes.grid(linestyle='--')
	axes.grid(False, axis='x')
	axes.tick_params(axis="y", labelsize=15)
	# sns.boxplot(x="time", y=data_col, hue='loc', data=data_df, palette="seismic", ax=axes)
	

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
	igtb_raw = load_data_basic.read_IGTB_Raw(tiles_data_path)
	
	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()

	# compare_method_list = ['shift', 'language', 'supervise', 'gender', 'icu', 'job', 'children', 'age', 'employer_duration']
	compare_method_list = ['supervise', 'icu', 'children', 'age']
	# realizd_col_list = ['total_time', 'mean_time', 'mean_inter', 'less_than_1min', 'above_1min']
	realizd_col_list = ['Total Usage Time', 'Average Session Length', 'Average Inter-session Time',
						'Session Frequency (<1min)', 'Session Frequency (>1min)']
	
	if os.path.exists(os.path.join('daily_realizd_fitbit_mr.csv.gz')) is True:
		final_df = pd.read_csv(os.path.join('daily_realizd_fitbit_mr.csv.gz'), index_col=0)
		nurse_df = final_df.loc[final_df['job'] == 'nurse']
	
		print(len(nurse_df))
		
		plot_df = pd.DataFrame()
		for i in range(len(nurse_df)):
			participant_df = nurse_df.iloc[i, :]
			participant_id = list(nurse_df.index)[i]
			row_df = pd.DataFrame(index=[participant_id])
			row_df['Total Usage Time'] = participant_df['total_time_mean']
			row_df['Average Session Length'] = participant_df['mean_time_mean']
			row_df['Average Inter-session Time'] = participant_df['mean_inter_mean']
			row_df['Session Frequency (<1min)'] = participant_df['less_than_1min_mean']
			row_df['Session Frequency (>1min)'] = participant_df['above_1min_mean']
			row_df['Data Type'] = 'Overall'
			
			if participant_df['supervise'] == 'Supervise':
				row_df['Management'] = 'Manager'
			else:
				row_df['Management'] = 'Non-manager'
				
			if participant_df['children'] > 0:
				row_df['Have Child or Not'] = 'Have Child'
			else:
				row_df['Have Child or Not'] = 'Don\'t Have Child'
			
			if participant_df['age'] > np.nanmedian(nurse_df['age']):
				row_df['Age'] = '>= Med. Age'
			else:
				row_df['Age'] = '< Med. Age'
				
			if participant_df['icu'] == 'icu':
				row_df['ICU/Non-ICU'] = 'ICU'
			else:
				row_df['ICU/Non-ICU'] = 'Non-ICU'
			plot_df = plot_df.append(row_df)
			
			row_df = pd.DataFrame(index=[participant_id])
			row_df['Total Usage Time'] = participant_df['total_time_work_mean']
			row_df['Average Session Length'] = participant_df['mean_time_work_mean']
			row_df['Average Inter-session Time'] = participant_df['mean_inter_work_mean']
			row_df['Session Frequency (<1min)'] = participant_df['less_than_1min_work_mean']
			row_df['Session Frequency (>1min)'] = participant_df['above_1min_work_mean']
			row_df['Data Type'] = 'Workday'
			
			if participant_df['supervise'] == 'Supervise':
				row_df['Management'] = 'Manager'
			else:
				row_df['Management'] = 'Non-manager'
			
			if participant_df['children'] > 0:
				row_df['Have Child or Not'] = 'Have Child'
			else:
				row_df['Have Child or Not'] = 'Don\'t Have Child'
			
			if participant_df['age'] > np.nanmedian(nurse_df['age']):
				row_df['Age'] = '>= Med. Age'
			else:
				row_df['Age'] = '< Med. Age'
			
			if participant_df['icu'] == 'icu':
				row_df['ICU/Non-ICU'] = 'ICU'
			else:
				row_df['ICU/Non-ICU'] = 'Non-ICU'
			plot_df = plot_df.append(row_df)
			
			row_df = pd.DataFrame(index=[participant_id])
			row_df['Total Usage Time'] = participant_df['total_time_work_mean']
			row_df['Average Session Length'] = participant_df['mean_time_work_mean']
			row_df['Average Inter-session Time'] = participant_df['mean_inter_work_mean']
			row_df['Session Frequency (<1min)'] = participant_df['less_than_1min_work_mean']
			row_df['Session Frequency (>1min)'] = participant_df['above_1min_work_mean']
			row_df['Data Type'] = 'Non-workday'
			
			if participant_df['supervise'] == 'Supervise':
				row_df['Management'] = 'Manager'
			else:
				row_df['Management'] = 'Non-manager'
			
			if participant_df['children'] > 0:
				row_df['Have Child or Not'] = 'Have Child'
			else:
				row_df['Have Child or Not'] = 'Don\'t Have Child'
			
			if participant_df['age'] > np.nanmedian(nurse_df['age']):
				row_df['Age'] = '>= Med. Age'
			else:
				row_df['Age'] = '< Med. Age'
			
			if participant_df['icu'] == 'icu':
				row_df['ICU/Non-ICU'] = 'ICU'
			else:
				row_df['ICU/Non-ICU'] = 'Non-ICU'
			plot_df = plot_df.append(row_df)
		
		for data_col in realizd_col_list:
			fig = plt.figure(figsize=(20, 3.25))
			axes = fig.subplots(nrows=1, ncols=len(compare_method_list))
			for i, compare_method in enumerate(compare_method_list):
				if compare_method == 'supervise':
					first_data_df = nurse_df.loc[nurse_df['supervise'] == 'Supervise']
					second_data_df = nurse_df.loc[nurse_df['supervise'] == 'Non-Supervise']
					cond_list = ['Nurse Manager', 'Non-nurse Manager']
					compare_str = 'Management'
					hue_str = 'seismic'
				elif compare_method == 'children':
					first_data_df = nurse_df.loc[final_df['children'] > 0]
					second_data_df = nurse_df.loc[final_df['children'] == 0]
					cond_list = ['Have Child', 'Don\'t Have Child']
					compare_str = 'Have Child or Not'
					hue_str = 'PuRd'
				elif compare_method == 'age':
					first_data_df = nurse_df.loc[nurse_df['age'] > np.nanmedian(nurse_df['age'])]
					second_data_df = nurse_df.loc[nurse_df['age'] <= np.nanmedian(nurse_df['age'])]
					cond_list = ['>= Median Age', '< Median Age']
					compare_str = 'Age'
					hue_str = 'YlGn'
				else:
					first_data_df = nurse_df.loc[nurse_df['icu'] == 'icu']
					second_data_df = nurse_df.loc[nurse_df['icu'] == 'non_icu']
					cond_list = ['ICU Unit', 'Non-ICU Unit']
					compare_str = 'ICU/Non-ICU'
					hue_str = 'BrBG'
				
				plot_comparison(first_data_df, second_data_df, plot_df, data_col, axes[i], compare_str, hue_str)
			plt.tight_layout()
			fig.subplots_adjust(top=0.825)
			fig.suptitle(data_col, fontsize=16, weight='bold', y=0.98, x=0.52)
			if os.path.exists(os.path.join('plot')) is False:
				os.mkdir(os.path.join('plot'))
			if os.path.exists(os.path.join('plot', 'daily_usage')) is False:
				os.mkdir(os.path.join('plot', 'daily_usage'))
			plt.savefig(os.path.join('plot', 'daily_usage', data_col.replace(' ', '_') + '.png'), dpi=300)


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()
	
	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment
	
	main(tiles_data_path, config_path, experiment)
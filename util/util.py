import os
import glob
import json
import pytz
import time
import datetime
import numpy as np
import pandas as pd
try:
   from StringIO import StringIO
except ImportError:
   from io import StringIO

zulu_timezone = pytz.utc
local_timezone = pytz.timezone("America/Los_Angeles")

# This function reads in a csv file and drops duplicate entries in time keeping
# the last entry
def DropCsvDuplicates(out_folder):
   csv_files = glob.glob(os.path.join(out_folder, '*.csv'))
   for csv_file in csv_files:
      data_frame = pd.read_csv(csv_file, sep=',')
      data_frame = data_frame.drop_duplicates(keep='last')
      data_frame.to_csv(csv_file, index=False, sep=',')
   return

def ReverseCsvRows(out_folder):
   csv_files = glob.glob(os.path.join(out_folder, '*.csv'))
   for csv_file in csv_files:
      data_frame = pd.read_csv(csv_file, sep=',')
      data_frame = data_frame.reindex(index=data_frame.index[::-1])
      data_frame.to_csv(csv_file, index=False, sep=',')
   return

def SortCsvRows(out_folder):
   csv_files = glob.glob(os.path.join(out_folder, '*.csv'))
   for csv_file in csv_files:
      data_frame = pd.read_csv(csv_file, sep=',')
      data_frame.sort_values(by=['Timestamp'], inplace=True)
      data_frame.to_csv(csv_file, index=False, sep=',')
   return

def GetLocalTimestamp(zulu_timestamp_list, parse_expr='%Y-%m-%dT%H:%M:%S.%fZ'):
   is_list = isinstance(zulu_timestamp_list, list) or isinstance(zulu_timestamp_list, np.ndarray)
   if not is_list:
      zulu_timestamp_list = [zulu_timestamp_list]
   if '%z' in parse_expr:
      zulu_timestamp_list = [t[:-3]+t[-2:] if t[-3:-2] == ':' else t for t in zulu_timestamp_list] # Workaround bug with +00:00 format vs. +0000
   zulu_times = [zulu_timezone.localize(datetime.datetime.now().strptime(t, parse_expr)) for t in zulu_timestamp_list]
   local_times = [t.astimezone(local_timezone) for t in zulu_times]
   time_stamps = [t.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] for t in local_times]
   return time_stamps if is_list else time_stamps[0]

def GetLocalTimestampFromUnixTime(unix_time_list):
   is_list = isinstance(unix_time_list, list) or isinstance(unix_time_list, np.ndarray)
   if not is_list:
      unix_time_list = [unix_time_list]

   utc_times = [zulu_timezone.localize(datetime.datetime.utcfromtimestamp(t)) for t in unix_time_list]
   local_times = [t.astimezone(local_timezone) for t in utc_times]
   time_stamps = [t.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] for t in local_times]
   return time_stamps if is_list else time_stamps[0]

def GetUnixTimeFromTimestamp(timestamp_list, parse_expr='%Y-%m-%dT%H:%M:%S.%f'):
   is_list = isinstance(timestamp_list, list) or isinstance(timestamp_list, np.ndarray)
   if not is_list:
      timestamp_list = [timestamp_list]
   timestamp_datetimes = [datetime.datetime.now().strptime(t, parse_expr) for t in timestamp_list]
   unix_times = [time.mktime(dt.timetuple()) for dt in timestamp_datetimes]
   return unix_times if is_list else unix_times[0]

def LoadJson(json_lines):
   df = pd.DataFrame([json.loads(line) for line in json_lines])
   return df

def LoadCsv(csv_lines, column_names=None):
   if column_names is not None:
      df = pd.read_csv(StringIO('\n'.join(csv_lines)), sep=',', header=None)
      df.columns = column_names
   else:
      df = pd.read_csv(StringIO('\n'.join(csv_lines)), sep=',')
   return df

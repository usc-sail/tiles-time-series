## 1. extract_work_schedule.py
#### The script is to extract start and end recording time of om signal

#### Sample usage cmd, please modify based your own usage, it is to extract start and end recording of om signal for each recording session: <br />
<br />

```

python3.6 extract_work_schedule.py -i ../../data -o ../output -v 6 

-i: main_data_path

-o: output path

-v: minimum gap between different recording
 
```
#### Sample saved csv format

user_id  |  work_shift_type  |  recording_date  |  start_recording_time     |  end_recording_time
---------|-------------------|------------------|---------------------------|-------------------------
use_id   |  1                |  2018-03-08      |  2018-03-08T06:51:00.000  |  2018-03-08T19:33:24.000
use_id   |  1                |  2018-03-09      |  2018-03-09T06:47:37.000  |  2018-03-09T19:34:26.000
use_id   |  1                |  2018-03-12      |  2018-03-12T06:45:51.000  |  2018-03-12T19:36:45.000
use_id   |  0                |  2018-03-13      |  2018-03-13T06:51:16.000  |  2018-03-13T20:05:09.000
use_id   |  0                |  2018-03-22      |  2018-03-22T06:44:44.000  |  2018-03-22T19:41:26.000

#### Colunms defination:

1.  **use_id:** <br />
user_id of participant
2.  **work_shift_type:** <br />
0, night shift nurse; 1, day shift nurse
3.  **recording_date:** <br />
4.  **start_recording_time:** <br />
5.  **end_recording_time:** <br />

### Issues/Artifacts on output

1.  **Participants are not wearing OM signal on time** <br />
2.  **Timestamp error on OM signal output** <br />
3.  **Other unknown artifacts** <br />

<br />

## 2. extract_sleep_timeline.py
#### The script is to extract sleep timeline and sleep quality metrics

#### Sample usage cmd, please modify based your own usage, it is to extract sleep start and end time as well sleep quality(nan if there is no corresponding data for a sleep) for each recording session: <br />
<br />

```

python3.6 extract_sleep_timeline.py -t combined -i ../../data -o ../output

-t: combined data type

    direct reading of sleep summary; indiret inferred from Fitbit step count; combine both

-i: main_data_path

-o: output path
 
```

#### Sample saved csv format

SleepBeginTimestamp      |  SleepEndTimestamp        |  SleepMinutesAwake  |  SleepMinutesStageDeep  |  SleepMinutesStageLight  |  SleepMinutesStageRem  |  SleepMinutesStageWake  |  SleepEfficiency  |  data_source
-------------------------|---------------------------|---------------------|-------------------------|--------------------------|------------------------|-------------------------|-------------------|-------------
2018-03-04T23:16:00.000  |  2018-03-05T07:09:00.000  |  63.0               |  65.0                   |  265.0                   |  80.0                  |  63.0                   |  94.0             |  1
2018-03-05T23:04:30.000  |  2018-03-06T07:17:30.000  |  50.0               |  59.0                   |  308.0                   |  76.0                  |  50.0                   |  97.0             |  1
2018-03-06T23:09:30.000  |  2018-03-07T06:18:00.000  |  41.0               |  59.0                   |  223.0                   |  105.0                 |  41.0                   |  96.0             |  1
2018-03-08T01:53:00.000  |  2018-03-08T10:55:30.000  |  68.0               |  81.0                   |  265.0                   |  128.0                 |  68.0                   |  92.0             |  1

#### Colunms defination:

1.  **SleepBeginTimestamp:** <br />
2.  **SleepEndTimestamp:** <br />
3.  **SleepMinutesAwake:** <br />
4.  **SleepMinutesStageDeep:** <br />
5.  **SleepMinutesStageLight:** <br />
6.  **SleepMinutesStageRem:** <br />
7.  **SleepMinutesStageWake:** <br />
8.  **SleepEfficiency:** <br />
9.  **data_source:** <br />
    0, step count; 1, summary

### Issues/Artifacts on output

1.  **Participants are sleeping without Fitbit** <br />
2.  **Unknow issues on Fitbit** <br />

 <br />

## 3. extract_work_day_sleep_pattern.py

#### The script is to extract sleep timeline and sleep quality metrics on workdays based on OM signal recordings

#### Sample usage cmd, please modify based your own usage, it is to extract sleep start and end time as well sleep quality(nan if there is no corresponding data for a sleep) on work days: <br />
<br />

```

python3.6 extract_work_day_sleep_pattern.py -t combined -i ../../data -o ../output

-t: data type

    combined; summary

-i: main_data_path

-o: output path

```

#### Sample saved csv format

user_id  |  work_date  |  work_shift_type  |  wake_before_work_standard_work_time  |  sleep_after_work_standard_work_time  |  wake_before_work_om_signal_start_time  |  sleep_after_work_om_signal_end_time  |  sleep_before_work_SleepBeginTimestamp  |  sleep_before_work_SleepEndTimestamp  |  sleep_before_work_DataResource  |  start_work_time  |  start_recording_time     |  end_work_time  |  end_recording_time       |  sleep_after_work_SleepBeginTimestamp  |  sleep_after_work_SleepEndTimestamp  |  sleep_after_work_DataResource  |  sleep_before_work_MinutesAwake  |  sleep_before_work_MinutesStageDeep  |  sleep_before_work_MinutesStageLight  |  sleep_before_work_MinutesStageRem  |  sleep_before_work_MinutesStageWake  |  sleep_before_work_Efficiency  |  sleep_after_work_MinutesAwake  |  sleep_after_work_MinutesStageDeep  |  sleep_after_work_MinutesStageLight  |  sleep_after_work_MinutesStageRem  |  sleep_after_work_MinutesStageWake  |  sleep_after_work_Efficiency
---------|-------------|-------------------|---------------------------------------|---------------------------------------|-----------------------------------------|---------------------------------------|-----------------------------------------|---------------------------------------|----------------------------------|-------------------|---------------------------|-----------------|---------------------------|----------------------------------------|--------------------------------------|---------------------------------|----------------------------------|--------------------------------------|---------------------------------------|-------------------------------------|--------------------------------------|--------------------------------|---------------------------------|-------------------------------------|--------------------------------------|------------------------------------|-------------------------------------|-----------------------------
SD1006   |  3/8/18     |  1                |  1.06                                 |  3.37                                 |  0.91                                   |  2.81                                 |  2018-03-07T23:13:00.000                |  2018-03-08T05:56:30.000              |  1                               |  3/8/18 7:00      |  2018-03-08T06:51:00.000  |  3/8/18 19:00   |  2018-03-08T19:33:24.000  |  2018-03-08T22:22:00.000               |  2018-03-09T06:12:30.000             |  1                              |  41                              |  59                                  |  194                                  |  109                                |  41                                  |  91                            |  73                             |  76                                 |  214                                 |  107                               |  73                                 |  90
SD1006   |  3/9/18     |  1                |  0.79                                 |  6.42                                 |  0.59                                   |  5.85                                 |  2018-03-08T22:22:00.000                |  2018-03-09T06:12:30.000              |  1                               |  3/9/18 7:00      |  2018-03-09T06:47:37.000  |  3/9/18 19:00   |  2018-03-09T19:34:26.000  |  2018-03-10T01:25:30.000               |  2018-03-10T08:20:00.000             |  1                              |  73                              |  76                                  |  214                                  |  107                                |  73                                  |  90                            |  45                             |  52                                 |  216                                 |  101                               |  45                                 |  96
SD1006   |  3/12/18    |  1                |                                       |  4.59                                 |                                         |  3.98                                 |                                         |                                       |  1                               |  3/12/18 7:00     |  2018-03-12T06:45:51.000  |  3/12/18 19:00  |  2018-03-12T19:36:45.000  |  2018-03-12T23:35:30.000               |  2018-03-13T07:10:00.000             |  1                              |                                  |                                      |                                       |                                     |                                      |                                |  60                             |  90                                 |  234                                 |  70                                |  60                                 |  88
SD1006   |  3/13/18    |  1                |                                       |  4.92                                 |                                         |  3.84                                 |                                         |                                       |  1                               |  3/13/18 7:00     |  2018-03-13T06:51:16.000  |  3/13/18 19:00  |  2018-03-13T20:05:09.000  |  2018-03-13T23:55:30.000               |  2018-03-14T07:49:00.000             |  1                              |                                  |                                      |                                       |                                     |                                      |                                |  51                             |  76                                 |  206                                 |  140                               |  51                                 |  94

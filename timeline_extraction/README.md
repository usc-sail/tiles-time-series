### 1. extract_work_schedule.py
#### The script is to extract start and end recording time of om signal

#### Sample usage cmd, please modify based your own usage, it is to extract start and end recording of om signal for each recording session: <br />
<br />

```
python3.6 extract_work_schedule.py -i ../../data/keck_wave1/2_preprocessed_data/omsignal/omsignal -g ../../data/keck_wave1/2_preprocessed_data/ground_truth -j ../../data/keck_wave1/2_preprocessed_data/job\ shift -o ../output/om_signal_timeline -v 6 

-i: omsignal data path

-g: groundtruth path

-j: job shift path

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

### 2. extract_sleep_timeline.py
#### The script is to extract sleep timeline and sleep quality metrics

#### Sample usage cmd, please modify based your own usage, it is to extract start and end recording of om signal for each recording session: <br />
<br />

```
python3.6 extract_sleep_timeline.py -t combined -f ../../data/keck_wave1/2_preprocessed_data/fitbit/fitbit -g ../../data/keck_wave1/2_preprocessed_data/ground_truth -i ../output/om_signal_timeline -o ../output/sleep_timeline

-t: combined data type

-g: groundtruth path

-f: Fitbit data path

-i: om signal timeline

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


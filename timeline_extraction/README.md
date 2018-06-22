## The script is to extract start and end recording time of om signal

#### Sample usage cmd, please modify based your own usage, it is to extract start and end recording of om signal for each recording session: <br />
<br />

```
python3.6 extract_work_schedule.py -i ../../data/keck_wave1/2_preprocessed_data/omsignal/omsignal -g ../../data/keck_wave1/2_preprocessed_data/ground_truth -j ../../data/keck_wave1/2_preprocessed_data/job\ shift -o ../output/om_signal_timeline -v 6
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


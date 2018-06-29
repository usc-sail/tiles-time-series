## 1. extract_sleep_timeline.py
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

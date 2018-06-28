# TILES-Daily-Inference

#### This is a repo for TILES project, topic is regarding daily activity inference.

### Prerequisites

Mainly used:

* Python3
* [Pandas](http://pandas.pydata.org/pandas-docs/version/0.15/index.html) -- `pip3 install pandas`
* [Numpy](http://www.numpy.org/) -- `pip3 install numpy`


### Recommended deploy file hierarchy

```
.
├── 
├── TILES-Daily-Inference
│   ├── output
│   │   ├── om_signal_timeline                  # OM Signal recording start and end timeline in csv, output by    extract_work_schedule.py
│   │   ├── sleep_routine_work                  # work day sleep pattern data in csv, output by extract_work_day_sleep_pattern.py
│   │   └── sleep_timeline                      # sleep timeline in csv, output by extract_sleep_timeline.py
│   └── timeline_extraction
|       ├── README.md                           # ReadME, usage of the scripts
|       ├── extract_work_day_sleep_pattern.sh   # Extract work day sleep pattern bash script
|       ├── extract_sleep_timeline.py           # Extract sleep timeline and summary
|       ├── extract_work_day_sleep_pattern.py   # Extract work day sleep pattern
|       ├── extract_work_schedule.py            # Extract work related recording timeline
└── data
```
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Authors

* **Tiantian Feng (Equal contribution)** 
* **Brandon Booth (Equal contribution)** 
* **Karel Mundnich (Equal contribution)** 

**Feel free to contact me if you want to be a collaborator.**

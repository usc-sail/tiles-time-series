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
├── Root
├── TILES-Daily-Inference
│   ├── output
│   │   ├── om_signal_timeline
│   │   ├── sleep_routine_work
│   │   └── sleep_timeline
│   └── timeline_extraction
|       ├── README.md                           # readme, usage of the scripts
|       ├── extract_sleep_timeline.py           # Extract sleep timeline and summary
|       ├── extract_work_day_sleep_pattern.py   # Extract work day sleep pattern
|       ├── extract_work_schedule.py            # Extract work related recording timeline
└── data
```
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Authors

* **Tiantian Feng** 
* **Brandon Booth** 

**Feel free to contact me if you want to be a collaborator.**

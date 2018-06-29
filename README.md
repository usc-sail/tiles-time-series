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
├── run.sh                                      # Sample usage: sh run.sh ../../data ../output
│   ├── output
│   │   ├── individual_timeline                 # individual timeline
│   │   ├── recording_timeline                  # om, owl-in-one, ground-truth based start and end time
│   │   ├── days_at_work                        # which dates people is at work
│   │   └── sleep_timeline                      # sleep timeline in csv
│   │
│   └── sleep_timeline
|   │   ├── README.md                           # ReadME, usage of the scripts
|   │   ├── extract_sleep_timeline.py           # Extract sleep timeline and summary
│   │
│   └── work_timeline (recording_timeline)
|   |   ├── README.md                           # ReadME, usage of the scripts
|   |   ├── signal_recording_start_end.py       # Extract recording start and end
|   |   ├── days_at_work.py                     # Extract days at work
│   │
│   └── individual_timeline
|   |   ├── README.md                           # ReadME, usage of the scripts
|   |   ├── individual_timeline.py              # Extract individual timeline
│   │
│   └── util (recording_timeline)
|       ├── files.py                            # Get common files path
|       ├── load_data_basic.py                  # Load basic information like participant id - user id mapping
└── data
```
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Authors

* **Tiantian Feng (Equal contribution)** 
* **Brandon Booth (Equal contribution)** 
* **Karel Mundnich (Equal contribution)** 

**Feel free to contact me if you want to be a collaborator.**

# config file creator

#### 1. Quick start - create config files in config_file folder naming baseline.cfg
```
python3 create_config.py --experiment baseline
```

#### 2. Items in config file (just showed fitbit here, other streams are similar and mostly with fewer params needed)

```
fitbit_param = {'data_type': 'fitbit', 
                'imputation': 'iterative', 
                'feature': 'original', 
                'offset': 60, 'overlap': 0,
                'preprocess_cols': ['HeartRatePPG', 'StepCount'],
                'cluster_method': 'ticc', 'num_cluster': 5, 
                'ticc_window': 10, 'ticc_switch_penalty': 100, 'ticc_sparsity': 1e-2,
                'segmentation_method': '', 'segmentation_lamb': 10e0}
```

#### Dictionary
```
data_type - data type, 'fitbit', 'om_signal', 'realizd', 'owl_in_one', 'audio'
```

```
imputation - imputation in preprocess, 'iterative', 'knn', 'rnn' (will add soon)
```

```
feature - what kind of feature we would like to get in preprocess, 'original'
```

```
offset  - aggregation window in preprocess, unit in seconds
overlap - aggregation window overlaps in preprocess, unit in seconds
```

```
preprocess_cols - cols to preprocess in data stream
```

```
cluster_method - clustering method on segmented data or preprocess data (if segmentation is None or ''), 
                 'ticc', 'casc', 'knn', others like ordinary clustering will be included soon
```

```
segmentation_method - clustering method on segmented data or preprocess data, '' or None(no segmentation performed), 'ggs'
```



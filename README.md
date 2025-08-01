# Granularity 

We want to compute the available metrics at a given granularity

## Rules
We want to compute the minimum granularity of a set metrics.
We can always downsample (never upsample)

If this isn't provided - we provide metrics of all provided granularities

if we don't provide the selected granularity
1. get the range of granularities i.e. 10d, 1m, 3m
2. for each granularity compute the metrics that we can.
3. for the next granularity - identify if we have a higher granularity input if so, downsample.


## Algorithm

### Preprocessing 
1. If selected gran hasn't been provided i.e [1m, 3m, 6m], automatically compute range of available granularities.

2. for each granularity in granularity range, check if we can get the temperature, salinity, SSH, U, V.

3. Resample from higher frequency data if it doesn't already exist. Provide option to store lazily or compute eagerly. 
(Some files can be 7GB or more - particularly if sampled at high frequency). Cache as appropriate. 

### Evaluate metrics
1. Loop over granularity range
2. If...

this just picks up the range of granularities i.e. 10d, 1m, 3m
we could use this upper limit to just run at a single granularity. otherwise run across all?

## Options
```
options =  ["all", "min"]
```

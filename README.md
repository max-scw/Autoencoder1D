# Autoencoder1D
PyTorch-based autoencoder for 1D signals

## Quickstart
The project provides two Python-based command line tools in [train.py](train.py) and [encode.py](encode.py).

It builds an autoencoder of depth `--depth` for signals of length `--signal-len`. 
Use CSV files or [Apache Parquet](https://parquet.apache.org/) files to feed signals to the autoencoder. For CSV files, use a text file that provides the path to the CSV files. An Apache Parquet file is a single large table, i.e. usually the concatenated version of all individual files. Therefore, provide the `--groupby` argument to specify the column that denotes to which record the values belong.

The project was developed with [Python 3.11](https://www.python.org/downloads/release/python-3119/) and [PyTorch 2.4.0](https://pytorch.org/blog/pytorch2-4/).

## Overview

Train an autoencoder from command line like this:

```shell
python train.py --epochs 200 --data DataTables.parquet  --groupby name --signal-len 11264 --depth 7 --batch-size 128 --normalize
```
Assuming an Apache Parquet file with a table like:

| Signal 1 | Signal 2 | name    |
|----------|----------|---------|
| 1        | 2        | "File1" |
| 1        | 2        | "File1" |
| 1        | 3        | "File2" |


This will save a file '<YEAR><MONTH><DAY>-<HOUR><MINUTE>_autoencoder.pth' than can be used to actually encode the same data with:

```shell
python encode.py --checkpoint <YEAR><MONTH><DAY>-<HOUR><MINUTE>_autoencoder.pth --data DataTables.parquet --batch-size 256 --workers 2
```

The results are saved as CSV file with the same name as the same name as the checkpoint + name of the data file.

### Project structure

````
Autoencoder1D
+-- utils
    |-- data.py  # wrapper for the specialized dataset classes
    |-- DatasetCSV.py  # torch-Dataset-based class for reading data from CSV files
    |-- DatasetParquet.py  # torch-Dataset-based class for reading a single parquet table, splitting into pandas.DataFrames by a <groupby> column
    |-- general.py  # save / load model
    |-- signals.py  # normalization and running statistics
|-- encode.py  # command line tool to encode data using existing weights for an autoencoder
|-- LICENSE
|-- models.py  # builds an autoencoder with 1D convolutions
|-- README.md
|-- requirements.txt
|-- train.py  # command line tool to train a data loader
````

## Acknowledgements

- Author max-scw

## Status

active

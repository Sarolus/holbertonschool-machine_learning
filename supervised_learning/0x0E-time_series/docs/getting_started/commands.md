[Documentation](../README.md) > [Getting started](README.md) > Commands

# Commands

## ![](https://img.shields.io/badge/import_csv-informational?style=flat&logoColor=white)

> **Description**: Import the content of a CSV file to a database

```bash
import_csv
    positional arguments:
        file_path     Path to the file to import
        currency      Currency to import
```

## ![](https://img.shields.io/badge/train-informational?style=flat&logoColor=white)

> **Description**: Train a model from dataset

```bash
train
    options:
        --epochs EPOCHS, -e EPOCHS                          Number of epochs
        --dataset_type DATASET_TYPE, -dst DATASET_TYPE      Dataset type (csv, db)
        --model_type MODEL_TYPE, -m MODEL_TYPE              Model type
        --model_path MODEL_PATH, -mp MODEL_PATH             Model path
        --csv_file_path CSV_FILE_PATH, -cfp CSV_FILE_PATH   CSV file path
        --graphs GRAPHS, -g GRAPHS                          Graphs
```

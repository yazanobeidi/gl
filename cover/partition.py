import argparse
import os
import pandas as pd
import numpy as np

def partition_covertype(path_to_train_set,
                        path_to_test_set,
                        path_to_validation_set,
                        output_folder,
                        partition_key,
                        original_train_test_split=True,
                        verbose=False):
    """
    This function transforms the original "centralized" Covertype dataset that
    contains all 4 wilderness areas into 4 "decentralized" datasets,
    i.e. separated private data silos.
    > Original dataset source: https://archive.ics.uci.edu/ml/datasets/covertype
    > Please see the covtype.info dataset description for more information.
    """
    train_set = pd.read_csv(path_to_train_set)
    test_set = pd.read_csv(path_to_test_set)
    validation_set = pd.read_csv(path_to_validation_set)

    train_dfs, test_dfs, val_dfs = list(), list(), list()

    def filter(df, label):
        return df.drop(df[df[partition_key] != label].index)

    for n, label in enumerate(np.unique(train_set[partition_key])):
        # note these are in-distribution test and validation sets
        filter(train_set, label).to_csv(os.path.join(output_folder, f'covertype11_train_{label}.csv'), index=False)
        filter(test_set, label).to_csv(os.path.join(output_folder,f'covertype11_test_{label}.csv'), index=False)
        filter(validation_set, label).to_csv(os.path.join(output_folder,f'covertype11_validation_{label}.csv'), index=False)


if __name__ == "__main__":
    default_partition_key = "wilderness_area"
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-train-set", type=str, default="cover/data/covertype11_train.csv")
    parser.add_argument("--path-to-test-set", type=str, default="cover/data/covertype11_test.csv")
    parser.add_argument("--path-to-validation-set", type=str, default="cover/data/covertype11_validation.csv")
    parser.add_argument("--partition-key", type=str, default=default_partition_key)
    parser.add_argument("--output-folder", type=str, default="cover/data/")
    parser.add_argument("--verbose", type=bool, default=False)

    args = parser.parse_args()

    if args.partition_key != default_partition_key:
        print(f'Warning, you are using a different partition key ({args.partition_key})'\
              f'than what whas used in the paper ({default_partition_key}).')

    partition_covertype(
        path_to_train_set=args.path_to_train_set,
        path_to_test_set=args.path_to_test_set,
        path_to_validation_set=args.path_to_validation_set,
        output_folder=args.output_folder,
        partition_key=args.partition_key,
        verbose=args.verbose
    )
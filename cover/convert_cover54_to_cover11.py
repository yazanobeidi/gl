import argparse
import os
import pandas as pd
import numpy as np

def convert_covertype(path_to_dataset, 
                      output_folder, 
                      verbose=False):
    """
    This function transforms the Covertype dataset from 54 features to 11 features
    by Ordinal encoding instead of one-hot-encoding. Numerical features are unchanged.
    We retain the identical train/test/validation proportions as in the original dataset 
    description (the following is taken from covtype.info):
    >    -- Classification performance
    >        -- first 11,340 records used for training data subset
    >        -- next 3,780 records used for validation data subset
    >        -- last 565,892 records used for testing data subset
    >        -- 70% Neural Network (backpropagation)
    >        -- 58% Linear Discriminant Analysis
    Original dataset source: https://archive.ics.uci.edu/ml/datasets/covertype
    Please see the covtype.info dataset description for more information.
    """
    df = pd.read_csv(path_to_dataset)
    initial_dtypes = df.dtypes
    # these are numerical - we don't modify them besides ensuring a float32 dtype
    float_columns = ['elevation', 'aspect', 'slope', 'hhdist', 'vhdist', 'hrdist', 'hfpdist']
    # these are onehot
    wilderness_columns = ['rawah', 'neota', 'comanche', 'poudre']
    # first digit=climate zone, second digit=geological zone, 3rd&4th digits have no inherent meaning
    soil_columns = ['2702', '2703', '2704', '2705', '2706', '2717', '3501',
                   '3502', '4201', '4703', '4704', '4744', '4758', '5101', '5151', '6101',
                   '6102', '6731', '7101', '7102', '7103', '7201', '7202', '7700', '7701',
                   '7702', '7709', '7710', '7745', '7746', '7755', '7756', '7757', '7790',
                   '8703', '8707', '8708', '8771', '8772', '8776']

    df[float_columns] = df[float_columns].astype(np.float32)
    ordinal_columns = wilderness_columns + soil_columns
    df[ordinal_columns] = df[ordinal_columns].astype(bool)

    ordinal_wilderness_column = df[wilderness_columns].idxmax(axis=1)
    ordinal_wilderness_column.name = 'wilderness_area'

    ordinal_soil_column = df[soil_columns].idxmax(axis=1)
    ordinal_soil_column.name = 'soil_type'

    climate_zone_column = ordinal_soil_column.apply(lambda x: x[0])
    climate_zone_column.name = 'climate_zone'

    geologic_zone_column = ordinal_soil_column.apply(lambda x: x[1])
    geologic_zone_column.name = 'geologic_zone'

    df = pd.concat([df[float_columns], 
                       ordinal_wilderness_column,
                       ordinal_soil_column,
                       climate_zone_column,
                       geologic_zone_column,
                       df.target.astype(np.float32)], axis=1)

    training_set = df.iloc[:11340]
    validation_set = df.iloc[11340:11340+3780]
    test_set = df.iloc[11340+3780:]

    training_set.to_csv(os.path.join(output_folder, 'covertype11_train.csv'), index=False)
    validation_set.to_csv(os.path.join(output_folder,'covertype11_validation.csv'), index=False)
    test_set.to_csv(os.path.join(output_folder,'covertype11_test.csv'), index=False)
    df.to_csv(os.path.join(output_folder,'covertype11_full.csv'), index=False)

    if verbose:
        print(f'Converted {path_to_dataset} from\n{initial_dtypes}\nto\n{df.dtypes}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-dataset", type=str, default="cover/data/covtype.data")
    parser.add_argument("--output-folder", type=str, default="cover/data/")
    parser.add_argument("--verbose", type=bool, default=False)

    args = parser.parse_args()

    convert_covertype(
        path_to_dataset=args.path_to_dataset,
        output_folder=args.output_folder,
        verbose=args.verbose
    )
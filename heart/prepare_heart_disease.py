import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as ttsplit

def load_uci_heart_disease(path, make_binary):
    dtypes = {'age':int, 
              'sex': int, 
              'cp': object, 
              'trestbps': int, 
              'chol': int, 
              'fbs': int, 
              'restecg': np.float32, 
              'thalach': np.float32, 
              'exang': int, 
              'oldpeak':np.float32, 
              'slope': int, 
              'ca': object, 
              'thal': np.float32, 
              'target': int}
    df = pd.read_csv(path, names=list(dtypes.keys()))
    df = df.replace('?', '-1').astype(dtypes)
    if make_binary:
        print(f'Converting multi-class to binary classification for {path}')
        df.target = (df.target == 0).astype(int)
    return df, make_binary

def load_sa(path, make_binary):
    dtypes = {'sbp': int,
              'tobacco': np.float32,
              'ldl': np.float32,
              'adiposity': np.float32,
              'famhist': object,
              'typea': int,
              'obesity': np.float32,
              'alcohol': np.float32,
              'age': int,
              'target': int}
    df = pd.read_csv(path).rename(columns={'chd':'target'})
    return df.drop('row.names', axis=1).astype(dtypes), False

def load_framingham(path, make_binary):
    dtypes = {'male': int,
              'age': int,
              'education': np.float32,
              'currentSmoker': int,
              'cigsPerDay': np.float32,
              'BPMeds': np.float32,
              'prevalentStroke': int,
              'prevalentHyp': int,
              'diabetes': int,
              'totChol': np.float32,
              'sysBP': np.float32,
              'diaBP': np.float32,
              'BMI': np.float32,
              'heartRate': np.float32,
              'glucose': np.float32,
              'target': int}
    df = pd.read_csv(path).rename(columns={'TenYearCHD':'target'})
    return df.replace(np.nan, '-1').astype(dtypes), False

def load_faisalabad(path, make_binary):
    dtypes = {'age': np.float32,
              'anaemia': int,
              'creatinine_phosphokinase': int,
              'diabetes': int,
              'ejection_fraction': int,
              'high_blood_pressure': int,
              'platelets': np.float32,
              'serum_creatinine': np.float32,
              'serum_sodium': int,
              'sex': int,
              'smoking': int,
              'time': int,
              'target': int}
    df = pd.read_csv(path).rename(columns={'DEATH_EVENT':'target'})
    return df.astype(dtypes), False

def prepare_heart_disease(path_to_dataset, 
                          output_folder, 
                          test_size,
                          val_size,
                          make_binary,
                          seed,
                          verbose=False):
    """
    This function transforms the various heart-disease datasets:
    1) UCI heart-disease: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
        -> this contains Cleveland, Hungary, Switzerland and Va locations.
    2) South Africa heart-diesase (SAHeart): http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html
    3) Framingham heart-disease: https://courses.washington.edu/b513/datasets/datasets.php?class=513
    4) UCI Faisalabad heart-disease (heart-failure): https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
    """
    datasets = [('cleveland.data', load_uci_heart_disease),
                ('hungarian.data', load_uci_heart_disease),
                ('switzerland.data', load_uci_heart_disease), 
                ('va.data', load_uci_heart_disease),
                ('sa.data', load_sa),
                ('framingham.csv', load_framingham),
                ('faisalabad.csv', load_faisalabad)]
    
    for dataset, loader in datasets:

        name = dataset.split('.')[0]
        path = path_to_dataset+dataset

        df, binarized = loader(path, make_binary)
        
        training_set, test_set = ttsplit(df, test_size=test_size, 
                             random_state=seed, shuffle=True, stratify=None)

        training_set, validation_set = ttsplit(training_set, test_size=val_size, 
                             random_state=seed, shuffle=True, stratify=None)

        training_set.to_csv(os.path.join(output_folder, f'heart_disease_train_{name}.csv'), index=False)
        validation_set.to_csv(os.path.join(output_folder, f'heart_disease_validation_{name}.csv'), index=False)
        test_set.to_csv(os.path.join(output_folder, f'heart_disease_test_{name}.csv'), index=False)
        df.to_csv(os.path.join(output_folder, f'heart_disease_{name}_full.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-dataset", type=str, default="heart/data/")
    parser.add_argument("--output-folder", type=str, default="heart/data/")
    parser.add_argument("--test-size", type=float, default=0.33)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--make-binary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=bool, default=False)

    args = parser.parse_args()

    prepare_heart_disease(
        path_to_dataset=args.path_to_dataset,
        output_folder=args.output_folder,
        test_size=args.test_size,
        val_size=args.val_size,
        make_binary=args.make_binary,
        seed=args.seed,
        verbose=args.verbose
    )
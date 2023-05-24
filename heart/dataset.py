import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
import os

from util import set_random_seed, unfitted
set_random_seed(42)

PATH_TO_DATASET_DIR = 'heart/data/{seed}/'
# HEART_DISEASE_SPLITS = ['cleveland', 'hungarian', 'switzerland', 'va', 'sa', 'framingham', 'faisalabad']
HEART_DISEASE_SPLITS = ['cleveland', 'sa', 'faisalabad']
HEART_DISEASE_TRAIN_PREFIX = PATH_TO_DATASET_DIR+'heart_disease_train_'
HEART_DISEASE_TEST_PREFIX = PATH_TO_DATASET_DIR+'heart_disease_test_'
HEART_DISEASE_VAL_PREFIX = PATH_TO_DATASET_DIR+'heart_disease_validation_'

class HeartDataset(Dataset):
    def __init__(self, 
                 path_to_csv, 
                 seed,
                 device,
                 encoder=None, 
                 label_encoder=None):
        set_random_seed(seed)
        self.seed = seed
        self.path_to_csv = path_to_csv
        self.encoder = self.get_feature_encoder() if encoder is None else encoder
        self.label_encoder = self.get_label_encoder() if label_encoder is None else label_encoder
        df = pd.read_csv(self.path_to_csv)
        # encode labels
        self.y = df.pop('target')
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(np.unique(self.y))
        self.y = self.label_encoder.transform(self.y)
        self.y = torch.tensor(self.y, dtype=torch.long, device=device)
        # encode features
        self.columns = df.columns
        if unfitted(self.encoder):
            self.encoder.fit(df)
        self.X = self.encoder.transform(df)#.astype(np.float32)
        del df
        self.X = torch.tensor(self.X, dtype=torch.float32, device=device)
        self.n_samples = len(self.X)
        self.name = self.path_to_csv.split('_')[-1][:-4]

    def get_input_shape(self):
        return tuple(self.X.shape[-1:])

    def get_unique_classes(self):
        return self.y.unique()

    def get_num_classes(self):
        return len(self.get_unique_classes())

    def get_output_shape(self):
        return tuple(self.get_output_classes().shape)

    def get_output_classes(self):
        return self.label_encoder.classes_

    def get_label_encoder(self):
        return LabelEncoder()

    def show_label_encoder_info(self):
        enc = self.y.unique().int().tolist()
        raw = self.label_encoder.inverse_transform(enc).astype(int).tolist()
        print(f'{self.name} label encodings {raw} => {enc}')
        
    def get_feature_encoder(self):
        return make_column_transformer(
                (StandardScaler(), make_column_selector(dtype_include=np.number)),
                (OrdinalEncoder(handle_unknown='use_encoded_value', 
                                unknown_value=-1, encoded_missing_value=-1), 
                                   make_column_selector(dtype_include=object)),
                verbose_feature_names_out=False).set_output(transform='default')
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y

def load_heart_disease(batches, drop_last, seed, device='cpu', 
                       force_consistent_feature_space=False,
                       force_consistent_target_space=False, verbose=True):
    
    cv_dataset_path = f"heart/data/{seed}/"
    if not os.path.isdir(cv_dataset_path):
        print(f'creating heart disease cross validation set for seed {seed} at {cv_dataset_path}')
        from heart.prepare_heart_disease import prepare_heart_disease
        os.mkdir(cv_dataset_path)
        prepare_heart_disease(
            path_to_dataset="heart/data/",
            output_folder=cv_dataset_path,
            test_size=0.33,
            val_size=0.1,
            make_binary=True,
            seed=seed,
            verbose=False
        )
    if force_consistent_target_space:
        print('Found option --force-consistent-target-space. '
              'Preloading training sets and extracting global target space ...')
        def get_global_label_encoder(verbose):
            train_data = {split:HeartDataset(HEART_DISEASE_TRAIN_PREFIX.format(seed=seed)+f'{split}.csv', 
                                         device=device, seed=seed) for split in HEART_DISEASE_SPLITS}
            psi = [d.label_encoder.inverse_transform(d.y.unique().tolist()) for d in train_data.values()]
            psi = list(set([float(e) for sublist in psi for e in sublist]))
            global_label_encoder = LabelEncoder()
            global_label_encoder.fit(psi)
            if verbose:
                print('Label encodings before forcing consistent target space (these will NOT be used)')
                for d in train_data.values():
                    d.show_label_encoder_info()
            del train_data
            return global_label_encoder

        global_label_encoder = get_global_label_encoder(verbose)
        print(f'... Now loading with obtained global target space: {global_label_encoder.classes_}:')
    else:
        global_label_encoder = None

    train_data, test_data, val_data = dict(), dict(), dict()

    for split in HEART_DISEASE_SPLITS:
        train_data[split] = HeartDataset(HEART_DISEASE_TRAIN_PREFIX.format(seed=seed)+f'{split}.csv', 
                                            device=device, seed=seed,
                                            label_encoder=global_label_encoder)
        test_data[split] = HeartDataset(HEART_DISEASE_TEST_PREFIX.format(seed=seed)+f'{split}.csv', 
                                            encoder=train_data[split].encoder,
                                            label_encoder=train_data[split].label_encoder, 
                                            device=device, seed=seed)
        val_data[split] = HeartDataset(HEART_DISEASE_VAL_PREFIX.format(seed=seed)+f'{split}.csv', 
                                            encoder=train_data[split].encoder,
                                            label_encoder=train_data[split].label_encoder, 
                                            device=device, seed=seed)

    if force_consistent_feature_space:
        print('Found option --force-consistent-target-space. '
              'Aligning feature spaces and imputing with 0s')
        def brute_force_align(datasets, train_data=None):
            # unfortunately sklearn does not support column_transformer.inverse_transform yet
            # dfs = {n:pd.DataFrame(d.encoder.inverse_transform(d.X.numpy()), 
            #                       columns=d.columns) for n, d in datasets.items()}
            dfs = {n:pd.read_csv(d.path_to_csv).drop('target', axis=1) for n, d in datasets.items()}
            n = len(dfs)
            k = list(dfs.keys())
            for i in range(n-1):
                for j in range(i+1, n):
                    dfs[k[i]], dfs[k[j]] = dfs[k[i]].align(dfs[k[j]], join="outer", axis=1, fill_value=0)
            for split, df in dfs.items():
                if train_data is None:
                    datasets[split].encoder.fit(df)
                else: # re-use train encoder on other cv splits
                    datasets[split].encoder = train_data[split].encoder
                datasets[split].X = torch.tensor(datasets[split].encoder.transform(df), 
                                                 dtype=torch.float32, device=device)
                datasets[split].columns, datasets[split].old_columns = df.columns, datasets[split].columns

        brute_force_align(train_data)
        brute_force_align(test_data, train_data=train_data)
        brute_force_align(val_data, train_data=train_data)

        if verbose:
            for split, dataset in train_data.items():
                print(f'aligned {split}: expanded feature space dim from'
                      f' {len(dataset.old_columns)} -> {len(dataset.columns)}')

    train_sizes = {n:len(s) for n,s in train_data.items()}

    def get_batch_sizes(train_sizes, batch_steps):
        return dict(zip(train_sizes.keys(), 
                        (np.array(list(train_sizes.values())) // batch_steps).tolist()))

    batch_sizes = get_batch_sizes(train_sizes, batches)

    if verbose:
        print('Batch sizes are set to: ', batch_sizes)

    trains, tests, vals = dict(), dict(), dict()

    for split in HEART_DISEASE_SPLITS:
        # if the last dataloader batch is size 1, this will raise an issue with batchnorm
        # which by default uses the batch statistics.. calling eval() makes it use running statistcs
        # another option is to set drop_last=True in the DataLoaders
        # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
        trains[split] = DataLoader(train_data[split], batch_size=batch_sizes[split], shuffle=True, drop_last=drop_last)
        tests[split] = DataLoader(test_data[split], batch_size=batch_sizes[split], shuffle=True)
        vals[split] = DataLoader(val_data[split], batch_size=batch_sizes[split], shuffle=True)
    
    if verbose:
        print('Label encodings are (we are using these):')
        for loader in trains.values():
            loader.dataset.show_label_encoder_info()

    return trains, tests, vals
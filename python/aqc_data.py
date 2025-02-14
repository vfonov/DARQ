# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 13/04/2018

import os
import collections

#from skimage import io, transform

import torch
import torchvision
import torchvision.transforms.functional as F

import numpy as np
import math

from torch.utils.data import Dataset, DataLoader


import gzip
import lz4.frame


QC_entry = collections.namedtuple( 
    'QC_entry', ['id', 'status', 'vol', 'variant', 'cohort', 'subject', 'visit', 'dist' ] )


def load_full_db(qc_db_path, 
                data_prefix, 
                table="qc_npy"
                ):
    """Load complete QC database into memory
    """
    import sqlite3

    with sqlite3.connect( qc_db_path ) as qc_db:

        query = f"""select variant,cohort,subject,visit,vol,pass,N from {table} """

        samples = []

        for line in qc_db.execute(query):
            variant, cohort, subject, visit, vol, _pass, N = line

            vol=os.path.join(data_prefix,vol)

            if _pass == 'TRUE': 
                status=1 
            else: 
                status=0 

            _id='{}:{}:{}:{}:{}'.format(variant, cohort, subject, visit,N)
            dist=-1
            if os.path.exists(vol):
                samples.append( QC_entry( _id, status, vol, variant, cohort, subject, visit, float(dist) ))
            else:
                print('Warning: file {} does not exist'.format(vol))

        return samples



def init_cv(dataset, fold=0, folds=8, validation=5, shuffle=False, seed=None):
    """
    Initialize Cross-Validation

    returns three indexes
    """
    n_samples   = len(dataset)
    whole_range = np.arange(n_samples)

    if shuffle:
        _state = None
        if seed is not None:
            _state = np.random.get_state()
            np.random.seed(seed)

        np.random.shuffle(whole_range)

        if seed is not None:
            np.random.set_state(_state)

    if folds > 0:
        training_samples = np.concatenate((whole_range[0:math.floor(fold * n_samples / folds)],
                                           whole_range[math.floor((fold + 1) * n_samples / folds):n_samples]))

        testing_samples = whole_range[math.floor(fold * n_samples / folds): 
                                      math.floor((fold + 1) * n_samples / folds)]
    else:
        training_samples = whole_range
        testing_samples = whole_range[0:0]
    #
    validation_samples = training_samples[0:validation]
    training_samples   = training_samples[validation:]

    return [ dataset[i] for i in training_samples   ], \
           [ dataset[i] for i in validation_samples ], \
           [ dataset[i] for i in testing_samples    ]

def split_dataset(all_samples, fold=0, folds=8, validation=5, 
    shuffle=False, seed=None):
    """
    Split samples, according to the subject field
    into testing,training and validation subsets
    sec_samples will be used for training subset, if provided
    """
    ### extract subject list
    subjects = set()
    for i in all_samples:
        subjects.add(i.subject)
    
    # split into three
    training_subjects, validation_subjects, testing_subjects = init_cv(
            sorted(list(subjects)), fold=fold, folds=folds, 
            validation=validation, shuffle=shuffle,seed=seed
        )
    
    training_subjects=set(training_subjects)
    validation_subjects=set(validation_subjects)
    testing_subjects=set(testing_subjects)

    # apply index
    validation = [i for i in all_samples if i.subject in validation_subjects]
    testing    = [i for i in all_samples if i.subject in testing_subjects]
    training   = [i for i in all_samples if i.subject in training_subjects]
    
    return training, validation, testing


def load_volume(vol):
    if vol.endswith('.gz'):
        with gzip.open(vol) as f:
            return np.load(f, allow_pickle=False )
    elif vol.endswith('.lz4'):
        with lz4.frame.open(vol) as f:
            return np.load(f, allow_pickle=False )
    else:
        return np.load(vol,allow_pickle=False )


class QCDataset(Dataset):
    """
    QC images dataset
    """

    def __init__(self, dataset, data_prefix):
        """
        Args:
            root_dir (string): Directory with all the data
            use_ref  (Boolean): use reference images
        """
        super(QCDataset, self).__init__()

        self.qc_samples = dataset
        self.data_prefix = data_prefix
        self.qc_subjects = set(i.subject for i in self.qc_samples)
        ### check sample size
        if len(self.qc_samples)>0:
            _vol = load_volume( self.qc_samples[0].vol )
            self.sample_size = _vol.shape
        else:
            self.sample_size = None


    def __len__(self):
        return len( self.qc_samples )

    def __getitem__(self, idx):
        _s = self.qc_samples[ idx ]
        # load images 

        # TODO: finish this 
        _vol = np.nan_to_num(load_volume( _s.vol )[None,:,:,:],nan=0.0,posinf=1.0,neginf=0.0,copy=False)

        return {'volume':_vol, 'status': _s.status, 'id':_s.id, 'dist': _s.dist }

    def n_subjects(self):
        """
        Return number of unique subjects
        """
        return len(self.qc_subjects)

    def get_balance(self):
        """
        Calculate class balance True/(True+False)
        """
        cnt=np.zeros(2)
        for i in self.qc_samples:
            cnt[i.status] = cnt[i.status]+1
        return cnt[1]/(cnt[1]+cnt[0]) if (cnt[1]+cnt[0])>0 else 0.0

    def balance(self):
        """
        Balance dataset by excluding some samples
        """
        # TODO: shuffle?
        pos_samples=[i for i in self.qc_samples if i.status==1]
        neg_samples=[i for i in self.qc_samples if i.status==0]
        
        n_both=min(len(pos_samples),len(neg_samples))

        self.qc_samples = pos_samples[0:n_both] + neg_samples[0:n_both]
        self.qc_subjects = set(i.subject for i in self.qc_samples)






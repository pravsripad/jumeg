#!/usr/bin/env python

"""
=================================================
Compute Power Spectral density based fingerprints
=================================================

"""

import os.path as op
import numpy as np
from scipy import stats
import mne

import matplotlib.pyplot as plt
import seaborn as sns

from jumeg.jumeg_utils import rescale_arr
from jumeg import get_jumeg_path
from jumeg.connectivity import plot_grouped_connectivity_circle

from mne_features.feature_extraction import extract_features

from rasmalai.meg_utils import (get_normalised_psds_welch, get_psd_fingerprint)

from rasmalai.network_utils import (make_count_based_stats,
                                    make_combined_average,
                                    get_con_metrics, get_coords_given_labels)

from glob import glob
import yaml
import pickle
import time
t_start = time.time()

sns.set_style('white')

epoch_fname = '/Users/psripad/kelsa/epochs_with_artifacts/207184_rest_EC_0.6-200bp_bcc,nr,ar,1-epo.fif'
epochs = mne.read_epochs(epoch_fname, preload=True)

# data = epochs.get_data()
# selected_funcs = {'pow_freq_bands'}
# freqs = np.array([1., 4., 8., 13., 18., 30., 80., 200.])
# X_new = extract_features(data, epochs.info['sfreq'], selected_funcs,
#                          funcs_params={'pow_freq_bands__freq_bands': freqs,
#                                        'pow_freq_bands__normalize': False,
#                                        'pow_freq_bands__ratios': None})
# X_new = X_new.reshape((430, -1, len(freqs) - 1))
# X_avg = X_new.mean(axis=(0, 1))

freq_bands = np.array([1., 4., 8., 13., 18., 30., 80., 200.])

# get the psds normalised using the gamma band
norm_psds, _ = get_normalised_psds_welch(epochs, norm_freq=[80., 200.],
                                         freq_bands=None,
                                         grand_average=True)
print(norm_psds.shape)

# get the PSD fingerprint (mean sum of PSDs across frequency bands)
psd_fingerprint = get_psd_fingerprint(epochs, freq_bands=freq_bands,
                                      normalize=True)
print(psd_fingerprint.shape)

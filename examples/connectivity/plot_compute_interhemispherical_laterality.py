"""
================================================================
Compute interhemispherical laterality between brain hemispheres.
================================================================
"""

# Author: Praveen Sripad <pravsripad@gmail.com>

import os.path as op
import numpy as np
import mne

from mne.datasets import sample
from jumeg.jumeg_utils import get_jumeg_path
from jumeg.connectivity import make_annot_from_csv
from jumeg.connectivity import plot_grouped_connectivity_circle

data_path = sample.data_path()
subject = 'sample'
subjects_dir = data_path + '/subjects'
parc_fname = 'standard_garces_2016'
csv_fname = op.join(get_jumeg_path(), 'data', 'standard_rsns.csv')

# set make_annot to True to save the annotation to disk
labels, coords, foci = make_annot_from_csv(subject, subjects_dir, csv_fname,
                                           parc_fname=parc_fname,
                                           make_annot=False,
                                           return_label_coords=True)

aparc = mne.read_labels_from_annot('sample', subjects_dir=subjects_dir)
aparc_names = [apa.name for apa in aparc]
lh_aparc = [mylab for mylab in aparc if mylab.hemi == 'lh']

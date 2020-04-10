"""
=============================================================================
Read grouped aparc labels from yaml file and plot grouped connectivity circle
=============================================================================
"""

# Author: Praveen Sripad <pravsripad@gmail.com>

import numpy as np
from jumeg import get_jumeg_path
from jumeg.connectivity import plot_grouped_connectivity_circle
import yaml

labels_fname = get_jumeg_path() + '/data/desikan_label_names.yaml'
yaml_fname = get_jumeg_path() + '/data/desikan_aparc_cortex_based_grouping.yaml'
replacer_dict_fname = get_jumeg_path() + '/data/replacer_dictionaries.yaml'

with open(labels_fname, 'r') as f:
    label_names = yaml.safe_load(f)['label_names']

with open(replacer_dict_fname, 'r') as f:
    replacer_dict = yaml.safe_load(f)['replacer_dict_aparc']

# make a random matrix with 68 nodes
# use simple seed for reproducibility
np.random.seed(42)
con = np.random.random((68, 68))
con[con < 0.5] = 0.

indices = (np.array((1, 2, 3)), np.array((5, 6, 7)))
plot_grouped_connectivity_circle(yaml_fname, con, label_names,
                                 labels_mode='cortex_only',
                                 replacer_dict=replacer_dict,
                                 out_fname='example_grouped_con_circle.png',
                                 colorbar_pos=(0.1, 0.1),
                                 n_lines=10, colorbar=True,
                                 colormap='viridis')

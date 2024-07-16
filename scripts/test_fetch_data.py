import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from joblib import Memory

import os

from nilearn.datasets import fetch_neurovault_ids

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

# Fetch data
fetch_neurovault_ids(collection_ids=[1952], mode="download_new")

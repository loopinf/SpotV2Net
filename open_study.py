# -*- coding: utf-8 -*-
"""
Created on Tue May 23 20:16:44 2023

@author: ab978
"""

import optuna
import pickle


# Specify the path to your pickle file
pickle_file_path = "output/20220522_RGNN_PriceCov_12_optuna/study.pkl"

# Load the study object from the pickle file
with open(pickle_file_path, "rb") as f:
    study = pickle.load(f)

# Access the study object and perform operations
study_name = study.study_name
best_trial = study.best_trial
trials = study.trials



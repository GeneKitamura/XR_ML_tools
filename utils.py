import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def array_print(*args):
    for i in args:
        print(i.shape)

def array_min_max(*args):
    for i in args:
        print(np.min(i), np.max(i))
        
def sex_age_sample(positive_df, non_pos_df, sample_n, params=None):
    if params is None:
        params = {
        "pos_age": 'Patient Age',
        "pos_sex": 'Patient Sex',
        "non_pos_age": 'Patient Age',
        "non_pos_sex": 'Patient Sex',
        'non_pos_fname': 'first_name',
        'non_pos_lname': 'last_name'
        }

    prior_age = positive_df[params['pos_age']]
    age_hist = plt.hist(prior_age, bins=5)
    age_count, age_bounds, _ = age_hist
    age_proportions = age_count / sum(age_count)
    expected_samples = np.floor(sample_n * age_proportions).astype(np.int32)

    male_n = positive_df[params['pos_sex']].value_counts()['Male']
    fem_n = positive_df[params['pos_sex']].value_counts()['Female']
    male_prop = male_n / (male_n + fem_n)
    fem_prop = fem_n / (male_n + fem_n)

    sampled_montage_df = pd.DataFrame()
    for n in range(5):
        c_df = non_pos_df[(non_pos_df[params['non_pos_age']] > age_bounds[n]) & (non_pos_df[params['non_pos_age']] <= age_bounds[n + 1])]
        expected_males = c_df[c_df[params['non_pos_sex']] == 'Male'].sample(int(male_prop * expected_samples[n]), replace=True)
        expected_fems = c_df[c_df[params['non_pos_sex']] == 'Female'].sample(int(fem_prop * expected_samples[n]), replace=True)
        sampled_montage_df = pd.concat([sampled_montage_df, expected_males, expected_fems])

    sampled_montage_df = sampled_montage_df.drop_duplicates(subset=[params['non_pos_fname'], params['non_pos_lname'], 'Exam Completed Date'])
    plt.hist(sampled_montage_df[params['non_pos_age']], bins=age_bounds)
    return sampled_montage_df
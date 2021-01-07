import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats

def array_print(*args):
    for i in args:
        print(i.shape)

def array_min_max(*args):
    for i in args:
        print(np.min(i), np.max(i))

# dt as accessor for datetime properties of series, HOWEVER NaT becomes 'NaT' string with dt accessor
# for Timestamp (single item), can use strftime directy.  NaT thorws error
def mix_date_parse(c_item): # use for map or applymap for pd.dt/series
    try:
        dt_item = pd.to_datetime(c_item)
        dt_item = dt_item.strftime("%m-%d-%Y") #NAT
        return dt_item
    except:
        return c_item

def sex_age_sample(positive_df, non_pos_df, sample_n, params=None):
    if params is None:
        params = {
        "pos_age": 'Patient Age',
        "pos_sex": 'Patient Sex',
        "non_pos_age": 'Patient Age',
        "non_pos_sex": 'Patient Sex',
        'non_pos_fname': 'Patient First Name',
        'non_pos_lname': 'Patient Last Name'
        }
    filtered_pos_df = positive_df[pd.to_numeric(positive_df[params['pos_age']], errors='coerce').notna()].copy()
    filtered_pos_df[params['pos_sex']] = filtered_pos_df[params['pos_sex']].fillna('')
    # if not filtered_pos_df['Patient Sex'].isin(['Male', 'Female', '']).all():
    #     raise Exception("Check Patient Sex for postive_df")
    filtered_pos_df = filtered_pos_df[(filtered_pos_df[params['pos_sex']] != '')]

    prior_age = filtered_pos_df[params['pos_age']]
    age_hist = plt.hist(prior_age, bins=5)
    age_count, age_bounds, _ = age_hist
    age_proportions = age_count / sum(age_count)
    expected_samples = np.floor(sample_n * age_proportions).astype(np.int32)

    male_n = filtered_pos_df[params['pos_sex']].value_counts()['Male']
    fem_n = filtered_pos_df[params['pos_sex']].value_counts()['Female']
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

def sex_age_comparison(a_df, b_df):
    a = a_df['Patient Age']
    a = a[a.notna()]
    b = b_df['Patient Age']
    b = b[b.notna()]
    print('Age stat: {}\n'.format(stats.ttest_ind(a, b)))

    c = a_df['Patient Sex'].value_counts()
    d = b_df['Patient Sex'].value_counts()
    print('Sex stat p_val: {}'.format(stats.chi2_contingency([[c['Male'], c['Female']], [d['Male'], d['Female']]])[1]))

def extract_idx_from_df(excel_n):
    cdf = pd.read_excel(excel_n)
    columns = cdf.columns
    e_dict = {}
    for item in columns:
        indices = cdf[item].dropna().map(int).values
        if len(indices) == 0:
            continue
        e_dict[item] = indices
    return e_dict

def categorize_angle(x):
    if x < 0:
        return 1
    elif x > 0:
        return 2
    else:
        return 0
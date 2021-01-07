import re
import pandas as pd
import numpy as np

from .nlp import void_term, validate_term

class WordParser():
    def __init__(self, c_df, terms_xlsx):
        self.c_df = c_df
        self.terms = terms_xlsx
        self.criteria = False
        self.text_df=None

    def prepare_criteria(self):
        terms = pd.read_excel(self.terms)
        self.inappropriate = terms['inappropriate'].dropna().tolist()
        self.incorrect = terms['incorrect'].dropna().tolist() #just skull
        self.post_op_terms = terms['postop'].dropna().tolist()
        self.criteria = True


    def clean_text(self, inclusion_list=None,
                   post_inclusion_validate_list=None, alias_name='merge_ID',  sentence_split=True,
                   get_findings=False, use_inclusion_regex=False):

        raw_excel = self.c_df.copy()
        raw_excel['merge_ID'] = raw_excel.index

        id_and_report_df = raw_excel[[alias_name, 'Report Text']].copy()  # avoid setting on slice
        # id_and_report_df = raw_excel
        # self.raw_text = id_and_report_df.copy()

        working_report_name = 'text_IP'
        id_and_report_df[working_report_name] = id_and_report_df['Report Text'].str.lower()
        addendum_or_mingle = re.compile(r'do not use|addendum')

        id_and_report_df['mingle_addendum'] = id_and_report_df[working_report_name].\
            map(lambda x: True if addendum_or_mingle.search(x) else False)

        right_bool = id_and_report_df[working_report_name].map(lambda x: True if re.search(r'right', x) else False)
        left_bool = id_and_report_df[working_report_name].map(lambda x: True if re.search(r'left', x) else False)
        laterality_discreptancy = right_bool & left_bool  # logical_and
        id_and_report_df['laterality_discreptancy'] = laterality_discreptancy

        id_and_report_df['lower_report'] = id_and_report_df[working_report_name].copy()

        if get_findings:
            findings_rg = re.compile(r'findings(.*)', re.I | re.S)
            id_and_report_df[working_report_name] = id_and_report_df[working_report_name].map(
                lambda x: x if findings_rg.search(x) else 'findingsUnicorn')  # put in findings if there are None.
            id_and_report_df[working_report_name] = id_and_report_df[working_report_name].map(
                lambda x: findings_rg.search(x).group(1))  # Then just extract after the findings

        else:  # get impression instead
            impression_rg = re.compile(r'impression(.*)', re.I | re.S)
            id_and_report_df[working_report_name] = id_and_report_df[working_report_name].map(lambda x: x if impression_rg.search(
                x) else 'impressionUnicorn')  # put in findings if there are None.
            id_and_report_df[working_report_name] = id_and_report_df[working_report_name].map(
                lambda x: impression_rg.search(x).group(1))  # Then just extract after the findings

        id_and_report_df['lower_find_or_imp'] = id_and_report_df[working_report_name].copy()

        # self.item_text = id_and_report_df.copy()

        # if you also want to split by \n, maybe replace \n with '.' with regex before split
        # pandas series str.split ignores nan.
        split_imp_series = id_and_report_df[working_report_name].str.split('.', expand=False)

        if not sentence_split:
            split_imp_series = split_imp_series.map(lambda x: ['.'.join(x)])

        if inclusion_list is not None:  # probably the same as the regex way
            dirty_frac_or_disloc = split_imp_series.map(lambda x: validate_term(x, word_list=inclusion_list))
        elif use_inclusion_regex:  # the way I did it, regex way
            inclusion_criteria = re.compile('fracture|disloc|widen|diastasis', re.I)
            dirty_frac_or_disloc = split_imp_series.map(lambda x: [i for i in x if inclusion_criteria.search(i)])
        else:
            dirty_frac_or_disloc = split_imp_series

        clean_series = dirty_frac_or_disloc.map(lambda x: [re.sub(r'\n', '', i) for i in x])
        stripped_series = clean_series.map(lambda x: [i.strip() for i in x])

        if post_inclusion_validate_list is not None:
            stripped_series = stripped_series.map(lambda x: validate_term(x, post_inclusion_validate_list))

        text_df = pd.DataFrame(stripped_series)
        text_df[alias_name] = id_and_report_df[alias_name]

        # text_df['report_len'] = text_df[working_report_name].map(lambda x: len(x))
        # text_df['report_bool'] = (text_df['report_len'] != 0)
        text_df['mingle_addendum'] = id_and_report_df['mingle_addendum']
        # text_df['laterality_discreptancy'] = id_and_report_df['laterality_discreptancy']
        text_df['No relevant text'] = text_df[working_report_name].map(lambda x: validate_term(x, ['Unicorn'], return_bool=True))
        text_df['lower_report'] = id_and_report_df['lower_report']
        # text_df['lower_find_or_imp'] = id_and_report_df['lower_find_or_imp']
        # text_df['Report Text'] = id_and_report_df['Report Text']

        self.text_series = stripped_series
        self.text_df = text_df

    def filter_text(self, print_word=False, exclude_mingle=False):

        if self.text_df is None:
            self.clean_text()

        if not self.criteria:
            self.prepare_criteria()

        text_df = self.text_df.copy()

        void_terms = self.inappropriate + self.incorrect
        post_op_terms = self.post_op_terms

        # Remember that void and validate terms return_bool will be True/False for the WHOLE LIST if even a single word is found
        void_bool = text_df['text_IP'].map(lambda x: void_term(x, void_terms, return_bool=True, print_word=print_word))
        no_relevant_text_bool = ~text_df['No relevant text']
        mingle_bool = ~text_df['mingle_addendum']
        filter_bool = void_bool & no_relevant_text_bool #all needs to be True (True & False is False)
        if exclude_mingle:
            filter_bool = filter_bool & mingle_bool

        text_df['safe_bool'] = filter_bool

        post_op_bool =text_df['text_IP'].map(lambda x: validate_term(x, post_op_terms, return_bool=True, print_word=print_word))
        text_df['postop_bool'] = post_op_bool

        self.safe_text = text_df

def read_montage(montage_file, terms_file, exclude_mingle=False):

    montage_df = pd.read_excel(montage_file)
    montage_df = montage_df.drop_duplicates()
    short_montage = montage_df[['Organization', 'Accession Number', 'Report Text', 'Patient Sex',
                                  'Patient Age', 'Patient MRN', 'Patient First Name', 'Patient Last Name',
                                  'Exam Completed Date']].copy()

    processed_montage = WordParser(montage_df, terms_file)
    processed_montage.prepare_criteria()

    # use impression since many reports don't have findings.
    # use impression for hardware so it's not too specific
    processed_montage.clean_text(get_findings=False)
    processed_montage.filter_text(exclude_mingle=exclude_mingle)

    safe_text_df = processed_montage.safe_text
    short_montage['postop_bool'] = safe_text_df['postop_bool']
    short_montage['mingle_addendum'] = safe_text_df['mingle_addendum']
    short_montage['text_IP'] = safe_text_df['text_IP']
    short_montage = short_montage[safe_text_df['safe_bool']].copy() # exclude no text and inappropriate/incorrect terms

    print('montage_n {}'.format(short_montage.shape))

    return short_montage

def merge_slice_dfs(gt_df, montage_df, params=None, prior_date=-60, after_date=120, fill_na=True,
                    prior_is_nonop=False, additional_columns=None, nonop_zero_is_prior=True,
                    age_diff_cuffoff=1):
    if params is None:
        params = {'orig_surg_date': 'ORIG_SERVICE_DATE',
        'pat_id': 'PAT_ID',
        'non_montage_age': 'AGE',
        'non_montage_first_name': 'first_name',
        'non_montage_last_name': 'last_name',
        'orig_idx': 'orig_idx'}

    orig_surg_date = params['orig_surg_date']
    c_id = params['pat_id']
    age = params['non_montage_age']
    first_name = params['non_montage_first_name']
    last_name = params['non_montage_last_name']
    orig_idx = params['orig_idx']

    both_df = pd.merge(gt_df, montage_df, how='left',
                       left_on=[first_name, last_name],
                       right_on=['Patient First Name', 'Patient Last Name'])
    if fill_na:
        both_df['Patient MRN'] = both_df['Patient MRN'].fillna(7).map(np.int32)
    print('pat_ID nunique: {}'.format(both_df[c_id].nunique()))

    if fill_na:
        both_df['Exam Completed Date'] = both_df['Exam Completed Date'].fillna(pd.Timestamp('2005-01-01'))
        both_df[orig_surg_date] = both_df[orig_surg_date].fillna(pd.Timestamp('2000-01-01'))
    both_df['time_diff'] = both_df['Exam Completed Date'] - both_df[orig_surg_date]
    both_df['time_diff'] = both_df['time_diff'].map(lambda x: x.days)

    # Montage Patient Age is rounded down, possibly to int floor.  So need age diff ~1.5.
    both_df['age_diff'] = both_df[age] - both_df['Patient Age']
    both_df['age_diff'] = both_df['age_diff'].map(np.abs)

    prefilter_df = both_df.copy()
    pre_filter = (both_df['age_diff'] < age_diff_cuffoff) & (both_df['time_diff'] > prior_date) & (both_df['time_diff'] < after_date)
    # print('before pre_filter pat_ID ', both_df[c_id].unique().shape)
    both_df = both_df[pre_filter].copy()
    # print('after pre_filter pat_ID ', both_df[c_id].unique().shape)

    prior_filter = (both_df['time_diff'] < 0) & (both_df['time_diff'] > prior_date)
    if prior_is_nonop:
        prior_filter = prior_filter & (both_df['postop_bool'] == False)
    def_prior_patid = both_df[prior_filter][c_id].unique()

    if nonop_zero_is_prior:
        zero_prior_idx = both_df[(both_df['time_diff'] == 0) & (both_df['postop_bool'] == False)].index
        zero_is_prior_patid = both_df.loc[zero_prior_idx, c_id].unique()  # if time zero and not postop
        def_prior_patid = np.concatenate([def_prior_patid, zero_is_prior_patid])
        def_prior_patid = pd.Series(def_prior_patid).unique()  # unique pat_id

    # after filter not important, since we're separating into priors vs. non-priors (all others)
    after_filter = (both_df['time_diff'] > 0) & (both_df['time_diff'] < after_date)
    def_after_patid = both_df[after_filter][c_id].unique()

    if nonop_zero_is_prior:
        zero_postop_filter = (both_df['time_diff'] == 0) & (both_df['postop_bool'] == True)
    else:
        zero_postop_filter = (both_df['time_diff'] == 0) # all time zero is an "after" case

    zero_postop_patid = both_df[zero_postop_filter][c_id].unique()
    combined_after_and_postop_zero_patid = pd.Series(np.concatenate([def_after_patid, zero_postop_patid])).unique()

    both_df['prior'] = both_df[c_id].isin(def_prior_patid)
    both_df['after'] = both_df[c_id].isin(combined_after_and_postop_zero_patid)

    both_df['prior_after'] = (both_df['prior'] & both_df['after'])
    both_df['only_prior'] = (both_df['prior'] & ~both_df['after'])
    both_df['only_after'] = (~both_df['prior'] & both_df['after'])

    columns = [orig_idx, 'Organization', orig_surg_date, age, first_name, last_name,
                       'Accession Number', 'Report Text', 'Patient Sex', 'Patient Age',
                       'Patient MRN', c_id,
                       'Exam Completed Date', 'time_diff', 'prior', 'after',
                       'prior_after', 'only_prior', 'only_after', 'postop_bool', 'mingle_addendum', 'text_IP']

    if additional_columns is not None:
        columns = columns + additional_columns

    both_df = both_df[columns]

    #get rid of invalid accession and associated PAT_ID
    prior_df = both_df[both_df[c_id].isin(def_prior_patid)].copy()
    try:
        prior_df['Accession Number'] = pd.to_numeric(prior_df['Accession Number'], errors='coerce', downcast='unsigned')
    except ValueError: # if a df is empty, will result in error when trying to downcast
        pass
    invalid_accession_idx = prior_df[~pd.to_numeric(prior_df['Accession Number'], errors='coerce').notna()].index
    prior_df = prior_df[~prior_df.index.isin(invalid_accession_idx)].copy()
    # redefine prior_patid
    def_prior_patid = prior_df[c_id].unique()

    # keep all cases without priors, post-op cases may have use
    # no_prior_df = prefilter_df[~prefilter_df[c_id].isin(def_prior_patid)].drop_duplicates(subset=[c_id]).copy()
    no_prior_df = prefilter_df[~prefilter_df[c_id].isin(def_prior_patid)].copy()
    no_prior_patID = no_prior_df[c_id].unique()  # id is unique among subjects

    print('prior_unique_patid: {}, no_prior_patID {}\n'.format(len(def_prior_patid), len(no_prior_patID)))

    return prior_df, no_prior_df

def format_no_montage_dfs(first_no_prior, remain_no_prior, params, to_save=None, additional_columns=None, fill_cols=None):

    pid = params['id']
    dob = params['dob']
    surg_date = params['op_date']
    fname = params['first_name']
    lname = params['last_name']
    age = params['AGE']
    sex = params['SEX']
    clin_mrn = params['clin_mrn']
    mont_mrn = "Patient MRN"

    first_no_prior['first_op'] = 'yes'
    remain_no_prior['first_op'] = 'no'
    tot_no_priors_df = pd.concat([first_no_prior, remain_no_prior], axis=0)
    tot_no_priors_df = tot_no_priors_df.drop_duplicates(subset=[pid]).copy()

    # sort by PID and surg_date
    tot_no_priors_df = tot_no_priors_df.sort_values([pid, surg_date], ascending=[True, True])

    relevant_columns = [surg_date, fname, lname, dob, pid, age, sex, clin_mrn, mont_mrn, 'first_op']

    if additional_columns is not None:
        relevant_columns = relevant_columns + additional_columns

    tot_no_priors_df = tot_no_priors_df[relevant_columns].copy()

    if fill_cols is None:
        fill_cols = ['Pre_op_Organization', 'Pre_op_Accession_#', 'Pre_op_study_date', 'same_day_contra_Accession']

    tot_no_priors_df = pd.concat([tot_no_priors_df, pd.DataFrame(columns=fill_cols)], sort=False)

    if to_save is not None:
        tot_no_priors_df[surg_date] = tot_no_priors_df[surg_date].dt.date
        tot_no_priors_df[dob] = tot_no_priors_df[dob].dt.date

        # pd.to_datetime(tot_no_priors_df.loc["DOB"]) # to change back to datetime

        tot_no_priors_df.to_excel(to_save)

    return tot_no_priors_df
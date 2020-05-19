import re
import pandas as pd

def void_term(x, word_list, print_word=False, return_bool=False):
    c_list = []
    breaker = False
    term_present = False
    for sentence in x:
        for word in word_list:
            if word in sentence:
                if print_word:
                    print("word is: ", word, "in sentence: ", sentence)
                term_present = True
                breaker = True
                break
        if not breaker:
            c_list.append(sentence.strip())
        breaker = False
    if return_bool:
        return not term_present
    else:
        return c_list

def validate_term(x, word_list, print_word=False, return_bool=False):
    c_list = []
    term_present = False
    for sentence in x:
        for word in word_list:
            if word in sentence:
                c_list.append(sentence.strip())
                term_present = True
                if print_word:
                    print("word is: ", word, "in sentence: ", sentence)
                break
    if return_bool:
        return term_present
    else:
        return c_list

class WordParser():
    def __init__(self, c_df):
        self.c_df = c_df
        self.criteria = False
        self.text_df=None

    def prepare_criteria(self):
        terms = pd.read_excel('./terms.xlsx')
        self.inappropriate = terms['inappropriate'].dropna().tolist()
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
        text_df['No relevant text'] = text_df[working_report_name].\
            map(lambda x: validate_term(x, ['Unicorn'], return_bool=True))
        text_df['lower_report'] = id_and_report_df['lower_report']
        # text_df['lower_find_or_imp'] = id_and_report_df['lower_find_or_imp']
        # text_df['Report Text'] = id_and_report_df['Report Text']

        self.text_series = stripped_series
        self.text_df = text_df

    def filter_text(self, print_word=False):

        if self.text_df is None:
            self.clean_text()

        if not self.criteria:
            self.prepare_criteria()

        text_df = self.text_df.copy()

        inappropriate_terms = self.inappropriate
        post_op_terms = self.post_op_terms

        # Remember that void and validate terms return_bool will be True/False for the WHOLE LIST if even a single word is found
        inappropriate_bool = text_df['text_IP'].map(lambda x: void_term(x, inappropriate_terms, return_bool=True, print_word=print_word))
        # no_relevant_text_bool = text_df['text_IP'].map(lambda x: void_term(x, ['Unicorn'], return_bool=True, print_word=print_word))
        no_relevant_text_bool = ~text_df['No relevant text']
        mingle_bool = ~text_df['mingle_addendum']
        filter_bool = inappropriate_bool & no_relevant_text_bool #all needs to be True
        text_df['safe_bool'] = filter_bool

        post_op_bool =text_df['text_IP'].map(lambda x: validate_term(x, post_op_terms, return_bool=True, print_word=print_word))
        text_df['postop_bool'] = post_op_bool

        self.safe_text = text_df

def read_montage(montage_file):

    montage_scfe = pd.read_excel(montage_file)
    short_montage = montage_scfe[['Organization', 'Accession Number', 'Report Text', 'Patient Sex', 'Patient Age', 'Patient MRN', 'Patient First Name', 'Patient Last Name', 'Exam Completed Date']]
    short_montage = short_montage.drop_duplicates(subset=['Accession Number'])

    montage = WordParser(montage_scfe)
    montage.prepare_criteria()

    # use impression since many reports don't have findings.
    # use impression for hardware so it's not too specific
    montage.clean_text(get_findings=False)
    montage.filter_text()

    safe_text_df = montage.safe_text
    short_montage['postop_bool'] = safe_text_df['postop_bool']
    short_montage = short_montage[safe_text_df['safe_bool']].copy()

    print('montage_n {}'.format(short_montage.shape))

    return short_montage
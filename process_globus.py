import pydicom
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
import shutil

from glob import glob

MAIN_PATH = '../L_spine_images/Images/'
F_IMG_MAP = './../PHI/DEXA/R3_1750_PT_MAP_FINAL.xlsx'
R3_REQ = './../PHI/DEXA/R1750.xlsx'
# PELVIC_FX_DF = pd.read_excel('./old_xlsx/pelvic_final_anon.xlsx') #actual useful 6263 cases

R60_DF = None
PELVIC_FX_DF = None
non_pelvis_list = None

# MAIN_PATH -> patient -> study (acc) -> series (one to many) -> DICOM (one to many)
# Different patients with different PATIENT_STUDY_ID may have studies with same ACCESSION_STUDY_ID.
# Look for Duplicated PAT_ID
# A patient may have 2 subdirs with same ACCESSION_STUDY_ID but different study titles
def count_dirs(c_path, del_error_path=False):
    dir_count = 0
    multi_pt_list = []
    all_pt_list = []
    single_pt_list = []
    study_acc_dir_list = []
    unique_acc_dir_list = []
    tot_patid_acc_tup_list = []
    unique_patid_acc_tup_list = []
    duplicated_patid_acc_tup_list = []
    dups_acc = []
    whole_path = []

    for pt in os.listdir(c_path):
        pt_id = pt.split('_')[1]
        try:
            subdir_path = next(os.walk(os.path.join(c_path, pt)))[1]
        except:
            error_path = os.path.join(c_path, pt)
            print(error_path)  # not directory such as .DS_Store
            if del_error_path:
                shutil.rmtree(error_path)
            continue

        for pt_acc in subdir_path:
            acc_num = re.search(r'_(\d{5,})', pt_acc).group(1)
            study_acc_dir_list.append(acc_num)
            if acc_num not in unique_acc_dir_list:
                unique_acc_dir_list.append(acc_num)
            else:
                dups_acc.append(acc_num)

            pt_id = int(pt_id)
            acc_num = int(acc_num)
            id_acc_tup = (pt_id, acc_num)
            whole_path.append([pt, pt_id, pt_acc, acc_num, id_acc_tup])
            tot_patid_acc_tup_list.append(id_acc_tup)
            if id_acc_tup not in unique_patid_acc_tup_list:
                unique_patid_acc_tup_list.append(id_acc_tup)
            else:
                duplicated_patid_acc_tup_list.append(id_acc_tup) #patient with different studies but same accession

        subdir_size = len(subdir_path)
        dir_count += subdir_size
        all_pt_list.append(pt_id)
        if subdir_size != 1:
            multi_pt_list.append(pt_id)
            # 1. pts with diff studies with same accession (associated/linked studies?)
            # 2. pts with diff acc due to collision in pt ids (really different patients)
            # 3. pts with true different studies (more than one study requested per patient)
        else:
            single_pt_list.append(pt_id)


    df_based_on_files = pd.DataFrame(whole_path, columns=['pat_path', 'pat_id', 'acc_path', 'acc_num', 'id_acc_tup'])

    dups_acc_series = pd.Series(dups_acc)

    diff_pt_same_acc = dups_acc_series[~dups_acc_series.isin([tup[1] for tup in duplicated_patid_acc_tup_list])]
    same_pt_same_acc = dups_acc_series[dups_acc_series.isin([tup[1] for tup in duplicated_patid_acc_tup_list])]

    diff_pt_same_acc_df = df_based_on_files[df_based_on_files['acc_num'].isin(diff_pt_same_acc)]
    same_pt_and_acc_df = df_based_on_files[df_based_on_files['acc_num'].isin(same_pt_same_acc)]

    pt_series = pd.Series(all_pt_list, name='pts').map(lambda x: int(x))
    pt_series_dups_list = pt_series[pt_series.duplicated()].tolist()
    print('Acc dups: {}, Pat dups: {}'.format(len(dups_acc_series), len(pt_series_dups_list)))

    print('total subdirectories/accessions: ', dir_count)
    print('single subfolder pt: ', len(single_pt_list))
    print('multiple subfolder pt: ', len(multi_pt_list))
    print('study_acc_dirs: ', len(study_acc_dir_list))
    print('unique_study_acc_dirs: ', len(unique_acc_dir_list))
    print('tot id_acc combos: ', len(tot_patid_acc_tup_list))
    print('unique id_acc combos: ', len(unique_patid_acc_tup_list))
    print('dups id_acc combos: ', len(duplicated_patid_acc_tup_list))
    print('diff_pt_same_acc_#: ', len(diff_pt_same_acc))
    print('same_pt_same_acc_#: ', len(same_pt_same_acc))

    return df_based_on_files

# On patient_MAP, Different patients may have same PATIENT_STUDY_ID and ACCESSION_STUDY_ID.
class Data_Fidelity():
    def __init__(self, map_file=F_IMG_MAP, img_root=MAIN_PATH, r3_req=R3_REQ, params=None):
        self.map_file = map_file
        self.img_root = img_root
        self.orig_req = pd.read_excel(r3_req)

        if params is None:
            params = {
                'r3_acc_id': 'ACCESSION_STUDY_ID',
                'r3_pt_id': 'PATIENT_STUDY_ID',
                'r3_pt_name': 'PAT_NAME',
                'r3_dob_lookup': 'DOBLookup',
                'r3_empi_lookup': 'EMPILookup',
                'r3_birth_date': 'BIRTH_DATE',
                'r3_pt_f_name': 'PatientFirstName',
                'r3_pt_l_name': 'PatientLastName',
                'r3_acc_num': 'AccessionNumber',
                'r3_pt_mrn': 'PatientMRN',
                'orig_acc_num': 'Accession Number',
                'orig_pt_mrn': 'Patient MRN'
            }

        self.params = params

    def last_name_slicer(self, x):
        try:
            return re.search(r'^(.*),', x, re.S).group(1)
        except:
            return 'Error'

    def first_name_slicer(self, x):
        try:
            return re.search(r',(\w*).*$', x, re.S).group(1)
        except:
            return 'Error'

    def acc_id_tup(self, row):
        acc = row[self.params['r3_acc_id']]
        pt_id = row[self.params['r3_pt_id']]
        return (pt_id, acc)

    def collision(self):
        img_map = pd.read_excel(self.map_file)
        img_map = img_map.drop_duplicates() #drop TRUE duplicated rows
        img_map = img_map[img_map[self.params['r3_pt_id']].notna() & img_map[self.params['r3_acc_id']].notna()].copy()
        img_map[self.params['r3_pt_id']] = pd.to_numeric(img_map[self.params['r3_pt_id']], downcast='integer')
        img_map[self.params['r3_acc_id']] = pd.to_numeric(img_map[self.params['r3_acc_id']], downcast='integer')

        # there should be no duplicates as each study should have unique PATIENT_STUDY_ID and ACCESSION_STUDY_ID
        dups = img_map[img_map.duplicated(subset=[self.params['r3_pt_id'], self.params['r3_acc_id']], keep=False)][[self.params['r3_pt_id'], self.params['r3_acc_id']]]
        dups = dups.drop_duplicates().copy() # drop duplicated dups since keep is False
        # dups.to_excel('../PHI/DEXA/R1750_collision_V2.xlsx', index=False)

        #dups are ID-ACC combo, so problem ID's are half of the dups #, as each ID has 2 ACC's with collision

        self.img_map = img_map
        self.dups = dups

    def correlate_data(self):
        img_map = self.img_map

        img_map['id_acc_tup'] = img_map.apply(self.acc_id_tup, axis=1)
        img_map['sliced_F_PATNAME'] = img_map[self.params['r3_pt_name']].map(self.first_name_slicer)
        img_map['sliced_L_PATNAME'] = img_map[self.params['r3_pt_name']].map(self.last_name_slicer)

        img_map_notNA = img_map[img_map[self.params['r3_empi_lookup']].notna()]  # not DOB due to dummy DOB
        bday_filter = img_map_notNA[self.params['r3_dob_lookup']] == img_map_notNA[self.params['r3_birth_date']]
        bday_idx = img_map_notNA[bday_filter].index

        l_name_filter = img_map['sliced_L_PATNAME'] == img_map[self.params['r3_pt_l_name']]
        f_name_filter = img_map['sliced_F_PATNAME'] == img_map[self.params['r3_pt_f_name']]
        name_idx = img_map[l_name_filter].index.union(img_map[f_name_filter].index)

        self.bday_idx = bday_idx
        self.name_idx = name_idx

        self.name_df = self.img_map.loc[name_idx]

        c_dups = self.name_df[self.name_df.duplicated(subset=[self.params['r3_pt_id'], self.params['r3_acc_id']], keep=False)]
        print("id-acc subset sups for name_df: {}".format(c_dups.shape[0]))

        outer_orig_vs_namedf = self.orig_req.merge(self.name_df, left_on=[self.params['orig_pt_mrn'], self.params['orig_acc_num']], right_on=[self.params['r3_pt_mrn'], self.params['r3_acc_num']], how='outer')
        missing_orig_req = outer_orig_vs_namedf[~outer_orig_vs_namedf[self.params['r3_pt_id']].notna()]
        self.missing_orig_req = missing_orig_req

    def check_collision(self, dups_iloc):

        name_df = self.name_df
        # names_df still has ID collision (half of dups), but no ID-ACC combo collision anymore

        non_idx_df = self.img_map[self.img_map[self.params['r3_pt_id']] == self.dups.iloc[dups_iloc][self.params['r3_pt_id']]]
        idx_df = name_df[name_df[self.params['r3_pt_id']] == self.dups.iloc[dups_iloc][self.params['r3_pt_id']]]

        return non_idx_df, idx_df

    def process_img_map(self, unique_accession_dirs_list, multi_pt_list, single_pt_list, pelvic_fx_df):

        img_map = self.name_df.copy()

        img_map[[self.params['r3_pt_id'], self.params['r3_acc_id']]] = img_map[[self.params['r3_pt_id'], self.params['r3_acc_id']]].applymap(lambda x: int(x))
        img_map['has_dir'] = img_map[self.params['r3_acc_id']].isin(unique_accession_dirs_list)  # correlating the db to the dirs, showing missing 48 cases with undefined values.

        img_map_nan = img_map[img_map[self.params['r3_acc_id']] == 999]
        non_dup_img_map = img_map[~img_map[self.params['r3_acc_num']].duplicated()]  # equal to 7677 cases, same as R60.  48 have undefined ACCESSION_STUDY_ID.

        img_map_valid = img_map[img_map[self.params['r3_acc_id']] != 999].copy()
        valid_non_dups_imgs = img_map_valid[~img_map_valid[self.params['r3_acc_num']].duplicated()]  # 7629 cases, excluded 48 with undefined ACCESSION_STUDY_ID.
        img_map_unique_acc_list = valid_non_dups_imgs[self.params['r3_acc_num']].nunique()

        valid_multi_pt = valid_non_dups_imgs[valid_non_dups_imgs[self.params['r3_pt_id']].isin(multi_pt_list)] # pts with multiple accessions
        unique_valid_mult_pt = valid_multi_pt[~valid_multi_pt[self.params['r3_pt_id']].duplicated()]
        valid_single_pt = valid_non_dups_imgs[valid_non_dups_imgs[self.params['r3_pt_id']].isin(single_pt_list)]

        pelvic_fx_df_copy = pelvic_fx_df.copy()

        joined_df = non_dup_img_map.join(pelvic_fx_df_copy.set_index('Accession Number'), on='Accession Number', lsuffix='_img', rsuffix='_fx').fillna(999)
        joined_df[['ID_fx', 'consold_label', 'sep_label']] = joined_df[['ID_fx', 'consold_label', 'sep_label']].applymap(lambda x: int(x))
        joined_df = joined_df.rename(columns={'ID_img': 'ID'}).drop(columns=['ID_fx', 'STUDY_DATE_DICOM'])

        joined_df['valid_map'] = joined_df['Accession Number'].isin(valid_non_dups_imgs['Accession Number'].tolist())  # is the required image valid/exist
        joined_df['valid_map'] = joined_df['valid_map'].map(lambda x: bool(x))

        return joined_df

    def get_final_df(self):
        unique_acc_dirs_list, multi_pt_list, single_pt_list = count_dirs()
        final_df = self.process_img_map(unique_acc_dirs_list, multi_pt_list, single_pt_list, PELVIC_FX_DF)

        return final_df

def consolidate_dup_acc_dirs(modify_files=False):
    dups_acc, dups_series = count_dirs(c_path=MAIN_PATH)

    study_ids_list = []
    prev_acc = './previous_accession'
    to_del_dups = []

    if not modify_files:
        print('Not modifying the files')

    for current_acc in dups_acc:

        study_id = (re.search(r'/.*_(\d{5,})$', current_acc).group(1))

        if study_id in study_ids_list:
            print('curr_acc: ', current_acc)
            print('prev_acc_path: ', prev_acc)
            to_del_dups.append(current_acc)
            series_dir = glob(os.path.join(current_acc, '*'))
            for single_series in series_dir:
                c_series = 'moved_' + re.search(r'/(Series.*$)', single_series).group(1)
                new_path = os.path.join(prev_acc, c_series)
                if modify_files:
                    shutil.copytree(single_series, new_path)

        study_ids_list.append(study_id)
        # study_path = re.search(r'(.*Patient_\d*)', current_acc).group(1)
        prev_acc = current_acc

    for i in to_del_dups:
        if modify_files:
            shutil.rmtree(i)

def place_SB_files(modify_files=False, del_error_files=False, save_excel=False, output_txt=False):
    # final_df = get_final_df()
    final_df = pd.DataFrame()

    valid_df = final_df[final_df['valid_map']]
    study_ids = valid_df['PT_STUDY_ID']
    accessions_ids = valid_df['ACCESSION_STUDY_ID']

    #Make sure new study_ids and accession_ids are fully above or below the current values
    #print(study_ids.min(), study_ids.max()) #7882157104 7888999786
    #print(accessions_ids.min(), accessions_ids.max()) #2000900 93697957

    new_sb_paths = []

    if not modify_files:
        print('not modifying files')

    for orig_id in os.listdir('./SB_images/'):
        orig_id = int(orig_id)
        new_pt_id = 5000000000 + orig_id
        new_accession = 1000000 + orig_id

        c_idx = final_df[final_df['ID'] == orig_id].index
        final_df.loc[c_idx, 'PT_STUDY_ID'] = new_pt_id
        final_df.loc[c_idx, 'ACCESSION_STUDY_ID'] = new_accession
        c_path = './images/PelvisImagesDICOM/Patient_{}/Study_{}/series_none/'.format(new_pt_id, new_accession)
        orig_file_path = os.path.join('./SB_images/', str(orig_id))
        new_sb_paths.append(c_path)
        if modify_files:
            shutil.copytree(orig_file_path, c_path)

    #TODO: Check new_sb_paths; probably should do before copying to other folder
    error_dcm_files = []
    path_is_dir = []

    for i in new_sb_paths:
        series_files = glob(os.path.join(i, '*'))
        for each_dcm_file in series_files:
            if os.path.isdir(each_dcm_file):
                path_is_dir.append(each_dcm_file)
                continue
            try:
                img = pydicom.dcmread(each_dcm_file).pixel_array
                # plt.imshow(img, 'gray')
                # plt.show()
            except:
                error_dcm_files.append(each_dcm_file)

    # Make sure that the failed and dir paths are saved
    with open('./post_move_fail_paths.txt', 'w') as f:
        for i in error_dcm_files:
            f.write(i + '\n')

    with open('./post_move_path_is_dirs.txt', 'w') as f:
        for i in path_is_dir:
            f.write(i + '\n')

    if del_error_files:
        for error_file in error_dcm_files:
            os.remove(error_file)

    if save_excel:
        final_df.to_excel('./after_SB_placement.xlsx')

    return final_df
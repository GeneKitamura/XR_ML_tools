import pydicom
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
import shutil
import cv2
import skimage

from skimage import transform, exposure, util, io
from collections import defaultdict

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from glob import glob

IMG_ROOT = '../L_spine_images/Images/'
MAP_FILE = './../PHI/DEXA/R3_1750_PT_MAP_FINAL.xlsx'
R3_REQ = './../PHI/DEXA/R1750.xlsx'
USER_XLSX = "../PHI/DEXA/L_spine_consolidated.xlsx"

# IMG_ROOT -> patient -> study (acc) -> series (one to many) -> DICOM (one to many)
# Different patients with different PATIENT_STUDY_ID may have studies with same ACCESSION_STUDY_ID.
# Look for Duplicated PAT_ID
# A patient may have 2 subdirs with same ACCESSION_STUDY_ID but different study titles
def count_dirs(img_root=IMG_ROOT):
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
    unique_pt_id = []
    dup_pt_id = []

    for pt in os.listdir(img_root):
        pt_id = pt.split('_')[1]
        pt_id = int(pt_id)
        if pt_id not in unique_pt_id:
            unique_pt_id.append(pt_id)
        else:
            dup_pt_id.append(dup_pt_id)

        try:
            subdir_path = next(os.walk(os.path.join(img_root, pt)))[1]
        except NotADirectoryError as e:
            error_path = os.path.join(img_root, pt)
            print(e)  # not directory such as .DS_Store
            continue

        for pt_acc in subdir_path:
            acc_num = re.search(r'_(\d{5,})', pt_acc).group(1)
            acc_num = int(acc_num)
            study_acc_dir_list.append(acc_num)
            if acc_num not in unique_acc_dir_list:
                unique_acc_dir_list.append(acc_num)
            else:
                dups_acc.append(acc_num)

            id_acc_tup = (pt_id, acc_num)
            whole_path.append([pt, pt_id, pt_acc, acc_num, id_acc_tup])
            tot_patid_acc_tup_list.append(id_acc_tup)
            if id_acc_tup not in unique_patid_acc_tup_list:
                unique_patid_acc_tup_list.append(id_acc_tup)
            else:
                duplicated_patid_acc_tup_list.append(id_acc_tup) #same pat and same accession

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

    print('\nFiles data')
    print('Acc dups: {}, Pat dups: {}'.format(len(dups_acc_series), len(pt_series_dups_list)))
    print('total subdirectories/accessions: ', dir_count)
    print('single subfolder pt: ', len(single_pt_list))
    print('multiple subfolder pt: ', len(multi_pt_list))
    print('study_acc_dirs: ', len(study_acc_dir_list))
    print('unique_study_acc_dirs: ', len(unique_acc_dir_list))
    print('unique pt: {}, dups pt: {}'.format(len(unique_pt_id), len(dup_pt_id)))
    print('tot id_acc combos: ', len(tot_patid_acc_tup_list))
    print('unique id_acc combos: ', len(unique_patid_acc_tup_list))
    print('dups id_acc combos: ', len(duplicated_patid_acc_tup_list))
    print('diff_pt_same_acc_#: ', len(diff_pt_same_acc))
    print('same_pt_same_acc_#: ', len(same_pt_same_acc))
    print('End Files data\n')

    return df_based_on_files

def get_full_paths(df_based_on_files):
    print('\n')
    pat_id_list = []
    acc_num_list = []
    id_acc_tup_list = []
    full_path_list = []
    empty_acc_id_tup_list = []

    for idx, row in df_based_on_files.iterrows():
        try:
            tail_acc_path = os.path.join(row['pat_path'], row['acc_path'])
            full_acc_path = os.path.join(IMG_ROOT, tail_acc_path)

            for series in os.listdir(full_acc_path):
                tail_series_path = os.path.join(tail_acc_path, series)
                one_series_path = os.path.join(full_acc_path, series)

                dicoms = os.listdir(one_series_path)
                if len(dicoms) == 0:
                    empty_acc_id_tup_list.append(row['id_acc_tup']) # multiple folders so one may be empty but others may not
                    continue

                for dicom in dicoms:
                    tail_dicom_path = os.path.join(tail_series_path, dicom)
                    dicom_path = os.path.join(one_series_path, dicom)
                    pat_id_list.append(row['pat_id'])
                    acc_num_list.append(row['acc_num'])
                    id_acc_tup_list.append(row['id_acc_tup'])
                    full_path_list.append(tail_dicom_path)

        except NotADirectoryError as e:
            print(e)
            continue

    full_path_df = pd.DataFrame({'pat_id': pat_id_list, 'acc_num': acc_num_list, 'id_acc_tup': id_acc_tup_list, 'full_path': full_path_list})
    no_files_df = df_based_on_files[~df_based_on_files['id_acc_tup'].isin(full_path_df['id_acc_tup'].unique())]

    print('full_path_df id_acc_tup with files: {}, without files: {}'.format(full_path_df['id_acc_tup'].nunique(), no_files_df['id_acc_tup'].nunique()))

    return full_path_df, no_files_df

# On patient_MAP, Different patients may have same PATIENT_STUDY_ID and ACCESSION_STUDY_ID.
# Need to correlate, either using DOB or first/last name
class Data_Fidelity():
    def __init__(self, map_file=MAP_FILE, img_root=IMG_ROOT, r3_req=R3_REQ, params=None, get_columns=None, user_xlsx=None):
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
                'orig_pt_mrn': 'Patient MRN',
                'orig_pt_age': 'Patient Age',
                'orig_pt_sex': 'Patient Sex'
            }

        self.params = params
        self.get_columns = get_columns
        self.user_xlsx = user_xlsx

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

    def place_columns(self, c_df, user_xlsx, sheet_name=1):
        get_columns = self.get_columns
        if get_columns is None:
            get_columns = [self.params['orig_pt_age'], self.params['orig_pt_sex']]
        get_columns = [self.params['orig_pt_mrn'], self.params['orig_acc_num']] + get_columns
        user_columns_df = pd.read_excel(user_xlsx, sheet_name=sheet_name)[get_columns]
        dups_cases = user_columns_df[user_columns_df.duplicated(subset=['Patient MRN', 'Accession Number'])]
        print('dups MRN-ACC in placed_columns: {}'.format(dups_cases.shape[0]))

        user_columns_df[self.params['orig_pt_mrn']] = pd.to_numeric(user_columns_df[self.params['orig_pt_mrn']], errors='coerce')
        user_columns_df[self.params['orig_acc_num']] = pd.to_numeric(user_columns_df[self.params['orig_acc_num']], errors='coerce')
        user_columns_df = user_columns_df[user_columns_df[self.params['orig_pt_mrn']].notna() & user_columns_df[self.params['orig_acc_num']].notna()].copy()

        user_columns_df[self.params['orig_pt_mrn']] = pd.to_numeric(user_columns_df[self.params['orig_pt_mrn']])
        user_columns_df[self.params['orig_acc_num']] = pd.to_numeric(user_columns_df[self.params['orig_acc_num']])
        placed_df = c_df.merge(user_columns_df, left_on=[self.params['r3_pt_mrn'], self.params['r3_acc_num']], right_on=[self.params['orig_pt_mrn'], self.params['orig_acc_num']], how='left')

        return placed_df

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
        img_map = self.img_map.copy()
        if self.user_xlsx is not None:
            img_map = self.place_columns(img_map, self.user_xlsx)

        img_map['id_acc_tup'] = img_map.apply(self.acc_id_tup, axis=1)
        img_map['sliced_F_PATNAME'] = img_map[self.params['r3_pt_name']].map(self.first_name_slicer)
        img_map['sliced_L_PATNAME'] = img_map[self.params['r3_pt_name']].map(self.last_name_slicer)

        img_map_notNA = img_map[img_map[self.params['r3_empi_lookup']].notna()]  # not DOB due to dummy DOB
        bday_filter = img_map_notNA[self.params['r3_dob_lookup']] == img_map_notNA[self.params['r3_birth_date']]
        bday_idx = img_map_notNA[bday_filter].index

        l_name_filter = img_map['sliced_L_PATNAME'] == img_map[self.params['r3_pt_l_name']]
        f_name_filter = img_map['sliced_F_PATNAME'] == img_map[self.params['r3_pt_f_name']]
        name_idx = img_map[l_name_filter].index.union(img_map[f_name_filter].index)

        self.img_map = img_map
        self.bday_idx = bday_idx
        self.name_idx = name_idx

        self.name_df = self.img_map.loc[name_idx]

        c_dups = self.name_df[self.name_df.duplicated(subset=[self.params['r3_pt_id'], self.params['r3_acc_id']], keep=False)]
        print("id-acc subset dups for name_df: {}".format(c_dups.shape[0]))

        outer_orig_vs_namedf = self.orig_req.merge(self.name_df, left_on=[self.params['orig_pt_mrn'], self.params['orig_acc_num']], right_on=[self.params['r3_pt_mrn'], self.params['r3_acc_num']], how='outer')
        missing_orig_req = outer_orig_vs_namedf[~outer_orig_vs_namedf[self.params['r3_pt_id']].notna()]
        self.missing_orig_req = missing_orig_req

    def check_collision(self, dups_iloc):

        name_df = self.name_df
        # names_df still has ID collision (half of dups), but no ID-ACC combo collision anymore

        non_idx_df = self.img_map[self.img_map[self.params['r3_pt_id']] == self.dups.iloc[dups_iloc][self.params['r3_pt_id']]]
        idx_df = name_df[name_df[self.params['r3_pt_id']] == self.dups.iloc[dups_iloc][self.params['r3_pt_id']]]

        return non_idx_df, idx_df

    def process_img_map(self, save=False):
        df_based_on_files = count_dirs(self.img_root)
        dir_and_map_merged_df = df_based_on_files.merge(self.name_df, on=['id_acc_tup'])

        outer_orig_vs_merged_df = self.orig_req.merge(dir_and_map_merged_df, left_on=[self.params['orig_pt_mrn'], self.params['orig_acc_num']], right_on=[self.params['r3_pt_mrn'], self.params['r3_acc_num']], how='outer')
        missing_orig_vs_merged_df = outer_orig_vs_merged_df[~outer_orig_vs_merged_df[self.params['r3_pt_id']].notna()]
        print('unique id-acc tup combo for final merged df', dir_and_map_merged_df['id_acc_tup'].nunique())
        print('final missing count from orig vs final df: ', missing_orig_vs_merged_df.shape[0])

        NA_id_df = dir_and_map_merged_df[~dir_and_map_merged_df['PATIENT_STUDY_ID'].notna()]
        print('Patient ID for final_merged_df is NA: {}'.format(NA_id_df.shape))

        self.missing_orig_vs_merged_df = missing_orig_vs_merged_df

        self.df_based_on_files = df_based_on_files
        self.dir_and_map_merged_df = dir_and_map_merged_df

        # need to look at full_paths since many directories are EMPTY
        full_path_df, no_files_df = get_full_paths(df_based_on_files)
        full_path_map_merged = full_path_df.merge(self.name_df, on=['id_acc_tup'])
        self.full_path_map_merged = full_path_map_merged

        valid_dir_map_merged = dir_and_map_merged_df[dir_and_map_merged_df['id_acc_tup'].isin(full_path_map_merged['id_acc_tup'].unique())].copy()
        self.valid_dir_map_merged = valid_dir_map_merged

        if save:
            full_path_map_merged.to_excel('../PHI/DEXA/full_path_map_merged.xlsx', index_label='index')
            valid_dir_map_merged.to_excel('../PHI/DEXA/valid_dir_map_merged.xlsx', index_label='index')

        short_full_merged = full_path_map_merged[['new_ID', 'PATIENT_STUDY_ID', 'ACCESSION_STUDY_ID',
                                                  'PatientFirstName', 'PatientLastName', 'full_path',
                                                  'ScheduledDate', 'Patient Age', 'Patient Sex']].copy()

        self.short_full_merged = short_full_merged

def check_dicom_files():
    # Cannot return from functions using concurrent module, so run on Jupyter
    file_path_series = None
    img_root = IMG_ROOT

    def img_shape_getter(x):
        c_path = os.path.join(img_root, x)
        try:
            img = pydicom.dcmread(c_path).pixel_array
            return 'pass'
        except Exception as e:  # dummy values when img.shape != 3
            print('Exception at {} as {}'.format(x, e))
            return x

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(img_shape_getter, i) for i in file_path_series}

    count_dict = defaultdict(int)
    for r in as_completed(futures, timeout=10):
        try:
            c_output = r.result()
            count_dict[c_output] += 1
        except Exception as e:
            print('exception as {}'.format(e))

    # Cannot return from functions using concurrent module, so run on Jupyter

def create_dicom_images():
    # Cannot return from functions using concurrent module, so run on Jupyter

    c_df = None
    output_size=238
    conv_to_uint8=False
    stack=False
    img_root=IMG_ROOT
    np_save_file = './2000_images_files'

    def img_creater(idx, row):
        dicom_path = os.path.join(img_root, row['full_path'])
        dcm = pydicom.dcmread(dicom_path)

        try:
            if dcm.DerivationDescription:  # derivative view
                return 0, idx, np.zeros((output_size, output_size))
        except AttributeError:  # no derivative view attribute, so move on to image creation
            pass

        _img = dcm.pixel_array
        bs = dcm.BitsStored

        try:
            _img = exposure.rescale_intensity(_img, in_range=('uint' + str(bs)))
        except ValueError: # odd number pixel
            pass

        if conv_to_uint8:
            _img = skimage.img_as_uint(_img)

        if dcm.PhotometricInterpretation == 'MONOCHROME1':
            _img = cv2.bitwise_not(_img)

        _img = transform.resize(_img, (output_size, output_size), mode='reflect', anti_aliasing=True, preserve_range=True)  # img_as_float

        if stack:
            _img = np.stack([_img, _img, _img], axis=-1)

        return 1, idx, _img

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(img_creater, idx, row) for idx, row in c_df.iterrows()}

    index_array = []
    image_array = []
    flag_array = []

    # return order is random
    for r in as_completed(futures, timeout=10):
        c_output = r.result()
        flag_array.append(c_output[0])
        index_array.append(c_output[1])
        image_array.append(c_output[2])

    image_array = np.array(image_array)
    index_array = np.array(index_array)
    flag_array = np.array(flag_array)

    # np.savez(np_save_file, image_array=image_array, index_array=index_array, flag_array=flag_array)

    # Cannot return from functions using concurrent module, so run on Jupyter


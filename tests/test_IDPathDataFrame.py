from IDPaths import *
import pandas as pd

id_regex = 'BASE_[0-9]{3}'
paths = {
  'artifact wp': '/Users/jason/Desktop/test/artifact wp/*WP*.xlsx',
  'artifact psg': '/Users/jason/Desktop/test/artifact psg/*.xlsx',
  'hypnogram': '/Users/jason/Desktop/test/hypnogram/*'
}

base = IDPaths(paths, id_regex, sort=True)
base.check_id_regex(print_mode=True)
base.subset_by_check(['check_id_regex'])
base.subset_by_regex(_dict={'artifact wp': '2.xlsx', 'artifact psg': '2.xlsx', 'hypnogram': '2.xlsx'}, negate=True)

path_df = base.create_id_df()

path_df.get_df()
path_df.get_ids()
path_df.get_complete_cases()
path_df.colnames()
path_df.shape()
path_df.summary()

path_df.add_read_function('hypnogram', pd.read_excel, {'engine': 'openpyxl'})
path_df.add_assertions('hypnogram', allow_missing_values=True)

path_df.add_read_function('artifact psg', pd.read_excel, {'engine': 'openpyxl'})
path_df.add_assertions('artifact psg', allow_missing_values=True)

path_df.add_read_function('artifact wp', pd.read_excel, {'engine': 'openpyxl'})
path_df.add_assertions('artifact wp', allow_missing_values=True)

path_df.get_assertions()
path_df.get_read_functions()

path_df.read_file('hypnogram', idx='BASE_101')
path_df.read_files(columns=['hypnogram', 'artifact wp'], ignore_missing=True, stop_on_error=False, keep_df=False)

View(path_df.get_df_log())
path_df.get_df_dict(idx='BASE_107')

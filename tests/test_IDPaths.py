from IDPaths import *
import pandas as pd

# TODO: basic assertions/checks
# TODO: documentation
# TODO: termcolor not working

id_regex = 'BASE_[0-9]{3}'
paths = {
  'artifact wp': '/Users/jason/Desktop/test/artifact wp/*WP*.xlsx',
  'artifact psg': '/Users/jason/Desktop/test/artifact psg/*.xlsx',
  'hypnogram': '/Users/jason/Desktop/test/hypnogram/*xlsx'
}

base = IDPaths(paths, id_regex, sort=True)
base.get_id_regex()
base.get_kept_files()
base.get_all_files()
base.get_removed_files()

base.print_paths()
base.print_kept_files()
base.print_removed_files()
base.print_all_files()
base.print_all_files(tags = ['artifact wp'], regex='2.xlsx')

base.check_id_regex(print_mode=True)
base.print_check_results()
base.get_check_results('check_id_regex')
base.subset_by_check(['check_id_regex'])
base.get_removed_files()
base.print_removed_files()
base.print_all_files(tags=['artifact wp'], regex='2.xlsx')
base.check_id_regex(print_mode=True)

base.check_duplicate_ids(print_mode=True, include_all=False)
base.get_check_results('check_duplicate_ids')
base.print_check_results(n_max = 5)
base.subset_by_regex(_dict={'artifact wp': '2.xlsx', 'hypnogram': '2.xlsx'}, negate=True)
base.print_removed_files()
base.print_kept_files()
base.print_all_files()
base.check_duplicate_ids(print_mode=True)
base.check_duplicate_files(print_mode=True)
base.get_removed_files()

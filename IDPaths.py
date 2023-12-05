import re
import glob  
import os
import shutil
import warnings
from numpy import shape
from itertools import combinations
from pandas import DataFrame, concat
from termcolor import colored
from tkinter import Tk
from collections import OrderedDict
from tkinter.filedialog import askopenfilenames

class IDPaths:
  
  def __init__(self, paths, id_regex, sort=False):
    assert isinstance(paths, dict), '`paths` must be a dictionary'
    assert isinstance(id_regex, str), '`id_regex` must be a string'
    assert isinstance(sort, bool), '`sort` must equal `True` or `False`'
    
    self.__path_dict = paths
    self.__id_regex = id_regex
    self.__store_files()
    self.__create_check_dicts()
    
    if sort:
      for k in self.__fdict_orig.keys():
        self.__fdict_orig[k] = sorted(self.__fdict_orig[k])
        
    self.__fdict_kept = self.__fdict_orig.copy()
      
    self.__fdict_removed = OrderedDict()
    for k in self.__fdict_orig.keys():
      self.__fdict_removed[k] = []
    
  def __store_files(self):
    """Stores a dictionary of file paths for each tag"""
    files = OrderedDict()
    for k in self.__path_dict.keys():
      files_globbed = glob.glob(self.__path_dict[k])
      if len(files_globbed) == 0:
        raise Exception(f'No files were found in {k}, check your paths!')
      else:
        files[k] = glob.glob(self.__path_dict[k])
    self.__fdict_orig = files
      
  def __create_check_dicts(self, methods=['check_id_regex', 'check_duplicate_ids', 'check_duplicate_files']):
    self.__results = OrderedDict()
    for method in methods:
      self.__results[method] = OrderedDict()
      for k in self.__fdict_orig.keys():
        self.__results[method][k] = []
        
  def print_paths(self):
    """Returns a print-friendly version of the paths dictionary"""
    for k in self.__path_dict.keys():
      print(f'{k}: {self.__path_dict[k]}', sep='\n')
    
  def get_id_regex(self):
    """Returns the regular expression used for all subsequent file-checking methods"""
    return self.__id_regex
  
  def get_all_files(self):
    return self.__fdict_orig.copy()
  
  def get_kept_files(self):
    return self.__fdict_kept.copy()
  
  def get_removed_files(self):
    return self.__fdict_removed.copy()
  
  def get_check_results(self, method):
    return self.__results[method].copy()
  
  def clear_kept_files(self, method):
    return self.__fdict_kept.clear()
  
  def clear_removed_files(self, method):
    return self.__fdict_removed.clear()
  
  def clear_check_results(self, method):
    return self.__results[method].clear()

  def __print_files(self, _dict, tags=None, n_max=None, basename=True, regex=None, regex_clr='yellow'):
    """Returns a print-friendly version of tags and files, and the number of files in each tag"""
    if tags is None:
      tags = _dict.keys()
    for ti,t in enumerate(tags):
      if len(_dict[t]) > 0:
        print(f'{t} files (N = {len(_dict[t])})')
        for i,file in enumerate(_dict[t]):
          if regex is not None:
            file_match = re.search(regex, os.path.basename(file))
          if n_max is None or i < n_max:
            if file in self.__fdict_removed[t]:
              if basename:
                file = os.path.basename(file)
              print('  ' + ''.join(['[' + str(i) + '] ', colored(file, 'red')]))
            elif regex is not None and file_match is not None:
              if basename:
                file = os.path.basename(file)
              print('  ' + ''.join(['[' + str(i) + '] ', colored(file, regex_clr)]))
            else:
              if basename:
                file = os.path.basename(file)
              print('  ' + ''.join(['[' + str(i) + '] ', file]))
        if ti < (len(tags) - 1):
          print('')
      else:
        print(f'{t} files (N = 0)')
   
  def print_all_files(self, **kwargs):
    self.__print_files(self.__fdict_orig, **kwargs)
    
  def print_removed_files(self, **kwargs):
    self.__print_files(self.__fdict_removed, **kwargs)
    
  def print_kept_files(self, **kwargs):
    self.__print_files(self.__fdict_kept, **kwargs)
    
  def print_check_results(self, methods=None, **kwargs):
    if methods is None:
      methods = self.__results.keys()
    for i,method in enumerate(methods):
      print(f'METHOD: {method}')
      self.__print_files(self.__results[method], **kwargs)
      if i < (len(methods) - 1):
        print('')
  
  def subset_by_name(self, _dict, remove=True):
    for k in _dict.keys():
      f_inter,f_diff = self.__compare_file_lists(_dict[k], self.__fdict_kept[k])
      if remove:
        self.__add_to_dict(self.__fdict_kept, k, f_diff, method='replace')
        self.__add_to_dict(self.__fdict_removed, k, f_inter, method='append')
      else:
        self.__add_to_dict(self.__fdict_kept, k, f_inter, method='replace')
        self.__add_to_dict(self.__fdict_removed, k, f_diff, method='append')
      print(f'{k} files removed')
      
  def __compare_file_lists(self, user_list, full_list):
    f_all = set(full_list)
    f_inter = sorted(list(f_all.intersection(set(user_list))))
    f_diff = sorted(list(f_all.difference(set(user_list))))
    return f_inter,f_diff 
      
  def __add_to_dict(self, _dict, tag, f_list, method):
    if method == 'append':
      for f in f_list:
        if f not in _dict[tag]:
          _dict[tag].append(f)
    else:
      _dict[tag] = f_list
      
  def subset_by_gui(self, remove=True):
    Tk().withdraw() 
    gui_dict = {}
    for tag,path in self.__path_dict.items():
      path = os.path.dirname(path)
      gui_dict[tag] = askopenfilenames(initialdir=path)
    self.subset_by_name(_dict=gui_dict, remove=remove)
        
  def __subset_by_index(self, _dict, remove, basename=True):
    for k,v in _dict.items():
      f_all = self.__fdict_kept[k]
      f_sub = []
      for i in range(len(f_all)):
        if remove:
          if i not in v:
            f_sub.append(f_all[i])
        else:
          if i in v:
            f_sub.append(f_all[i])
      
      f_inter,f_diff = self.__compare_file_lists(f_sub, f_all)
      if remove:
        self.__add_to_dict(self.__fdict_kept, k, f_diff, method='replace')
        self.__add_to_dict(self.__fdict_removed, k, f_inter, method='append')
      else:
        self.__add_to_dict(self.__fdict_kept, k, f_inter, method='replace')
        self.__add_to_dict(self.__fdict_removed, k, f_diff, method='append')
        
    if remove:
      print('Files removed!')
    else:
      print('Files kept!')
          
  def subset_by_check(self, methods, remove=True):
    for method in methods:
      check_res = self.get_check_results(method)
      if all([len(check_res[k]) == 0 for k in check_res.keys()]):
        warnings.warn('The dictionary is empty and nothing was removed!')
      else:
        for k in check_res.keys():
          f_inter,f_diff = self.__compare_file_lists(check_res[k], self.__fdict_kept[k])
          if remove:
            self.__add_to_dict(self.__fdict_kept, k, f_diff, method='replace')
            self.__add_to_dict(self.__fdict_removed, k, f_inter, method='append')
          else:
            self.__add_to_dict(self.__fdict_kept, k, f_inter, method='replace')
            self.__add_to_dict(self.__fdict_removed, k, f_diff, method='append')
        print(f'Removed all files in `{method}` results!')

  def subset_by_regex(self, _dict, negate=True, basename=True):
    tag_indices = {}
    for k,v in _dict.items():
      match_indices, non_match_indices = [],[]
      for i,file in enumerate(self.__fdict_kept[k]):
        file_match = re.search(v, os.path.basename(file))
        if file_match is not None:
          match_indices.append(i)
        else:
          non_match_indices.append(i)
      if negate:
        tag_indices[k] = non_match_indices
      else:
        tag_indices[k] = match_indices
    self.__subset_by_index(_dict=tag_indices, basename=basename, remove=negate)  
    
  def __as_bool(self, counter_list):
    return all([_x == 0 for _x in counter_list])
  
  def __check_print_helper(self, print_mode, counter_list, as_bool):
    if self.__as_bool(counter_list) and print_mode:
      print('Passed check!')
    if not print_mode:
      if as_bool:
        return self.__as_bool(counter_list)
      else:
        return counter_list
        
  def check_id_regex(self, tags=None, print_mode=False, as_bool=False, basename=True):
    """Checks whether all files contain the specfied regular expression. Returns a bool."""
    
    if tags is None:
      tags = self.__fdict_kept.keys()
    
    counter_list = []
    for ti,t in enumerate(tags):
      tag_print_bool = False
      counter = 0
      for i,file in enumerate(self.__fdict_kept[t]):
        id_match = re.search(self.__id_regex, os.path.basename(file))
        if id_match is None:
          counter += 1
          if file not in self.__results['check_id_regex'][t]:
            self.__results['check_id_regex'][t].append(file)
          if print_mode:
            if not tag_print_bool:
              print(t)
              tag_print_bool = True
            if basename:
              file = os.path.basename(file)
            print('  ' + ''.join(['[' + str(i) + '] ', file]))
            
      counter_list.append(counter)
    return self.__check_print_helper(print_mode, counter_list, as_bool)
  
  def check_duplicate_ids(self, tags=None, force=False, print_mode=False, as_bool=False, basename=True, include_all=True):
    """Checks whether any IDs are duplicated. Returns a bool."""
    
    ids_verified = self.check_id_regex(tags=tags, as_bool=True)
    
    if not ids_verified and not force:
      raise Exception('Not all ids are verified. Run `.check_id_regex()`, or use force=True to skip these ids')
    
    if tags is None:
      tags = self.__fdict_kept.keys()
      
    counter_list = []
    for ti,t in enumerate(tags):
      counter = 0
      id_dict = {}
      files = self.__fdict_kept[t]
      for i,file in enumerate(files):
        id_match = re.search(self.__id_regex, os.path.basename(file))
        if id_match is None and force:
          continue
        if id_match[0] not in id_dict.keys():
          id_dict[id_match[0]] = [(i, file)]
        else:
          counter += 1
          id_dict[id_match[0]].append((i, file))
          if include_all:
            first_file = id_dict[id_match[0]][0][1]
            if first_file not in self.__results['check_duplicate_ids'][t]:
              self.__results['check_duplicate_ids'][t].append(first_file)
          if file not in self.__results['check_duplicate_ids'][t]:
            self.__results['check_duplicate_ids'][t].append(file)
            
      counter_list.append(counter)
    
      if print_mode:
        if counter > 0:
          print(colored(t, 'yellow'))
        for k,v in id_dict.items():
          if len(v) > 1:
            print('  ' + k)
            for i,file in v:
              if basename:
                file = os.path.basename(file)
              print(f'    [{i}] {file}')
      
    return self.__check_print_helper(print_mode, counter_list, as_bool)
  
  def check_duplicate_files(self, tags_to_compare=None, print_mode=False, as_bool=False, basename=True):
    
    if tags_to_compare is None:
      tags_to_compare = self.__fdict_kept.keys()
    
    combos = list(combinations(tags_to_compare, 2))
    
    counter_list = []
    for c1,c2 in combos:
      counter = 0
      s1 = set(self.__fdict_kept[c1])
      s2 = set(self.__fdict_kept[c2])
      set_inter = list(s1.intersection(s2))
      if len(set_inter) != 0:
        
        for dup_file in set_inter:
          counter += 1
          if dup_file not in self.__results['check_duplicate_files'][c1]:
            self.__results['check_duplicate_files'][c1].append(dup_file)
          if dup_file not in self.__results['check_duplicate_files'][c2]:
            self.__results['check_duplicate_files'][c2].append(dup_file)
            
        if print_mode:
          if basename:
            set_inter = [os.path.basename(x) for x in set_inter]
          print(f'{c1} and {c2} have the following files in common:')
          print('  ' + '\n  '.join(set_inter), end='\n')
      
      counter_list.append(counter)
      return self.__check_print_helper(print_mode, counter_list, as_bool)
  
  def run_all_checks(self):
    ids_verified = self.check_id_regex(as_bool=True)
    all_ids_unique = self.check_duplicate_ids(as_bool=True)
    all_files_unique = self.check_duplicate_files(as_bool=True)

    if not ids_verified:
      raise Exception('Not all ids are verified. Run `.check_id_regex()`')

    if not all_ids_unique:
      raise Exception('Some ids are duplicated. Run `.check_duplicate_ids()`')

    if not all_files_unique:
      raise Exception('Some files are duplicated. Run `.check_duplicate_files()`')
  
    return True
  
  def create_id_df(self):
    if self.run_all_checks():
      id_df = IDPathDataFrame(self.__id_regex, self.__fdict_kept)
      return id_df
    
    
class IDPathDataFrame():
  
  def __init__(self, id_regex, _dict):
    self.__id_col = 'id'
    self.__copy_root_path = None
    self.__df = None
    self.__read_dict = {}
    self.__assertion_dict = {}
    self.__create_data_frame(id_regex, _dict)
    
  def __create_id_dict(self, id_regex, _dict):
    """Creates a dictionary of files and tags, organized by ID."""
    id_dict = OrderedDict()
    id_list = []
    for t in _dict.keys():
      for file in _dict[t]:
        id_match = re.search(id_regex, os.path.basename(file))[0]
        if id_match not in id_list:
          id_dict[id_match] = [(t, file)]
          id_list.append(id_match)
        else:
          id_dict[id_match].append((t, file))
    return id_dict
  
  def __create_data_frame(self, id_regex, _dict):
    id_dict = self.__create_id_dict(id_regex, _dict)

    cols = list(_dict.keys())
    cols.insert(0, self.__id_col)
    df = DataFrame(columns=cols)

    for idx in sorted(id_dict.keys()):
      vals = id_dict[idx]
      df_dict = {self.__id_col: idx}
      for tag,file in vals:
        df_dict[tag] = file
      df = concat([df, DataFrame(df_dict, index=[0])], ignore_index=True)

    self.__df = df
  
  def get_df(self):
    return self.__df.copy()
  
  def get_ids(self):
    return self.__df[self.__id_col].copy()
  
  def get_complete_cases(self, columns=None, how='any'):
    if columns is None:
      columns = list(self.colnames())
    return self.__df[columns].dropna(axis=0, how=how).copy()
  
  def colnames(self):
    return self.__df.columns
  
  def shape(self):
    return shape(self.__df)
  
  def summary(self):
    cnt_miss = self.__df.isna().sum()
    cnt_nonmiss = self.__df.count()
    df = DataFrame({
      'nonmiss_count': cnt_nonmiss,
      'nonmiss_perc': round(cnt_nonmiss/self.shape()[0], 2),
      'miss_count': cnt_miss,
      'miss_perc': round(cnt_miss/self.shape()[0], 2)
    })
    return df
  
  def add_read_function(self, column, FUN, kw_dict=None):
    self.__read_dict[column] = {'FUN': FUN, 'kwargs': kw_dict}
    
  def add_assertions(self, column, colnames=None, ignore_case=False, check_col_order=True, 
    ncols=None, nrows=None, allow_missing_values=False):
  
    self.__assertion_dict[column] = {
      'colnames': colnames,
      'ignore_case': ignore_case,
      'check_col_order': check_col_order,
      'ncols': ncols,
      'nrows': nrows,
      'allow_missing_values': allow_missing_values
    }
    
  def get_assertions(self, column=None):
    return self.__assertion_dict.copy()
  
  def check_assertions(self, df, colnames=None, ignore_case=False, check_col_order=True, 
    ncols=None, nrows=None, allow_missing_values=False):
  
    if ncols is not None:
      assert ncols == df.shape[1], f'`ncols` is {ncols}, but there are {df.shape[1]} columns'
    
    if nrows is not None:
      assert nrows == df.shape[0], f'`nrows` is {nrows}, but there are {df.shape[0]} columns'
    
    if colnames is not None:
      cn = list(df.columns)
      if ignore_case:
        cn = [x.lower() for x in cn]
      if check_col_order:
        assert colnames == cn, 'Not all columns match in order'
      else:
        cn_set = set(cn)
        assert len(cn_set.difference(set(colnames))) == 0, 'Not all column names match'
        
    if not allow_missing_values:
      assert all([all(df.notna()[x]) for x in list(df.columns)]), 'Some columns have missing values'
    
  def get_read_functions(self):
    return self.__read_dict.copy()
  
  def read_file(self, column, idx, check_assertions=True):
    assert column in list(self.colnames()), f'{column} not found in data frame'
    assert column in self.__read_dict.keys(), f'You must first add a reading function for {column} column'
    assert idx in list(self.get_ids()), f'{idx} does not exist in data frame'

    f = self.__df[self.__df[self.__id_col] == idx][column]
    if f.isna().all():
      raise Exception(f'{idx} has a missing value for {column}')
    else:
      f = f.to_list()[0]
      
    read = self.__read_dict[column]
    try:
      res = read['FUN'](f, **read['kwargs']) if read['kwargs'] is not None else read['FUN'](f)
    except Exception as e:
      return e
      
    if check_assertions:
      try:
        self.check_assertions(res, **self.__assertion_dict[column])
      except AssertionError as e:
        return e
      
    return res
  
  def read_files(self, columns=None, ids=None, ignore_missing=True, check_assertions=True, 
    log=True, keep_df=True, stop_on_error=True):
    
    self.__df_dict = OrderedDict()
    all_columns = [x for x in list(self.colnames()) if x not in self.__id_col]
    
    if log:
      df_log = self.__df.copy()
      df_log.loc[:, df_log.columns != self.__id_col] = ''
      for c in all_columns:
        c_index = df_log.columns.get_loc(c)
        missing_indices = self.__df[self.__df[c].isna()].index
        df_log.iloc[missing_indices, c_index] = 'MISSING'
      
    if columns is None:
      columns = all_columns
      
    s_columns = set(columns)
    missing_read_fun = s_columns.difference(set(self.__read_dict.keys()))
    if len(missing_read_fun) > 0:
      raise Exception(f'The following columns are missing a read function: {missing_read_fun}')
    
    if ids is None:
      ids = list(self.get_ids())
    
    for c in columns:
      self.__df_dict[c] = OrderedDict()
      if ignore_missing:
        df = self.__df[self.__df[c].notna()]
        ids_sub = df[self.__id_col].to_list()
        ids_loop = [x for x in ids if x in ids_sub]
      else:
        ids_loop = ids
      for idx in ids_loop:
        res = self.read_file(column=c, idx=idx, check_assertions=check_assertions)
        if isinstance(res, (Exception, AssertionError)):
          if stop_on_error:
            raise Exception(f'{idx} in {c} column: {res}')
        if log:
          c_index = df_log.columns.get_loc(c)
          r_index = df_log[df_log[self.__id_col] == idx].index
          if isinstance(res, (Exception, AssertionError)):
            df_log.iloc[r_index, c_index] = str(res)
          else:
            df_log.iloc[r_index, c_index] = 'Read'
        if keep_df:
          if not isinstance(res, (Exception, AssertionError)):
            self.__df_dict[c][idx] = res
          
    if log:
      self.__df_log = df_log
        
  def get_df_log(self):
    return self.__df_log.copy()
        
  def get_df_dict(self, column=None, idx=None):
    x = self.__df_dict.copy()
    if column is not None and idx is not None:
      x = x.get(column).get(idx)
    elif column is not None:
      x = x.get(column)
    elif idx is not None:
      tmp = {}
      for k in x.keys():
        tmp[k] = x[k].get(idx)
      x = tmp
    return x
  
  def copy_files(self, root_path, df=None, id_col='id', 
    copy_cols=None, mk_dirs=False, overwrite=False, tag_dict=None):
    
    self.__copy_root_path = root_path
    
    if df is None:
      df = self.__df
      
    if copy_cols is None:
      copy_cols = df[df.columns.difference([id_col])].columns
      
    for c in copy_cols:
      x = df[c]
      cn = x.name
      if tag_dict is not None:
        if cn in tag_dict.keys():
          cn = tag_dict[cn]
      subpath = os.path.join(root_path, cn)
      if not os.path.isdir(subpath):
        if mk_dirs:
          os.makedirs(subpath)
        else:
          raise Exception('The directory does not exist and `mk_dirs` was set to False!')
      if len(os.listdir(subpath)) > 0 and not overwrite:
        raise Exception('The directory is not empty. Set `overwrite=True` to overwrite current files!')
      for r in range(len(x)):
        if x.notna().values[r]:
          shutil.copy2(x[r], os.path.join(subpath, os.path.basename(x[r])))

  def replace_paths(self, _dict=None, use_copy_root_path=False, tag_dict=None):
    
    if _dict is not None and use_copy_root_path:
      raise Exception('You cannot provide a dictionary and also use the copy root path')
    
    x = self.__df
    if _dict is None:
      if use_copy_root_path:
        if self.__copy_root_path is None:
          raise Exception('You must first copy files before using the copy root path!')
        else:
          _dict = {}
          tmp = x[x.columns.difference([self.__id_col])]
          for c in tmp.columns:
            _dict[c] = self.__copy_root_path
      else:
        raise Exception('Either a dictionary must be specified or `use_copy_root_path` should be True')
        
    for c,path in _dict.items():
      if tag_dict is not None:
        if c in tag_dict.keys():
          c_rename = tag_dict[c]
        else:
          c_rename = c
      else:
        c_rename = c
      for i,f in enumerate(x[c][x[c].notna()].to_list()): 
        x[c][x[c].notna()][i] = os.path.join(path, c_rename, os.path.basename(f)) 
    self.__df = x
    return self.__df.copy()
  
  def save(self, path, df=None, **kwargs):
    if df is None:
      df = self.__df
    df.to_csv(path, **kwargs)

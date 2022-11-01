import os
import math
import json


def dump_file(file_path, data):
  with open(file_path, 'w') as fp:
    json.dump(data, fp, indent=2)

root = "./"
clas_dirs = os.listdir(root)
total_train_files = []
total_valid_files = []
total_test_files = []
min_num = 10e9
for clas_dir in clas_dirs:
  if not os.path.isdir(root + '/' + clas_dir):
    continue
  files = os.listdir(root + '/' + clas_dir)
  num_files = len(files)
  min_num = min(min_num, num_files)
print(min_num)

min_num
for clas_dir in clas_dirs:
  if not os.path.isdir(root + '/' + clas_dir):
    continue
  files = os.listdir(root + '/' + clas_dir)
  # num_files = len(files)
  train_files = []
  valid_files = []
  test_files = []
  for i in range(math.floor(min_num * 0.8)):
    if '.json' in files[i]:
      train_files.append(clas_dir + '/' + files[i])
  for i in range(math.floor(min_num * 0.8), math.floor(min_num * 0.9), 1):
    if '.json' in files[i]:
      valid_files.append(clas_dir + '/' + files[i])
  for i in range(math.floor(min_num * 0.9), min_num, 1):
    if '.json' in files[i]:
      test_files.append(clas_dir + '/' + files[i])
  print(clas_dir + ' files num train: ', len(train_files), " val: ", len(valid_files), " test: ", len(test_files))
  total_train_files.extend(train_files)
  total_valid_files.extend(valid_files)
  total_test_files.extend(test_files)
info = {'train': total_train_files, 'val': total_valid_files, 'test': total_test_files}
dump_file(root + 'train_val_test.json', info)
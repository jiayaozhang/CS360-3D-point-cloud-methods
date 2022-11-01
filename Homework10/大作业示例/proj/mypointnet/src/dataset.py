import open3d as o3d
import torch
import os
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


def show_np_pts3d(np_pts):
    pc_view = o3d.geometry.PointCloud()
    pc_view.points = o3d.utility.Vector3dVector(np_pts)
    o3d.visualization.draw_geometries([pc_view])


def load_json_file(file_path):
  data = None
  with open(file_path, 'r') as fp:
    data = json.load(fp)
  return data


def read_file_names_from_file(file):
  with open(file, 'r') as f:
    files = []
    for line in f:
      files.append(line.split('\n')[0])
  return files


class PointNetDataset(Dataset):
  def __init__(self, root_dir, train):
    super(PointNetDataset, self).__init__()

    self._train = train
    self._classes = []

    self._features = []
    self._labels = []

    # nums of points are different in different clusters,
    # so we must record the max num to keep each cluster having same num of points
    self._max_size_pts = 0 

    self.load(root_dir)

  def classes(self):
    return self._classes

  def __len__(self):
    return len(self._features)
  
  def __getitem__(self, idx):
    feature, label = self._features[idx], self._labels[idx]
    # if self._max_size_pts == feature.shape[0]:
    #   print(label)
    #   show_np_pts3d(feature)
    # normalize
    center = np.expand_dims(np.mean(feature, axis = 0), 0)
    feature = feature - center
    dist = np.max(np.sqrt(np.sum(feature ** 2, axis = 1)),0)
    feature = feature / dist #scale
    # rotation
    theta = np.random.uniform(0, np.pi*2)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
        [np.sin(theta), np.cos(theta)]])
    feature[:,[0,2]] = feature[:,[0,2]].dot(rotation_matrix)
    # jitter
    feature += np.random.normal(0, 0.02, size=feature.shape)
    # keep each cluster with same points

    if self._max_size_pts > feature.shape[0]:
      compenstate = np.zeros((self._max_size_pts - feature.shape[0], 3))
      feature = np.concatenate((feature, compenstate), axis=0)
  
    l_lable = [0 for _ in range(len(self._classes))]
    l_lable[self._classes.index(label)] = 1

    feature = torch.Tensor(feature.T)
    label = torch.Tensor(l_lable)

    return feature, label
  
  def load(self, root_dir):
    self._classes = ['car', 'ped', 'cyclist', 'genobj']
    # get train/val/test files
    files = []
    things = os.listdir(root_dir)
    for f in things:
      if f == "train_val_test.json":
        train_val_test_files = load_json_file(root_dir + '/' + f)
        if self._train == 0:
          files = train_val_test_files['train']
        elif self._train == 1:
          files = train_val_test_files['test']
        elif self._train == 2:
          files = train_val_test_files['val']
        elif self._train == 3:
          files = train_val_test_files['test']
        
      if len(files) > 0:
        break
    
    tmp_classes = []
    for file in files:
      kind = file.split("/")[0]
      # print(kind)
      if kind not in tmp_classes:
        tmp_classes.append(kind)
      else:
        if self._train == 3:
          continue
      pcd_file = root_dir + '/' + file
      np_pts = np.array(load_json_file(pcd_file))
      # if np_pts.shape[0] == 11846:
      #   print(pcd_file)
      #   show_np_pts3d(np_pts)
      self._max_size_pts = max(self._max_size_pts, np_pts.shape[0])
      self._features.append(np_pts)
      self._labels.append(kind)
    print('max size: ', self._max_size_pts)
    if self._train == 0:
      print("There are " + str(len(self._labels)) + " trian files.")
    elif self._train == 1:
      print("There are " + str(len(self._labels)) + " test files.")
    elif self._train == 2:
      print("There are " + str(len(self._labels)) + " valid files.")
    elif self._train == 3:
      print("There are " + str(len(self._labels)) + " app files.")
      

if __name__ == "__main__":
  train_data = PointNetDataset("../../dataset", train=0)
  train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
  cnt = 0
  for pts, label in train_loader:
    print(pts.shape)
    cnt += 1
    if cnt > 3:
      break
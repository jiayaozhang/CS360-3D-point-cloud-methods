import open3d as o3d
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial import KDTree
from sklearn.neighbors import KDTree
import pretty_errors
import faulthandler
from numpy.linalg import *


def read_pcd_from_file(file):
  np_pts = np.zeros(0)
  with open(file, 'r') as f:
    pts = []
    for line in f:
      one_pt = list(map(float, line[:-1].split(',')))
      pts.append(one_pt[:3])
    np_pts = np.array(pts)
  return np_pts


def show_np_pts3d(np_pts):
  pc_view = o3d.geometry.PointCloud()
  pc_view.points = o3d.utility.Vector3dVector(np_pts)
  o3d.visualization.draw_geometries([pc_view])


def point_cloud_show(point_cloud, feature_point):
  fig = plt.figure(dpi=150)
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], cmap='spectral', s=2, linewidths=0, alpha=1, marker=".")
  ax.scatter(feature_point[:, 0], feature_point[:, 1], feature_point[:, 2], cmap='spectral', s=2, linewidths=5, alpha=1,marker=".",color='red')
  plt.title('Point Cloud')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.show()


def process_pcd_files(dir):
  files = []
  things = os.listdir(dir)
  for f in things:
    sub_dir = dir + '/' + f
    if os.path.isdir(sub_dir) and f[0] != '.':
      sub_files = os.listdir(sub_dir)
      for sub_file in sub_files:
        files.append(sub_dir + '/' + sub_file)
        break # choose one file in one cat.
  return files


def get_cov_matrix(pt, neighbor_pts, tree):
  pt_matrix = np.array([pt])
  # ngbsngb_ids_array = tree.query_radius(neighbor_pts, 0.03)
  # weights = [1.0 / len(ngb_ids) for ngb_ids in ngbsngb_ids_array]
  # sum_weights = np.sum([w ** 2 for w in weights])
  # cov_matrix = np.cov(np.array([weights]).T * (neighbor_pts - pt_matrix), rowvar=0)
  cov_matrix = np.cov(neighbor_pts - pt_matrix, rowvar=0)
  return cov_matrix


def get_eigenvalues(cov_matrix):
  eigenvalues, eigenvectors = np.linalg.eig(np.mat(cov_matrix))
  sort = eigenvalues.argsort()[::-1]
  eigenvalues = eigenvalues[sort]
  eigenvectors = eigenvectors[:, sort]
  return eigenvalues


def is_key_point(eigenvalues, threshold):
  if eigenvalues[2] < threshold:
    return False
  threshold1 = 0.5
  threshold2 = 0.5
  if (eigenvalues[1] / eigenvalues[0] < threshold1) and \
     (eigenvalues[2] / eigenvalues[1] < threshold2):
    return True
  else:
    return False


def iss(np_pts):
  tree = KDTree(np_pts, leaf_size=4)
  neighbor_ids_array = tree.query_radius(np_pts, 4.0)
  eigenvalues_array = []
  for i, neighbor_ids in enumerate(neighbor_ids_array):
    if neighbor_ids.size < 10:
      eigenvalues = [0.0, 0.0, 0.0]
    else:
      cov_matrix = get_cov_matrix(np_pts[i], np_pts[neighbor_ids], tree)
      eigenvalues = get_eigenvalues(cov_matrix)
    eigenvalues_array.append(eigenvalues)
  # threshold = np.median(np.array(eigenvalues_array), axis=0)[2]
  threshold = 0.000001
  
  key_ids = []
  for i, vals in enumerate(eigenvalues_array):
    if is_key_point(vals, threshold):
      key_ids.append(i)
  
  filtered_key_ids = nms(key_ids, np.array([vals[2] for vals in eigenvalues_array]), np_pts)
  return filtered_key_ids


def nms(key_ids, lamda3_array, np_pts):
  lamda3s = lamda3_array[key_ids]
  pts = np_pts[key_ids]
  key_ids = np.array(key_ids)
  
  keep = [] # nms后剩余的对象
  index = lamda3s.argsort()[::-1]
  while index.size > 0:
    i = index[0]
    keep.append(i)
    diffs = np_pts[list(key_ids[index[1:]])] - np_pts[key_ids[i]]
    dists = np.array([norm(diff) for diff in diffs])
    thresh = 10.0
    idx = np.where(dists >= thresh)[0]
    index = index[idx+1]
  return key_ids[keep]


def main():
  files = process_pcd_files("../modelnet40_normal_resampled")
  for file in files:
    print(file)
    np_pts = read_pcd_from_file(file)
    key_ids = iss(np_pts)
    point_cloud_show(np_pts, np_pts[key_ids])


if __name__ == "__main__":
  # w = np.array([[1.0, 2.0, 3.0, 4.0]])
  # a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
  # a = w.T * a
  # print(a)
  # print(np.cov(a, rowvar=0) / (1 + 4.0 + 9.0 + 16.0))

  faulthandler.enable()
  main()

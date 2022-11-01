import libPCLKeypoint as libpcl
import numpy as np
import math
import os
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
import struct
import faulthandler
from scipy.spatial.transform import Rotation
from iss import iss
from evaluate_rt import read_reg_results, read_oxford_bin, reg_result_row_to_array


def show_np_pts3d(np_pts):
  pc_view = o3d.geometry.PointCloud()
  pc_view.points = o3d.utility.Vector3dVector(np_pts)
  o3d.visualization.draw_geometries([pc_view])


def point_cloud_show(point_cloud, feature_points):
  fig = plt.figure(dpi=150)
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], cmap='spectral', s=2, linewidths=0, alpha=1, marker=".")
  ax.scatter(feature_points[:, 0], feature_points[:, 1], feature_points[:, 2], cmap='spectral', s=2, linewidths=5, alpha=1, marker=".", color='red')
  plt.title('Point Cloud')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.show()


def dist_between_descriptor(descriptor1, descriptor2):
  assert(len(descriptor1) == len(descriptor2))
  dist = 0.0
  for i in range(len(descriptor1)):
    dist += math.fabs(descriptor1[i] - descriptor2[i])
  dist /= len(descriptor1)
  return dist


def dist_between_descriptor(descriptor1, descriptor2):
  assert(len(descriptor1) == len(descriptor2))
  assert(len(descriptor1) == 352)
  dist = 0.0
  for i in range(len(descriptor1)):
    dist += math.fabs(descriptor1[i] - descriptor2[i])
  dist /= len(descriptor1)
  return dist


def find_nn_dep(ref_dep, deps):
  min_dist = 10e9
  min_id = -1
  for i in range(deps.shape[0]):
    dep = deps[i]
    dist = dist_between_descriptor(ref_dep, dep)
    if dist < min_dist:
      min_dist = dist
      min_id = i
  assert(min_id >= 0 and min_id < deps.shape[0])
  return min_id


def build_correspondence(descriptors_1, descriptors_2, key_pts_1, key_pts_2):
  index = np.where(np.isnan(descriptors_1[:, 0]))[0]
  descriptors_1 = np.delete(descriptors_1, index, axis=0)
  key_pts_1 = np.delete(key_pts_1, index, axis=0)
  index = np.where(np.isnan(descriptors_2[:, 0]))[0]
  descriptors_2 = np.delete(descriptors_2, index, axis=0)
  key_pts_2 = np.delete(key_pts_2, index, axis=0)
  # # build tree
  # tree_1 = KDTree(descriptors_1, leaf_size=4)
  # tree_2 = KDTree(descriptors_2, leaf_size=4)
  # # find nn description for each other
  # match_2_in_1 = tree_1.query(descriptors_2, k=1, return_distance=False)
  # match_1_in_2 = tree_2.query(descriptors_1, k=1, return_distance=False)
  # # build correspondence
  # pairs = []
  # for i in range(descriptors_1.shape[0]):
  #   match_in_2 = match_1_in_2[i][0]
  #   if match_2_in_1[match_in_2][0] == i:
  #     pairs.append([i, match_in_2])

  pairs = []
  for i in range(descriptors_1.shape[0]):
    nnid_of_1_in_2 = find_nn_dep(descriptors_1[i], descriptors_2)
    dep_2 = descriptors_2[nnid_of_1_in_2]
    nnid_of_2_in_1 = find_nn_dep(dep_2, descriptors_1)
    if nnid_of_2_in_1 == i:
      pairs.append([i, nnid_of_1_in_2])

  return pairs, key_pts_1, key_pts_2


def normalize_pts(np_pts):
  center = np.expand_dims(np.mean(np_pts, axis=0), 0)
  np_pts = np_pts - center
  return np_pts


def procrustes(pts_1, pts_2):
  pts_1_nor = normalize_pts(pts_1)
  pts_2_nor = normalize_pts(pts_2)
  u, s, vT = np.linalg.svd(np.matmul(pts_2_nor.T, pts_1_nor), full_matrices=1, compute_uv=1)
  R = np.matmul(u, vT)
  t = np.expand_dims(np.mean(pts_2 - np.matmul(R, pts_1.T).T, axis=0), 0)
  return R, t


def solve_rt_with_ransac(key_pts_1, key_pts_2, pairs):
  matched_key_pts_1 = key_pts_1[[p[0] for p in pairs]]
  matched_key_pts_2 = key_pts_2[[p[1] for p in pairs]]
  assert(matched_key_pts_1.shape[0] == matched_key_pts_2.shape[0])
  N = 70
  outlier_p = 0.1
  tau = 10.0
  max_inlier_vote = -1
  rt_final = []
  for step in range(N):
    # selected 3 pairs of points randomly
    selected_pairs = [np.random.randint(0, len(pairs)) for _ in range(3)]
    pts_1 = key_pts_1[[pairs[i][0] for i in selected_pairs]]
    pts_2 = key_pts_2[[pairs[i][1] for i in selected_pairs]]
    # cal r, t
    R, t = procrustes(pts_1, pts_2)
    # vote
    diffs = matched_key_pts_2 - (np.matmul(R, matched_key_pts_1.T).T + t)
    vote = 0
    for i in range(diffs.shape[0]):
      diff = diffs[i]
      if np.linalg.norm(diff) < tau:
        vote += 1
    # record the best vote
    if vote > max_inlier_vote:
      rt_final = [R, t]
      max_inlier_vote = vote
    # stop earlier if reach expectation
    if max_inlier_vote > (1 - outlier_p) * len(pairs):
      # print("stop at[" + str(step) + "].")
      break
  # print("max vote: ", max_inlier_vote)
  return rt_final[0], rt_final[1]


def init_rt(np_pts_1, np_pts_2):
  # extract key point

  # key_pts_1 = libpcl.keypointIss(np_pts_1, iss_salient_radius=3.0, iss_gamma_21=0.5, iss_gamma_32=0.5, \
  #                                iss_non_max_radius=5.0, threads=8)
  # key_pts_2 = libpcl.keypointIss(np_pts_2, iss_salient_radius=3.0, iss_gamma_21=0.5, iss_gamma_32=0.5, \
  #                                iss_non_max_radius=5.0, threads=8)

  # key_pts_1 = np_pts_1[iss(np_pts_1)]
  # key_pts_2 = np_pts_2[iss(np_pts_2)]

  key_pts_1 = libpcl.keypointSift(np_pts_1, min_contrast=0.09)
  key_pts_2 = libpcl.keypointSift(np_pts_2, min_contrast=0.09)
  # vis for debug
  # point_cloud_show(np_pts_1, key_pts_1)
  # point_cloud_show(np_pts_2, key_pts_2)
  # extract description
  descriptors_1 = libpcl.featureSHOT352(np_pts_1, key_pts_1, feature_radius=5.0)
  descriptors_2 = libpcl.featureSHOT352(np_pts_2, key_pts_2, feature_radius=5.0)
  # build correspondence
  pairs, key_pts_1, key_pts_2 = build_correspondence(descriptors_1, descriptors_2, key_pts_1, key_pts_2)
  # for i in range(len(pairs)):
  #   print('1')
  #   point_cloud_show(np_pts_1, key_pts_1[[pairs[i][0]]])
  #   print('2')
  #   point_cloud_show(np_pts_2, key_pts_2[[pairs[i][1]]])
  assert(len(pairs) > 3)
  # print("Num of key points pairs[" + str(len(pairs)) + "].")
  R, t = solve_rt_with_ransac(key_pts_1, key_pts_2, pairs)
  # print("Q init matrix:")
  # print(Rotation.from_matrix(R).as_quat())
  # print("t init matrix:")
  # print(t)
  return R, t


def cal_diff(np_pts_1, np_pts_2):
  assert(np_pts_1.shape[0] == np_pts_2.shape[0])
  diffs = np_pts_1 - np_pts_2
  sum_diff = 0.0
  for i in range(diffs.shape[0]):
    sum_diff += np.linalg.norm(diffs[i])
  return sum_diff / np_pts_1.shape[0]


def visualize_pc_pair(src_np, dst_np):
  pcd_src = o3d.geometry.PointCloud()
  pcd_src.points = o3d.utility.Vector3dVector(np.transpose(src_np))
  pcd_src.paint_uniform_color([1, 0, 0])

  pcd_dst = o3d.geometry.PointCloud()
  pcd_dst.points = o3d.utility.Vector3dVector(np.transpose(dst_np))
  pcd_dst.paint_uniform_color([0, 1, 0])

  o3d.visualization.draw_geometries([pcd_src, pcd_dst])


def icp(np_pts_1, np_pts_2, R_init, t_init):
  R_final = np.identity(3)
  t_final = np.array([[0.0, 0.0, 0.0]])
  R, t = R_init, t_init
  last_diff = 10e9
  for step in range(100):
    # update data
    np_pts_1 = np.matmul(R, np_pts_1.T).T + t
    R_final = np.matmul(R, R_final)
    t_final += t
    diff = cal_diff(np_pts_1, np_pts_2)
    if diff < 0.01:
      break
    if step % 20 == 1:
      # visualize_pc_pair(np_pts_1.T, np_pts_2.T)
      # print ("---------------------" + str(step) + " loop----------------------")
      # print("R matrix:")
      # print(R_final)
      # print("t matrix:")
      # print(t_final)
      # print("diff:")
      # print(diff)
      pass
    last_diff = diff
    # build correspondence
    tree = KDTree(np_pts_2, leaf_size=4)
    nn_id_array = tree.query(np_pts_1, k=1, return_distance=False)
    pairs = []
    for i in range(np_pts_1.shape[0]):
      nn_id = nn_id_array[i][0]
      pt_1 = np_pts_1[i]
      pt_2 = np_pts_2[nn_id]
      if np.linalg.norm(pt_1 - pt_2) < 100.0:
        pairs.append([i, nn_id])
    assert(len(pairs) > 3)
    # solve r, t
    R, t = procrustes(np_pts_1[[p[0] for p in pairs]], np_pts_2[[p[1] for p in pairs]])
  return R_final, t_final


if __name__ == "__main__":
  faulthandler.enable()
  np.random.seed(1)
  # read points
  dataset_path = '/home/zhangj0o/Downloads/3D-Point-Cloud-Analytics/workspace/data/registration_dataset'
  your_reg_result_path = os.path.join(dataset_path, 'reg_result.txt')
  result_file_path = './result.txt'
  with open(result_file_path, 'w') as f:
      f.write("idx1,idx2,t_x,t_y,t_z,q_w,q_x,q_y,q_z\r\n")

  reg_list = read_reg_results(os.path.join(dataset_path, 'reg_result.txt'), splitter=',')

  for row in range(1, len(reg_list)):
    idx1, idx2, t, rot = reg_result_row_to_array(reg_list[row])
    np_pts_1 = (read_oxford_bin(os.path.join(dataset_path, 'point_clouds', '%d.bin' % idx2))[0:3, :]).T
    np_pts_2 = (read_oxford_bin(os.path.join(dataset_path, 'point_clouds', '%d.bin' % idx1))[0:3, :]).T
    # visualize_pc_pair(np_pts_1.T, np_pts_2.T)

    # init R, t
    R_init, t_init = init_rt(np_pts_1, np_pts_2)
    # R_init = np.identity(3)
    # t_init = np.array([[0.0, 0.0, 0.0]])
    R, t = icp(np_pts_1, np_pts_2, R_init, t_init)
    np_pts_1 = np.matmul(R, np_pts_1.T).T + t
    quat = Rotation.from_matrix(R).as_quat()

    record_str = str(idx1) + "," + str(idx2) + "," + str(t[0][0]) + ',' + str(t[0][1]) + "," + str(t[0][2]) + \
                 "," + str(quat[3]) + "," + str(quat[0]) + "," + str(quat[1]) + "," + str(quat[2])
    # print(record_str)
    print("Processing[" + str(row) + "] of " + str(len(reg_list)) + " files.")
    with open(result_file_path, 'a') as f:
      f.write(record_str + '\r\n')
    # visualize_pc_pair(np_pts_1.T, np_pts_2.T)

  

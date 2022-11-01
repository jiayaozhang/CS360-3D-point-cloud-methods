# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类
import open3d as o3d
import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import time
# from voxel_filter import voxel_filter
# from KMeans import K_Means
# from Spectral import Spectral
import cv2


def PCA(data, correlation=False, sort=True):
    # 屏蔽开始
    # remove mean
    mean = np.mean(data, axis=0)
    mean_removed = data - mean
    # get the cov matrix of sample
    cov_matrix = np.cov(mean_removed, rowvar=0)
    # cal eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(np.mat(cov_matrix))
    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors, mean_removed


def angle_of_two_vectors(v1, v2):
    costheta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return math.acos(costheta) / math.pi * 180


def show_np_pts3d(np_pts):
    pc_view = o3d.geometry.PointCloud()
    pc_view.points = o3d.utility.Vector3dVector(np_pts)
    o3d.visualization.draw_geometries([pc_view])

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    np.random.seed(1)
    outlier_p = 0.7
    N = 70
    iter_num = 0
    tau = 0.05
    nor_vec_final = None
    pt_final = None
    max_inlier_vote = -1
    while iter_num < N:
        # 取三个随机点
        idxs = [np.random.randint(0, data.shape[0]) for _ in range(3)]
        pts = data[idxs]
        # 计算平面法向量
        p0p1 = pts[1] - pts[0]
        p0p2 = pts[2] - pts[0]
        # 三点共线
        diff = p0p2 - (p0p2[0] / p0p1[0]) * p0p1
        if np.linalg.norm(diff) < 0.001:
            continue
        nor_vec = np.cross(p0p2, p0p1)
        norm = np.linalg.norm(nor_vec)
        nor_vec /= norm
        # 根据点到平面的距离投票
        inlier_vote = 0
        for i in range(data.shape[0]):
            pt = data[i]
            dist = math.fabs(np.matmul(nor_vec, (pt - pts[0]).T) / np.linalg.norm(nor_vec))
            if dist < tau:
                inlier_vote += 1
        if inlier_vote > max_inlier_vote:
            nor_vec_final = nor_vec
            max_inlier_vote = inlier_vote
            pt_final = pts[0]
        if max_inlier_vote > (1 - outlier_p) * data.shape[0]:
            # print("stop at[" + str(iter_num) + "].")
            break
        iter_num += 1
    
    left_idx = []
    rm_idx = []
    for i in range(data.shape[0]):
        pt = data[i]
        dist = math.fabs(np.matmul(nor_vec_final, (pt - pt_final).T) / np.linalg.norm(nor_vec_final))
        if dist > 0.3 and dist < 2.0:
            left_idx.append(i)
        else:
            rm_idx.append(i)
    segmengted_cloud = data[left_idx]
    rm_cloud = data[rm_idx]
    # 屏蔽结束
    # show_np_pts3d(rm_cloud)
    # show_np_pts3d(segmengted_cloud)
    
    # print('origin data points num:', data.shape[0])
    # print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud, [nor_vec_final, pt_final]


def cluster_in_image(data):
    # 获取图像的高和宽
    x_min = float("inf")
    x_max = -float("inf")
    y_min = float("inf")
    y_max = -float("inf")
    for i in range(data.shape[0]):
        pt = data[i]
        x_min = min(x_min, pt[0, 0])
        x_max = max(x_max, pt[0, 0])
        y_min = min(y_min, pt[0, 1])
        y_max = max(y_max, pt[0, 1])
    width = math.ceil((y_max - y_min) / 0.2)
    height = math.ceil((x_max - x_min) / 0.2)
    # 点云转图像
    mat = np.zeros((height, width, 1), np.uint8)
    for i in range(data.shape[0]):
        pt = data[i]
        x = pt[0, 0]
        y = pt[0, 1]
        p_x = math.floor((x - x_min) / 0.2)
        p_y = math.floor((y - y_min) / 0.2)
        mat[p_x, p_y, 0] = 255
    # cv2.imwrite("pointclouds.png", mat)
    kernel = np.ones((2, 2), np.uint8)
    mat = cv2.dilate(mat, kernel)
    h = cv2.findContours(mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 提取轮廓
    contours = h[0]
    # background = np.zeros((mat.shape[0], mat.shape[1], 1), np.uint8)
    # cv2.drawContours(background, contours, -1, 255, 1)
    # cv2.imwrite("contours.png", background)
    # cv2.imshow("demo1", background)
    # cv2.waitKey(0)
    clusters_index = []
    all_kinds = []
    for i in range(data.shape[0]):
        pt = data[i]
        x = pt[0, 0]
        y = pt[0, 1]
        p_x = math.floor((x - x_min) / 0.2)
        p_y = math.floor((y - y_min) / 0.2)
        clas = 0
        final_clas = 0
        max_dist = -float('inf')
        find = False
        for contour in contours:
            # hull = cv2.convexHull(contour, returnPoints = False)
            dist = cv2.pointPolygonTest(contour, (p_y, p_x), True)
            if dist > 0.0:
                find = True
                clusters_index.append(clas)
                break
            elif dist > max_dist:
                max_dist = dist
                final_clas = clas
            clas += 1
        if not find:
            # print("pt is classifered to " + str(final_clas))
            clusters_index.append(final_clas)
        else:
            # print("pt is in " + str(clas))
            pass
        if clusters_index[-1] not in all_kinds:
            all_kinds.append(clusters_index[-1])
    # print(len(contours))
    # print(len(all_kinds))
    return clusters_index
        

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始
    w, v, mean_removed = PCA(data)
    point_cloud_vector = v[:, :2]  # 主方向向量
    low_dim_data = mean_removed * point_cloud_vector
    clusters_index = cluster_in_image(low_dim_data)

    # vis = np.zeros(shape=[low_dim_data.shape[0], 3])
    # vis[:, [1, 2]] = low_dim_data
    # show_np_pts3d(vis)

    # kmeans = K_Means(n_clusters=20)
    # kmeans.fit(low_dim_data)
    # clusters_index = kmeans.predict(low_dim_data)

    # spectral = cluster.SpectralClustering(eigen_solver='arpack', affinity="nearest_neighbors")
    # spectral.fit(low_dim_data)
    # clusters_index = algorithm.labels_.astype(np.int)
    # 屏蔽结束

    return clusters_index, point_cloud_vector

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def main():
    root_dir = '/home/zhangj0o/Downloads/3D-Point-Cloud-Analytics/workspace/data/kitti-3d-object-detection/training/velodyne' # 数据集路径
    cat = os.listdir(root_dir)
    # cat = cat[1:]
    iteration_num = len(cat)
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        # print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        segmented_points = ground_segmentation(data=origin_points)
        time_start = time.time()
        cluster_index, pca_vec = clustering(segmented_points)
        time_end = time.time()
        # print("time: " + str(time_end - time_start))

        plot_clusters(segmented_points, cluster_index)


if __name__ == '__main__':
    main()

import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getdata(d_path):
    tmpdata = np.loadtxt(d_path, delimiter=',',usecols=(0,1,2))
    return tmpdata


def iss(data,gamma21,gamma32,r,NMS_radius,k_num = 100):

    print("-"*10,"start to do iss",data.shape[0],'points','-'*10)
    
    leaf_size = 32
    tree = KDTree(data,leaf_size)
    eps = 0.0001
    radius_neighbor = tree.query_ball_point(data,r)
    key_pointset = []
    featvalue = []

    print("-"*10,"start to search keypoints",'-'*10)

    for index in range(len(radius_neighbor)):
        neighbor_list = radius_neighbor[index]
        
        neighbor_list.remove(index)
        if len(neighbor_list) == 0:
            continue
        # svd
        weightmatrix = np.linalg.norm(data[neighbor_list] - data[index],axis=1)
        weightmatrix[weightmatrix==0] = eps # 避免除0的情况出现
        weightmatrix = 1/weightmatrix
        tmp = (data[neighbor_list] - data[index])[:,:,np.newaxis]# N,3,1
        convmatrix = np.sum(weightmatrix[:,np.newaxis,np.newaxis] * (tmp@tmp.transpose(0,2,1)),axis=0)/np.sum(weightmatrix)
        s = np.linalg.svd(convmatrix,compute_uv=False)

        if s[1]/s[0] < gamma21 and s[2]/s[1] < gamma32:
            key_pointset.append(data[index])
            featvalue.append(s[2])

    print("search keypoints finished",key_pointset.__len__()," points")

    print("-"*10,"NMS to filter keypoints",'-'*10)

    respointset = []
    leaf_size = 10
    index_matrix = [i for i in range(len(key_pointset))]
    restree = KDTree(key_pointset,leaf_size)
    # NMS STEP
    for iteration in range(k_num):
        max_index = featvalue.index(max(featvalue))
        tmp_point = key_pointset[max_index]
        del_index = restree.query_ball_point(tmp_point,NMS_radius)
        for d_index in del_index:
            if d_index in index_matrix:
                del featvalue[index_matrix.index(d_index)]
                del key_pointset[index_matrix.index(d_index)]
                del index_matrix[index_matrix.index(d_index)]
        respointset.append(tmp_point)
        if len(key_pointset) == 0:
            break
    
    print("NMS finished,find ",len(respointset)," points")

    return np.array(respointset)

def main():

    pts = getdata('./laptop_0168.txt')
    keypoint = iss(pts,0.6,0.6,0.1,0.15)

    pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
    key_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(keypoint))

    key_view.paint_uniform_color([1, 1, 0])
    pc_view.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([key_view,pc_view])

if __name__ == "__main__":
    main()
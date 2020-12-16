import open3d as o3d
import numpy as np
import copy
import math
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns

# get color from http://colormind.io/
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

sns.set(font='Franklin Gothic Book',
        rc={
    'axes.axisbelow': False,
    'axes.edgecolor': 'lightgrey',
    'axes.facecolor': 'None',
    'axes.grid': False,
    'axes.labelcolor': 'dimgrey',
    'axes.spines.right': False,
    'axes.spines.top': False,
    'figure.facecolor': 'white',
    'lines.solid_capstyle': 'round',
    'patch.edgecolor': 'w',
    'patch.force_edgecolor': True,
    'text.color': 'dimgrey',
    'xtick.bottom': False,
    'xtick.color': 'dimgrey',
    'xtick.direction': 'out',
    'xtick.top': False,
    'ytick.color': 'dimgrey',
    'ytick.direction': 'out',
    'ytick.left': False,
    'ytick.right': False})
sns.set_context("notebook", rc={"font.size":16,
                                "axes.titlesize":20,
                                "axes.labelsize":18})


ModelNet_category = ["airplane","bathtub","bed","bench","bookshelf","bottle","bowl","car","chair","cone","cup","curtain","desk","door","dresser","flower_pot","glass_box","guitar","keyboard","lamp","laptop","mantel","monitor","night_stand","person","piano","plant","radio","range_hood","sink","sofa","stairs","stool","table","tent","toilet","tv_stand","vase","wardrobe","xbox"]

def generatePcd(test_group, object_num, complete):
    category_arrs = []
    for _ in range(0, object_num):
        category_arrs.append(np.random.randint(len(ModelNet_category), size=test_group))

    pcd_arrs = []
    for category_arr in category_arrs:
        pcd_arr = []
        for index in range(0,test_group):
            filename = random.choice(os.listdir("pcd_data/"+str(ModelNet_category[category_arr[index]])+"/"))
            filename = "pcd_data/"+str(ModelNet_category[category_arr[index]])+"/" + filename
            pcd = o3d.io.read_point_cloud(filename)
            if(not complete):
                pcd_points = np.asarray(pcd.points)
                random_indexes = np.random.randint(pcd_points.shape[0], size = 400//object_num)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcd_points[random_indexes])
            pcd_arr.append(pcd)
        pcd_arrs.append(pcd_arr)
    return pcd_arrs

def transformPcd(pcd, t_and_r):
    transform_matrix = np.eye(4)
    transform_matrix[:3,:3] = pcd.get_rotation_matrix_from_xyz((t_and_r[3],t_and_r[4],t_and_r[5]))
    transform_matrix[0,3] = t_and_r[0]
    transform_matrix[1,3] = t_and_r[1]
    transform_matrix[2,3] = t_and_r[2]
    transformed_pcd = copy.deepcopy(pcd).transform(transform_matrix)
    return transformed_pcd, transform_matrix

def generateTestCase(object_num, test_group, rot_file_name, pcd_arrs):
    os.makedirs("./test_cases/",exist_ok=True)
    rot_arr = np.loadtxt(open(rot_file_name, "rb"), delimiter=",")

    original_pcd_transform_index_arr = []
    transformed_pcd_transform_index_arr = []
    for _ in range(0, object_num):
        original_pcd_transform_index_arr.append(np.random.randint(rot_arr.shape[0], size=test_group))
        transformed_pcd_transform_index_arr.append(np.random.randint(rot_arr.shape[0], size=test_group))
   
    
    for index_test in range(0,test_group):
        pcd_original_xyz = []
        pcd_transformed_xyz = []
        for index_object in range(0, object_num):
            original_pcd_transform_index = original_pcd_transform_index_arr[index_object]
            transformed_pcd_transform_index  = transformed_pcd_transform_index_arr[index_object]


            # 1*6 : "Trans_x","Trans_y","Trans_z","Rot_x(deg)","Rot_y(deg)","Rot_z(deg)"
            t_and_r_pcd_original = rot_arr[original_pcd_transform_index[index_test]] 
            pcd_original, _ = transformPcd(pcd_arrs[index_object][index_test], t_and_r_pcd_original)
            pcd_original_xyz.append(np.asarray(pcd_original.points))

            t_and_r_pcd_transformed = rot_arr[transformed_pcd_transform_index[index_test]] 
            pcd_transformed, _ = transformPcd(pcd_original, t_and_r_pcd_transformed)
            pcd_transformed_xyz.append(np.asarray(pcd_transformed.points))

        np.savez("./test_cases/test_case_"+str(index_test)+".npz", pcd_original=pcd_original_xyz, 
                pcd_transformed=pcd_transformed_xyz)

def plotPcdSideBySide(testcase):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    pcd_original = testcase["pcd_original"]
    pcd_transformed = testcase["pcd_transformed"]
 
    for index in range(0,len(pcd_original)):
        ax.scatter(pcd_original[index][:,0], pcd_original[index][:,1], pcd_original[index][:,2], marker='o', c=color_list[index])
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    for index in range(0,len(pcd_transformed)):
        ax.scatter(pcd_transformed[index][:,0], pcd_transformed[index][:,1], pcd_transformed[index][:,2], marker='o', c=color_list[index])
 

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def add_gaussian_noise(test_case ,noise_level,test_number):
    pcd_original = test_case["pcd_original"]
    pcd_transformed = test_case["pcd_transformed"]

    for index in range(0,len(pcd_original)):
        random_indexes_original = np.random.randint(pcd_original[index].shape[0], size = pcd_original[index].shape[0]//8)
        for row in random_indexes_original:
            pcd_original[index][row][0] += np.random.normal(0,noise_level)
            pcd_original[index][row][1] += np.random.normal(0,noise_level)
            pcd_original[index][row][2] += np.random.normal(0,noise_level)
        random_indexes_transformed = np.random.randint(pcd_transformed[index].shape[0], size = pcd_transformed[index].shape[0]//8)
        for row in random_indexes_transformed:
            pcd_transformed[index][row][0] += np.random.normal(0,noise_level)
            pcd_transformed[index][row][1] += np.random.normal(0,noise_level)
            pcd_transformed[index][row][2] += np.random.normal(0,noise_level)
    
    np.savez("./test_cases/test_case_"+str(test_number)+"_noise.npz", pcd_original=pcd_original, pcd_transformed=pcd_transformed)

def split_pcd(test_group, split_number, rot_file_name):
    # pcd_arr = generatePcd(test_group, 1, True)
    pcd = o3d.io.read_point_cloud("./pcd_data/chair/data_0000.ply")
    pcd_arr = [[pcd]]
    for index in range(0,test_group):
        pcd_split_arr = []
        pcd = pcd_arr[index][0]
        pcd_points = np.asarray(pcd.points)
        pcd_points_copy= copy.deepcopy(pcd_points)

        x_min = np.min(pcd_points_copy[:,0])
        x_max = np.max(pcd_points_copy[:,0])
        for index_split in range(1,split_number+1):
            x_upper_limit = x_min + (x_max - x_min+1)/split_number*index_split
            x_lower_limit = x_min + (x_max - x_min+1)/split_number*(index_split-1)
            pcd_split_arr.append(pcd_points_copy[np.where((pcd_points_copy[:,0] < x_upper_limit) * (pcd_points_copy[:,0] >= x_lower_limit))])
        pcd_original = [pcd_points]

        rot_arr = np.loadtxt(open(rot_file_name, "rb"), delimiter=",")
        split_tranform_index = np.random.randint(rot_arr.shape[0], size=len(pcd_split_arr))
        pcd_split = []
        
        for pcd_split_index in range(0,len(pcd_split_arr)):
            t_and_r = rot_arr[split_tranform_index[pcd_split_index]] 
            pcd = o3d.geometry.PointCloud()
            pcd_split_arr[pcd_split_index]
            pcd.points = o3d.utility.Vector3dVector(pcd_split_arr[pcd_split_index])
            pcd_transformed, _ = transformPcd(pcd, t_and_r)
            pcd_split.append(np.asarray(pcd_transformed.points))
            
        np.savez("./test_cases/split_pcd_"+str(index)+".npz", pcd_original=pcd_original, 
                    pcd_transformed=pcd_split) 

if __name__ == "__main__": 
    # test_group = 1
    # object_number = 2
    # pcd_arrs = generatePcd(test_group,object_number,False)
    # pcd_arrs = generatePcd(test_group,object_number,True)
    
    # generateTestCase(object_number, test_group,"./transform_matrix/trans_rot.csv",pcd_arrs)

    # Add noise
    # test_case_0 = np.load("./test_cases/test_case_0.npz")
    # test_case_1 = np.load("./test_cases/test_case_1.npz")
    # add_gaussian_noise(test_case_0 ,0.01,0)
    # add_gaussian_noise(test_case_1 ,0.03,1)
     
    # add_gaussian_noise(test_case_0 ,0.05,0)
    # add_gaussian_noise(test_case_1 ,0.05,1)

    # Some visualization
    # test_case_0_noise = np.load("./test_cases/test_case_0_noise.npz",allow_pickle=True)
    # test_case_1_noise = np.load("./test_cases/test_case_1_noise.npz",allow_pickle=True)
    # plotPcdSideBySide(test_case_0_noise)
    # plotPcdSideBySide(test_case_1_noise)


    # test_case_0 = np.load("./test_cases/test_case_0_.npz",allow_pickle=True)
    # test_case_1 = np.load("./test_cases/test_case_1.npz",allow_pickle=True)
    # plotPcdSideBySide(test_case_0)
    # plotPcdSideBySide(test_case_1)

    # Split application
    test_group = 1
    split_number = 5
    split_pcd(test_group, split_number, "./transform_matrix/trans_rot.csv")
    split_pcd_0 = np.load("./test_cases/split_pcd_0.npz",allow_pickle=True)
    plotPcdSideBySide(split_pcd_0)


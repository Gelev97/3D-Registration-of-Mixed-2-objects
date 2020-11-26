import open3d as o3d
import numpy as np
import copy
import math
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


ModelNet_category = ["airplane","bathtub","bed","bench","bookshelf","bottle","bowl","car","chair","cone","cup","curtain","desk","door","dresser","flower_pot","glass_box","guitar","keyboard","lamp","laptop","mantel","monitor","night_stand","person","piano","plant","radio","range_hood","sink","sofa","stairs","stool","table","tent","toilet","tv_stand","vase","wardrobe","xbox"]

''' 
generatePcd : randomly chose pcd to generate test cases
Input : 
    test_group -> the number of tests need to be created
    object_num -> the number of pcd need to be created
Output: 
    pcd_arrs -> randomly chosen pcd to generate testcase (object_num*test_group array)
'''
def generatePcd(test_group, object_num):
    category_arrs = []
    for _ in range(0, object_num):
        category_arrs.append(np.random.randint(len(ModelNet_category), size=test_group))

    pcd_arrs = []
    for category_arr in category_arrs:
        pcd_arr = []
        for index in range(0,test_group):
            filename = random.choice(os.listdir("pcd_data/"+str(ModelNet_category[category_arr[index]])+"/"))
            filename = "pcd_data/"+str(ModelNet_category[category_arr[index]])+"/" + filename
            pcd1 = o3d.io.read_point_cloud(filename)
            pcd_arr.append(pcd1)
        pcd_arrs.append(pcd_arr)
    return pcd_arrs

''' 
transformPcd : transform pcd with the given transform element including R and T
Input : 
    pcd -> the pcd need to be transform
    t_and_r -> 1*6 : "Trans_x","Trans_y","Trans_z","Rot_x(deg)","Rot_y(deg)","Rot_z(deg)"
Output: 
    transformed_pcd -> transformed pcd
    transform_matrix -> the tranform matrix consisting t_and_r's elements (4*4 transform matrix)
'''
def transformPcd(pcd, t_and_r):
    transform_matrix = np.eye(4)
    transform_matrix[:3,:3] = pcd.get_rotation_matrix_from_xyz((t_and_r[3],t_and_r[4],t_and_r[5]))
    transform_matrix[0,3] = t_and_r[0]
    transform_matrix[1,3] = t_and_r[1]
    transform_matrix[2,3] = t_and_r[2]
    transformed_pcd = copy.deepcopy(pcd).transform(transform_matrix)
    return transformed_pcd, transform_matrix

'''
generateTestCase : generate test case of original pcd points' locations 
                    and transformed pcd points' locations
Input : 
    object_num -> the number of pcd need to be created
    test_group -> the number of tests need to be created
    rot_file_name -> the transform's elements file need to be chosen
                    (3 different rotation's limits 45, 90,180)
    pcd_arrs -> randomly chosen pcd to generate testcase (object_num*test_group array)
Output: 
    test_cases written to the "./test_cases/" folder
'''
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

'''
plotPcdSideBySide : plot four pcd, the original pcd in one plot and the transformed pcd in one plot
Input : 
    test_case -> the test case which one wants to visualize
Output:
    modified test_cases written to the "./test_cases/" folder
'''
def plotPcdSideBySide(testcase):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    pcd_original = testcase["pcd_original"]
    pcd_transformed = testcase["pcd_transformed"]

    colors = cm.rainbow(np.linspace(0, 1, max(len(pcd_original),len(pcd_transformed))))
    for index in range(0,len(pcd_original)):
        ax.scatter(pcd_original[index][:,0], pcd_original[index][:,1], pcd_original[index][:,2], marker='o', c=colors[index])
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    for index in range(0,len(pcd_transformed)):
        ax.scatter(pcd_transformed[index][:,0], pcd_transformed[index][:,1], pcd_transformed[index][:,2], marker='o', c=colors[index])
 

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

'''
add_gaussian_noise : add gaussian noise to all the pcd with noise level
Input : 
    test_case -> the test case which one wants to visualize
    noise_level -> the noise level of the gaussian noise
    test_number -> the test case's number
Output:
    Showing a plot
'''
def add_gaussian_noise(test_case ,noise_level,test_number):
    pcd_original = test_case["pcd_original"]
    pcd_transformed = test_case["pcd_transformed"]

    for index in range(0,len(pcd_original)):
        for row in range(0,pcd_original[index].shape[0]):
            pcd_original[index][row][0] += np.random.normal(0,noise_level)
            pcd_original[index][row][1] += np.random.normal(0,noise_level)
            pcd_original[index][row][2] += np.random.normal(0,noise_level)
        for row in range(0,pcd_transformed[index].shape[0]):
            pcd_transformed[index][row][0] += np.random.normal(0,noise_level)
            pcd_transformed[index][row][1] += np.random.normal(0,noise_level)
            pcd_transformed[index][row][2] += np.random.normal(0,noise_level)
    
    np.savez("./test_cases/test_case_"+str(test_number)+".npz", pcd_original=pcd_original, pcd_transformed=pcd_transformed)

def split_pcd(test_group, split_number, rot_file_name):
    pcd_arr = generatePcd(test_group, 1)
    for index in range(0,test_group):
        pcd_split_arr = []
        pcd = pcd_arr[index][0]
        pcd_points = np.asarray(pcd.points)
        pcd_points_copy= copy.deepcopy(pcd_points)

        x_min = np.min(pcd_points_copy[:,0])
        x_max = np.max(pcd_points_copy[:,0])
        for index_split in range(1,split_number+1):
            x_upper_limit = x_min + (x_max - x_min)/split_number*index_split
            x_lower_limit = x_min + (x_max - x_min)/split_number*(index_split-1)
            pcd_split_arr.append(pcd_points_copy[np.where((pcd_points_copy[:,0] < x_upper_limit) * (pcd_points_copy[:,0] >= x_lower_limit))])
        pcd_original = [pcd_points]

        rot_arr = np.loadtxt(open(rot_file_name, "rb"), delimiter=",")
        split_tranform_index = np.random.randint(rot_arr.shape[0], size=len(pcd_split_arr))
        pcd_split = []
        
        for pcd_split_index in range(0,len(pcd_split_arr)):
            t_and_r = rot_arr[split_tranform_index[pcd_split_index]] 
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_split_arr[pcd_split_index])
            pcd_transformed, _ = transformPcd(pcd, t_and_r)
            pcd_split.append(np.asarray(pcd_transformed.points))

        np.savez("./test_cases/split_pcd_"+str(index)+".npz", pcd_original=pcd_original, 
                    pcd_transformed=pcd_split) 
# '''
# checkCorrectness : check the correctness of calculated transform,
# Input : 
#     test_case -> the test case which one wants to check
#     transform_1 -> the first object's calculated transform
#     transform_2 -> the second object's calculated transform
# Output:
#     Showing a plot
# '''
# def checkCorrectness(test_case, transform_1, transform_2):
#     pcd1_homo = np.hstack((test_case["pcd1_xyz"],np.ones((test_case["pcd1_xyz"].shape[0],1))))
#     transformed_pcd1_homo = np.matmul(transform_1,pcd1_homo.T)
#     transformed_pcd1 = transformed_pcd1_homo[:3,:]/(transformed_pcd1_homo[-1,:].reshape((1,transformed_pcd1_homo.shape[1])))
#     plotPcd(transformed_pcd1, test_case["transformed_pcd1_xyz"])

#     pcd2_homo = np.hstack((test_case["pcd2_xyz"],np.ones((test_case["pcd2_xyz"].shape[0],1))))
#     transformed_pcd2_homo = np.matmul(transform_2,pcd2_homo.T)
#     transformed_pcd2 = transformed_pcd2_homo[:3,:]/(transformed_pcd2_homo[-1,:].reshape((1,transformed_pcd2_homo.shape[1])))
#     plotPcd(transformed_pcd2, test_case["transformed_pcd2_xyz"])
    
if __name__ == "__main__": 
    test_group = 2
    object_number = 5
    pcd_arrs = generatePcd(test_group,object_number)

    generateTestCase(object_number, test_group,"./transform_matrix/rot_45.csv",pcd_arrs)

    # Add noise
    # test_case_0 = np.load("./test_cases/test_case_0.npz")
    # test_case_1 = np.load("./test_cases/test_case_1.npz")
    # add_gaussian_noise(test_case_0 ,0.03,0)
    # add_gaussian_noise(test_case_1 ,0.03,1)
     
    # add_gaussian_noise(test_case_0 ,0.05,0)
    # add_gaussian_noise(test_case_1 ,0.05,1)

    # Some visualization
    # test_case_0 = np.load("./test_cases/test_case_0.npz")
    # test_case_1 = np.load("./test_cases/test_case_1.npz")
    # plotPcdSideBySide(test_case_0)
    # plotPcdSideBySide(test_case_1)

    # Split application
    # test_group = 1
    # split_number = 5
    # split_pcd(test_group, split_number, "./transform_matrix/rot_45.csv")
    # split_pcd_0 = np.load("./test_cases/split_pcd_0.npz",allow_pickle=True)
    # plotPcdSideBySide(split_pcd_0)


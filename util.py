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
Output: 
    pcd1_arr -> pcd1 randomly chosen to generate testcase (1*test_group array)
    pcd2_arr -> pcd2 randomly chosen to generate testcase (1*test_group array)
'''
def generatePcd(test_group):
    category_arr_1 = np.random.randint(len(ModelNet_category), size=test_group)
    category_arr_2 = np.random.randint(len(ModelNet_category), size=test_group)

    pcd1_arr = []
    pcd2_arr = []
    for index in range(0,test_group):
        filename_1 = random.choice(os.listdir("pcd_data/"+str(ModelNet_category[category_arr_1[index]])+"/"))
        filename_1 = "pcd_data/"+str(ModelNet_category[category_arr_1[index]])+"/" + filename_1
        pcd1 = o3d.io.read_point_cloud(filename_1)
        pcd1_arr.append(pcd1)

        filename_2 = random.choice(os.listdir("pcd_data/"+str(ModelNet_category[category_arr_2[index]])+"/"))
        filename_2 = "pcd_data/"+str(ModelNet_category[category_arr_2[index]])+"/" + filename_2
        pcd2 = o3d.io.read_point_cloud(filename_2)
        pcd2_arr.append(pcd2)
    return pcd1_arr, pcd2_arr

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
    test_group -> the number of tests need to be created
    rot_file_name -> the transform's elements file need to be chosen
                    (3 different rotation's limits 45, 90,180)
    pcd1_arr -> pcd1 randomly chosen to generate testcase (1*test_group array)
    pcd2_arr -> pcd2 randomly chosen to generate testcase (1*test_group array)
Output: 
    test_cases written to the "./test_cases/" folder
'''
def generateTestCase(test_group, rot_file_name, pcd1_arr, pcd2_arr):
    os.makedirs("./test_cases/",exist_ok=True)
    rot_arr = np.loadtxt(open(rot_file_name, "rb"), delimiter=",")
    transform_index_pcd_1 = np.random.randint(rot_arr.shape[0], size=test_group)
    transform_index_pcd_2 = np.random.randint(rot_arr.shape[0], size=test_group)
    
    transform_index_pcd_transformed_1 = np.random.randint(rot_arr.shape[0], size=test_group)
    transform_index_pcd_transformed_2 = np.random.randint(rot_arr.shape[0], size=test_group)

    for index in range(0,test_group):
        # 1*6 : "Trans_x","Trans_y","Trans_z","Rot_x(deg)","Rot_y(deg)","Rot_z(deg)"
        t_and_r_pcd_1 = rot_arr[transform_index_pcd_1[index]] 
        pcd1, _ = transformPcd(pcd1_arr[index], t_and_r_pcd_1)
        pcd1_xyz = np.asarray(pcd1.points)

        t_and_r_pcd_2 = rot_arr[transform_index_pcd_2[index]] 
        pcd2, _ = transformPcd(pcd2_arr[index], t_and_r_pcd_2)
        pcd2_xyz = np.asarray(pcd2.points)

        t_and_r_pcd_transformed_1 = rot_arr[transform_index_pcd_transformed_1[index]] 
        transformed_pcd1, transform_matrix_1 = transformPcd(pcd1, t_and_r_pcd_transformed_1)
        transformed_pcd1_xyz = np.asarray(transformed_pcd1.points)

        t_and_r_pcd_transformed_2 = rot_arr[transform_index_pcd_transformed_2[index]] 
        transformed_pcd2, transform_matrix_2 = transformPcd(pcd2, t_and_r_pcd_transformed_2)
        transformed_pcd2_xyz = np.asarray(transformed_pcd2.points)

        np.savez("./test_cases/test_case_"+str(index)+".npz", pcd1_xyz=pcd1_xyz,pcd2_xyz=pcd2_xyz,
            transformed_pcd1_xyz=transformed_pcd1_xyz,transformed_pcd2_xyz=transformed_pcd2_xyz,
            transform_matrix_1=transform_matrix_1, 
            transform_matrix_2=transform_matrix_2)

'''
plotPcdSideBySide : plot four pcd, the original 2 in one plot and the transformed 2 in one plot
Input : 
    pcd_original -> [original pcd1 , original pcd 2]
    pcd_transformed -> [transformed pcd1 , transformed pcd 2]
Output:
    Showing a plot
'''
def plotPcdSideBySide(pcd_original, pcd_transformed):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.scatter(pcd_original[0][:,0], pcd_original[0][:,1], pcd_original[0][:,2], marker='o', c='r')
    ax.scatter(pcd_original[1][:,0], pcd_original[1][:,1], pcd_original[1][:,2], marker='o', c='b')
 
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(pcd_transformed[0][:,0], pcd_transformed[0][:,1], pcd_transformed[0][:,2], marker='o', c='r')
    ax.scatter(pcd_transformed[1][:,0], pcd_transformed[1][:,1], pcd_transformed[1][:,2], marker='o', c='b')
 

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

'''
plotPcdSideBySide : plot two pcd in one picture
Input : 
    pcd_1 -> one of the pcd tio plot
    pcd_2 -> the other pcd to plot
Output:
    Showing a plot
'''
def plotPcd(pcd_1, pcd_2):
    fig = plt.figure()
    plt3d = fig.gca(projection='3d')
    ax = plt.gca()

    ax.scatter(pcd_2[:,0], pcd_2[:,1], pcd_2[:,2], marker='o', c='r')
    ax.scatter(pcd_1[:,0], pcd_1[:,1], pcd_1[:,2], marker='o', c='b')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

'''
checkCorrectness : check the correctness of calculated transform,
Input : 
    test_case -> the test case which one wants to check
    transform_1 -> the first object's calculated transform
    transform_2 -> the second object's calculated transform
Output:
    Showing a plot
'''
def checkCorrectness(test_case, transform_1, transform_2):
    pcd1_homo = np.hstack((test_case["pcd1_xyz"],np.ones((test_case["pcd1_xyz"].shape[0],1))))
    transformed_pcd1_homo = np.matmul(transform_1,pcd1_homo.T)
    transformed_pcd1 = transformed_pcd1_homo[:3,:]/(transformed_pcd1_homo[-1,:].reshape((1,transformed_pcd1_homo.shape[1])))
    plotPcd(transformed_pcd1, test_case["transformed_pcd1_xyz"])

    pcd2_homo = np.hstack((test_case["pcd2_xyz"],np.ones((test_case["pcd2_xyz"].shape[0],1))))
    transformed_pcd2_homo = np.matmul(transform_2,pcd2_homo.T)
    transformed_pcd2 = transformed_pcd2_homo[:3,:]/(transformed_pcd2_homo[-1,:].reshape((1,transformed_pcd2_homo.shape[1])))
    plotPcd(transformed_pcd2, test_case["transformed_pcd2_xyz"])
    

if __name__ == "__main__": 
    test_group = 2
    pcd1_arr, pcd2_arr = generatePcd(test_group)

    generateTestCase(test_group,"./transform_matrix/rot_45.csv",pcd1_arr,pcd2_arr)
    
    test_case_0 = np.load("./test_cases/test_case_0.npz")
    
    # Some visualization
    o3d.visualization.draw_geometries([pcd1_arr[0], pcd2_arr[0]])
    plotPcdSideBySide([test_case_0["pcd1_xyz"],test_case_0["pcd2_xyz"]], 
        [test_case_0["transformed_pcd1_xyz"],test_case_0["transformed_pcd2_xyz"]])
    plotPcd(test_case_0["pcd1_xyz"],test_case_0["pcd2_xyz"])
    plotPcd(test_case_0["transformed_pcd1_xyz"],test_case_0["transformed_pcd2_xyz"])

    checkCorrectness(test_case_0, test_case_0["transform_matrix_1"], test_case_0["transform_matrix_2"])


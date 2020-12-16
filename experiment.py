import util
import numpy as np
import cluster
import register
import time

def testPerformance(file_name, object_number, test_group, rot_file_name, noise, complete, noise_level=0.01):
    pcd_arrs = util.generatePcd(test_group, object_number, complete)
    util.generateTestCase(object_number, test_group, rot_file_name, pcd_arrs)

    total_time_arr = []
    total_err_arr = []
    for index in range(0,test_group):
        print(index)
        test_case = np.load("./test_cases/test_case_" +str(index) + ".npz",allow_pickle=True)
        if(noise):
            util.add_gaussian_noise(test_case,noise_level,index)
            test_case_noise = np.load("./test_cases/test_case_" +str(index) + "_noise.npz",allow_pickle=True)
            pcd_original_arr = test_case_noise["pcd_original"]
            pcd_transformed_arr = test_case_noise["pcd_transformed"]
        else:
            pcd_original_arr = test_case["pcd_original"]
            pcd_transformed_arr = test_case["pcd_transformed"]
        
        orig = np.concatenate(pcd_original_arr, axis=0)
        transformed = np.concatenate(pcd_transformed_arr, axis=0)
        start_time = time.time()
        if(noise):
            cat, corres = cluster.find_cluster_noise(orig,transformed,object_number)
        else:
            cat, corres = cluster.find_cluster(orig,transformed)
        end_time = time.time()
        total_time_arr.append(end_time-start_time)
        
        err_arr = []
        for index_object in range(0, object_number):
            orig_homo = np.vstack(((orig[cat==index_object]).T,np.ones((1,(orig[cat==index_object]).shape[0]))))
            transformed_homo = np.vstack(((transformed[cat==index_object]).T,np.ones((1,(transformed[cat==index_object]).shape[0]))))

            mat = register.register(orig_homo, transformed_homo)
            if(noise):
                pcd_original_arr = test_case["pcd_original"]
                pcd_transformed_arr = test_case["pcd_transformed"]
                orig = np.concatenate(pcd_original_arr, axis=0)
                transformed = np.concatenate(pcd_transformed_arr, axis=0)        
                orig_homo = np.vstack(((orig[cat==index_object]).T,np.ones((1,(orig[cat==index_object]).shape[0]))))
                transformed_homo = np.vstack(((transformed[cat==index_object]).T,np.ones((1,(transformed[cat==index_object]).shape[0]))))
                err = register.errFunc(mat, orig_homo, transformed_homo).sum()
            else:
                err = register.errFunc(mat, orig_homo, transformed_homo).sum()
            print("Reprojection Error: {}".format(err))
            err_arr.append(err)

        total_err_arr.append(np.sum(np.array(err_arr))/len(err_arr))

if __name__ == "__main__": 
    object_number = 2
    test_group = 100
    rot_file_name = "./transform_matrix/trans_rot.csv"

    testPerformance("object2_test100_45",object_number, test_group, rot_file_name, False, True, noise_level=0.01)
    testPerformance("object2_test100_total400",object_number, test_group, rot_file_name, False, False, noise_level=0.01)

    testPerformance("object2_test100_total400_noise1",object_number, test_group, rot_file_name, True, False, noise_level=0.01)
    testPerformance("object2_test100_total400_noise2",object_number, test_group, rot_file_name, True, False, noise_level=0.02)
    testPerformance("object2_test100_total400_noise3",object_number, test_group, rot_file_name, True, False, noise_level=0.03)
    testPerformance("object2_test100_total400_noise4",object_number, test_group, rot_file_name, True, False, noise_level=0.04)
    testPerformance("object2_test100_total400_noise5",object_number, test_group, rot_file_name, True, False, noise_level=0.05)

    object_number = 3
    testPerformance("object3_test100_total400",object_number, test_group, rot_file_name, False, False, noise_level=0.01)

    object_number = 4
    testPerformance("object4_test100_total400",object_number, test_group, rot_file_name, False, False, noise_level=0.01)

    object_number = 5
    testPerformance("object5_test100_total400",object_number, test_group, rot_file_name, False, False, noise_level=0.01)

import open3d as o3d
import numpy as np
import os

ModelNet_category = ["airplane","bathtub","bed","bench","bookshelf","bottle","bowl","car","chair","cone","cup","curtain","desk","door","dresser","flower_pot","glass_box","guitar","keyboard","lamp","laptop","mantel","monitor","night_stand","person","piano","plant","radio","range_hood","sink","sofa","stairs","stool","table","tent","toilet","tv_stand","vase","wardrobe","xbox"]

'''
readH5 : read h5 file and change it to numpy array
Input : 
    filename -> h5's file name
Output: 
    numpy array of the h5's content
'''
def readH5(filename):
    import h5py
    file = h5py.File(filename,'r')
    return [np.array(file.get('data')),file.get('label')]

'''
writePly : write data into a .ply file with the given name
Input : 
    filename -> save .ply file' filename
    data -> data saved into the .ply file
Output: 
    a categorized saved .ply file in the pcd_data/{$category}
'''
def writePly(filename, data):
    with open(filename,'w') as file:
        header = 'ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nend_header\n'.format(data.shape[0], 0)
        file.write(header)
        for i in range(data.shape[0]):
            file.write(str(data[i,0])+' '+str(data[i,1])+' '+str(data[i,2])+'\n')

'''
convertH5ToPly : convert H5's data and label to a .ply file
Input : 
    data_arr -> H5's extracted data array 
    label_arr -> H5's extracted label array 
Output: 
    a categorized saved .ply file in the pcd_data/{$category}
'''
def convertH5ToPly(data_arr,label_arr):
    for index in range(0,len(data_arr)):
        data = data_arr[index]
        label = label_arr[index]
        dir_path = "./pcd_data/" + ModelNet_category[int(label)] + "/"
        os.makedirs(dir_path,exist_ok=True)
        files_num = len(next(os.walk(dir_path))[2])
        writePly(dir_path+'data_%04d.ply'%files_num,data)

'''
convert : convert H5 file to a categorized saved .ply file 
Input : 
    input_path -> h5's path
Output: 
    a categorized saved .ply file in the pcd_data/{$category}
'''
def convert(input_path):
    [data,label] = readH5(input_path)
    convertH5ToPly(data,label)

def main():
    convert("./h5_data/source_00.h5")
    convert("./h5_data/source_01.h5")
    convert("./h5_data/source_02.h5")
    convert("./h5_data/source_03.h5")
    convert("./h5_data/source_04.h5")

main()

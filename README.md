# 3D-Registration-of-Mixed-2-objects

16-811 Fall 2020 Term Project
- Ruohai Ge 
- Ruoyang Xu
- Zhihao Zhang

Raw pcd files are store in the h5_data folder
- run ht_to_ply.py to turn them in to pcd files(.ply format)
- They should be stored under pcd_data/ categorized.

util.py
- generate test cases need for this Project
- test cases are saved in .npz file with x,y,z locations of original and transformed PCD

experiment.py
- script to do different experiments

plot.py
- script to plot the experiment results

cluster.py
- the RFM method to cluster point cloud data

resgister.py
- SVM and RANSAC to get the transform matrix

transform_matrix/
- contains one csv files
- each row of these files contain 6 elements: "Trans_x","Trans_y","Trans_z","Rot_x(deg)","Rot_y(deg)","Rot_z(deg)"

results/
- contains some test cases in .npz

src/
- contains raw algorithm code, simple test cases

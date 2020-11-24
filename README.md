# 3D-Registration-of-Mixed-2-objects

16-811 Fall 2020 Term Project
- Ruohai Ge 
- Ruoyang Xu
- Zhihao Zhang

Raw pcd files are store in the h5_data folder
- run ht_to_ply.py to turn them in to pcd files(.ply format)
- They should be stored under pcd_data/ categorized.

util.ply
- generate test cases need for this Project

transform_matrix/
- contains three csv files
- each row of these files contain 6 elements: "Trans_x","Trans_y","Trans_z","Rot_x(deg)","Rot_y(deg)","Rot_z(deg)"
- rot_45.csv - rotation angle limit is 45 degress
- rot_90.csv - rotation angle limit is 90 degress
- rot_180.csv - rotation angle limit is 180 degress

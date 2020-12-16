import open3d as o3d
import numpy as np
import cluster
import register
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

split_pcd_0 = np.load("./test_cases/split_pcd_0.npz",allow_pickle=True)

orig = np.concatenate(split_pcd_0["pcd_original"], axis=0)
transformed = np.concatenate(split_pcd_0["pcd_transformed"], axis=0)
cat, corres = cluster.find_cluster(orig,transformed)

mat_arr = []
for index_object in range(0, 5):
    orig_homo = np.vstack(((orig[cat==index_object]).T,np.ones((1,(orig[cat==index_object]).shape[0]))))
    transformed_homo = np.vstack(((transformed[cat==index_object]).T,np.ones((1,(transformed[cat==index_object]).shape[0])))) 
    mat = register.register(transformed_homo, orig_homo)
    mat_arr.append(mat)

pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(transformed[cat==0])
pcd1.transform(mat_arr[0])

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(transformed[cat==1])
pcd2.transform(mat_arr[1])

pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(transformed[cat==2])
pcd3.transform(mat_arr[2])

pcd4 = o3d.geometry.PointCloud()
pcd4.points = o3d.utility.Vector3dVector(transformed[cat==3])
pcd4.transform(mat_arr[3])

pcd5 = o3d.geometry.PointCloud()
pcd5.points = o3d.utility.Vector3dVector(transformed[cat==4])
pcd5.transform(mat_arr[4])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pcd_arr = [pcd1, pcd2, pcd3, pcd4, pcd5]

colors = [[5.00000000e-01,0.00000000e+00,1.00000000e+00,1.00000000e+00],
 [1.96078431e-03,7.09281308e-01,9.23289106e-01,1.00000000e+00],
 [5.03921569e-01,9.99981027e-01,7.04925547e-01,1.00000000e+00],
 [1.00000000e+00,7.00543038e-01,3.78411050e-01,1.00000000e+00],
 [1.00000000e+00,1.22464680e-16,6.12323400e-17, 1.00000000e+00]]
for index in range(0,5):
    pcd = pcd_arr[index]
    points = np.asarray(pcd.points)
    ax.scatter(points[:,0], points[:,1], points[:,2], marker='o', c=colors[index])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
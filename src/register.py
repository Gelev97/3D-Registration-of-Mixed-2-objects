import numpy as np
import scipy.io

def tf(pts1, pts2, gmat):
    # Use DLT to solve
    assert(pts1.shape == pts2.shape)
    (r, c) = pts1.shape
    if (r == 4):
        pts1_nh = pts1[:-1, :]
    else:
        pts1_nh = pts1
    
    l1 = np.hstack((-pts1_nh.T, -np.ones((c, 1)), np.zeros((c, 8)), pts1_nh.T * (pts2[0, :].T).reshape((-1, 1)), (pts2[0, :].T).reshape((-1, 1))))
    l2 = np.hstack((np.zeros((c, 4)), -pts1_nh.T,  -np.ones((c, 1)), np.zeros((c, 4)), pts1_nh.T * (pts2[1, :].T).reshape((-1, 1)), (pts2[1, :].T).reshape((-1, 1))))
    l3 = np.hstack((np.zeros((c, 8)), -pts1_nh.T,  -np.ones((c, 1)), pts1_nh.T * (pts2[2, :].T).reshape((-1, 1)), (pts2[2, :].T).reshape(-1, 1)))
    A_mat = np.vstack((l1, l2, l3))
    
    true_err = np.dot(A_mat, gmat.flatten())
    print("True error {}".format((np.abs(true_err)).sum()))

    (u, s, vt) = np.linalg.svd(A_mat)
    mat = vt[-1, :].reshape((4, 4))
    return mat

if __name__ == "__main__":
    mat_dict = scipy.io.loadmat('example.mat')
    pts1 = mat_dict['pt1'].astype(float)
    pts2 = mat_dict['pt1_tf'].astype(float)
    gmat = mat_dict['gmat'].astype(float)

    mat = tf(pts1, pts2, gmat)
    mat = mat / mat[3, 3]
    print(mat)
    recovered = np.dot(mat, pts1)
    err = (np.abs(pts2 - recovered)).sum()
    print("Reprojection Error {}".format(err))
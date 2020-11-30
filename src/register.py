import numpy as np
import scipy.io
import scipy.optimize as optimize

def tf(pts1, pts2):
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
    
    # true_err = np.dot(A_mat, gmat.flatten())
    # print("True error {}".format((np.abs(true_err)).sum()))

    (u, s, vt) = np.linalg.svd(A_mat)
    mat = vt[-1, :].reshape((4, 4))
    return mat

def errFunc(gmat, pts, pts_orig):
    (r, c) = pts.shape
    if r == 3:
        pts = np.vstack((pts, np.ones((1, c))))
    
    tf_ed = np.dot(gmat, pts)
    tf_ed = tf_ed / tf_ed[3, :]
    err = np.abs(pts_orig - tf_ed)
    return err

def capErr(err, c):
    return np.where(err > c, c, err)

def ransac(pts1, pts2):
    # Given pts1 and pts2
    # Run ransac on returned value
    n = 3
    p = 0.8
    w = 0.5
    k = np.ceil(np.log(1 - p) / np.log(1 - w ** n)).astype(int)

    (r, c) = pts1.shape

    min_err = np.inf
    min_gmat = None

    for i in range(k):
        rand_idx = np.random.choice(c, n + 2, replace = False)
        this_pts1 = pts1[:, rand_idx]
        this_pts2 = pts2[:, rand_idx]
        gmat = tf(this_pts1, this_pts2)
        err = errFunc(gmat, pts1, pts2).sum()
        if err < min_err:
            min_gmat = gmat
            min_err = err
    
    return min_gmat

def rodrigues(r):
    # Replace pass by your implementation
    theta = np.linalg.norm(r)
    
    if theta == 0:
        return np.eye(3)

    u = r / theta
    u = u.reshape((-1, 1))
    (u1, u2, u3) = (u[0, 0], u[1, 0], u[2, 0])
    ucross = np.array([[0, -u3, u2], [u3, 0, -u1], [-u2, u1, 0]])
    return np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * u.dot(u.T) + ucross * np.sin(theta)

def invRodrigues(R):
    A = (R - R.T)/2
    rho = np.array([A[2, 1], A[0, 2], A[1, 0]]).reshape((-1, 1))
    s = np.linalg.norm(rho)
    c = (np.diagonal(R).sum() - 1) / 2
    if (s == 0) and (c == 1):
        return np.zeros((3,1))
    
    if (s == 0) and (c == -1):
        temp = R + np.eye(3)
        # Pick the first non zero column of R + I
        idx = 0
        val = 0
        while val <= 0:
            v = temp[:, idx]
            val = np.abs(temp).sum()
            idx += 1
        
        u = v / np.linalg.norm(v)
        sfunc_r = u * np.pi
        # No need to check for pi since it is already normalized
        # print(sfunc_r.shape)
        r1 = sfunc_r[0]
        r2 = sfunc_r[2]
        r3 = sfunc_r[3]
        if (r1 == 0 and r2 == 0 and r3 < 0) or (r1 == 0 and r2 < 0) or (r1 < 0):
            sfunc_r = -sfunc_r
        return sfunc_r

    if  (s != 0):
        u = rho / s
        theta = np.arctan2(s, c)
        r = u * theta
        r = np.reshape(r, (3, 1))
        return r

def costFunc(x, pts1, pts2, rigid=True):
    if rigid:
        rotMat = rodrigues(x[:3])
        t = x[3:]
        gmat = np.zeros((4, 4))
        gmat[3, 3] = 1
        gmat[:3, :3] = rotMat
        gmat[:3, 3] = t
    else:
        gmat = x.reshape((4, 4))

    res = errFunc(gmat, pts1, pts2).sum(axis = 0)
    return capErr(res, 50)

def register(pts1, pts2, rigid=True):
    # use ransac as initial step
    # truncated nonlinear least square for subsequent steps.
    (r, c) = pts1.shape
    if r == 3:
        pts1 = np.vstack((pts1, np.ones((1, c))))
        pts2 = np.vstack((pts2, np.ones((1, c))))

    func_sig = lambda x : (costFunc(x, pts1, pts2, rigid).flatten())
    gmat_init = ransac(pts1, pts2)
    gmat_init /= gmat_init[3, 3]
    print("Ransac result err {}".format(errFunc(gmat_init, pts1, pts2).sum()))

    if rigid:
        r = invRodrigues(gmat_init[:3, :3])
        t = gmat_init[:3, 3]
        g0 = np.vstack((r.reshape((-1, 1)), t.reshape((-1, 1)))).reshape((-1, ))
        # print("gmat init\n{}".format(gmat_init[:3, :3]))
        # print("r recon\n{}".format(rodrigues(g0[:3])))
        # print("r \n {}".format(g0[:3]))
        g_star = optimize.leastsq(func_sig, g0)[0]
        gmat = np.zeros((4, 4))
        gmat[3, 3] = 1
        gmat[:3, :3] = rodrigues(g_star[:3])
        gmat[:3, 3] = g_star[3:]
    else:
        g_star = optimize.leastsq(func_sig, gmat_init.flatten())[0]
        gmat = g_star.reshape((4, 4))
    return gmat

if __name__ == "__main__":
    mat_dict = scipy.io.loadmat('example.mat')
    pts1 = mat_dict['pt1'].astype(float)
    pts2 = mat_dict['pt1_tf'].astype(float)
    gmat = mat_dict['gmat'].astype(float)

    mat = tf(pts1, pts2)
    mat = mat / mat[3, 3]
    print(mat)
    recovered = np.dot(mat, pts1)
    err = (np.abs(pts2 - recovered)).sum()
    print("Reprojection Error {}".format(err))

    mat = register(pts1, pts2)
    # print(mat)
    print("Register err: {}".format(errFunc(mat, pts1, pts2).sum()))
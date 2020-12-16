import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from numpy.linalg import svd
from numpy.linalg import lstsq
from scipy.spatial.distance import cdist
from scipy.stats import ortho_group

def find_cluster(orig, transformed):
    a = np.expand_dims(orig, axis=1).repeat(orig.shape[0], axis=1)
    b = np.transpose(a, [1,0,2])
    feat1 = np.sqrt(np.sum((a-b)**2, axis=2))
    a = np.expand_dims(transformed, axis=1).repeat(transformed.shape[0], axis=1)
    b = np.transpose(a, [1,0,2])
    feat2 = np.sqrt(np.sum((a-b)**2, axis=2))
#     dm = cdist(feat1, feat2, lambda u, v: len(np.intersect1d(u, v)))
#     match = np.argmax(dm, axis=1)
    point_num = orig.shape[0]
    
    cat = np.ones(point_num)*-1
    
    corres = np.ones(point_num)*-1
    
    unmatched_point = set(np.arange(point_num))
    
    cat_id = 0
    
    while len(unmatched_point)>0:
        while True:
            ref_point_index = np.random.choice(np.array(list(unmatched_point)), 4, replace=False)
            dm = cdist(feat1[ref_point_index], feat2, lambda u, v: len(np.intersect1d(u, v)))
            match = np.argmax(dm, axis=1)
            m = feat1[ref_point_index, :][:, ref_point_index]
            n = feat2[match, :][:, match]
            if np.isclose(m,n).all():
                break
        ref_feat1 = feat1[ref_point_index, :].T  # shape (n,4)
        ref_feat2 = feat2[match, :].T  # shape (n,4)
        
        dist = cdist(ref_feat1, ref_feat2)
        
        in_class = np.min(dist, axis=1) < 1e-9
        matched_point = np.argmin(dist, axis=1)
        cat[in_class] = cat_id
        unmatched_point = unmatched_point - set(in_class.nonzero()[0])
        corres[in_class] = matched_point[in_class]
        print(len(unmatched_point))
        if(len(unmatched_point) < 4):
            break
        cat_id += 1
    
    return cat, corres

def find_cluster_noise(orig, transformed, object_num):
    a = np.expand_dims(orig, axis=1).repeat(orig.shape[0], axis=1)
    b = np.transpose(a, [1,0,2])
    feat1 = np.sqrt(np.sum((a-b)**2, axis=2))
    a = np.expand_dims(transformed, axis=1).repeat(transformed.shape[0], axis=1)
    b = np.transpose(a, [1,0,2])
    feat2 = np.sqrt(np.sum((a-b)**2, axis=2))
#     dm = cdist(feat1, feat2, lambda u, v: len(np.intersect1d(u, v)))
#     match = np.argmax(dm, axis=1)
    point_num = orig.shape[0]
    
    cat = np.ones(point_num)*-1
    
    corres = np.ones(point_num)*-1
    
    unmatched_point = set(np.arange(point_num))
    
    cat_id = 0
    iteration = 0
    while(iteration < object_num+1 and len(unmatched_point) > 0):
        max_iteration = 0
        find_flag = False
        while(max_iteration < 100):
            ref_point_index = np.random.choice(np.array(list(unmatched_point)), 4, replace=False)
            dm = cdist(feat1[ref_point_index], feat2, lambda u, v: len(np.intersect1d(u, v)))
            match = np.argmax(dm, axis=1)
            m = feat1[ref_point_index, :][:, ref_point_index]
            n = feat2[match, :][:, match]
            if np.isclose(m,n).all():
                find_flag = True
                break
            max_iteration += 1
        if(find_flag):
            ref_feat1 = feat1[ref_point_index, :].T  # shape (n,4)
            ref_feat2 = feat2[match, :].T  #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          (n,4)
            
            dist = cdist(ref_feat1, ref_feat2)
            
            in_class = np.min(dist, axis=1) < 1e-9
            matched_point = np.argmin(dist, axis=1)
            cat[in_class] = cat_id
            unmatched_point = unmatched_point - set(in_class.nonzero()[0])
            corres[in_class] = matched_point[in_class]
            
            cat_id += 1
        iteration += 1
    
    return cat, corres


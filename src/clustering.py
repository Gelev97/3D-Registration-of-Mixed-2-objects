import numpy as np

def cluster(pts_before, pts_after, match_idx, thres=1e-3, iters = 10):

    matched_pts = pts_after[match_idx, :]
    (pts_count, c) = pts_before.shape

    result_class = np.zeros((pts_count, ))
    for i in range(1, iters + 1):
        anchor_point = np.random.randint(pts_count)
        anchor_feat = pts_before[anchor_point, :]
        feat_before = np.linalg.norm(pts_before - anchor_feat, axis=1)
        feat_after = np.linalg.norm(pts_after - anchor_feat, axis=1)
        idx = (np.abs(feat_before - feat_after) < thres).astype(float)

        if (consensus(idx, result_class, i)):
            result_class += idx
        else:
            result_class += (1 - idx)

    return np.round(result_class / iters)

def consensus(curr, past, iters):
    (r, ) = past.shape
    past = np.round(past / iters)
    consensus = (curr == past).astype(float).sum()
    if consensus > r / 2:
        return True
    
    return False
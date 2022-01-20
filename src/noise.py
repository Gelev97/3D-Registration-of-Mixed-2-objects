import numpy as np

def gaussian_noise(pts, sigma = 1):
    mean = 0
    (r, c) = pts.shape
    noise = np.random.normal(mean, sigma, (3, c))
    pts[:3, :] += noise
    return pts

def white_noise(pts, r = 1):
    (r, c) = pts.shape
    noise = np.random.uniform(-r, r, (3, c))
    pts[:3, :] += noise
    return pts

def destroy_match(corres, ratio = 0.2):
    (r, ) = corres.shape
    countable = np.floor(ratio * r).astype(int)
    start_idx = np.random.randint(0, )
    rand_idx = np.random.choice(r, countable, replace = False)
    print(rand_idx)
    np.random.shuffle(corres[rand_idx])
    print(corres)
    return corres


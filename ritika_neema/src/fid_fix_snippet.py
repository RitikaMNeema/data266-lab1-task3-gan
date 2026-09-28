def compute_fid(feat_real, feat_fake):
    mu1, sigma1 = feat_real.mean(axis=0), np.cov(feat_real, rowvar=False)
    mu2, sigma2 = feat_fake.mean(axis=0), np.cov(feat_fake, rowvar=False)
    diff = mu1 - mu2
    try:
        covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    except TypeError:
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

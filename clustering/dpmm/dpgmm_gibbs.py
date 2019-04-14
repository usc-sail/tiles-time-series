"""
Posterior sampling for Gaussian Mixture Model with CRP prior (DPGMM) using Gibbs sampler
Reference: https://pdfs.semanticscholar.org/9ece/0336316d78837076ef048f3d07e953e38072.pdf
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


def DPMM(X, alpha=100, iter=1, K=30):
    N, D = X.shape
    
    # GMM params
    mus = []  # List of 2x1 vector (mean vector of each gaussian)
    sigma = np.eye(D)
    prec = np.linalg.inv(sigma)  # Fixed precision matrix for all Gaussians
    zs = np.zeros([N], dtype=int)  # Assignments
    # C = []  # Cluster: binary matrix of K x M
    Ns = []  # Count of each cluster
    
    # CRP prior
    alpha, last_z = alpha, zs
    
    # Base distribution prior: N(mu0, prec0)
    mu0 = np.ones(D)
    sigma0 = np.eye(D)
    prec0 = np.linalg.inv(np.eye(D))
    G0 = st.multivariate_normal(mean=mu0, cov=np.eye(D))
    
    # Initialize with ONE cluster
    random_state = 170
    # y_pred = KMeans(n_clusters=10, random_state=random_state).fit_predict(X)
    zs[:] = KMeans(n_clusters=K, random_state=random_state).fit_predict(X)
    for k in np.unique(zs):
        # C.append(np.ones(N, dtype=int))
        # index = np.where(zs == i)[0]
        # Ns.append(len(index))
        # mus.append(np.mean(X[np.where(zs == 0)], axis=0).reshape([D]))

        Xk = X[zs == k]
        Ns.append(Xk.shape[0])

        # Covariance of posterior
        lambda_post = prec0 + Ns[k] * prec
        cov_post = np.linalg.inv(lambda_post)

        # Mean of posterior
        left = cov_post
        right = prec0 @ mu0 + Ns[k] * prec @ np.mean(Xk, axis=0)
        mus_post = left @ right

        # Draw new mean sample from posterior
        mus.append(st.multivariate_normal.rvs(mus_post, cov_post))
        
    # G = st.multivariate_normal(allow_singular=True, mean=np.mean(X[np.where(zs == 0)], axis=0).reshape([D]),
    # cov=np.cov(X[np.where(zs == 0)].T))

    # C.append(np.ones(N, dtype=int))
    # zs[:] = 0
    # Ns.append(N)
    # mus.append(G0.rvs())
    # K = 1
    
    mvn = st.multivariate_normal
    
    # Gibbs sampler
    for it in range(iter):
        # --------------------------------------------------------
        # Sample from full conditional of assignment from CRP prior
        # z ~ GEM(alpha)
        # --------------------------------------------------------
        
        # For each data point, draw the cluster assignment
        for i in range(N):
            # Remove assignment from cluster
            # ------------------------------
            if i % 100 == 0:
                print('iteration: %d, sample index %d' % (it, i))

            zi = zs[i]
            # C[zi][i] = 0
            Ns[zi] -= 1
    
            # If empty, remove cluster
            if Ns[zi] == 0:
                # Fix indices
                zs[zs > zi] -= 1
    
                # Delete cluster
                # del C[zi]
                del Ns[zi]
                del mus[zi]
    
                # Decrement cluster count
                K -= 1
    
            # Draw new assignment zi weighted by CRP prior
            # --------------------------------------------
    
            probs = np.zeros(K+1)
            zs_minus_i = zs[np.arange(len(zs)) != i]
    
            # Probs of joining existing cluster
            for k in range(K):
                nk_minus = zs_minus_i[zs_minus_i == k].shape[0]
                crp = nk_minus / (N + alpha - 1)
                probs[k] = crp * mvn.pdf(X[i], mus[k], sigma)
    
            # Prob of creating new cluster
            crp = alpha / (N + alpha - 1)
            lik = mvn.pdf(X[i], mu0, sigma0+sigma)  # marginal dist. of x
            probs[K] = crp*lik
    
            # Normalize
            probs /= np.sum(probs)
    
            # Draw new assignment for i
            z = np.random.multinomial(n=1, pvals=probs).argmax()
    
            # Update assignment trackers
            if z == K:
                # C.append(np.zeros(N, dtype=int))
                Ns.append(0)
                mus.append(G0.rvs())
                K += 1
    
            zs[i] = z
            Ns[z] += 1
            # C[z][i] = 1
            
        # -------------------------------------------------
        # Sample from full conditional of cluster parameter
        # -------------------------------------------------
    
        # Assume fixed covariance => posterior is Normal
        # mu ~ N(mu, sigma)
        for k in range(K):
            Xk = X[zs == k]
            Ns[k] = Xk.shape[0]
    
            # Covariance of posterior
            lambda_post = prec0 + Ns[k]*prec
            cov_post = np.linalg.inv(lambda_post)
    
            # Mean of posterior
            left = cov_post
            right = prec0 @ mu0 + Ns[k]*prec @ np.mean(Xk, axis=0)
            mus_post = left @ right
    
            # Draw new mean sample from posterior
            mus[k] = mvn.rvs(mus_post, cov_post)
        
        if it == 0:
            last_z = zs
            continue
        
        if np.sum(last_z == zs) == 0:
            break
            
        print(Ns)

        last_z = zs
        
    
    # Even though we only initialize with one cluster, the result should be:
    #
    # Expected output:
    # ----------------
    # 20 data in cluster-0, mean: [ 5  5 ]
    # 20 data in cluster-1, mean: [ 8  8 ]
    # 10 data in cluster-2, mean: [ 20  20 ]
    #
    # Note: cluster label is exchangeable
    # for k in range(K):
    #    print('{} data in cluster-{}, mean: {}'.format(Ns[k], k, mus[k]))
    
    return zs
## Which version
import os

import numpy as np
import pandas as pd
from coxdev import CoxDeviance

d = pd.read_csv("foo.csv", header=None)
beta = pd.read_csv("beta.csv", header=None)
start = None
coxdev = CoxDeviance(event=np.asarray(d[0]),
                     start=start,
                     status=np.asarray(d[1]),
                     tie_breaking='breslow')

n = d.shape[0]
p = d.shape[1] - 3
# X = rng.standard_normal((n, p))
# beta = rng.standard_normal(p) / np.sqrt(n)
# weight = sample_weight(n)
weight = np.asarray(d[170])
X = np.asarray(d.iloc[:, 2:170].values)

C = coxdev(X @ beta, weight)

eta = X @ beta

H = coxdev.information(eta,
                       weight)
v = rng.standard_normal(H.shape[0])
Hv = H @ v

I = X.T @ (H @ X)

    assert np.allclose(I, I.T)
    cov_ = np.linalg.inv(I)

    (G_coxph,
     D_coxph,
     cov_coxph) = get_coxph(event=np.asarray(data['event']),
                            status=np.asarray(data['status']),
                            beta=beta,
                            sample_weight=weight,
                            start=start,
                            ties=tie_breaking,
                            X=X)

    print(D_coxph, C.deviance - 2 * C.loglik_sat)
    assert np.allclose(D_coxph[0], C.deviance - 2 * C.loglik_sat)
    delta_ph = np.linalg.norm(G_coxph - X.T @ C.gradient) / np.linalg.norm(X.T @ C.gradient)
    assert delta_ph < tol
    assert np.linalg.norm(cov_ - cov_coxph) / np.linalg.norm(cov_) < tol

    
test_coxph(tie_types = all_combos[100], tie_breaking = 'efron', sample_weight = sample_weights,
           have_start_times = True, nrep = 1, size =5, tol = 1e-10)

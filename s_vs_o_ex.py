# pca_pce_demo.py
import numpy as np
import matplotlib.pyplot as plt
import openturns as ot

from cardio_ex import simulate_cardio_multiple_params

#########################
# 1) PCA routine
#########################

def comb(n, k):
    """
    Return binomial coefficient C(n, k).
    Equivalent to: n! / (k! * (n-k)!)
    """
    if 0 <= k <= n:
        c = 1
        for i in range(min(k, n - k)):
            c = c * (n - i) // (i + 1)
        return c
    else:
        return 0
def do_pca_centered(data_mat, n_components=None):
    """
    data_mat shape=(N_samples, N_features)
      e.g. (N_params, N_time)
    We'll center the columns => subtract mean(t).
    Return:
      mean_vec: shape (N_features,)
      components: shape (n_components, N_features) => top principal axes
      scores: shape (N_samples, n_components) => coefficients alpha_ij
    """
    # 1) center
    mean_vec = np.mean(data_mat, axis=0)
    centered = data_mat - mean_vec

    # 2) SVD
    U,S,VT = np.linalg.svd(centered, full_matrices=False)
    # U shape=(N_samples, N_samples)
    # S shape=(min(N_samples,N_features),)
    # VT shape=(min(N_samples,N_features), N_features)

    # we pick n_components => if None => pick rank or smaller
    rank = np.linalg.matrix_rank(centered)
    if n_components is None or n_components>rank:
        n_components = rank

    # top n_components
    components = VT[:n_components,:]  # shape (n_components, N_features)
    scores     = U[:,:n_components]*S[:n_components]  # shape (N_samples, n_components)
    return mean_vec, components, scores

#########################
# 2) Build PCE in param space for each PC coefficient
#########################
def build_pce_for_scores(param_samples, scores, polynomial_degree=3):
    """
    param_samples shape=(N_params, d)
    scores shape=(N_params,) for 1 PC
    We'll do a PCE in dimension d => param distribution => scores
    Return meta_model for that PC
    """
    # We define Uniform(...) for each dimension => fix your param bounds
    # For demonstration, let's do a small snippet => we assume we have param_bounds 
    # We'll treat param_samples' min..max as bounding box => or you define them externally

    d = param_samples.shape[1]
    lower = np.min(param_samples, axis=0)
    upper = np.max(param_samples, axis=0)
    marginals=[]
    for j in range(d):
        marginals.append(ot.Uniform(lower[j], upper[j]))
    distribution=ot.ComposedDistribution(marginals)

    # Convert param,score => OT Sample
    N= param_samples.shape[0]
    input_sample=ot.Sample(param_samples.tolist())
    output_sample=ot.Sample(scores.reshape(-1,1).tolist())

    # Build a polynomial basis => dimension=d, Legendre for Uniform
    poly_coll=ot.PolynomialFamilyCollection(d)
    for j in range(d):
        poly_coll[j]=ot.LegendreFactory()

    from openturns import LinearEnumerateFunction, OrthogonalProductPolynomialFactory, FixedStrategy, LeastSquaresStrategy, FunctionalChaosAlgorithm
    enum_func=LinearEnumerateFunction(d)
    product_factory=OrthogonalProductPolynomialFactory(poly_coll, enum_func)

    # number of terms

    n_terms=comb(d+polynomial_degree,d)
    strategy=FixedStrategy(product_factory, n_terms)
    projection=LeastSquaresStrategy()

    chaos_algo=FunctionalChaosAlgorithm(
        input_sample,
        output_sample,
        distribution,
        strategy,
        projection
    )
    chaos_algo.run()
    return chaos_algo.getResult().getMetaModel()

def evaluate_pce_for_scores(meta_model, param_array):
    # param_array shape=(N_eval, d)
    ot_input = ot.Sample(param_array.tolist())
    ot_output= meta_model(ot_input)
    return np.array([row[0] for row in ot_output])

#########################
# 3) Full PCA+PCE approach
#########################
def main():
    # 1) param sampling
    np.random.seed(0)
    N_params=15
    d=9
    param_bounds = [
        (0.21, 0.34),
        (0.15, 0.205),
        (0.042, 0.078),
        (0.0231, 0.0429),
        (0.777, 1.443),
        (0.791, 1.469),
        (7.7, 14.3),
        (1.05, 1.95),
        (0.021, 0.039),
    ]
    param_samples=np.zeros((N_params,d))
    for j in range(d):
        low,high = param_bounds[j]
        param_samples[:,j] = np.random.uniform(low, high, size=N_params)

    # 2) get time series for pLV => shape (N_params, N_time)
    # using the 'simulate_cardio_multiple_params' approach
    t_ref, pLV_mat, pSA_mat, Vlv_mat = simulate_cardio_multiple_params(param_samples, end_time=35.0, dt=0.01)
    # shape => pLV_mat=(N_params, N_time)

    # we'll do PCA on pLV_mat => so data_mat shape=(N_params,N_time)
    data_mat= pLV_mat
    mean_vec, components, scores = do_pca_centered(data_mat, n_components=5)
    # mean_vec shape=(N_time,)
    # components shape=(5, N_time)
    # scores shape=(N_params,5)

    # 3) build PCE for each PC
    polynomial_degree=3
    metas=[]
    for jPC in range(components.shape[0]):
        # scores[:,jPC] => the coefficient alpha_{i,jPC}
        meta_model_j= build_pce_for_scores(
            param_samples,
            scores[:,jPC],
            polynomial_degree
        )
        metas.append(meta_model_j)

    # Reconstruct => for a new param
    test_param = np.array([
        0.3, 0.2, 0.06, 0.03, 1.2, 1.2, 10.0, 1.6, 0.035
    ]) # random test

    # Evaluate each PC's coefficient => shape=(5,)
    test_param_array = test_param.reshape(1,-1) # shape(1,9)
    alphas=[]
    for meta_j in metas:
        val_j= evaluate_pce_for_scores(meta_j, test_param_array)  # shape(1,)
        alphas.append(val_j[0])
    alphas=np.array(alphas) # shape(5,)

    # Reconstruct pLV(t) => mean_vec + sum_{j=1..5} alpha_j * components[j]
    # dimension of mean_vec => (N_time,)
    # components => shape(5, N_time)
    pLV_reconstructed = mean_vec + np.dot(alphas, components)

    # let's see how it compares to direct ODE with test_param
    # We'll do a direct run:
    from cardio_ex import simulate_cardio_model
    t_test, pLV_test, pSA_test, Vlv_test= simulate_cardio_model(test_param, end_time=35, dt=0.01)

    # possibly interpolate pLV_reconstructed => it's on t_ref, shape(N_time,)
    # t_test might differ => let's just do nearest or direct plot vs t_ref
    import matplotlib.pyplot as plt

    # We'll unify time => let's do parted => or we can do simpler approach => directly compare
    # We'll do partial => let's do t_ref => shape(N_time,)
    # We'll also do a small mask => for 33..35
    mask_33_35_ref= (t_ref>=33)&(t_ref<=35)
    t_ref_crop= t_ref[mask_33_35_ref]
    pLV_crop= pLV_reconstructed[mask_33_35_ref]

    mask_33_35_test=(t_test>=33)&(t_test<=35)
    t_test_crop=t_test[mask_33_35_test]
    pLV_test_crop=pLV_test[mask_33_35_test]

    plt.figure(figsize=(10,5))
    plt.plot(t_test_crop, pLV_test_crop,'b-',label='ODE (Test Param) pLV')
    plt.plot(t_ref_crop, pLV_crop,'r--', label='PCA+PCE Reconstructed pLV')
    plt.xlabel("Time (s)")
    plt.ylabel("pLV")
    plt.title("PCA+PCE for time-varying pLV, compare new param from 33..35s")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    main()

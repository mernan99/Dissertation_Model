import numpy as np
import openturns as ot
import matplotlib.pyplot as plt

from cardio_model import CardiovascularModel, HRV

###############################################
# 1) Single-run function with finer steps
###############################################
def run_full_simulation(param_vector, end_time=43):
    """
    Runs one simulation for 'param_vector' (9D).
    We'll use a smaller step to gather more training data.
    """
    from assimulo.problem import Implicit_Problem
    from assimulo.solvers.sundials import IDA

    t_τL = HRV(end_time)  # random cycle times, each >= 0.4s
    u0 = np.array([8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0])
    udot0 = np.zeros(7)

    model = CardiovascularModel(u0, udot0, param_vector, t_τL)
    problem = Implicit_Problem(model.res, u0, udot0, 0.0)
    problem.root = model.root
    problem.handle_event = model.handle_event
    problem.nroots = 1

    solver = IDA(problem)
    solver.report_continuously = True
    solver.atol = 1e-6
    solver.rtol = 1e-6
    solver.maxord = 5
    solver.maxh = 0.01    # Smaller max step => capture the wave more finely
    solver.maxsteps = 50000

    tfinal = t_τL[-1] + 0.1
    # Use a smaller interval as well
    t_eval = np.arange(0, tfinal + 0.001, 0.001)

    t_values, Y, Ydot = solver.simulate(tfinal, ncp_list=t_eval)
    t_values = np.array(t_values)
    Y = np.array(Y)
    event_times = np.array(model.event_times)
    return t_values, Y, event_times, param_vector


def build_multi_cycle_data(t_values, Y, event_times, param_vector):
    """
    9 param + cycle_index + phase => 11D input
    Y => columns [0,1,2,3] => pLV, pSA, pSV, Vlv
    """
    pLV = Y[:,0]
    pSA = Y[:,1]
    pSV = Y[:,2]
    Vlv = Y[:,3]

    # If no events => fallback => entire run = 1 cycle
    if len(event_times) == 1:
        event_times = np.append(event_times, t_values[-1])

    X_all = []
    Y_all_pLV = []
    Y_all_pSA = []
    Y_all_pSV = []
    Y_all_Vlv = []

    n_cycles = len(event_times) - 1
    for j in range(n_cycles):
        start = event_times[j]
        end   = event_times[j+1]
        idx   = np.where((t_values>=start)&(t_values<=end))[0]
        if len(idx) == 0:
            continue
        dt = end - start
        if dt < 1e-12:
            continue
        t_cycle = t_values[idx]
        phase = (t_cycle - start)/dt
        cycle_index = j/(n_cycles-1) if n_cycles>1 else 0.0

        n_pts = len(idx)
        X_params = np.tile(param_vector.reshape(-1,1), (1,n_pts))
        X_cycle_idx = cycle_index*np.ones((1,n_pts))
        X_phase = phase.reshape(1,n_pts)

        X_cycle = np.vstack([X_params, X_cycle_idx, X_phase])
        X_all.append(X_cycle)

        Y_all_pLV.append(pLV[idx])
        Y_all_pSA.append(pSA[idx])
        Y_all_pSV.append(pSV[idx])
        Y_all_Vlv.append(Vlv[idx])

    if not X_all:
        return None, None
    X_data = np.hstack(X_all).T
    Y_data_pLV = np.concatenate(Y_all_pLV)
    Y_data_pSA = np.concatenate(Y_all_pSA)
    Y_data_pSV = np.concatenate(Y_all_pSV)
    Y_data_Vlv = np.concatenate(Y_all_Vlv)
    return X_data, (Y_data_pLV, Y_data_pSA, Y_data_pSV, Y_data_Vlv)

###############################################
# 2) Build PCE with a higher polynomial degree + LARS
###############################################
def build_pce_surrogates_openturns(X_train, Y_train_list, bounds_list, polynomial_degree=3):
    """
    Builds a PCE surrogate for each output in Y_train_list (e.g. pLV, pSA,...)
    using the 5-argument constructor of FunctionalChaosAlgorithm:
       (X, Y, dist, adaptive_strategy, projection_strategy).

    This matches older versions of OT that do not accept a 3-arg constructor.
    """
    import openturns as ot

    # 1) Create the input distribution
    marginals = [ot.Uniform(low, high) for (low, high) in bounds_list]
    distribution = ot.ComposedDistribution(marginals)

    # 2) Convert X to an OpenTURNS Sample
    input_sample = ot.Sample(X_train.tolist())  # shape (n_samples, n_inputs)
    n_samples, n_inputs = input_sample.getSize(), input_sample.getDimension()

    # 3) Build polynomial basis
    poly_coll = ot.PolynomialFamilyCollection(n_inputs)
    for j in range(n_inputs):
        poly_coll[j] = ot.LegendreFactory()
    enum_func = ot.LinearEnumerateFunction(n_inputs)
    product_factory = ot.OrthogonalProductPolynomialFactory(poly_coll, enum_func)

    # 4) Number of terms for total degree = polynomial_degree
    def comb(n, k):
        if k < 0 or k > n:
            return 0
        c = 1
        for i in range(min(k, n-k)):
            c = c * (n - i) // (i + 1)
        return c
    max_terms = comb(n_inputs + polynomial_degree, n_inputs)

    # 5) Create an "AdaptiveStrategy". For instance, a simple FixedStrategy 
    #    that includes all polynomials up to 'max_terms':
    adaptive_strategy = ot.FixedStrategy(product_factory, max_terms)

    # 6) Create a "ProjectionStrategy". Options:
    #    - Plain OLS: projection_strategy = ot.LeastSquaresStrategy()
    #    - LARS-based approach:
    #         lars_factory = ot.ApproximationAlgorithmImplementationFactory(ot.LARS())
    #         projection_strategy = ot.LeastSquaresStrategy(lars_factory)
    #      (But it may fail in certain older OT versions if LARS is missing.)
    projection_strategy = ot.LeastSquaresStrategy()  # plain OLS

    meta_models = []
    for Y_train in Y_train_list:
        # Convert Y to an OT Sample
        output_sample = ot.Sample(Y_train.reshape(-1, 1).tolist())

        # 7) Create the FunctionalChaosAlgorithm with 5 arguments
        chaos_algo = ot.FunctionalChaosAlgorithm(
            input_sample,         # X
            output_sample,        # Y
            distribution,         # Joint dist of X
            adaptive_strategy,    # which polynomials to consider
            projection_strategy   # how to do the least squares
        )

        # 8) Run the algorithm
        chaos_algo.run()

        # 9) Extract the result
        result = chaos_algo.getResult()
        meta_models.append(result.getMetaModel())

    return meta_models, distribution

###############################################
# 3) Evaluate Surrogate using REAL event boundaries
###############################################
def evaluate_surrogate_with_events(meta_models, param_vector, t_values, event_times):
    """
    For each t in t_values, find the correct cycle j => compute phase => evaluate PCE.
    """
    import openturns as ot

    def evaluate_pce_model(metamodels, X):
        input_sample = ot.Sample(X.tolist())
        preds = np.zeros((X.shape[0], len(metamodels)))
        for j, mm in enumerate(metamodels):
            vals = mm(input_sample)
            preds[:, j] = [v[0] for v in vals]
        return preds

    results = np.zeros((len(t_values), 4))
    n_cycles = len(event_times) - 1
    if n_cycles<1:
        return results

    for i, t in enumerate(t_values):
        j = np.searchsorted(event_times, t) - 1
        if j<0: j=0
        if j>=n_cycles: j=n_cycles-1
        start = event_times[j]
        end   = event_times[j+1]
        dt = end - start
        if dt<1e-12:
            phase=0.0
        else:
            phase=(t-start)/dt
        if phase<0: phase=0
        if phase>1: phase=1
        cycle_idx = j/(n_cycles-1) if n_cycles>1 else 0.0
        x_row = np.concatenate([param_vector,[cycle_idx,phase]]).reshape(1,-1)
        res_i = evaluate_pce_model(meta_models, x_row)
        results[i,:] = res_i
    return results

###############################################
# 4) Main demonstration
###############################################
def main():
    ###############################################
    # A) Create many param sets for training
    ###############################################
    np.random.seed(42)
    N_param = 10  # number of param sets
    # bounds for the 9 parameters
    p_lower = [0.21, 0.15, 0.042, 0.0231, 0.777, 0.791, 7.7, 1.05, 0.021]
    p_upper = [0.34, 0.205,0.078, 0.0429,1.443,1.469,14.3,1.95, 0.039]

    param_sets = np.random.uniform(low=p_lower, high=p_upper, size=(N_param,9))

    X_all = []
    Y_pLV_all = []
    Y_pSA_all = []
    Y_pSV_all = []
    Y_Vlv_all = []

    # B) Build training data from multiple runs
    for i, p_i in enumerate(param_sets):
        print(f"Running param set {i+1}/{N_param}: {p_i}")
        t_vals, Y, event_times, used_p = run_full_simulation(p_i, end_time=35)
        X_run, Y_run = build_multi_cycle_data(t_vals, Y, event_times, used_p)
        if X_run is None:
            print("No data for param:", p_i)
            continue
        pLV_run, pSA_run, pSV_run, Vlv_run = Y_run
        X_all.append(X_run)
        Y_pLV_all.append(pLV_run)
        Y_pSA_all.append(pSA_run)
        Y_pSV_all.append(pSV_run)
        Y_Vlv_all.append(Vlv_run)

    X_train = np.vstack(X_all)
    Y_train_pLV = np.concatenate(Y_pLV_all)
    Y_train_pSA = np.concatenate(Y_pSA_all)
    Y_train_pSV = np.concatenate(Y_pSV_all)
    Y_train_Vlv = np.concatenate(Y_Vlv_all)

    # C) Build a higher-order PCE with LARS
    bounds_list_params = [
        (0.21,0.34),
        (0.15,0.205),
        (0.042,0.078),
        (0.0231,0.0429),
        (0.777,1.443),
        (0.791,1.469),
        (7.7,14.3),
        (1.05,1.95),
        (0.021,0.039),
    ]
    # cycle_index, phase => [0,1]
    bounds_11 = bounds_list_params + [(0.0,1.0),(0.0,1.0)]
    Y_train_list = [Y_train_pLV, Y_train_pSA, Y_train_pSV, Y_train_Vlv]

    poly_degree = 3
    meta_models, distribution = build_pce_surrogates_openturns(
        X_train, Y_train_list, bounds_11, polynomial_degree=poly_degree
    )
    print("Finished building PCE with polynomial_degree =", poly_degree)
    print("Number of total training points:", X_train.shape[0])

    # D) Now test a new param
    p_test = np.array([0.31,0.42,0.065,0.03,1.12,1.05,12.0,1.7,0.028])
    print("Testing surrogate with param:", p_test)

    # Re-run solver with p_test => compare
    t_test, Y_test, ev_times_test, used_p = run_full_simulation(p_test, end_time=35)
    
    # Plot 33..35
    mask = (t_test>=33)&(t_test<=35)
    t_zoom = t_test[mask]
    Y_sur = evaluate_surrogate_with_events(meta_models, p_test, t_zoom, ev_times_test)

    # Compare
    plt.figure(figsize=(10,6))
    # plt.plot(t_zoom, Y_test[mask,0], label='True pLV', alpha=0.7)
    # plt.plot(t_zoom, Y_test[mask,1], label='True pSA', alpha=0.7)
    # plt.plot(t_zoom, Y_test[mask,2], label='True pSV', alpha=0.7)
    plt.plot(t_zoom, Y_sur[:,0], '--', label='Surrogate pLV')
    plt.plot(t_zoom, Y_sur[:,1], '--', label='Surrogate pSA')
    plt.plot(t_zoom, Y_sur[:,2], '--', label='Surrogate pSV')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure')
    plt.title('Compare Surrogate vs. True Model (Pressures) - new param')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,6))
    # plt.plot(t_zoom, Y_test[mask,3], label='True Vlv', alpha=0.7)
    plt.plot(t_zoom, Y_sur[:,3], '--', label='Surrogate Vlv')
    plt.xlabel('Time (s)')
    plt.ylabel('Volume')
    plt.title('Compare Surrogate vs. True Model (Vlv) - new param')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__=="__main__":
    main()

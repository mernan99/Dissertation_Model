# surrogate.py


# def build_pce_surrogates_openturns(
#     X_train, 
#     Y_train_list,
#     bounds_list,
#     polynomial_degree=3
# ):
#     """
#     Build OpenTURNS-based PCE surrogates for multiple outputs (e.g., pLV, psa, Vlv).

#     Parameters
#     ----------
#     X_train : np.ndarray, shape (n_samples, n_params)
#         Training inputs (parameter sets).
#     Y_train_list : list of np.ndarray, each shape (n_samples,)
#         A list of output arrays, one for each response variable 
#         (e.g. [pLV_final, psa_final, Vlv_final]).
#     bounds_list : list of (low, high) for each parameter
#         Bounds for each parameter, used to define Uniform distributions.
#     polynomial_degree : int
#         Maximum total polynomial degree for the PCE expansions.

#     Returns
#     -------
#     meta_models : list of ot.Function
#         A list of OpenTURNS metamodels, one for each output in Y_train_list.
#     distribution : ot.ComposedDistribution
#         The joint input distribution used for PCE.
#     """
#     import openturns as ot
#     n_samples, n_params = X_train.shape

#     # 1) Create input distribution from user-provided bounds (Uniform)
#     marginals = [ot.Uniform(low, high) for (low, high) in bounds_list]
#     distribution = ot.ComposedDistribution(marginals)

#     # 2) Convert training data to OpenTURNS format
#     input_sample = ot.Sample(X_train.tolist())  # shape (n_samples, n_params)

#     meta_models = []

#     # 3) Build a PCE for each output
#     for Y_train in Y_train_list:
#         # Convert output to ot.Sample
#         output_sample = ot.Sample(Y_train.reshape(-1, 1).tolist())

#         # Build polynomial basis factories (Legendre for Uniform)
#         poly_basis_factory = ot.PolynomialFamilyCollection(n_params)
#         for j in range(n_params):
#             poly_basis_factory[j] = ot.LegendreFactory()

#         enumeration_function = ot.LinearEnumerateFunction(n_params)
#         total_poly_basis = ot.OrthogonalProductPolynomialFactory(
#             poly_basis_factory, enumeration_function
#         )

#         # limit to 1000 basis terms; or the library chooses adaptively
#         # adaptive_strategy = ot.FixedStrategy(total_poly_basis, 4000)
#         lars_factory = ot.ApproximationAlgorithmImplementationFactory(ot.LARS())
#         projection_strategy = ot.LeastSquaresStrategy(lars_factory)
#         adaptive_strategy = ot.SequentialStrategy(total_poly_basis, projection_strategy)

#         # Use a selection-based approach (LARS) instead of a fixed strategy
#         # projection_strategy = ot.LeastSquaresStrategy(ot.LARS(), ot.Normal())
#         # adaptive_strategy = ot.SequentialStrategy(total_poly_basis, projection_strategy)
#         # projection_strategy = ot.LeastSquaresStrategy(ot.LARS())  # Just pass LARS
#         # adaptive_strategy = ot.SequentialStrategy(total_poly_basis, projection_strategy)


#         # 4) Create the FunctionalChaosAlgorithm with no custom projection strategy
#         # chaos_algo = ot.FunctionalChaosAlgorithm(
#         #     input_sample, 
#         #     output_sample, 
#         #     distribution, 
#         #     adaptive_strategy
#         # )


#         # chaos_algo.run()
#         # 1. Instantiate FunctionalChaosAlgorithm without projection strategy
#         chaos_algo = ot.FunctionalChaosAlgorithm (
#             input_sample,
#             output_sample,
#             distribution
#         )

#         # 2. Set the polynomial basis + degree
#         chaos_algo.setOrthogonalPolynomials(total_poly_basis)
#         chaos_algo.setMaximumTotalDegree(polynomial_degree)

#         # 3. Use internal LARS model selection logic
#         chaos_algo.setUseLARSModelSelection(True)

#         # Optional: set model selection criterion like Corrected LOO
#         # chaos_algo.setModelSelectionCriterion(ot.CorrectedLeaveOneOut())

#         # 4. Run the algorithm
#         chaos_algo.run()



#         # 6) Extract the result:
#         chaos_result = chaos_algo.getResult()
#         meta_model = chaos_result.getMetaModel()


#         # 5) Extract the resulting metamodel
#         chaos_result = chaos_algo.getResult()
#         meta_model = chaos_result.getMetaModel()
#         meta_models.append(meta_model)

#     return meta_models, distribution

import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli

import openturns as ot


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
    
def build_pce_surrogates_openturns(
    X_train,
    Y_train_list,
    bounds_list,
    polynomial_degree=3
):
    """
    Build PCE surrogates of total degree <= polynomial_degree
    in OpenTURNS 1.19 using a "FixedStrategy" (no LARS).
    
    Parameters
    ----------
    X_train : np.ndarray of shape (n_samples, n_params)
        Input parameter samples
    Y_train_list : list of np.ndarray, each (n_samples,)
        One output array per response variable
    bounds_list : list of (low, high)
        Uniform bounds for each parameter dimension
    polynomial_degree : int
        Maximum total polynomial degree

    Returns
    -------
    meta_models : list of ot.Function
        One PCE model per output
    distribution : ot.ComposedDistribution
        The joint uniform distribution
    """
    # 1) Distribution from uniform bounds
    marginals = [ot.Uniform(low, high) for (low, high) in bounds_list]
    distribution = ot.ComposedDistribution(marginals)

    # 2) Convert X to OpenTURNS Sample
    input_sample = ot.Sample(X_train.tolist())
    n_params = input_sample.getDimension()

    # 3) Build the polynomial basis factory (Legendre for Uniform):
    poly_coll = ot.PolynomialFamilyCollection(n_params)
    for j in range(n_params):
        poly_coll[j] = ot.LegendreFactory()
    enumerate_function = ot.LinearEnumerateFunction(n_params)
    product_factory = ot.OrthogonalProductPolynomialFactory(poly_coll, enumerate_function)

    # 4) Determine how many terms appear up to total degree p
    #    For dimension d, total degree p => binomial(d+p, d)
    def n_terms_for_degree(dim, p):
        return comb(p + dim, dim)

    max_terms = n_terms_for_degree(n_params, polynomial_degree)
    
    # 5) Create a "FixedStrategy" with exactly 'max_terms' polynomials
    adaptive_strategy = ot.FixedStrategy(product_factory, max_terms)

    # 6) Plain least-squares projection strategy
    projection_strategy = ot.LeastSquaresStrategy()  # Standard OLS

    meta_models = []
    for Y_train in Y_train_list:
        output_sample = ot.Sample(Y_train.reshape(-1, 1).tolist())

        # 7) Use the 5-argument constructor: (X, Y, dist, adaptiveStrategy, projectionStrategy)
        chaos_algo = ot.FunctionalChaosAlgorithm(
            input_sample,
            output_sample,
            distribution,
            adaptive_strategy,
            projection_strategy
        )

        # 8) Run the PCE
        chaos_algo.run()
        result = chaos_algo.getResult()
        meta_models.append(result.getMetaModel())

    return meta_models, distribution


def evaluate_pce_model_openturns(meta_models, X):
    """
    Evaluate a list of metamodels on new parameter sets.
    Each meta_model in meta_models maps R^n -> R^1.

    Parameters
    ----------
    meta_models : list of ot.Function
        Each is an OpenTURNS function that takes (n_params,) -> (1,).
    X : np.ndarray, shape (n_samples, n_params)
        The new parameter sets to evaluate.

    Returns
    -------
    predictions : np.ndarray, shape (n_samples, n_outputs)
        The predicted outputs from each meta-model.
    """
    n_samples, n_params = X.shape
    n_outputs = len(meta_models)

    # Convert X into an ot.Sample
    input_sample = ot.Sample(X.tolist())

    # Evaluate each meta-model
    predictions = np.zeros((n_samples, n_outputs))
    for j, model in enumerate(meta_models):
        # model(input_sample) returns an ot.Sample of shape (n_samples, 1)
        result_sample = model(input_sample)
        # Convert each row to a float
        result_array = np.array([val[0] for val in result_sample])
        predictions[:, j] = result_array

    return predictions


def sobol_analysis_with_surrogate(
    meta_model,   # a single-output meta-model
    problem_dict, # SALib problem dict
    N_sobol=1000
):
    """
    Perform Sobol analysis with SALib, using an OpenTURNS PCE meta-model
    for a single output.

    Parameters
    ----------
    meta_model : ot.Function
        The PCE surrogate for a single output (R^n -> R^1).
    problem_dict : dict
        Dictionary defining parameter bounds and names for SALib.
    N_sobol : int
        Base sample size for Saltelli's method.

    Returns
    -------
    si : dict
        Dictionary with keys 'S1', 'ST', 'S2', etc.
    """
    from SALib.sample import saltelli
    from SALib.analyze import sobol

    # 1) Generate param values for SA using Saltelli
    param_values_sobol = saltelli.sample(
        problem_dict, 
        N_sobol, 
        calc_second_order=True
    )
    # shape: (N_sobol*(2*n_vars+2), n_vars)

    # 2) Evaluate the meta_model on each sample
    Y_sobol = evaluate_pce_model_openturns([meta_model], param_values_sobol)[:, 0]
    # shape: (n_samples,)

    # 3) Run SALib's Sobol analysis
    si = sobol.analyze(problem_dict, Y_sobol, calc_second_order=True, print_to_console=False)
    return si
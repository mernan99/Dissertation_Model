# surrogate.py
import numpy as np
import openturns as ot
from SALib.analyze import sobol
from SALib.sample import saltelli

def build_pce_surrogates_openturns(
    X_train, 
    Y_train_list,
    bounds_list,
    polynomial_degree=3
):
    """
    Build OpenTURNS-based PCE surrogates for multiple outputs (e.g., pLV, psa, Vlv).

    Parameters
    ----------
    X_train : np.ndarray, shape (n_samples, n_params)
        Training inputs (parameter sets).
    Y_train_list : list of np.ndarray, each shape (n_samples,)
        A list of output arrays, one for each response variable 
        (e.g. [pLV_final, psa_final, Vlv_final]).
    bounds_list : list of (low, high) for each parameter
        Bounds for each parameter, used to define Uniform distributions.
    polynomial_degree : int
        Maximum total polynomial degree for the PCE expansions.

    Returns
    -------
    meta_models : list of ot.Function
        A list of OpenTURNS metamodels, one for each output in Y_train_list.
    distribution : ot.ComposedDistribution
        The joint input distribution used for PCE.
    """
    import openturns as ot
    n_samples, n_params = X_train.shape

    # 1) Create input distribution from user-provided bounds (Uniform)
    marginals = [ot.Uniform(low, high) for (low, high) in bounds_list]
    distribution = ot.ComposedDistribution(marginals)

    # 2) Convert training data to OpenTURNS format
    input_sample = ot.Sample(X_train.tolist())  # shape (n_samples, n_params)

    meta_models = []

    # 3) Build a PCE for each output
    for Y_train in Y_train_list:
        # Convert output to ot.Sample
        output_sample = ot.Sample(Y_train.reshape(-1, 1).tolist())

        # Build polynomial basis factories (Legendre for Uniform)
        poly_basis_factory = ot.PolynomialFamilyCollection(n_params)
        for j in range(n_params):
            poly_basis_factory[j] = ot.LegendreFactory()

        enumeration_function = ot.LinearEnumerateFunction(n_params)
        total_poly_basis = ot.OrthogonalProductPolynomialFactory(
            poly_basis_factory, enumeration_function
        )

        # limit to 1000 basis terms; or the library chooses adaptively
        adaptive_strategy = ot.FixedStrategy(total_poly_basis, 10000)
        # Use a selection-based approach (LARS) instead of a fixed strategy
        # projection_strategy = ot.LeastSquaresStrategy(ot.LARS(), ot.Normal())
        # adaptive_strategy = ot.SequentialStrategy(total_poly_basis, projection_strategy)
        # projection_strategy = ot.LeastSquaresStrategy(ot.LARS())  # Just pass LARS
        # adaptive_strategy = ot.SequentialStrategy(total_poly_basis, projection_strategy)


        # 4) Create the FunctionalChaosAlgorithm with no custom projection strategy
        chaos_algo = ot.FunctionalChaosAlgorithm(
            input_sample, 
            output_sample, 
            distribution, 
            adaptive_strategy
        )

        # default approach (no setProjectionStrategy())
        # chaos_algo.run()

        # # 1) Create the FunctionalChaosAlgorithm the older/manual way:
        # chaos_algo = ot.FunctionalChaosAlgorithm(input_sample,
        #                                         output_sample,
        #                                         distribution)

        # # 2) Manually set the polynomial basis and total polynomial degree:
        # chaos_algo.setOrthogonalPolynomials(total_poly_basis)
        # chaos_algo.setMaximumTotalDegree(polynomial_degree)

        # # 3) (Optional) If your version has setUseLARSModelSelection:
        # #    This toggles LARS-based term selection internally
        # chaos_algo.setUseLARSModelSelection(True)

        # # 4) Possibly set a model selection criterion (like corrected LOO):
        # # chaos_algo.setModelSelectionCriterion(ot.CorrectedLeaveOneOut())

        # 5) Now run:
        chaos_algo.run()

        # 6) Extract the result:
        chaos_result = chaos_algo.getResult()
        meta_model = chaos_result.getMetaModel()


        # 5) Extract the resulting metamodel
        chaos_result = chaos_algo.getResult()
        meta_model = chaos_result.getMetaModel()
        meta_models.append(meta_model)

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
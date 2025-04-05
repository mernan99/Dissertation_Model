# surrogate_time_only.py
import openturns as ot
import numpy as np

def build_time_only_pce(
    time_array,    # shape (N,) - raw time in [0, end_time]
    Y_array,       # shape (N,)
    end_time,      # float - so we know to scale time to [0,1]
    polynomial_degree=3
):
    """
    Build a 1D polynomial chaos expansion in "normalized time" t_norm = t / end_time.
    Returns meta_model that can map t => Y.
    """
    N = len(time_array)
    if len(Y_array) != N:
        raise ValueError("time_array and Y_array mismatch length")

    # 1) Normalize time to [0,1]
    t_norm = time_array / end_time  # shape (N,)

    # 2) Convert to OT Sample
    input_sample  = ot.Sample(t_norm.reshape(-1,1).tolist())  # shape (N,1)
    output_sample = ot.Sample(Y_array.reshape(-1,1).tolist()) # shape (N,1)

    # 3) Distribution for t_norm: Uniform(0,1)
    distribution = ot.Uniform(0.0, 1.0)

    # 4) Build polynomial basis (Legendre) in 1D
    poly_coll = [ot.LegendreFactory()]
    enum_func = ot.LinearEnumerateFunction(1)
    product_factory = ot.OrthogonalProductPolynomialFactory(poly_coll, enum_func)

    # 5) # of polynomials
    # total degree p => #terms = p+1 in 1D
    n_terms = polynomial_degree + 1

    # Create a "FixedStrategy" with 'n_terms' polynomials
    adaptive_strategy = ot.FixedStrategy(product_factory, n_terms)

    # Plain least-squares
    projection_strategy = ot.LeastSquaresStrategy()

    # 6) Build & run the PCE
    chaos_algo = ot.FunctionalChaosAlgorithm(
        input_sample,
        output_sample,
        distribution,
        adaptive_strategy,
        projection_strategy
    )
    chaos_algo.run()

    # 7) Return the final meta-model
    result = chaos_algo.getResult()
    meta_model = result.getMetaModel()
    return meta_model

def evaluate_time_only_pce(meta_model, t_eval, end_time):
    """
    Evaluate the 1D polynomial chaos at times t_eval (in [0..end_time]).
    """
    # 1) Normalize
    t_norm = t_eval / end_time
    # shape (N_eval, 1)
    ot_input = ot.Sample(t_norm.reshape(-1,1).tolist())
    ot_output = meta_model(ot_input)  # shape (N_eval, 1)
    return np.array([row[0] for row in ot_output])

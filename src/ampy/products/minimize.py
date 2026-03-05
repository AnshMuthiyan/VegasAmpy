import json
from pathlib import Path

import argparse
import numpy as np
from scipy import optimize

from ampy.ampy import AMPy
from ampy.core import utils
from ampy.core.structs import ScaleType
from ampy.inference.engine import log_posterior_model


def safe_log_posterior_fn(theta, engine, param_view):
    """
    Calculates the natural log of the posterior
    probability. The minimizer will crash if the
    posterior is not finite. This method wraps
    and returns a large, negative number instead
    of negative infinity.

    Parameters
    ----------
    theta : np.ndarray of float
        The MCMC sampled values.

    engine : ampy.modeling.engine.ModelingEngine
        The modeling engine.

    param_view : ampy.core.params.ParameterView
        The parameter viewer.

    Returns
    -------
    float
        The natural log of the posterior.
    """
    lp = log_posterior_model(theta, engine, param_view)
    return lp if np.isfinite(lp) else -1e10


def run_minimizer(x0, func_args=(), bounds=(), minimizer='minimize'):
    """
    Runs the minimizer on the negative maximum a
    posteriori (MAP).

    Parameters
    ----------
    x0 : np.ndarray
        Initial positions.

    func_args : tuple
        Any args needed for the ``log_posterior_model``.

    bounds : np.ndarray
        Sequence of ``(min, max)`` pairs for each
        element in `x`. None is used to specify no bound.

    minimizer : str
        Which minimizer to use. Must be 'minimize' or 'basinhopping'.
        Use basinhopping for difficult posteriors.

    Returns
    -------
    OptimizeResult
        The optimization result represented as a ``OptimizeResult``
        object. Important attributes are: ``x`` the solution array,
        ``success`` a Boolean flag indicating if the optimizer exited
        successfully and ``message`` which describes the cause of the
        termination.
    """
    nmap = lambda *lp_args: -safe_log_posterior_fn(*lp_args)

    if minimizer == 'minimize':
        return optimize.minimize(nmap, x0, args=func_args, bounds=bounds)

    elif minimizer == 'basinhopping':
        return optimize.basinhopping(
            nmap, x0, minimizer_kwargs={"method": "L-BFGS-B", "args": func_args, "bounds": bounds}
        )

    raise ValueError(f"Unknown minimizer '{minimizer}'.")


def log_results(p, out_dir):
    """
    Log the minimized parameters to a JSON file.

    Parameters
    ----------
    p : dict
        The minimized parameters.

    out_dir : Path
        The output directory.
    """
    with open(Path(out_dir) / 'minimized.json', "w") as f:
        json.dump(p, f, indent=4)  # type: ignore


def main(obs_path, registry_path, report_path, results_dir):
    """
    Runs the BestFitinator and plots the light curve.

    Parameters
    ----------
    obs_path : Path
        The path to the observation CSV file.

    registry_path : Path
        The path to the registry TOML file.

    report_path : Path
        The path to the report JSON file containing the best fit
        parameters and their inference configuration.

    results_dir : Path
        The output directory.

    Returns
    -------
    OptimizeResult
        See `run_minimizer` for details.
    """
    # Build AMPy from the registry
    ampy = AMPy.from_registry(obs_path, registry_path)
    param_view = ampy.inference_engine.param_view

    # Load the best fitting results from the report
    with open(report_path, "r") as f:
        report = json.load(f)

    # Set the initial search pos and bounds
    initial, bounds = [], []

    for p in param_view.fitting:
        # Best fit values are stored in linear space under report[plugin][stage][name]
        val = report[p.plugin][p.stage][p.name]
        # print(p.infer_scale, val)
        initial.append(utils.to_scale(val, ScaleType.LINEAR, p.infer_scale))
        bounds.append((p.prior.lower, p.prior.upper))

    # Run the minimizer
    func_args = (ampy.inference_engine.modeling_engine, param_view)

    results = run_minimizer(
        x0=np.array(initial), func_args=func_args, bounds=np.array(bounds)
    )

    # Add some additional logging info
    min_params = param_view.samples_to_dict(results.x)
    min_params['nmap'] = -2 * log_posterior_model(results.x, *func_args)
    min_params['success'] = results.success
    min_params['message'] = results.message

    # Write the results to a JSON file
    log_results(min_params, results_dir)

    return results


if __name__ == "__main__":
    # Run the BestFitinator via the command line
    parser = argparse.ArgumentParser(description="BestFitinator Parameters")
    parser.add_argument('--obs',       help='The input observation file.')
    parser.add_argument('--registry',  help='The registry TOML file.')
    parser.add_argument('--report',    help='The report JSON file.')
    parser.add_argument('--results',   help='The results directory.')
    args = parser.parse_args()

    event, sub_dir = '220101A', 'grbs'

    p_obs      = Path(rf"C:\Server\FINAL\analytic\{event}\obs.csv")
    p_registry = Path(rf"C:\Server\FINAL\analytic\{event}\registry.toml")
    p_report   = Path(rf"C:\Server\FINAL\analytic\{event}\report.json")
    d_results  = Path(rf"C:\Server\FINAL\analytic\{event}\minimized")

    response = main(
        obs_path=     args.obs      or p_obs,
        registry_path=args.registry or p_registry,
        report_path=  args.report   or p_report,
        results_dir=  args.results  or d_results,
    )

    print(f"Was the BestFitinator successful? {response.success}")

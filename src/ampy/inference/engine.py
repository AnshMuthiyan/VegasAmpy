import copy
import numpy as np

from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from ampy.inference.samplers import EnsembleSampler
from ampy.inference.samplers import PTSampler


def get_pool_context(workers=None, executor='process'):
    """
    If ``executor==process`` and ``workers>1``:
        Returns ``ProcessPoolExecutor(max_workers=workers)``.

        Since processes need to load everything into memory, this should only
        be used if the likelihood calculation takes about one-second or more
        to calculate.

    If ``executor==thread`` and ``workers>1``:
        Returns ``ThreadPoolExecutor(max_workers=workers)``.

        Note that unless a free-threaded Python is installed, multithreading
        will not yield any benefits. Even if a no-GIL Python version is used,
        the performance increase depends on the likelihood implementation.
        Pure Python implementations will see a large performance increase. If
        the likelihood uses Cython, then it depends on how the code is
        compiled and optimized.

    Parameters
    ----------
    workers : int, optional, default=None
        The max number of workers.

    executor : str, optional, default='process'
        See above docstring for details. Must be ``process`` or ``thread``.

    Returns
    -------
    ``ProcessPoolExecutor`` or ``ThreadPoolExecutor`` or ``nullcontext``
        The pool context manager.
    """
    if workers and workers > 1:

        if executor == 'process':
            return ProcessPoolExecutor(max_workers=workers)

        if executor == 'thread':
            return ThreadPoolExecutor(max_workers=workers)

    return nullcontext()


class InferenceEngine:
    """
    High-level driver for MCMC inference over an :class:`~ampy.modeling.engine.ModelingEngine`.

    It manages sampler construction, initialization of starting positions,
    optional burn-in, and the production run.

    Attributes
    ----------
    modeling_engine : ampy.modeling.engine.ModelingEngine
        The engine used for forward modeling.

    param_view : ampy.core.params.ParameterView
        Parameter and plugin view used by the sampler callbacks.

    sampler : object or None
        The active sampler instance (e.g.,
        :class:`~ampy.inference.samplers.EnsembleSampler` or
        :class:`~ampy.inference.samplers.PTSampler`).

    burn_chain : np.ndarray or None
        Copy of the burn-in chain, if burn-in was run and stored.
    """
    def __init__(self, modeling_engine, param_view):
        self.modeling_engine = modeling_engine
        self.param_view = param_view

        # Sampler
        self.sampler = None
        self.burn_chain = None

    @property
    def ndim(self):
        """ The number of fitting dimensions. """
        return len(self.param_view.fitting)

    def set_sampler(self, sampler, nwalkers, pool, ntemps=None, **kwargs):
        """
        Construct and assign the sampler used for inference.

        Parameters
        ----------
        sampler : {"ensemble", "tempered"}
             Sampler backend to use.

            - ``"ensemble"`` uses :class:`~ampy.inference.samplers.EnsembleSampler`
              (affine-invariant ensemble sampling).
            - ``"tempered"`` uses :class:`~ampy.inference.samplers.PTSampler`
              (parallel tempering).


        nwalkers : int
            The number of walkers.

        pool : `ProcessPoolExecutor` or `ThreadPoolExecutor` or `nullcontext`
            The pool to use for multithreading/processing.

        ntemps : int, optional
            Number of temperatures (required for ``"tempered"``).

        kwargs
            Any kwargs accepted by the sampler.
        """
        if sampler == 'ensemble':
            self.sampler = EnsembleSampler.load(
                nwalkers, self.ndim, log_posterior_model,
                args=(self.modeling_engine, self.param_view), pool=pool,
                **kwargs
            )

        elif sampler == 'tempered':
            self.sampler = PTSampler(
                ntemps, nwalkers, self.ndim,
                log_likelihood_model, log_prior_model,
                log_l_args=(self.modeling_engine, self.param_view), pool=pool,
                log_p_args=(self.param_view,), **kwargs
            )

    def set_start_positions(self, resume=False):
        """
        Determine starting positions for the sampler.

        Parameters
        ----------
        resume : bool, optional, default=False
            If `True` and the sampler exposes ``get_last_sample()``, start
            from the last recorded sample (typically from a backend).

        Returns
        -------
        np.ndarray or `emcee.State`
            Starting positions in the format expected by the active sampler.
        """
        if resume and hasattr(self.sampler, 'get_last_sample'):
            # Use the backend to determine start positions
            return self.sampler.get_last_sample()

        # Use priors to determine start positions
        return self.sampler.draw_positions(
            params=self.param_view,
            engine=self.modeling_engine,
            log_post_fn=log_posterior_model
        )

    def run(
        self, nwalkers, iterations, burn=0, sampler='ensemble',
        workers=None, ntemps=None, sampler_kw=None, run_kw=None, resume=False
    ):
        """
        Runs the MCMC sampling routine.

        Parameters
        ----------
        nwalkers : int
            The number of walkers.

        iterations : int
            The number of post-burn-in iterations.

        burn : int, optional, default=0
            The number of iterations to burn. If `burn>0`, the burn sampler
            is stored to `self.burn_sampler` before resetting it for the main
            run.

        sampler : str, optional, default='ensemble'
            Must be `ensemble` or `tempered`.

        workers : int, optional, default=None
            The max number of workers to use.

        ntemps : int, optional, default=None
            The number of temperatures for `PTSampler`.

        sampler_kw : dict, optional
            Any kwargs accepted by the sampler.

        run_kw : dict, optional
            Any kwargs accepted by `run_mcmc`.

        resume : bool, optional, default=False
            Resume from a previous run?

        Notes
        -----
        Burn-in storage uses a deepcopy of the burn chain rather than
        deep-copying the sampler, because copying the sampler can detach or
        invalidate the pool used for parallelism.
        """
        with get_pool_context(workers) as pool:
            self.set_sampler(
                sampler, nwalkers, pool, ntemps, **(sampler_kw or {})
            )

            start_pos = self.set_start_positions(resume)

            if burn < 1:
                start_run_pos = start_pos

            else:
                start_burn_pos = start_pos

                # Run burn in and save the last position
                start_run_pos = (
                    self.sampler.run_mcmc(
                        start_burn_pos, burn, **(run_kw or {})
                    )
                )

                # Save the chain if desired for diagnostics. Cannot
                # save the entire sampler because deepcopy detaches
                # the pool which prevents multiprocessing/threading
                self.burn_chain = copy.deepcopy(self.sampler.get_chain())
                self.sampler.reset()

            # Run production
            self.sampler.run_mcmc(
                start_run_pos, iterations, **(run_kw or {})
            )

    def summary(self):
        """
        Summarize the current sampling run.

        Returns
        -------
        dict
            Dictionary containing:

            - ``sampler``: sampler name
            - ``burn_iters``: number of burn-in iterations stored (0 if none)
            - ``prod_iters``: number of production iterations completed
            - ``nwalkers``: number of walkers
            - ``nmap_idx``: index of the maximum a-posteriori (MAP) sample in
                the flattened log-probability array
            - ``nmap_val``: ``-2 * max(log_prob)``, which is often used as a
              deviance-like quantity (smaller is better)
        """
        flat_chain = self.sampler.get_log_prob(flat=True)

        out = {
            'sampler': self.sampler.name, 'burn_iters': 0,
            'prod_iters': int(self.sampler.iteration),
            'nwalkers': self.sampler.nwalkers,
            'nmap_idx': int(np.nanargmax(flat_chain)),
            'nmap_val': -2.0 * flat_chain.max()
        }

        if self.burn_chain is not None:
            out['burn_iters'] = int(len(self.burn_chain))

        return out


# Must be global to be compliant with multi-threading
def log_posterior_model(theta, engine, param_view) -> float:
    """
    Calculates the natural log of the posterior probability.

    The posterior probability is the probability of the parameters, `theta`,
    given the evidence X denoted by p(theta | X).

    Parameters
    ----------
    theta : np.ndarray of float, with length of `params.fitting`
        The MCMC sampled values.

    engine : ampy.modeling.engine.ModelingEngine

    param_view : ampy.core.params.ParameterView

    Returns
    -------
    float
        The natural log of the posterior.
    """
    if np.isfinite(lp := log_prior_model(theta, param_view)):
        ll = log_likelihood_model(theta, engine, param_view)

        if np.isfinite(ll):
            return lp + ll

    return -np.inf


# Must be global to be compliant with multi-threading
def log_prior_model(theta, params) -> float:
    """
    Evaluates the natural log of the priors.

    Parameters
    ----------
    theta : np.ndarray of float, with length of `params.fitting`
        The sampled MCMC parameter values.

    params : Parameters
        MCMC parameter container.

    Returns
    -------
    float
        The log of the evaluated priors.
    """
    lp = 0.0

    for i, p in enumerate(params.fitting):
        lp += p.prior.evaluate(theta[i])

        if np.isinf(lp):
            return -np.inf

    return lp


# Must be global to be compliant with multi-threading
def log_likelihood_model(theta, engine, param_view):
    """
    Calculates the natural log of the likelihood.

    Parameters
    ----------
    theta : np.ndarray of float
        The MCMC sampled values.

    engine : ``ampy.modeling.engine.ModelingEngine``
        The MCMC Modeling engine.

    param_view : ``ampy.core.params.ParameterView``
        The parameter viewer used to map the MCMC samples to dicts.

    Returns
    -------
    float or -np.inf
        The log of the likelihood if the parameters were valid. Else, -np.inf.
    """
    # Map array to a dict of plugin params
    params = param_view.samples_to_dict(theta)

    # Model the observed afterglow
    try:
        modeled = engine(param_view.modeling_plugins, params)
    except ValueError as e:
        print(f"[DEBUG] Likelihood model raised ValueError: {e}")
        print(f"[DEBUG] theta = {theta}")
        return -np.inf

    for plugin in param_view.inference_plugins:
        ll = -0.5 * plugin(modeled, engine.observation, params.get(f"{plugin.name}"))

        if not np.isfinite(ll):
            print(f"[DEBUG] Likelihood returned {ll} | theta = {theta}")

    return ll

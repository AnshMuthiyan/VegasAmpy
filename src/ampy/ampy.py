import pathlib
import tomllib

import json
import numpy as np

from ampy.core.obs import Observation
from ampy.core.params import ParameterView
from ampy.core.structs import StatusType, requires_complete
from ampy.inference.engine import InferenceEngine
from ampy.modeling.plugins import load_plugins
from ampy.modeling.engine import ModelingEngine
from ampy.products import plotting
from ampy.products import utils


class AMPy:
    """
    High-level interface for Afterglow Modeling in Python (AMPy).

    Parameters
    ----------
    inference_engine : ampy.inference.engine.InferenceEngine
        The engine to be used for inference.
    """
    def __init__(self, inference_engine):
        self.inference_engine = inference_engine
        self._status = StatusType.READY

    @classmethod
    def from_registry(cls, obs, path):
        """
        Load the modeling engine from a registry file.

        Parameters
        ----------
        obs : str or ``pathlib.Path`` or ``ampy.core.obs.Observation``
            The input observation. If a filepath is provided, the observation
            is loaded using ``ampy.core.obs.Observation.load(obs)``.

        path : str or ``pathlib.Path``
            The path to the registry TOML file.

        Returns
        -------
        ampy.ampy.AMPy
        """
        if isinstance(obs, (str, pathlib.Path)):
            obs = Observation.load(obs)

        # Load the registry config
        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Load the plugins
        plugins = load_plugins(
            data['plugins'], base_dir=pathlib.Path(path).parent, obs=obs
        )

        # Create the parameter mapper
        param_view = ParameterView.from_plugins(plugins, obs=obs)

        return cls(InferenceEngine(ModelingEngine(obs), param_view))

    @property
    def status(self):
        return self._status

    @property
    def modeling_engine(self):
        return self.inference_engine.modeling_engine

    def get(self, attr):
        return getattr(self, f"get_{attr}")

    def get_observation(self):
        return self.modeling_engine.observation

    def get_sampler(self):
        return self.inference_engine.sampler

    def get_plugins(self):
        return self.inference_engine.param_view.plugins

    @requires_complete
    def get_best_params(self):
        """
        Returns the most likely samples as a dictionary.

        Returns
        -------
        dict
            The samples with the highest log probability.
        """
        return self.inference_engine.param_view.samples_to_dict(
            self.get_best_samples()
        )

    @requires_complete
    def get_best_samples(self, return_index=False):
        """
        Returns the samples with the highest log probability.

        Parameters
        ----------
        return_index : bool, optional, default=False
            Should the index of the samples in the flattened chain be returned?
            If `True`, returns a tuple of (int, np.ndarray). Else, it returns
            only the array of samples.

        Returns
        -------
        np.ndarray of float or tuple of [int, np.ndarray]
            The samples with the highest log probability.
        """
        if self.status != StatusType.COMPLETE:
            raise RuntimeError("AMPy has not run an inference yet!")

        idx = np.nanargmax(self.get_sampler().get_log_prob(flat=True))
        samples = self.get_sampler().get_chain(flat=True)[idx]

        return (idx, samples) if return_index else samples

    @requires_complete
    def get_random_samples(self, nsamps, return_index=False):
        """
        Randomly draws `nsamps` sets of samples from the chain.

        Parameters
        ----------
        nsamps : int
            The number of samples to draw.

        return_index : bool, optional, default=False
            Should the index of the samples in the flattened chain be returned?
            If `True`, returns a tuple of (int, np.ndarray). Else, it returns
            only the array of samples.

        Returns
        -------
        np.ndarray
            The randomly drawn sets of sampled values.
        """
        chain = self.inference_engine.sampler.get_chain(flat=True)
        indices = np.random.randint(len(chain), size=nsamps)

        return (indices, chain[indices]) if return_index else chain[indices]

    def run_mcmc(
        self, nwalkers, iterations, burn=0, sampler='ensemble',
        workers=None, ntemps=None, sampler_kw=None, run_kw=None,
        resume=False, path=None
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
            The max number of workers (cores) to use.

        ntemps : int, optional, default=None
            The number of temperatures for tempered sampling.

        sampler_kw : dict, optional
            Any kwargs to pass to the sampler.

        run_kw : dict, optional
            Any kwargs to pass to the `run_mcmc` method.

        resume : bool, optional, default=False
            Resume from a previous run?

        path : str, optional
            The path to save the sampler contents. If using the ensemble
            sampler, it is recommended to use `emcee.backends.HDFBackend`
            instead of providing `path`. When using `path`, the state is only
            saved after the run has completed. If the run stops before
            completing, then the sampler state is lost.
        """
        self._status = StatusType.READY

        self.inference_engine.run(
            nwalkers, iterations, burn, sampler,workers,
            ntemps, sampler_kw, run_kw, resume
        )
        self._status = StatusType.COMPLETE

        if (sampler_kw or {}).get('backend') is None and path:
            self.inference_engine.sampler.save(path)

        return self.get_best_params()

    @requires_complete
    def generate_products(self, output_dir):
        """
        Generates and saves the standard data products.

        Saves the lightcurve, spectral plot, density profile, corner plot,
        and the summary file in the ``output_dir``.

        Parameters
        ----------
        output_dir : str or ``pathlib.Path``
            The directory to save the data products.
        """
        if isinstance(output_dir, str):
            output_dir = pathlib.Path(output_dir)

        self.summary(output_dir / "report.json")
        self.light_curve(path=output_dir / "light_curve.pdf")
        try:
            self.light_curve(spread = 1, sigma= 1, path=output_dir / "light_curve_pretty.pdf")
        except:
            print("Pretty light curve failed to generate.")
            pass
        self.spectral_plot(path=output_dir / "spectral_plot.pdf")
        self.density_profile(path=output_dir / "density_profile.pdf")
        self.corner_plot(path=output_dir / "corner.pdf")

    @requires_complete
    def light_curve(self, spread=None, sigma=None, path=None):
        """
        Generates the light curve using the best parameters.

        Parameters
        ----------
        spread : dict, optional

        sigma : bool, optional
            If truthy, compute a 16th–84th percentile uncertainty band from
            the top 100 posterior samples by log-probability.

        path : str or ``pathlib.Path``, optional
            The path to save the light curve.

        Returns
        -------
        matplotlib.axes.Axes
            The light curve axes.
        """
        samples = log_probs = param_view = None
        if sigma:
            samples = self.get_sampler().get_chain(flat=True)
            log_probs = self.get_sampler().get_log_prob(flat=True)
            param_view = self.inference_engine.param_view

        fig, ax = plotting.generate_light_curve(
            self.get_observation(), self.get_plugins(), self.get_best_params(),
            spread=spread,
            samples=samples,
            log_probs=log_probs,
            param_view=param_view
        )

        if path is not None:
            utils.save_plot_unique(path)

        return fig, ax
    
    @requires_complete
    def pretty_light_curve(self, spread=None, simga =None, path=None):
        """Generates the light curve using the best parameters with a spacing for legibility."""

        fig, ax = plotting.generate_light_curve(
            self.get_observation(), self.get_plugins(), self.get_best_params(),
            spread={'afterglow_flux': 0.5}
        )
        if path is not None:
            utils.save_plot_unique(path)

        return fig, ax

    @requires_complete
    def density_profile(self, ndata=100, path=None):
        """
        Generates the density profile (density vs. radius) plot.

        Parameters
        ----------
        ndata : int, optional, default=100
            The number of data points to generate for each band.

        path : str or ``pathlib.Path``, optional
            The path to save the light curve.

        Returns
        -------
        tuple
            matplotlib.pyplot.Figure and iterable of matplotlib.axes.Axes
        """
        fig, ax =  plotting.generate_density_profile_plot(
            self.get_observation(), self.get_random_samples(ndata),
            self.inference_engine.param_view, self.get_best_samples(),
            ndata=ndata
        )

        if path is not None:
            utils.save_plot_unique(path)

        return fig, ax

    @requires_complete
    def spectral_plot(self, ndata=100, path=None):
        """
        Generates the spectral plot (spectral breaks and spectral indices).

        Parameters
        ----------
        ndata : int, optional, default=100
            The number of data points to generate for each band.

        path : str or ``pathlib.Path``, optional
            The path to save the light curve.

        Returns
        -------
        tuple
            matplotlib.pyplot.Figure and iterable of matplotlib.axes.Axes
        """
        fig, ax = plotting.generate_spectral_plot(
            self.get_observation(), self.get_random_samples(ndata),
            self.inference_engine.param_view, self.get_best_samples(),
            ndata=ndata
        )

        if path is not None:
            utils.save_plot_unique(path)

        return fig, ax

    @requires_complete
    def corner_plot(self, plugins=None, path=None):
        """
        Generates the corner plot.

        Parameters
        ----------
        plugins : iterable of str, optional
            The names of the plugins to plot. If not provided, then all
            parameters from all plugins are plotted onto the figure.

        path : str or pathlib.Path, optional
            The path to save the figure.

        Returns
        -------
        matplotlib.pyplot.Figure
        """
        fig = plotting.plot_corner(
            self.inference_engine.sampler.get_chain(flat=True),
            self.inference_engine.param_view, plugins=plugins
        )

        if path is not None:
            utils.save_plot_unique(path)

        return fig

    @requires_complete
    def summary(self, path=None):
        """
        Generates a summary dict.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            The path to save the file.

        Returns
        -------
        dict
        """
        # Add in the plugin parameters
        out = self.get_best_params()

        # Add in the inference information
        out['inference'] = self.inference_engine.summary()

        # Add in the inference parameters
        out['inference']['params'] = []

        params = np.concatenate([
            self.inference_engine.param_view.fitting,
            self.inference_engine.param_view.fixed
        ])

        for i, param in enumerate(params):
            out['inference']['params'].append(param.serialize())

        if path is not None:
            with open(path, "w") as f:
                json.dump(out, f, indent=4)

        return out

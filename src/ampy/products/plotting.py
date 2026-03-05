from collections import defaultdict
from inspect import isfunction

import arviz as az
import corner
import numpy as np
from matplotlib import pyplot as plt

import ampy.core.utils as utils2
from ampy.core.structs import DataType
from ampy.defaults import SpectralPlotDefaults, CornerPlotDefaults
from ampy.defaults import DensityProfileDefaults, BandColorMap
from ampy.modeling.engine import ModelingEngine
from ampy.modeling.models.base import MassP
from ampy.modeling.plugins import CalibrationPlugin
from ampy.products import utils


# <editor-fold desc="Statistics">
def plot_corner(chain, param_view, plugins=None, fig=None):
    """

    Parameters
    ----------
    chain: np.ndarray of float
        The flattened MCMC chain.

    param_view : ampy.core.params.ParameterView
        The parameter viewer used to map the samples to dicts.

    plugins : iterable of str, optional
        The names of the plugins to plot. If not provided, then all parameters
        from all plugins are plotted onto the figure.

    fig : matplotlib.pyplot.Figure, optional
        An existing figure to plot onto.

    Returns
    -------
    matplotlib.pyplot.Figure
    """
    plugins = plugins or param_view.plugin_names

    pos, labels = [], []
    for i, param in enumerate(param_view.fitting):

        # Store the position and name of the parameter
        if param.plugin in plugins:
            labels.append(utils.latex(param.name))
            pos.append(i)

    if fig is None:
        fig = corner.corner(
            chain[:, pos], labels=labels, **CornerPlotDefaults().kwargs()
        )

    return fig


def plot_trace(params, path=None, sampler=None, chain=None) -> None:
    """
    Generate the trace plots using arviz.

    Parameters
    ----------
    params : Parameters
        The model Parameters object.

    path : str or pathlib.Path
        The directory to save the results.

    sampler : emcee.EnsembleSampler, optional
        The MCMC sampler. Required if `chain=None`.

    chain : np.ndarray, optional
        The MCMC chain. Required if `sampler=None`.

    Returns
    -------
    matplotlib.axes:
        The figure axes
    """
    if chain is sampler is None:
        raise ValueError('`chain`  or `sampler` must be specified')

    # Use arviz style
    az.style.use("arviz-darkgrid")

    # Create the production inference data object
    var_names = [p.name for p in params.fitting]

    if sampler is not None:
        # Create inference data from EnsembleSampler
        inf_data = az.from_emcee(sampler, var_names=var_names)

    else:
        # Create inference data from the chain
        chain = np.transpose(chain, axes=(1, 0, 2))
        burn = {name: chain[..., i] for i, name in enumerate(var_names)}
        inf_data = az.from_dict(posterior=burn)

    # Save summary statistics to a csv
    # az.summary(inf_data).to_csv(out_dir / "summary.csv")

    # Plot the trace plot
    axes = az.plot_trace(inf_data)
    if path:
        utils.save_plot_unique(path)

    return axes
# </editor-fold>


# <editor-fold desc="Light Curve">
def default_light_curve_axes(plot_kwargs=None):
    """
    Creates the default plt.axes for the light curve.

    Parameters
    ----------
    plot_kwargs : dict, optional
        Any keyword arguments accepted by matplotlib.pyplot.subplots. If any
        kwargs overlap with the default kwargs, then the provided kwargs will
        overwrite the defaults.

    Returns
    -------
    matplotlib.axes.Axes
        The light curve axes.
    """
    kwargs = {'figsize': (8, 10)} | (plot_kwargs or {})

    fig, ax = plt.subplots(**kwargs)

    # Add the main axes options
    ax.set_ylabel('Flux Density [mJy]')
    ax.set_xlabel('Time Since Trigger [days]')
    ax.set_yscale('log'), ax.set_xscale('log')
    ax.tick_params(axis='x', top=False, bottom=True, reset=True)
    ax.grid(alpha=0.3)

    # Add secondary x-axis in units of seconds
    ax2 = ax.secondary_xaxis(
        'top', functions=(utils2.days_to_sec, utils2.sec_to_days)
    )
    ax2.set_xlabel("Time Since Trigger [seconds]", labelpad=10)
    ax2.xaxis.set_ticks_position('none')
    ax2.tick_params(axis='x', top=True, bottom=False)

    return fig, ax


def generate_light_curve(
    obs,
    plugins,
    params,
    ndata=100,
    spread=None,
    ax=None,
    sigma=None,
    ll_kwargs=None,
    plot_kwargs=None
):
    """
    Generates the light curve.

    Parameters
    ----------
    obs : ampy.core.obs.Observation
        The observational data.

    plugins : iterable of ampy.modeling.plugins.<plugin>
        The plugins used to model the observation.

    params : dict
        The plugin parameters.

    ndata : int, optional, default=100
        The number of data points to generate for each band.

    spread : dict[str, float], optional
        Multiplicative values to spread the bands.

    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If not provided, a new one will be created
        using the `.default_light_curve_axes` function.

    ll_kwargs : dict[str, Any], optional
        Any keyword arguments accepted by matplotlib.pyplot.loglog. If any
        kwargs overlap with the default kwargs, then the provided kwargs will
        overwrite the defaults.

    plot_kwargs : dict[str, Any], optional
        Any keyword arguments accepted by matplotlib.pyplot.subplots. If any
        kwargs overlap with the default kwargs, then the provided kwargs will
        overwrite the defaults.

    Returns
    -------
    matplotlib.axes.Axes
        The light curve axes.
    """
    # Best-fit flux for each observed band
    flux = generate_model(obs, plugins, params, ndata)

    # Log-spaced time array spanning the observation epoch
    times = np.geomspace(
        obs.epoch()[0], obs.epoch()[1], num=ndata, dtype=float
    )

    # If MCMC samples are provided, compute the 16th–84th percentile
    # uncertainty band for each band by stacking models across samples
    sigma_bands = None
    if sigma is not None:
        stacked = defaultdict(list)
        for s in sigma:
            try:
                for band, f in generate_model(obs, plugins, s, ndata).items():
                    stacked[band].append(f)
            except Exception:
                continue
        sigma_bands = {
            band: np.percentile(fluxes, [16, 84], axis=0)
            for band, fluxes in stacked.items()
        }

    # Plot model lines (and uncertainty bands if sigma was provided)
    fig, ax = plot_lightcurve_model(
        flux, times, spread, ax, sigma_bands, ll_kwargs, plot_kwargs
    )

    # Overlay the observed data points with per-band error bars
    ax = plot_observation(obs, spread=spread, ax=ax)

    ax.set_xlim(times.min(), times.max())

    # Label the y-axis to indicate arbitrary scaling when spread is applied
    if spread is not None:
        ax.set_ylabel('Arbitrarily Scaled Flux Density')

    ax.legend(
        loc='lower left', ncols=2, columnspacing=0.25,
        handletextpad=0.25, fontsize=12
    )

    return fig, ax


def generate_model(obs, plugins, params, ndata=100) -> dict:
    """
    Generates the light curve flux organized by the observed bands.

    Parameters
    ----------
    obs : ampy.core.obs.Observation
        The observational data.

    plugins : iterable of ampy.modeling.plugins.<plugin>
        The plugins used to model the observation.

    params : dict
        The plugin parameters.

    ndata : int, optional, default=100
        The number of data points to generate for each band.

    Returns
    -------
    dict[str, np.ndarray of float]
        The modeled flux values [mJy] organized by band name.
    """
    # Create a mock observation with `ndata` points for each band
    mock_obs = obs.mock(
        subset=('Band',), ndata=ndata, exclude=['spectral index']
    )

    # The plugins may have already been used to model another observation.
    # If the plugins were optimized to that observation, then the optimized
    # data needs to be updated for the mock observation. Do that here using
    # copies of the provided plugins.
    plugins_copy = utils.copy_plugins(plugins, mock_obs)

    # Remove the calibration plugin since we do not apply offsets here.
    for plugin in plugins_copy:
        if isinstance(plugin, CalibrationPlugin):
            plugins_copy.remove(plugin)
            break

    # Model the mock observation
    modeled = ModelingEngine(mock_obs)(plugins_copy, params)

    # Organize the flux data by bands
    bands = mock_obs.bands(mask=mock_obs.flux_loc)

    res = {}
    for band, loc in bands.items():

        for i in loc:
            # Convert to mJy for consistency when plotting
            if mock_obs.data[i].value.unit != 'mJy':
                modeled[i] = utils.convert_value_to_mjy(
                    mock_obs.data[i], modeled[i]
                )

        res[band] = modeled[loc]

    return res


def plot_lightcurve_model(
    flux,
    times,
    spread=None,
    ax=None,
    sigma=None,
    ll_kwargs=None,
    plot_kwargs=None
):
    """
    Generate the light curve flux axes.

    Parameters
    ----------
    flux : dict[str, np.ndarray of float]
        The light curve plox to be plotted. Organized by band names.

    times : np.ndarray
        The times to plot the light curve. Assume that all the fluxes are to be
        plotted with the same time array.

    spread : dict[str, float], optional
        Multiplicative values to spread the bands.

    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If not provided, a new one will be created
        using the `.default_light_curve_axes` function.

    ll_kwargs : dict[str, Any], optional
        Any keyword arguments accepted by matplotlib.pyplot.loglog. If any
        kwargs overlap with the default kwargs, then the provided kwargs will
        overwrite the defaults.

    plot_kwargs : dict[str, Any], optional
        Any keyword arguments accepted by matplotlib.pyplot.subplots. If any
        kwargs overlap with the default kwargs, then the provided kwargs will
        overwrite the defaults.

    Returns
    -------
    matplotlib.axes.Axes
    """
    colors = BandColorMap()
    fig, default_ax = default_light_curve_axes(plot_kwargs)

    ax = ax or default_ax
    ll_kwargs = {'linewidth': 1.0} | (ll_kwargs or {})

    for band, flux in flux.items():
        # Apply per-band multiplicative offset for readability (default 1 = no scaling)
        scale = spread.get(band, 1.0) if spread is not None else 1.0

        # Best-fit model line
        ax.loglog(times, flux * scale, '--', color=colors.get(band), **ll_kwargs)

        # 16th–84th percentile shading from MCMC samples (pretty plot only)
        if sigma is not None and band in sigma:
            p16, p84 = sigma[band]
            ax.fill_between(
                times, p16 * scale, p84 * scale,
                color=colors.get(band), alpha=0.2
            )

    return fig, ax


def plot_observation(obs, spread=None, ax=None):
    """

    Parameters
    ----------
    obs : ampy.core.obs.Observation
        The observational data.

    spread : dict[str, float], optional
        Multiplicative values to spread the bands.

    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If not provided, a new one will be created
        using the `.default_light_curve_axes` function.

    Returns
    -------
    matplotlib.axes.Axes
    """
    colors = BandColorMap()

    for band, loc in obs.bands().items():
        e, x, y = [], [], []

        for i, data in enumerate(obs.data[loc]):

            if data.type == DataType.SPECTRAL_INDEX:
                continue

            if data.type == DataType.INTEGRATED_FLUX:
                data = data.to_spectral()

            # Store the flux, time, and uncertainty
            x.append(data.time.to_value('d'))
            y.append(data.value.to_value('mJy'))
            e.append(data.uncertainty.center.to_value('mJy'))

            if spread and band in spread:
                y[i] *= spread[band]

        if e and x and y:
            ax.errorbar(
                x, y, yerr=e, fmt='.', markersize=3.0, elinewidth=0.5,
                label=band, color=colors.get(band)
            )

    return ax


# </editor-fold>


# <editor-fold desc="Density Profile">
def _n_to_rho(n):
    """ number density to mass density """
    return MassP * n


def _rho_to_n(rho):
    """ mass density to number density """
    return rho / MassP


def _cm_to_pc(val):
    """ centimeters to parsecs """
    return val * 3.2407792896664E-19


def _pc_to_cm(val):
    """ parsec to centimeters """
    return val / 3.2407792896664E-19


def default_density_axes(plot_kwargs=None):
    """
    Creates the default plt.axes for the density profile.

    Parameters
    ----------
    plot_kwargs : dict, optional
        Any keyword arguments accepted by matplotlib.pyplot.subplots. If any
        kwargs overlap with the default kwargs, then the provided kwargs will
        overwrite the defaults.

    Returns
    -------
    matplotlib.axes.Axes
        The light curve axes.
    """
    kwargs = {'figsize': (8, 8)} | (plot_kwargs or {})

    fig, ax = plt.subplots(**kwargs)

    # Add the main axes options
    ax.set_ylabel(r'n [cm$^{-3}$]', fontsize=14)
    ax.set_xlabel('Radius [cm]', fontsize=14)
    ax.set_yscale('log'), ax.set_xscale('log')
    ax.tick_params(axis='x', top=False, right=False, bottom=True, reset=True)
    ax.grid(alpha=0.3)

    # Add secondary y-axis in units of g cm-3
    ax2 = ax.secondary_yaxis(
        'right', functions=(_n_to_rho, _rho_to_n)
    )
    ax2.set_ylabel(r'$\rho$ [g cm$^{-3}$]', fontsize=14, labelpad=10)
    ax2.tick_params(right=True, labelright=True)

    # Add secondary x-axis in units of parsecs
    ax3 = ax.secondary_xaxis(
        'top', functions=(_cm_to_pc, _pc_to_cm)
    )
    ax3.set_xlabel("Radius [Parsecs]", labelpad=10)
    ax3.tick_params(top=True, labeltop=True)

    return fig, ax


def generate_density_profile_plot(
    obs, samples, param_view, best_sample, ndata=100, plot_kwargs=None
):
    """
    Generates the density profile (density vs. radius) plot.

    Parameters
    ----------
    obs : ampy.core.obs.Observation
        The observational data.

    samples : iterable of np.ndarray of float
        The samples to generate the spectral break curves.

    param_view : ampy.core.params.ParameterView
        The parameter viewer used to map the samples to dicts.

    best_sample : np.ndarray of float
        The most likely set of samples.

    ndata : int, optional, default=100
        The number of data points to generate for each band.

    plot_kwargs : dict[str, Any], optional
        Any keyword arguments accepted by matplotlib.pyplot.subplots. If any
        kwargs overlap with the default kwargs, then the provided kwargs will
        overwrite the defaults.

    Returns
    -------
    tuple
        matplotlib.pyplot.Figure and iterable of matplotlib.axes.Axes
    """
    fig, ax = default_density_axes(plot_kwargs)

    # Assume that the first plugin is the base afterglow model
    model = param_view.plugins[0].model
    plugin = param_view.plugins[0].name

    # Model and plot the density profiles for a series of samples
    ax = plot_density_profile(
        model, plugin, obs, samples, param_view, ax, ndata,
        **DensityProfileDefaults().distribution.kwargs()
    )

    # Model and plot the density profiles for the best samples
    ax = plot_density_profile(
        model, plugin, obs, [best_sample], param_view, ax, ndata,
        **DensityProfileDefaults().best.kwargs()
    )

    # Legend has to be added after the artists are added
    ax.legend(loc='best')

    return fig, ax


def plot_density_profile(
    model, plugin, obs, samples, param_view, ax, ndata=100, **kwargs
):
    """
    Generates the density profile curves.

    Parameters
    ----------
    model : callable
        The model containing `density_profile` and `blast_radius`.

    plugin : str
        The name of the base afterglow plugin.

    obs : ampy.core.obs.Observation
        The observational data.

    samples : iterable of np.ndarray of float
        The samples to generate the spectral break curves.

    param_view : ampy.core.params.ParameterView
        The parameter viewer used to map the samples to dicts.

    ax : matplotlib.pyplot.Axes
        The axes on which to plot.

    ndata : int, optional, default=100
        The number of data points to generate for each band.

    Returns
    -------
    matplotlib.axes.Axes
    """
    times = np.geomspace(
        obs.epoch()[0], obs.epoch()[1], num=ndata, dtype=float
    )

    for sample in samples:
        params = param_view.samples_to_dict(sample)

        if isfunction(model):
            model_obj = model(**params[plugin]['eval'])
        else:
            model_obj = model(**params[plugin]['init'])

        r = model_obj.blast_radius(times)
        n = model_obj.density(times)

        ax.plot(r, n, **kwargs)

    return ax
# </editor-fold>


# <editor-fold desc="Spectral Plot">
def _create_legend(fig, axes, rows=3):
    """
    Creates the main, detached legend above the figure.

    Parameters
    ----------
    fig : matplotlib.pyplot.Figure

    axes : iterable of matplotlib.axes.Axes

    rows : int, optional
        The number of rows to use for the legend.

    Returns
    -------
    tuple
        matplotlib.pyplot.Figure and iterable of matplotlib.axes.Axes
    """
    h, l = axes[0].get_legend_handles_labels()

    # Determine the number of columns based on how many
    # rows are requested and how many labels there are.
    n_col = int((len(l) + rows - 1) // rows)

    # Tell matplotlib to only use 90% of the height for
    # the figure so that there is room for the legend.
    fig.legend(h, l, ncol=n_col, **SpectralPlotDefaults().legend.kwargs())
    fig.tight_layout(rect=[0, 0, 1, 0.9])  # noqa

    return fig, axes


def default_spectral_axes(kwargs=None):
    """
    Creates the default plt.axes for the spectral plot.

    Parameters
    ----------
    kwargs : dict, optional
        Any keyword arguments accepted by matplotlib.pyplot.subplots. If any
        kwargs overlap with the default kwargs, then the provided kwargs will
        overwrite the defaults.

    Returns
    -------
    tuple
        matplotlib.pyplot.Figure and iterable of matplotlib.axes.Axes
    """
    sub_kwargs = SpectralPlotDefaults().layout.kwargs() | (kwargs or {})

    # Create a two plot figure (top: breaks, bottom: indices)
    fig, axes = plt.subplots(nrows=2, ncols=1, **sub_kwargs)

    # Remove the space between the axes
    fig.subplots_adjust(hspace=0)

    # Add the main axes options
    axes[0].grid(alpha=0.3, axis='x')
    axes[0].set_ylabel('Frequency [Hz]', fontsize=14)
    axes[1].set_xlabel('Time Since Trigger [days]', fontsize=14)
    axes[1].set_ylabel('Spectral Index', fontsize=14)

    return fig, axes


def generate_spectral_plot(
    obs, samples, param_view, best_sample, ndata=100, plot_kwargs=None
):
    """
    Generates the spectral plot.

    Parameters
    ----------
    obs : ampy.core.obs.Observation
        The observational data.

    samples : iterable of np.ndarray of float
        The samples to generate the spectral break curves.

    param_view : ampy.core.params.ParameterView
        The parameter viewer used to map the samples to dicts.

    best_sample : np.ndarray of float


    ndata : int, optional, default=100
        The number of data points to generate for each band.

    plot_kwargs : dict[str, Any], optional
        Any keyword arguments accepted by matplotlib.pyplot.subplots. If any
        kwargs overlap with the default kwargs, then the provided kwargs will
        overwrite the defaults.

    Returns
    -------
    tuple
        matplotlib.pyplot.Figure and iterable of matplotlib.axes.Axes
    """
    # Model the spectral breaks for a series of samples
    fig, axes = generate_spectral_break_fig(
        obs, samples, param_view, best_sample, ndata, plot_kwargs
    )

    # Plot the observational photometry and indices
    axes = plot_spectral_observation(obs, axes)

    # Model the spectral indices and plot them
    axes = generate_spectral_index_axes(
        obs, samples, param_view, axes, ndata
    )

    # Add the main legend
    fig, axes = _create_legend(fig, axes, rows=3)

    return fig, axes


def generate_spectral_break_fig(
    obs, samples, param_view, best_sample, ndata=100, plot_kwargs=None
):
    """
    Generates the spectral break curve figure and axes.

    Parameters
    ----------
    obs : ampy.core.obs.Observation
        The observational data.

    samples : iterable of np.ndarray of float
        The samples to generate the spectral break curves.

    param_view : ampy.core.params.ParameterView
        The parameter viewer used to map the samples to dicts.

    best_sample : np.ndarray of float


    ndata : int, optional, default=100
        The number of data points to generate for each band.

    plot_kwargs : dict[str, Any], optional
        Any keyword arguments accepted by matplotlib.pyplot.subplots. If any
        kwargs overlap with the default kwargs, then the provided kwargs will
        overwrite the defaults.

    Returns
    -------
    tuple
        matplotlib.pyplot.Figure and iterable of matplotlib.axes.Axes
    """
    # Model the spectral breaks for a series of samples
    times, nu_a, nu_m, nu_c = generate_spectral_breaks(
        obs, samples, param_view, ndata
    )

    # Plot the series of spectral breaks
    fig, axes = plot_spectral_breaks(
        times, nu_a, nu_m, nu_c, plot_kwargs=plot_kwargs, nu_kwargs={
        'nu_a': {'alpha': 0.1}, 'nu_m': {'alpha': 0.1}, 'nu_c': {'alpha': 0.1}
        }
    )

    # Overplot the most likely spectral breaks
    times, nu_a, nu_m, nu_c = generate_spectral_breaks(
        obs, [best_sample], param_view, ndata
    )

    _, axes = plot_spectral_breaks(
        times, nu_a, nu_m, nu_c, axes=axes, plot_kwargs=plot_kwargs,
        nu_kwargs={
            'nu_a': {'label': r'$\nu_a$'}, 'nu_m': {'label': r'$\nu_m$'},
            'nu_c': {'label': r'$\nu_c$'}
        }
    )

    return fig, axes


def plot_spectral_breaks(
    times, nu_a=None, nu_m=None, nu_c=None,
    axes=None, plot_kwargs=None, nu_kwargs=None
):
    """
    Plots the spectral frequency breaks.

    Parameters
    ----------
    times : np.ndarray
        The time series to plot.

    nu_a : iterable of np.ndarray of float, optional
        The self-absorption frequencies [Hz].

    nu_m : iterable of np.ndarray of float, optional
        The synchrotron frequencies [Hz].

    nu_c : iterable of np.ndarray of float, optional
        The cooling frequencies [Hz].

    axes : matplotlib.axes.Axes, optional
        The axes to plot onto.

    plot_kwargs : dict[str, Any], optional

    nu_kwargs: dict[str, Any], optional

    Returns
    -------
    tuple
        matplotlib.pyplot.Figure and iterable of matplotlib.axes.Axes
    """
    fig = None

    if axes is None:
        fig, axes = default_spectral_axes(plot_kwargs)

    spectral_breaks = {
        'nu_a': nu_a if nu_a is not None else [],
        'nu_m': nu_m if nu_a is not None else [],
        'nu_c': nu_c if nu_a is not None else []
    }

    # Get the default color mapping
    colors = SpectralPlotDefaults().colors.mapping()

    for name, spectral_break in spectral_breaks.items():
        for curve in spectral_break:
            axes[0].loglog(
                times, curve, color=colors[name], **nu_kwargs[name]
            )

    return fig, axes


def generate_spectral_breaks(obs, samples, param_view, ndata=100):
    """
    Models the spectral break curves.

    Parameters
    ----------
    obs : ampy.core.obs.Observation
        The observational data.

    samples : iterable of np.ndarray of float
        The samples to generate the spectral break curves.

    param_view : ampy.core.params.ParameterView
        The parameter viewer used to map the samples to dicts.

    ndata : int, optional, default=100
        The number of data points to generate for each band.

    Returns
    -------
    tuple of size 4
        The times, nu_a, nu_m, and nu_c.
    """
    # Assume that the first plugin is the base afterglow model
    model = param_view.plugins[0].model

    times = np.geomspace(
        obs.epoch()[0], obs.epoch()[1], num=ndata, dtype=float
    )

    nu_a, nu_m, nu_c = [], [], []
    for sample in samples:
        params = param_view.samples_to_dict(sample)

        model_obj = model(**params['afterglow_flux']['init'])

        if hasattr(model_obj, 'nu_m'):
            nu_m.append(model_obj.nu_m(times))

        if hasattr(model_obj, 'nu_c'):
            nu_c.append(model_obj.nu_c(times))

        if hasattr(model_obj, 'nu_a'):
            nu_a.append(model_obj.nu_a(times, nu_m=nu_m, nu_c=nu_c))

    return times, nu_a, nu_m, nu_c


def generate_spectral_index_axes(
    obs, samples, param_view, axes=None, ndata=100
):
    """
    Generates the spectral index axes.

    Parameters
    ----------
    obs : ampy.core.obs.Observation
        The observational data.

    samples : iterable of np.ndarray of float
        The samples to generate the spectral break curves.

    param_view : ampy.core.params.ParameterView
        The parameter viewer used to map the samples to dicts.

    axes : iterable of matplotlib.axes.Axes, optional
        The axes to add the spectral indices.

    ndata : int, optional, default=100
        The number of data points to generate for each band.

    Returns
    -------
    iterable of matplotlib.axes.Axes
    """
    if axes is None:
        _, axes = default_spectral_axes()

    times = np.geomspace(
        obs.epoch()[0], obs.epoch()[1], num=ndata, dtype=float
    )

    # Model the spectral indices and plot them
    modeled = generate_spectral_indices(obs, samples, param_view, ndata)

    for i, indices in enumerate(modeled):
        axes[1].plot(
            times, indices, alpha=0.2, color='royalblue',
            label='AMPy' if i == 0 else None
        )

    # Truncate the edges and add the legend for axes[1]
    axes[1].set_xlim(times.min(), times.max())
    axes[1].legend(loc='best')

    return axes


def generate_spectral_indices(obs, samples, param_view, ndata=100):
    """
    Model the spectral indices.

    Parameters
    ----------
    obs : ampy.core.obs.Observation
        The observational data.

    samples : iterable of np.ndarray of float
        The plugins used to model the observation.

    param_view : ampy.core.params.ParameterView
        The parameter viewer used to map the samples to dicts.

    ndata : int, optional, default=100
        The number of data points to generate for each band.

    Returns
    -------
    np.ndarray of np.ndarray of float
        The modeled spectral indices.
    """
    plugins = param_view.plugins

    # Create a mock observation with `ndata` index points
    mock_obs = obs.mock(
        subset=('CalGroup', 'Band'), ndata=ndata,
        exclude=['spectral flux', 'integrated flux']
    )

    # The plugins may have already been used to model another observation.
    # If the plugins were optimized to that observation, then the optimized
    # data needs to be updated for the mock observation. Do that here using
    # copies of the provided plugins. Since we are only modeling the indices,
    # we only need the first (base afterglow) plugin.
    plugins_copy = utils.copy_plugins(plugins=[plugins[0]], obs=mock_obs)

    indices = []
    # Model the spectral indices
    for sample in samples:
        params = param_view.samples_to_dict(sample)

        try:
            # Prevent a bad sample set from causing a crash
            index = ModelingEngine(mock_obs)(plugins_copy, params)
            indices.append(index)
        except ValueError as e:
            print("Modeled an invalid index from a random sample.")

    return indices


def plot_spectral_observation(obs, axes):
    """
    Add the photometry and indices to the spectral break plot.

    Parameters
    ----------
    obs : ampy.core.obs.Observation
        The observational data.

    axes : iterable of matplotlib.axes.Axes
        The spectral break axes.

    Returns
    -------
    iterable of matplotlib.axes.Axes
    """
    # Plot the photometry as frequency vs. time
    axes = plot_observation_photometry(obs, axes)

    # Plot the spectral indices vs. time
    axes = plot_observation_indices(obs, axes)

    return axes


def plot_observation_photometry(obs, axes):
    """
    Add the photometry points to the spectral break plot.

    Parameters
    ----------
    obs : ampy.core.obs.Observation
        The observational data.

    axes : iterable of matplotlib.axes.Axes
        The spectral break axes.

    Returns
    -------
    iterable of matplotlib.axes.Axes
    """
    colors = BandColorMap()
    inc_bands = obs.bands()

    for band, loc in inc_bands.items():
        x, y = [], []

        for data in obs.data[loc]:

            if data.type == DataType.SPECTRAL_INDEX:
                continue

            # Store the flux, time, and uncertainty
            x.append(data.time.to_value('d'))
            y.append(data.frequency.to_value('Hz'))

        if x and y:
            axes[0].scatter(
                x, y, label=band, color=colors.get(band), marker='.'
            )

    return axes


def plot_observation_indices(obs, axes):
    """
    Add the spectral indices to the spectral break plot.

    Parameters
    ----------
    obs : ampy.core.obs.Observation
        The observational data.

    axes : iterable of matplotlib.axes.Axes
        The spectral break axes.

    Returns
    -------
    iterable of matplotlib.axes.Axes
    """
    indices = obs.data[obs.sindex_loc]

    for i, index in enumerate(indices):
        x_err, y_err = None, None

        # Optional: Determine the time uncertainty
        if index.time_lower is not None and index.time_upper is not None:
            x_err = (
                (index.time_lower.to_value('d'),),
                (index.time_upper.to_value('d'),)
            )

        # Optional: Determine the value uncertainty
        if index.uncertainty is not None:
            y_err = (
                (index.uncertainty.lower.value,),
                (index.uncertainty.upper.value,)
            )

        axes[1].errorbar(
            x=index.time.to_value('d'), y=index.value.value,
            yerr=y_err, xerr=x_err, label='XRT' if i == 0 else None,
            **SpectralPlotDefaults().observed.kwargs()
        )

    return axes
# </editor-fold>


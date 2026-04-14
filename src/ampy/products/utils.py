import copy
import os
import pathlib

import astropy.units as u
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = False


def convert_value_to_mjy(data, value):
    """
    Converts a value to MJY. Intended for internal use.

    In the plotting routine, there is often a modeled float value in cgs units
    that corresponds to an observation data point. Instead of rewriting the
    conversion logic, use the corresponding observation data point that has the
    logic already built in.

    Parameters
    ----------
    data : astropy.units.Quantity
        The quantity containing the `value` information.

    value : float
        The value to convert using the `data` quantity.

    Returns
    -------
    float
        The converted value in mJy.
    """
    if data.value.unit == 'mJy':
        return data.value.value

    quantity = u.Quantity(value, unit=data.value.unit)

    return (quantity / data.int_range.width).to_value('mJy')


def copy_plugins(plugins, obs=None) -> list:
    """
    Deep-copies the list of plugins into a new list.

    Parameters
    ----------
    plugins : iterable of ampy.modeling.plugins.<plugin>
        The plugins to deepcopy

    obs : ampy.core.obs.Observation, optional
        The observational data. If provided, the plugins are optimized.

    Returns
    -------
    list
        The deep-copied plugins.
    """
    plugins_copy = []

    for plugin in plugins:
        plugin_copy = copy.deepcopy(plugin)

        if obs:
            plugin_copy.optimize(obs)

        plugins_copy.append(plugin_copy)

    return plugins_copy


def save_plot_unique(path, **kw):
    """
    Save a matplotlib plot to disk, adding a suffix if the file exists.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to save to.

    kw : int, optional
        Any keyword arguments to pass to the `plt.savefig` function.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)

    # Remove the `.` from the suffix
    ext = path.suffix.replace('.', '')

    i = 0
    # Try to save the plot, incrementing a number
    # until the filename becomes unique.
    while True:
        filename = (
            f"{path.stem}.{ext}" if i == 0 else f"{path.stem}_{i}.{ext}"
        )

        filepath = os.path.join(path.parent, filename)

        if not os.path.exists(filepath):
            plt.savefig(filepath, **kw)
            return filepath
        i += 1


# TODO: Make this general for plugins.
def latex(key, plugin=None) -> str:
    """
    Returns LaTeX label for the provided fitting parameter.

    Parameters
    ----------
    key : str
        The parameter name

    plugin : ampy.modeling.plugins.<plugin>, optional
        The associated plugin.

    Returns
    -------
    str
        The LaTeX formatted `key` or just `key`.
    """
    if '_host' in key:
        return r'$log_{10}$(' + f'{key.split('_')[0]}' + r'$_{host}$)'

    if '_offset' in key:
        return r'$\delta_{' + f'{key.split('_')[0]}' r'}$'

    try:
        return {
            'slop': r'$\sigma$',
            'slop_uvot': r'$\sigma_{uvot}$',
            'slop_other': r'$\sigma_{other}$',

            # Jetsimpy
            'Eiso': r'$log_{10}E_{iso}$',
            'lf': r'$log_{10}\Gamma$',
            'theta_c': r'$\theta_c$',
            'theta_v': r'$\theta_v$',
            'A': r'$log_{10}A$',
            'n0': r'$log_{10}n_0$',

            # Stratified Fireball Model
            'n0t': r'$log_{10}n_{0, t}$',
            'rt': r'$log_{10}R_t$',
            'k1': r'$k_{pre}$',
            'k2': r'$k_{post}$',
            'sn': r'$s_n$',
            'sni': r'$s_n^{-1}$',
            'sj': r'$s_j$',
            'sji': r'$s_j^{-1}$',
            'tj': r'$log_{10}t_j$',

            # Generic Fireball Model
            'lf0': r'$log_{10}\Gamma_0$',
            'E52': r'$log_{10}E_{52}$',
            'eps_e': r'$log_{10}\epsilon_e$',
            'eps_b': r'$log_{10}\epsilon_B$',
            'rv_milky_way': r'$log_{10}R_{v}^{MW}$',
            'ebv_source_frame': r'$E(B-V)_{sf}$',
            'ebv_milky_way': r'$E(B-V)_{MW}$',
            'n017': r'$log_{10}n_{0, 17}$',
        }[key]
    except KeyError:
        return key

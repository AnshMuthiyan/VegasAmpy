import importlib
import warnings

import numpy as np
from dust_extinction.parameter_averages import CCM89
warnings.filterwarnings('ignore', message='x has no units', module='dust_extinction')

from ampy.modeling.models.base import DAY2SEC, MassP, SpectralFluxModel, \
    IntegratedFluxModel, SpectralIndexModel
from ampy.modeling.models.base import BlastWaveModel, ObservedSpectrumModel
from ampy.modeling.models.base import RadiationModel, BlastWaveModel2

from ampy import core


class ImportFromStringError(ImportError):
    pass


def import_from_string(path: str):
    """"""
    if "." not in path:
        raise ImportFromStringError(
            f"Invalid import path '{path}'. "
            "Expected a dotted path like 'package.module.ClassName'."
        )

    module_path, attr = path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportFromStringError(
            f"Could not import module '{module_path}' "
            f"from '{path}'."
        ) from e

    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ImportFromStringError(
            f"Module '{module_path}' has no attribute '{attr}'."
        ) from e


# <editor-fold desc="Dust Extinction Models">
def source_dust_extinction_model(obs, params) -> np.ndarray:
    """
    Calculates the source-frame CCM89 dust extinction values.

    Parameters
    ----------
    obs : `ampy.core.obs.Observation`
        The input observation containing the extinction positions stored as an
        array.

    params : dict[str, float]
        The `Rv` and `Ebv` values stored as a dictionary.

    Returns
    -------
    np.ndarray of float
        The fractional source dust extinction values.
    """
    res = np.ones(len(obs.extinguishable))

    # Source-frame wavenumbers
    x = obs.as_arrays.wave_numbers[obs.extinguishable] * (1.0 + params['z'])

    # Fractional dust extinction values
    frac_ext = CCM89(Rv=params['Rv']).extinguish(x, Ebv=params['Ebv'])

    res[obs.extinguishable] = res[obs.extinguishable] * frac_ext

    return res


def galactic_dust_extinction_model(obs, params) -> np.ndarray:
    """
    Calculates the Milky Way CCM89 dust extinction values.

    Parameters
    ----------
    obs : `ampy.core.obs.Observation`
        The input observation containing the extinction positions stored as an
        array.

    params : dict[str, float]
        The `Rv` and `Ebv` values stored as a dictionary.

    Returns
    -------
    np.ndarray of float
        The fractional Milky Way dust extinction values.
    """
    res = np.ones(len(obs.extinguishable))

    # Observer-frame wavenumbers
    x = obs.as_arrays.wave_numbers[obs.extinguishable]

    # Fractional dust extinction values
    frac_ext = CCM89(Rv=params['Rv']).extinguish(x, Ebv=params['Ebv'])

    res[obs.extinguishable] = res[obs.extinguishable] * frac_ext

    return res
# </editor-fold>


# <editor-fold desc="Calibration Models">
def calibration_offset_model(obs, params) -> np.ndarray:
    """
    Calculates the calibration (systematic) offset values.

    Parameters
    ----------
    obs : `ampy.core.obs.Observation`
        The input observation containing the calibration offset positions
        stored as a dictionary and sorted by the calibration group name.

    params : dict[str, float]
        The calibration offset values stored as a dictionary and sorted by the
        calibration group name.

    Returns
    -------
    np.ndarray of float
        The multiplicative calibration offsets.
    """
    offsets = np.ones(len(obs.flux_loc))

    for name, offset in params.items():
        offsets[obs.offsets[name]] = 10.0 ** -(0.4 * offset)

    return offsets
# </editor-fold>


# <editor-fold desc="Host Galaxy Models">
def host_galaxy_model(obs, params) -> np.ndarray:
    """
    Calculates the host galaxy contribution as an additive offset.

    Parameters
    ----------
    obs : `ampy.core.obs.Observation`
        The input observation containing the host galaxy contribution positions
        stored as a dictionary and sorted by the host galaxy group name.

    params : dict[str, float]
        The host galaxy values stored as a dictionary and sorted by the host
        galaxy group name.

    Returns
    -------
    np.ndarray of float
        The additive host galaxy contributions.
    """
    hosts = np.zeros(len(obs.flux_loc))

    for name, host in params.items():
        hosts[obs.hosts[name]] = host

    return hosts
# </editor-fold>


#<editor-fold desc="Inference Models">
def chi_squared_model(modeled, obs, params):
    """
    Calculates the combined chi-squared between the modeled and observational
    data for both the flux and spectral indices.

    The flux chi-squared calculation uses a so-called chi-squared effective
    which uses a slop parameter. Spectral indices use the standard
    chi-squared formulation.

    Parameters
    ----------
    modeled : np.ndarray of float
        The modeled values.

    obs : ampy.core.obs.Observation
        The observational data.

    params : dict
        The params for the chi-squared model. Must contain 'eval'.

    Returns
    -------
    float
        The effective chi-squared value.
    """
    slop = params['eval']

    # Flux and indices are treated separately
    flux_mask = obs.flux_loc
    index_mask = obs.sindex_loc

    # Format the slop
    if isinstance(slop, dict):
        slop_fmt = np.empty(obs.length)

        for name, val in slop.items():
            slop_fmt[obs.slops[name]] = val

        slop = slop_fmt[flux_mask]

    # Handle flux and indices the same
    if slop is None:
        return core.utils.chi_squared(
            modeled, obs.as_arrays.values, obs.as_arrays.errors
        )

    # Chi-squared for flux (uses slop)
    cs_flux = core.utils.chi_squared(
        modeled[flux_mask],
        obs.as_arrays.values[flux_mask],
        obs.as_arrays.errors[flux_mask],
        slop  # noqa
    )

    # Chi-squared for spectral indices (does not use slop)
    if not index_mask.any():
        return cs_flux

    cs_indices = core.utils.chi_squared(
        modeled[index_mask],
        obs.as_arrays.values[index_mask],
        obs.as_arrays.errors[index_mask]
    )

    # return combined chi-squared
    return cs_flux + cs_indices
#</editor-fold>


# <editor-fold desc="Afterglow Models">
class BasePowerLawModel:
    """
    Base model. Not intended for direct use.

    Implement the ultra-relativistic shock moving into an external medium
    with density rho = rho_0 * R^-k.

    Parameters
    ----------
    E52 : float
        The explosion energy normalized to 1e52 ergs.

    dL28 : float
        The luminosity distance to the event normalized to 1e28 cm.
        Requiring the distance to be provided in addition to the
        redshift prevents the need to assume a cosmology here.

    p : float
        The electron energy index. Must be > 2.

    eps_b : float
        The fraction of thermal energy in the magnetic field.
        Must be in the range [0, 1].

    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons. Must be in the range [0, 1].

    z : float
        The redshift of the event.

    hmf : float
        The hydrogen mass fraction. Must be in the range [0, 1].
        0 indicates hydrogen depleted. 1 indicates hydrogen rich.

    tj : float, optional
        The jet break time [d].

    sj, sji : float, optional
        The jet break smoothing factor. ``sji`` is the inverse
        smoothing factor. Useful for changing MCMC basis.

    use_sa : bool, optional, default=True
        Should self-absorption be modeled?

    References
    ----------
    [1] Broadband view of blast wave physics: A study
        of gamma-ray burst afterglows
    """

    # noinspection PyPep8Naming
    def __init__(
        self, E52, p, eps_b, eps_e, z, dL28, hmf,
        lf0=None, tj=None, sj=None, sji=None, use_sa=True
    ):
        # Intrinsic properties
        self.E52 = E52
        self.p = p
        self.eps_b = eps_b
        self.eps_e = eps_e
        self.hmf = hmf
        self.lf0 = lf0

        # Extrinsic properties
        self.dL28 = dL28
        self.z = z

        # Jet properties
        self.tj = tj
        self.sj = (sj or 1 / sji) if (sj or sji) else None

        self.use_sa = use_sa

    def __repr__(self):
        """ Human-readable representation. """
        return f'{self.__class__.__name__}(E={self.E}, p={self.p}, .., z={self.z})'

    # noinspection PyPep8Naming
    @property
    def E(self):
        """ Returns the initial explosion energy [erg]. """
        return 1e52 * self.E52

    # noinspection PyPep8Naming
    @property
    def dL(self):
        """ Returns the luminosity distance [cm]. """
        return 1e28 * self.dL28

    @property
    def is_valid(self) -> bool:
        """ Whether the model is parameters are valid. """
        # Smoothing parameters are unstable around 0
        if self.sj is not None and abs(self.sj) <= 0.1:
            return False
        return ((self.eps_b + self.eps_e) < 1.0) and (self.p > 2.0)

    @property
    def radiative(self):
        """ Should radiative evolution be considered? """
        return self.eps_e > 0.4 and self.lf0 is not None

    def spectrum(self, *args, **kwargs):
        """ Placeholder. """
        raise NotImplementedError('spectrum is not implemented.')

    def spectral_flux(self, t, nu, fts=False):
        """
        Calculates the spectral fluxes at time(s) ``t`` for
        the frequencies ``nu``.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer-frame time(s) [d].

        nu : float or np.ndarray of float
            The average band frequencies [Hz].

        fts : bool, optional, default=False
            Is there a fast-to-slow cooling transition?

        Returns
        -------
        float np.ndarray of float
            The modeled spectral flux [mJy].
        """
        return SpectralFluxModel(**self.spectrum(t)).evaluate(nu, fts)

    def integrated_flux(self, t, lower, upper, fts=False):
        """
        Calculates the integrated fluxes at time(s) ``t`` for
        the integration bounds, ``lower`` and ``upper``.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer-frame time(s) [d].

        lower, upper : float or np.ndarray of float
            The integration bounds [Hz].

        fts : bool, optional, default=False
            Is there a fast-to-slow cooling transition?

        Returns
        -------
        float or np.ndarray of float
            The modeled spectral flux [erg cm-2 s-1].
        """
        return IntegratedFluxModel(**self.spectrum(t)).evaluate(
            lower, upper, fts
        )

    def spectral_index(self, t, lower, upper, fts=False):
        """
        Calculates the spectral indices at time(s) ``t`` for
        the integration bounds, ``lower`` and ``upper``.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer-frame time(s) [d].

        lower, upper : float or np.ndarray of float
            The integration bounds [Hz].

        fts : bool, optional, default=False
            Is there a fast-to-slow cooling transition?

        Returns
        -------
        float np.ndarray of float
            The modeled spectral index.
        """
        return SpectralIndexModel(**self.spectrum(t)).evaluate(
            lower, upper, fts
        )


class PowerLawModel(BasePowerLawModel):
    """
    A fully analytic description of an ultra-relativistic shock moving into an
    external medium with density:
    :math:`\\rho = \\rho_0 R^{-k}`.

    Parameters
    ----------
    E52 : float
        The explosion energy normalized to 1e52 ergs.

    p : float
        The electron energy index.

    eps_b : float
        The fraction of thermal energy in the magnetic field.
        Must be in the range [0, 1].

    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons. Must be in the range [0, 1].

    z : float
        The redshift of the event.

    dL28 : float
        The luminosity distance to the event [1e28 cm]. Requiring the distance
        to be provided in addition to the redshift prevents the need to assume
        a cosmology here.

    n017 : float
        The density normalization normalized to 1e17 cm.

    k : float
        The density power-law index.

    hmf : float
        The hydrogen mass fraction. Must be in the range [0, 1].
        0 indicates hydrogen depleted. 1 indicates hydrogen rich.

    lf0 : float, optional, default=None
        The initial Lorentz factor.

    tj : float, optional, default=None
        The jet break observer-frame time [d].

    sj : float, optional, default=None
        The jet break smoothing factor. Required if ``tj != None``.

    use_sa : bool, optional, default=True
        Should self-absorption be modeled?
    """
    # noinspection PyPep8Naming
    def __init__(
        self, E52, p, eps_b, eps_e, z, dL28, n017, k, hmf,
        lf0=None, tj=None, sj=None, sji=None, use_sa=True
    ):
        super().__init__(
            E52, p, eps_b, eps_e, z, dL28, hmf, lf0, tj, sj, sji, use_sa
        )

        self.ref_radius = 1.0e17  # [cm]
        self.n017 = n017
        self.k = k

        # Blast wave model
        if lf0 is not None:
            self.blast = BlastWaveModel2(self.E, self.lf0, self.n0, self.k)

        # Radiation model
        self.radiation = RadiationModel(
            self.n0, self.k, self.p, self.eps_b, self.eps_e, self.dL, self.z,
            self.hmf
        )

    def __call__(self, obs, params):
        """ Behaves the same as `.model(obs, params)` """
        return self.model(obs, params)

    @property
    def n0(self):
        """ Returns the number density normalization. """
        return self.n017 * self.ref_radius ** self.k

    @property
    def rho0(self):
        """ Returns the mass density normalization normalized to 1e17 cm. """
        return MassP * self.n017

    @property
    def rho(self):
        """ Returns the mass density [g cm-3]. """
        return MassP * self.n017 * self.ref_radius ** self.k

    def model(self, obs, params=None):
        """
        Models the observational data, `obs``.

        Parameters
        ----------
        obs : ampy.core.obs.Observation
            The observational data.

        params : dict, optional
            Not used. For compatibility with AMPy inference.

        Returns
        -------
        np.ndarray of float
            The modeled observational data.
        """
        if not self.is_valid:
            return np.array([np.nan])

        # Return the modeled observational data
        return ObservedSpectrumModel(
            **self.spectrum(obs.times()), arrays=obs.as_arrays
        ).model()

    def density(self, t):
        """
        Returns the number density at all times `t`.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer times [d].

        Returns
        -------
        np.ndarray of float
            The number density [cm-3] at all blast wave radii.
        """
        return self.n017 * (self.blast_radius(t) / self.ref_radius) ** -self.k

    def blast_radius(self, t):
        """
        Calculates the radius traversed by the blast wave during time `t` in a
        stratified medium defined by the power-law index `k` density at the
        reference radius.

        Parameters
        ----------
        t : float or np.ndarray
            The observer times [d].

        Returns
        -------
        float or np.ndarray
            The radii traversed by the blast wave [cm].
        """
        bwm = BlastWaveModel(self.E52, self.n017, self.k, ref=self.ref_radius)

        return bwm.shock_radius(
            self.z, t, bwm.decel_time(self.lf0 or 300.0) / DAY2SEC
        )

    def spectrum(self, t):
        """
        Returns the characteristics that define the GRB spectrum.

        Parameters
        ----------
        t : float or np.ndarray
            The observer-frame time(s) [d].

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        t = np.atleast_1d(t)

        # Default to adiabatic evolution
        spec = self.spectrum_adiabatic(t)

        # Radiative evolution
        if self.radiative:
            rad = np.logical_and(spec['nu_m'] > spec['nu_c'], spec['nu_a'] < spec['nu_m'])

            # Is there a radiative solution?
            if not rad.any():
                return spec

            # Radiative evolution spectrum
            spec_rad = self.spectrum_radiative(t)

            # Is there still a radiative solution?
            if not (spec_rad['nu_m'] > spec_rad['nu_c']).any():
                return spec

            # When does radiative end and adiabatic begin?
            t_trans = self.radiation.rad_to_ad_time(self.E / self.lf0) / DAY2SEC

            # Will the transition affect the light curve?
            if not ((t.min() / 100.0) < t_trans < (t.max() * 100.0)):
                return spec_rad

            # Recalculate adiabatic functions using diminished energy
            spec_ad = self.spectrum_adiabatic(t, post_rad=True)

            return self.radiation.rad_to_ad_smooth(t, t_trans, spec_rad, spec_ad)

        return spec

    def spectrum_adiabatic(self, t, post_rad=False):
        """
        Returns the characteristics that define the GRB spectrum assuming
        adiabatic evolution.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer-frame time(s) [d].

        post_rad : bool, optional, default=False
            Is the adiabatic evolution following a radiative evolution? If so,
            accounts for the energy loss during radiative evolution.

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        nrg = self.E

        if post_rad:
            # When does radiative end and adiabatic begin?
            t_trans = self.radiation.rad_to_ad_time(self.E / self.lf0)

            # What is the energy after radiative loss?
            nrg = self.E * self.blast.energy_loss(t_trans / (1 + self.z))

        # Optional jet break parameters
        tj, sj = (self.tj or -1.0), (self.sj or 1.0)

        return {
            'p': self.p, 'k': self.k,
            'f_peak': self.radiation.peak_flux(nrg, t, tj=tj, sj=sj),
            'nu_c': self.radiation.cooling_frequency(nrg, t, tj=tj, sj=sj),
            'nu_a': self.radiation.absorption_frequency(nrg, t, tj=tj, sj=sj),
            'nu_m': self.radiation.synchrotron_frequency(nrg, t, tj=tj, sj=sj)
        }

    def spectrum_radiative(self, t):
        """
        Returns the characteristics that define the GRB spectrum assuming
        radiative evolution.

        Parameters
        ----------
        t : float or np.ndarray
            The observer-frame time(s) [d].

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        nrg = self.E / self.lf0

        return {
            'p': self.p, 'k': self.k,
            'f_peak': self.radiation.peak_flux(nrg, t, adiabatic=False),
            'nu_c': self.radiation.cooling_frequency(nrg, t, adiabatic=False),
            'nu_a': self.radiation.absorption_frequency(nrg, t, adiabatic=False),
            'nu_m': self.radiation.synchrotron_frequency(nrg, t, adiabatic=False)
        }

    def f_peak(self, t):
        """
        Calculates the peak flux in the case of an ultra-relativistic shock
        moving into an external medium with density:
        :math:`\\rho = \\rho_0 R^{-k}`.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        Returns
        -------
        float or np.ndarray of float
            The peak flux value(s) [mJy] at time(s) `t`.
        """
        return self.spectrum(t)['f_peak']

    def nu_c(self, t):
        """
        Calculates the cooling frequency in the case of an ultra-relativistic
        shock moving into an external medium with density:
        :math:`\\rho = \\rho_0 R^{-k}`.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        Returns
        -------
        float or np.ndarray of float
            The cooling frequency value(s) [Hz] at time(s) `t`.
        """
        return self.spectrum(t)['nu_c']

    def nu_m(self, t):
        """
        Calculates the synchrotron frequency in the case of an
        ultra-relativistic shock moving into an external medium with density:
        :math:`\\rho = \\rho_0 R^{-k}`.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        Returns
        -------
        float or np.ndarray of float
            The synchrotron frequency value(s) [Hz] at time(s) `t`.
        """
        return self.spectrum(t)['nu_m']

    def nu_a(self, t, nu_m=None, nu_c=None):
        """
        Calculates the self-absorption frequency.

        The self-absorption frequency has a circular definition.
        For example, to calculate nu_a, you must first know how
        nu_a relates to nu_m and nu_c, but nu_a isn't known
        because it needs to be known before it can be known >:)

        This solution is weak, but the self-absorption frequency
        is calculated for every case (except nu_a > both nu_c and
        nu_m, not supported). The result is a combined array where
        the self-absorptions are compared to the synchrotron and
        cooling frequencies.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        nu_m : float or np.ndarray of float, optional
            The synchrotron frequencies [Hz] at time `t`.

        nu_c : float or np.ndarray of float, optional
            The cooling frequencies [Hz] at time `t`.

        Returns
        -------
        float or np.ndarray of float
            The self-absorption frequency value(s) [Hz] at time(s) `t`.
        """
        return self.spectrum(t)['nu_a']


class SBPowerLawModel(BasePowerLawModel):
    """
    A fully analytic description of an ultra-relativistic shock moving into
    an external stratified medium with density:
    :math:`\\rho = \\rho_0 R^{-k}`.
    Where `k` transitions from one asymptotic value to another.

    Parameters
    ----------
    E52 : float
        The explosion energy normalized to 1e52 ergs.

    p : float
        The electron energy index (dimensionless).

    eps_b : float
        The fraction of thermal energy in the magnetic field.
        Must be in the range [0, 1].

    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons. Must be in the range [0, 1].

    z : float
        The redshift to the event.

    dL28 : float
        The luminosity distance to the event [1e28 cm]. Requiring the distance
        to be provided in addition to the redshift prevents the need to assume
        a cosmology here.

    sn, sni : float, optional
        The density smoothing factor. ``sni`` is the inverse of the smoothing
        factor. Useful for changing the fitting basis in MCMC.

    k1 : float
        The density power-law index before the transition.

    k2 : float
        The density power-law index after the transition.

    hmf : float
        The hydrogen mass fraction.

    tj : float, optional, default=None
        The jet break observer-frame time [d].

    sj, sji : float, optional, default=None
        The jet break smoothing factor. `sji` is the inverse of the smoothing
        factor. Useful for changing the fitting basis in MCMC.
        Required if ``tj != None``.

    use_sa : bool, optional, default=True
        Should self-absorption be modeled?
    """
    # noinspection PyPep8Naming
    def __init__(
        self, E52, p, eps_b, eps_e, z, dL28, n0t, rt, hmf,
        k1=None, k2=None, lf0=None, tj=None, sj=None, sji=None,
        sn=None, sni=None, k1i=None, k2i=None, use_sa=True
    ):
        super().__init__(E52, p, eps_b, eps_e, z, dL28, hmf, lf0, tj, sj, sji, use_sa)

        if sn is None and sni is None:
            raise ValueError("Must specify either sn or sni.")

        # Medium
        self.k1 = k1 if k1 is not None else 1 / k1i
        self.k2 = k2 if k2 is not None else 1 / k2i

        self.sn = (sn or 1 / sni) if (sn or sni) else None
        self.rt = rt
        self.n0t = n0t

    def __call__(self, obs, params):
        """ Behaves the same as `.model(obs, params)` """
        return self.model(obs, params)

    @property
    def ref_radius(self):
        """ Returns the reference radius [cm]."""
        return self.rt

    @property
    def is_valid(self) -> bool:
        """ Whether the model is parameters are valid. """
        if self.sn < 0 and self.k1 < self.k2:
            return False

        if self.sn > 0 and self.k1 > self.k2:
            return False

        return super().is_valid and abs(self.sn) > 0.1

    def smooth(self, t):
        """
        Empirically smooths the number density normalizations and the power-law
        indices over the observer times `t`.

        Parameters
        ----------
        t : np.ndarray
            The observer-frame time(s) [d].

        Returns
        -------
        tuple of np.ndarray of float
            The smoothed number density normalizations [cm-3] and the smoothed
            density power-law indices.
        """
        # Rename for convenience
        s, k1, k2 = self.sn, self.k1, self.k2

        r = self.blast_radius(t)
        x = r / self.rt

        # Calculate the effective number densities
        n = self.n0t * (2 ** (1 / s)) * (
            x ** (k1 * s) + x ** (k2 * s)
        ) ** -(1 / s)

        # Calculate the effective density power-law indices
        k_eff_num = k1 * x ** (k1 * s) + k2 * x ** (k2 * s)
        k_eff_den = x ** (k1 * s) + x ** (k2 * s)
        k_eff = k_eff_num / k_eff_den

        # Number density normalized to `rt`
        n0 = n * (r / self.rt) ** k_eff

        return n0, k_eff

    def density(self, t):
        """
        Returns the number density at all times `t`.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer times [d].

        Returns
        -------
        np.ndarray of float
            The number density [cm-3] at all blast wave radii.
        """
        # Rename for convenience
        s, k1, k2 = self.sn, self.k1, self.k2

        r = self.blast_radius(t)
        x = r / self.rt

        # Calculate the effective number densities
        return self.n0t * (2 ** (1 / s)) * (
            x ** (k1 * s) + x ** (k2 * s)) ** -(1 / s)

    def blast_radius(self, t):
        """
        Calculates the radius traversed by the blast wave during time `t` in a
        stratified medium defined by the power-law indices `k1` and `k2`, and
        the radius and density at the transition, `n0t` and `rt`.

        Parameters
        ----------
        t : float or np.ndarray
            The observer-frame time(s) [d].

        Returns
        -------
        float or np.ndarray
            The radii traversed by the blast wave [cm].
        """
        bwm1 = BlastWaveModel(self.E52, self.n0t, self.k1, ref=self.rt)
        bwm2 = BlastWaveModel(self.E52, self.n0t, self.k2, ref=self.rt)
        t_decel = bwm1.decel_time(self.lf0 or 300.0) / DAY2SEC

        r1 = bwm1.shock_radius(self.z, t, t_decel)
        r2 = bwm2.shock_radius(self.z, t, t_decel)

        # Rename for convenience
        s, k1, k2 = self.sn, self.k1, self.k2
        x1, x2 = r1 / self.rt, r2 / self.rt

        return (2 ** (1 / s)) * self.rt * (x1 ** -s + x2 ** -s) ** -(1 / s)

    def model(self, obs, params=None):
        """
        Models the observational data, `obs`.

        Parameters
        ----------
        obs : ampy.core.obs.Observation
            The observational data.

        params : dict, optional
            For compatability with ``ampy.modeling.engin.ModelingEngine``.

        Returns
        -------
        np.ndarray of float
            The modeled observational data.
        """
        if not self.is_valid:
            return np.array([np.nan])

        # return the modeled smoothed, unextinguished flux
        return ObservedSpectrumModel(
            **self.spectrum(obs.times()), arrays=obs.as_arrays
        ).model()

    def spectrum(self, t, n=None, k=None):
        r"""
        Compute synchrotron spectral characteristics of the GRB afterglow.

        This method evaluates the characteristic quantities that define the
        broadband synchrotron spectrum at observer-frame time(s) ``t`` for the
        current afterglow model parameters. The returned quantities correspond
        to the standard broken power-law synchrotron spectrum:

        * :math:`F_{\nu,\mathrm{peak}}` — peak flux normalization
        * :math:`\nu_a` — self-absorption frequency
        * :math:`\nu_m` — characteristic synchrotron frequency
        * :math:`\nu_c` — cooling frequency
        * :math:`p` — electron energy index
        * :math:`k` — circumburst density power-law index

        The spectrum is computed assuming adiabatic evolution by default.
        If ``self.radiative`` is enabled and a radiative solution exists
        (:math:`\nu_m > \nu_c`), the method transitions from radiative to
        adiabatic evolution at the radiative–adiabatic transition time and
        smoothly joins the two regimes.

        Parameters
        ----------
        t : float or array_like of float
            Observer-frame time(s) in days.
        n : float or array_like of float, optional
            Effective number density normalization at the reference radius
            ``self.ref_radius`` in cm⁻³. If not provided, the density evolution
            is obtained from :meth:`smooth`.
        k : float or array_like of float, optional
            Effective circumburst density power-law index such that
            :math:`\rho(R) \propto R^{-k}`. If not provided, values are obtained
            from :meth:`smooth`.

        Returns
        -------
        dict[str, numpy.ndarray]
            Dictionary containing the spectral characteristics evaluated at
            each input time:

            ``"f_peak"`` : ndarray
                Peak flux normalization :math:`F_{\nu,\mathrm{peak}}` [mJy].
            ``"nu_a"`` : ndarray
                Synchrotron self-absorption frequency :math:`\nu_a` [Hz].
            ``"nu_m"`` : ndarray
                Characteristic synchrotron frequency :math:`\nu_m` [Hz].
            ``"nu_c"`` : ndarray
                Cooling frequency :math:`\nu_c` [Hz].
            ``"p"`` : float
                Electron energy power-law index.
            ``"k"`` : ndarray
                Effective density power-law index.

        Notes
        -----
        * If ``n`` and ``k`` are not supplied, they are interpolated from the
          smooth density profile using :meth:`smooth`.
        * When radiative evolution is enabled, the method determines whether a
          radiative solution exists (:math:`\nu_m > \nu_c`). If so, it computes
          the radiative spectrum, determines the radiative–adiabatic transition
          time, adjusts the blast-wave energy for radiative losses, and smoothly
          transitions to the adiabatic solution.
        * All spectral quantities are returned in the observer frame.

        Examples
        --------
        Evaluate spectral characteristics at 1 day:

        >>> spec = self.spectrum(1.0)
        >>> spec["nu_m"], spec["nu_c"]

        Evaluate over a time grid:

        >>> t = np.logspace(-2, 2, 100)
        >>> spec = self.spectrum(t)
        >>> spec["f_peak"].shape
        (100,)
        """
        t = np.atleast_1d(t)

        if n is None or k is None:
            n, k = self.smooth(t)

        # Number density normalization [cm(k-3)]
        n = n * self.ref_radius ** k

        radiation = RadiationModel(
            n, k, self.p, self.eps_b, self.eps_e, self.dL, self.z, self.hmf
        )

        # Default to the adiabatic spectrum
        spec = self.spectrum_adiabatic(radiation, self.E, t)

        # Should we consider radiative evolution?
        if self.radiative:

            # OK, but is there actually a radiative solution?
            if not (spec['nu_m'] > spec['nu_c']).any():
                return spec

            spec_rad = self.spectrum_radiative(radiation, t)

            # Is there still a radiative solution?
            if not (spec_rad['nu_m'] > spec_rad['nu_c']).any():
                return spec

            # When does radiative end and adiabatic begin?
            t_trans = radiation.rad_to_ad_time(
                self.E / self.lf0, t_obs=t, nu_m=spec_rad['nu_m'], nu_c=spec_rad['nu_c']
            )

            if t_trans is None:
                return spec_rad

            # What is the energy after radiative loss?
            k_t = np.interp(t_trans / DAY2SEC, t, k)
            n_t = np.interp(t_trans / DAY2SEC, t, n)

            nrg = self.E * BlastWaveModel2(self.E, self.lf0, n_t, k_t).energy_loss(
                t_trans / (1 + self.z)
            )

            # Recalculate adiabatic functions using diminished energy
            spec_ad = self.spectrum_adiabatic(radiation, nrg, t)

            # Smooth the spectral functions
            return radiation.rad_to_ad_smooth(t, t_trans / DAY2SEC, spec_rad, spec_ad)

        return spec

    def spectrum_adiabatic(self, radiation, E, t):
        """
        Returns the characteristics that define the GRB spectrum assuming
        adiabatic evolution.

        Parameters
        ----------
        radiation : RadiationModel
            The radiation model to use.

        E : float
            The energy [erg] at the start of the adiabatic evolution.

        t : float or np.ndarray
            The observer-frame time(s) [d].

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        # Optional jet break parameters
        tj, sj = (self.tj or -1.0), (self.sj or 1.0)

        return {
            'p': self.p, 'k': radiation.k,
            'f_peak': radiation.peak_flux(E, t, tj=tj, sj=sj),
            'nu_c': radiation.cooling_frequency(E, t, tj=tj, sj=sj),
            'nu_a': radiation.absorption_frequency(E, t, tj=tj, sj=sj),
            'nu_m': radiation.synchrotron_frequency(E, t, tj=tj, sj=sj)
        }

    def spectrum_radiative(self, radiation, t):
        """
        Returns the characteristics that define the GRB spectrum assuming
        radiative evolution.

        Parameters
        ----------
        radiation : RadiationModel
            The radiation model to use.

        t : float or np.ndarray
            The observer-frame time(s) [d].

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        nrg = self.E / self.lf0

        return {
            'p': radiation.p, 'k': radiation.k,
            'f_peak': radiation.peak_flux(nrg, t, adiabatic=False),
            'nu_c': radiation.cooling_frequency(nrg, t, adiabatic=False),
            'nu_a': radiation.absorption_frequency(nrg, t, adiabatic=False),
            'nu_m': radiation.synchrotron_frequency(nrg, t, adiabatic=False)
        }

    def f_peak(self, t, n=None, k=None):
        """
        Calculates the peak flux in the case of an ultra-relativistic shock
        moving into an external medium with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        n : float or np.ndarray of float, optional
            The smoothed density normalization [cm-3].

        k : float or np.ndarray of float, optional
            The density power-law indices.

        Returns
        -------
        float or np.ndarray of float
            The peak flux [mJy] at time `t` [d].
        """
        if k is None or n is None:
            n, k = self.smooth(t)

        return self.spectrum(t, n, k)['f_peak']

    def nu_c(self, t, n=None, k=None):
        """
        Calculates the cooling frequency in the case of an ultra-relativistic
        shock moving into an external medium with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        n : np.ndarray, optional
            The smoothed density normalization [cm-3].

        k : np.ndarray, optional
            The density power-law indices.

        Returns
        -------
        float or np.ndarray of float
            The cooling frequency in Hz at time `t`.
        """
        if n is None or k is None:
            n, k = self.smooth(t)

        return self.spectrum(t, n, k)['nu_c']

    def nu_m(self, t, n=None, k=None):
        """
        Calculates the synchrotron frequency in the case of an
        ultra-relativistic shock moving into an external medium with density
        rho = rho_0 * R^-k.

        Parameters
        ----------
        n : np.ndarray, optional
            The smoothed density normalization [cm-3].

        k : np.ndarray, optional
            The density power-law indices.

        t : float or np.ndarray of float
            The observer time(s) [d].

        Returns
        -------
        float or np.ndarray of float
            The synchrotron frequency [Hz] at time(s) `t`.
        """
        if k is None:
            n, k = self.smooth(t)

        return self.spectrum(t, n, k)['nu_m']

    def nu_a(self, t, n=None, k=None, nu_m=None, nu_c=None):
        """
        Calculates the self-absorption frequency.

        The self-absorption frequency has a circular definition.
        For example, to calculate nu_a, you must first know how
        nu_a relates to the nu_m and nu_c, but nu_a isn't known
        because it needs to be known before it can be known >:)

        This solution is weak, but the self-absorption frequency
        is calculated for every case (except nu_a > both nu_c and
        nu_m, not supported). The result is a combined array where
        the self-absorptions are compared to the synchrotron and
        cooling frequencies.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        n : np.ndarray of float, optional
            The effective density normalization [cm-3].

        k : np.ndarray of float, optional
            The effective density power-law indices.

        nu_m : float or np.ndarray of float, optional
            The synchrotron frequencies [Hz] at time `t`.

        nu_c : float or np.ndarray of float, optional
            The cooling frequencies [Hz] at time `t`.

        Returns
        -------
        float or np.ndarray of float
            The self-absorption frequency [Hz] at time(s) `t`.
        """
        if n is None or k is None:
            n, k = self.smooth(t)

        return self.spectrum(t, n, k)['nu_a']
# </editor-fold>

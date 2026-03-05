import numpy as np

# from jetfit.core.utils import days_to_sec

try:
    from VegasAfterglow import Model, ISM, Wind, Medium, TophatJet, GaussianJet, PowerLawJet, powerLaw
    from VegasAfterglow import Observer, Radiation
    _HAS_VEGASAFTERGLOW = True
except ImportError:
    _HAS_VEGASAFTERGLOW = False


"""
Adapter for integrating VegasAfterglow with JetFit's MCMC infrastructure.

This module wraps the VegasAfterglow numerical simulation package and 
adapts it for use with JetFit's MCMC framework.

Notes
-----
VegasAfterglow is a high-performance numerical afterglow simulator that
computes radiation from relativistic blast waves. It supports various jet
structures (tophat, Gaussian, power-law) and circumburst media (ISM, wind).

The model is computationally intensive compared to analytical models, so
multiprocessing is recommended for MCMC fitting.

References
----------
.. [1] VegasAfterglow: https://github.com/YihanWangAstro/VegasAfterglow
"""


class powerlawVegasModel:
    """
    Adapter for the VegasAfterglow numerical afterglow simulator.

    Parameters
    ----------
    E_iso52 : float
        Isotropic equivalent energy [erg] normalized to e52.

    lf0 : float
        Initial Lorentz factor (dimensionless).

    theta_c : float
        Half-opening angle of the jet core [rad].

    theta_v : float
        Viewing angle [rad].

    n_ism : float, optional
        ISM number density [cm^-3]. Use for constant density medium.

    A_star : float, optional
        Wind density parameter. Use for wind medium (rho = A_star / r^2).

    eps_e : float
        Fraction of shock energy in electrons. Must be in [0, 1].

    eps_b : float
        Fraction of shock energy in magnetic field. Must be in [0, 1].

    p : float
        Electron power-law index (dimensionless). Typically p >= 2.

    z : float
        Redshift.

    lumi_dist : float
        Luminosity distance [cm].

    jet_type : str, optional, default='tophat'
        Type of jet structure: 'tophat', 'gaussian', or 'powerlaw'.

    medium_type : str, optional, default='ism'
        Type of circumburst medium: 'ism' or 'wind'.

    k_e : float, optional
        Energy power-law index for structured jets. Default=None (tophat).

    k_g : float, optional
        Lorentz factor power-law index for structured jets. Default=None.

    References
    ----------
    .. [1] VegasAfterglow: A Numerical Code for GRB Afterglow
    """
    
    def __init__(
        self,
        E52,
        lf0,
        theta_c,
        theta_v,
        eps_e,
        eps_b,
        p,
        z,
        dL28,
        n017,  # number density at r = 10^17 cm [cm^-3] (log scale in parameters.toml)
        k,
        k_e,
        k_g,
        s,
        hmf = 0.7,     # density power-law index: n(r) = n017 * (r/r0)^(-k)
        n_ism=None,
        A_star=None,

        jet_type='powerlaw',
        medium_type='powerlawC',
        ref_radius=1.0e17,

    ):
        if not _HAS_VEGASAFTERGLOW:
            raise ImportError(
                "VegasAfterglow is not installed or failed to import. "
                "Install it from: https://github.com/YihanWangAstro/VegasAfterglow"
            )

        # NOTE: JetFit already converts "scale=log" parameters to linear
        # So we receive linear values here, NOT log10 values
        self.E_iso52 = E52 * 1e52  # E52 is linear multiplier, convert to erg
        self.lf0 = lf0             # Already linear (JetFit converted from log10)
        self.theta_c = theta_c
        self.theta_v = theta_v
        self.eps_e = eps_e         # Already linear (JetFit converted from log10)
        self.eps_b = eps_b         # Already linear (JetFit converted from log10)
        self.p = p
        self.z = z
        self.hmf = hmf
        self.lumi_dist = dL28 * 1e28  # dl28 is linear multiplier, convert to cm
        self.jet_type = jet_type
        self.medium_type = medium_type
        self.n017 = n017           # Already linear (JetFit converted from log10)
        self.k = k                 # Power-law index (linear scale)
        self.n_ism = n_ism
        self.A_star = A_star
        self.k_e = k_e
        self.k_g = k_g
        self.s = s
        self.ref_radius = ref_radius  # Reference radius [cm]


        # Initialize VegasAfterglow components
        self._setup_model()

    def _get_details(self, t_sec):
        """
        Helper to cache VegasAfterglow details for a given (t_min, t_max) range.
        """
        t_key = (float(np.min(t_sec)), float(np.max(t_sec)))
        if t_key in self._details_cache:
            return self._details_cache[t_key]
        details = self.vegas_model.details(t_key[0], t_key[1])
        self._details_cache[t_key] = details
        return details

    def days_to_sec(x):
        """ Convert days to seconds. """
        return x * 86400

    def _setup_model(self):
        """Initialize the VegasAfterglow model with current parameters."""
        self._details_cache = {}
        self._flux_cache = {}
        r0 = 1e17  # reference radius [cm]

        def power_law_medium(phi, theta, r):
            """
            Single power-law mass density profile.

            Parameters
            ----------
            phi : float or np.ndarray
                Azimuthal angle (not used in this profile)
            theta : float or np.ndarray
                Polar angle (not used in this profile)
            r : float or np.ndarray
                Radius [cm]

            Returns
            -------
            float or np.ndarray
                Mass density [g/cm^3]
            """
            # n017 is already linear (JetFit converted from log10)
            # n(r) = n017 * (r / r0)^(-k)
            n = self.n017 * (r / r0) ** (-self.k)

            # Convert number density to mass density
            # rho = m_p * n, where m_p = 1.67262192e-24 g
            X = 0.7  # Hydrogen mass fraction
            m_p = 1.67262192e-24  # proton mass in g
            rho = m_p * n * X

            return rho

        # Create medium
        if self.medium_type.lower() == 'ism':
            if self.n_ism is None:
                raise ValueError("n_ism must be specified for ISM medium")
            medium = ISM(n_ism=self.n_ism)
        elif self.medium_type.lower() == 'wind':
            if self.A_star is None:
                raise ValueError("A_star must be specified for wind medium")
            medium = Wind(A_star=self.A_star)
        elif self.medium_type.lower() == 'powerlaw':
            # Wrap the single power-law function in a Medium object
            self._medium_rho = power_law_medium
            medium = Medium(rho=power_law_medium)
        elif self.medium_type == 'powerlawC':
            # Use the powerLaw medium in C++ side of VegasAfterglow
            medium = powerLaw(n0=self.n017, k=self.k)
        else:
            raise ValueError(f"Unknown medium_type: {self.medium_type}")

        # Create jet
        if self.jet_type.lower() == 'tophat':
            jet = TophatJet(
                theta_c=self.theta_c,
                E_iso=self.E_iso52,
                Gamma0=self.lf0
            )
        elif self.jet_type.lower() == 'gaussian':
            jet = GaussianJet(
                theta_c=self.theta_c,
                E_iso=self.E_iso52,
                Gamma0=self.lf0
            )
        elif self.jet_type.lower() == 'powerlaw':
            if self.k_e is None or self.k_g is None:
                raise ValueError("k_e and k_g must be specified for powerlaw jet")
            jet = PowerLawJet(
                theta_c=self.theta_c,
                E_iso=self.E_iso52,
                Gamma0=self.lf0,
                k_e=self.k_e,
                k_g=self.k_g,
                s = self.s
            )
        else:
            raise ValueError(f"Unknown jet_type: {self.jet_type}")

        # Create observer and radiation
        observer = Observer(
            lumi_dist=self.lumi_dist,
            z=self.z,
            theta_obs=self.theta_v
        )

        radiation = Radiation(
            eps_e=self.eps_e,
            eps_B=self.eps_b,
            p=self.p
        )

        # Create the model - Note: Model expects (jet, medium, observer, fwd_rad)
        # print(f"DEBUG: Creating VegasAfterglow Model...")
        try:
            self.vegas_model = Model(jet=jet, medium=medium, observer=observer, fwd_rad=radiation)
            # print(f"DEBUG: Model created successfully!")
        except Exception as e:
            print(f"DEBUG: Model creation FAILED: {e}")
            raise

    @property
    def is_valid(self) -> bool:
        """Check if model parameters are physically valid."""
        valid = (
            (self.eps_b + self.eps_e) < 1.0 and
            self.p >= 2.0 and
            self.theta_c > 0 and
            self.lf0 > 1
        )
        # if not valid:
            # print(f"DEBUG: Model is_valid=False:")
            # print(f"  eps_b + eps_e = {self.eps_b + self.eps_e} < 1.0? {(self.eps_b + self.eps_e) < 1.0}")
            # print(f"  p = {self.p} >= 2.0? {self.p >= 2.0}")
            # print(f"  theta_c = {self.theta_c} > 0? {self.theta_c > 0}")
            # print(f"  lf0 = {self.lf0} > 1? {self.lf0 > 1}")
        return valid

    def nu_m(self, t, **kwargs):
        """
        Calculate the minimum synchrotron frequency (cooling break).
        
        This extracts nu_m from VegasAfterglow's details.
        
        Parameters
        ----------
        t : np.ndarray of float
            Observer times [days].
            
        Returns
        -------
        np.ndarray of float
            Minimum synchrotron frequency [Hz].
        """
        t_sec = self.days_to_sec(t)
        t_sec = np.atleast_1d(t_sec)
        try:
            details = self._get_details(t_sec)
            nu_m = np.asarray(details.fwd.nu_m[0,0,:]*details.fwd.Doppler[0,0,:]/(1+self.z))
            if len(nu_m) != len(t_sec):
                t_details = np.asarray(details.fwd.t_obs[0, 0, :])
                nu_m = np.interp(t_sec, t_details, nu_m)
        except Exception as e:
            raise RuntimeError(f"Failed to extract nu_m from VegasAfterglow: {e}")
        return np.atleast_1d(nu_m)
    
    def nu_c(self, t, **kwargs):
        """
        Calculate the cooling frequency.
        
        This extracts nu_c from VegasAfterglow's details.
        
        Parameters
        ----------
        t : np.ndarray of float
            Observer times [days].
            
        Returns
        -------
        np.ndarray of float
            Cooling frequency [Hz].
        """
        t_sec = self.days_to_sec(t)
        t_sec = np.atleast_1d(t_sec)
        try:
            details = self._get_details(t_sec)
            nu_c = np.asarray(details.fwd.nu_c[0,0,:]*details.fwd.Doppler[0,0,:]/(1+self.z))
            if len(nu_c) != len(t_sec):
                t_details = np.asarray(details.fwd.t_obs[0, 0, :])
                nu_c = np.interp(t_sec, t_details, nu_c)
        except Exception as e:
            raise RuntimeError(f"Failed to extract nu_c from VegasAfterglow: {e}")
        return np.atleast_1d(nu_c)
    
    def nu_a(self, t, **kwargs):
        """
        Calculate the self-absorption frequency.
        
        This extracts nu_a from VegasAfterglow's details.
        
        Parameters
        ----------
        t : np.ndarray of float
            Observer times [days].
            
        Returns
        -------
        np.ndarray of float
            Self-absorption frequency [Hz].
        """
        t_sec = self.days_to_sec(t)
        t_sec = np.atleast_1d(t_sec)
        try:
            details = self._get_details(t_sec)
            nu_a = np.asarray(details.fwd.nu_a[0,0,:]*details.fwd.Doppler[0,0,:]/(1+self.z))
            if len(nu_a) != len(t_sec):
                t_details = np.asarray(details.fwd.t_obs[0, 0, :])
                nu_a = np.interp(t_sec, t_details, nu_a)
        except Exception as e:
            raise RuntimeError(f"Failed to extract nu_a from VegasAfterglow: {e}")
        return np.atleast_1d(nu_a)
    
    # def spectrum(self, t, **kwargs):
    #     """
    #     Return a dictionary of spectral characteristics at given times.
    #     ...
    #     """
    #     # Get density parameters
    #     n_eff, k_eff = self.smooth(t, **kwargs)
        
    #     # Get break frequencies
    #     nu_m = self.nu_m(t, **kwargs)
    #     nu_c = self.nu_c(t, **kwargs)
    #     nu_a = self.nu_a(t, **kwargs)
        
    #     # Compute f_peak by evaluating flux at nu_m (the spectral peak)
    #     # In slow cooling (nu_m < nu_c), peak is at nu_m
    #     # In fast cooling (nu_c < nu_m), peak is at nu_c
    #     nu_peak = np.minimum(nu_m, nu_c)  # Peak is at min(nu_m, nu_c)
    #     f_peak = self.spectral_flux(t, nu_peak)
        
    #     return {
    #         'nu_m': nu_m,
    #         'nu_c': nu_c,
    #         'nu_a': nu_a,
    #         'f_peak': f_peak,
    #         'p': self.p,
    #         'k': k_eff,
    #     }

    def smooth(self, t, **kwargs):
        """
        Return effective density normalization and power-law index.
        
        For VegasAfterglow with smooth_broken medium, returns the
        density parameters at the transition radius.
        
        Parameters
        ----------
        t : np.ndarray of float
            Observer times [days].
            
        Returns
        -------
        n_eff : np.ndarray
            Effective number density [cm^-3] at transition radius.
        k_eff : np.ndarray
            Effective density power-law index.
        """
        # For smooth_broken profile, return fixed values at rt
        if self.medium_type.lower() == 'smooth_broken':
            n0t = 10**self.nt  # Number density at transition radius
            # At the transition radius, k_eff is approximately the average
            k_avg = (self.k1 + self.k2) / 2.0
            
            # Return arrays of constant values matching input size
            t_arr = np.atleast_1d(t)
            n_eff = np.full_like(t_arr, n0t)
            k_eff = np.full_like(t_arr, k_avg)
            return n_eff, k_eff
        elif self.medium_type.lower() == 'powerlaw':
            t_arr = np.atleast_1d(t)
            return np.full_like(t_arr, self.n017), np.full_like(t_arr, self.k)
        elif self.medium_type.lower() == 'ism':
            t_arr = np.atleast_1d(t)
            return np.full_like(t_arr, self.n_ism), np.zeros_like(t_arr)
        elif self.medium_type.lower() == 'wind':
            t_arr = np.atleast_1d(t)
            # Wind: k=2 always
            return np.full_like(t_arr, self.A_star), np.full_like(t_arr, 2.0)
        else:
            raise ValueError(f"Unknown medium_type '{self.medium_type}' in smooth(). "
                           f"Supported types: 'powerlaw', 'ism', 'wind', 'smooth_broken'")
    
    def radii(self, t, **kwargs):
        """
        Return blast wave radius at given times.
        
        Extracts the actual computed radius from VegasAfterglow's details.
        
        Parameters
        ----------
        t : np.ndarray of float
            Observer times [days].
            
        Returns
        -------
        np.ndarray
            Blast wave radius [cm].
        """
        t_sec = self.days_to_sec(t)
        t_sec = np.atleast_1d(t_sec)
        try:
            details = self._get_details(t_sec)
            r = np.asarray(details.fwd.r[0, 0, :])
            if len(r) != len(t_sec):
                t_details = np.asarray(details.fwd.t_obs[0, 0, :])
                r = np.interp(t_sec, t_details, r)
            return r
        except Exception as e:
            raise RuntimeError(f"Failed to extract radius from VegasAfterglow: {e}")

    def spectral_flux(self, t, nu, **kwargs):
        """
        Calculate spectral flux density at given times and frequencies.

        Parameters
        ----------
        t : np.ndarray of float
            Observer times [days].

        nu : float or np.ndarray of float
            Frequencies [Hz].

        Returns
        -------
        np.ndarray of float
            Spectral flux density [mJy].
        """
        t_sec = self.days_to_sec(t)
        t_sec = np.atleast_1d(t_sec)
        nu = np.atleast_1d(nu)
        # Use a tuple of (tuple(t_sec), tuple(nu)) as cache key
        cache_key = (tuple(np.round(t_sec, 8)), tuple(np.round(nu, 8)))
        if cache_key in self._flux_cache:
            return self._flux_cache[cache_key]
        if t_sec.size == nu.size:
            flux_dict = self.vegas_model.flux_density(t_sec, nu)
            flux_cgs = np.asarray(flux_dict.total)
        elif nu.size == 1:
            nu_array = np.full_like(t_sec, nu[0])
            flux_dict = self.vegas_model.flux_density(t_sec, nu_array)
            flux_cgs = np.asarray(flux_dict.total)
        elif t_sec.size == 1:
            t_array = np.full_like(nu, t_sec[0])
            flux_dict = self.vegas_model.flux_density(t_array, nu)
            flux_cgs = np.asarray(flux_dict.total)
        else:
            raise ValueError(
                f"Time array size ({t_sec.size}) and frequency array size ({nu.size}) "
                f"must match or one must be scalar. Use flux_density_grid for grid output."
            )
        result = flux_cgs / 1e-26
        self._flux_cache[cache_key] = result
        return result

    def integrated_flux(self, t, lower, upper, **kwargs):
        """
        Calculate integrated flux over a frequency band.

        Parameters
        ----------
        t : np.ndarray of float
            Observer times [days].

        lower, upper : float or np.ndarray of float
            Frequency integration bounds [Hz].

        Returns
        -------
        np.ndarray of float
            Integrated flux [erg cm^-2 s^-1].
        """
        # Use numerical integration over the band
        # For now, use spectral index approximation
        sflux = self.spectral_flux(t, lower)
        beta = self.spectral_index(t, lower, upper)

        return 1e-26 * (
            (sflux * lower / (beta + 1)) *
            (((upper / lower) ** (beta + 1)) - 1)
        )

    def spectral_index(self, t, lower, upper, **kwargs):
        # print("Hi")
        """
        Calculate spectral index between two frequencies.

        Parameters
        ----------
        t : np.ndarray of float
            Observer times [days].

        lower, upper : float or np.ndarray of float
            Frequency bounds [Hz].

        Returns
        -------
        np.ndarray of float
            Spectral index (dimensionless).
        """
        flux_lower = self.spectral_flux(t, lower)
        flux_upper = self.spectral_flux(t, upper)

        return np.log10(flux_upper / flux_lower) / np.log10(upper / lower)

    def blast_radius(self, t):
        """Return blast wave radius at observer times `t` [days], in cm."""
        return self.radii(t)

    def density(self, t):
        """Return number density [cm^-3] at the blast wave radius at times `t` [days]."""
        r = np.atleast_1d(self.blast_radius(t))
        m_p = 1.67262192e-24
        rho = np.asarray(self.vegas_model.medium(0.0, 0.0, r))
        return rho / (m_p * self.hmf)

    def __call__(self, obs, _params):
        """Behaves the same as `.model(obs)`"""
        return self.model(obs)

    def model(self, obs, subset=None):
        """
        Model an Observation object.

        Parameters
        ----------
        obs : Observation
            The observational data.

        subset : np.ndarray of bool, optional
            Subset of data to model.

        Returns
        -------
        np.ndarray of float
            Modeled observational data.
        """
        if not self.is_valid:
            return np.array([np.nan])

        res = np.full(obs.as_arrays.times.size, np.nan)

        sfm = obs.as_arrays.sflux_loc
        ifm = obs.as_arrays.iflux_loc
        sim = obs.as_arrays.sindex_loc

        if subset is not None:
            sfm = sfm & subset
            ifm = ifm & subset
            sim = sim & subset

        has_sfm, has_ifm, has_sim = sfm.any(), ifm.any(), sim.any()

        if not (has_sfm or has_ifm or has_sim):
            return res

        # Collect all (t, nu) pairs and make a single flux_density call.
        # Layout: [sfm | ifm_lo | ifm_up | sim_lo | sim_up]
        t_parts, nu_parts = [], []

        if has_sfm:
            t_parts.append(self.days_to_sec(obs.times()[sfm]))
            nu_parts.append(obs.freqs()[sfm])

        if has_ifm:
            t_ifm = self.days_to_sec(obs.times()[ifm])
            t_parts += [t_ifm, t_ifm]
            nu_parts += [obs.int_lowers()[ifm], obs.int_uppers()[ifm]]

        if has_sim:
            t_sim = self.days_to_sec(obs.times()[sim])
            t_parts += [t_sim, t_sim]
            nu_parts += [obs.int_lowers()[sim], obs.int_uppers()[sim]]

        t_all = np.concatenate(t_parts)
        nu_all = np.concatenate(nu_parts)

        # flux_density requires ascending time order; sort, evaluate, then restore.
        sort_idx = np.argsort(t_all, kind='stable')
        unsort_idx = np.argsort(sort_idx, kind='stable')
        flux_sorted = np.asarray(
            self.vegas_model.flux_density(t_all[sort_idx], nu_all[sort_idx]).total
        ) / 1e-26
        flux_mJy = flux_sorted[unsort_idx]

        idx = 0

        if has_sfm:
            n = sfm.sum()
            res[sfm] = flux_mJy[idx:idx + n]
            idx += n

        if has_ifm:
            n = ifm.sum()
            f_lo = flux_mJy[idx:idx + n]
            f_up = flux_mJy[idx + n:idx + 2 * n]
            lo, up = obs.int_lowers()[ifm], obs.int_uppers()[ifm]
            beta = np.log10(f_up / f_lo) / np.log10(up / lo)
            res[ifm] = 1e-26 * (f_lo * lo / (beta + 1)) * ((up / lo) ** (beta + 1) - 1)
            idx += 2 * n

        if has_sim:
            n = sim.sum()
            f_lo = flux_mJy[idx:idx + n]
            f_up = flux_mJy[idx + n:idx + 2 * n]
            lo, up = obs.int_lowers()[sim], obs.int_uppers()[sim]
            res[sim] = np.log10(f_up / f_lo) / np.log10(up / lo)

        return res

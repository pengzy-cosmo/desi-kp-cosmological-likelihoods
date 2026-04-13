import cosmoprimo
import numpy as np
from cobaya.log import LoggedError
from cobaya.theory import Theory
from reptvelocileptors import (
    Cosmology,
    PowerSpectrumBAOFilter,
    PowerSpectrumInterpolator1D,
    _convert_camb_or_classy_to_cosmoprimo_params,
    reptvelocileptors,
)


class reptvelocileptors_class(reptvelocileptors):
    """DESI FS wrapper that reconstructs the cb inputs needed by REPT from CLASS."""

    def must_provide(self, **requirements):
        Theory.must_provide(self, **requirements)
        for key, value in requirements.items():
            if key == "pkpoles":
                self.z = np.asarray(value["z"], dtype=float)
                self.k = np.asarray(value["k"], dtype=float)
                self.ells = tuple(value["ells"])
                fiducial = value["fiducial"]

        self.kin = np.geomspace(
            min(self._kinlim[0], self.k[0] / 2.0),
            max(self._kinlim[1], self.k[-1] * 2.0),
            self._kinlim[2],
        )
        self.fiducial = getattr(cosmoprimo.fiducial, fiducial)(engine="camb")
        pk_dd_interpolator_fid = self.fiducial.get_fourier().pk_interpolator(of="delta_cb").to_1d(z=self.z)
        self._template = {
            "filter": PowerSpectrumBAOFilter(
                pk_dd_interpolator_fid,
                engine="peakaverage",
                cosmo=self.fiducial,
                cosmo_fid=self.fiducial,
            )
        }

        return {
            "CLASS_primordial": None,
            "Hubble": {"z": np.concatenate([[0.0], self.z])},
            "angular_diameter_distance": {"z": self.z},
            "rdrag": None,
            "Omega_nu_massive": {"z": [0.0]},
            # In Cobaya/classy this requirement calls add_z_for_matter_power(), which
            # makes CLASS populate z_pk at the exact redshifts needed below.
            "sigma8_z": {"z": self.z},
        }

    @staticmethod
    def _interp_loglog_with_extrapolation(x, y, x_new):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x_new = np.asarray(x_new, dtype=float)
        order = np.argsort(x)
        logx = np.log(x[order])
        logy = np.log(y[order])
        logx_new = np.log(x_new)
        logy_new = np.interp(logx_new, logx, logy)

        left = logx_new < logx[0]
        if np.any(left):
            slope = (logy[1] - logy[0]) / (logx[1] - logx[0])
            logy_new[left] = logy[0] + slope * (logx_new[left] - logx[0])

        right = logx_new > logx[-1]
        if np.any(right):
            slope = (logy[-1] - logy[-2]) / (logx[-1] - logx[-2])
            logy_new[right] = logy[-1] + slope * (logx_new[right] - logx[-1])

        return np.exp(logy_new)

    def _get_class_component(self):
        component = self.provider.requirement_providers.get("CLASS_primordial")
        if component is None or not hasattr(component, "classy"):
            raise LoggedError(
                self.log,
                "Could not locate the CLASS component providing `CLASS_primordial`.",
            )
        return component

    def _get_pk_cb(self, class_backend, h):
        k_h = self.kin
        k = k_h * h
        pk_cb = class_backend.get_pk_cb_array(
            np.asarray(k, dtype=float),
            np.asarray(self.z, dtype=float),
            k.size,
            self.z.size,
            False,
        )
        pk_cb = np.asarray(pk_cb, dtype=float).reshape(self.z.size, k.size).T * h**3
        return k_h, pk_cb

    def _get_pk_tt(self, class_backend, h, primordial):
        transfer0 = class_backend.get_transfer(z=float(self.z[0]), output_format="class")
        needed = ["k (h/Mpc)", "t_b", "h_prime", "eta_prime"]
        missing = [name for name in needed if name not in transfer0]
        if missing:
            raise LoggedError(
                self.log,
                "CLASS direct outputs are missing %r. Add `output: mPk mTk vTk`, `gauge: synchronous` and `extra_metric_transfer_functions: yes` to the classy block.",
                missing,
            )

        k_h = np.asarray(transfer0["k (h/Mpc)"], dtype=float)
        k = k_h * h
        theta_b = np.empty((k.size, self.z.size), dtype=float)
        h_prime = np.empty_like(theta_b)
        eta_prime = np.empty_like(theta_b)

        for iz, z in enumerate(self.z):
            transfer = class_backend.get_transfer(z=float(z), output_format="class")
            if not np.allclose(np.asarray(transfer["k (h/Mpc)"], dtype=float), k_h):
                raise LoggedError(
                    self.log,
                    "CLASS transfer k-grid changed across requested redshifts.",
                )
            theta_b[:, iz] = np.asarray(transfer["t_b"], dtype=float)
            h_prime[:, iz] = np.asarray(transfer["h_prime"], dtype=float)
            eta_prime[:, iz] = np.asarray(transfer["eta_prime"], dtype=float)

        k_prim = np.asarray(primordial["k [1/Mpc]"], dtype=float)
        p_prim = np.asarray(primordial["P_scalar(k)"], dtype=float)
        p_prim_interp = self._interp_loglog_with_extrapolation(k_prim, p_prim, k)
        prefactor = 2.0 * np.pi**2 * p_prim_interp[:, None] / k[:, None] ** 3

        Omega_b = float(self.provider.get_param("omega_b")) / h**2
        Omega_cdm = float(self.provider.get_param("omega_cdm")) / h**2
        f_b = Omega_b / (Omega_b + Omega_cdm)
        # CLASS stores t_b = theta_b = k * v_b in the current gauge. Shifting to
        # Newtonian gauge adds alpha * k^2 = (h' + 6 eta') / 2, so this matches
        # CAMB's v_newtonian = -v_N k / (aH) = -theta_N / (aH).
        theta_cb_newtonian = f_b * theta_b + 0.5 * (h_prime + 6.0 * eta_prime)
        aH = (
            np.asarray(self.provider.get_Hubble(z=self.z, units="1/Mpc"), dtype=float)[None, :]
            / (1.0 + self.z)[None, :]
        )
        pk_tt = prefactor * (-theta_cb_newtonian / aH) ** 2 * h**3
        return k_h, pk_tt

    def set_template(self):
        class_component = self._get_class_component()
        gauge = str(class_component.extra_args.get("gauge", "synchronous")).lower()
        if gauge != "synchronous":
            raise LoggedError(
                self.log,
                "`reptvelocileptors_class` requires `gauge: synchronous`.",
            )
        if str(class_component.extra_args.get("nbody_gauge_transfer_functions", "no")).lower() in {"yes", "true", "1"}:
            raise LoggedError(
                self.log,
                "`reptvelocileptors_class` requires `nbody_gauge_transfer_functions: no`.",
            )

        h = float(np.squeeze(self.provider.get_Hubble(0.0)) / 100.0)
        primordial = self.provider.get_CLASS_primordial()
        class_backend = class_component.classy
        k_h_dd, pk_dd = self._get_pk_cb(class_backend, h)
        k_h_tt, pk_tt = self._get_pk_tt(class_backend, h, primordial)
        pk_dd_interpolator = PowerSpectrumInterpolator1D(k_h_dd, pk_dd)
        pk_tt_interpolator = PowerSpectrumInterpolator1D(k_h_tt, pk_tt)

        Omega_b = float(self.provider.get_param("omega_b")) / h**2
        Omega_cdm = float(self.provider.get_param("omega_cdm")) / h**2
        Omega_ncdm = float(np.squeeze(self.provider.get_Omega_nu_massive(z=0.0)))

        params = dict(self.provider.params)
        cstate = {
            _convert_camb_or_classy_to_cosmoprimo_params[param]: value
            for param, value in params.items()
            if param in _convert_camb_or_classy_to_cosmoprimo_params
        }
        cstate = {name: value for name, value in cstate.items() if name in ["n_s"]}
        cstate["Omega_b"] = Omega_b
        cstate["Omega_cdm"] = Omega_cdm
        cstate["Omega_ncdm"] = Omega_ncdm
        cstate["H0"] = 100.0 * h
        cosmo = Cosmology(**cstate)
        cosmo.rs_drag = self.provider.get_param("rdrag") * h

        self._template["filter"](pk_dd_interpolator, cosmo=cosmo)
        pknow_dd_interpolator = self._template["filter"].smooth_pk_interpolator()

        self._template.update(
            pk_dd_interpolator=pk_dd_interpolator,
            pknow_dd_interpolator=pknow_dd_interpolator,
            pk_tt_interpolator=pk_tt_interpolator,
        )


import h5py as h5
import jax.numpy as jnp
import numpy as np
from jax.scipy.integrate import trapezoid
from jax.scipy.interpolate import RegularGridInterpolator

from .data_vector import DataVector

datavector_requires = {
    "c_cmbkcmbk": [],
    "c_dcmbk": ["nz_d"],
    "c_kk": ["nz_s"],
    "c_bb": ["nz_s"],
    "c_dk": ["nz_d", "nz_s"],
    "c_dd": ["nz_d"],
    "xi_plus":  ["nz_s"],
    "xi_minus": ["nz_s"],
    "w_theta":  ["nz_d"],
    "gamma_t":  ["nz_d", "nz_s"]
}

covariance_field_types = {
    "c_cmbkcmbk": ["cmbk", "cmbk"],
    "c_dcmbk": ["d", "cmbk"],
    "c_kk": ["k", "k"],
    "c_bb": ["b", "b"],
    "c_dk": ["d", "k"],
    "c_dd": ["d", "d"],
    "xi_plus":  ["k", "k"],
    "xi_minus": ["k", "k"],
    "w_theta":  ["d", "d"],
    "gamma_t":  ["d", "k"]
}

field_types = {
    "c_cmbkcmbk": ["cmbk", "cmbk"],
    "c_dcmbk": ["d", "cmbk"],
    "c_kk": ["gamma_e", "gamma_e"],
    "c_bb": ["gamma_b", "gamma_b"],
    "c_dk": ["d", "gamma_e"],
    "c_dd": ["d", "d"],
    "xi_plus":  ["gamma_e", "gamma_e"],
    "xi_minus": ["gamma_e", "gamma_e"],
    "w_theta":  ["d", "d"],
    "gamma_t":  ["d", "gamma_e"],
}


class TwoPointSpectrum(DataVector):
    """Data vector for two-point angular power spectra (C_ell).

    Handles loading observed spectra and covariance from HDF5, applying
    scale cuts, computing Gaussian covariance matrices, interpolating
    redshift distributions, and managing bandpower window functions.
    """

    _field_types = field_types

    def __init__(
        self,
        data_vector_info_filename,
        spectrum_info,
        covariance_info=None,
        scale_cuts=None,
        zeff_weighting=True,
        dummy_cov=False,
        generate_data_vector=False,
        zmin=0,
        zmax=2.0,
        nz=125,
    ):
        self.data_vector_info_filename = data_vector_info_filename
        self.spectrum_info = spectrum_info
        self.scale_cuts = scale_cuts
        self.spectrum_types = list(self.spectrum_info.keys())
        self.zeff_weighting = zeff_weighting
        self.dummy_cov = dummy_cov
        self.generate_data_vector = generate_data_vector
        self.covariance_info = covariance_info
        self.z = jnp.linspace(zmin, zmax, nz)

    def _compute_ell_binning(self):
        """Compute the bandpower ell_eff and delta_ell arrays and store them as
        instance attributes.  Returns the bpws index array so callers that also
        need to build a window matrix can reuse it."""
        ells = np.arange(3 * 2048, dtype="int32")
        bpws = np.zeros_like(ells) - 1

        i = 0
        counter = 25  # ell_start
        delta_ell = int(3 * np.sqrt(counter))
        bpw_widths = [delta_ell]

        while counter + delta_ell < ells.shape[0]:
            bpws[counter : counter + delta_ell] = i
            counter = counter + delta_ell
            delta_ell = int(3 * np.sqrt(ells[counter]))
            bpw_widths.append(delta_ell)
            i += 1

        self.delta_ell = np.array(bpw_widths)[:-1]
        ell_eff = np.bincount(bpws + 1, weights=ells * (2 * ells + 1)) / np.bincount(
            bpws + 1, weights=(2 * ells + 1)
        )
        self.ell_eff = ell_eff[1:]
        return bpws

    def generate_data(self):
        required_spectra = []
        for si in self.spectrum_info:
            for sj in self.spectrum_info:
                c0 = f"c_{covariance_field_types[si][0]}{covariance_field_types[sj][0]}"
                if c0 not in field_types:
                    c0 = f"c_{covariance_field_types[sj][0]}{covariance_field_types[si][0]}"

                c1 = f"c_{covariance_field_types[si][1]}{covariance_field_types[sj][1]}"
                if c1 not in field_types:
                    c1 = f"c_{covariance_field_types[sj][1]}{covariance_field_types[si][1]}"

                c2 = f"c_{covariance_field_types[si][0]}{covariance_field_types[sj][1]}"
                if c2 not in field_types:
                    c2 = f"c_{covariance_field_types[sj][1]}{covariance_field_types[si][0]}"

                c3 = f"c_{covariance_field_types[si][1]}{covariance_field_types[sj][0]}"
                if c3 not in field_types:
                    c3 = f"c_{covariance_field_types[sj][0]}{covariance_field_types[si][1]}"

                required_spectra.extend([c0, c1, c2, c3])
        required_spectra = np.unique(required_spectra)

        bin_pairs = {}
        n_bins = 0
        for t in required_spectra:
            bin_pairs[t] = []
            for i in self.spectrum_info[t]["bins0"]:
                use_cross = self.spectrum_info[t].get("use_cross", True)
                if use_cross:
                    for j in self.spectrum_info[t]["bins1"]:
                        if (
                            covariance_field_types[t][0] != covariance_field_types[t][1]
                        ) | (j >= i):
                            bin_pairs[t].append((i, j))
                            n_bins += 1
                else:
                    j = i
                    bin_pairs[t].append((i, j))
                    n_bins += 1

        ells = np.arange(3 * 2048, dtype="int32")  # Array of multipoles
        weights = np.zeros(len(ells))  # Array of weights
        bpws = np.zeros_like(ells) - 1  # Array of bandpower indices
        ell_start = 25
        delta_ell = 30

        i = 0
        counter = ell_start
        delta_ell = int(3 * np.sqrt(ell_start))
        bpw_widths = [delta_ell]

        while counter + delta_ell < ells.shape[0]:
            bpws[counter : counter + delta_ell] = i
            weights[counter : counter + delta_ell] = 1 / float(delta_ell)
            counter = counter + delta_ell
            delta_ell = int(3 * np.sqrt(ells[counter]))
            bpw_widths.append(delta_ell)
            i += 1

        self.delta_ell = np.array(bpw_widths)[:-1]
        ell_eff = np.bincount(bpws + 1, weights=ells * (2 * ells + 1)) / np.bincount(
            bpws + 1, weights=(2 * ells + 1)
        )
        ell_eff = ell_eff[1:]
        window = np.zeros((len(ell_eff), len(ells)))
        self.ell_eff = ell_eff
        for i, ell in enumerate(ell_eff):
            window[i, bpws == i] = 1 / float(self.delta_ell[i])

        dt = np.dtype(
            [
                ("spectrum_type", "S10"),
                ("zbin0", int),
                ("zbin1", int),
                ("separation", float),
                ("value", float),
            ]
        )
        spectra = np.zeros(n_bins * len(ell_eff), dtype=dt)
        self.cW = {}
        counter = 0
        for t in required_spectra:
            self.cW[t] = {}
            for bin_pair in bin_pairs[t]:
                spectra[counter : counter + ell_eff.shape[0]]["spectrum_type"] = t
                spectra[counter : counter + ell_eff.shape[0]]["zbin0"] = bin_pair[0]
                spectra[counter : counter + ell_eff.shape[0]]["zbin1"] = bin_pair[1]
                spectra[counter : counter + ell_eff.shape[0]]["separation"] = ell_eff

                self.cW[t][f"{bin_pair[0]}_{bin_pair[1]}"] = window
                counter += ell_eff.shape[0]
        #            print(self.cW[t], flush=True)

        return spectra

    def save_data_vector(self, filename, model):
        with h5.File(filename, "w") as f:
            dt = np.dtype(
                [
                    ("spectrum_type", "S10"),
                    ("zbin0", int),
                    ("zbin1", int),
                    ("separation", float),
                    ("value", float),
                ]
            )
            data = np.zeros(len(model), dtype=dt)
            data["spectrum_type"] = self.spectra["spectrum_type"]
            data["zbin0"] = self.spectra["zbin0"]
            data["zbin1"] = self.spectra["zbin1"]
            data["separation"] = self.spectra["separation"]
            data["value"] = model

            f.create_dataset("spectra", data=data)
            f.create_dataset("covariance", data=self.cov.flatten())
            
            for k_i in self.data_vector_info.keys():
                if k_i in ["spectra", "covariance"]:
                    continue
                elif 'window' in k_i:
                    grp = f.create_group(k_i)
                    for k_j in self.data_vector_info[k_i].keys():
                        grpp = grp.create_group(k_j)
                        for k_k in self.data_vector_info[k_i][k_j].keys():
                            grpp.create_dataset(k_k, data=self.data_vector_info[k_i][k_j][k_k][:])
                else:
                    f.create_dataset(k_i, data=self.data_vector_info[k_i][:])

    def load_requirements(self):
        requirements = []
        for t in self.spectrum_info:
            requirements = datavector_requires[t]
            if self.zeff_weighting:
                if t == "c_dk":
                    requirements.append("nz_d_dk")
                elif t == "c_dcmbk":
                    requirements.append("nz_d_dcmbk")
            for r in requirements:
                nz_ = self.data_vector_info[r][:]
                nbins = nz_.shape[1] - 1
                nz = jnp.zeros((nbins, len(self.z)))

                for i in range(nbins):
                    idx = nz_[:, i + 1] > 0
                    nz = nz.at[i, :].set(
                        RegularGridInterpolator(
                            [(nz_[idx, 0])],
                            np.atleast_2d(nz_[idx, i + 1]).T,
                            fill_value=0,
                        )(self.z)[:, 0]
                    )

                nz = nz / trapezoid(nz, x=self.z, axis=-1)[:, None]
                self.spectrum_info[t][r] = nz
                setattr(self, r, nz)

        if ("cell_windows" in self.data_vector_info.keys()) & (
            not self.generate_data_vector
        ):
            window_matrix_files = self.data_vector_info["cell_windows"]

            self.cW = {}
            for k in list(window_matrix_files.keys()):
                if k not in self.spectrum_types:
                    continue

                self.cW[k] = {}
                for ij in list(window_matrix_files[k].keys()):
                    self.cW[k][ij] = window_matrix_files[k][ij][:]

    def _ensure_covariance_info(self):
        """Prompt interactively for any f_sky or noise terms missing from covariance_info.
        Also ensures ell_eff and delta_ell are computed if not already set."""
        if not hasattr(self, 'ell_eff') or self.ell_eff is None:
            self._compute_ell_binning()

        if self.covariance_info is None:
            self.covariance_info = {}

        if ("f_sky" not in self.covariance_info):
            for s in self.covariance_info:
                try:
                    f_sky = self.covariance_info[s]["f_sky"]
                except KeyError:
                    val = input("f_sky not found in config. Enter f_sky: ")
                    self.covariance_info[s]["f_sky"] = float(val)

        for t in self.spectrum_info:
            if t not in self.covariance_info:
                self.covariance_info[t] = {}
            for (b0, b1) in self.spectrum_info[t]["bin_pairs"]:
                key = f"{b0}_{b1}"
                entry = self.covariance_info[t].get(key, {})
                if "noise" not in entry:
                    val = input(
                        f"noise for {t} bin pair ({b0}, {b1}) not found in config. Enter noise: "
                    )
                    entry["noise"] = float(val)
                    self.covariance_info[t][key] = entry

    def _lookup_spectrum(self, spec, za, zb, model_spectra):
        """Look up a spectrum value from model_spectra dict or observed data."""
        if model_spectra is not None and (spec, za, zb) in model_spectra:
            return model_spectra[(spec, za, zb)]
        s = self.spectra["value"][
            (self.spectra["spectrum_type"] == spec.encode('utf-8'))
            & (self.spectra["zbin0"] == za)
            & (self.spectra["zbin1"] == zb)
        ]
        if len(s)>0:
            return s
        else:
            raise(ValueError(f"No spectrum {spec} with zbin comination {za}, {zb} found in model spectra or observed data."))

    def gaussian_variance(self, si, sj, z00, z01, z10, z11, model_spectra=None):
        """Compute the diagonal Gaussian variance for a pair of spectrum blocks.

        Args:
            si: First spectrum type string.
            sj: Second spectrum type string.
            z00: First redshift bin of spectrum si.
            z01: Second redshift bin of spectrum si.
            z10: First redshift bin of spectrum sj.
            z11: Second redshift bin of spectrum sj.
            model_spectra: Optional dict of model spectra to use instead of observed.

        Returns:
            Array of variance values per ell bin.
        """
        c0 = f"c_{covariance_field_types[si][0]}{covariance_field_types[sj][0]}"
        if c0 not in field_types:
            c0 = f"c_{covariance_field_types[sj][0]}{covariance_field_types[si][0]}"

        c1 = f"c_{covariance_field_types[si][1]}{covariance_field_types[sj][1]}"
        if c1 not in field_types:
            c1 = f"c_{covariance_field_types[sj][1]}{covariance_field_types[si][1]}"

        c2 = f"c_{covariance_field_types[si][0]}{covariance_field_types[sj][1]}"
        if c2 not in field_types:
            c2 = f"c_{covariance_field_types[sj][1]}{covariance_field_types[si][0]}"

        c3 = f"c_{covariance_field_types[si][1]}{covariance_field_types[sj][0]}"
        if c3 not in field_types:
            c3 = f"c_{covariance_field_types[sj][0]}{covariance_field_types[si][1]}"

        spec_w_n = []
        for spec, za, zb, f0, f1 in zip(
            [c0, c1, c2, c3],
            [z00, z01, z00, z01],
            [z10, z11, z11, z10],
            [
                covariance_field_types[si][0],
                covariance_field_types[si][1],
                covariance_field_types[si][0],
                covariance_field_types[si][1],
            ],
            [
                covariance_field_types[sj][0],
                covariance_field_types[sj][1],
                covariance_field_types[sj][1],
                covariance_field_types[sj][0],
            ],
        ):
            try:
                c_w_n = self._lookup_spectrum(spec, za, zb, model_spectra) \
                        + float(self.covariance_info[spec][f"{za}_{zb}"]["noise"])
            
            except:
                if f0 == f1:
                    c_w_n = self._lookup_spectrum(spec, zb, za, model_spectra) \
                        + float(self.covariance_info[spec][f"{zb}_{za}"]["noise"])
                else:
                    raise(ValueError(f"No spectrum {spec} with zbin comination {za}, {zb} or {zb}, {za}"))

            spec_w_n.append(c_w_n)

        var = (spec_w_n[0] * spec_w_n[1] + spec_w_n[2] * spec_w_n[3]) / (
            float(self.covariance_info["f_sky"]) * self.delta_ell * (2 * self.ell_eff)
        )

        return var

    def gaussian_covariance(self):
        dt = np.dtype(
            [
                ("spectrum_type0", "S10"),
                ("spectrum_type1", "S10"),
                ("zbin00", int),
                ("zbin01", int),
                ("zbin10", int),
                ("zbin11", int),
                ("separation0", float),
                ("separation1", float),
                ("value", float),
            ]
        )

        cov = np.zeros((self.n_dv, self.n_dv), dtype=dt)
        counter_i = 0
        for si in self.spectrum_info:
            n_ell_i = self.spectrum_info[si]["n_dv_per_bin"]
            for z00, z01 in self.spectrum_info[si]["bin_pairs"]:
                counter_j = 0
                for sj in self.spectrum_info:
                    n_ell_j = self.spectrum_info[sj]["n_dv_per_bin"]
                    for z10, z11 in self.spectrum_info[sj]["bin_pairs"]:
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["spectrum_type0"] = si
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["spectrum_type1"] = sj
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["zbin00"] = z00
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["zbin01"] = z01
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["zbin10"] = z10
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["zbin11"] = z11
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["separation0"] = self.spectrum_info[si]["separation"][:, None]
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["separation1"] = self.spectrum_info[sj]["separation"][None, :]

                        var = self.gaussian_variance(si, sj, z00, z01, z10, z11)
                        np.fill_diagonal(
                            cov[
                                counter_i : counter_i + n_ell_i,
                                counter_j : counter_j + n_ell_j,
                            ]["value"],
                            var,
                        )
                        counter_j += n_ell_j

                counter_i += n_ell_i

        return cov


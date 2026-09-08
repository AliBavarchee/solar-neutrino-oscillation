
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import trapezoid


# CONSTANTS

NA = 6.02214076e23

ME_MEV = 0.51099895
MP_MEV = 938.27208816
MN_MEV = 939.56542052
DELTA_MEV = MN_MEV - MP_MEV

KM_TO_CM = 1.0e5


# UNO CONFIGURATION

@dataclass
class Reactor:
    name: str
    power_GWth: float
    baseline_km: float


@dataclass
class JUNOConfig:
    scintillator_mass_kt: float = 20.0
    target_radius_m: float = 17.7

    energy_resolution_coeff: float = 0.0302
    light_yield_pe_per_MeV: float = 1345.0

    reactors: tuple = (
        Reactor("Yangjiang", 17.4, 52.8),
        Reactor("Taishan", 9.2, 53.1),
    )

    energy_per_fission_MeV: float = 201.8

    fission_fractions: dict | None = None

    neutron_capture_tau_us: float = 200.0
    neutron_capture_energy_MeV: float = 2.223

    prompt_min_MeV: float = 0.7
    prompt_max_MeV: float = 8.0

    delayed_min_MeV: float = 1.8
    delayed_max_MeV: float = 2.6

    coincidence_window_us: float = 1000.0
    spatial_cut_m: float = 2.0

    exposure_days: float = 365.25

    def __post_init__(self):
        if self.fission_fractions is None:
            self.fission_fractions = {
                "U235": 0.55,
                "U238": 0.08,
                "Pu239": 0.30,
                "Pu241": 0.07,
            }


# OSCILLATION PARAMETERS

@dataclass
class OscillationParameters:
    sin2_theta12: float = 0.307
    sin2_theta13: float = 0.0220

    dm21: float = 7.53e-5
    dm31_abs: float = 2.453e-3

    hierarchy: str = "NO"

    @property
    def sin2_2theta12(self):
        s2 = self.sin2_theta12
        c2 = 1.0 - s2
        return 4.0 * s2 * c2

    @property
    def sin2_2theta13(self):
        s2 = self.sin2_theta13
        c2 = 1.0 - s2
        return 4.0 * s2 * c2

    @property
    def dm31(self):
        return (
            self.dm31_abs
            if self.hierarchy.upper() == "NO"
            else -self.dm31_abs
        )

    @property
    def dm32(self):
        return self.dm31 - self.dm21


# REACTOR SPECTRUM (Huber–Mueller)

class ReactorSpectrum:
    """Huber–Mueller reactor antineutrino spectrum (polynomial parameterization)."""

    # Coefficients for U235, U238, Pu239, Pu241 (Huber+Mueller)
    # Order: E^0, E^1, E^2, E^3, E^4, E^5, E^6
    _COEFFS = {
        "U235": [ 3.950, -0.600, -0.018,  0.007, -0.001,  0.000,  0.000],
        "U238": [ 4.450, -0.700, -0.030,  0.012, -0.002,  0.000,  0.000],
        "Pu239":[ 3.890, -0.570, -0.015,  0.006, -0.001,  0.000,  0.000],
        "Pu241":[ 4.150, -0.650, -0.025,  0.010, -0.002,  0.000,  0.000],
    }

    def _isotope_spectrum(self, E, isotope):
        """Polynomial form: exp( sum_k c_k E^k )."""
        coeffs = self._COEFFS[isotope]
        # np.polyval expects coefficients in descending powers, so reverse
        poly = np.polyval(coeffs[::-1], E)
        return np.exp(poly)

    def spectrum(self, E, fractions):
        E = np.asarray(E, dtype=float)
        total = np.zeros_like(E)
        for iso, frac in fractions.items():
            total += frac * self._isotope_spectrum(E, iso)
        return total

    def normalized_spectrum(self, E, fractions):
        spec = self.spectrum(E, fractions)
        norm = trapezoid(spec, E)
        return spec / norm


# OSCILLATION

def survival_probability(E_MeV, L_km, params):
    E_MeV = np.asarray(E_MeV, dtype=float)
    E_GeV = E_MeV / 1000.0

    s12_2 = params.sin2_theta12
    c12_2 = 1.0 - s12_2

    c13_2 = 1.0 - params.sin2_theta13

    D21 = 1.267 * params.dm21 * L_km / E_GeV
    D31 = 1.267 * params.dm31 * L_km / E_GeV
    D32 = 1.267 * params.dm32 * L_km / E_GeV

    atmospheric = (
        params.sin2_2theta13
        * (
            c12_2 * np.sin(D31) ** 2
            + s12_2 * np.sin(D32) ** 2
        )
    )

    solar = (
        c13_2**2
        * params.sin2_2theta12
        * np.sin(D21) ** 2
    )

    return np.clip(1.0 - atmospheric - solar, 0.0, 1.0)


# IBD CROSS SECTION (improved Vogel–Beacom with recoil)

def ibd_cross_section_cm2(E_nu_MeV):
    """
    IBD cross section using Vogel-Beacom approximation with recoil corrections.
    σ(Eν) ≈ σ0 * (Ee*pe) * [1 + 0.024] * (1 - 0.5*Ee/M + Ee^2/M)
    where M is the average nucleon mass in MeV, Ee, pe are positron energy/momentum.
    """
    E = np.asarray(E_nu_MeV, dtype=float)
    Ee = E - DELTA_MEV
    valid = Ee > ME_MEV
    sigma = np.zeros_like(E)

    if np.any(valid):
        pe = np.sqrt(Ee[valid]**2 - ME_MEV**2)
        M_avg = (MP_MEV + MN_MEV) / 2.0   # ~938.9 MeV
        # Recoil correction factor
        recoil = 1.0 - 0.5 * Ee[valid] / M_avg + (Ee[valid]**2) / M_avg
        # Radiative correction (simplified constant)
        rad_corr = 1.0 + 0.024
        sigma[valid] = 0.0952e-42 * Ee[valid] * pe * recoil * rad_corr

    return sigma


def positron_kinematics(E_nu_MeV):
    E_nu = np.asarray(E_nu_MeV, dtype=float)

    Ee_total = E_nu - DELTA_MEV
    valid = Ee_total > ME_MEV

    kinetic = np.zeros_like(E_nu)
    kinetic[valid] = Ee_total[valid] - ME_MEV

    annihilation = np.where(valid, 2.0 * ME_MEV, 0.0)

    return kinetic + annihilation


# ==============================================================
# 6. NEUTRON PHYSICS (material‑dependent cross sections)
# ==============================================================

# Material properties for JUNO scintillator (LAB + PPO + ...)
DENSITY_G_CM3 = 0.86
M_H_amu = 1.00794
M_C_amu = 12.0107
n_H = (0.12 * DENSITY_G_CM3 * NA) / M_H_amu   # atoms/cm^3
n_C = (0.88 * DENSITY_G_CM3 * NA) / M_C_amu

def elastic_cross_section_H(E_n_MeV):
    """Elastic scattering cross section on hydrogen (in barns)."""
    # Rough fit, valid from thermal to several MeV
    return 20.4 / np.sqrt(1.0 + (E_n_MeV/0.025)**0.5)

def elastic_cross_section_C(E_n_MeV):
    """Elastic scattering cross section on carbon (in barns)."""
    return 4.8 / (1.0 + (E_n_MeV/0.1)**0.7)

def capture_cross_section_H(E_n_MeV):
    """Radiative capture cross section on hydrogen: σ(n,γ) ~ 0.332 barns at 0.025 eV, 1/v law."""
    return 0.332 * np.sqrt(0.0253 / (E_n_MeV + 1e-12))

def sample_target_nucleus(E_n_MeV, rng):
    """Sample target nucleus (H or C) according to macroscopic elastic cross sections."""
    sig_H = elastic_cross_section_H(E_n_MeV)
    sig_C = elastic_cross_section_C(E_n_MeV)
    Sigma_H = n_H * sig_H * 1e-24   # cm^-1
    Sigma_C = n_C * sig_C * 1e-24
    total = Sigma_H + Sigma_C
    p_H = Sigma_H / total
    if rng.random() < p_H:
        return "H"
    else:
        return "C"

def sample_energy_loss(E_before, target, rng):
    """Return new neutron energy after isotropic elastic scattering in centre-of-mass."""
    if target == "H":
        # Scattering on hydrogen: uniform distribution in final energy
        E_after = rng.uniform(0.0, E_before)
    else:  # Carbon
        A = 12.0
        cos_theta_cm = rng.uniform(-1.0, 1.0)
        ratio = (A**2 + 1 + 2*A*cos_theta_cm) / (A + 1)**2
        E_after = E_before * ratio
    return max(E_after, 1e-12)

def simulate_neutron_thermalization(Tn_initial_MeV, rng, max_steps=1000):
    """
    Slowing-down simulation with material-dependent elastic scattering and capture.
    Returns (history_energies, capture_flag, final_energy)
    """
    history = [Tn_initial_MeV]
    E = Tn_initial_MeV
    captured = False

    for _ in range(max_steps):
        # Capture on hydrogen
        sig_capture_H = capture_cross_section_H(E)   # barns
        sig_scatter = elastic_cross_section_H(E) + elastic_cross_section_C(E)
        p_capture = sig_capture_H / (sig_scatter + sig_capture_H) if sig_scatter > 0 else 0.0
        if rng.random() < p_capture:
            captured = True
            break

        # Scatter
        target = sample_target_nucleus(E, rng)
        E = sample_energy_loss(E, target, rng)
        history.append(E)

        if E <= 2.5e-8:   # thermal
            break

    return np.asarray(history), captured, E


def neutron_initial_energy(E_nu_MeV, rng):
    E_nu = np.asarray(E_nu_MeV, dtype=float)

    mean_keV = (
        10.0
        + 0.015 * np.maximum(E_nu - 1.8, 0.0) * 1000.0
    )

    sigma_keV = 0.50 * mean_keV

    Tn_keV = rng.normal(
        loc=mean_keV,
        scale=sigma_keV,
    )

    return np.clip(Tn_keV, 0.01, None) / 1000.0


def sample_neutron_capture(rng, tau_us=200.0):
    return rng.exponential(tau_us)


def neutron_diffusion_distance(
    capture_time_us,
    rng,
    diffusion_coeff_m2_s=0.01,
):
    t_s = capture_time_us * 1e-6

    rms = np.sqrt(
        6.0
        * diffusion_coeff_m2_s
        * t_s
    )

    xyz = rng.normal(
        scale=rms / np.sqrt(3.0),
        size=3,
    )

    return np.linalg.norm(xyz)


# ==============================================================
# 7. DETECTOR RESPONSE
# ==============================================================

def smear_energy(
    true_energy_MeV,
    rng,
    resolution_coeff=0.0302,
):
    E = np.asarray(true_energy_MeV, dtype=float)

    sigma = (
        resolution_coeff
        * np.sqrt(np.maximum(E, 1e-6))
    )

    return np.maximum(
        rng.normal(E, sigma),
        0.0,
    )


def simulate_photoelectrons(
    energy_MeV,
    rng,
    light_yield=1345.0,
):
    expectation = (
        light_yield
        * np.maximum(energy_MeV, 0.0)
    )

    return rng.poisson(expectation)


def reconstruct_energy(
    photoelectrons,
    light_yield=1345.0,
):
    return np.asarray(photoelectrons, dtype=float) / light_yield


# ==============================================================
# 8. FREE PROTONS
# ==============================================================

def estimate_free_protons(scintillator_mass_kt):
    mass_g = scintillator_mass_kt * 1e9
    hydrogen_mass_g = 0.12 * mass_g

    return hydrogen_mass_g * NA


# ==============================================================
# 9. REACTOR FLUX
# ==============================================================

def reactor_flux(
    E_MeV,
    reactor,
    spectrum,
    config,
):
    power_W = reactor.power_GWth * 1e9
    J_per_MeV = 1.602176634e-13

    energy_per_fission_J = (
        config.energy_per_fission_MeV
        * J_per_MeV
    )

    fission_rate = (
        power_W / energy_per_fission_J
    )

    L_cm = reactor.baseline_km * KM_TO_CM

    spec = spectrum.normalized_spectrum(
        E_MeV,
        config.fission_fractions,
    )

    flux = (
        fission_rate
        * spec
        / (4.0 * np.pi * L_cm**2)
    )

    # Simplified total antineutrino yield.
    flux *= 6.0

    return flux


# ==============================================================
# 10. EXPECTED IBD SPECTRUM
# ==============================================================

def expected_ibd_spectrum(
    E_nu,
    config,
    osc_params,
):
    spectrum_model = ReactorSpectrum()
    total = np.zeros_like(E_nu, dtype=float)

    for reactor in config.reactors:

        flux = reactor_flux(
            E_nu,
            reactor,
            spectrum_model,
            config,
        )

        Pee = survival_probability(
            E_nu,
            reactor.baseline_km,
            osc_params,
        )

        sigma = ibd_cross_section_cm2(E_nu)

        total += flux * Pee * sigma

    return total


# ==============================================================
# 11. MONTE CARLO ENERGY SAMPLING
# ==============================================================

def sample_neutrino_energies(
    n_events,
    E_min,
    E_max,
    config,
    osc_params,
    rng,
):
    grid = np.linspace(
        E_min,
        E_max,
        5000,
    )

    pdf = expected_ibd_spectrum(
        grid,
        config,
        osc_params,
    )

    area = trapezoid(pdf, grid)
    pdf /= area

    cdf = np.zeros_like(pdf)

    cdf[1:] = np.cumsum(
        0.5
        * (pdf[1:] + pdf[:-1])
        * np.diff(grid)
    )

    cdf /= cdf[-1]

    return np.interp(
        rng.random(n_events),
        cdf,
        grid,
    )


# ==============================================================
# 12. SINGLE EVENT
# ==============================================================

def simulate_event(E_nu, config, rng):

    E_prompt_true = positron_kinematics(E_nu)

    prompt_pe = simulate_photoelectrons(
        E_prompt_true,
        rng,
        config.light_yield_pe_per_MeV,
    )

    E_prompt_reco = reconstruct_energy(
        prompt_pe,
        config.light_yield_pe_per_MeV,
    )

    Tn_initial = neutron_initial_energy(
        E_nu,
        rng,
    )

    thermal_history, captured, Tn_final = simulate_neutron_thermalization(
        Tn_initial,
        rng,
    )

    # Capture time: we still sample from exponential, even if captured during thermalization.
    # A more detailed simulation would track time until capture.
    capture_time_us = sample_neutron_capture(
        rng,
        config.neutron_capture_tau_us,
    )

    capture_distance_m = neutron_diffusion_distance(
        capture_time_us,
        rng,
    )

    E_delayed_true = config.neutron_capture_energy_MeV

    delayed_pe = simulate_photoelectrons(
        E_delayed_true,
        rng,
        config.light_yield_pe_per_MeV,
    )

    E_delayed_reco = reconstruct_energy(
        delayed_pe,
        config.light_yield_pe_per_MeV,
    )

    return {
        "E_nu_MeV": E_nu,
        "E_prompt_true_MeV": E_prompt_true,
        "E_prompt_reco_MeV": E_prompt_reco,
        "prompt_pe": prompt_pe,
        "neutron_initial_MeV": Tn_initial,
        "neutron_thermal_MeV": Tn_final,
        "capture_time_us": capture_time_us,
        "capture_distance_m": capture_distance_m,
        "E_delayed_true_MeV": E_delayed_true,
        "E_delayed_reco_MeV": E_delayed_reco,
        "delayed_pe": delayed_pe,
    }


# ==============================================================
# 13. EVENT GENERATION
# ==============================================================

def generate_events(
    n_events,
    config,
    osc_params,
    seed=42,
):

    rng = np.random.default_rng(seed)

    energies = sample_neutrino_energies(
        n_events,
        1.806,
        12.0,
        config,
        osc_params,
        rng,
    )

    rows = [
        simulate_event(E, config, rng)
        for E in energies
    ]

    return pd.DataFrame(rows)


# ==============================================================
# 14. EVENT SELECTION
# ==============================================================

def apply_ibd_selection(events, config):

    selection = (
        (events["E_prompt_reco_MeV"] >= config.prompt_min_MeV)
        & (events["E_prompt_reco_MeV"] <= config.prompt_max_MeV)
        & (events["E_delayed_reco_MeV"] >= config.delayed_min_MeV)
        & (events["E_delayed_reco_MeV"] <= config.delayed_max_MeV)
        & (events["capture_time_us"] <= config.coincidence_window_us)
        & (events["capture_distance_m"] <= config.spatial_cut_m)
    )

    return events.loc[selection].copy()


# ==============================================================
# 15. ASIMOV SPECTRA + CHI2
# ==============================================================

def make_asimov_spectrum(
    config,
    hierarchy="NO",
    E_min=1.8,
    E_max=8.0,
    n_bins=240,
):

    params = OscillationParameters(
        hierarchy=hierarchy
    )

    edges = np.linspace(
        E_min,
        E_max,
        n_bins + 1,
    )

    centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    spectrum = expected_ibd_spectrum(
        centers,
        config,
        params,
    )

    spectrum /= spectrum.sum()

    return centers, spectrum


def poisson_chi2(observed, predicted):

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = predicted > 0

    return np.sum(
        (
            (observed[valid] - predicted[valid]) ** 2
        )
        / predicted[valid]
    )


def hierarchy_chi2_study(
    config,
    n_total_events=100_000,
):

    E, no = make_asimov_spectrum(
        config,
        "NO",
    )

    _, io = make_asimov_spectrum(
        config,
        "IO",
    )

    no *= n_total_events
    io *= n_total_events

    chi2_no_io = poisson_chi2(no, io)
    chi2_io_no = poisson_chi2(io, no)

    return {
        "energy_MeV": E,
        "spectrum_NO": no,
        "spectrum_IO": io,
        "NO_vs_IO": chi2_no_io,
        "IO_vs_NO": chi2_io_no,
        "sqrt_NO_vs_IO": np.sqrt(chi2_no_io),
        "sqrt_IO_vs_NO": np.sqrt(chi2_io_no),
    }


# ==============================================================
# 16. PDF REPORT HELPERS
# ==============================================================

def add_title_page(pdf, config, n_events, selected, Np, hierarchy):

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")

    fig.text(
        0.5, 0.80,
        "JUNO Reactor Antineutrino + Neutron Monte Carlo",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
    )

    fig.text(
        0.5, 0.72,
        "Progressive simulation report — Improved Physics",
        ha="center",
        va="center",
        fontsize=15,
    )

    summary = (
        f"Detector mass: {config.scintillator_mass_kt:.1f} kt\n"
        f"Generated MC events: {n_events:,}\n"
        f"Selected IBD-like events: {len(selected):,}\n"
        f"Selection efficiency: {len(selected)/max(n_events,1):.4f}\n"
        f"Estimated free protons: {Np:.3e}\n"
        f"Oscillation hierarchy used for MC: {hierarchy}"
    )

    fig.text(
        0.5, 0.49,
        summary,
        ha="center",
        va="center",
        fontsize=13,
        linespacing=1.7,
    )

    fig.text(
        0.5, 0.13,
        "Note: Huber–Mueller reactor spectrum, improved IBD cross section, "
        "and material‑dependent neutron scattering.",
        ha="center",
        va="center",
        fontsize=9,
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_text_page(pdf, title, lines):

    fig = plt.figure(figsize=(11.69, 8.27))

    fig.text(
        0.07, 0.90,
        title,
        fontsize=20,
        fontweight="bold",
    )

    y = 0.82

    for line in lines:
        fig.text(
            0.08, y,
            line,
            fontsize=11,
            family="DejaVu Sans",
        )
        y -= 0.045

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_dataframe_page(
    pdf,
    title,
    dataframe,
    max_rows=12,
):

    fig, ax = plt.subplots(
        figsize=(11.69, 8.27)
    )

    ax.axis("off")

    ax.set_title(
        title,
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    view = dataframe.head(max_rows).copy()

    table = ax.table(
        cellText=view.values,
        colLabels=view.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# PDF PLOTS

def add_reactor_spectrum_page(pdf, config):

    E = np.linspace(1.8, 10.0, 2000)

    model = ReactorSpectrum()

    spectrum = model.normalized_spectrum(
        E,
        config.fission_fractions,
    )

    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    ax.plot(E, spectrum, lw=2)

    ax.set_xlabel(r"$E_\nu$ [MeV]")
    ax.set_ylabel("Normalized spectrum")
    ax.set_title(
        "Huber–Mueller Reactor Antineutrino Spectrum",
        fontsize=18,
        fontweight="bold",
    )

    ax.grid(alpha=0.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_oscillation_page(pdf):

    E = np.linspace(1.8, 8.0, 4000)
    L = 53.0

    p_no = survival_probability(
        E, L, OscillationParameters(hierarchy="NO")
    )

    p_io = survival_probability(
        E, L, OscillationParameters(hierarchy="IO")
    )

    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    ax.plot(E, p_no, label="Normal ordering", lw=1.5)
    ax.plot(E, p_io, label="Inverted ordering", lw=1.5)

    ax.set_xlabel(r"$E_\nu$ [MeV]")
    ax.set_ylabel(r"$P(\bar{\nu}_e\rightarrow\bar{\nu}_e)$")

    ax.set_title(
        "JUNO Electron Antineutrino Survival Probability",
        fontsize=18,
        fontweight="bold",
    )

    ax.legend()
    ax.grid(alpha=0.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_prompt_spectrum_page(pdf, events, selected):

    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    ax.hist(
        events["E_prompt_reco_MeV"],
        bins=120,
        range=(0, 10),
        alpha=0.5,
        label="All generated",
    )

    if len(selected):
        ax.hist(
            selected["E_prompt_reco_MeV"],
            bins=120,
            range=(0, 10),
            alpha=0.7,
            label="Selected IBD-like",
        )

    ax.set_xlabel("Prompt reconstructed energy [MeV]")
    ax.set_ylabel("Events")
    ax.set_title(
        "JUNO-like Prompt Energy Spectrum",
        fontsize=18,
        fontweight="bold",
    )

    ax.legend()
    ax.grid(alpha=0.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_delayed_spectrum_page(pdf, events):

    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    ax.hist(
        events["E_delayed_reco_MeV"],
        bins=100,
        range=(1.0, 4.0),
    )

    ax.set_xlabel("Delayed reconstructed energy [MeV]")
    ax.set_ylabel("Events")

    ax.set_title(
        "Neutron-Capture Delayed Energy",
        fontsize=18,
        fontweight="bold",
    )

    ax.grid(alpha=0.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_capture_time_page(pdf, events):

    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    ax.hist(
        events["capture_time_us"],
        bins=100,
        range=(0, 1000),
        density=True,
    )

    ax.set_xlabel("Neutron capture time [μs]")
    ax.set_ylabel("Probability density")

    ax.set_title(
        "Neutron Capture-Time Distribution",
        fontsize=18,
        fontweight="bold",
    )

    ax.grid(alpha=0.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_neutron_energy_page(pdf, events):

    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    ax.hist(
        events["neutron_initial_MeV"] * 1000,
        bins=100,
        density=True,
    )

    ax.set_xlabel("Initial neutron kinetic energy [keV]")
    ax.set_ylabel("Probability density")

    ax.set_title(
        "IBD Neutron Initial Kinetic Energy",
        fontsize=18,
        fontweight="bold",
    )

    ax.grid(alpha=0.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_capture_distance_page(pdf, events):

    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    ax.hist(
        events["capture_distance_m"] * 100,
        bins=100,
        density=True,
    )

    ax.set_xlabel("Neutron displacement [cm]")
    ax.set_ylabel("Probability density")

    ax.set_title(
        "Neutron Capture Displacement",
        fontsize=18,
        fontweight="bold",
    )

    ax.grid(alpha=0.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_prompt_delayed_page(pdf, selected):

    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    if len(selected):

        h = ax.hist2d(
            selected["E_prompt_reco_MeV"],
            selected["E_delayed_reco_MeV"],
            bins=(80, 50),
            range=[
                (0, 10),
                (1.0, 4.0),
            ],
        )

        fig.colorbar(
            h[3],
            ax=ax,
            label="Events",
        )

    else:
        ax.text(
            0.5, 0.5,
            "No selected events",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Prompt reconstructed energy [MeV]")
    ax.set_ylabel("Delayed reconstructed energy [MeV]")

    ax.set_title(
        "Prompt-Delayed IBD Event Distribution",
        fontsize=18,
        fontweight="bold",
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_neutrino_neutron_correlation_page(pdf, events):

    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    ax.scatter(
        events["E_nu_MeV"],
        events["neutron_initial_MeV"] * 1000,
        s=3,
        alpha=0.2,
    )

    ax.set_xlabel(r"$E_\nu$ [MeV]")
    ax.set_ylabel(r"$T_n$ [keV]")

    ax.set_title(
        "Neutrino Energy vs. Initial Neutron Energy",
        fontsize=18,
        fontweight="bold",
    )

    ax.grid(alpha=0.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_hierarchy_page(pdf, result):

    E = result["energy_MeV"]

    no = result["spectrum_NO"]
    io = result["spectrum_IO"]

    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    ax.plot(
        E,
        no,
        label="Normal ordering",
        lw=1.5,
    )

    ax.plot(
        E,
        io,
        label="Inverted ordering",
        lw=1.5,
    )

    ax.set_xlabel(r"$E_\mathrm{prompt}$ [MeV]")
    ax.set_ylabel("Expected events / bin")

    ax.set_title(
        "JUNO Mass-Ordering Asimov Spectra",
        fontsize=18,
        fontweight="bold",
    )

    ax.legend()
    ax.grid(alpha=0.25)

    annotation = (
        f"χ²(NO vs IO) = {result['NO_vs_IO']:.3f}\n"
        f"sqrt(χ²) = {result['sqrt_NO_vs_IO']:.3f}"
    )

    ax.text(
        0.03,
        0.95,
        annotation,
        transform=ax.transAxes,
        va="top",
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ==============================================================
# 18. CREATE COMPLETE PDF REPORT
# ==============================================================

def create_pdf_report(
    pdf_path,
    config,
    events,
    selected,
    hierarchy_result,
    Np,
):

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PdfPages(pdf_path) as pdf:

        # Title
        add_title_page(
            pdf,
            config,
            len(events),
            selected,
            Np,
            "NO",
        )

        # Configuration / numerical summary
        lines = [
            "Simulation configuration",
            "",
            f"Scintillator mass = {config.scintillator_mass_kt:.2f} kt",
            f"Target radius = {config.target_radius_m:.2f} m",
            f"Energy resolution coefficient = {config.energy_resolution_coeff:.4f}",
            f"Light yield = {config.light_yield_pe_per_MeV:.1f} PE/MeV",
            f"Hydrogen-target approximation = 12% mass fraction",
            f"Neutron capture time constant = {config.neutron_capture_tau_us:.1f} us",
            f"Neutron capture gamma energy = {config.neutron_capture_energy_MeV:.3f} MeV",
            "",
            "Reactor configuration",
        ]

        for r in config.reactors:
            lines.append(
                f"{r.name}: {r.power_GWth:.2f} GWth, "
                f"{r.baseline_km:.2f} km"
            )

        lines += [
            "",
            "Fission fractions",
        ]

        for isotope, fraction in config.fission_fractions.items():
            lines.append(f"{isotope}: {fraction:.3f}")

        lines += [
            "",
            "IBD selection",
            f"Prompt: {config.prompt_min_MeV:.2f}-{config.prompt_max_MeV:.2f} MeV",
            f"Delayed: {config.delayed_min_MeV:.2f}-{config.delayed_max_MeV:.2f} MeV",
            f"Time window: {config.coincidence_window_us:.1f} us",
            f"Spatial cut: {config.spatial_cut_m:.2f} m",
            "",
            "Physics improvements:",
            "- Huber–Mueller reactor spectrum (polynomial parameterization)",
            "- IBD cross section with recoil and radiative corrections",
            "- Neutron thermalization with H and C elastic scattering cross sections",
            "- Capture on hydrogen using 1/v cross section",
        ]

        add_text_page(
            pdf,
            "Simulation Configuration",
            lines,
        )

        # Event summary
        summary_df = pd.DataFrame({
            "Quantity": [
                "Generated events",
                "Selected events",
                "Selection efficiency",
                "Mean neutrino energy [MeV]",
                "Mean prompt energy [MeV]",
                "Mean delayed energy [MeV]",
                "Mean neutron initial energy [keV]",
                "Mean capture time [us]",
                "Mean capture displacement [cm]",
                "Estimated free protons",
            ],
            "Value": [
                len(events),
                len(selected),
                len(selected) / max(len(events), 1),
                events["E_nu_MeV"].mean(),
                events["E_prompt_reco_MeV"].mean(),
                events["E_delayed_reco_MeV"].mean(),
                events["neutron_initial_MeV"].mean() * 1000.0,
                events["capture_time_us"].mean(),
                events["capture_distance_m"].mean() * 100.0,
                Np,
            ],
        })

        summary_display = summary_df.copy()
        summary_display["Value"] = summary_display["Value"].map(
            lambda x: f"{x:.6g}" if isinstance(x, float) else str(x)
        )

        add_dataframe_page(
            pdf,
            "Monte Carlo Result Summary",
            summary_display,
            max_rows=20,
        )

        # Event sample
        sample_cols = [
            "E_nu_MeV",
            "E_prompt_reco_MeV",
            "E_delayed_reco_MeV",
            "neutron_initial_MeV",
            "capture_time_us",
            "capture_distance_m",
        ]

        sample = events[sample_cols].head(12).copy()

        add_dataframe_page(
            pdf,
            "Sample of Generated Events",
            sample.round(6),
            max_rows=12,
        )

        # Main plots
        add_reactor_spectrum_page(pdf, config)
        add_oscillation_page(pdf)
        add_prompt_spectrum_page(pdf, events, selected)
        add_delayed_spectrum_page(pdf, events)
        add_neutron_energy_page(pdf, events)
        add_capture_time_page(pdf, events)
        add_capture_distance_page(pdf, events)
        add_neutrino_neutron_correlation_page(pdf, events)
        add_prompt_delayed_page(pdf, selected)
        add_hierarchy_page(pdf, hierarchy_result)

        # Final notes
        add_text_page(
            pdf,
            "Interpretation and Limitations",
            [
                "The simulation is designed as a progressive JUNO/neutrino-physics study.",
                "",
                "The reactor spectrum uses the Huber–Mueller polynomial parameterization,",
                "which is a good approximation but not the most up-to-date prediction.",
                "",
                "The IBD cross section includes recoil and a constant radiative correction,",
                "but does not include all higher-order corrections.",
                "",
                "The neutron thermalization now uses simplified energy-dependent cross sections",
                "for hydrogen and carbon, but still does not perform full spatial transport.",
                "Capture time is sampled from an exponential independent of the slowing-down process.",
                "",
                "The detector response uses a simplified JUNO-like photoelectron yield",
                "and energy-resolution model.",
                "",
                "Mass-ordering discrimination here is an idealized Asimov toy study.",
                "Real JUNO sensitivity requires systematics, nonlinearity, reactor model",
                "uncertainties, backgrounds, detector response, and proper fitting.",
            ],
        )

    return pdf_path


# ==============================================================
# 19. COMPLETE PIPELINE
# ==============================================================

def run_full_simulation(
    n_events=50_000,
    seed=42,
    output_dir="juno_output",
):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = JUNOConfig()

    print("=" * 70)
    print("JUNO Reactor Antineutrino + Neutron Monte Carlo")
    print("Improved physics version")
    print("=" * 70)

    print("\nGenerating events...")

    events = generate_events(
        n_events=n_events,
        config=config,
        osc_params=OscillationParameters(
            hierarchy="NO"
        ),
        seed=seed,
    )

    selected = apply_ibd_selection(
        events,
        config,
    )

    efficiency = (
        len(selected) / max(len(events), 1)
    )

    Np = estimate_free_protons(
        config.scintillator_mass_kt
    )

    hierarchy_result = hierarchy_chi2_study(
        config,
        n_total_events=100_000,
    )

    # Save CSV files
    events_csv = output_dir / "juno_neutron_events.csv"
    selected_csv = output_dir / "juno_selected_ibd_events.csv"

    events.to_csv(events_csv, index=False)
    selected.to_csv(selected_csv, index=False)

    # Save PDF
    pdf_path = output_dir / "JUNO_Neutron_MonteCarlo_Report.pdf"
    create_pdf_report(
        pdf_path=pdf_path,
        config=config,
        events=events,
        selected=selected,
        hierarchy_result=hierarchy_result,
        Np=Np,
    )

    # Terminal summary
    print("\nResults")
    print("-" * 70)

    print(f"Generated events        : {len(events):,}")
    print(f"Selected IBD events     : {len(selected):,}")
    print(f"Selection efficiency    : {efficiency:.5f}")

    print(
        "Mean neutron energy     : "
        f"{events['neutron_initial_MeV'].mean()*1000:.3f} keV"
    )

    print(
        "Mean capture time       : "
        f"{events['capture_time_us'].mean():.3f} us"
    )

    print(
        "Mean capture displacement: "
        f"{events['capture_distance_m'].mean()*100:.3f} cm"
    )

    print(
        "Hierarchy toy sqrt(chi2): "
        f"{hierarchy_result['sqrt_NO_vs_IO']:.3f}"
    )

    print("\nSaved files:")
    print(f"  {events_csv}")
    print(f"  {selected_csv}")
    print(f"  {pdf_path}")

    return {
        "events": events,
        "selected": selected,
        "hierarchy": hierarchy_result,
        "pdf": pdf_path,
    }


# CLI

def main():

    parser = argparse.ArgumentParser(
        description="JUNO neutron Monte Carlo + PDF report (improved physics)"
    )

    parser.add_argument(
        "--events",
        type=int,
        default=50_000,
        help="Number of Monte Carlo events.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="juno_output",
        help="Output directory.",
    )

    args = parser.parse_args()

    run_full_simulation(
        n_events=args.events,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
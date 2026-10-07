"""Shared helpers for the PharmaPy workshop notebooks.

apply_pharmapy_patches()
    Fixes for seven crystallizer and filter bugs in older PharmaPy versions. Each fix is
    applied only if a quick check shows that the installed PharmaPy still
    has the bug, so on a current PharmaPy this does nothing.
csd_quantiles(x_grid, distrib, dx)
    d10, d50, d90 and span of a finite-volume (1D-FVM) size distribution.
volume_density(cryst)
    Volume-weighted size density of a crystallizer's final CSD, for plotting.
plot_* functions
    Plotting helpers, so that everyone's figures look the same and answers are
    easy to compare.
"""

import inspect
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

from PharmaPy.Mechanisms import (PopulationBalanceMechanism,
                                 MomentsPopulationBalance, OneDFVMMechanism)
from PharmaPy.Crystallizers_Refactored import _BaseCrystallizer
from PharmaPy.DataClasses import PhaseRef

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, 'data', 'compounds_workshop.json')


# ---------------------------------------------------------------------------
# Fixes for older PharmaPy versions
# ---------------------------------------------------------------------------
#
# 1. Crystal inflow. PopulationBalanceMechanism.get_inlet_contributions
#    divided the slurry flow of a crystal-laden feed by the receiving SOLID
#    phase volume (the crystal volume) instead of the slurry volume, so the
#    second MSMPR of a cascade received far too many crystals (and none at all
#    while it was still empty). The MSMPR balance is
#    dn/dt = (Q / V_slurry) * (n_in - n).
#
# 2. Stale liquid. The population balance kept a reference to the liquid that
#    was in the vessel when CrystKinetics was set. When an upstream batch unit
#    hands over its liquid (SimulationExec), that reference was not updated,
#    so the slurry volume and crystal mass came from the placeholder liquid.
#
# 3. Empty population. Just as crystals first appear, the integrator can hold
#    a population with tiny negative densities. Asking it for zero crystal
#    mass raised an error and stalled the integrator.
#
# 4. "Inlet not detected". CheckModule warned that continuous and semibatch
#    units had no inlet, even when they did.
#
# 5. Semibatch dilution. The crystal population is a number density per m3
#    of slurry. A liquid feed into a semibatch crystallizer grows the volume,
#    which must dilute the population, dn/dt = -(Q_in / V) n; without that
#    term the crystal mass was overstated (about 2x in the flowsheet case).
#
# 6. Filter cycles. With slurry_div > 1 the Filter counted the filtrate
#    against one cycle's liquid while holding the whole charge, so no
#    filtrate ever left and the run went on until its time limit.
#
# 7. Cake crystals. The crystals of a filter cake are counted per m3 of
#    slurry, but their population balance was never linked to the filter's
#    liquid (nor kept its count as that liquid drained), so the cake reported
#    no crystal mass.

def _slurry_scaled_inlet(self, stream_phase, vessel_phase, amount,
                         completed_state):
    if not amount:
        return {}

    density = self._inlet_population_density(stream_phase)
    if density is None:
        return {}

    key = self._flow_state_key()
    state = completed_state.get(key)
    if state is None:
        state = getattr(self, self.flow_state_name)

    vol_slurry = self._slurry_volume(
        self.true_state(np.asarray(state, dtype=float)))
    if vol_slurry <= 0:
        return {}

    return {key: (amount / vol_slurry) * density}


_original_post_set_phases = _BaseCrystallizer._post_set_phases


def _post_set_phases_refresh_liquid(self):
    _original_post_set_phases(self)
    try:
        solid = self.Phases.get_phase_from_ref(PhaseRef('solid', 0))
        liquid = self.Phases.get_phase_from_ref(PhaseRef('liquid', 0))
    except (IndexError, AttributeError):
        return

    pbm = solid.get_mechanism(PopulationBalanceMechanism)
    if pbm is not None and getattr(pbm, 'liquid_phase', None) is not None:
        pbm.liquid_phase = liquid


_original_moments_set_mass = MomentsPopulationBalance.set_mass
_original_fvm_set_third_moment = OneDFVMMechanism.set_third_moment


def _moments_set_mass(self, value):
    moments = np.asarray(getattr(self, self.moments_state_name), dtype=float)
    if len(moments) >= 4 and self.true_state(moments)[3] <= 0:
        return
    return _original_moments_set_mass(self, value)


def _fvm_set_third_moment(self, target_m3):
    distrib = getattr(self, self.distribution_state_name)
    if self.compute_third_moment(self.true_state(distrib)) <= 0:
        return
    return _original_fvm_set_third_moment(self, target_m3)


_original_material_contributions = _BaseCrystallizer.calculate_material_contributions


def _material_contributions_with_dilution(self, time, completed_state,
                                          resolved_inlets, resolved_outlets):
    buffer = _original_material_contributions(self, time, completed_state,
                                              resolved_inlets, resolved_outlets)

    vol_in = sum(t.vol_flow or 0.0 for conn in resolved_inlets for t in conn)
    vol_out = sum(t.vol_flow or 0.0 for conn in resolved_outlets for t in conn)
    net_vol_flow = vol_in - vol_out
    if not net_vol_flow:
        return buffer

    for phase in self.Phases.Phases:
        for pbm in getattr(phase, 'mechanisms', None) or ():
            if not isinstance(pbm, PopulationBalanceMechanism):
                continue

            key = pbm._flow_state_key()
            state = completed_state.get(key)
            if state is None:
                state = getattr(pbm, pbm.flow_state_name)
            state = np.asarray(state, dtype=float)

            vol_slurry = pbm._slurry_volume(pbm.true_state(state))
            state_slice = self.solver_state_collection.material_slices.get(key)
            if vol_slurry > 0 and state_slice is not None:
                buffer.contributions[buffer.INLET, state_slice] -= \
                    (net_vol_flow / vol_slurry) * state

    return buffer


def _filter_liquid_held(completed_state):
    from PharmaPy.DataClasses import StateKey
    return np.sum(np.asarray(completed_state[StateKey('mass_j', PhaseRef('liquid', 0))], dtype=float))


def _patch_filter_cycles():
    from PharmaPy.SolidLiquidSep_Refactored import Filter
    original_constants = Filter.compute_filtration_constants

    def compute_filtration_constants(self):
        original_constants(self)
        self.mass_liquid_init *= self.slurry_div      # the vessel holds the whole charge

    def filtrate_mass(self, completed_state):         # filtrate of one cycle
        return (self.mass_liquid_init - _filter_liquid_held(completed_state)) / self.slurry_div

    def filtrate_volumetric_flow(self, completed_state):   # all cycles drain at once
        return self.slurry_div * self.filtrate_mass_flow(completed_state) / self.dens_liq

    Filter.compute_filtration_constants = compute_filtration_constants
    Filter.filtrate_mass = filtrate_mass
    Filter.filtrate_volumetric_flow = filtrate_volumetric_flow


def _patch_filter_cake():
    from PharmaPy.SolidLiquidSep_Refactored import Filter
    original_constants = Filter.compute_filtration_constants
    original_results = Filter.retrieve_results

    def phases(filt):
        liquid = filt.Phases.get_phase_from_ref(PhaseRef('liquid', 0))
        solid = filt.Phases.get_phase_from_ref(PhaseRef('solid', 0))
        return liquid, solid, solid.get_mechanism(PopulationBalanceMechanism)

    def compute_filtration_constants(self):
        original_constants(self)
        liquid, solid, pbm = phases(self)
        self._workshop_vol_crystals = None
        if pbm is not None:
            pbm.liquid_phase = liquid
            state = pbm.true_state(np.asarray(getattr(pbm, pbm.flow_state_name), dtype=float))
            self._workshop_vol_slurry = pbm.slurry_volume_from(liquid, state)
            self._workshop_vol_crystals = pbm.solid_volume_fraction(state) * self._workshop_vol_slurry

    def retrieve_results(self, time, solver_states):
        result = original_results(self, time, solver_states)
        vol_crystals = getattr(self, '_workshop_vol_crystals', None)
        liquid, _, pbm = phases(self)
        if vol_crystals and pbm is not None:
            # keep the crystal count as the liquid around the crystals drains
            pbm.liquid_phase = liquid
            factor = self._workshop_vol_slurry / (float(liquid.vol) + vol_crystals)
            state = np.asarray(getattr(pbm, pbm.flow_state_name), dtype=float)
            setattr(pbm, pbm.flow_state_name, state * factor)
            self._workshop_vol_crystals = None
        return result

    Filter.compute_filtration_constants = compute_filtration_constants
    Filter.retrieve_results = retrieve_results


# ---- checks: does the installed PharmaPy still have each bug? -------------

def _test_kinetics():
    from PharmaPy.Kinetics import CrystKinetics
    return CrystKinetics(np.array([-28.13909202, 0.001, 5.900800253]),
                         nucl_prim=(1.0448e17, 0, 8.30671806),
                         growth=(2.4297e4, 0, 4.52241037),
                         solubility_type='apelblat',
                         solubility_basis='mass_per_volume_solvent')


def _test_solid(distrib):
    from PharmaPy.Phases_Refactored import SolidPhase
    solid = SolidPhase(_DATA, temp=313.15, mass=0, mass_frac=[0, 0, 1, 0, 0])
    solid.mechanisms = OneDFVMMechanism(solid, target_components='C', solvent_name='solvent',
                                        x_grid=np.geomspace(1, 1500, 20), distrib_init=distrib)
    return solid


def _test_liquid(mass=1.0):
    from PharmaPy.Phases_Refactored import LiquidPhase
    return LiquidPhase(_DATA, temp=313.15, mass=mass, mass_frac=[0, 0, 0.2, 0, 0.8],
                       name_solv='solvent')


def _test_msmpr(with_feed):
    from PharmaPy.Crystallizers_Refactored import ContinuousCrystallizer
    from PharmaPy.ProcessControl_Refactored import ContinuousVesselController
    from PharmaPy.Streams_Refactored import LiquidStream

    cryst = ContinuousCrystallizer(integrator=None,
                                   controller=ContinuousVesselController(temp_func=lambda t: 313.15))
    cryst.Phases = [_test_liquid(), _test_solid(np.zeros(20))]
    cryst.CrystKinetics = _test_kinetics()
    if with_feed:
        cryst.Inlet = LiquidStream(_DATA, temp=313.15, vol_flow=1e-6,
                                   mass_frac=[0, 0, 0.2, 0, 0.8], name_solv='solvent')
    return cryst


def _has_inflow_bug():
    """An empty vessel must still receive the crystals of its feed."""
    cryst = _test_msmpr(with_feed=False)
    solid = cryst.Phases.get_phase_from_ref(PhaseRef('solid', 0))
    pbm = solid.get_mechanism(PopulationBalanceMechanism)
    feed = _test_solid(np.full(20, 1e6))
    return not pbm.get_inlet_contributions(feed, solid, 1e-6, {})


def _has_stale_liquid_bug():
    """The population balance must follow a liquid handed over by SimulationExec."""
    from PharmaPy.Crystallizers_Refactored import BatchCrystallizer
    from PharmaPy.MixedPhases_Refactored import MixedPhase
    from PharmaPy.ProcessControl_Refactored import SimpleTemperatureController

    cryst = BatchCrystallizer(integrator=None,
                              controller=SimpleTemperatureController(temp_func=lambda t: 313.15))
    cryst.Phases = [_test_liquid(), _test_solid(np.zeros(20))]
    cryst.CrystKinetics = _test_kinetics()

    handed_over = MixedPhase([_test_liquid(mass=2.0)])
    handed_over.transferred_from_uo = True
    cryst.Phases = handed_over

    solid = cryst.Phases.get_phase_from_ref(PhaseRef('solid', 0))
    liquid = cryst.Phases.get_phase_from_ref(PhaseRef('liquid', 0))
    return solid.get_mechanism(PopulationBalanceMechanism).liquid_phase is not liquid


def _has_empty_population_bug():
    """Zero crystal mass must be accepted by a population with tiny negative noise."""
    noise = np.zeros(20)
    noise[[5, 10]] = [1e-5, -1e-5]
    pbm = _test_solid(noise).get_mechanism(OneDFVMMechanism)
    try:
        pbm.set_third_moment(0.0)
    except ValueError:
        return True
    return False


def _has_inlet_warning_bug():
    """A unit with a feed must not be reported as missing its Inlet."""
    from PharmaPy.CheckModule import check_modeling_objects

    cryst = _test_msmpr(with_feed=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        check_modeling_objects(cryst, 'CR01')
    return any('Inlet' in str(w.message) for w in caught)


def _has_dilution_bug():
    """A liquid feed must dilute the crystals of a semibatch crystallizer."""
    from PharmaPy.Crystallizers_Refactored import SemiBatchCrystallizer
    from PharmaPy.Streams_Refactored import LiquidStream

    # Undersaturated (313 K), so crystallization is idle and only the feed acts.
    cryst = SemiBatchCrystallizer(integrator=None, isothermal=True)
    cryst.Phases = [_test_liquid(), _test_solid(np.full(20, 1e6))]
    cryst.CrystKinetics = _test_kinetics()
    cryst.Inlet = LiquidStream(_DATA, temp=313.15, vol_flow=1e-6,
                               mass_frac=[0, 0, 0.2, 0, 0.8], name_solv='solvent')
    cryst.compile_structure()

    states = cryst.create_solver_init_states()
    rates = np.asarray(cryst.unit_model(0.0, states, mat_bce=True))
    collection = cryst.solver_state_collection
    key = next(k for k in collection.states if k.name == 'distrib')
    return np.all(rates[collection.slices[key]] == 0)


def _test_filter(slurry_div):
    """A small filter charge (0.05 kg of crystals per litre), solved."""
    from PharmaPy.IntegratorBackends import ScipyBackend
    from PharmaPy.Phases_Refactored import SolidPhase
    from PharmaPy.SolidLiquidSep_Refactored import Filter

    x_grid = np.geomspace(1, 1500, 20)
    solid = SolidPhase(_DATA, temp=313.15, mass=0, mass_frac=[0, 0, 1, 0, 0])
    solid.mechanisms = OneDFVMMechanism(solid, target_components='C', solvent_name='solvent', x_grid=x_grid,
                                        distrib_init=np.exp(-0.5 * (np.log(x_grid / 100) / 0.3)**2),
                                        distrib_basis='vol_perc', basis_mass=0.05 * slurry_div,
                                        vol_slurry=1e-3 * slurry_div)
    filt = Filter(0.05, deltaP=1e5, alpha=1e10, slurry_div=slurry_div, integrator=ScipyBackend())
    filt.Phases = [_test_liquid(mass=0.9 * slurry_div), solid]
    filt.solve_unit(runtime=100, verbose=False)
    return filt


def _has_filter_cycles_bug():
    """A charge split into two filtration cycles must still drain."""
    held = _test_filter(slurry_div=2).result.mass_j_liquid0.sum(axis=1)
    return held[-1] > 0.9 * held[0]


def _has_filter_cake_bug():
    """The cake must keep the 0.05 kg of crystals that were filtered."""
    return _test_filter(slurry_div=1).Outlet['cake'].Solids[0].mass < 0.025


def _check(has_bug):
    # If the check itself fails (e.g. a very different PharmaPy version),
    # assume the bug is there: the fix is what the notebooks were tested with.
    try:
        return has_bug()
    except Exception:
        return True


def apply_pharmapy_patches(verbose=True):
    """Fix the crystallizer and filter bugs described above, where still present."""
    applied = []

    if _check(_has_stale_liquid_bug):
        _BaseCrystallizer._post_set_phases = _post_set_phases_refresh_liquid
        applied.append('stale liquid')

    if _check(_has_inflow_bug):
        PopulationBalanceMechanism.get_inlet_contributions = _slurry_scaled_inlet
        applied.append('crystal inflow')

    if _check(_has_empty_population_bug):
        MomentsPopulationBalance.set_mass = _moments_set_mass
        OneDFVMMechanism.set_third_moment = _fvm_set_third_moment
        applied.append('empty population')

    if _check(_has_dilution_bug):
        _BaseCrystallizer.calculate_material_contributions = _material_contributions_with_dilution
        applied.append('semibatch dilution')

    if _check(_has_inlet_warning_bug):
        # Only the message where Inlet is the sole missing object.
        warnings.filterwarnings(
            'ignore', category=UserWarning, module='PharmaPy.CheckModule',
            message=r'The following PharmaPy modeling objects were not detected in [^\n]*:\n\tInlet\.\n')
        applied.append('inlet warning')

    if _check(_has_filter_cycles_bug):
        _patch_filter_cycles()
        applied.append('filter cycles')

    if _check(_has_filter_cake_bug):
        _patch_filter_cake()
        applied.append('cake crystals')

    if verbose:
        if applied:
            print('PharmaPy workshop fixes applied: %s.' % ', '.join(applied))
        else:
            print('Your PharmaPy version needs no workshop fixes.')

    return applied


# ---------------------------------------------------------------------------
# Crystal size distribution statistics
# ---------------------------------------------------------------------------

def csd_quantiles(x_grid, distrib, dx):
    """Volume-weighted d10, d50, d90 [um] and span of a 1D-FVM distribution.

    Parameters
    ----------
    x_grid : array
        Size grid [um] (``mechanism.x_grid``).
    distrib : array
        Number density [#/(um m3)] on that grid, e.g.
        ``result.distrib_solid0[-1]``.
    dx : float or array
        Cell widths [um] (``mechanism.dx``).

    Returns
    -------
    dict
        ``d10``, ``d50``, ``d90`` [um] and ``span`` [-]. All NaN if there are
        no crystals.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    vol_density = np.clip(np.asarray(distrib, dtype=float), 0, None) * x_grid**3 * dx

    total = vol_density.sum()
    if not total > 0:
        return {'d10': np.nan, 'd50': np.nan, 'd90': np.nan, 'span': np.nan}

    cumulative = np.cumsum(vol_density) / total
    d10, d50, d90 = np.interp([0.1, 0.5, 0.9], cumulative, x_grid)

    return {'d10': d10, 'd50': d50, 'd90': d90, 'span': (d90 - d10) / d50}


def volume_density(cryst, time_index=-1):
    """Volume-weighted density q3(ln L) of a 1D-FVM crystallizer, for plotting.

    Returns the size grid [um] and q3 [-], normalized so that its integral
    over ln(L) is 1 (so distributions with different crystal masses can be
    compared by shape).
    """
    mech = cryst.Phases.Solids[0].mechanisms[0]
    x_grid = np.asarray(mech.x_grid, dtype=float)
    distrib = np.clip(cryst.result.distrib_solid0[time_index], 0, None)
    vol = distrib * x_grid**3 * mech.dx
    if not vol.sum() > 0:
        return x_grid, np.zeros_like(x_grid)
    return x_grid, vol / vol.sum() / np.gradient(np.log(x_grid))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

SPECIES_ABCD = ['A', 'B', 'C', 'D']


def plot_concentrations(time, conc, ax=None, title=None, time_unit='s'):
    """Concentration profiles of A, B, C and D (first four columns of conc)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    scale = {'s': 1, 'min': 60, 'h': 3600}[time_unit]
    ax.plot(np.asarray(time) / scale, np.asarray(conc)[:, :4])
    ax.legend(SPECIES_ABCD)
    ax.set_xlabel('time (%s)' % time_unit)
    ax.set_ylabel('$C_j$ (mol/L)')
    if title:
        ax.set_title(title)
    return ax


def plot_fit(t_exp, c_exp, t_model, c_model, ax=None, title=None, c_seed=None):
    """Measured (markers) and predicted (lines) concentrations of B and C.

    c_model (and the optional c_seed, dashed) have one column per species
    (A, B, C, D, solvent); c_exp has two columns (B, C).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    lines = ax.plot(t_model, np.asarray(c_model)[:, [1, 2]])
    for line, measured in zip(lines, np.asarray(c_exp).T):
        ax.plot(t_exp, measured, 'o', mfc='None', color=line.get_color())
    if c_seed is not None:
        ax.plot(t_model, np.asarray(c_seed)[:, [1, 2]], 'k--', alpha=0.5)
    ax.legend(lines, ['B', 'C'])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('$C_j$ (mol/L)')
    if title:
        ax.set_title(title)
    return ax


def plot_arrhenius(temps, ln_k, ln_A, Ea_R, rxn_names):
    """ln(k_i) vs 1000/T, with the regression line ln(A_i) - (E_i/R)/T.

    ln_k has one row per temperature and one column per reaction.
    """
    temps = np.asarray(temps, dtype=float)
    ln_k = np.asarray(ln_k, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))
    inv_T = np.linspace(1 / temps.max(), 1 / temps.min(), 20)
    for ind, ax in enumerate(axes.flat[:len(rxn_names)]):
        ax.plot(1000 / temps, ln_k[:, ind], 'o', mfc='None')
        ax.plot(1000 * inv_T, ln_A[ind] - Ea_R[ind] * inv_T, '--')
        ax.set_title(rxn_names[ind])
        ax.set_xlabel('1000/T (1/K)')
        ax.set_ylabel(r'$\ln(k_%i)$' % (ind + 1))
    fig.tight_layout()
    plt.show()


def plot_intervals(names, estimate, half_width):
    """Estimates with their confidence intervals (error bars)."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    pos = np.arange(len(names))
    ax.errorbar(pos, estimate, yerr=half_width, fmt='o', capsize=4)
    ax.set_xticks(pos, ['$%s$' % name for name in names], rotation=90)
    ax.set_ylabel('estimate and 95% CI')
    fig.tight_layout()
    plt.show()


def plot_correlation(corr, names):
    """Heat map of a parameter correlation matrix."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    labels = ['$%s$' % name for name in names]
    ax.set_xticks(range(len(names)), labels, rotation=90)
    ax.set_yticks(range(len(names)), labels)
    fig.colorbar(im, label='correlation')
    fig.tight_layout()
    plt.show()


def plot_batches(runs, axes=None):
    """Temperature, supersaturation and final CSD of 1D-FVM batch crystallizers.

    runs : dict {label: crystallizer}
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    else:
        fig = axes[0].figure
    for label, cryst in runs.items():
        res = cryst.result
        axes[0].plot(res.time / 60, res.global_temp, label=label)
        axes[1].plot(res.time / 60, np.ravel(res.supersat), label=label)
        axes[2].semilogx(*volume_density(cryst), label=label)

    axes[0].set_ylabel('T (K)')
    axes[1].set_ylabel('supersaturation S (-)')
    axes[2].set_ylabel(r'volume density $q_3(\ln L)$')
    axes[0].set_xlabel('time (min)')
    axes[1].set_xlabel('time (min)')
    axes[2].set_xlabel(r'size ($\mu$m)')
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_csds(runs, ax=None, title=None):
    """Final volume-weighted CSDs of several 1D-FVM crystallizers.

    runs : dict {label: crystallizer}
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for label, cryst in runs.items():
        ax.semilogx(*volume_density(cryst), label=label)
    ax.set_xlabel(r'size ($\mu$m)')
    ax.set_ylabel(r'volume density $q_3(\ln L)$')
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    plt.show()


def plot_profiles(temp_funcs, t_end, ax=None):
    """Temperature profiles T(t) before simulating them.

    temp_funcs : dict {label: function of time in s returning K}
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    time = np.linspace(0, t_end, 300)
    for label, func in temp_funcs.items():
        ax.plot(time / 60, [func(t) for t in time], label=label)
    ax.set_xlabel('time (min)')
    ax.set_ylabel('T (K)')
    ax.legend(fontsize=8)
    plt.show()


def plot_kpi_bars(table, columns=None):
    """Horizontal bar charts of KPIs, one panel per column of a DataFrame.

    The index of the DataFrame labels the bars; the first row is highlighted.
    """
    columns = list(table.columns) if columns is None else columns
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 0.5 * len(table) + 1.5),
                             sharey=True)
    axes = np.atleast_1d(axes)
    labels = [str(label) for label in table.index]
    for ax, col in zip(axes, columns):
        ax.barh(labels, table[col], color=['C3'] + ['C0'] * (len(table) - 1))
        ax.set_xlabel(col)
    axes[0].invert_yaxis()
    fig.tight_layout()
    plt.show()


def plot_pareto(points, front, x, y, xlabel=None, ylabel=None):
    """All evaluated designs (grey) and a Pareto front (red line).

    points, front : DataFrames with columns x and y.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(points[x], points[y], 'o', color='0.7', label='evaluated designs')
    front = front.sort_values(x)
    ax.plot(front[x], front[y], 'o-', color='C3', label='Pareto front')
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.legend()
    fig.tight_layout()
    plt.show()

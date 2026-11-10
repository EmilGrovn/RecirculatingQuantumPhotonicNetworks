"""Figure of the CZ gate with a finite round-trip delay (one column, style of the paper's figures).

(a) Minimal CZ duration T_min(tau) at infidelity <= 1e-3 (results/tmin.json) for kappa_m <= 240 chi3 (blue) and
    kappa_m <= 15 chi3 (grey), with the causal bound T_CZ + 2 tau and the SLH-optimised gate. Circles mark (b) and (c).
(b) kappa_max = 15, tau = 0.001, T = T_min = 0.835 (smoothed): couplings kappa_m(t) of block A (kappa_1 = kappa_3,
    kappa_4 = kappa_2 by the mirror symmetry), photon number in the waveguide loop for the computational inputs
    (|00>: cav 1+3, |01> = |10> (mirror): cav 1+4 / 2+3, |11>: cav 4+2), averaged with a Gaussian of rms width 4 tau,
    and the gate fidelity F(t), with the SLH model (same controls, tau = 0) as symbols.
(c) kappa_max = 240, tau = 0.26, T = 1.38 (smoothed): couplings and loop photon number.

usage: python delay_model/scripts/plot_delay_figure.py
Reads results/tmin.json and results/trajectories/*.json (scripts/trajectories.jl). Writes figures/fig_delay.{pdf,png}.
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures')
TRAJ_B = 'cz_k15_tau0.0010_T0.8350_smoothed.json'      # RQPN_NSAMPLE=1000000 (every time step)
TRAJ_C = 'cz_k240_tau0.2600_T1.3800_smoothed.json'     # RQPN_NSAMPLE=2500
KMAX = 240.0
T_NL = 0.754          # pi/4 - asin(sqrt(1e-3)): fastest CZ at tau = 0 with infidelity 1e-3
T_SLH = 0.7737        # the paper's SLH-optimised CZ (kappa_m <= 15 chi3)

# ---------------------------------------------------------------- style
def first_font(*names):
    """First installed font family of `names` (matplotlib warns about every missing family in a list)."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in names if n in have), 'DejaVu Serif')


BLUE, RED, GRID = '#0000FF', '#FF0000', '#DDDDDD'
POP = ('k', '#2687BD', '#DB7626')
POP_LS = ('-', (0, (1.0, 1.3)), (0, (3.2, 2.2)))
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': [first_font('cmr10')], 'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True, 'axes.unicode_minus': True,
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.9, 'lines.linewidth': 1.3,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
    'xtick.major.width': 0.9, 'ytick.major.width': 0.9, 'xtick.major.pad': 2.0, 'ytick.major.pad': 2.0,
    'axes.grid': True, 'grid.linestyle': ':', 'grid.color': GRID, 'grid.linewidth': 0.7,
    'axes.labelpad': 2.0,
    'legend.frameon': True, 'legend.fancybox': True, 'legend.edgecolor': '0.25', 'legend.framealpha': 1.0,
    'legend.borderpad': 0.3, 'legend.handlelength': 1.8, 'legend.handletextpad': 0.5,
    'legend.columnspacing': 1.0, 'legend.labelspacing': 0.25,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600,
})
LETTER = dict(fontfamily=first_font('Times New Roman', 'Times', 'Nimbus Roman', 'TeX Gyre Termes'), fontsize=10,
              va='top', ha='left')                  # panel letters in Times, as in the paper
SMALL_LEG = dict(fontsize=6.3, handlelength=1.4, handletextpad=0.4, borderpad=0.25, labelspacing=0.15, columnspacing=0.8)


def frame_legend(leg):
    leg.get_frame().set_linewidth(0.7)
    return leg


# ---------------------------------------------------------------- (a) data
rows = json.load(open(os.path.join(RES, 'tmin.json')))['rows']


def curve(k):
    r = sorted((x for x in rows if abs(x['kappa_max'] - k) < 1e-9), key=lambda x: x['tau_nominal'])
    return np.array([x['tau_nominal'] for x in r]), np.array([x['T_min'] for x in r])


tmin_t, tmin_T = curve(KMAX)
ref_t, ref_T = curve(15.0)


# ---------------------------------------------------------------- (b), (c) data
def overlaps(src):
    # a(t) = <input (x) vac|psi(t)> of block A; |00> = one photon in cav 1 and one in its mirror image 3 => a1^2,
    # |01> = |10> => a12, |11> => -a2^2;  F(t) = |sum_n <target_n|psi_n(t)> / 4|^2 (= 1 - I at t = T, 1/4 at t = 0)
    amp = lambda k: np.r_[1.0, np.array(src[k]['a_re']) + 1j * np.array(src[k]['a_im'])]
    a1, a2, a12 = amp('single1'), amp('single2'), amp('double')
    return abs((a1 ** 2 + 2 * a12 - a2 ** 2) / 4) ** 2


def load_traj(f):
    d = json.load(open(os.path.join(RES, 'trajectories', f)))
    kap = np.array(d['kappa'])                       # rows: block cavities 1, 2
    T = d['T']
    s1, s2, dd = d['single1'], d['single2'], d['double']
    t = np.r_[0.0, s1['t']]                          # prepend t = 0 (all photons in the cavities)
    w10 = np.r_[0.0, 1 - np.array(s1['n1']) - np.array(s1['n2'])]
    w01 = np.r_[0.0, 1 - np.array(s2['n1']) - np.array(s2['n2'])]
    w11 = np.r_[0.0, 2 - np.array(dd['n1']) - np.array(dd['n2'])]
    e = d['slh']
    return dict(T=T, tau=d['tau'], I=d['infidelity'], edges=np.linspace(0, T, kap.shape[1] + 1), kap=kap, t=t,
                n00=2 * w10, n01=w11, n11=2 * w01, fgate=overlaps(d),
                slh=dict(I=e['infidelity'], t=np.r_[0.0, e['single1']['t']], fgate=overlaps(e)))


small, large = load_traj(TRAJ_B), load_traj(TRAJ_C)
for nm, d in (('(b)', small), ('(c)', large)):
    print(f'{nm}: T = {d["T"]:.3f}, tau = {d["tau"]:.4f}, I = {d["I"]:.2e}, I_SLH = {d["slh"]["I"]:.2e}, '
          f'max kappa = {d["kap"].max():.2f}, max n_WG = {max(d["n00"].max(), d["n01"].max(), d["n11"].max()):.3f}')

# (b): <n_WG> low-pass filtered (Gaussian, rms width 4 tau). The instantaneous value carries a ripple (about 35% of the
# maximum) from loop modes excited by the piecewise-constant control steps; it converges in dt (physical in the model)
# but lies outside the SLH description. Its slowest component is the beat with the loop mode nearest to the cavities,
# at theta_k/tau with cot(theta_k/2) = 2 lambda_k (lambda_k: eigenvalues of C), 630/chi3 here (period about 10 tau);
# the filter suppresses it to 4%. The trajectory must be sampled at every time step (no aliasing).
dts = small['t'][2] - small['t'][1]
sig = 4 * small['tau'] / dts                         # in samples
ker = np.exp(-0.5 * (np.arange(-int(4 * sig), int(4 * sig) + 1) / sig) ** 2)
ker /= ker.sum()
h = len(ker) // 2
for k in ('n00', 'n01', 'n11'):
    x = small[k]
    xa = np.convolve(np.pad(x, h, mode='edge'), ker, mode='valid')
    print(f'(b) {k}: Gaussian low-pass (sigma = {sig:.0f} samples = 4 tau) removes a ripple of '
          f'{np.abs(x - xa)[2 * h:-2 * h].max() / x.max():.1%} of the maximum')
    small[k] = xa

# ---------------------------------------------------------------- layout
W, H = 3.40, 5.05                                   # revtex two-column: \columnwidth = 246 pt
ratios = [1.50, 0.62, 0.76, 0.76, 0.76, 0.50, 0.76, 0.76]
fig = plt.figure(figsize=(W, H))
gs = fig.add_gridspec(len(ratios), 1, left=0.155, right=0.975, top=1 - 0.015 * 4.45 / H, bottom=0.072 * 4.45 / H,
                      height_ratios=ratios, hspace=0.0)
ax = fig.add_subplot(gs[0])
bk, bn, bf = fig.add_subplot(gs[2]), fig.add_subplot(gs[3]), fig.add_subplot(gs[4])
ck, cn = fig.add_subplot(gs[6]), fig.add_subplot(gs[7])

# ---------------------------------------------------------------- (a)
TAULIM = (6e-4, 0.5)
tt = np.geomspace(*TAULIM, 80)
ax.plot(tt, T_NL + 2 * tt, color='k', ls='-', lw=1.0, label=r'$T_{\rm CZ}+2\tau$')
ax.axhline(T_SLH, color='k', ls='--', lw=1.0, label=r'SLH ($\kappa_m\leq15\chi_3$)')
ax.plot(ref_t, ref_T, 's-', color='0.6', ms=2.4, lw=1.0, clip_on=False, zorder=3, label=r'WQED ($\kappa_m\leq15\chi_3$)')
ax.plot(tmin_t, tmin_T, 's-', color=BLUE, ms=3.0, lw=1.3, clip_on=False, zorder=4, label=rf'WQED ($\kappa_m\leq{KMAX:g}\chi_3$)')
for d, lab, ct, cT in ((small, '(b)', ref_t, ref_T), (large, '(c)', tmin_t, tmin_T)):
    j = np.argmin(abs(ct - d['tau']))
    assert abs(ct[j] - d['tau']) < 2e-3, (lab, ct[j], d['tau'])
    if abs(cT[j] - d['T']) > 1e-6:
        print(f'note {lab}: shown solution T = {d["T"]:.3f}, curve T_min = {cT[j]:.3f}; circle at the solution')
    ax.plot(ct[j], d['T'], 'o', ms=7.5, mfc='none', mec='k', mew=1.0, zorder=5, clip_on=False)
    ax.annotate(lab, (ct[j], d['T']), xytext=(0, 7), textcoords='offset points', ha='center', va='bottom', color='k',
                family=LETTER['fontfamily'], fontsize=8.5)
ax.set_xscale('log')
ax.set_xlim(*TAULIM)
ax.set_ylim(0.6, 2.45)
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.set_xlabel(r'Round-trip delay $\tau\chi_3$')
ax.set_ylabel(r'$T\chi_3$')
frame_legend(ax.legend(loc='upper left'))

# ---------------------------------------------------------------- (b), (c)
TX = max(small['T'], large['T'])                    # common time axis
POPS = (('n00', r'$|00\rangle$'), ('n01', r'$|01\rangle, |10\rangle$'), ('n11', r'$|11\rangle$'))


def slh_marks(a, t, y, col, j, n=26):
    # SLH model as open symbols on top of the WQED lines
    k = np.unique(np.round(np.linspace(0, len(t) - 1, n) + (j - 1) * len(t) / (3.0 * n)).astype(int).clip(0, len(t) - 1))
    a.plot(t[k], y[k], ls='none', marker='o', ms=2.6, mfc='none', mec=col, mew=0.65, zorder=4)


def traj_panels(axk, axn, d, kmax, kap_leg, pop_leg, nlim, nstep, bottom=True):
    # kappa_4 dashed: the two couplings coincide on the kappa_max plateaus
    axk.stairs(d['kap'][0], d['edges'], color=BLUE, baseline=None, label=r'$\kappa_1=\kappa_3$')
    axk.stairs(d['kap'][1], d['edges'], color=RED, ls=(0, (3.2, 1.6)), baseline=None, label=r'$\kappa_4=\kappa_2$')
    for (key, lab), col, ls in zip(POPS, POP, POP_LS):
        axn.plot(d['t'], d[key], color=col, ls=ls, label=lab)
    for a in (axk, axn):
        a.set_xlim(0, TX)
        a.xaxis.set_major_locator(MultipleLocator(0.25))
    axk.set_ylim(-0.053 * kmax, 1.3 * kmax)
    axk.yaxis.set_major_locator(MultipleLocator(100 if kmax > 50 else 5))
    axn.set_ylim(*nlim)
    axn.yaxis.set_major_locator(MultipleLocator(nstep))
    axk.tick_params(labelbottom=False)
    axk.set_ylabel(r'$\kappa_m(t)/\chi_3$')
    axn.set_ylabel(r'$\langle\hat n_{\rm WG}\rangle$')
    if bottom:
        axn.set_xlabel(r'$t\chi_3$', labelpad=1.0)
    else:
        axn.tick_params(labelbottom=False)
    if kap_leg is not None:
        frame_legend(axk.legend(**kap_leg, **SMALL_LEG))
    if pop_leg is not None:
        frame_legend(axn.legend(**pop_leg, **SMALL_LEG))


def fid_panel(a, d):
    # gate fidelity F(t): WQED line, SLH (same controls, tau = 0) symbols
    a.plot(d['t'], d['fgate'], color='k', zorder=3)
    slh_marks(a, d['slh']['t'], d['slh']['fgate'], POP[2], 1, n=34)
    a.set_xlim(0, TX)
    a.xaxis.set_major_locator(MultipleLocator(0.25))
    a.set_ylim(-0.08, 1.2)
    a.yaxis.set_major_locator(MultipleLocator(0.5))
    a.set_ylabel(r'$\mathcal{F}(t)$')
    a.set_xlabel(r'$t\chi_3$', labelpad=1.0)
    hs = [Line2D([], [], color='k', label='WQED'),
          Line2D([], [], ls='none', marker='o', ms=2.6, mfc='none', mec=POP[2], mew=0.65, label=r'SLH ($\tau=0$)')]
    frame_legend(a.legend(handles=hs, loc='center right', ncol=1, **SMALL_LEG))


def mark_times(axn, d, y=1.0):
    # strong coupling: both photons travel the loop together (<n_WG> = 2 for a time tau), then sit in the cavities
    # while the CZ phase accrues; edges = half-maximum crossings of <n_WG> for |01>
    n = d['n01']
    up = np.flatnonzero(np.diff((n > 0.5 * n.max()).astype(int)))
    t0, t1, t2 = d['t'][up[:3]]
    for a, b, lab in ((t0, t1, r'$\tau$'), (t1, t2, r'$T_{\rm CZ}$')):
        axn.annotate('', (a, y), (b, y), arrowprops=dict(arrowstyle='<->', color='k', lw=0.8, shrinkA=0, shrinkB=0,
                                                         mutation_scale=7))
        axn.text(0.5 * (a + b), y + 0.12, lab, ha='center', va='bottom', fontsize=8)


# (b): <n_WG> << 1; couplings legend in the empty part t > T, populations legend above the curves
nmax = max(small[k].max() for k in ('n00', 'n01', 'n11'))
nstep = next(s for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if 1.9 * nmax / s <= 4.5)   # at most 4 ticks
traj_panels(bk, bn, small, 15.0, dict(loc='center right', ncol=1), dict(loc='upper center', ncol=3),
            nlim=(-0.06 * nmax, 1.9 * nmax), nstep=nstep, bottom=False)
bn.yaxis.set_major_locator(FixedLocator(np.arange(0, 0.85 * 1.9 * nmax, nstep)))   # no label in the top 15%
fid_panel(bf, small)
traj_panels(ck, cn, large, KMAX, None, None, nlim=(-0.15, 2.45), nstep=1.0)
mark_times(cn, large)
axs = [ax, bk, bn, bf, ck, cn]
for a in axs:
    a.set_axisbelow(True)
fig.align_ylabels(axs)
for a, lab in ((ax, '(a)'), (bk, '(b)'), (ck, '(c)')):
    pos = a.get_position()
    fig.text(0.005, pos.y1 + (0.004 if a is ax else 0.026) * 4.45 / H, lab, **LETTER)

os.makedirs(FIG, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, f'fig_delay.{ext}'), dpi=600 if ext == 'png' else None)
print('saved', os.path.normpath(os.path.join(FIG, 'fig_delay.pdf')))

"""Convert the paper's SLH-optimised CZ gate (src/results/CZ_Chi3_res.pkl) to physical units for the Julia
delay model, and re-check its SLH infidelity with an independent numpy implementation.

usage: python delay_model/scripts/convert_slh_cz.py      (writes delay_model/results/slh_cz_chi3.json)

Convention of the optimisation code (src/rqpnn/rqpnn_hamiltonian.py): the time grid has unit-length bins and, in bin p,
    H_p = tau_p * (-chi3 * sum n(n-1)) + sum delta_p n + sum C_nm sqrt(kappa_n,p kappa_m,p) a_n^dag a_m
with chi3 = 1. Dividing by tau_p gives the physical Hamiltonian applied for a physical duration tau_p (units 1/chi3):
    delta_phys = delta_p / tau_p,  kappa_phys = kappa_p / tau_p.
"""
import itertools
import json
import os
import pickle

import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(HERE, '..', '..', 'src', 'results', 'CZ_Chi3_res.pkl')
OUT = os.path.join(HERE, '..', 'results', 'slh_cz_chi3.json')


def _reconstruct_array(fun, args, arr_state, aval_state):
    a = fun(*args)
    a.__setstate__(arr_state)
    return a


class _Unpickler(pickle.Unpickler):
    """Reads the result pickles without jax installed (jax arrays become numpy arrays)."""
    def find_class(self, module, name):
        if module == 'jax._src.array' and name == '_reconstruct_array':
            return _reconstruct_array
        return super().find_class(module, name)


def physical(d):
    p, a = d[0], d[1]
    tau = 1 / (1 + np.exp(-p['tau'].astype(float))) * (a['max_tau'] - a['min_tau']) + a['min_tau']
    kap = (np.pi / 2 + np.arctan(p['kappa'].astype(float))) * a['kappa'] / np.pi
    dc = np.arctan(p['delta_cav'].astype(float)) * a['delta_cav'] / np.pi
    M = d[3]['M_cav']
    iu = np.triu_indices(M, k=1)
    A = np.zeros((M, M)); A[iu] = p['coeff_vec'].astype(float)
    C = A + A.T
    return dict(dt=tau.astype(float), kappa=(kap / tau).astype(float), delta=(dc / tau).astype(float), C=C, M=M)


def slh_cz_infidelity(ph):
    """SLH CZ infidelity in the full two-photon Fock space of the four cavities."""
    M = ph['M']
    basis = [s for s in itertools.product(range(3), repeat=M) if sum(s) == 2]
    idx = {s: i for i, s in enumerate(basis)}
    D = len(basis)
    hop = np.zeros((M, M, D, D))                       # a_n^dag a_m
    for i, s in enumerate(basis):
        for n in range(M):
            for m in range(M):
                if s[m] == 0:
                    continue
                t = list(s); amp = np.sqrt(t[m]); t[m] -= 1
                amp *= np.sqrt(t[n] + 1); t[n] += 1
                hop[n, m, idx[tuple(t)], i] += amp
    nn = np.array([[hop[m, m, i, i] for i in range(D)] for m in range(M)])
    kerr = -np.sum(nn * (nn - 1), axis=0)
    U = np.eye(D, dtype=complex)
    for p in range(len(ph['dt'])):
        k = ph['kappa'][:, p]
        H = np.diag(kerr + ph['delta'][:, p] @ nn).astype(complex)
        for n in range(M):
            for m in range(M):
                H += ph['C'][n, m] * np.sqrt(k[n] * k[m]) * hop[n, m]
        U = expm(-1j * H * ph['dt'][p]) @ U
    ins = [(1, 0, 1, 0), (1, 0, 0, 1), (0, 1, 1, 0), (0, 1, 0, 1)]
    amps = [sgn * U[idx[s], idx[s]] for sgn, s in zip([1, 1, 1, -1], ins)]
    return 1 - abs(np.mean(amps)) ** 2


if __name__ == '__main__':
    with open(PKL, 'rb') as f:
        ph = physical(_Unpickler(f).load())
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ph.items()}
    out['source'] = 'src/results/CZ_Chi3_res.pkl'
    out['units'] = 'chi3 = 1; dt = duration of each piecewise-constant bin (1/chi3); kappa, delta: rows = cavities'
    out['T'] = float(np.sum(ph['dt']))
    out['slh_infidelity_numpy'] = slh_cz_infidelity(ph)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {OUT}: T = {out["T"]:.4f}, {len(ph["dt"])} bins, max kappa = {ph["kappa"].max():.2f}, '
          f'max |delta| = {np.abs(ph["delta"]).max():.2f}, SLH infidelity (numpy) = {out["slh_infidelity_numpy"]:.4e}')

"""Reference solution for the 1D Allen-Cahn benchmark.

    u_t - 1e-4 u_xx + 5 u^3 - 5 u = 0,   x in [-1, 1] (periodic),   t in [0, 1]
    u(0, x) = x^2 cos(pi x)

Method: Fourier pseudo-spectral discretization in x, ETDRK4 time stepping
(Kassam & Trefethen, SIAM J. Sci. Comput. 26, 2005), phi-functions evaluated
by contour integrals.

Note on resolution: u0 is periodic but its slope jumps at x = +-1
(u0'(-1) = 2, u0'(1) = -2), so spectral convergence is only algebraic.
We therefore solve on a fine grid (default N = 4096) and subsample to the
512-point output grid. Default settings are self-converged to ~1e-6
relative L2 (N = 2048 vs 4096 differ by 7e-7; dt = 1e-4 vs 1e-5 by <1e-11).

Output (.npz): t (nt,), x (nx,) periodic grid x_j = -1 + 2 j / nx (x = 1 is
the same point as x = -1), usol (nt, nx), plus the solver settings.

Usage:
    python data/generate_allen_cahn.py                  # writes data/allen_cahn_ref.npz
    python data/generate_allen_cahn.py --check          # also runs a convergence check
"""
import argparse
import numpy as np


def solve(n_solve=4096, dt=1e-4, T=1.0, nt_out=201, eps=1e-4, n_contour=32):
    x = -1.0 + 2.0 * np.arange(n_solve) / n_solve
    u = x**2 * np.cos(np.pi * x)
    v = np.fft.fft(u)
    k = np.pi * np.fft.fftfreq(n_solve, d=1.0 / n_solve)   # wavenumbers, period 2
    L = -eps * k**2 + 5.0                                   # linear part: diffusion + 5u
    E, E2 = np.exp(dt * L), np.exp(dt * L / 2)
    r = np.exp(1j * np.pi * (np.arange(1, n_contour + 1) - 0.5) / n_contour)
    LR = dt * L[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))

    def nonlin(v):                                          # -5 u^3 in Fourier space
        return -5.0 * np.fft.fft(np.real(np.fft.ifft(v)) ** 3)

    n_steps = int(round(T / dt))
    if n_steps % (nt_out - 1):
        raise ValueError("T/dt must be a multiple of (nt_out - 1)")
    every = n_steps // (nt_out - 1)
    t_out, U = [0.0], [u.copy()]
    for n in range(1, n_steps + 1):
        Nv = nonlin(v)
        a = E2 * v + Q * Nv
        Na = nonlin(a)
        b = E2 * v + Q * Na
        Nb = nonlin(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = nonlin(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if n % every == 0:
            t_out.append(n * dt)
            U.append(np.real(np.fft.ifft(v)))
    return np.array(t_out), x, np.array(U)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-solve", type=int, default=4096)
    p.add_argument("--n-out", type=int, default=512)
    p.add_argument("--nt-out", type=int, default=201)
    p.add_argument("--dt", type=float, default=1e-4)
    p.add_argument("--out", default="data/allen_cahn_ref.npz")
    p.add_argument("--check", action="store_true", help="compare against a half-resolution solve")
    a = p.parse_args()
    if a.n_solve % a.n_out:
        raise ValueError("--n-solve must be a multiple of --n-out")

    t, x, U = solve(a.n_solve, a.dt, nt_out=a.nt_out)
    s = a.n_solve // a.n_out
    np.savez(a.out, t=t, x=x[::s], usol=U[:, ::s], n_solve=a.n_solve, dt=a.dt,
             equation="u_t - 1e-4 u_xx + 5u^3 - 5u = 0, u0 = x^2 cos(pi x), periodic on [-1,1]")
    print(f"wrote {a.out}: usol {U[:, ::s].shape}, range [{U.min():.4f}, {U.max():.4f}]")

    if a.check:
        _, _, Uc = solve(a.n_solve // 2, a.dt, nt_out=a.nt_out)
        ref, coarse = U[:, ::s], Uc[:, :: s // 2]
        print(f"rel L2 diff, N={a.n_solve // 2} vs N={a.n_solve}: "
              f"{np.linalg.norm(coarse - ref) / np.linalg.norm(ref):.2e}")


if __name__ == "__main__":
    main()

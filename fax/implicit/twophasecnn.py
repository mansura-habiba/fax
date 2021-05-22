import functools
import logging
import operator

import jax
import jax.numpy as np

from faxutil import converge
from faxutil import loop

logger = logging.getLogger(__name__)

_add = functools.partial(jax.tree_multimap, operator.add)


def default_convergence_test(rtol=1e-10, atol=1e-10, dtype=np.float32):

    adjusted_tol = converge.adjust_tol_for_dtype(rtol, atol, dtype)

    def convergence_test(x_new, x_old):
        return converge.max_diff_test(x_new, x_old, *adjusted_tol)

    return convergence_test


def default_solver(convergence_test=None, max_iter=5000, batched_iter_size=1):


    def _default_solve(param_func, init_x, params):

        _convergence_test = convergence_test
        if convergence_test is None:
            _convergence_test = default_convergence_test(
                dtype=converge.tree_smallest_float_dtype(init_x),
            )

        func = param_func(params)
        sol = loop.fixed_point_iteration(
            init_x=init_x,
            func=func,
            convergence_test=_convergence_test,
            max_iter=max_iter,
            batched_iter_size=batched_iter_size,
        )

        return sol.value
    return _default_solve


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,3))
def two_phase_solve(param_func, init_xs, params, solvers=()):


    # If no solver is given or if None is found in its place, use the default
    # fixed-point iteration solver.
    if solvers and solvers[0] is not None:
        fwd_solver = solvers[0]
    else:
        fwd_solver = default_solver()

    return fwd_solver(param_func, init_xs, params)


def two_phase_fwd(param_func, init_xs, params, solvers):
    sol = two_phase_solve(param_func, init_xs, params, solvers)
    return sol, (sol, params)


def two_phase_rev(param_func, solvers, res, sol_bar):

    def param_dfp_fn(packed_params):
        v, p, dvalue = packed_params
        _, fp_vjp_fn = jax.vjp(lambda x: param_func(p)(x), v)

        def dfp_fn(dout):
            dout = _add(fp_vjp_fn(dout)[0], dvalue)
            return dout

        return dfp_fn

    sol, params = res
    dsol = two_phase_solve(param_dfp_fn,
                           sol_bar,
                           (sol, params, sol_bar),
                           solvers[1:])
    _, dparam_vjp = jax.vjp(lambda p: param_func(p)(sol), params)
    return jax.tree_map(np.zeros_like, sol), dparam_vjp(dsol)[0]


two_phase_solve.defvjp(two_phase_fwd, two_phase_rev)
"""Base MPC optimizer functions."""

import functools
import jax
import jax.numpy as jnp
import trajax.optimizers as trajax_opt


def ilqr_solve(
    cost, dynamics, x0, U, params, cost_args, dynamics_args, trajax_ilqr_kwargs
):
    def wrapped_cost(x, u, t):
        return cost(x, u, t, params, *cost_args)

    def wrapped_dynamics(x, u, t):
        return dynamics(x, u, t, params, *dynamics_args)

    return trajax_opt.ilqr(
        wrapped_cost, wrapped_dynamics, x0, U, **trajax_ilqr_kwargs
    )


def objective(cost, dynamics, U, x0):
    return jnp.sum(
        trajax_opt.evaluate(
            cost,
            trajax_opt.rollout(dynamics, U, x0),
            trajax_opt.pad(U),
        )
    )


# @functools.partial(jax.jit, static_argnums=(0, 1, 2))
def bilevel_optimization(
    cost,
    dynamics,
    loss,
    x0,
    init_U,
    params,
    cost_args,
    dynamics_args,
    loss_args,
    trajax_ilqr_kwargs,
):
    def wrapped_cost(x, u, t):
        return cost(x, u, t, params, *cost_args)

    def wrapped_dynamics(x, u, t):
        return dynamics(x, u, t, params, *dynamics_args)

    T, m = init_U.shape

    X, U, _, low_level_grad, _, _, itr = trajax_opt.ilqr(
        wrapped_cost, wrapped_dynamics, x0, init_U, **trajax_ilqr_kwargs
    )

    B = loss_grad_wrt_control(
        loss, wrapped_dynamics, x0, U, loss_args
    ).reshape((T * m,))
    A = cost_hessian_wrt_control(
        wrapped_cost, wrapped_dynamics, x0, U
    ).reshape((T * m, T * m))
    H = jax.scipy.linalg.solve(A, B).reshape((T * m,))

    high_level_grad = cost_vjp(
        cost, wrapped_dynamics, H, x0, U, params, cost_args
    )

    high_level_loss = loss(X, U, *loss_args)

    return (high_level_loss, low_level_grad, high_level_grad, itr)


def loss_grad_wrt_control(loss, dynamics, x0, U, loss_args):
    def func(U):
        X = trajax_opt.rollout(dynamics, U, x0)
        return loss(X, U, *loss_args)

    return jax.grad(func)(U)


def cost_hessian_wrt_control(cost, dynamics, x0, U):
    def func(U):
        return objective(cost, dynamics, U, x0)

    return jax.hessian(func)(U)


def cost_vjp(cost, dynamics, V, x0, U, params, cost_args):
    v_size = V.shape[0]

    def outter(params):
        def inner(U):
            def wrapped_cost(x, u, t):
                return cost(x, u, t, params, *cost_args)

            return objective(wrapped_cost, dynamics, U, x0)

        return V @ jax.grad(inner)(U).reshape((v_size,))

    return jax.grad(outter)(params)

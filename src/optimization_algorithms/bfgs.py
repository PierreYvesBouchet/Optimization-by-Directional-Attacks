#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Libraries import

import torch
import time

from src.optimization_algorithms.tools.fill_history import fill_history



#%% Optimization based on L-BFGS (unconstrained)

def optim_bfgs(f, df, Phi, x_0, r_0,
                r_min              = 0,
                nb_points_max      = float("inf"),
                runtime_max        = float("inf"),
                k_max              = float("inf"),
                history_size       = 10,
                line_search_fn     = 'strong_wolfe',
                t_stall            = 0,
                verbose_iterations = 0,
                seed               = 0
                ):

    rng = torch.Generator()
    rng.manual_seed(seed)
    Phi.nb_forward_calls = 0


    history = fill_history([], "x", "f(Phi(x))", "k", "runtime", "cache size", "iteration status",
                           additional=["step norm"], is_header=True)
    t_sum = 0
    v_sum = 0
    converged = False

    # x must be a leaf tensor with gradient tracking so that LBFGS can update
    # it in-place and autograd can accumulate gradients through Phi.
    x = x_0.clone().detach().requires_grad_(True)

    # LBFGS minimizes; we maximize f(Phi(x)) by minimizing -f(Phi(x)).
    # lr=r_0 is the initial step size fed to the line search.
    # max_iter=1 gives one LBFGS update per step() call, matching the
    # one-move-per-iteration spirit of the other methods here.
    optimizer = torch.optim.LBFGS(
        [x],
        lr               = r_0,
        max_iter         = 1,
        history_size     = history_size,
        line_search_fn   = line_search_fn,
        tolerance_grad   = 0.0,   # disable internal stopping; we control it
        tolerance_change = 0.0,
    )

    with torch.no_grad(): o = f(Phi(x))
    k = 0; t = 0; v = 1; step_norm = float("inf"); s = "init"
    history = fill_history(history, x.detach().clone(), o, k, t, v, s, additional=[step_norm])
    v_sum += v

    if verbose_iterations > 0:
        with torch.no_grad():
            print("optim_bfgs from obj value = {:>+9.3E} with seed {}".format(o, seed))

    while not converged:

        k   += 1
        x_prev = x.detach().clone()
        o_prev = o
        t_in = time.perf_counter()

        def closure():
            optimizer.zero_grad()

            phi_x = Phi(x)                           # keep graph through Phi

            with torch.no_grad():
                grad_f = df(phi_x.detach())          # ∇f(Phi(x)), no graph needed

            # d/dx [-(phi_x · grad_f).sum()] = -J_Phi(x)^T grad_f = ∇_x(-f(Phi(x)))
            dummy_loss = -(phi_x * grad_f).sum()
            dummy_loss.backward()                    # sets x.grad correctly

            # The scalar returned is only used by the line search for comparisons;
            # it does not need to carry a computation graph.
            with torch.no_grad(): loss_val = f(phi_x.detach())
            time.sleep(t_stall)
            return torch.tensor(-float(loss_val), dtype=x.dtype)

        optimizer.step(closure)

        with torch.no_grad(): o_new = f(Phi(x))
        step_norm = torch.linalg.norm(x.detach() - x_prev, ord=float("inf")).item()

        if o_new > o_prev:
            o = o_new
            s = "linesearch"
        else:
            s = "failure"

        t_out = time.perf_counter(); t = t_out-t_in; t_sum += t
        v = Phi.nb_forward_calls-v_sum;     v_sum += v
        history = fill_history(history, x.detach().clone(), o, k, t, v, s, additional=[step_norm])

        if verbose_iterations > 0 and k % verbose_iterations == 0:
            print("k = {:>4d}, obj = {:>+9.3E}, step = {:>7.1E}, v = {:>8d}, t = {:>6.2f}, s = {:s}".format(k, o, step_norm, v_sum, t_sum, s))

        if k >= k_max:
            converged = True
            if verbose_iterations > 0: print("stopping criterion \"number of iterations\" triggered")

        if v_sum >= nb_points_max:
            converged = True
            if verbose_iterations > 0: print("stopping criterion \"number of evaluated points\" triggered")

        if step_norm <= r_min:
            converged = True
            if verbose_iterations > 0: print("stopping criterion \"step norm < r_min\" triggered")

        if t_sum >= runtime_max:
            converged = True
            if verbose_iterations > 0: print("stopping criterion \"excessive runtime\" triggered")

    if verbose_iterations > 0:
        print("k = {:>4d}, obj = {:>+9.3E}, step = {:>7.1E}, v = {:>8d}, t = {:>6.2f}".format(k, o, step_norm, v_sum, t_sum))
        print()

    return history
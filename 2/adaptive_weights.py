"""
adaptive_weights.py -- optional adaptive loss-term balancing via gradient
statistics, per Wang, S., Teng, Y. & Perdikaris, P. (2021), "Understanding
and mitigating gradient flow pathologies in physics-informed neural
networks," SIAM Journal on Scientific Computing, 43(5), A3055-A3081.

Carried over UNCHANGED from the 2-D plane-channel project -- this module
is purely a generic gradient-norm balancing utility over a dict of scalar
loss terms and a parameter list; it has no dependence on channel vs. pipe
geometry, so no adaptation was needed for the axisymmetric pipe project.
It is included here, as there, as an OPTIONAL alternative to
train_hybrid.py's default fixed weights (10 data / 1 PDE / 5 inlet+wall+
axis / 1 outlet) -- pass --adaptive on the command line to enable it.

SIMPLIFICATION, stated explicitly rather than presented as the paper's
exact method: Wang et al. (2021) maintain a running (exponential moving
average) estimate of each loss term's gradient statistics over all of
training. This implementation instead recomputes each term's weight from
a single fresh gradient-norm measurement every N iterations (an
engineering simplification for a small student project, not a claim of
reproducing the paper's full scheme). The core idea it does keep: rescale
each loss term so it produces a gradient magnitude comparable to the
PDE-residual term (held as the fixed reference, weight 1.0, as in the
paper's own convention), instead of relying only on hand-picked constants.
"""
import torch


def _max_grad_norm(loss, params):
    """Max abs. gradient this loss term induces across `params` -- the
    quantity Wang et al. (2021) balance terms against (their Eq. for
    lambda_i uses the max, not the mean, of the gradient magnitude)."""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    norms = [g.abs().max().item() for g in grads if g is not None]
    return max(norms) if norms else 0.0


def compute_adaptive_weights(loss_terms, reference_key, params, eps=1e-8):
    """
    loss_terms   : dict[str, torch.Tensor] of scalar loss terms, each still
                   attached to the autograd graph built from `params`.
    reference_key: key whose weight is held fixed at 1.0 (the PDE-residual
                   term, matching the paper's own convention of anchoring
                   on the PDE loss).
    params       : iterable of model parameters to measure gradients
                   against -- the paper measures against a single
                   representative (e.g. last hidden) layer rather than the
                   whole network, which is what this project also does
                   (see train_hybrid.py, `adaptive_ref_params`).

    Returns dict[str, float] of weights (reference term always 1.0).
    NOTE: this calls autograd.grad once per loss term with
    retain_graph=True, so it is noticeably more expensive per call than a
    normal training step -- call it only every N iterations, not every
    step (see train_hybrid.py's ADAPTIVE_EVERY).
    """
    ref_norm = _max_grad_norm(loss_terms[reference_key], params)
    weights = {reference_key: 1.0}
    for k, loss in loss_terms.items():
        if k == reference_key:
            continue
        term_norm = _max_grad_norm(loss, params)
        weights[k] = ref_norm / (term_norm + eps)
    return weights
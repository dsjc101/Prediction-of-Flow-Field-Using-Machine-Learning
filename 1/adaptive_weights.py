"""
adaptive_weights.py -- self-adaptive loss-term weighting for the hybrid
PINN, following the gradient-statistics scheme of:

    Wang, S., Teng, Y., & Perdikaris, P. (2021), "Understanding and
    Mitigating Gradient Flow Pathologies in Physics-Informed Neural
    Networks," SIAM Journal on Scientific Computing, 43(5), A3055-A3081.
    DOI: 10.1137/20M1318043

NOTE ON PROVENANCE: train_hybrid.py imports compute_adaptive_weights from
this module and README.md describes it as "New. Optional Wang-et-al.-
style adaptive loss balancing," but the file itself was never actually
included in any upload -- like model.py/physics.py, it has to be written
from the paper it's supposed to implement, not assumed.

Core idea (Wang et al. 2021, Sec. 3): when several loss terms are simply
summed with fixed scalar weights, the term whose gradient happens to be
smallest at the shared network parameters is effectively starved during
back-propagation, even if its scalar weight is not literally zero. This
is exactly the failure mode fixed weights (DEFAULT_WEIGHTS in
train_hybrid.py) cannot detect or correct, because they never look at
what actually reaches the weights during training.

The fix: periodically measure, at a FIXED reference set of parameters
(the paper diagnoses this at the last hidden layer, Sec. 3.2 -- reused
here for the same reason: it is close enough to the output to reflect
each loss term's real influence, without the extra noise of the full
parameter set), the gradient each loss term produces there, and rescale
every term so all of them push comparably hard on those parameters:

    lambda_i = max_k |dL_ref/dtheta_k|  /  mean_k |dL_i/dtheta_k|

with the reference term (here the PDE residual -- the term a hybrid
PINN most needs to protect from being drowned out by a strong data-loss
gradient) fixed at weight 1. To keep the weights from oscillating between
the periodic recomputations, an exponential moving average is applied
(Wang et al. Eq. 3.4; alpha=0.9 is the value used in the paper).
"""
import torch

ALPHA = 0.9  # EMA smoothing factor, Wang et al. 2021 Eq. 3.4

# Module-level EMA state. Persists across calls within one training run
# (train_hybrid.py calls this every ADAPTIVE_EVERY iterations); reset by
# reloading the module (i.e. a fresh `python train_hybrid.py ... --fresh`
# process) rather than by any explicit API, since there is only ever one
# training run alive in this project's process at a time.
_ema_weights = {}


def _grad_stats(loss, params):
    """Flatten d(loss)/d(params) into one vector and return (max |.|,
    mean |.|). retain_graph=True: the caller still needs each term's
    graph intact both for the other terms' grad_stats calls and for the
    real backward() pass that follows."""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
    return flat.abs().max().item(), flat.abs().mean().item()


def compute_adaptive_weights(terms, reference_key, params, alpha=ALPHA,
                              clip_range=(0.2, 20.0)):
    """
    terms         : dict[str, Tensor] of scalar loss terms, e.g. the
                     output of train_hybrid.compute_loss_terms().
    reference_key  : which term's gradient sets the scale; that term's
                     weight is fixed at 1.0 (use "pde" for this project,
                     per the hybrid-PINN rationale above).
    params         : list of Tensors to differentiate against -- pass a
                     single representative layer's parameters (e.g. the
                     last hidden Linear layer's weight+bias), not the
                     whole network, matching Wang et al.'s own choice.
    alpha          : EMA smoothing factor (default: paper's alpha=0.9).
    clip_range     : (min, max) bound applied to every non-reference
                      weight after the EMA update. SAFEGUARD, not in the
                      original 2021 formula: a term whose own gradient
                      shrinks fastest (e.g. this project's "outlet" term,
                      a single scalar p=0 condition that gets satisfied
                      early) has lambda_i = ref_max/mean_i grow WITHOUT
                      BOUND as mean_i -> 0, even though its absolute loss
                      is already negligible -- observed directly in this
                      project's own training log (outlet's weight climbed
                      to 13-31x pde's fixed weight of 1, visibly crowding
                      out the wall/data terms' share of the gradient).
                      Bounding runaway weights once they're observed is
                      standard stabilization practice for this class of
                      scheme -- see e.g. the follow-up practical guidance
                      in Wang, Sankaran, Wang & Perdikaris (2023), "An
                      Expert's Guide to Training Physics-Informed Neural
                      Networks," arXiv:2308.08468. The exact bounds here
                      are a hyperparameter, not a value taken from that
                      paper -- tune clip_range if your own loss terms sit
                      at very different natural scales.

    Returns a plain dict[str, float] of weights, EMA-smoothed against
    the previous call's weights within this process, then clipped.
    """
    ref_max, _ = _grad_stats(terms[reference_key], params)

    raw = {}
    for k, L in terms.items():
        if k == reference_key:
            raw[k] = 1.0
            continue
        _, mean_k = _grad_stats(L, params)
        raw[k] = ref_max / (mean_k + 1e-12)

    global _ema_weights
    if not _ema_weights:
        _ema_weights = dict(raw)
    else:
        for k in raw:
            prev = _ema_weights.get(k, raw[k])
            _ema_weights[k] = alpha * prev + (1.0 - alpha) * raw[k]

    lo, hi = clip_range
    for k in _ema_weights:
        if k != reference_key:
            _ema_weights[k] = min(max(_ema_weights[k], lo), hi)

    return dict(_ema_weights)
"""
adaptive_weights.py
=====================
Carried over UNCHANGED from the channel and pipe projects (it is a
generic utility over loss-term/parameter gradient-norm dictionaries with
no geometry dependence, exactly as noted in the pipe report Sec 12).

Wang, S., Teng, Y. & Perdikaris, P. (2021), "Understanding and mitigating
gradient flow pathologies in physics-informed neural networks," SIAM J.
Sci. Comput., 43(5), A3055-A3081, https://doi.org/10.1137/20M1318043

Enabled with --adaptive; fixed weights (config.LOSS_WEIGHTS) are used by
default, matching both earlier reports' reported results.
"""

import torch


def gradient_norm_weights(loss_terms: dict, model: torch.nn.Module, alpha=0.9,
                            prev_weights: dict = None):
    """
    loss_terms: {name: scalar tensor loss}. Returns updated weight dict
    using the Wang/Teng/Perdikaris (2021) gradient-norm balancing rule:
    weight_i propto (sum_j ||grad L_j||) / ||grad L_i||, smoothed with an
    exponential moving average (alpha) against prev_weights to avoid
    oscillation, exactly the same smoothing device used in the pipe
    project's carried-over implementation.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    norms = {}
    for name, loss in loss_terms.items():
        grads = torch.autograd.grad(loss, params, retain_graph=True,
                                     create_graph=False, allow_unused=True)
        gnorm = torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None) + 1e-12)
        norms[name] = float(gnorm.item())

    total = sum(norms.values()) + 1e-12
    raw_weights = {k: total / (n + 1e-12) for k, n in norms.items()}
    # normalize so weights average to 1 (keeps overall loss scale stable)
    mean_w = sum(raw_weights.values()) / len(raw_weights)
    raw_weights = {k: w / mean_w for k, w in raw_weights.items()}

    if prev_weights is None:
        return raw_weights
    return {k: alpha * prev_weights.get(k, raw_weights[k]) + (1 - alpha) * raw_weights[k]
            for k in raw_weights}
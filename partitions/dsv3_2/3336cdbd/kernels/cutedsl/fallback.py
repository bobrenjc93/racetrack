from __future__ import annotations

try:
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False


def fused_norm_rope(
    q_c,
    q_weight,
    kv_c,
    kv_weight,
    k_pe,
    positions,
    *,
    eps,
    rope_base,
    fallback,
):
    return fallback(
        q_c, q_weight, kv_c, kv_weight, k_pe, positions,
        eps=eps, rope_base=rope_base,
    )


def fused_residual_norm(
    residual,
    update,
    norm_weight,
    *,
    eps,
    fallback,
):
    return fallback(residual, update, norm_weight, eps=eps)


def fused_swiglu(gate, up, *, fallback):
    return fallback(gate, up)


def hc_head(
    hidden_states,
    hc_fn,
    hc_scale,
    hc_base,
    *,
    rms_norm_eps,
    hc_eps,
    fallback,
):
    return fallback(
        hidden_states, hc_fn, hc_scale, hc_base,
        rms_norm_eps=rms_norm_eps, hc_eps=hc_eps,
    )

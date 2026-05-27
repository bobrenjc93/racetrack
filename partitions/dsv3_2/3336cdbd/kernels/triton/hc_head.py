from __future__ import annotations

try:
    import triton  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False


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

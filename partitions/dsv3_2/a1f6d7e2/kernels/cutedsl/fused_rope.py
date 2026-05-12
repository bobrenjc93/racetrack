BACKEND_AVAILABLE = False


def fused_norm_rope(*args, **kwargs):
    raise RuntimeError("CUTEDSL fused_norm_rope is not implemented")


def hc_head(*args, **kwargs):
    raise RuntimeError("CUTEDSL hc_head is not implemented")

__all__ = ["BACKEND_AVAILABLE", "fused_norm_rope", "hc_head"]

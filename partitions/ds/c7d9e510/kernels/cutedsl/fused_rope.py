from racetrack.runtime.package_kernels import fused_norm_rope, hc_head, package_available

BACKEND_AVAILABLE = package_available("cutlass") or package_available("cutedsl")

__all__ = ["BACKEND_AVAILABLE", "fused_norm_rope", "hc_head"]

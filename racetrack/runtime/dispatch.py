from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch


Fallback = Callable[..., Any]


class KernelDispatcher:
    """Loads partition-local kernel modules and routes calls by env/backend.

    The public switch is ``RACETRACK_KERNEL_BACKEND``. Supported values are
    ``torch``, ``triton``, ``cutedsl``, ``cutedl`` (alias), ``helion``, and
    ``best``. ``best`` times all callable candidates on first use and caches
    the fastest backend for the operation signature.
    """

    BACKENDS = ("triton", "cutedsl", "helion")

    def __init__(self, kernel_root: str | Path | None = None):
        self.kernel_root = Path(kernel_root) if kernel_root is not None else None
        self._modules: dict[tuple[str, str], ModuleType | None] = {}
        self._best: dict[tuple[Any, ...], str] = {}
        self._best_ops: dict[str, set[str]] = {}
        self._best_fast_path: dict[str, str] = {}

    @staticmethod
    def selected_backend(default: str = "torch") -> str:
        raw = os.getenv("RACETRACK_KERNEL_BACKEND", default).strip().lower()
        if raw == "cutedl":
            return "cutedsl"
        return raw

    def backend_status(self, backend: str) -> str:
        if backend == "torch":
            return "native"
        module = self._load_module(backend, "fused_rope")
        if module is None:
            return "missing"
        return "native" if bool(getattr(module, "BACKEND_AVAILABLE", False)) else "missing"

    def call(
        self,
        op_name: str,
        fallback: Fallback,
        *args: Any,
        backend: str | None = None,
        **kwargs: Any,
    ) -> Any:
        selected = backend or self.selected_backend(default="torch")
        if selected == "all":
            selected = "torch"
        if selected == "torch":
            return fallback(*args, **kwargs)
        if selected == "best":
            selected = self._best_fast_path.get(op_name)
            if selected is None:
                selected = self._select_best(op_name, fallback, *args, **kwargs)
                self._best_fast_path[op_name] = selected
        fn = self._resolve(selected, op_name)
        if fn is None:
            self._handle_missing(selected, op_name)
        return fn(*args, fallback=fallback, **kwargs)

    def best_summary(self) -> str:
        if not self._best_ops:
            return "best"
        unique_backends = sorted(
            {backend for backends in self._best_ops.values() for backend in backends}
        )
        if len(unique_backends) == 1:
            return f"mixed={unique_backends[0]}"
        parts = [
            f"{op_name}={'+'.join(sorted(backends))}"
            for op_name, backends in sorted(self._best_ops.items())
        ]
        return "mixed=" + ";".join(parts)

    def _handle_missing(self, backend: str, op_name: str) -> None:
        raise RuntimeError(f"No available {backend} kernel found for {op_name}")

    def _resolve(self, backend: str, op_name: str) -> Callable[..., Any] | None:
        module = self._load_module(backend, "fused_rope")
        if module is None:
            return None
        if not bool(getattr(module, "BACKEND_AVAILABLE", False)):
            return None
        fn = getattr(module, op_name, None)
        return fn if callable(fn) else None

    def _load_module(self, backend: str, module_name: str) -> ModuleType | None:
        if self.kernel_root is None:
            return None
        key = (backend, module_name)
        if key in self._modules:
            return self._modules[key]
        path = self.kernel_root / backend / f"{module_name}.py"
        if not path.exists():
            self._modules[key] = None
            return None
        spec_name = f"racetrack_partition_kernel_{abs(hash(path))}_{backend}_{module_name}"
        spec = importlib.util.spec_from_file_location(spec_name, path)
        if spec is None or spec.loader is None:
            self._modules[key] = None
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._modules[key] = module
        return module

    def _select_best(
        self,
        op_name: str,
        fallback: Fallback,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        key = self._best_key(op_name, args, kwargs)
        if key in self._best:
            selected = self._best[key]
            self._best_ops.setdefault(op_name, set()).add(selected)
            return selected

        candidates = [
            backend for backend in self.BACKENDS if self._resolve(backend, op_name)
        ]
        timings: list[tuple[float, str]] = []
        for candidate in candidates:
            fn = self._resolve(candidate, op_name)
            if fn is None:
                continue
            try:
                elapsed = self._time_candidate(candidate, fn, fallback, *args, **kwargs)
            except Exception:
                if os.getenv("RACETRACK_KERNEL_STRICT", "0") == "1":
                    raise
                continue
            timings.append((elapsed, candidate))
        if not timings:
            raise RuntimeError(f"No available kernels found for best {op_name}")
        selected = min(timings)[1]
        self._best[key] = selected
        self._best_ops.setdefault(op_name, set()).add(selected)
        return selected

    @staticmethod
    def _best_key(
        op_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, ...]:
        tensor_parts = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor_parts.append(
                    (
                        tuple(arg.shape),
                        str(arg.dtype),
                        str(arg.device),
                        tuple(arg.stride()),
                    )
                )
        scalar_parts = tuple(
            sorted(
                (key, value)
                for key, value in kwargs.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            )
        )
        return (op_name, tuple(tensor_parts), scalar_parts)

    @staticmethod
    def _time_candidate(
        backend: str,
        fn: Callable[..., Any],
        fallback: Fallback,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        def run_once() -> Any:
            return fn(*args, fallback=fallback, **kwargs)

        run_once()
        iterations = int(os.getenv("RACETRACK_BEST_ITERS", "10"))
        tensor_arg = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
        if tensor_arg is not None and tensor_arg.device.type == "cuda":
            torch.cuda.synchronize(tensor_arg.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                run_once()
            end.record()
            torch.cuda.synchronize(tensor_arg.device)
            return float(start.elapsed_time(end)) / iterations

        start_time = time.perf_counter()
        for _ in range(iterations):
            run_once()
        return (time.perf_counter() - start_time) * 1000.0 / iterations

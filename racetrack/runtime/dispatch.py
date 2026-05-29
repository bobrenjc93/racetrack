from __future__ import annotations

import ast
import importlib.util
import json
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
    ``best``. ``best`` times all implemented callable candidates on first use
    and caches the fastest backend for the operation signature.
    """

    BACKENDS = ("triton", "cutedsl", "helion")

    def __init__(self, kernel_root: str | Path | None = None):
        self.kernel_root = Path(kernel_root) if kernel_root is not None else None
        self._modules: dict[tuple[str, str], ModuleType | None] = {}
        self._backend_modules_cache: dict[str, list[ModuleType]] = {}
        self._best: dict[str, str] = {}
        self._best_ops: dict[str, set[str]] = {}
        self._best_fast_path: dict[str, str] = {}
        self._load_best_config()

    @staticmethod
    def selected_backend(default: str = "torch") -> str:
        raw = os.getenv("RACETRACK_KERNEL_BACKEND", default).strip().lower()
        if raw == "cutedl":
            return "cutedsl"
        return raw

    def backend_status(self, backend: str) -> str:
        if backend == "torch":
            return "native"
        modules = self._load_backend_modules(backend)
        if not modules:
            return "missing"
        if any(bool(getattr(m, "BACKEND_AVAILABLE", False)) for m in modules):
            return "native"
        return "missing"

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
            selected = self._select_best(op_name, fallback, *args, **kwargs)
        if selected == "torch":
            return fallback(*args, **kwargs)
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
        for module in self._load_backend_modules(backend):
            if not bool(getattr(module, "BACKEND_AVAILABLE", False)):
                continue
            fn = getattr(module, op_name, None)
            if callable(fn):
                return fn
        return None

    def _load_backend_modules(self, backend: str) -> list[ModuleType]:
        if backend in self._backend_modules_cache:
            return self._backend_modules_cache[backend]
        if self.kernel_root is None:
            return []
        backend_dir = self.kernel_root / backend
        if not backend_dir.is_dir():
            self._backend_modules_cache[backend] = []
            return []
        modules: list[ModuleType] = []
        for path in sorted(backend_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            module = self._load_module(backend, path.stem)
            if module is not None:
                modules.append(module)
        self._backend_modules_cache[backend] = modules
        return modules

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
        import sys
        sys.modules[spec_name] = module
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
        cached = self._best.get(key)
        if cached is not None and (cached == "torch" or self._resolve(cached, op_name) is not None):
            if cached != "torch":
                self._best_ops.setdefault(op_name, set()).add(cached)
            return cached

        timings: list[tuple[float, str]] = []
        for candidate in self.BACKENDS:
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
            selected = "torch"
        else:
            selected = min(timings)[1]
        self._best[key] = selected
        if selected != "torch":
            self._best_ops.setdefault(op_name, set()).add(selected)
            self._best_fast_path[op_name] = selected
        self._save_best_config()
        return selected

    @staticmethod
    def _best_key(
        op_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
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
        return repr((op_name, tuple(tensor_parts), scalar_parts))

    @staticmethod
    def _op_name_from_signature(signature: str) -> str:
        try:
            parsed = ast.literal_eval(signature)
        except (ValueError, SyntaxError):
            return signature
        if isinstance(parsed, tuple) and parsed and isinstance(parsed[0], str):
            return parsed[0]
        return signature

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

    def _load_best_config(self) -> None:
        if self.kernel_root is None:
            return
        path = self.kernel_root / "best.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            return
        for op_name, per_shape in data.items():
            # Backward-compat: older best.json caches used a flat
            # {op_name: backend} layout before selection became shape-keyed.
            # Treat a bare string as an op-level fast-path default so legacy
            # caches (and any non-shape-keyed report) keep working until a fresh
            # 'best' run rewrites the file in the nested {op: {sig: backend}} form.
            if isinstance(per_shape, str):
                if per_shape != "torch":
                    self._best_fast_path[op_name] = per_shape
                continue
            if not isinstance(per_shape, dict):
                continue
            for signature, backend in per_shape.items():
                if not isinstance(signature, str) or not isinstance(backend, str):
                    continue
                self._best[signature] = backend
                if backend != "torch":
                    self._best_fast_path[op_name] = backend

    def _save_best_config(self) -> None:
        if self.kernel_root is None:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
        nested: dict[str, dict[str, str]] = {}
        for signature, backend in self._best.items():
            op_name = self._op_name_from_signature(signature)
            nested.setdefault(op_name, {})[signature] = backend
        path = self.kernel_root / "best.json"
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with open(tmp_path, "w") as f:
            json.dump(nested, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)

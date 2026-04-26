"""Small adapter for using AndroidWorld as an external library."""

from __future__ import annotations

import os
import random
import sys
import time
import urllib.request
import importlib
import importlib.util
from pathlib import Path
from typing import Any


NEW_ROOT = Path(__file__).resolve().parents[1]
ANDROID_WORLD_ROOT = NEW_ROOT / "android_world"
GENERATED_PROTO_ROOT = Path(__file__).resolve().parent / "generated_proto"
A11Y_FORWARDER_URL = (
    "https://storage.googleapis.com/android_env-tasks/"
    "2024.05.13-accessibility_forwarder.apk"
)
MIN_A11Y_FORWARDER_BYTES = 1_000_000


def prepare_android_world_imports() -> None:
    """Ensure imports resolve to `new/android_world` and Python 3.10 has Self."""
    if str(ANDROID_WORLD_ROOT) not in sys.path:
        sys.path.insert(0, str(ANDROID_WORLD_ROOT))
    try:
        import typing
        from typing_extensions import Self

        if not hasattr(typing, "Self"):
            typing.Self = Self  # type: ignore[attr-defined]
    except Exception:
        pass
    ensure_information_retrieval_protos()


def ensure_information_retrieval_protos() -> None:
    """Make missing AndroidWorld generated proto modules importable."""
    module_names = (
        "android_world.task_evals.information_retrieval.proto.state_pb2",
        "android_world.task_evals.information_retrieval.proto.task_pb2",
    )
    if all(importlib.util.find_spec(name) is not None for name in module_names):
        return

    proto_dir = ANDROID_WORLD_ROOT / "android_world" / "task_evals" / "information_retrieval" / "proto"
    output_dir = GENERATED_PROTO_ROOT
    state_out = output_dir / "android_world" / "task_evals" / "information_retrieval" / "proto" / "state_pb2.py"
    task_out = output_dir / "android_world" / "task_evals" / "information_retrieval" / "proto" / "task_pb2.py"
    if not state_out.exists() or not task_out.exists():
        _generate_proto_modules(proto_dir=proto_dir, output_dir=output_dir)

    parent = importlib.import_module("android_world.task_evals.information_retrieval.proto")
    state_module = _load_module(module_names[0], state_out)
    setattr(parent, "state_pb2", state_module)
    task_module = _load_module(module_names[1], task_out)
    setattr(parent, "task_pb2", task_module)


def find_adb() -> str:
    for path in (
        os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
        os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
        "adb",
    ):
        if path == "adb" or os.path.isfile(path):
            return path
    raise EnvironmentError("adb not found. Set --adb_path.")


def _generate_proto_modules(*, proto_dir: Path, output_dir: Path) -> None:
    try:
        from grpc_tools import protoc
    except ImportError as exc:
        raise RuntimeError("grpc_tools is required to generate AndroidWorld proto modules.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "grpc_tools.protoc",
        f"-I{ANDROID_WORLD_ROOT}",
        f"--python_out={output_dir}",
        str(proto_dir / "state.proto"),
        str(proto_dir / "task.proto"),
    ]
    status = protoc.main(args)
    if status != 0:
        raise RuntimeError(f"protoc failed with exit status {status}.")


def _load_module(module_name: str, path: Path) -> Any:
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generated module {module_name} from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def patch_a11y_forwarder_download() -> None:
    """Patch android_env's one-shot APK download with a checked local cache."""
    prepare_android_world_imports()
    try:
        from android_env.wrappers import a11y_grpc_wrapper
    except Exception:
        return

    def _get_cached_accessibility_forwarder_apk() -> bytes:
        cache_path = Path(
            os.environ.get(
                "ANDROID_WORLD_A11Y_FORWARDER_APK",
                Path.home() / ".cache" / "android_world" / "accessibility_forwarder.apk",
            )
        )
        if cache_path.is_file() and cache_path.stat().st_size >= MIN_A11Y_FORWARDER_BYTES:
            return cache_path.read_bytes()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                data = _download_checked_bytes(A11Y_FORWARDER_URL)
                tmp_path = cache_path.with_suffix(".apk.tmp")
                tmp_path.write_bytes(data)
                tmp_path.replace(cache_path)
                return data
            except Exception as exc:  # pylint: disable=broad-exception-caught
                last_error = exc
                if attempt < 3:
                    time.sleep(attempt)
        raise RuntimeError(f"Failed to download accessibility forwarder APK: {last_error}") from last_error

    a11y_grpc_wrapper._get_accessibility_forwarder_apk = _get_cached_accessibility_forwarder_apk  # pylint: disable=protected-access


def load_env(*, console_port: int, adb_path: str, perform_emulator_setup: bool = False) -> Any:
    prepare_android_world_imports()
    patch_a11y_forwarder_download()
    from android_world.env import env_launcher

    return env_launcher.load_and_setup_env(
        console_port=console_port,
        emulator_setup=perform_emulator_setup,
        adb_path=adb_path,
    )


def create_task(task_name: str, *, seed: int) -> Any:
    prepare_android_world_imports()
    from android_world import registry

    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
    if task_name not in aw_registry:
        raise ValueError(f"Task {task_name} not found in AndroidWorld registry.")
    random.seed(seed)
    task_type = aw_registry[task_name]
    params = task_type.generate_random_params()
    return task_type(params)


def _download_checked_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "gui-demo-android-world/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        expected = int(response.headers.get("Content-Length") or 0)
        chunks = []
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    data = b"".join(chunks)
    if expected and len(data) != expected:
        raise IOError(f"incomplete APK download: got {len(data)} bytes, expected {expected}")
    if len(data) < MIN_A11Y_FORWARDER_BYTES:
        raise IOError(f"APK download is unexpectedly small: {len(data)} bytes")
    return data

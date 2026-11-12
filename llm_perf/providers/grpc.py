"""gRPC provider — dynamic stubs compiled from .proto files.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path

from .base import NormalizedResponse, RequestSpec

_generated_cache: dict = {}


def compile_proto(
    proto_path: str | Path, out_dir: str | Path | None = None, triton_fallback: bool = True
) -> str:
    """Compile a .proto file with grpc_tools; returns the proto's python package path."""
    proto_path = Path(proto_path)
    out_dir = Path(out_dir or proto_path.parent / "_generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    from grpc_tools import protoc

    args = [
        "grpc_tools.protoc",
        f"-I{proto_path.parent}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        str(proto_path),
    ]
    rc = protoc.main(args)
    if rc != 0:
        raise RuntimeError(f"protoc failed for {proto_path} (rc={rc})")
    # add out_dir to sys.path for generated imports
    _ensure_on_syspath(out_dir)
    return str(out_dir.resolve())


def get_triton_proto_path(workdir: Path | None = None) -> Path:
    """Ensure the bundled/downloaded Triton inference proto exists locally."""
    workdir = workdir or Path.cwd() / "protos"
    workdir.mkdir(parents=True, exist_ok=True)
    dest = workdir / "inference.proto"
    if not dest.exists():
        bundled = Path(__file__).resolve().parents[2] / "protos" / "inference.proto"
        if bundled.exists():
            dest.write_bytes(bundled.read_bytes())
        else:
            raise FileNotFoundError("no inference.proto found; place one in protos/ or pass grpc.proto_path")
    return dest


def _ensure_on_syspath(path: Path) -> None:
    """Add a directory to sys.path once (for generated stubs)."""
    s = str(path.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


class GrpcAdapter:
    """gRPC provider: compiles .proto stubs and invokes RPCs dynamically."""

    name = "grpc"

    def __init__(self):
        """GrpcAdapter(): per-instance stub cache."""
        self._channels = {}
        self._stubs = {}

    def _get_stub(self, spec: RequestSpec):
        """Compile proto, open a channel, and return (rpc, pb2, metadata)."""
        gcfg = spec.extra.get("grpc", {})
        # method must be in the key — two targets sharing proto/service/host
        # but calling different RPCs need different cached stubs
        key = (
            gcfg.get("proto_path"),
            gcfg.get("service"),
            gcfg.get("method"),
            gcfg.get("tls"),
            spec.base_url,
        )
        if key in self._stubs:
            return self._stubs[key]
        import grpc

        proto_path = gcfg.get("proto_path")
        if not proto_path:
            proto_path = get_triton_proto_path()
        compile_proto(proto_path)
        stem = Path(proto_path).stem
        _ensure_on_syspath(Path(proto_path).parent / "_generated")
        pb2_mod = importlib.import_module(f"{stem}_pb2")
        grpc_mod = importlib.import_module(f"{stem}_pb2_grpc")

        target = spec.base_url or "localhost:8001"
        if gcfg.get("tls", True):
            channel = grpc.secure_channel(target, grpc.ssl_channel_credentials())
        else:
            channel = grpc.insecure_channel(target)
        # call credentials from metadata
        md = dict(gcfg.get("metadata", {}))
        stub_cls = None
        service = gcfg.get("service", "inference.GRPCInferenceService")
        for attr in dir(grpc_mod):
            if (
                attr.endswith("Stub") and service.split(".")[-1].replace("Service", "") in attr
            ) or attr == f"{service.split('.')[-1]}Stub":
                stub_cls = getattr(grpc_mod, attr)
                break
        if stub_cls is None:
            for attr in dir(grpc_mod):
                if attr.endswith("Stub"):
                    stub_cls = getattr(grpc_mod, attr)
                    break
        stub = stub_cls(channel)
        method_name = gcfg.get("method", "ModelInfer")
        rpc = getattr(stub, method_name, None)
        if rpc is None:
            # snake_case fallback
            snake = "".join("_" + c.lower() if c.isupper() else c for c in method_name).lstrip("_")
            rpc = getattr(stub, snake)
        self._stubs[key] = (rpc, pb2_mod, md)
        return self._stubs[key]

    def call(self, spec: RequestSpec) -> NormalizedResponse:
        """Build the request message, invoke the RPC, extract text/usage."""
        rpc, pb2_mod, md = self._get_stub(spec)
        # build request message: map prompt into configured fields
        fields = spec.extra.get("grpc", {}).get("request_fields", {}) or {}
        text_field = fields.get("text", fields.get("prompt", "text"))
        model_field = fields.get("model", "model_name")

        # find the request message type used by the RPC
        req_cls = None
        for attr in dir(pb2_mod):
            if attr.endswith("Request"):
                req_cls = getattr(pb2_mod, attr)
                break
        if req_cls is None:
            req_cls = pb2_mod.DESCRIPTOR.message_types_by_name.get(
                next(iter(pb2_mod.DESCRIPTOR.message_types_by_name))
            )
        msg = req_cls()
        # set fields best-effort
        for f in msg.DESCRIPTOR.fields:
            name = f.name
            if name == text_field or name in ("text", "prompt", "query", "input"):
                with contextlib.suppress(Exception):
                    setattr(msg, name, spec.prompt)
            elif name == model_field or name in ("model", "model_name"):
                with contextlib.suppress(Exception):
                    setattr(msg, name, spec.params.get("model", ""))
        try:
            metadata = list(md.items())
            resp = rpc(msg, metadata=metadata, timeout=spec.timeout_s)
        except Exception as e:
            return NormalizedResponse(text="", raw={"error": str(e)})
        # extract text via response_fields mapping
        rfields = spec.extra.get("grpc", {}).get("response_fields", {}) or {}
        text_path = rfields.get("text", "text")
        text = _extract_field(resp, text_path)
        usage = _extract_field(resp, rfields.get("usage", "")) or {}
        p = int(_dig(usage, "prompt_tokens") or _dig(usage, "input_tokens") or 0)
        c = int(_dig(usage, "completion_tokens") or _dig(usage, "output_tokens") or 0)
        return NormalizedResponse(
            text=str(text or ""),
            prompt_tokens=p,
            completion_tokens=c,
            total_tokens=p + c,
            raw=str(resp)[:4000],
        )


def _extract_field(msg, path: str):
    """Dotted-path getattr over a protobuf response message."""
    if not path:
        return None
    cur = msg
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _dig(obj, key):
    """Dict-key or attribute lookup helper."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)

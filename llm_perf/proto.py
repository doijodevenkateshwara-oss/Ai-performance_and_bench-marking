"""Proto compilation helpers — see providers/grpc.py for runtime path.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

from pathlib import Path


def proto_gen(proto_path: str) -> Path:
    """CLI helper: compile a .proto and print output dir."""
    from .providers.grpc import compile_proto, get_triton_proto_path

    p = Path(proto_path)
    if not p.exists():
        if proto_path == "triton":
            p = get_triton_proto_path()
        else:
            raise FileNotFoundError(proto_path)
    out = compile_proto(p)
    return Path(out)

"""gRPC proto compilation test using the bundled Triton proto.

Author: Venkateshwara Doijode
"""

from pathlib import Path


def test_bundled_proto_compiles(tmp_path):
    """The bundled Triton proto compiles with grpc_tools."""
    proto = Path("protos/inference.proto")
    assert proto.exists()
    from grpc_tools import protoc

    out = tmp_path / "gen"
    out.mkdir()
    rc = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{proto.parent}",
            f"--python_out={out}",
            f"--grpc_python_out={out}",
            str(proto),
        ]
    )
    assert rc == 0
    assert (out / "inference_pb2.py").exists()
    assert (out / "inference_pb2_grpc.py").exists()


def test_triton_proto_path():
    """get_triton_proto_path resolves the bundled proto."""
    from llm_perf.providers.grpc import get_triton_proto_path

    p = get_triton_proto_path(Path("/tmp/llm_perf_protos"))
    assert p.exists()
    assert "service GRPCInferenceService" in p.read_text()

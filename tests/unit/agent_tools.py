"""Helper callables for agent-stage tests.

Author: Venkateshwara Doijode
"""


def echo(input=None, **kwargs):
    """Test tool: echoes its input argument."""
    return f"echo:{input}"

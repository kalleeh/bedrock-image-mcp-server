# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Guards the mcp dependency bounds.

mcp 2.0.0 (published 2026-07-28) removed `mcp.server.fastmcp` outright — FastMCP became
MCPServer under `mcp.server.mcpserver`. Because 0.3.2 declared `mcp[cli]>=1.11.0` with no
upper bound, every fresh install resolved to 2.x and crashed at import with
ModuleNotFoundError before the server could speak protocol. Existing users with a warm uv
cache or a lockfile were unaffected, which is what made it look intermittent.

0.4.0 ported to the 2.0 API, so the range moved to `>=2,<3`: the floor is required because
`mcp.server.mcpserver` does not exist before 2.0, and the ceiling now guards the 2.x -> 3.x
boundary the same way the old one guarded 1.x -> 2.x.

These tests fail if the bounds and the import site in server.py drift apart in either
direction, so a future major bump has to move both together.
"""

import re
from pathlib import Path


def _mcp_requirement() -> str:
    """Return the declared mcp requirement string from pyproject.toml.

    Parsed with a regex rather than tomllib/tomli: tomllib needs 3.11 but this package
    supports 3.10, and tomli is only present transitively (via coverage[toml]), so
    importing it would make this guard depend on an undeclared package.
    """
    pyproject = Path(__file__).resolve().parent.parent / 'pyproject.toml'
    for line in pyproject.read_text().splitlines():
        match = re.match(r"""^\s*["'](mcp\b[^"']*)["']\s*,?\s*$""", line)
        if match:
            return match.group(1)
    raise AssertionError('mcp is not declared in pyproject.toml dependencies')


def _server_source() -> str:
    """Return the source of the server module that declares the mcp import site."""
    server = (
        Path(__file__).resolve().parent.parent
        / 'awslabs'
        / 'bedrock_image_mcp_server'
        / 'server.py'
    )
    return server.read_text()


def test_mcp_dependency_has_upper_bound():
    """The mcp requirement must exclude the next major while the code targets 2.x."""
    requirement = _mcp_requirement()
    assert '<3' in requirement, (
        f'mcp requirement {requirement!r} has no <3 ceiling. mcp 2.0 already broke this '
        'project once by deleting mcp.server.fastmcp, so an unbounded requirement lets the '
        'next major do it again to every fresh install. Drop the ceiling only together '
        'with a port to the 3.0 API.'
    )


def test_mcp_dependency_floor_covers_the_ported_api():
    """The floor must exclude 1.x, which has no mcp.server.mcpserver to import."""
    requirement = _mcp_requirement()
    assert '>=2' in requirement, (
        f'mcp requirement {requirement!r} allows 1.x, but server.py imports '
        'mcp.server.mcpserver, which only exists from 2.0 on. Installing 1.x would crash '
        'at import with ModuleNotFoundError before the server could speak protocol.'
    )


def test_server_imports_the_api_the_bounds_pin():
    """The bounds and the import site must agree on which mcp generation is in use."""
    source = _server_source()
    imports_mcpserver = 'from mcp.server.mcpserver import' in source
    targets_2x = '>=2' in _mcp_requirement()
    assert imports_mcpserver == targets_2x, (
        'server.py and the mcp requirement disagree: the >=2 floor is only correct while '
        'server.py imports from mcp.server.mcpserver. Update both together.'
    )


def test_server_does_not_import_the_removed_fastmcp_module():
    """The removed module has no alias in mcp 2.x, so any reference is a hard crash."""
    assert 'mcp.server.fastmcp' not in _server_source(), (
        'server.py still references mcp.server.fastmcp, which does not exist in mcp 2.x. '
        'There is no compatibility shim: importing it raises ModuleNotFoundError.'
    )

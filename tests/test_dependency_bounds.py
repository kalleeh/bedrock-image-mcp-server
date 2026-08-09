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
"""Guards the mcp dependency ceiling.

mcp 2.0.0 (published 2026-07-28) removed `mcp.server.fastmcp` outright — FastMCP became
MCPServer under `mcp.server.mcpserver`. Because 0.3.2 declared `mcp[cli]>=1.11.0` with no
upper bound, every fresh install resolved to 2.x and crashed at import with
ModuleNotFoundError before the server could speak protocol. Existing users with a warm uv
cache or a lockfile were unaffected, which is what made it look intermittent.

These tests fail if the ceiling is dropped without porting to the 2.0 API.
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


def test_mcp_dependency_has_upper_bound():
    """The mcp requirement must exclude 2.x while the code targets the FastMCP API."""
    requirement = _mcp_requirement()
    assert '<2' in requirement, (
        f'mcp requirement {requirement!r} has no <2 ceiling. mcp 2.x removed '
        'mcp.server.fastmcp, so an unbounded requirement breaks every fresh install. '
        'Drop the ceiling only together with a port to mcp.server.mcpserver.MCPServer.'
    )


def test_server_imports_the_api_the_ceiling_pins():
    """The ceiling and the import site must agree on which mcp generation is in use."""
    server = (
        Path(__file__).resolve().parent.parent
        / 'awslabs'
        / 'bedrock_image_mcp_server'
        / 'server.py'
    )
    source = server.read_text()
    imports_fastmcp = 'from mcp.server.fastmcp import' in source
    ceiling = '<2' in _mcp_requirement()
    assert imports_fastmcp == ceiling, (
        'server.py and the mcp requirement disagree: the <2 ceiling is only correct '
        'while server.py imports from mcp.server.fastmcp. Update both together.'
    )

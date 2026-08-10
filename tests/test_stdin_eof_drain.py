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
"""Tests that a tool call outlives stdin EOF instead of being discarded.

The MCP stdio transport tears the session down the moment stdin hits EOF, cancelling whatever
tool call is running: the client sees ``-32000 Connection closed`` rather than the image it paid
for (python-sdk#2678). ``server._defer_stdin_eof`` interposes a pipe on file descriptor 0 to
hold the transport open until in-flight work finishes.

These tests drive a real server process over real pipes. That is the only way to cover this:
the behaviour lives in file-descriptor handling and process teardown, so an in-process test with
a fake stream would exercise none of it. Only the Bedrock call is stubbed, and it is replaced
with a sleep so a call is reliably in flight when EOF lands.
"""

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from typing import Dict, List, Optional


# Stand-in for the 30-90s a real generation takes, kept short enough to run in CI.
WORK_SECONDS = 3.0

DRIVER = textwrap.dedent("""
    import asyncio, os
    import awslabs.bedrock_image_mcp_server.server as server
    from awslabs.bedrock_image_mcp_server.models.common import ImageGenerationResponse

    async def _slow_generate(**kwargs):
        await asyncio.sleep(float(os.environ['WORK_SECONDS']))
        return ImageGenerationResponse(
            status='success',
            paths=['/tmp/fake.png'],
            message='ok',
            model_id='stability.sd3-5-large-v1:0',
        )

    # Replace only the Bedrock call, so the transport, the middleware and the drain are real.
    server.generate_text_to_image = _slow_generate

    if os.environ.get('DRAIN') == 'off':
        server.mcp.run()          # the unfixed behaviour, for contrast
    else:
        grace = os.environ.get('GRACE_OVERRIDE')
        if grace:
            server.STDIN_EOF_GRACE_SECONDS = float(grace)
        server.main()
""")


def _drive(
    work_seconds: float,
    drain: str,
    grace: Optional[str] = None,
    close_stdin_after: float = 0.5,
) -> Dict:
    """Run a server process, call a slow tool, close stdin mid-call, and report the outcome."""
    env = {
        **os.environ,
        'WORK_SECONDS': str(work_seconds),
        'DRAIN': drain,
        'AWS_REGION': 'us-west-2',
        # Keep the child from trying to reach AWS while building its client.
        'AWS_ACCESS_KEY_ID': 'testing',
        'AWS_SECRET_ACCESS_KEY': 'testing',
        'AWS_SESSION_TOKEN': 'testing',
    }
    env.pop('AWS_PROFILE', None)
    if grace is not None:
        env['GRACE_OVERRIDE'] = grace

    proc = subprocess.Popen(
        [sys.executable, '-c', DRIVER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    assert proc.stdin and proc.stdout and proc.stderr

    # Read on threads: the point of the test is that the child keeps writing after we stop.
    responses: List[Dict] = []

    def read_stdout() -> None:
        assert proc.stdout
        for raw in proc.stdout:
            try:
                responses.append(json.loads(raw))
            except ValueError:
                pass

    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()
    threading.Thread(target=proc.stderr.read, daemon=True).start()

    def send(message: Dict) -> None:
        assert proc.stdin
        proc.stdin.write((json.dumps(message) + '\n').encode())
        proc.stdin.flush()

    started = time.monotonic()
    send(
        {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'initialize',
            'params': {
                'protocolVersion': '2025-06-18',
                'capabilities': {},
                'clientInfo': {'name': 'test', 'version': '1'},
            },
        }
    )
    # Wait for the handshake before calling a tool, so the call is not rejected pre-init.
    while not any(r.get('id') == 1 for r in responses):
        if time.monotonic() - started > 30:
            raise AssertionError('server never answered initialize')
        time.sleep(0.05)

    send({'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {}})
    call_sent = time.monotonic()
    send(
        {
            'jsonrpc': '2.0',
            'id': 2,
            'method': 'tools/call',
            'params': {'name': 'generate_image_sd35', 'arguments': {'prompt': 'a cat'}},
        }
    )

    time.sleep(close_stdin_after)
    proc.stdin.close()  # EOF while the tool is still running

    proc.wait(timeout=work_seconds + 90)
    reader.join(timeout=5)

    call_response = next((r for r in responses if r.get('id') == 2), None)
    return {
        'response': call_response,
        'exit_code': proc.returncode,
        'elapsed': time.monotonic() - call_sent,
    }


class TestStdinEofDrain:
    """Tests for holding the stdio session open past EOF while work is in flight."""

    def test_tool_result_survives_stdin_closing_mid_call(self):
        """A call still running when stdin closes must return its result, not a dropped session.

        This is the regression: without the drain the client gets ``-32000 Connection closed``
        and the generated image is thrown away.
        """
        outcome = _drive(WORK_SECONDS, drain='on')

        assert outcome['response'] is not None, (
            'no response to the tool call: the session died at stdin EOF'
        )
        assert 'error' not in outcome['response'], (
            f'tool call failed after stdin EOF: {outcome["response"]["error"]}'
        )
        assert outcome['response']['result']['isError'] is False
        # The call really did outlive EOF rather than finishing before it.
        assert outcome['elapsed'] >= WORK_SECONDS

    def test_without_the_drain_the_call_is_dropped(self):
        """Pin the upstream behaviour the drain exists to correct.

        If this ever stops failing, python-sdk#2678 was fixed upstream and the drain in
        ``server.main()`` can be reconsidered.
        """
        outcome = _drive(WORK_SECONDS, drain='off')

        dropped = outcome['response'] is None or 'error' in outcome['response']
        assert dropped, (
            'the stdio transport now survives stdin EOF on its own; '
            're-evaluate whether _defer_stdin_eof is still needed'
        )

    def test_grace_period_bounds_the_wait(self):
        """A call that outruns the grace period must not wedge the process open.

        EOF-triggered shutdown is deliberate upstream (python-sdk#2231) because it stops a
        server outliving a dead client, so the drain has to give up.
        """
        grace = 1.0
        work = 30.0
        outcome = _drive(work, drain='on', grace=str(grace))

        assert outcome['exit_code'] is not None, 'server did not exit'
        assert outcome['elapsed'] < work, (
            f'waited {outcome["elapsed"]:.1f}s for a {work}s call: the grace period did not bound it'
        )

    def test_eof_while_idle_shuts_down_promptly(self):
        """With nothing in flight, EOF must shut down at once rather than wait out the grace."""
        env = {
            **os.environ,
            'WORK_SECONDS': '0',
            'DRAIN': 'on',
            'GRACE_OVERRIDE': '120',
            'AWS_REGION': 'us-west-2',
            'AWS_ACCESS_KEY_ID': 'testing',
            'AWS_SECRET_ACCESS_KEY': 'testing',
            'AWS_SESSION_TOKEN': 'testing',
        }
        env.pop('AWS_PROFILE', None)

        proc = subprocess.Popen(
            [sys.executable, '-c', DRIVER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        assert proc.stdin and proc.stdout
        threading.Thread(target=proc.stdout.read, daemon=True).start()

        started = time.monotonic()
        proc.stdin.write(
            (
                json.dumps(
                    {
                        'jsonrpc': '2.0',
                        'id': 1,
                        'method': 'initialize',
                        'params': {
                            'protocolVersion': '2025-06-18',
                            'capabilities': {},
                            'clientInfo': {'name': 'test', 'version': '1'},
                        },
                    }
                )
                + '\n'
            ).encode()
        )
        proc.stdin.flush()
        time.sleep(1.0)
        proc.stdin.close()

        proc.wait(timeout=60)
        assert time.monotonic() - started < 30, 'idle server waited out the grace period'

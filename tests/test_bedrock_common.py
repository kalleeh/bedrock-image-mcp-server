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
"""Tests for the bedrock_common module of the bedrock-image-mcp-server."""

import base64
import json
import os
import pytest
from awslabs.bedrock_image_mcp_server.models.common import OutputFormat
from awslabs.bedrock_image_mcp_server.services.bedrock_common import (
    BedrockAPIError,
    ContentFilterError,
    invoke_bedrock_model,
    resolve_output_path,
    sanitize_filename,
    save_images,
)
from botocore.exceptions import ClientError
from unittest.mock import MagicMock


MODEL_ID = 'us.stability.stable-image-core-v1:1'


def _client_raising(error: Exception) -> MagicMock:
    """Build a mock Bedrock client whose invoke_model raises the given error.

    Args:
        error: Exception the client should raise when invoked.

    Returns:
        A mock client suitable for passing to invoke_bedrock_model.
    """
    client = MagicMock()
    client.invoke_model.side_effect = error
    return client


def _client_returning(payload: dict) -> MagicMock:
    """Build a mock Bedrock client that returns the given payload as a JSON body.

    Args:
        payload: Response document the fake Bedrock body should contain.

    Returns:
        A mock client suitable for passing to invoke_bedrock_model.
    """
    client = MagicMock()
    body = MagicMock()
    body.read.return_value = json.dumps(payload).encode('utf-8')
    client.invoke_model.return_value = {'body': body}
    return client


def _client_error(code: str, message: str = 'boom') -> ClientError:
    """Build a real botocore ClientError with the given AWS error code.

    Args:
        code: AWS error code, e.g. 'ThrottlingException'.
        message: AWS error message.

    Returns:
        A ClientError instance shaped like a real InvokeModel failure.
    """
    return ClientError(
        {'Error': {'Code': code, 'Message': message}},
        'InvokeModel',
    )


class TestInvokeBedrockModelSuccess:
    """Tests for the successful path of invoke_bedrock_model."""

    async def test_returns_decoded_response_and_serializes_request(self):
        """Test the response body is decoded and the request body sent as JSON."""
        client = _client_returning({'images': ['abc'], 'seeds': [7]})

        result = await invoke_bedrock_model(
            model_id=MODEL_ID,
            request_body={'prompt': 'a cat'},
            bedrock_client=client,
        )

        assert result == {'images': ['abc'], 'seeds': [7]}
        call_kwargs = client.invoke_model.call_args[1]
        assert call_kwargs['modelId'] == MODEL_ID
        assert json.loads(call_kwargs['body']) == {'prompt': 'a cat'}

    async def test_null_finish_reasons_are_treated_as_success(self):
        """Test a finish_reasons list of nulls does not trigger content filtering."""
        client = _client_returning({'images': ['abc'], 'finish_reasons': [None, None]})

        result = await invoke_bedrock_model(
            model_id=MODEL_ID,
            request_body={'prompt': 'a cat'},
            bedrock_client=client,
        )

        assert result['images'] == ['abc']


class TestContentFiltering:
    """Tests for content-filter detection in invoke_bedrock_model."""

    async def test_non_null_finish_reason_raises_content_filter_error(self):
        """Test a non-null finish reason raises ContentFilterError with that reason."""
        client = _client_returning(
            {'images': ['abc'], 'finish_reasons': ['Filter reason: prompt']}
        )

        with pytest.raises(ContentFilterError) as exc_info:
            await invoke_bedrock_model(
                model_id=MODEL_ID,
                request_body={'prompt': 'blocked'},
                bedrock_client=client,
            )

        error = exc_info.value
        assert error.reason == 'Filter reason: prompt'
        assert error.error_code == 'ContentFiltered'
        assert error.retryable is False
        assert 'Filter reason: prompt' in str(error)

    async def test_content_filter_error_is_not_reclassified(self):
        """Test ContentFilterError is re-raised rather than caught by the catch-all."""
        client = _client_returning(
            {'images': ['abc'], 'finish_reasons': [None, 'CONTENT_FILTERED']}
        )

        with pytest.raises(ContentFilterError) as exc_info:
            await invoke_bedrock_model(
                model_id=MODEL_ID,
                request_body={'prompt': 'blocked'},
                bedrock_client=client,
            )

        # The generic handler would rewrite these to 'UnexpectedError'.
        assert exc_info.value.error_code == 'ContentFiltered'
        assert 'Unexpected error' not in str(exc_info.value)


class TestClientErrorClassification:
    """Tests for the ClientError to BedrockAPIError classification ladder."""

    @pytest.mark.parametrize(
        ('error_code', 'retryable'),
        [
            ('ValidationException', False),
            ('AccessDeniedException', False),
            ('ThrottlingException', True),
            ('ModelNotReadyException', True),
            ('ServiceUnavailableException', True),
            ('InternalServerException', True),
        ],
    )
    async def test_known_error_codes_are_classified(self, error_code, retryable):
        """Test each known AWS error code maps to the right code and retryable flag."""
        client = _client_raising(_client_error(error_code))

        with pytest.raises(BedrockAPIError) as exc_info:
            await invoke_bedrock_model(
                model_id=MODEL_ID,
                request_body={'prompt': 'a cat'},
                bedrock_client=client,
            )

        error = exc_info.value
        assert error.error_code == error_code
        assert error.retryable is retryable
        # A ClientError must not be downgraded to the generic catch-all classification.
        assert error.error_code != 'UnexpectedError'
        assert not isinstance(error, ContentFilterError)

    async def test_validation_exception_surfaces_aws_message(self):
        """Test the AWS-provided message is included for validation failures."""
        client = _client_raising(
            _client_error('ValidationException', 'height must be a multiple of 64')
        )

        with pytest.raises(BedrockAPIError) as exc_info:
            await invoke_bedrock_model(
                model_id=MODEL_ID,
                request_body={'height': 100},
                bedrock_client=client,
            )

        assert 'height must be a multiple of 64' in exc_info.value.message

    async def test_access_denied_message_mentions_model_id(self):
        """Test the access-denied message names the model that was refused."""
        client = _client_raising(_client_error('AccessDeniedException'))

        with pytest.raises(BedrockAPIError) as exc_info:
            await invoke_bedrock_model(
                model_id=MODEL_ID,
                request_body={'prompt': 'a cat'},
                bedrock_client=client,
            )

        assert MODEL_ID in exc_info.value.message

    async def test_model_not_ready_message_mentions_model_id(self):
        """Test the model-not-ready message names the model that was not ready."""
        client = _client_raising(_client_error('ModelNotReadyException'))

        with pytest.raises(BedrockAPIError) as exc_info:
            await invoke_bedrock_model(
                model_id=MODEL_ID,
                request_body={'prompt': 'a cat'},
                bedrock_client=client,
            )

        assert MODEL_ID in exc_info.value.message

    async def test_unknown_error_code_is_preserved_and_not_retryable(self):
        """Test an unrecognized AWS error code is passed through as non-retryable."""
        client = _client_raising(_client_error('ResourceNotFoundException', 'no such model'))

        with pytest.raises(BedrockAPIError) as exc_info:
            await invoke_bedrock_model(
                model_id=MODEL_ID,
                request_body={'prompt': 'a cat'},
                bedrock_client=client,
            )

        error = exc_info.value
        assert error.error_code == 'ResourceNotFoundException'
        assert error.retryable is False
        assert 'no such model' in error.message


class TestUnexpectedErrorClassification:
    """Tests for the catch-all error handler in invoke_bedrock_model."""

    async def test_non_client_error_becomes_unexpected_error(self):
        """Test a non-AWS exception is wrapped as a non-retryable UnexpectedError."""
        client = _client_raising(RuntimeError('socket closed'))

        with pytest.raises(BedrockAPIError) as exc_info:
            await invoke_bedrock_model(
                model_id=MODEL_ID,
                request_body={'prompt': 'a cat'},
                bedrock_client=client,
            )

        error = exc_info.value
        assert error.error_code == 'UnexpectedError'
        assert error.retryable is False
        assert 'socket closed' in error.message

    async def test_malformed_response_body_becomes_unexpected_error(self):
        """Test an undecodable response body is reported as an UnexpectedError."""
        client = MagicMock()
        body = MagicMock()
        body.read.return_value = b'not json'
        client.invoke_model.return_value = {'body': body}

        with pytest.raises(BedrockAPIError) as exc_info:
            await invoke_bedrock_model(
                model_id=MODEL_ID,
                request_body={'prompt': 'a cat'},
                bedrock_client=client,
            )

        assert exc_info.value.error_code == 'UnexpectedError'


class TestSaveImagesContainment:
    """Tests that save_images confines writes to the workspace output directory."""

    @pytest.fixture
    def image_b64(self):
        """Return base64-encoded placeholder image bytes."""
        return base64.b64encode(b'image-bytes').decode('utf-8')

    @pytest.mark.parametrize(
        'prefix',
        [
            '../escaped',
            '../../escaped',
            'nested/escaped',
            '..\\escaped',
            '/etc/escaped',
        ],
    )
    def test_traversal_prefixes_stay_inside_output_dir(self, tmp_path, image_b64, prefix):
        """Test that traversal and absolute prefixes cannot write outside the output dir."""
        workspace = tmp_path / 'ws'
        workspace.mkdir()
        expected_dir = workspace / 'output'

        paths = save_images(
            base64_images=[image_b64],
            workspace_dir=str(workspace),
            filename_prefix=prefix,
            output_format=OutputFormat.PNG,
        )

        assert len(paths) == 1
        assert os.path.dirname(paths[0]) == str(expected_dir)
        assert os.path.exists(paths[0])
        # Nothing was created next to the workspace or anywhere above it.
        assert sorted(p.name for p in tmp_path.iterdir()) == ['ws']
        assert sorted(p.name for p in workspace.iterdir()) == ['output']

    def test_prefix_of_only_dots_falls_back_to_default(self, tmp_path, image_b64):
        """Test a prefix that sanitizes to nothing produces a usable default filename."""
        workspace = tmp_path / 'ws'
        workspace.mkdir()

        paths = save_images(
            base64_images=[image_b64],
            workspace_dir=str(workspace),
            filename_prefix='../..',
            output_format=OutputFormat.PNG,
        )

        assert os.path.basename(paths[0]).startswith('image_')
        assert os.path.dirname(paths[0]) == str(workspace / 'output')

    def test_invalid_base64_is_rejected(self, tmp_path):
        """Test that undecodable image data raises instead of writing a corrupt file."""
        workspace = tmp_path / 'ws'
        workspace.mkdir()

        with pytest.raises(IOError):
            save_images(
                base64_images=['not!valid!base64'],
                workspace_dir=str(workspace),
                filename_prefix='img',
                output_format=OutputFormat.PNG,
            )


class TestSanitizeFilenameHardening:
    """Tests for adversarial inputs to sanitize_filename."""

    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [
            ('foo\x00bar', 'foobar'),
            ('C:evil', 'C'),
            ('stream:hidden', 'stream'),
            ('....', 'fallback'),
            ('   ', 'fallback'),
            ('/', 'fallback'),
            ('a' * 400, 'a' * 100),
        ],
    )
    def test_hostile_names_are_reduced_to_safe_basenames(self, raw, expected):
        """Test control characters, drive/stream qualifiers, and overlong names are handled."""
        assert sanitize_filename(raw, 'fallback') == expected

    def test_sanitized_names_are_writable(self, tmp_path):
        """Test that a sanitized hostile filename can actually be written to disk."""
        image = base64.b64encode(b'bytes').decode('utf-8')

        paths = save_images(
            base64_images=[image],
            workspace_dir=str(tmp_path),
            filename_prefix='x' * 400 + '\x00',
            output_format=OutputFormat.PNG,
        )

        assert os.path.exists(paths[0])
        assert len(os.path.basename(paths[0])) < 255


class TestResolveOutputPath:
    """Tests for the output-directory containment check."""

    def test_names_inside_the_directory_are_allowed(self, tmp_path):
        """Test a plain basename resolves inside the output directory."""
        assert resolve_output_path(str(tmp_path), 'img.png') == str(tmp_path / 'img.png')

    @pytest.mark.parametrize('name', ['../escape.png', 'sub/../../escape.png'])
    def test_escaping_names_are_rejected(self, tmp_path, name):
        """Test that a name resolving outside the output directory raises."""
        with pytest.raises(ValueError, match='outside the output directory'):
            resolve_output_path(str(tmp_path), name)

    def test_symlinked_output_dir_is_resolved_not_escaped(self, tmp_path):
        """Test a symlinked output directory still contains its own files."""
        real = tmp_path / 'real'
        real.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real)

        resolved = resolve_output_path(str(link), 'img.png')

        assert resolved == str(real / 'img.png')


class TestBase64Whitespace:
    """Tests that line-wrapped base64 is accepted while corrupt data is not."""

    def test_line_wrapped_base64_is_accepted(self, tmp_path):
        """Test 76-column wrapped base64 with a trailing newline still saves."""
        wrapped = base64.encodebytes(b'x' * 200).decode('utf-8')
        assert '\n' in wrapped

        paths = save_images(
            base64_images=[wrapped],
            workspace_dir=str(tmp_path),
            filename_prefix='img',
            output_format=OutputFormat.PNG,
        )

        with open(paths[0], 'rb') as f:
            assert f.read() == b'x' * 200

    def test_non_alphabet_characters_are_still_rejected(self, tmp_path):
        """Test data whose padding is valid but alphabet is not fails instead of truncating."""
        with pytest.raises(IOError):
            save_images(
                base64_images=['aGVsbG8*d29ybGQ='],
                workspace_dir=str(tmp_path),
                filename_prefix='img',
                output_format=OutputFormat.PNG,
            )

"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import logging
import typing
from typing import ClassVar
from time import time
from datetime import datetime

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from azure.identity import DefaultAzureCredential

from .config import DEFAULT_MAX_TOKENS, LLMConfig
from .openai_base_client import BaseOpenAIClient

logger = logging.getLogger(__name__)

# Pyparus configuration
PYPARUS_DEFAULT_MODEL = os.getenv('PYPARUS_DEFAULT_MODEL', 'GPT4oMini-Batch')
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gpt-4o-mini')

PAPYRUS_HEADER = {
    "Authorization": "",  # Will be set by refresh_access_token()
    "Content-Type": "application/json",
    "papyrus-model-name": PYPARUS_DEFAULT_MODEL,
    "papyrus-quota-id": "DeepSeekAMDAds",
    "papyrus-timeout-ms": "1800000"
}
verify_scope = os.getenv('VERIFY_SCOPE')
cur_credential = DefaultAzureCredential()

def refresh_access_token():
    """Refresh the access token and update the Authorization header only when necessary"""
    global PAPYRUS_HEADER, access_token_expiry
   
    current_time = time()
   
    # Only refresh if the token doesn't exist or is about to expire (within 3 minutes)
    if not hasattr(refresh_access_token, 'access_token_expiry') or current_time >= getattr(refresh_access_token, 'access_token_expiry', 0) - 180:
        if verify_scope is None:
            raise ValueError("VERIFY_SCOPE environment variable is not set")
        token_response = cur_credential.get_token(verify_scope)
        access_token = token_response.token
       
        # Store token expiry time (convert to Unix timestamp)
        # Default to 1 hour (3600 seconds) if expires_on is not available
        if hasattr(token_response, 'expires_on'):
            refresh_access_token.access_token_expiry = token_response.expires_on
        else:
            refresh_access_token.access_token_expiry = current_time + 3600
           
        PAPYRUS_HEADER["Authorization"] = f"Bearer {access_token}"
        logger.info(f"Access token refreshed, expires at {datetime.fromtimestamp(refresh_access_token.access_token_expiry).strftime('%Y-%m-%d %H:%M:%S')}")
        return access_token
    else:
        # Token is still valid
        logger.debug("Using existing access token")
        return PAPYRUS_HEADER["Authorization"].replace("Bearer ", "")


class OpenAIClient(BaseOpenAIClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the BaseOpenAIClient and provides OpenAI-specific implementation
    for creating completions.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
        total_tokens (int): The total number of tokens used in all requests.
        total_prompt_tokens (int): The total number of prompt tokens used in all requests.
        total_completion_tokens (int): The total number of completion tokens used in all requests.
        request_count (int): The total number of API requests made.
        token_usage_history (list): History of token usage by request.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 10

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the OpenAIClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
        """
        super().__init__(config, cache, max_tokens)

        if config is None:
            config = LLMConfig()

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

        self.is_papyrus = "papyrus" in str(self.client.base_url)
        self.max_tokens = max_tokens
       
        # Initialize token and request tracking
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.request_count = 0
        self.token_usage_history = []

    def _track_token_usage(self, response, model: str):
        """Track token usage from the response"""
        # Increment request count
        self.request_count += 1
       
        # Track token usage if available in the response
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
            completion_tokens = getattr(response.usage, 'completion_tokens', 0)
            total_tokens = getattr(response.usage, 'total_tokens', 0)
           
            # Update total tokens
            self.total_tokens += total_tokens
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
           
            # Record token usage for this request
            usage_record = {
                'timestamp': datetime.now().isoformat(),
                'model': model,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens
            }
            if len(self.token_usage_history) < 50:
                self.token_usage_history.append(usage_record)
           
            logger.debug(f"Request #{self.request_count}: Used {total_tokens} tokens ({prompt_tokens} prompt, {completion_tokens} completion)")

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
    ):
        """Create a structured completion using OpenAI's beta parse API."""
        if self.is_papyrus:
            refresh_access_token()
            
        response = await self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_model,  # type: ignore
            extra_headers=PAPYRUS_HEADER if self.is_papyrus else None,
        )
        
        # Track token usage
        self._track_token_usage(response, model)
        return response

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
    ):
        """Create a regular completion with JSON format."""
        if self.is_papyrus:
            refresh_access_token()
            
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={'type': 'json_object'},
            extra_headers=PAPYRUS_HEADER if self.is_papyrus else None,
        )
        
        # Track token usage
        self._track_token_usage(response, model)
        return response


    def get_token_stats(self) -> dict:
        """
        Returns statistics about token usage and request count.
       
        Returns:
            dict: A dictionary containing statistics about token usage and request count.
        """
        avg_tokens = 0
        if self.request_count > 0:
            avg_tokens = self.total_tokens / self.request_count

        return {
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "request_count": self.request_count,
            "token_usage_history": self.token_usage_history[-10:] if self.token_usage_history else [],
            "average_tokens_per_request": avg_tokens
        }

    def reset_token_stats(self):
        """
        Resets the token usage statistics.
        """
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.request_count = 0
        self.token_usage_history = []
        logger.info("Token usage statistics have been reset")
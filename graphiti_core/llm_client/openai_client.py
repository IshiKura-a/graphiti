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

import asyncio
import os
import logging
import typing
from typing import ClassVar

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from time import time
from datetime import datetime
from azure.identity import DefaultAzureCredential
 
from ..prompts.models import Message
from .client import MULTILINGUAL_EXTRACTION_RESPONSES, LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError, RefusalError
logger = logging.getLogger(__name__)
 
# gpt-4.1-mini
# PYPARUS_DEFAULT_MODEL = 'gpt-41-mini-2025-04-14-eval'
# PYPARUS_DEFAULT_SMALL_MODEL = 'gpt-41-mini-2025-04-14-eval'
# DEFAULT_SMALL_MODEL = 'gpt-4.1-nano'
# DEFAULT_MODEL = 'gpt-4.1-mini'
# DEFAULT_SMALL_MODEL = 'gpt-4.1-mini'
 
# gpt-4o-mini

PYPARUS_DEFAULT_MODEL = os.getenv('PYPARUS_DEFAULT_MODEL', 'GPT4oMini-Batch')
PYPARUS_DEFAULT_SMALL_MODEL = os.getenv('PYPARUS_DEFAULT_SMALL_MODEL', 'GPT4oMini-Batch')
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gpt-4o-mini')
DEFAULT_SMALL_MODEL = os.getenv('DEFAULT_SMALL_MODEL', 'gpt-4o-mini')

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
 
class OpenAIClient(LLMClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the LLMClient and provides methods to initialize the client,
    get an embedder, and generate responses from the language model.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.
        total_tokens (int): The total number of tokens used in all requests.
        total_prompt_tokens (int): The total number of prompt tokens used in all requests.
        total_completion_tokens (int): The total number of completion tokens used in all requests.
        request_count (int): The total number of API requests made.
        token_usage_history (list): History of token usage by request.
 
    Methods:
        __init__(config: LLMConfig | None = None, cache: bool = False, client: typing.Any = None):
            Initializes the OpenAIClient with the provided configuration, cache setting, and client.

        _generate_response(messages: list[Message]) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
           
        get_token_stats() -> dict:
            Returns statistics about token usage and request count.
           
        reset_token_stats():
            Resets the token usage statistics.
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
        # removed caching to simplify the `generate_response` override
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenAI')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

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
 
    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})

        if self.is_papyrus:
            refresh_access_token()
        try:
            if model_size == ModelSize.small:
                model = self.small_model or (PYPARUS_DEFAULT_SMALL_MODEL if self.is_papyrus else DEFAULT_SMALL_MODEL)
            else:
                model = self.model or (PYPARUS_DEFAULT_MODEL if self.is_papyrus else DEFAULT_MODEL)

            response = await self.client.beta.chat.completions.parse(
                model=model,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                response_format=response_model,  # type: ignore
                extra_headers=PAPYRUS_HEADER if self.is_papyrus else None,
            )
           
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
 
            response_object = response.choices[0].message

            if response_object.parsed:
                return response_object.parsed.model_dump()
            elif response_object.refusal:
                raise RefusalError(response_object.refusal)
            else:
                raise Exception(f'Invalid response from LLM: {response_object.model_dump()}')
        except openai.LengthFinishReasonError as e:
            raise Exception(f'Output length exceeded max tokens {self.max_tokens}: {e}') from e
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        retry_count = 0
        last_error = None

        # Add multilingual extraction instructions
        messages[0].content += MULTILINGUAL_EXTRACTION_RESPONSES

        while retry_count <= self.MAX_RETRIES:
            try:
                response = await self._generate_response(
                    messages, response_model, max_tokens, model_size
                )
                return response
            except (RefusalError):
                # These errors should not trigger retries
                raise
            except Exception as e:
                last_error = e
 
                if "Could not parse response content" in str(e):
                    print(f"Skip, Error parsing response: {e}, ", flush=True)
                    break  # Stop retrying on parsing errors
                # Don't retry if we've hit the max retries
                if retry_count >= self.MAX_RETRIES:
                    logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}')
                    raise

                retry_count += 1

                # Construct a detailed error message for the LLM
                error_context = (
                    f'The previous response attempt was invalid. '
                    f'Error type: {e.__class__.__name__}. '
                    f'Error details: {str(e)}. '
                    f'Please try again with a valid response, ensuring the output matches '
                    f'the expected format and constraints.'
                )

                error_message = Message(role='user', content=error_context)
                messages.append(error_message)
                if "Request timed out" not in str(e):
                    logger.warning(
                        f'Retrying after application error (attempt {retry_count}/{self.MAX_RETRIES}): {e}'
                    )

                await asyncio.sleep(4) # Optional: Add a delay before retrying
 
        # If we somehow get here, raise the last error
        raise last_error or Exception('Max retries exceeded with no specific error')

    def get_token_stats(self) -> dict:
        """
        Returns statistics about token usage and request count.
       
        Returns:
            dict: A dictionary containing statistics about token usage and request count.
                - total_tokens (int): Total number of tokens used across all requests
                - total_prompt_tokens (int): Total number of prompt tokens used across all requests
                - total_completion_tokens (int): Total number of completion tokens used across all requests
                - request_count (int): Total number of requests made
                - token_usage_history (list): Detailed history of token usage by request
                - average_tokens_per_request (float): Average number of tokens per request
        """
        avg_tokens = 0
        if self.request_count > 0:
            avg_tokens = self.total_tokens / self.request_count

        return {
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "request_count": self.request_count,
            "token_usage_history": self.token_usage_history[0],
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
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
import logging
from typing import Any

import numpy as np
import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI

from ..helpers import semaphore_gather
from ..llm_client import LLMConfig, RateLimitError
from ..prompts import Message
from .client import CrossEncoderClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4.1-nano'


class OpenAIRerankerClient(CrossEncoderClient):
    def __init__(
        self,
        config: LLMConfig | None = None,
        client: AsyncOpenAI | AsyncAzureOpenAI | None = None,
    ):
        """
        Initialize the OpenAIRerankerClient with the provided configuration and client.

        This reranker uses the OpenAI API to run a simple boolean classifier prompt concurrently
        for each passage. Log-probabilities are used to rank the passages.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            client (AsyncOpenAI | AsyncAzureOpenAI | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
        """
        if config is None:
            config = LLMConfig()

        self.config = config
        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        openai_messages_list: Any = [
            [
                Message(
                    role='system',
                    content='You are an expert tasked with determining whether the passage is relevant to the query',
                ),
                Message(
                    role='user',
                    content=f"""
                           Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise.
                           <PASSAGE>
                           {passage}
                           </PASSAGE>
                           <QUERY>
                           {query}
                           </QUERY>
                           """,
                ),
            ]
            for passage in passages
        ]
        
        # Create individual tasks with error handling and retry logic
        async def score_passage(openai_messages):
            max_retries = 10
            base_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=DEFAULT_MODEL,
                        messages=openai_messages,
                        temperature=0,
                        max_tokens=1,
                        logit_bias={'6432': 1, '7983': 1},
                        logprobs=True,
                        top_logprobs=2,
                    )
                    return response, None
                except openai.RateLimitError as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f'Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})')
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f'Rate limit hit after {max_retries} attempts')
                        return None, e
                except Exception as e:
                    return None, e
        
        try:
            # Execute all scoring tasks
            results = await semaphore_gather(
                *[score_passage(openai_messages) for openai_messages in openai_messages_list]
            )
            
            # Count failures
            failed_count = sum(1 for _, error in results if error is not None)
            total_count = len(results)
            failure_rate = failed_count / total_count if total_count > 0 else 0
            
            # If failure rate exceeds 10%, raise error
            if failure_rate > 0.1:
                logger.error(f'Too many scoring failures: {failed_count}/{total_count} ({failure_rate:.1%})')
                # Find the first error to raise
                for _, error in results:
                    if error is not None:
                        raise error
            
            # Process results, setting score to 0 for failed cases
            scores: list[float] = []
            for i, (response, error) in enumerate(results):
                if error is not None:
                    logger.warning(f'Failed to score passage {i}: {error}')
                    scores.append(0.0)
                    continue
                
                if response is None:
                    scores.append(0.0)
                    continue
                
                top_logprobs = (
                    response.choices[0].logprobs.content[0].top_logprobs
                    if response.choices[0].logprobs is not None
                    and response.choices[0].logprobs.content is not None
                    else []
                )
                
                if len(top_logprobs) == 0:
                    scores.append(0.0)
                    continue
                norm_logprobs = np.exp(top_logprobs[0].logprob)
                if bool(top_logprobs[0].token):
                    scores.append(norm_logprobs)
                else:
                    scores.append(1 - norm_logprobs)

            results = [(passage, score) for passage, score in zip(passages, scores, strict=True)]
            results.sort(reverse=True, key=lambda x: x[1])
            return results
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise

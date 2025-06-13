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

from collections.abc import Iterable
from typing import Optional, Union
import asyncio

from sentence_transformers import SentenceTransformer

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'BAAI/bge-m3'


class BGEEmbedderConfig(EmbedderConfig):
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    device: Optional[str] = None  # 'cpu', 'cuda', etc.


class BGEEmbedder(EmbedderClient):
    """
    BGE Embedder Client

    This client uses SentenceTransformer with BGE models, particularly 'BAAI/bge-m3'.
    """

    def __init__(
        self,
        config: BGEEmbedderConfig = None,
        model: SentenceTransformer = None,
    ):

        self.config = config or BGEEmbedderConfig()

        self.model = model or SentenceTransformer(
            self.config.embedding_model, 
            device=self.config.device
        )

    async def create(
        self, input_data: Union[str, list[str], Iterable[int], Iterable[Iterable[int]]]
    ) -> list[float]:
        """
        Create embeddings for a single input or list of inputs.
        Returns a single embedding for the first input.
        """
        # Wrapping synchronous encode in asyncio to match the async interface
        # The model.encode() function is synchronous
        if isinstance(input_data, str):
            input_data = [input_data]
        
        result = await asyncio.to_thread(self.model.encode, input_data, show_progress_bar=False)
        
        return result[0, :self.config.embedding_dim].tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """
        Create embeddings for a batch of inputs.
        Returns a list of embeddings.
        """
        result = await asyncio.to_thread(self.model.encode, input_data_list, show_progress_bar=False)
        
        return result[:, :self.config.embedding_dim].tolist()

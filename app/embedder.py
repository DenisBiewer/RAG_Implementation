from __future__ import annotations

"""
Dense embedding utilities built around the Contriever encoder.

This module provides the retrieval-side text encoder used by the current RAG
pipeline. It is responsible for turning raw query or chunk text into dense
vector representations that can be stored in, or searched against, the FAISS
index.

Implementation notes:
- the encoder backbone is loaded through Hugging Face Transformers
- token-level hidden states are mean pooled over non-padding tokens
- each embedding is L2-normalized before it is returned

The normalization step is important because the downstream FAISS index uses
inner-product search. Once vectors are normalized to unit length, inner product
is equivalent to cosine similarity.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class ContrieverEmbedder:
    """
    Encode text into dense, retrieval-oriented Contriever embeddings.

    This wrapper is intentionally lightweight: it loads a pretrained Contriever
    encoder, tokenizes input text, pools token representations into one vector
    per text, and returns a NumPy array suitable for FAISS indexing or search.

    The embedding procedure is:
    1. tokenize each input string with truncation and padding
    2. run the encoder to obtain token-level hidden states
    3. mean-pool across non-padding positions only
    4. L2-normalize the pooled vector

    The final vectors are `float32` and have shape
    `(number_of_texts, hidden_size)`.
    """

    def __init__(self, model_name: str = "facebook/contriever", device: str = "cpu") -> None:
        """
        Initialize the tokenizer and encoder model.

        Args:
            model_name: Hugging Face model identifier for the embedding model.
                The default uses Facebook's Contriever checkpoint.
            device: Torch device string used for inference, for example
                `"cpu"` or `"cuda"`.

        Notes:
            The model is switched to evaluation mode immediately because this
            component is only used for inference in the current pipeline.
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 16,
        max_length: int = 512,
    ) -> np.ndarray:
        """
        Encode one or more texts into dense `float32` embeddings.

        Args:
            texts: Input texts to encode. Each string is converted into one
                embedding vector.
            batch_size: Number of texts processed in a single forward pass.
                Larger values can improve throughput but require more memory.
            max_length: Maximum token length after truncation. Text longer than
                this limit is truncated by the tokenizer.

        Returns:
            A NumPy array with shape `(len(texts), hidden_size)` and dtype
            `float32`. The returned vectors are L2-normalized.

        Raises:
            ValueError: If `batch_size` or `max_length` is not a positive
                integer.

        Special cases:
            If `texts` is empty, the method returns an empty array with shape
            `(0, hidden_size)` instead of failing.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**toks).last_hidden_state
            attention_mask = toks["attention_mask"].unsqueeze(-1).to(outputs.dtype)

            # Mean-pool only across real tokens so padding does not affect the embedding.
            summed = (outputs * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1.0)
            emb = summed / counts

            # Normalize so cosine similarity can be computed with a dot product downstream.
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)

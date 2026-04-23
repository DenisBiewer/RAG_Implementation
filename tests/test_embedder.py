import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from app.embedder import ContrieverEmbedder


class FakeBatch(dict):
    """
    Minimal fake tokenizer output that behaves like Hugging Face BatchEncoding
    for the methods used in ContrieverEmbedder.
    """

    def to(self, device):
        moved = {}
        for k, v in self.items():
            if hasattr(v, "to"):
                moved[k] = v.to(device)
            else:
                moved[k] = v
        return FakeBatch(moved)


class FakeModelOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class TestContrieverEmbedder(unittest.TestCase):
    @patch("app.embedder.AutoModel.from_pretrained")
    @patch("app.embedder.AutoTokenizer.from_pretrained")
    def test_init_loads_tokenizer_and_model(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        # model.to(device) should return the model object itself
        mock_model.to.return_value = mock_model

        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model

        embedder = ContrieverEmbedder(model_name="facebook/contriever", device="cpu")

        mock_tokenizer_from_pretrained.assert_called_once_with("facebook/contriever")
        mock_model_from_pretrained.assert_called_once_with("facebook/contriever")
        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()

        self.assertIs(embedder.tokenizer, mock_tokenizer)
        self.assertIs(embedder.model, mock_model)
        self.assertEqual(embedder.device, "cpu")

    @patch("app.embedder.AutoModel.from_pretrained")
    @patch("app.embedder.AutoTokenizer.from_pretrained")
    def test_encode_returns_float32_numpy_array(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        mock_model.to.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model

        # Two texts, each with 3 tokens, hidden dim = 2
        attention_mask = torch.tensor([
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=torch.float32)

        input_ids = torch.tensor([
            [10, 11, 0],
            [20, 21, 22],
        ], dtype=torch.long)

        tokenized = FakeBatch({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

        mock_tokenizer.return_value = tokenized

        last_hidden_state = torch.tensor([
            [[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]],  # masked token should be ignored
            [[1.0, 1.0], [1.0, 1.0], [2.0, 0.0]],
        ], dtype=torch.float32)

        mock_model.return_value = FakeModelOutput(last_hidden_state)

        embedder = ContrieverEmbedder()

        result = embedder.encode(["text 1", "text 2"], batch_size=16)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (2, 2))

    @patch("app.embedder.AutoModel.from_pretrained")
    @patch("app.embedder.AutoTokenizer.from_pretrained")
    def test_encode_applies_mean_pooling_and_masking(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        mock_model.to.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model

        # One text, 3 tokens, hidden dim = 2
        # Third token masked out
        attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.float32)
        input_ids = torch.tensor([[10, 11, 0]], dtype=torch.long)

        tokenized = FakeBatch({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

        mock_tokenizer.return_value = tokenized

        # Only first two tokens should count:
        # mean = ([2,0] + [0,2]) / 2 = [1,1]
        # normalized = [1/sqrt(2), 1/sqrt(2)]
        last_hidden_state = torch.tensor([
            [[2.0, 0.0], [0.0, 2.0], [100.0, 100.0]]
        ], dtype=torch.float32)

        mock_model.return_value = FakeModelOutput(last_hidden_state)

        embedder = ContrieverEmbedder()
        result = embedder.encode(["text 1"])

        expected = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)]], dtype=np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    @patch("app.embedder.AutoModel.from_pretrained")
    @patch("app.embedder.AutoTokenizer.from_pretrained")
    def test_encode_outputs_are_l2_normalized(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        mock_model.to.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model

        attention_mask = torch.tensor([
            [1, 1],
            [1, 1],
        ], dtype=torch.float32)

        input_ids = torch.tensor([
            [1, 2],
            [3, 4],
        ], dtype=torch.long)

        tokenized = FakeBatch({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

        mock_tokenizer.return_value = tokenized

        last_hidden_state = torch.tensor([
            [[3.0, 4.0], [3.0, 4.0]],   # mean = [3,4], norm = 5
            [[5.0, 12.0], [5.0, 12.0]], # mean = [5,12], norm = 13
        ], dtype=torch.float32)

        mock_model.return_value = FakeModelOutput(last_hidden_state)

        embedder = ContrieverEmbedder()
        result = embedder.encode(["a", "b"])

        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), rtol=1e-5, atol=1e-6)

    @patch("app.embedder.AutoModel.from_pretrained")
    @patch("app.embedder.AutoTokenizer.from_pretrained")
    def test_encode_respects_batching(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        mock_model.to.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model

        def tokenizer_side_effect(batch, padding, truncation, max_length, return_tensors):
            batch_size = len(batch)
            input_ids = torch.ones((batch_size, 2), dtype=torch.long)
            attention_mask = torch.ones((batch_size, 2), dtype=torch.float32)
            return FakeBatch({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })

        def model_side_effect(**kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            # hidden dim = 2
            last_hidden_state = torch.ones((batch_size, 2, 2), dtype=torch.float32)
            return FakeModelOutput(last_hidden_state)

        mock_tokenizer.side_effect = tokenizer_side_effect
        mock_model.side_effect = model_side_effect

        embedder = ContrieverEmbedder()
        texts = ["t1", "t2", "t3", "t4", "t5"]

        result = embedder.encode(texts, batch_size=2)

        # 5 texts total, still should return 5 embeddings
        self.assertEqual(result.shape, (5, 2))

        # tokenizer should be called 3 times: batches of size 2, 2, 1
        self.assertEqual(mock_tokenizer.call_count, 3)
        self.assertEqual(mock_model.call_count, 3)

    @patch("app.embedder.AutoModel.from_pretrained")
    @patch("app.embedder.AutoTokenizer.from_pretrained")
    def test_encode_empty_input_returns_empty_array(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        mock_model.to.return_value = mock_model
        mock_model.config.hidden_size = 768
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model

        embedder = ContrieverEmbedder()
        result = embedder.encode([])

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (0, 768))
        mock_tokenizer.assert_not_called()
        mock_model.assert_not_called()

    @patch("app.embedder.AutoModel.from_pretrained")
    @patch("app.embedder.AutoTokenizer.from_pretrained")
    def test_encode_rejects_non_positive_batch_size(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        mock_model.to.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model

        embedder = ContrieverEmbedder()

        with self.assertRaisesRegex(ValueError, "batch_size must be a positive integer"):
            embedder.encode(["text"], batch_size=0)

    @patch("app.embedder.AutoModel.from_pretrained")
    @patch("app.embedder.AutoTokenizer.from_pretrained")
    def test_encode_rejects_non_positive_max_length(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        mock_model.to.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model

        embedder = ContrieverEmbedder()

        with self.assertRaisesRegex(ValueError, "max_length must be a positive integer"):
            embedder.encode(["text"], max_length=0)

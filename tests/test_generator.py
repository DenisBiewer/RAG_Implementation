from __future__ import annotations

import unittest

import torch

from app.generator import QwenGenerator


class FakeBatch(dict):
    def to(self, device):
        self["input_ids"] = self["input_ids"].to(device)
        return self


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ):
        del messages, tokenize, add_generation_prompt, return_tensors, return_dict
        return FakeBatch({"input_ids": torch.tensor([[11, 12, 13]], dtype=torch.long)})

    def decode(self, generated_ids, skip_special_tokens=True):
        del generated_ids, skip_special_tokens
        return "sampled output"


class FakeModel:
    def __init__(self) -> None:
        self.last_generate_kwargs = None

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        appended = torch.tensor([[21, 22]], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, appended], dim=1)


class TestQwenGeneratorSampledGeneration(unittest.TestCase):
    def build_generator(self) -> QwenGenerator:
        generator = object.__new__(QwenGenerator)
        generator.device = "cpu"
        generator.tokenizer = FakeTokenizer()
        generator.model = FakeModel()
        return generator

    def test_generate_sampled_accepts_seed(self):
        generator = self.build_generator()

        output = generator.generate_sampled(
            [{"role": "user", "content": "Hello"}],
            seed=123,
        )

        generation_generator = generator.model.last_generate_kwargs["generator"]
        self.assertEqual(output, "sampled output")
        self.assertIsInstance(generation_generator, torch.Generator)
        self.assertEqual(generation_generator.initial_seed(), 123)

    def test_generate_sampled_prefers_explicit_generator(self):
        generator = self.build_generator()
        explicit_generator = torch.Generator(device="cpu")
        explicit_generator.manual_seed(999)

        generator.generate_sampled(
            [{"role": "user", "content": "Hello"}],
            seed=123,
            generator=explicit_generator,
        )

        self.assertIs(generator.model.last_generate_kwargs["generator"], explicit_generator)

    def test_generate_sampled_preserves_existing_calls_without_seed(self):
        generator = self.build_generator()

        generator.generate_sampled([{"role": "user", "content": "Hello"}])

        self.assertIsNone(generator.model.last_generate_kwargs["generator"])


if __name__ == "__main__":
    unittest.main()

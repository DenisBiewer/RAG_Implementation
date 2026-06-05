from __future__ import annotations

import unittest
from contextlib import contextmanager

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

    def attach_seed_recorder(self, generator: QwenGenerator) -> list[int | None]:
        recorded_seeds: list[int | None] = []

        @contextmanager
        def record_seed(seed):
            recorded_seeds.append(seed)
            yield

        generator._temporary_generation_seed = record_seed
        return recorded_seeds

    def test_generate_sampled_accepts_seed(self):
        generator = self.build_generator()
        recorded_seeds = self.attach_seed_recorder(generator)

        output = generator.generate_sampled(
            [{"role": "user", "content": "Hello"}],
            seed=123,
        )

        self.assertEqual(output, "sampled output")
        self.assertEqual(recorded_seeds, [123])
        self.assertNotIn("generator", generator.model.last_generate_kwargs)

    def test_generate_sampled_uses_generator_seed_when_seed_missing(self):
        generator = self.build_generator()
        recorded_seeds = self.attach_seed_recorder(generator)
        explicit_generator = torch.Generator(device="cpu")
        explicit_generator.manual_seed(999)

        generator.generate_sampled(
            [{"role": "user", "content": "Hello"}],
            generator=explicit_generator,
        )

        self.assertEqual(recorded_seeds, [999])
        self.assertNotIn("generator", generator.model.last_generate_kwargs)

    def test_generate_sampled_prefers_explicit_seed_over_generator_seed(self):
        generator = self.build_generator()
        recorded_seeds = self.attach_seed_recorder(generator)
        explicit_generator = torch.Generator(device="cpu")
        explicit_generator.manual_seed(999)

        generator.generate_sampled(
            [{"role": "user", "content": "Hello"}],
            seed=123,
            generator=explicit_generator,
        )

        self.assertEqual(recorded_seeds, [123])
        self.assertNotIn("generator", generator.model.last_generate_kwargs)

    def test_generate_sampled_preserves_existing_calls_without_seed(self):
        generator = self.build_generator()
        recorded_seeds = self.attach_seed_recorder(generator)

        generator.generate_sampled([{"role": "user", "content": "Hello"}])

        self.assertEqual(recorded_seeds, [None])
        self.assertNotIn("generator", generator.model.last_generate_kwargs)


if __name__ == "__main__":
    unittest.main()

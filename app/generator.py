from __future__ import annotations

import importlib.util
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app.model_profiles import GeneratorProfile


class QwenGenerator:
    """
    Wrapper around Qwen for deterministic baseline generation.

    Practical defaults for this project:
    - prefer CUDA when available
    - use fp16 on CUDA for non-quantized models
    - use fp32 on CPU for non-quantized models
    - allow 4-bit / 8-bit quantized loading for larger experimental models
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        quantization: str = "none",
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.quantization = quantization

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        model_kwargs = self._build_model_kwargs()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if self.quantization == "none":
            self.model.to(self.device)

        self.model.eval()

    @classmethod
    def from_profile(cls, profile: GeneratorProfile) -> QwenGenerator:
        return cls(
            model_name=profile.model_name,
            device=profile.device,
            quantization=profile.quantization,
        )

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _build_model_kwargs(self) -> dict:
        if self.quantization == "none":
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            return {"dtype": dtype}

        if importlib.util.find_spec("bitsandbytes") is None:
            raise ImportError(
                "Quantized loading requires the 'bitsandbytes' package. "
                "Install it or switch the generator profile to quantization='none'."
            )

        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(
                f"Unsupported quantization mode: {self.quantization!r}. "
                "Expected 'none', '4bit', or '8bit'."
            )

        # `device_map="auto"` lets Accelerate place the quantized model correctly.
        return {
            "dtype": "auto",
            "device_map": "auto",
            "quantization_config": quantization_config,
        }

    @contextmanager
    def _temporary_generation_seed(self, seed: int | None):
        """
        Temporarily seed torch RNG state for one sampled generation call.

        Transformers 5.x validates `generate()` kwargs more strictly and
        rejects the `generator=` argument used previously in this project.
        """
        if seed is None:
            yield
            return

        cpu_state = torch.random.get_rng_state()
        cuda_states = None
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()

        try:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            yield
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    @torch.no_grad()
    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate a deterministic answer from the provided chat messages.
        """
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return text.strip()

    @torch.no_grad()
    def generate_sampled(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> str:
        """
        Generate a sampled answer from the provided chat messages.

        This is intended for stages that need output diversity, such as
        candidate PoisonedRAG passage generation. The default deterministic
        `generate` method remains unchanged for the rest of the project.
        """
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        effective_seed = seed
        if effective_seed is None and generator is not None:
            effective_seed = int(generator.initial_seed())

        with self._temporary_generation_seed(effective_seed):
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return text.strip()

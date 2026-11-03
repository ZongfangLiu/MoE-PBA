from __future__ import annotations
from abc import ABC, abstractmethod
import random
import logging
from typing import Any, TYPE_CHECKING


import torch
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict


logger = logging.getLogger(__name__)


class BaseDatasetProcessor(ABC):
    messages_field: str = "messages"

    def __init__(
        self,
        dataset: Dataset | DatasetDict,
        tokenizer: AutoTokenizer,
        pack_samples: bool = True,
        max_input_len: int | None = None,
        split: str | None = None,
        truncate: bool = False,
    ):
        try:
            from datasets import DatasetDict
        except ImportError as exc:
            raise RuntimeError("Calibration dataset processing requires the `datasets` package.") from exc

        if isinstance(dataset, DatasetDict):
            if split is None:
                split = list(dataset.keys())[0]
                logging.warning(
                    f"Using split '{split}' as default for dataset. Available "
                    f"splits: {list(dataset.keys())}",
                )
            dataset = dataset[split]
        if max_input_len is None:
            max_input_len = tokenizer.model_max_length
            logger.warning(
                f"max_input_len is set to {max_input_len} as per tokenizer's "
                f"model_max_length. This will be used for truncation.",
            )
        self.pack_samples = pack_samples
        self.max_input_len = max_input_len
        if self.max_input_len <= 0:
            raise ValueError("max_input_len must be positive.")
        self.dataset = dataset
        self._mapped_dataset = None
        self.tokenizer = tokenizer
        self.truncate = truncate

    @staticmethod
    @abstractmethod
    def _map_fn(sample: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            "This method should be implemented by subclasses.",
        )

    @abstractmethod
    def _encode_sample(self, sample: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError(
            "This method should be implemented by subclasses.",
        )

    def get_processed_dataset(
        self,
        num_samples: int,
        return_loss_mask: bool = False,
        sample_seed: int | None = None,
    ) -> list[torch.Tensor] | list[dict]:
        """Sample calibration sequences from the dataset without replacement."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if self._mapped_dataset is None:
            self._mapped_dataset = self.dataset.map(self._map_fn)
        rng = random.Random(sample_seed) if sample_seed is not None else random
        if self.pack_samples:
            return self._process_samples_packed(
                num_samples,
                return_loss_mask,
                rng,
            )
        else:
            return self._process_samples_unpacked(
                num_samples,
                return_loss_mask,
                rng,
            )

    def _process_samples_unpacked(
        self,
        num_samples: int,
        return_loss_mask: bool = False,
        rng=None,
    ) -> list[torch.Tensor] | list[dict]:
        rng = rng or random
        dataset = self._mapped_dataset
        processed_samples = []
        sampled = []  # sample without replacement
        while len(processed_samples) < num_samples:
            if len(sampled) >= len(dataset):
                logger.warning(
                    "Requested %d calibration sequences, but only %d usable sequences "
                    "were produced after inspecting %d samples.",
                    num_samples, len(processed_samples), len(sampled),
                )
                break
            sample_idx = rng.randint(0, len(dataset) - 1)
            if sample_idx in sampled:
                continue
            sampled.append(sample_idx)
            sample = dataset[sample_idx]
            labels_mask = None
            if return_loss_mask and hasattr(self, "_encode_sample_with_mask"):
                encoded_sample, labels_mask = self._encode_sample_with_mask(sample)
            else:
                encoded_sample = self._encode_sample(sample)
                if return_loss_mask:
                    labels_mask = torch.ones_like(encoded_sample)
            if encoded_sample.shape[-1] == 0:
                continue
            if encoded_sample.shape[-1] > self.max_input_len:
                if self.truncate:
                    encoded_sample = encoded_sample[:, : self.max_input_len]
                    if labels_mask is not None:
                        labels_mask = labels_mask[:, : self.max_input_len]
                else:
                    continue

            if return_loss_mask:
                encoded_sample = {
                    "input_ids": encoded_sample,
                    "attention_mask": torch.ones_like(encoded_sample),
                    "labels_mask": labels_mask if labels_mask is not None else torch.ones_like(encoded_sample),
                }
            processed_samples.append(encoded_sample)
        return processed_samples

    def _process_samples_packed(
        self,
        num_samples: int,
        return_loss_mask: bool = False,
        rng=None,
    ) -> list[torch.Tensor] | list[dict]:
        rng = rng or random
        dataset = self._mapped_dataset
        processed_samples = []
        sampled = []
        while len(processed_samples) < num_samples:
            if len(sampled) >= len(dataset):
                logger.warning(
                    "Requested %d calibration sequences, but only %d were produced "
                    "after packing all %d samples.",
                    num_samples, len(processed_samples), len(sampled),
                )
                break
            # Token ID 0 separates packed samples and remains visible in the attention mask.
            seq = torch.zeros((1, self.max_input_len), dtype=torch.long)
            labels_seq = torch.zeros_like(seq) if return_loss_mask else None
            seq_idx = 0
            while seq_idx < self.max_input_len:
                if len(sampled) >= len(dataset):
                    logger.warning(
                        "Not enough samples to pack the last sequence to max_input_len."
                    )
                    break
                sample_idx = rng.randint(0, len(dataset) - 1)
                if sample_idx in sampled:
                    continue
                sampled.append(sample_idx)
                sample = dataset[sample_idx]
                labels_mask = None
                if return_loss_mask and hasattr(self, "_encode_sample_with_mask"):
                    encoded_sample, labels_mask = self._encode_sample_with_mask(sample)
                else:
                    encoded_sample = self._encode_sample(sample)  # shape (batch, seq)
                    if return_loss_mask:
                        labels_mask = torch.ones_like(encoded_sample)
                if encoded_sample.shape[-1] == 0:
                    continue
                end_seq = seq_idx + encoded_sample.shape[-1]
                if end_seq > self.max_input_len:
                    encoded_sample = encoded_sample[:, : (self.max_input_len - seq_idx)]
                    if labels_mask is not None:
                        labels_mask = labels_mask[:, : (self.max_input_len - seq_idx)]
                    end_seq = self.max_input_len
                seq[:, seq_idx:end_seq] = encoded_sample
                if labels_mask is not None and labels_seq is not None:
                    labels_seq[:, seq_idx:end_seq] = labels_mask
                seq_idx = end_seq + 1
            valid_length = min(seq_idx, self.max_input_len)
            if valid_length == 0:
                continue
            if return_loss_mask:
                attention_mask = torch.zeros_like(seq)
                attention_mask[:, :valid_length] = 1
                encoded_sample = {
                    "input_ids": seq,
                    "attention_mask": attention_mask,
                    "labels_mask": labels_seq if labels_seq is not None else torch.ones_like(seq),
                }
            else:
                encoded_sample = seq[:, :valid_length]
            processed_samples.append(encoded_sample)
        return processed_samples


class ChatDatasetProcessor(BaseDatasetProcessor):
    def _encode_sample(self, sample: dict[str, Any]) -> torch.Tensor:
        chat_sample = self.tokenizer.apply_chat_template(
            sample[self.messages_field],
            add_generation_prompt=False,
            tokenize=False,
        )
        return self.tokenizer(
            chat_sample,
            truncation=self.truncate,
            max_length=self.max_input_len if self.truncate else None,
            return_tensors="pt",
        )["input_ids"]

    def _encode_sample_with_mask(self, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tokenized chat sample plus a mask over assistant tokens for loss."""
        messages = sample[self.messages_field]
        chat_sample = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        tokenized = self.tokenizer(
            chat_sample,
            truncation=self.truncate,
            max_length=self.max_input_len if self.truncate else None,
            return_tensors="pt",
        )["input_ids"]
        total_len = tokenized.shape[-1]
        loss_mask = torch.zeros_like(tokenized)

        prev_len = 0
        for idx, message in enumerate(messages):
            partial = self.tokenizer.apply_chat_template(
                messages[: idx + 1],
                add_generation_prompt=False,
                tokenize=False,
            )
            partial_ids = self.tokenizer(
                partial,
                truncation=self.truncate,
                max_length=self.max_input_len if self.truncate else None,
                return_tensors="pt",
            )["input_ids"]
            curr_len = partial_ids.shape[-1]
            start = min(prev_len, total_len)
            end = min(curr_len, total_len)
            if message.get("role") == "assistant":
                loss_mask[:, start:end] = 1
            prev_len = curr_len
        return tokenized, loss_mask


class PersonasMathChatDataset(ChatDatasetProcessor):
    @staticmethod
    def _map_fn(sample: dict[str, Any]) -> dict[str, Any]:
        return sample


class CodeAlpacaChatDataset(ChatDatasetProcessor):
    @staticmethod
    def _map_fn(sample: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["output"]},
            ],
        }


DATASET_REGISTRY: dict[str, type[BaseDatasetProcessor]] = {
    "theblackcat102/evol-codealpaca-v1": CodeAlpacaChatDataset,
    "allenai/tulu-3-sft-personas-math": PersonasMathChatDataset,
}

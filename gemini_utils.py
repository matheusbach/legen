from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pysrt

import gemini_srt_translator as gst


@dataclass(frozen=True)
class GeminiTranslationConfig:
    api_keys: Sequence[str]
    input_file: Path
    output_file: Path
    target_language: str
    batch_size: int = 400
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 50
    free_quota: bool = True
    resume: bool = False
    thinking: bool = False
    progress_log: bool = False
    thoughts_log: bool = False


class MultiKeyGeminiTranslator(gst.GeminiSRTTranslator):
    """Gemini translator that rotates across an arbitrary number of API keys."""

    def __init__(self, api_keys: Sequence[str], **kwargs) -> None:
        cleaned: List[str] = [key.strip() for key in api_keys if key and key.strip()]
        if not cleaned:
            raise ValueError("At least one Gemini API key is required.")

        primary = cleaned[0]
        secondary = cleaned[1] if len(cleaned) > 1 else None

        super().__init__(gemini_api_key=primary, gemini_api_key2=secondary, **kwargs)

        self._api_keys = cleaned
        self._api_index = 0
        self.current_api_key = primary
        self.current_api_number = 1
        self.backup_api_number = 2 if len(cleaned) > 1 else 1

    def _switch_api(self) -> bool:  # type: ignore[override]
        if len(self._api_keys) <= 1:
            return False

        previous_number = self.current_api_number
        total_keys = len(self._api_keys)

        for step in range(1, total_keys + 1):
            next_index = (self._api_index + step) % total_keys
            if next_index == self._api_index:
                continue

            next_key = self._api_keys[next_index]
            if next_key:
                self._api_index = next_index
                self.current_api_key = next_key
                self.current_api_number = next_index + 1
                self.backup_api_number = previous_number
                return True

        return False


def translate_with_gemini(config: GeminiTranslationConfig) -> pysrt.SubRipFile:
    translator = MultiKeyGeminiTranslator(
        api_keys=config.api_keys,
        target_language=config.target_language,
        input_file=str(config.input_file),
        output_file=str(config.output_file),
        batch_size=config.batch_size,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        free_quota=config.free_quota,
        resume=config.resume,
        thinking=config.thinking,
        progress_log=config.progress_log,
        thoughts_log=config.thoughts_log,
    )

    translator.translate()

    return pysrt.open(config.output_file, encoding="utf-8")


def normalize_api_keys(keys: Iterable[str] | str | None) -> List[str]:
    if keys is None:
        return []

    if isinstance(keys, str):
        raw = [keys]
    else:
        raw = list(keys)

    candidates: List[str] = []
    for value in raw:
        if not value:
            continue
        parts = [part.strip() for part in str(value).replace("\n", ",").split(",")]
        candidates.extend(part for part in parts if part)

    unique: List[str] = []
    seen = set()
    for key in candidates:
        if key not in seen:
            unique.append(key)
            seen.add(key)

    return unique

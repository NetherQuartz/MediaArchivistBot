import json
from types import SimpleNamespace

import pytest

from archivistbot.config import Settings
from archivistbot.llm_api import (
    SYSTEM_PROMPT,
    EmbeddingService,
    MediaDescription,
    _parse_media_description,
    build_search_text,
)


class FakeOllama:
    def __init__(self) -> None:
        self.inputs: list[str] | None = None

    async def embed(self, *, model, input, truncate, dimensions):
        self.inputs = input
        return SimpleNamespace(embeddings=[[1.0, 0.0, 0.0] for _ in input])


@pytest.mark.asyncio
async def test_query_uses_qwen_instruction() -> None:
    service = EmbeddingService(
        Settings(
            embedding_dimensions=3,
            embedding_model="test-model",
            ollama_host="http://example.invalid",
        )
    )
    fake = FakeOllama()
    service.client = fake

    result = await service.embed_query("  capybara on a person's head  ")

    assert result == [1.0, 0.0, 0.0]
    assert fake.inputs is not None
    assert fake.inputs[0].startswith("Instruct:")
    assert fake.inputs[0].endswith("Query: capybara on a person's head")


def test_search_text_contains_exact_ocr_caption_and_transcript() -> None:
    description = MediaDescription(
        summary="Stilgar looks on in reverence",
        visible_text=["As it was written"],
        people_and_characters=["Stilgar", "Javier Bardem"],
        meme_context=["Dune", "a prophecy being fulfilled"],
        controversial_context=["political satire about an election"],
    )

    text = build_search_text(
        description,
        caption="the best political forecast",
        transcript="Lisan al-Gaib",
    )

    assert "As it was written" in text
    assert "the best political forecast" in text
    assert "Lisan al-Gaib" in text
    assert "political satire about an election" in text


def test_prompt_requires_bilingual_search_aliases() -> None:
    normalized_prompt = " ".join(SYSTEM_PROMPT.split())
    assert "both English and Russian" in normalized_prompt
    assert "visible_text verbatim" in normalized_prompt


@pytest.mark.parametrize(
    "wrapper",
    [
        lambda payload: payload,
        lambda payload: f"```json\n{payload}\n```",
        lambda payload: f"```\n{payload}\n```",
        lambda payload: f"Here is the requested description:\n```json\n{payload}\n```",
    ],
)
def test_media_description_accepts_plain_or_fenced_json(wrapper) -> None:
    payload = json.dumps({"summary": "A red square"})

    description = _parse_media_description(wrapper(payload))

    assert description.summary == "A red square"

import asyncio
import base64
from collections.abc import Sequence
from dataclasses import dataclass

from ollama import AsyncClient as AsyncOllama
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field

from .config import Settings, get_settings

PROMPT_VERSION = "search-facets-v4"

SYSTEM_PROMPT = """
You create an accurate search index for a collection of memes, images, and
short videos. Write descriptions in English, while preserving the original
spelling of visible text, names, slang, and quotations.

This is neutral archival classification, not endorsement. Do not refuse to
describe media solely because it contains profanity, insults, slurs, dark
humor, political propaganda, extremist symbols, or religious or sexual jokes.
Record those elements accurately and without euphemisms when they are actually
present or necessary for retrieval. Do not add moral commentary or invent
offensive wording that is absent from the source.

Describe only what can be observed. Do not infer a real person's ethnicity,
religion, sexual orientation, medical condition, or political beliefs from
appearance. Name public figures, fictional characters, and known meme
templates only when recognizable; put doubts in uncertainties. Explain the
joke, irony, and context, but keep them separate from factual observations.
In aliases_and_search_phrases, include common search terms and synonyms in both
English and Russian, even when the source media uses only one of those
languages. Keep visible_text verbatim in its original language.
Do not use Markdown and follow the JSON schema exactly.
""".strip()


class MediaDescription(BaseModel):
    summary: str
    visible_text: list[str] = Field(default_factory=list)
    people_and_characters: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    setting: list[str] = Field(default_factory=list)
    emotions: list[str] = Field(default_factory=list)
    meme_context: list[str] = Field(default_factory=list)
    controversial_context: list[str] = Field(default_factory=list)
    aliases_and_search_phrases: list[str] = Field(
        default_factory=list,
        description=(
            "Likely retrieval phrases and synonyms in both English and Russian."
        ),
    )
    uncertainties: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ImageInput:
    data: bytes
    mime_type: str = "image/jpeg"


def _parse_media_description(output: str) -> MediaDescription:
    cleaned = output.strip()
    fence_start = cleaned.find("```")
    if fence_start != -1:
        content_start = cleaned.find("\n", fence_start + 3)
        fence_end = cleaned.find("```", content_start + 1)
        if content_start != -1 and fence_end != -1:
            fence_label = cleaned[fence_start + 3 : content_start].strip().lower()
            if fence_label in {"", "json"}:
                cleaned = cleaned[content_start + 1 : fence_end].strip()
    return MediaDescription.model_validate_json(cleaned)


class VisionService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = AsyncOpenAI(
            api_key=self.settings.vision_api_key.get_secret_value(),
            base_url=self.settings.vision_base_url,
            timeout=self.settings.vision_timeout_seconds,
            max_retries=0,
        )

    async def describe(
        self,
        images: Sequence[ImageInput],
        *,
        media_type: str,
        caption: str | None = None,
        transcript: str | None = None,
    ) -> MediaDescription:
        if not images:
            raise ValueError("At least one image is required for vision analysis")

        context = [
            f"Media type: {media_type}.",
            (
                "Extract attributes that let a user retrieve this media with a "
                "short, conversational query in any language."
            ),
        ]
        if caption:
            context.append(f"Telegram caption: {caption}")
        if transcript:
            context.append(f"Automatic audio transcript: {transcript}")

        content: list[dict[str, object]] = [
            {"type": "text", "text": "\n".join(context)}
        ]
        content.extend(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{image.mime_type};base64,"
                        f"{base64.b64encode(image.data).decode('ascii')}"
                    )
                },
            }
            for image in images
        )

        schema = MediaDescription.model_json_schema()
        schema["required"] = list(schema["properties"])
        schema["additionalProperties"] = False
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "media_description",
                "strict": True,
                "schema": schema,
            },
        }

        for attempt in range(self.settings.vision_max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.settings.vision_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    response_format=response_format,
                    temperature=0.1,
                )
                output = response.choices[0].message.content
                if not output:
                    raise ValueError("Vision provider returned an empty response")
                return _parse_media_description(output)
            except (APIConnectionError, APITimeoutError, RateLimitError):
                if attempt + 1 >= self.settings.vision_max_retries:
                    raise
                await asyncio.sleep(2**attempt)
            except APIStatusError as error:
                if error.status_code < 500 or attempt + 1 >= self.settings.vision_max_retries:
                    raise
                await asyncio.sleep(2**attempt)

        raise RuntimeError("Vision request retry loop finished unexpectedly")


class EmbeddingService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = AsyncOllama(host=self.settings.ollama_host)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(list(texts))

    async def embed_query(self, query: str) -> list[float]:
        instruction = (
            "Instruct: Given a user's natural-language request, retrieve the most "
            "relevant meme, image, or video description.\n"
            f"Query: {query.strip()}"
        )
        return (await self._embed([instruction]))[0]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be non-empty strings")
        response = await self.client.embed(
            model=self.settings.embedding_model,
            input=texts,
            truncate=True,
            dimensions=self.settings.embedding_dimensions,
        )
        embeddings = [list(vector) for vector in response.embeddings]
        for vector in embeddings:
            if len(vector) != self.settings.embedding_dimensions:
                raise ValueError(
                    "Unexpected embedding size: "
                    f"{len(vector)} != {self.settings.embedding_dimensions}"
                )
        return embeddings


def build_search_text(
    description: MediaDescription,
    *,
    caption: str | None = None,
    transcript: str | None = None,
) -> str:
    sections: list[tuple[str, Sequence[str] | str]] = [
        ("Description", description.summary),
        ("Visible text", description.visible_text),
        ("People and characters", description.people_and_characters),
        ("Objects", description.objects),
        ("Actions", description.actions),
        ("Setting", description.setting),
        ("Emotions", description.emotions),
        ("Meme context", description.meme_context),
        ("Controversial context", description.controversial_context),
        ("Search phrases", description.aliases_and_search_phrases),
    ]
    if caption:
        sections.append(("Caption", caption))
    if transcript:
        sections.append(("Speech", transcript))

    lines: list[str] = []
    for label, value in sections:
        if isinstance(value, str):
            cleaned = value.strip()
        else:
            cleaned = "; ".join(item.strip() for item in value if item.strip())
        if cleaned:
            lines.append(f"{label}: {cleaned}")
    return "\n".join(lines)

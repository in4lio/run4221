from __future__ import annotations

import logging

from run4221.ai.extraction_provider import ExtractorProvider
from run4221.ai.openai_provider import OpenAIExtractorProvider
from run4221.config import Settings
from run4221.db.prompts import (
    DISCOVER_EVENT_PROFILE_PROMPT,
    PromptConfigError,
    get_file_prompt,
    get_runtime_prompt,
)

logger = logging.getLogger(__name__)


class ExtractorProviderConfigError(RuntimeError):
    pass


def get_extractor_provider(settings: Settings) -> ExtractorProvider | None:
    if settings.ai_extractor_provider == "heuristic":
        return None

    if settings.ai_extractor_provider == "openai":
        if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
            raise ExtractorProviderConfigError(
                "AI_EXTRACTOR_PROVIDER=openai requires OPENAI_API_KEY."
            )
        try:
            if settings.run4221_prompt_source == "file":
                prompt = get_file_prompt(
                    DISCOVER_EVENT_PROFILE_PROMPT,
                    prompts_dir=settings.run4221_prompt_dir,
                )
            else:
                prompt = get_runtime_prompt(
                    DISCOVER_EVENT_PROFILE_PROMPT,
                    database_url=settings.database_url,
                )
        except PromptConfigError as error:
            raise ExtractorProviderConfigError(str(error)) from error

        if prompt.fallback_reason:
            logger.warning(
                "Using fallback %s prompt %s v%s: %s",
                prompt.source,
                prompt.prompt_key,
                prompt.version,
                prompt.fallback_reason,
            )
        else:
            logger.info(
                "Using %s prompt %s%s",
                prompt.source,
                prompt.prompt_key,
                f" from {prompt.file_path}" if prompt.file_path else f" v{prompt.version}",
            )

        return OpenAIExtractorProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_extract_model,
            instructions=prompt.content,
        )

    raise ExtractorProviderConfigError(
        f"Unknown AI_EXTRACTOR_PROVIDER: {settings.ai_extractor_provider}"
    )

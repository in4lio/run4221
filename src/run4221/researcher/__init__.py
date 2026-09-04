"""Contracts and configuration for the standalone event researcher.

This package intentionally exposes no Telegram, moderation, or persistence tools.
"""

from run4221.researcher.config import ResearcherSettings
from run4221.researcher.schemas import (
    ArtifactReference,
    ResearchBudget,
    ResearchCandidate,
    ResearchDecision,
    ResearchRunStatus,
)

__all__ = (
    "ArtifactReference",
    "ResearchBudget",
    "ResearchCandidate",
    "ResearchDecision",
    "ResearchRunStatus",
    "ResearcherSettings",
)

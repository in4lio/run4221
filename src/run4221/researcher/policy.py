from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from run4221.ingestion.page_snapshot import PageSnapshot
from run4221.researcher.schemas import validate_http_url


def source_domain(url: str) -> str:
    validate_http_url(url)
    hostname = urlsplit(url).hostname
    assert hostname is not None
    return hostname.rstrip(".").casefold()


def optional_source_domain(url: str) -> str | None:
    try:
        return source_domain(url)
    except ValueError:
        return None


def domain_is_trusted(hostname: str, trusted_domains: frozenset[str]) -> bool:
    return any(
        hostname == trusted or hostname.endswith(f".{trusted}")
        for trusted in trusted_domains
    )


@dataclass(frozen=True)
class SourceTrustDecision:
    trusted: bool
    reason: str
    registry_snapshot: PageSnapshot | None = None


@dataclass(frozen=True)
class SourceTrustPolicy:
    """Closed deterministic source policy; page claims never grant trust."""

    trusted_domains: frozenset[str] = frozenset()
    trusted_registry_urls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_domains = frozenset(
            domain.strip().rstrip(".").casefold()
            for domain in self.trusted_domains
            if domain.strip()
        )
        if any("/" in domain or ":" in domain for domain in normalized_domains):
            raise ValueError("Trusted domains must be hostnames, not URLs.")
        normalized_registry_urls = tuple(
            dict.fromkeys(validate_http_url(url.strip()) for url in self.trusted_registry_urls)
        )
        for url in normalized_registry_urls:
            if not domain_is_trusted(source_domain(url), normalized_domains):
                raise ValueError("Every trusted registry URL must use a trusted domain.")
        object.__setattr__(self, "trusted_domains", normalized_domains)
        object.__setattr__(self, "trusted_registry_urls", normalized_registry_urls)

    def evaluate(
        self,
        candidate: PageSnapshot,
        *,
        registry_snapshots: tuple[PageSnapshot, ...] = (),
    ) -> SourceTrustDecision:
        candidate_domain = source_domain(candidate.final_url)
        if domain_is_trusted(candidate_domain, self.trusted_domains):
            return SourceTrustDecision(True, "configured trusted domain")

        configured_registries = set(self.trusted_registry_urls)
        for registry in registry_snapshots:
            if registry.source_url not in configured_registries:
                continue
            if not domain_is_trusted(
                source_domain(registry.final_url), self.trusted_domains
            ):
                continue
            linked_domains = {
                domain
                for link in registry.links
                if (domain := optional_source_domain(link.url)) is not None
            }
            if candidate_domain in linked_domains:
                return SourceTrustDecision(
                    True,
                    "captured trusted registry link",
                    registry_snapshot=registry,
                )

        return SourceTrustDecision(False, "source domain is not trusted")

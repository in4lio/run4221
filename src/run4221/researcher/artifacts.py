from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ValidationError

from run4221.ingestion.page_snapshot import PageSnapshot
from run4221.researcher.schemas import (
    ArtifactReference,
    DecisionAction,
    ResearchDecision,
    ResearchRunStatus,
    RunOutcome,
    RunState,
    validate_http_url,
)

DEFAULT_MAX_TEXT_CHARS = 50_000
DEFAULT_MAX_COLLECTION_ITEMS = 100
DEFAULT_MAX_DEPTH = 8
DEFAULT_MAX_JSON_BYTES = 256_000
MAX_ARTIFACT_TYPE_CHARS = 48
MAX_METADATA_KEY_CHARS = 120
MAX_QUEUE_REFERENCE_CHARS = 240
# Legacy prepared.json files carry no producer wall cap; assume the largest
# cap any job may configure so reconciliation never races an in-flight run.
LEGACY_PRODUCER_DEADLINE_SECONDS = 900.0
TRUNCATION_MARKER = "...[truncated]"
_ARTIFACT_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,47}$")


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ArtifactError(RuntimeError):
    """Base error for the append-only artifact store."""


class ArtifactExistsError(ArtifactError):
    """Raised when a singleton artifact would be overwritten."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when stored bytes no longer match their reference."""


class ArtifactLimitError(ArtifactError):
    """Raised when bounded artifact input exceeds a non-truncatable limit."""


class ArtifactLifecycleError(ArtifactError):
    """Raised when a run attempts an invalid lifecycle transition."""


class QueueResolutionState(StrEnum):
    COMMITTED = "committed"
    ABSENT = "absent"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class QueueResolution:
    """The only queue truth reconciliation may receive from application code."""

    state: QueueResolutionState
    queue_reference: str | None = None
    detail: str | None = None

    @classmethod
    def committed(cls, queue_reference: str) -> QueueResolution:
        return cls(QueueResolutionState.COMMITTED, queue_reference=queue_reference)

    @classmethod
    def absent(cls) -> QueueResolution:
        return cls(QueueResolutionState.ABSENT)

    @classmethod
    def inconclusive(cls, detail: str | None = None) -> QueueResolution:
        return cls(QueueResolutionState.INCONCLUSIVE, detail=detail)


@dataclass(frozen=True)
class PreparedDecision:
    reference: ArtifactReference
    decision: ResearchDecision
    committed_status: ResearchRunStatus


@dataclass(frozen=True)
class ReconciliationResult:
    run_id: str
    state: QueueResolutionState
    terminal_reference: ArtifactReference | None = None
    reconciliation_reference: ArtifactReference | None = None


type PreparedResolver = Callable[[PreparedDecision], QueueResolution]


class ResearchArtifactStore:
    """Bounded append-only storage for researcher run evidence and decisions.

    Queue persistence stays outside this module. A caller first prepares a decision,
    commits its own queue transaction, and only then reports that commit here. Startup
    reconciliation can ask a narrow resolver whether that exact prepared reference is
    committed, absent, or currently unknowable.
    """

    def __init__(
        self,
        root: str | Path = "data/research_runs",
        *,
        now: Callable[[], datetime] | None = None,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
        max_collection_items: int = DEFAULT_MAX_COLLECTION_ITEMS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
    ) -> None:
        if min(max_text_chars, max_collection_items, max_depth, max_json_bytes) < 1:
            raise ValueError("Artifact limits must be positive.")
        self.root = Path(root)
        self._now = now or (lambda: datetime.now(UTC))
        self.max_text_chars = max_text_chars
        self.max_collection_items = max_collection_items
        self.max_depth = max_depth
        self.max_json_bytes = max_json_bytes

    def create_run(
        self,
        *,
        job_type: str,
        metadata: Mapping[str, object] | None = None,
    ) -> str:
        run_id = str(uuid4())
        run_dir = self.root / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(mode=0o700)
        try:
            self._write_named_artifact(
                run_id,
                artifact_name="run.json",
                artifact_type="run",
                source_url=None,
                content={"job_type": job_type, "metadata": dict(metadata or {})},
            )
        except Exception:
            run_dir.rmdir()
            raise
        return run_id

    def write_artifact(
        self,
        run_id: str,
        *,
        artifact_type: str,
        source_url: str,
        content: Mapping[str, object],
    ) -> ArtifactReference:
        self._validate_artifact_type(artifact_type)
        artifact_name = f"{artifact_type}-{uuid4()}.json"
        return self._write_named_artifact(
            run_id,
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            source_url=source_url,
            content=content,
        )

    def write_page_snapshot(
        self,
        run_id: str,
        snapshot: PageSnapshot,
    ) -> ArtifactReference:
        return self.write_artifact(
            run_id,
            artifact_type="page_snapshot",
            source_url=snapshot.source_url,
            content=snapshot.to_dict(),
        )

    def prepare_decision(
        self,
        run_id: str,
        *,
        source_url: str,
        decision: ResearchDecision,
        committed_status: ResearchRunStatus,
        producer_deadline_seconds: float | None = None,
    ) -> ArtifactReference:
        self._require_run(run_id)
        if (self._run_dir(run_id) / "terminal.json").exists():
            raise ArtifactLifecycleError("A terminal run cannot prepare another decision.")
        self._validate_prepared_status(decision, committed_status)
        if producer_deadline_seconds is not None and (
            not isinstance(producer_deadline_seconds, int | float)
            or isinstance(producer_deadline_seconds, bool)
            or not math.isfinite(producer_deadline_seconds)
            or producer_deadline_seconds <= 0
        ):
            raise ArtifactLimitError("Producer deadline must be a positive finite number.")
        for evidence in decision.evidence:
            if evidence.run_id != run_id:
                raise ArtifactLifecycleError("Prepared evidence must belong to the same run.")
            self.verify_artifact(evidence)
        content: dict[str, object] = {
            "decision": decision.model_dump(mode="json"),
            "committed_status": committed_status.model_dump(mode="json"),
        }
        if producer_deadline_seconds is not None:
            content["producer_deadline_seconds"] = float(producer_deadline_seconds)
        return self._write_named_artifact(
            run_id,
            artifact_name="prepared.json",
            artifact_type="prepared_decision",
            source_url=source_url,
            content=content,
        )

    def finalize_committed(
        self,
        prepared_reference: ArtifactReference,
        *,
        queue_reference: str,
    ) -> ArtifactReference:
        prepared = self._read_prepared(prepared_reference)
        queue_reference = self._bounded_queue_reference(queue_reference)
        return self._write_terminal(
            prepared,
            queue_state=QueueResolutionState.COMMITTED,
            status=prepared.committed_status,
            queue_reference=queue_reference,
        )

    def finalize_uncommitted(
        self,
        prepared_reference: ArtifactReference,
        *,
        status: ResearchRunStatus,
    ) -> ArtifactReference:
        """Record a truthful terminal after a queue admission rejects the attempt."""

        prepared = self._read_prepared(prepared_reference)
        if status.outcome is RunOutcome.PROPOSAL_CREATED:
            raise ArtifactLifecycleError(
                "An uncommitted decision cannot report a queue-created outcome."
            )
        return self._write_terminal(
            prepared,
            queue_state=QueueResolutionState.ABSENT,
            status=status,
        )

    def finalize_without_queue(
        self,
        run_id: str,
        *,
        source_url: str,
        status: ResearchRunStatus,
    ) -> ArtifactReference:
        run_dir = self._require_run(run_id)
        if (run_dir / "prepared.json").exists():
            raise ArtifactLifecycleError(
                "A prepared queue decision must be finalized through commit reporting."
            )
        if status.outcome is RunOutcome.PROPOSAL_CREATED:
            raise ArtifactLifecycleError(
                "Queue-created outcomes require a prepared decision and commit report."
            )
        return self._write_named_artifact(
            run_id,
            artifact_name="terminal.json",
            artifact_type="terminal",
            source_url=source_url,
            content={
                "queue_state": None,
                "queue_reference": None,
                "prepared_artifact": None,
                "status": status.model_dump(mode="json"),
            },
        )

    def reconcile_prepared(
        self,
        resolver: PreparedResolver,
    ) -> tuple[ReconciliationResult, ...]:
        if not self.root.exists():
            return ()
        results: list[ReconciliationResult] = []
        for run_dir in sorted(self.root.iterdir(), key=lambda path: path.name):
            if not run_dir.is_dir() or run_dir.is_symlink():
                continue
            try:
                run_id = str(UUID(run_dir.name))
            except ValueError:
                continue
            if run_id != run_dir.name or (run_dir / "terminal.json").exists():
                continue
            prepared_path = run_dir / "prepared.json"
            if not prepared_path.is_file() or prepared_path.is_symlink():
                continue
            prepared_reference = self._reference_from_path(prepared_path)
            if self._prepared_may_still_be_running(prepared_reference):
                continue
            try:
                prepared = self._read_prepared(prepared_reference)
            except (ArtifactIntegrityError, ValidationError):
                # A legacy prepared.json whose decision no longer validates
                # (e.g. a deleted action) must not poison reconciliation for
                # every other run: park it behind a truthful failed terminal
                # and keep processing.
                terminal = self._write_incompatible_prepared_terminal(prepared_reference)
                results.append(
                    ReconciliationResult(
                        run_id,
                        QueueResolutionState.INCONCLUSIVE,
                        terminal_reference=terminal,
                    )
                )
                continue
            try:
                resolution = resolver(prepared)
            except Exception as error:
                resolution = QueueResolution.inconclusive(
                    f"Resolver failed with {type(error).__name__}."
                )
            self._validate_resolution(resolution)
            if resolution.state is QueueResolutionState.COMMITTED:
                terminal = self.finalize_committed(
                    prepared.reference,
                    queue_reference=resolution.queue_reference or "",
                )
                results.append(
                    ReconciliationResult(run_id, resolution.state, terminal_reference=terminal)
                )
                continue
            if resolution.state is QueueResolutionState.ABSENT:
                terminal = self._write_terminal(
                    prepared,
                    queue_state=QueueResolutionState.ABSENT,
                    status=ResearchRunStatus(
                        status=RunState.FAILED,
                        outcome=RunOutcome.INCONCLUSIVE,
                        detail="Prepared decision was not found in a moderation queue.",
                    ),
                )
                results.append(
                    ReconciliationResult(run_id, resolution.state, terminal_reference=terminal)
                )
                continue
            reconciliation = self._write_inconclusive_once(
                prepared,
                resolution.detail or "Queue state could not be determined.",
            )
            results.append(
                ReconciliationResult(
                    run_id,
                    resolution.state,
                    reconciliation_reference=reconciliation,
                )
            )
        return tuple(results)

    def read_artifact(self, reference: ArtifactReference) -> dict[str, Any]:
        path = self._path_for_reference(reference)
        raw = path.read_bytes()
        self._verify_bytes(reference, raw)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ArtifactIntegrityError("Artifact is not valid JSON.") from error
        if not isinstance(payload, dict):
            raise ArtifactIntegrityError("Artifact root must be a JSON object.")
        if payload.get("run_id") != reference.run_id:
            raise ArtifactIntegrityError("Artifact run ID does not match its reference.")
        if payload.get("source_url") != reference.source_url:
            raise ArtifactIntegrityError("Artifact source URL does not match its reference.")
        if reference.artifact_name != "run.json":
            manifest_hash = payload.get("run_manifest_hash")
            current_manifest_hash = sha256(
                (path.parent / "run.json").read_bytes()
            ).hexdigest()
            if manifest_hash != current_manifest_hash:
                raise ArtifactIntegrityError("Artifact run manifest hash does not match.")
        return payload

    def hash_artifact(self, reference: ArtifactReference) -> str:
        path = self._path_for_reference(reference)
        return sha256(path.read_bytes()).hexdigest()

    def verify_artifact(self, reference: ArtifactReference) -> None:
        self.read_artifact(reference)

    def _write_terminal(
        self,
        prepared: PreparedDecision,
        *,
        queue_state: QueueResolutionState,
        status: ResearchRunStatus,
        queue_reference: str | None = None,
    ) -> ArtifactReference:
        return self._write_named_artifact(
            prepared.reference.run_id,
            artifact_name="terminal.json",
            artifact_type="terminal",
            source_url=prepared.reference.source_url,
            content={
                "queue_state": queue_state.value,
                "queue_reference": queue_reference,
                "prepared_artifact": prepared.reference.model_dump(mode="json"),
                "status": status.model_dump(mode="json"),
            },
        )

    def _write_inconclusive_once(
        self,
        prepared: PreparedDecision,
        detail: str,
    ) -> ArtifactReference:
        path = self._run_dir(prepared.reference.run_id) / "reconciliation-inconclusive.json"
        try:
            return self._write_named_artifact(
                prepared.reference.run_id,
                artifact_name=path.name,
                artifact_type="reconciliation",
                source_url=prepared.reference.source_url,
                content={
                    "queue_state": QueueResolutionState.INCONCLUSIVE.value,
                    "detail": detail,
                    "prepared_artifact": prepared.reference.model_dump(mode="json"),
                },
            )
        except ArtifactExistsError:
            return self._reference_from_path(path)

    def _write_incompatible_prepared_terminal(
        self,
        reference: ArtifactReference,
    ) -> ArtifactReference:
        """Terminate a run whose prepared decision no longer parses (append-only)."""

        status = ResearchRunStatus(
            status=RunState.FAILED,
            outcome=RunOutcome.INCONCLUSIVE,
            detail="Prepared decision is incompatible with the current schema.",
        )
        return self._write_named_artifact(
            reference.run_id,
            artifact_name="terminal.json",
            artifact_type="terminal",
            source_url=reference.source_url,
            content={
                "queue_state": QueueResolutionState.INCONCLUSIVE.value,
                "queue_reference": None,
                "prepared_artifact": reference.model_dump(mode="json"),
                "status": status.model_dump(mode="json"),
            },
        )

    def _prepared_may_still_be_running(self, reference: ArtifactReference) -> bool:
        """Skip prepared runs younger than twice their producer-stamped wall cap."""

        payload = self.read_artifact(reference)
        content = payload.get("content")
        cap = (
            content.get("producer_deadline_seconds")
            if isinstance(content, Mapping)
            else None
        )
        if (
            not isinstance(cap, int | float)
            or isinstance(cap, bool)
            or not math.isfinite(cap)
            or cap <= 0
        ):
            cap = LEGACY_PRODUCER_DEADLINE_SECONDS
        try:
            created_at = datetime.fromisoformat(str(payload.get("created_at")))
        except ValueError:
            # An unreadable creation time cannot prove the producer finished:
            # treat the run as possibly still in flight and skip it this cycle.
            return True
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        return self._utc_now() - created_at < timedelta(seconds=2 * float(cap))

    def _read_prepared(self, reference: ArtifactReference) -> PreparedDecision:
        if reference.artifact_name != "prepared.json":
            raise ArtifactLifecycleError("Commit finalization requires prepared.json.")
        payload = self.read_artifact(reference)
        if payload.get("artifact_type") != "prepared_decision":
            raise ArtifactIntegrityError("Prepared reference has the wrong artifact type.")
        content = payload.get("content")
        if not isinstance(content, dict):
            raise ArtifactIntegrityError("Prepared artifact content must be an object.")
        try:
            decision = ResearchDecision.model_validate(content.get("decision"))
            status = ResearchRunStatus.model_validate(content.get("committed_status"))
        except ValueError as error:
            raise ArtifactIntegrityError("Prepared artifact has an invalid payload.") from error
        return PreparedDecision(reference, decision, status)

    def _write_named_artifact(
        self,
        run_id: str,
        *,
        artifact_name: str,
        artifact_type: str,
        source_url: str | None,
        content: Mapping[str, object],
    ) -> ArtifactReference:
        run_dir = self._require_run(run_id, allow_missing_manifest=artifact_name == "run.json")
        if Path(artifact_name).name != artifact_name:
            raise ArtifactLimitError("Artifact name must be a basename.")
        if source_url is not None:
            source_url = self._validate_source_url(source_url)
        bounded_content = self._bound_value(dict(content), key="content", depth=0)
        envelope = {
            "schema_version": 1,
            "run_id": run_id,
            "artifact_type": artifact_type,
            "created_at": self._utc_now().isoformat(),
            "source_url": source_url,
            "content": bounded_content,
        }
        if artifact_name != "run.json":
            envelope["run_manifest_hash"] = sha256(
                (run_dir / "run.json").read_bytes()
            ).hexdigest()
        serialized = json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        raw = f"{serialized}\n".encode()
        if len(raw) > self.max_json_bytes:
            raise ArtifactLimitError(
                f"Artifact exceeds maximum stored bytes ({self.max_json_bytes})."
            )
        path = run_dir / artifact_name
        self._atomic_write(path, raw)
        if source_url is None:
            # The run manifest is an internal lifecycle anchor, not a persisted finding.
            return ArtifactReference(
                run_id=run_id,
                artifact_name=artifact_name,
                source_url="https://run4221.invalid/research-run",
                content_hash=sha256(raw).hexdigest(),
            )
        return ArtifactReference(
            run_id=run_id,
            artifact_name=artifact_name,
            source_url=source_url,
            content_hash=sha256(raw).hexdigest(),
        )

    def _atomic_write(self, path: Path, raw: bytes) -> None:
        temporary = path.parent / f".{path.name}.{uuid4()}.tmp"
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise ArtifactExistsError(f"Artifact already exists: {path.name}") from error
            fsync_directory(path.parent)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def _reference_from_path(self, path: Path) -> ArtifactReference:
        raw = path.read_bytes()
        try:
            payload = json.loads(raw)
            run_id = payload["run_id"]
            source_url = payload["source_url"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ArtifactIntegrityError(f"Invalid artifact envelope: {path.name}") from error
        reference = ArtifactReference(
            run_id=run_id,
            artifact_name=path.name,
            source_url=source_url,
            content_hash=sha256(raw).hexdigest(),
        )
        self.read_artifact(reference)
        return reference

    def _path_for_reference(self, reference: ArtifactReference) -> Path:
        run_dir = self._require_run(reference.run_id)
        path = run_dir / reference.artifact_name
        if path.is_symlink():
            raise ArtifactIntegrityError("Artifact symlinks are not allowed.")
        return path

    def _run_dir(self, run_id: str) -> Path:
        try:
            canonical_run_id = str(UUID(run_id))
        except ValueError as error:
            raise ArtifactLifecycleError("Invalid research run ID.") from error
        if canonical_run_id != run_id:
            raise ArtifactLifecycleError("Research run ID must use canonical UUID form.")
        return self.root / canonical_run_id

    def _require_run(self, run_id: str, *, allow_missing_manifest: bool = False) -> Path:
        run_dir = self._run_dir(run_id)
        if not run_dir.is_dir() or run_dir.is_symlink():
            raise ArtifactLifecycleError(f"Research run does not exist: {run_id}")
        if not allow_missing_manifest and not (run_dir / "run.json").is_file():
            raise ArtifactLifecycleError(f"Research run manifest is missing: {run_id}")
        return run_dir

    def _bound_value(self, value: object, *, key: str, depth: int) -> Any:
        if depth > self.max_depth:
            raise ArtifactLimitError(f"Artifact metadata exceeds maximum depth ({self.max_depth}).")
        if isinstance(value, BaseModel):
            return self._bound_value(value.model_dump(mode="json"), key=key, depth=depth)
        if value is None or isinstance(value, bool | int):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ArtifactLimitError("Artifact numbers must be finite.")
            return value
        if isinstance(value, str):
            if key == "url" or key.endswith("_url"):
                return self._validate_source_url(value)
            return self._truncate(value)
        if isinstance(value, Mapping):
            if len(value) > self.max_collection_items:
                raise ArtifactLimitError(
                    f"Artifact exceeds maximum collection items ({self.max_collection_items})."
                )
            bounded: dict[str, Any] = {}
            for item_key, item_value in value.items():
                if not isinstance(item_key, str) or not item_key:
                    raise ArtifactLimitError("Artifact metadata keys must be non-empty strings.")
                if len(item_key) > MAX_METADATA_KEY_CHARS:
                    raise ArtifactLimitError("Artifact metadata key is too long.")
                bounded[item_key] = self._bound_value(
                    item_value,
                    key=item_key,
                    depth=depth + 1,
                )
            return bounded
        if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
            if len(value) > self.max_collection_items:
                raise ArtifactLimitError(
                    f"Artifact exceeds maximum collection items ({self.max_collection_items})."
                )
            return [
                self._bound_value(item, key=key, depth=depth + 1) for item in value
            ]
        raise ArtifactLimitError(f"Unsupported artifact metadata type: {type(value).__name__}")

    def _truncate(self, value: str) -> str:
        if len(value) <= self.max_text_chars:
            return value
        marker = TRUNCATION_MARKER[: self.max_text_chars]
        return value[: self.max_text_chars - len(marker)] + marker

    @staticmethod
    def _validate_source_url(value: str) -> str:
        if len(value) > 2_048:
            raise ArtifactLimitError("Artifact URL exceeds 2048 characters.")
        try:
            return validate_http_url(value)
        except ValueError as error:
            raise ArtifactLimitError(str(error)) from error

    @staticmethod
    def _validate_artifact_type(value: str) -> None:
        if len(value) > MAX_ARTIFACT_TYPE_CHARS or not _ARTIFACT_TYPE_PATTERN.fullmatch(value):
            raise ArtifactLimitError("Artifact type must be a short lowercase identifier.")

    def _bounded_queue_reference(self, value: str) -> str:
        value = value.strip()
        if not value or len(value) > MAX_QUEUE_REFERENCE_CHARS or "\n" in value:
            raise ArtifactLimitError("Queue reference must be a short single-line value.")
        return value

    def _validate_resolution(self, resolution: QueueResolution) -> None:
        if not isinstance(resolution, QueueResolution):
            raise ArtifactLifecycleError("Queue resolver must return QueueResolution.")
        if not isinstance(resolution.state, QueueResolutionState):
            raise ArtifactLifecycleError("Queue resolution has an invalid state.")
        if resolution.state is QueueResolutionState.COMMITTED:
            self._bounded_queue_reference(resolution.queue_reference or "")
        elif resolution.queue_reference is not None:
            raise ArtifactLifecycleError(
                "Only a committed resolution may include a queue reference."
            )

    @staticmethod
    def _validate_prepared_status(
        decision: ResearchDecision,
        status: ResearchRunStatus,
    ) -> None:
        if decision.action is not DecisionAction.PROPOSE_UPDATE:
            raise ArtifactLifecycleError("Only a queue-writing decision can be prepared.")
        if (
            status.status is not RunState.SUCCEEDED
            or status.outcome is not RunOutcome.PROPOSAL_CREATED
        ):
            raise ArtifactLifecycleError(
                "Prepared decision status must match its committed queue outcome."
            )

    @staticmethod
    def _verify_bytes(reference: ArtifactReference, raw: bytes) -> None:
        if sha256(raw).hexdigest() != reference.content_hash:
            raise ArtifactIntegrityError("Artifact content hash does not match its reference.")

    def _utc_now(self) -> datetime:
        value = self._now()
        if value.tzinfo is None:
            raise ArtifactLifecycleError("Artifact clock must return a timezone-aware datetime.")
        return value.astimezone(UTC)

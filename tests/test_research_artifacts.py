from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from run4221.ingestion.page_snapshot import PageLink, PageSnapshot
from run4221.researcher.artifacts import (
    ArtifactExistsError,
    ArtifactIntegrityError,
    ArtifactLimitError,
    QueueResolution,
    QueueResolutionState,
    ResearchArtifactStore,
)
from run4221.researcher.schemas import (
    ArtifactReference,
    ResearchDecision,
    ResearchRunStatus,
)

SOURCE_URL = "https://example.com/marathon"


def decision(evidence: ArtifactReference) -> ResearchDecision:
    return ResearchDecision(
        action="propose_update",
        summary="Registration is open on the captured official page.",
        confidence=0.9,
        proposed_fields={"registration_status": "open"},
        evidence=[evidence],
        applicability=[
            {
                "evidence": evidence,
                "event_identity": "confirmed",
                "event_edition": "confirmed",
                "distance_category": "confirmed",
                "applicable_fields": ["registration_status"],
            }
        ],
        field_support=[
            {
                "field": "registration_status",
                "evidence": [evidence],
            }
        ],
    )


def write_registration_evidence(
    store: ResearchArtifactStore,
    run_id: str,
) -> ArtifactReference:
    return store.write_artifact(
        run_id,
        artifact_type="evidence",
        source_url=SOURCE_URL,
        content={"text": "Registration is open."},
    )


def committed_status() -> ResearchRunStatus:
    return ResearchRunStatus(status="succeeded", outcome="proposal_created")


def test_same_second_runs_and_artifacts_are_unique(tmp_path: Path) -> None:
    def now() -> datetime:
        return datetime(2026, 8, 31, 12, 0, tzinfo=UTC)

    store = ResearchArtifactStore(tmp_path, now=now)

    first_run = store.create_run(job_type="refresh", metadata={"event_id": 1})
    second_run = store.create_run(job_type="refresh", metadata={"event_id": 1})
    first = store.write_artifact(
        first_run,
        artifact_type="search_summary",
        source_url=SOURCE_URL,
        content={"summary": "First"},
    )
    second = store.write_artifact(
        first_run,
        artifact_type="search_summary",
        source_url=SOURCE_URL,
        content={"summary": "Second"},
    )

    assert first_run != second_run
    assert first.artifact_name != second.artifact_name
    assert store.read_artifact(first)["content"]["summary"] == "First"
    assert store.read_artifact(second)["content"]["summary"] == "Second"


def test_atomic_write_failure_leaves_no_partial_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ResearchArtifactStore(tmp_path)
    run_id = store.create_run(job_type="refresh")

    def fail_link(_source: os.PathLike[str], _destination: os.PathLike[str]) -> None:
        raise OSError("simulated interrupted publish")

    monkeypatch.setattr("run4221.researcher.artifacts.os.link", fail_link)

    with pytest.raises(OSError, match="simulated interrupted publish"):
        store.write_artifact(
            run_id,
            artifact_type="assessment",
            source_url=SOURCE_URL,
            content={"summary": "Captured"},
        )

    run_dir = tmp_path / run_id
    assert list(run_dir.glob("assessment-*.json")) == []
    assert list(run_dir.glob(".*.tmp")) == []


def test_exact_stored_byte_hash_detects_tampering_and_reference_has_no_path(
    tmp_path: Path,
) -> None:
    store = ResearchArtifactStore(tmp_path)
    run_id = store.create_run(job_type="refresh")
    reference = store.write_artifact(
        run_id,
        artifact_type="evidence",
        source_url=SOURCE_URL,
        content={"text": "Registration opens today."},
    )
    serialized_reference = json.dumps(reference.model_dump(mode="json"))

    assert reference.content_hash == store.hash_artifact(reference)
    assert str(tmp_path) not in serialized_reference
    assert str(Path.home()) not in serialized_reference
    assert Path(reference.artifact_name).name == reference.artifact_name

    artifact_path = tmp_path / run_id / reference.artifact_name
    artifact_path.write_bytes(artifact_path.read_bytes() + b"\n")

    with pytest.raises(ArtifactIntegrityError, match="content hash"):
        store.verify_artifact(reference)


def test_artifact_reference_detects_run_manifest_tampering(tmp_path: Path) -> None:
    store = ResearchArtifactStore(tmp_path)
    run_id = store.create_run(job_type="refresh")
    reference = store.write_artifact(
        run_id,
        artifact_type="evidence",
        source_url=SOURCE_URL,
        content={"text": "Registration opens today."},
    )
    manifest_path = tmp_path / run_id / "run.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    with pytest.raises(ArtifactIntegrityError, match="run manifest hash"):
        store.read_artifact(reference)


def test_artifact_content_is_bounded_and_excess_metadata_is_rejected(
    tmp_path: Path,
) -> None:
    store = ResearchArtifactStore(tmp_path, max_text_chars=80, max_collection_items=3)
    run_id = store.create_run(job_type="refresh")
    reference = store.write_artifact(
        run_id,
        artifact_type="assessment",
        source_url=SOURCE_URL,
        content={"summary": "<script>" + ("x" * 200)},
    )

    summary = store.read_artifact(reference)["content"]["summary"]
    assert summary.startswith("<script>")
    assert summary.endswith("...[truncated]")
    assert len(summary) == 80

    with pytest.raises(ArtifactLimitError, match="collection items"):
        store.write_artifact(
            run_id,
            artifact_type="search_summary",
            source_url=SOURCE_URL,
            content={"sources": [1, 2, 3, 4]},
        )


def test_page_snapshot_is_written_as_bounded_evidence(tmp_path: Path) -> None:
    store = ResearchArtifactStore(tmp_path)
    run_id = store.create_run(job_type="refresh")
    snapshot = PageSnapshot(
        source_url=SOURCE_URL,
        final_url=SOURCE_URL,
        fetched_at=datetime(2026, 8, 31, 12, 0, tzinfo=UTC),
        status_code=200,
        content_type="text/html",
        title="Example Marathon",
        normalized_text="Registration is open.",
        text_hash="b" * 64,
        links=(PageLink(url=f"{SOURCE_URL}/register", text="Register"),),
    )

    reference = store.write_page_snapshot(run_id, snapshot)
    payload = store.read_artifact(reference)

    assert payload["artifact_type"] == "page_snapshot"
    assert payload["content"] == snapshot.to_dict()
    assert reference.source_url == SOURCE_URL


def test_prepared_decision_requires_committed_evidence_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    store = ResearchArtifactStore(tmp_path)
    run_id = store.create_run(job_type="refresh")
    evidence = write_registration_evidence(store, run_id)
    prepared = store.prepare_decision(
        run_id,
        source_url=SOURCE_URL,
        decision=decision(evidence),
        committed_status=committed_status(),
    )

    assert prepared.artifact_name == "prepared.json"
    with pytest.raises(ArtifactExistsError, match="prepared.json"):
        store.prepare_decision(
            run_id,
            source_url=SOURCE_URL,
            decision=decision(evidence),
            committed_status=committed_status(),
        )


def test_reconcile_absent_prepared_decision_records_truthful_terminal(
    tmp_path: Path,
) -> None:
    store = ResearchArtifactStore(tmp_path)
    run_id = store.create_run(job_type="refresh")
    evidence = write_registration_evidence(store, run_id)
    prepared = store.prepare_decision(
        run_id,
        source_url=SOURCE_URL,
        decision=decision(evidence),
        committed_status=committed_status(),
    )

    results = store.reconcile_prepared(
        lambda found: QueueResolution.absent()
        if found.reference == prepared
        else QueueResolution.inconclusive("unexpected prepared decision")
    )

    assert len(results) == 1
    assert results[0].state is QueueResolutionState.ABSENT
    terminal = store.read_artifact(results[0].terminal_reference)
    assert terminal["content"]["queue_state"] == "absent"
    assert terminal["content"]["status"] == {
        "detail": "Prepared decision was not found in a moderation queue.",
        "outcome": "inconclusive",
        "status": "failed",
    }


def test_non_queue_outcome_can_finalize_without_prepared_decision(tmp_path: Path) -> None:
    store = ResearchArtifactStore(tmp_path)
    run_id = store.create_run(job_type="refresh")

    terminal_reference = store.finalize_without_queue(
        run_id,
        source_url=SOURCE_URL,
        status=ResearchRunStatus(status="succeeded", outcome="no_change"),
    )

    terminal = store.read_artifact(terminal_reference)
    assert terminal["content"]["prepared_artifact"] is None
    assert terminal["content"]["queue_state"] is None
    assert terminal["content"]["status"]["outcome"] == "no_change"


def test_reconcile_committed_after_finalize_crash_records_one_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ResearchArtifactStore(tmp_path)
    run_id = store.create_run(job_type="refresh")
    evidence = write_registration_evidence(store, run_id)
    prepared = store.prepare_decision(
        run_id,
        source_url=SOURCE_URL,
        decision=decision(evidence),
        committed_status=committed_status(),
    )
    real_link = os.link
    failed = False

    def fail_terminal_once(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        nonlocal failed
        if str(destination).endswith("terminal.json") and not failed:
            failed = True
            raise OSError("crash before terminal finalization")
        real_link(source, destination)

    monkeypatch.setattr("run4221.researcher.artifacts.os.link", fail_terminal_once)

    with pytest.raises(OSError, match="crash before terminal finalization"):
        store.finalize_committed(prepared, queue_reference="proposed_event_update:42")

    assert not (tmp_path / run_id / "terminal.json").exists()
    results = store.reconcile_prepared(
        lambda found: QueueResolution.committed("proposed_event_update:42")
        if found.reference == prepared
        else QueueResolution.inconclusive("unexpected prepared decision")
    )

    assert len(results) == 1
    assert results[0].state is QueueResolutionState.COMMITTED
    terminal = store.read_artifact(results[0].terminal_reference)
    assert terminal["content"]["queue_state"] == "committed"
    assert terminal["content"]["queue_reference"] == "proposed_event_update:42"
    assert terminal["content"]["status"]["outcome"] == "proposal_created"
    assert [path.name for path in (tmp_path / run_id).glob("terminal*.json")] == [
        "terminal.json"
    ]
    with pytest.raises(ArtifactExistsError, match="terminal.json"):
        store.finalize_committed(prepared, queue_reference="proposed_event_update:42")


def test_inconclusive_reconciliation_does_not_invent_terminal_state(tmp_path: Path) -> None:
    store = ResearchArtifactStore(tmp_path)
    run_id = store.create_run(job_type="refresh")
    evidence = write_registration_evidence(store, run_id)
    store.prepare_decision(
        run_id,
        source_url=SOURCE_URL,
        decision=decision(evidence),
        committed_status=committed_status(),
    )

    results = store.reconcile_prepared(
        lambda _prepared: QueueResolution.inconclusive("database unavailable")
    )

    assert results[0].state is QueueResolutionState.INCONCLUSIVE
    assert results[0].terminal_reference is None
    assert not (tmp_path / run_id / "terminal.json").exists()
    reconciliation = store.read_artifact(results[0].reconciliation_reference)
    assert reconciliation["content"]["queue_state"] == "inconclusive"
    assert reconciliation["content"]["detail"] == "database unavailable"

    repeated = store.reconcile_prepared(
        lambda _prepared: QueueResolution.inconclusive("still unavailable")
    )
    assert repeated[0].reconciliation_reference == results[0].reconciliation_reference

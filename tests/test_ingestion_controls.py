# ruff: noqa: E402
from datetime import datetime, timezone
import sys

sys.argv = sys.argv[:1]

import pytest
from fastapi import BackgroundTasks, Form, HTTPException

from lightrag.api.routers import document_routes

from lightrag.api.routers.document_routes import (
    _cleanup_expired_preflight_entries,
    _get_preflight_entry,
    _optional_form_bool,
    _optional_form_str,
    _raise_large_ingestion_error_with_preflight,
    _store_preflight_entry,
    enforce_max_ingestion_chunks,
)
from lightrag.api.routers.ingestion_routes import (
    request_ingestion_stop,
    start_over_ingestion_job,
)
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.kg.shared_storage import (
    finalize_share_data,
    get_namespace_data,
    get_namespace_lock,
    initialize_pipeline_status,
    initialize_share_data,
)


class WhitespaceTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


class GuardRAG:
    chunk_token_size = 5
    chunk_overlap_token_size = 1
    tokenizer = WhitespaceTokenizer()


class MemoryDocStatus:
    def __init__(self, docs: dict[str, DocProcessingStatus]):
        self.docs = docs

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        return {
            doc_id: doc for doc_id, doc in self.docs.items() if doc.track_id == track_id
        }

    async def upsert(self, docs: dict[str, dict]) -> None:
        for doc_id, data in docs.items():
            existing = self.docs[doc_id]
            self.docs[doc_id] = DocProcessingStatus(
                content_summary=data.get("content_summary", existing.content_summary),
                content_length=data.get("content_length", existing.content_length),
                file_path=data.get("file_path", existing.file_path),
                status=data.get("status", existing.status),
                created_at=data.get("created_at", existing.created_at),
                updated_at=data.get("updated_at", existing.updated_at),
                track_id=data.get("track_id", existing.track_id),
                chunks_count=data.get("chunks_count", existing.chunks_count),
                chunks_list=data.get("chunks_list", existing.chunks_list),
                error_msg=data.get("error_msg"),
                metadata=data.get("metadata", existing.metadata),
            )


class ControlRAG:
    def __init__(self, workspace: str, doc_status: MemoryDocStatus):
        self.workspace = workspace
        self.doc_status = doc_status
        self.process_called = False

    async def apipeline_process_enqueue_documents(self) -> None:
        self.process_called = True


def make_doc(status: DocStatus, track_id: str = "job-1") -> DocProcessingStatus:
    now = datetime.now(timezone.utc).isoformat()
    return DocProcessingStatus(
        content_summary="summary",
        content_length=42,
        file_path="doc.txt",
        status=status,
        created_at=now,
        updated_at=now,
        track_id=track_id,
        chunks_count=2,
        chunks_list=["chunk-1", "chunk-2"],
        metadata={},
    )


@pytest.mark.offline
def test_large_ingestion_guard_requires_confirmation(monkeypatch):
    monkeypatch.setattr(document_routes.global_args, "max_ingestion_chunks", 2)

    with pytest.raises(HTTPException) as exc_info:
        enforce_max_ingestion_chunks(
            GuardRAG(),
            ["one two three four five six seven eight nine"],
            confirm_large_ingestion=False,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["estimated_chunks"] == 3
    assert exc_info.value.detail["confirm_large_ingestion_required"] is True


@pytest.mark.offline
def test_large_ingestion_guard_allows_explicit_confirmation(monkeypatch):
    monkeypatch.setattr(document_routes.global_args, "max_ingestion_chunks", 2)

    estimated_chunks = enforce_max_ingestion_chunks(
        GuardRAG(),
        ["one two three four five six seven eight nine"],
        confirm_large_ingestion=True,
    )

    assert estimated_chunks == 3


@pytest.mark.offline
def test_large_ingestion_guard_error_includes_preflight_id():
    with pytest.raises(HTTPException) as exc_info:
        _raise_large_ingestion_error_with_preflight(
            estimated_chunks=12, max_chunks=10, preflight_id="preflight_abc123"
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "large_ingestion_requires_confirmation"
    assert exc_info.value.detail["preflight_id"] == "preflight_abc123"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_stop_marks_queued_job_documents_stopped():
    workspace = "test_stop_marks_queued"
    initialize_share_data()
    try:
        await initialize_pipeline_status(workspace)
        doc_status = MemoryDocStatus({"doc-1": make_doc(DocStatus.PENDING)})
        rag = ControlRAG(workspace, doc_status)

        response = await request_ingestion_stop(rag, "job-1")

        assert response.status == "stopped"
        assert doc_status.docs["doc-1"].status == DocStatus.STOPPED
        assert doc_status.docs["doc-1"].error_msg == "Stopped safely by user request"
    finally:
        finalize_share_data()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_start_over_requeues_stopped_job_documents():
    workspace = "test_start_over_requeues_stopped"
    initialize_share_data()
    try:
        await initialize_pipeline_status(workspace)
        doc_status = MemoryDocStatus({"doc-1": make_doc(DocStatus.STOPPED)})
        rag = ControlRAG(workspace, doc_status)
        background_tasks = BackgroundTasks()

        response = await start_over_ingestion_job(rag, "job-1", background_tasks)

        assert response.status == "start_over_started"
        assert doc_status.docs["doc-1"].status == DocStatus.PENDING
        assert doc_status.docs["doc-1"].error_msg is None
        assert len(background_tasks.tasks) == 1
    finally:
        finalize_share_data()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_stop_active_job_sets_pipeline_stop_flag():
    workspace = "test_stop_active_job"
    initialize_share_data()
    try:
        await initialize_pipeline_status(workspace)
        doc_status = MemoryDocStatus({"doc-1": make_doc(DocStatus.PROCESSING)})
        rag = ControlRAG(workspace, doc_status)

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=workspace
        )
        async with pipeline_status_lock:
            pipeline_status["busy"] = True
            pipeline_status["active_track_ids"] = ["job-1"]
            pipeline_status["history_messages"][:] = []

        response = await request_ingestion_stop(rag, "job-1")

        assert response.status == "stop_requested"
        async with pipeline_status_lock:
            assert pipeline_status["stop_requested"] is True
            assert pipeline_status["stopped_job_id"] == "job-1"
    finally:
        finalize_share_data()


@pytest.mark.offline
def test_optional_form_str_ignores_unparsed_form_default():
    assert _optional_form_str(Form(default=None)) is None
    assert _optional_form_str("  preflight_abc  ") == "preflight_abc"
    assert _optional_form_str("   ") is None


@pytest.mark.offline
def test_optional_form_bool_ignores_unparsed_form_default():
    assert _optional_form_bool(Form(False)) is False
    assert _optional_form_bool(True) is True
    assert _optional_form_bool("true") is True
    assert _optional_form_bool("0") is False


@pytest.mark.offline
@pytest.mark.asyncio
async def test_preflight_entry_expires_on_lookup():
    workspace = "test_preflight_expiry"
    initialize_share_data()
    try:
        rag = ControlRAG(workspace, MemoryDocStatus({}))
        preflight_id = "preflight_expired"
        await _store_preflight_entry(
            rag,
            {
                "preflight_id": preflight_id,
                "file_name": "doc.txt",
                "file_sha256": "abc",
                "estimated_chunks": 1,
                "expires_at_ts": 1,
                "created_at_ts": 1,
            },
        )

        assert await _get_preflight_entry(rag, preflight_id) is None
    finally:
        finalize_share_data()


@pytest.mark.offline
def test_cleanup_expired_preflight_entries_removes_stale_rows():
    state = {
        "entries": {
            "fresh": {"expires_at_ts": 9_999_999_999, "created_at_ts": 2},
            "stale": {"expires_at_ts": 1, "created_at_ts": 1},
        }
    }
    _cleanup_expired_preflight_entries(state)
    assert "stale" not in state["entries"]
    assert "fresh" in state["entries"]

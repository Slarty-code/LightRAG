"""
API routes and helpers for ingestion job controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from lightrag import LightRAG
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock


router = APIRouter(prefix="/api/ingestion", tags=["ingestion"])


@dataclass
class IngestionJobRecord:
    """Best-effort in-memory view of API-submitted ingestion jobs."""

    job_id: str
    state: str
    updated_at: str
    processed_docs: int = 0
    chunk_index: int = 0
    active_track_ids: list[str] = dataclass_field(default_factory=list)


INGESTION_JOB_REGISTRY: dict[str, IngestionJobRecord] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_ingestion_job(job_id: str, state: str = "queued") -> None:
    """Register a job returned by the API ingestion routes."""
    INGESTION_JOB_REGISTRY[job_id] = IngestionJobRecord(
        job_id=job_id,
        state=state,
        updated_at=_utc_now(),
        active_track_ids=[job_id],
    )


def update_ingestion_job(
    job_id: str,
    state: str,
    *,
    processed_docs: int | None = None,
    chunk_index: int | None = None,
) -> None:
    """Update the in-memory job registry without making it the source of truth."""
    record = INGESTION_JOB_REGISTRY.get(job_id)
    if record is None:
        register_ingestion_job(job_id, state)
        record = INGESTION_JOB_REGISTRY[job_id]
    record.state = state
    record.updated_at = _utc_now()
    if processed_docs is not None:
        record.processed_docs = processed_docs
    if chunk_index is not None:
        record.chunk_index = chunk_index


class IngestionControlResponse(BaseModel):
    """Response model for ingestion pause and resume operations."""

    status: Literal[
        "pause_requested",
        "paused",
        "already_paused",
        "resume_started",
        "already_running",
        "not_busy",
        "not_paused",
    ] = Field(description="Result of the ingestion control request")
    message: str = Field(description="Human-readable operation summary")
    job_id: str = Field(description="Ingestion track/job identifier")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "pause_requested",
                "message": "Pause requested. The pipeline will stop at the next safe checkpoint.",
                "job_id": "upload_20250729_170612_abc123",
            }
        }
    )


class IngestionJobStatusResponse(BaseModel):
    """Response model for the in-memory ingestion job registry."""

    job_id: str
    state: str
    updated_at: str
    processed_docs: int = 0
    chunk_index: int = 0
    active_track_ids: list[str] = Field(default_factory=list)


def _doc_status_value(doc: object) -> str:
    status = getattr(doc, "status", None)
    if isinstance(status, DocStatus):
        return status.value
    return str(status or "")


def _doc_field(doc: object, name: str, default=None):
    if isinstance(doc, dict):
        return doc.get(name, default)
    return getattr(doc, name, default)


def _build_doc_status_payload(
    doc: object,
    status: DocStatus,
    *,
    error_msg: Optional[str],
    metadata_extra: dict,
) -> dict:
    metadata = dict(_doc_field(doc, "metadata", {}) or {})
    metadata.update(metadata_extra)
    return {
        "status": status,
        "content_summary": _doc_field(doc, "content_summary", ""),
        "content_length": _doc_field(doc, "content_length", 0),
        "created_at": _doc_field(doc, "created_at", _utc_now()),
        "updated_at": _utc_now(),
        "file_path": _doc_field(doc, "file_path", "unknown_source"),
        "track_id": _doc_field(doc, "track_id", None),
        "chunks_count": _doc_field(doc, "chunks_count", None),
        "chunks_list": _doc_field(doc, "chunks_list", []),
        "error_msg": error_msg,
        "metadata": metadata,
    }


async def mark_job_docs_paused(rag: LightRAG, job_id: str) -> int:
    """Mark queued documents for a job as paused before they enter the pipeline."""
    docs = await rag.doc_status.get_docs_by_track_id(job_id)
    paused_updates = {}
    for doc_id, doc in docs.items():
        if _doc_status_value(doc) in {
            DocStatus.PENDING.value,
            DocStatus.PROCESSING.value,
            DocStatus.FAILED.value,
        }:
            paused_updates[doc_id] = _build_doc_status_payload(
                doc,
                DocStatus.PAUSED,
                error_msg="Paused by user request",
                metadata_extra={"paused_at": _utc_now()},
            )

    if paused_updates:
        await rag.doc_status.upsert(paused_updates)
    return len(paused_updates)


async def restore_paused_job_docs(rag: LightRAG, job_id: str) -> int:
    """Move paused job documents back to pending so the pipeline can resume them."""
    docs = await rag.doc_status.get_docs_by_track_id(job_id)
    resume_updates = {}
    for doc_id, doc in docs.items():
        if _doc_status_value(doc) == DocStatus.PAUSED.value:
            resume_updates[doc_id] = _build_doc_status_payload(
                doc,
                DocStatus.PENDING,
                error_msg=None,
                metadata_extra={"resumed_at": _utc_now()},
            )

    if resume_updates:
        await rag.doc_status.upsert(resume_updates)
    return len(resume_updates)


async def request_ingestion_pause(
    rag: LightRAG, job_id: str
) -> IngestionControlResponse:
    """Request a cooperative pause for the active or queued ingestion job."""
    docs = await rag.doc_status.get_docs_by_track_id(job_id)
    if not docs:
        raise HTTPException(status_code=404, detail=f"Ingestion job '{job_id}' not found")

    pipeline_status = await get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )

    async with pipeline_status_lock:
        active_track_ids = set(pipeline_status.get("active_track_ids") or [])
        busy = bool(pipeline_status.get("busy", False))
        if busy:
            if active_track_ids and job_id not in active_track_ids:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Ingestion job '{job_id}' is not active in the current pipeline run"
                    ),
                )

            pipeline_status["pause_requested"] = True
            pipeline_status["paused"] = False
            pipeline_status["paused_job_id"] = job_id
            message = (
                "Pause requested. The pipeline will stop at the next safe checkpoint."
            )
            pipeline_status["latest_message"] = message
            pipeline_status["history_messages"].append(message)
            update_ingestion_job(
                job_id,
                "pausing",
                processed_docs=int(pipeline_status.get("cur_batch", 0) or 0),
                chunk_index=int(pipeline_status.get("chunks_done", 0) or 0),
            )
            return IngestionControlResponse(
                status="pause_requested",
                message=message,
                job_id=job_id,
            )

    paused_count = await mark_job_docs_paused(rag, job_id)
    if paused_count:
        update_ingestion_job(job_id, "paused")
        return IngestionControlResponse(
            status="paused",
            message=f"Paused {paused_count} queued document(s).",
            job_id=job_id,
        )

    if any(_doc_status_value(doc) == DocStatus.PAUSED.value for doc in docs.values()):
        update_ingestion_job(job_id, "paused")
        return IngestionControlResponse(
            status="already_paused",
            message="Ingestion job is already paused.",
            job_id=job_id,
        )

    return IngestionControlResponse(
        status="not_busy",
        message="No active or queued documents are available to pause for this job.",
        job_id=job_id,
    )


async def resume_ingestion_job(
    rag: LightRAG, job_id: str, background_tasks: BackgroundTasks
) -> IngestionControlResponse:
    """Resume a paused ingestion job by re-queuing its paused documents."""
    docs = await rag.doc_status.get_docs_by_track_id(job_id)
    if not docs:
        raise HTTPException(status_code=404, detail=f"Ingestion job '{job_id}' not found")

    resumed_count = await restore_paused_job_docs(rag, job_id)
    if resumed_count == 0 and not any(
        _doc_status_value(doc) == DocStatus.PAUSED.value for doc in docs.values()
    ):
        return IngestionControlResponse(
            status="not_paused",
            message="Ingestion job has no paused documents to resume.",
            job_id=job_id,
        )

    pipeline_status = await get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )
    async with pipeline_status_lock:
        pipeline_status["pause_requested"] = False
        pipeline_status["paused"] = False
        pipeline_status["paused_job_id"] = None
        if pipeline_status.get("busy", False):
            pipeline_status["request_pending"] = True
            update_ingestion_job(job_id, "queued")
            return IngestionControlResponse(
                status="already_running",
                message="Pipeline is already running; resume request has been queued.",
                job_id=job_id,
            )

    update_ingestion_job(job_id, "resuming")
    background_tasks.add_task(rag.apipeline_process_enqueue_documents)
    return IngestionControlResponse(
        status="resume_started",
        message=f"Resume started for {resumed_count} paused document(s).",
        job_id=job_id,
    )


def create_ingestion_routes(rag: LightRAG, api_key: Optional[str] = None) -> APIRouter:
    """Create ingestion control routes backed by pipeline shared state."""
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "/pause/{job_id}",
        response_model=IngestionControlResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def pause_ingestion(job_id: str) -> IngestionControlResponse:
        return await request_ingestion_pause(rag, job_id)

    @router.post(
        "/resume/{job_id}",
        response_model=IngestionControlResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def resume_ingestion(
        job_id: str, background_tasks: BackgroundTasks
    ) -> IngestionControlResponse:
        return await resume_ingestion_job(rag, job_id, background_tasks)

    @router.get(
        "/jobs/{job_id}",
        response_model=IngestionJobStatusResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_ingestion_job(job_id: str) -> IngestionJobStatusResponse:
        record = INGESTION_JOB_REGISTRY.get(job_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"Ingestion job '{job_id}' not found"
            )
        return IngestionJobStatusResponse(**record.__dict__)

    return router

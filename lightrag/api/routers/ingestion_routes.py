"""
API routes for ingestion pause and resume controls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from lightrag import LightRAG
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock


router = APIRouter(prefix="/api/ingestion", tags=["ingestion"])

_PAUSABLE_DOC_STATUSES = {
    DocStatus.PENDING.value,
    DocStatus.PARSING.value,
    DocStatus.ANALYZING.value,
    DocStatus.PROCESSING.value,
    DocStatus.FAILED.value,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestionControlResponse(BaseModel):
    status: Literal[
        "pause_requested",
        "paused",
        "already_paused",
        "resume_started",
        "already_running",
        "not_busy",
        "not_paused",
    ]
    message: str
    job_id: str


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
    docs = await rag.doc_status.get_docs_by_track_id(job_id)
    paused_updates = {}
    for doc_id, doc in docs.items():
        if _doc_status_value(doc) in _PAUSABLE_DOC_STATUSES:
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


async def count_paused_job_docs(rag: LightRAG, job_id: str) -> int:
    docs = await rag.doc_status.get_docs_by_track_id(job_id)
    return sum(
        1 for doc in docs.values() if _doc_status_value(doc) == DocStatus.PAUSED.value
    )


async def request_ingestion_pause(
    rag: LightRAG, job_id: str
) -> IngestionControlResponse:
    docs = await rag.doc_status.get_docs_by_track_id(job_id)
    if not docs:
        raise HTTPException(
            status_code=404, detail=f"Ingestion job '{job_id}' not found"
        )

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
            return IngestionControlResponse(
                status="pause_requested",
                message=message,
                job_id=job_id,
            )

    paused_count = await mark_job_docs_paused(rag, job_id)
    if paused_count:
        return IngestionControlResponse(
            status="paused",
            message=f"Paused {paused_count} queued document(s).",
            job_id=job_id,
        )

    if any(_doc_status_value(doc) == DocStatus.PAUSED.value for doc in docs.values()):
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
    pipeline_status = await get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )

    requested_docs = await rag.doc_status.get_docs_by_track_id(job_id)
    if not requested_docs:
        raise HTTPException(
            status_code=404, detail=f"Ingestion job '{job_id}' not found"
        )

    # Resolve the best resume target. When multiple tracks are active, the UI may
    # send a valid-but-not-current track id; fall back to the currently paused id
    # and active track ids to find where PAUSED docs actually live.
    candidates: list[str] = []
    for candidate in [
        job_id,
        pipeline_status.get("paused_job_id"),
        *(pipeline_status.get("active_track_ids") or []),
    ]:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    target_job_id = job_id
    resumed_count = 0
    for candidate in candidates:
        paused_count = await count_paused_job_docs(rag, candidate)
        if paused_count > 0:
            target_job_id = candidate
            resumed_count = await restore_paused_job_docs(rag, candidate)
            break

    if resumed_count == 0:
        return IngestionControlResponse(
            status="not_paused",
            message="Ingestion job has no paused documents to resume.",
            job_id=job_id,
        )

    async with pipeline_status_lock:
        pipeline_status["pause_requested"] = False
        pipeline_status["paused"] = False
        pipeline_status["paused_job_id"] = None
        if pipeline_status.get("busy", False):
            pipeline_status["request_pending"] = True
            return IngestionControlResponse(
                status="already_running",
                message="Pipeline is already running; resume request has been queued.",
                job_id=target_job_id,
            )

    background_tasks.add_task(rag.apipeline_process_enqueue_documents)
    return IngestionControlResponse(
        status="resume_started",
        message=f"Resume started for {resumed_count} paused document(s).",
        job_id=target_job_id,
    )


def create_ingestion_routes(rag: LightRAG, api_key: Optional[str] = None) -> APIRouter:
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

    return router

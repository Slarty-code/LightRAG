"""
API routes for ingestion stop-safely and start-over controls.
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

# Legacy doc_status value before STOPPED rename (dev/fork compatibility).
_LEGACY_STOPPED_STATUS = "paused"

_STOPPABLE_DOC_STATUSES = {
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
        "stop_requested",
        "stopped",
        "already_stopped",
        "start_over_started",
        "already_running",
        "not_busy",
        "not_stopped",
    ]
    message: str
    job_id: str


def _doc_status_value(doc: object) -> str:
    status = getattr(doc, "status", None)
    if isinstance(status, DocStatus):
        return status.value
    return str(status or "")


def _is_stopped_doc_status(value: str) -> bool:
    return value in {DocStatus.STOPPED.value, _LEGACY_STOPPED_STATUS}


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


async def mark_job_docs_stopped(rag: LightRAG, job_id: str) -> int:
    docs = await rag.doc_status.get_docs_by_track_id(job_id)
    stopped_updates = {}
    for doc_id, doc in docs.items():
        if _doc_status_value(doc) in _STOPPABLE_DOC_STATUSES:
            stopped_updates[doc_id] = _build_doc_status_payload(
                doc,
                DocStatus.STOPPED,
                error_msg="Stopped safely by user request",
                metadata_extra={"stopped_at": _utc_now()},
            )

    if stopped_updates:
        await rag.doc_status.upsert(stopped_updates)
    return len(stopped_updates)


async def requeue_stopped_job_docs(rag: LightRAG, job_id: str) -> int:
    """Re-queue stopped documents for full reprocessing (Start Over)."""
    docs = await rag.doc_status.get_docs_by_track_id(job_id)
    requeue_updates = {}
    for doc_id, doc in docs.items():
        if _is_stopped_doc_status(_doc_status_value(doc)):
            requeue_updates[doc_id] = _build_doc_status_payload(
                doc,
                DocStatus.PENDING,
                error_msg=None,
                metadata_extra={"start_over_at": _utc_now()},
            )

    if requeue_updates:
        await rag.doc_status.upsert(requeue_updates)
    return len(requeue_updates)


async def count_stopped_job_docs(rag: LightRAG, job_id: str) -> int:
    docs = await rag.doc_status.get_docs_by_track_id(job_id)
    return sum(
        1 for doc in docs.values() if _is_stopped_doc_status(_doc_status_value(doc))
    )


async def request_ingestion_stop(
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

            pipeline_status["stop_requested"] = True
            pipeline_status["stopped"] = False
            pipeline_status["stopped_job_id"] = job_id
            # Clear legacy keys if present in shared pipeline_status.
            pipeline_status.pop("pause_requested", None)
            pipeline_status.pop("paused", None)
            pipeline_status.pop("paused_job_id", None)
            message = (
                "Stop requested. The pipeline will stop at the next safe checkpoint."
            )
            pipeline_status["latest_message"] = message
            pipeline_status["history_messages"].append(message)
            return IngestionControlResponse(
                status="stop_requested",
                message=message,
                job_id=job_id,
            )

    stopped_count = await mark_job_docs_stopped(rag, job_id)
    if stopped_count:
        return IngestionControlResponse(
            status="stopped",
            message=f"Stopped {stopped_count} queued document(s) safely.",
            job_id=job_id,
        )

    if any(_is_stopped_doc_status(_doc_status_value(doc)) for doc in docs.values()):
        return IngestionControlResponse(
            status="already_stopped",
            message="Ingestion job is already stopped.",
            job_id=job_id,
        )

    return IngestionControlResponse(
        status="not_busy",
        message="No active or queued documents are available to stop for this job.",
        job_id=job_id,
    )


async def start_over_ingestion_job(
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

    candidates: list[str] = []
    for candidate in [
        job_id,
        pipeline_status.get("stopped_job_id")
        or pipeline_status.get("paused_job_id"),
        *(pipeline_status.get("active_track_ids") or []),
    ]:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    target_job_id = job_id
    requeued_count = 0
    for candidate in candidates:
        stopped_count = await count_stopped_job_docs(rag, candidate)
        if stopped_count > 0:
            target_job_id = candidate
            requeued_count = await requeue_stopped_job_docs(rag, candidate)
            break

    if requeued_count == 0:
        return IngestionControlResponse(
            status="not_stopped",
            message="Ingestion job has no stopped documents to start over.",
            job_id=job_id,
        )

    async with pipeline_status_lock:
        pipeline_status["stop_requested"] = False
        pipeline_status["stopped"] = False
        pipeline_status["stopped_job_id"] = None
        if pipeline_status.get("busy", False):
            pipeline_status["request_pending"] = True
            return IngestionControlResponse(
                status="already_running",
                message=(
                    "Pipeline is already running; start-over request has been queued."
                ),
                job_id=target_job_id,
            )

    background_tasks.add_task(rag.apipeline_process_enqueue_documents)
    return IngestionControlResponse(
        status="start_over_started",
        message=(
            f"Start Over started for {requeued_count} stopped document(s); "
            "each will be reprocessed from the beginning."
        ),
        job_id=target_job_id,
    )


def create_ingestion_routes(rag: LightRAG, api_key: Optional[str] = None) -> APIRouter:
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "/stop/{job_id}",
        response_model=IngestionControlResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def stop_ingestion(job_id: str) -> IngestionControlResponse:
        return await request_ingestion_stop(rag, job_id)

    @router.post(
        "/start-over/{job_id}",
        response_model=IngestionControlResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def start_over_ingestion(
        job_id: str, background_tasks: BackgroundTasks
    ) -> IngestionControlResponse:
        return await start_over_ingestion_job(rag, job_id, background_tasks)

    return router

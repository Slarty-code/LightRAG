"""Offline tests for ingestion pause propagation into PARSE and ANALYZE.

Mirrors ``tests/pipeline/test_pipeline_cancellation.py`` for the cooperative
pause contract:

* ``_parse_worker`` and ``_analyze_worker`` check ``pause_requested`` at the
  top of every loop iteration, drain queued items as PAUSED with a
  ``"User paused during {stage}: ..."`` ``error_msg``, and ``task_done()``
  each one so ``q.join()`` in ``_run_pipeline_batch`` returns.
* ``analyze_multimodal`` fails fast on ``pause_requested`` (pre-schedule and
  in-flight poll) and re-raises :class:`PipelinePausedException`.
* ``_finalize_doc_failure`` writes ``DocStatus.PAUSED`` for pause exceptions
  at the PROCESS stage.

Tests construct ``_BatchRunContext`` and call worker methods directly to
avoid the cross-task races inherent in driving the full
``apipeline_process_enqueue_documents`` entry point.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.exceptions import PipelinePausedException
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
from lightrag.pipeline import _BatchRunContext
from lightrag.parser.registry import parser_specs_snapshot
from lightrag.utils import EmbeddingFunc, Tokenizer


pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 8)


async def _noop_llm(prompt, **kwargs):  # pragma: no cover - never invoked
    return ""


def _build_rag(tmp_path: Path, *, vlm_func=None) -> LightRAG:
    role_configs = {}
    for spec in ROLES:
        if spec.name == "vlm" and vlm_func is not None:
            role_configs[spec.name] = RoleLLMConfig(func=vlm_func)
        else:
            role_configs[spec.name] = RoleLLMConfig()
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"pause-{tmp_path.name}",
        llm_model_func=vlm_func or _noop_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=1024,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        vlm_process_enable=True,
        role_llm_configs=role_configs,
    )


async def _shutdown_role_workers(rag: LightRAG) -> None:
    for func in rag.role_llm_funcs.values():
        try:
            await rag._shutdown_llm_wrapper(func)
        except Exception as exc:
            logging.getLogger("lightrag").warning(
                f"role worker shutdown raised during test teardown: {exc}"
            )


async def _make_ctx(rag: LightRAG) -> tuple[_BatchRunContext, dict, Any]:
    pipeline_status = await get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status.clear()
    pipeline_status.update(
        {
            "busy": True,
            "history_messages": [],
            "latest_message": "",
            "cancellation_requested": False,
            "pause_requested": False,
            "paused": False,
        }
    )
    ctx = _BatchRunContext(
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        semaphore=asyncio.Semaphore(2),
        total_files=0,
        parse_queues={
            "native": asyncio.Queue(),
            "mineru": asyncio.Queue(),
            "docling": asyncio.Queue(),
        },
        parser_specs=parser_specs_snapshot(),
        q_analyze=asyncio.Queue(),
        q_process=asyncio.Queue(),
    )
    return ctx, pipeline_status, pipeline_status_lock


def _make_status_doc(doc_id: str) -> DocProcessingStatus:
    now = datetime.now(timezone.utc).isoformat()
    return DocProcessingStatus(
        content_summary=f"summary-{doc_id}",
        content_length=10,
        file_path=f"{doc_id}.pdf",
        status=DocStatus.PENDING,
        created_at=now,
        updated_at=now,
        track_id=None,
        content_hash=f"hash-{doc_id}",
    )


async def _run_worker_until_drained(
    worker_coro_factory,
    queue: asyncio.Queue,
    *,
    timeout: float = 2.0,
) -> None:
    worker = asyncio.create_task(worker_coro_factory())
    try:
        await asyncio.wait_for(queue.join(), timeout=timeout)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_parse_worker_drains_queue_when_paused_before_start(
    tmp_path, monkeypatch
):
    """Pause set BEFORE the worker pulls any item: parser must not run,
    every queued doc is PAUSED with a friendly message, q.join() returns."""
    rag = _build_rag(tmp_path)
    await rag.initialize_storages()
    try:
        ctx, pipeline_status, _ = await _make_ctx(rag)

        get_parser_spy = Mock(side_effect=AssertionError("parser must not be resolved"))
        monkeypatch.setattr("lightrag.pipeline.get_parser", get_parser_spy)

        for i in range(3):
            doc_id = f"doc-{i}"
            await rag.full_docs.upsert(
                {doc_id: {"content": "hello", "file_path": f"{doc_id}.pdf"}}
            )
            await rag.doc_status.upsert(
                {
                    doc_id: {
                        "status": DocStatus.PENDING.value,
                        "content_summary": f"sum-{doc_id}",
                        "content_length": 5,
                        "file_path": f"{doc_id}.pdf",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "track_id": "t",
                    }
                }
            )
            await ctx.parse_queues["native"].put((doc_id, _make_status_doc(doc_id)))

        pipeline_status["pause_requested"] = True

        start = time.monotonic()
        await _run_worker_until_drained(
            lambda: rag._parse_worker("native", ctx.parse_queues["native"], ctx),
            ctx.parse_queues["native"],
        )
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"queue drain should be fast, took {elapsed:.2f}s"
        assert get_parser_spy.call_count == 0

        pause_messages = [
            m
            for m in pipeline_status["history_messages"]
            if "User paused during parse" in m
        ]
        assert len(pause_messages) == 3

        for i in range(3):
            doc_id = f"doc-{i}"
            row = await rag.doc_status.get_by_id(doc_id)
            assert row is not None
            assert row.get("status") == DocStatus.PAUSED.value
            assert "User paused during parse" in (row.get("error_msg") or "")
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_analyze_worker_drains_queue_when_paused_before_start(tmp_path):
    """ANALYZE-worker symmetric to the PARSE pause test above."""
    rag = _build_rag(tmp_path)
    await rag.initialize_storages()
    try:
        ctx, pipeline_status, _ = await _make_ctx(rag)

        rag.analyze_multimodal = AsyncMock(
            side_effect=AssertionError("analyze_multimodal must not be called")
        )

        for i in range(3):
            doc_id = f"doc-{i}"
            await rag.doc_status.upsert(
                {
                    doc_id: {
                        "status": DocStatus.ANALYZING.value,
                        "content_summary": f"sum-{doc_id}",
                        "content_length": 5,
                        "file_path": f"{doc_id}.pdf",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "track_id": "t",
                    }
                }
            )
            await ctx.q_analyze.put(
                (doc_id, _make_status_doc(doc_id), {"content": "x"})
            )

        pipeline_status["pause_requested"] = True

        start = time.monotonic()
        await _run_worker_until_drained(
            lambda: rag._analyze_worker(ctx),
            ctx.q_analyze,
        )
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"queue drain should be fast, took {elapsed:.2f}s"
        assert rag.analyze_multimodal.await_count == 0

        pause_messages = [
            m
            for m in pipeline_status["history_messages"]
            if "User paused during analyze" in m
        ]
        assert len(pause_messages) == 3

        for i in range(3):
            row = await rag.doc_status.get_by_id(f"doc-{i}")
            assert row is not None
            assert row.get("status") == DocStatus.PAUSED.value
            assert "User paused during analyze" in (row.get("error_msg") or "")
    finally:
        await rag.finalize_storages()


def _write_three_item_sidecar(tmp_path: Path) -> tuple[str, dict, Path]:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir(exist_ok=True)
    blocks_path = parsed_dir / "doc.blocks.jsonl"
    blocks_path.write_text(
        json.dumps({"type": "meta", "doc_id": "doc-1"}) + "\n",
        encoding="utf-8",
    )
    sidecar_path = parsed_dir / "doc.drawings.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "drawings": {
                    "im-A": {"caption": "A", "path": "ignored-A"},
                    "im-B": {"caption": "B", "path": "ignored-B"},
                    "im-C": {"caption": "C", "path": "ignored-C"},
                }
            }
        ),
        encoding="utf-8",
    )
    parsed_data = {"blocks_path": str(blocks_path)}
    return "doc-1", parsed_data, sidecar_path


@pytest.mark.asyncio
async def test_analyze_multimodal_inflight_pause_polls_flag(tmp_path):
    """User sets pause_requested while VLM tasks are running.
    analyze_multimodal should observe the flag at the next poll boundary,
    cancel pending tasks, write the sidecar with partial results, and raise
    PipelinePausedException."""

    async def slow_vlm(prompt, **kwargs):
        await asyncio.sleep(1.2)
        return json.dumps(
            {"name": "x", "type": "Chart", "description": "should not arrive"}
        )

    rag = _build_rag(tmp_path, vlm_func=slow_vlm)
    await rag.initialize_storages()
    try:
        doc_id, parsed_data, sidecar_path = _write_three_item_sidecar(tmp_path)

        from .test_pipeline_analyze_multimodal import PNG_BYTES

        for letter in ("A", "B", "C"):
            (tmp_path / "parsed" / f"im-{letter}.png").write_bytes(PNG_BYTES)
        sidecar_path.write_text(
            json.dumps(
                {
                    "drawings": {
                        f"im-{letter}": {
                            "caption": letter,
                            "path": str(tmp_path / "parsed" / f"im-{letter}.png"),
                        }
                        for letter in ("A", "B", "C")
                    }
                }
            ),
            encoding="utf-8",
        )

        pipeline_status: dict = {
            "busy": True,
            "history_messages": [],
            "latest_message": "",
            "pause_requested": False,
        }
        pipeline_status_lock = asyncio.Lock()

        async def flip_after(delay: float):
            await asyncio.sleep(delay)
            async with pipeline_status_lock:
                pipeline_status["pause_requested"] = True

        flipper = asyncio.create_task(flip_after(0.1))

        start = time.monotonic()
        with pytest.raises(PipelinePausedException):
            await asyncio.wait_for(
                rag.analyze_multimodal(
                    doc_id=doc_id,
                    file_path="fixture.pdf",
                    parsed_data=parsed_data,
                    process_options="i",
                    pipeline_status=pipeline_status,
                    pipeline_status_lock=pipeline_status_lock,
                ),
                timeout=15.0,
            )
        elapsed = time.monotonic() - start
        await flipper

        assert elapsed < 1.0, f"in-flight pause took {elapsed:.2f}s (>1.0s)"

        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        for letter in ("A", "B", "C"):
            item = payload["drawings"][f"im-{letter}"]
            assert "llm_analyze_result" in item
            assert item["llm_analyze_result"]["status"] in ("failure", "success")
    finally:
        await _shutdown_role_workers(rag)
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_analyze_multimodal_pre_schedule_pause_skips_task_creation(
    tmp_path, monkeypatch
):
    """``pause_requested`` is already True when analyze_multimodal enters the
    sidecar processing loop. The pre-schedule check must raise immediately,
    before any per-item VLM task is constructed."""
    from .test_pipeline_analyze_multimodal import PNG_BYTES

    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    image_path = parsed_dir / "im-X.png"
    image_path.write_bytes(PNG_BYTES)
    blocks_path = parsed_dir / "doc.blocks.jsonl"
    blocks_path.write_text(
        json.dumps({"type": "meta", "doc_id": "doc-1"}) + "\n",
        encoding="utf-8",
    )
    sidecar_path = parsed_dir / "doc.drawings.json"
    sidecar_path.write_text(
        json.dumps({"drawings": {"im-X": {"caption": "X", "path": str(image_path)}}}),
        encoding="utf-8",
    )
    parsed_data = {"blocks_path": str(blocks_path)}

    vlm_invocations = 0

    async def tripwire_vlm(prompt, **kwargs):
        nonlocal vlm_invocations
        vlm_invocations += 1
        return json.dumps(
            {"name": "X", "type": "Chart", "description": "must not be called"}
        )

    progress_log_tasks_created = 0
    original_create_task = asyncio.create_task

    def spy_create_task(coro, *args, **kwargs):
        nonlocal progress_log_tasks_created
        name = getattr(coro, "__qualname__", "") or getattr(
            getattr(coro, "cr_code", None), "co_qualname", ""
        )
        if "_run_with_progress_log" in name:
            progress_log_tasks_created += 1
        return original_create_task(coro, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", spy_create_task)

    rag = _build_rag(tmp_path, vlm_func=tripwire_vlm)
    await rag.initialize_storages()
    try:
        pipeline_status: dict = {
            "busy": True,
            "history_messages": [],
            "latest_message": "",
            "pause_requested": True,
        }
        pipeline_status_lock = asyncio.Lock()

        with pytest.raises(PipelinePausedException):
            await rag.analyze_multimodal(
                doc_id="doc-1",
                file_path="fixture.pdf",
                parsed_data=parsed_data,
                process_options="i",
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
            )

        assert progress_log_tasks_created == 0
        assert vlm_invocations == 0
    finally:
        await _shutdown_role_workers(rag)
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_finalize_doc_failure_marks_paused_at_process_stage(tmp_path):
    """PROCESS-stage epilogue must persist PAUSED (not FAILED) for pause."""
    rag = _build_rag(tmp_path)
    await rag.initialize_storages()
    try:
        ctx, pipeline_status, pipeline_status_lock = await _make_ctx(rag)
        doc_id = "doc-pause"
        status_doc = _make_status_doc(doc_id)
        await rag.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.PROCESSING.value,
                    "content_summary": status_doc.content_summary,
                    "content_length": status_doc.content_length,
                    "file_path": status_doc.file_path,
                    "created_at": status_doc.created_at,
                    "updated_at": status_doc.updated_at,
                    "track_id": "job-1",
                }
            }
        )

        await rag._finalize_doc_failure(
            doc_id=doc_id,
            status_doc=status_doc,
            file_path=status_doc.file_path,
            error=PipelinePausedException("User paused during entity extraction"),
            stage_label="extract",
            current_file_number=1,
            total_files=1,
            failed_chunks_snapshot=(["chunk-1"], 1),
            pending_tasks=[],
            metadata_extra={"process_start_time": int(time.time())},
            pipeline_status=ctx.pipeline_status,
            pipeline_status_lock=ctx.pipeline_status_lock,
        )

        row = await rag.doc_status.get_by_id(doc_id)
        assert row is not None
        assert row.get("status") == DocStatus.PAUSED.value
        assert "User paused" in (row.get("error_msg") or "")
        assert row.get("chunks_list") == ["chunk-1"]
        assert row.get("chunks_count") == 1

        pause_messages = [
            m
            for m in pipeline_status["history_messages"]
            if "User paused" in m
        ]
        assert pause_messages
    finally:
        await rag.finalize_storages()

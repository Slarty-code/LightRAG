"""Offline tests for pipeline chunk progress helpers."""

import pytest

from lightrag.kg.shared_storage import (
    clear_pipeline_chunk_progress,
    increment_pipeline_chunk_progress,
    register_pipeline_chunk_progress,
    sync_pipeline_chunk_aggregates,
    unregister_pipeline_chunk_progress,
)


@pytest.mark.offline
def test_chunk_progress_lifecycle():
    ps: dict = {}
    register_pipeline_chunk_progress(ps, "doc-a", 10)
    assert ps["chunks_total"] == 10
    assert ps["chunks_done"] == 0
    assert ps["current_doc_id"] == "doc-a"
    for _ in range(3):
        increment_pipeline_chunk_progress(ps, "doc-a")
    assert ps["chunks_done"] == 3
    unregister_pipeline_chunk_progress(ps, "doc-a")
    assert ps["chunks_total"] == 0
    assert ps["current_doc_id"] is None


@pytest.mark.offline
def test_chunk_progress_parallel_docs():
    ps: dict = {}
    register_pipeline_chunk_progress(ps, "doc-a", 5)
    register_pipeline_chunk_progress(ps, "doc-b", 3)
    assert ps["chunks_total"] == 8
    increment_pipeline_chunk_progress(ps, "doc-a")
    increment_pipeline_chunk_progress(ps, "doc-b")
    assert ps["chunks_done"] == 2
    unregister_pipeline_chunk_progress(ps, "doc-a")
    assert ps["chunks_total"] == 3
    assert ps["chunks_done"] == 1
    unregister_pipeline_chunk_progress(ps, "doc-b")
    assert ps["chunks_total"] == 0


@pytest.mark.offline
def test_clear_pipeline_chunk_progress():
    ps: dict = {}
    register_pipeline_chunk_progress(ps, "doc-a", 4)
    clear_pipeline_chunk_progress(ps)
    assert ps["chunks_total"] == 0
    assert ps["chunks_done"] == 0
    assert ps["chunk_progress"] == {}
    assert ps["current_doc_id"] is None


@pytest.mark.offline
def test_sync_pipeline_chunk_aggregates_empty():
    ps: dict = {"chunk_progress": {}}
    sync_pipeline_chunk_aggregates(ps)
    assert ps["chunks_total"] == 0
    assert ps["chunks_done"] == 0

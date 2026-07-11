from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, OpenAI

from .data_utils import getenv_required, save_json, save_jsonl


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=getenv_required("OPENAI_API_KEY"))


def create_batch_input(path: str | Path, requests: list[dict[str, Any]]) -> Path:
    save_jsonl(path, requests)
    return Path(path)


def submit_batch(input_file: str | Path, endpoint: str = "/v1/responses", completion_window: str = "24h") -> dict[str, Any]:
    client = get_openai_client()
    input_path = Path(input_file)
    print(f"[batch_api] Uploading batch input file: {input_path}")
    with input_path.open("rb") as fp:
        uploaded = client.files.create(file=fp, purpose="batch")

    print(f"[batch_api] Uploaded input file id: {uploaded.id}")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=endpoint,
        completion_window=completion_window,
    )
    print(f"[batch_api] Submitted batch: {batch.id} status={batch.status}")
    return {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "status": batch.status,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }


def download_file_with_retry(
    client: OpenAI,
    file_id: str,
    output_path: str | Path,
    *,
    label: str,
    max_retries: int = 8,
    retry_delay_seconds: int = 10,
    chunk_size: int = 1024 * 1024,
) -> Path:
    target = Path(output_path)
    temp_path = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None
    retryable_errors = (APIConnectionError, httpx.RemoteProtocolError, httpx.ReadError, httpx.TimeoutException)

    for attempt in range(1, max_retries + 1):
        try:
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[batch_api] Downloading {label} file: {file_id} attempt={attempt}/{max_retries}")
            with client.files.with_streaming_response.content(file_id) as response:
                response.stream_to_file(temp_path, chunk_size=chunk_size)
            temp_path.replace(target)
            print(f"[batch_api] Saved {label} file to {target}")
            return target
        except retryable_errors as exc:
            last_error = exc
            if temp_path.exists():
                temp_path.unlink()
            if attempt == max_retries:
                break
            print(
                f"[batch_api] Connection error while downloading {label} file {file_id}. "
                f"Retrying in {retry_delay_seconds}s..."
            )
            time.sleep(retry_delay_seconds)

    assert last_error is not None
    raise last_error


def wait_for_batch(batch_id: str, poll_seconds: int = 60, output_path: str | Path | None = None, error_path: str | Path | None = None) -> dict[str, Any]:
    client = get_openai_client()
    print(f"[batch_api] Waiting for batch {batch_id} with poll interval {poll_seconds}s")

    while True:
        batch = client.batches.retrieve(batch_id)
        request_counts = getattr(batch, "request_counts", None)
        counts_text = ""
        if request_counts is not None:
            completed = getattr(request_counts, "completed", None)
            failed = getattr(request_counts, "failed", None)
            total = getattr(request_counts, "total", None)
            counts_text = f" completed={completed} failed={failed} total={total}"
        print(f"[batch_api] Poll status: {batch.status}{counts_text}")
        if batch.status in {"completed", "failed", "cancelled", "expired"}:
            break
        time.sleep(poll_seconds)

    result = {
        "batch_id": batch.id,
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }

    if batch.output_file_id and output_path is not None:
        download_file_with_retry(client, batch.output_file_id, output_path, label="batch output")
        result["output_file"] = str(output_path)

    if batch.error_file_id and error_path is not None:
        download_file_with_retry(client, batch.error_file_id, error_path, label="batch error")
        result["error_file"] = str(error_path)

    return result


def _extract_text_from_response_body(body: dict[str, Any]) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output_items = body.get("output")
    if isinstance(output_items, list):
        texts: list[str] = []
        for item in output_items:
            if not isinstance(item, dict):
                continue
            content_items = item.get("content")
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                if not isinstance(content, dict):
                    continue
                text_value = content.get("text") or content.get("output_text")
                if isinstance(text_value, str) and text_value.strip():
                    texts.append(text_value.strip())
        if texts:
            return "\n".join(texts)

    return ""


def parse_batch_output(output_file: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(output_file)
    results: dict[str, dict[str, Any]] = {}
    print(f"[batch_api] Parsing batch output: {path}")

    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            custom_id = str(row.get("custom_id"))
            response = row.get("response") or {}
            body = response.get("body") or {}
            error = row.get("error")
            results[custom_id] = {
                "custom_id": custom_id,
                "status_code": response.get("status_code"),
                "text": _extract_text_from_response_body(body),
                "raw_body": body,
                "error": error,
            }
    print(f"[batch_api] Parsed {len(results)} batch results")
    return results


def save_batch_metadata(path: str | Path, payload: dict[str, Any]) -> Path:
    save_json(path, payload)
    return Path(path)

import base64
import json
import mimetypes
import os
import time
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image


def setup_client(env_file):
    load_dotenv(env_file)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(f"GOOGLE_API_KEY not found. Checked env file: {env_file}")
    return genai.Client(api_key=api_key)


def guess_mime_type(path):
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def create_batch_request_file(output_dir, filename, requests):
    batch_file = Path(output_dir) / filename
    write_jsonl(batch_file, requests)
    return batch_file


def upload_batch_file(client, batch_file, display_name):
    return client.files.upload(
        file=str(batch_file),
        config=types.UploadFileConfig(
            display_name=display_name,
            mime_type="application/jsonl",
        ),
    )


def submit_batch_job(client, batch_file, model_name, job_name):
    uploaded_file = upload_batch_file(client, batch_file, display_name=job_name)
    batch_job = client.batches.create(
        model=model_name,
        src=uploaded_file.name,
        config={"display_name": job_name},
    )
    return uploaded_file, batch_job


def wait_for_batch_job(client, job_name, poll_interval=60, max_wait_hours=25):
    completed_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    deadline = time.time() + max_wait_hours * 3600
    while time.time() < deadline:
        batch_job = client.batches.get(name=job_name)
        state = batch_job.state.name
        print(f"Batch state: {state}")
        if state in completed_states:
            return batch_job
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for batch job: {job_name}")


def download_batch_output(client, batch_job, output_path):
    if not batch_job.dest or not batch_job.dest.file_name:
        raise ValueError(f"Batch job has no output file: {batch_job.name}")
    output_bytes = client.files.download(file=batch_job.dest.file_name)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        file.write(output_bytes)
    return output_path


def upload_image(client, image_path):
    return client.files.upload(file=str(image_path))


def make_file_part(uploaded_file, image_path=None):
    file_uri = getattr(uploaded_file, "uri", None)
    mime_type = getattr(uploaded_file, "mime_type", None) or guess_mime_type(image_path or "")
    if not file_uri:
        raise ValueError("Uploaded Gemini file is missing uri")
    return {
        "file_data": {
            "file_uri": file_uri,
            "mime_type": mime_type,
        }
    }


def extract_text_from_response_row(row):
    response = row.get("response") or {}
    candidates = response.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text")
            if text:
                return str(text).strip()
    return ""


def extract_inline_images_from_response_row(row):
    response = row.get("response") or {}
    candidates = response.get("candidates") or []
    image_payloads = []
    for candidate in candidates:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            inline_data = part.get("inlineData") or part.get("inline_data")
            if inline_data and inline_data.get("data"):
                image_payloads.append(inline_data)
    return list(reversed(image_payloads))


def save_images_from_response_row(row, images_dir):
    key = row.get("key", "")
    if not key.startswith("article_"):
        return []
    article_id = key.replace("article_", "", 1)
    saved_paths = []
    for save_idx, inline_data in enumerate(extract_inline_images_from_response_row(row)):
        image_bytes = base64.b64decode(inline_data["data"])
        image = Image.open(BytesIO(image_bytes))
        image_path = Path(images_dir) / f"id_{article_id}_image_{save_idx}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        image.close()
        saved_paths.append(str(image_path))
    return saved_paths

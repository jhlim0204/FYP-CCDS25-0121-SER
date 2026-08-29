"""HTTP wrapper around the SER inference pipeline.

Exposes a single POST /detect endpoint that mirrors the ots-vad / ots-lid /
ots-sed family (multipart upload in, JSON out). Internally it shells out to
``run_inference.sh``, which drives the two-stage, two-conda-env pipeline
(audio preprocessing in venv_audio, then the DeepSpeed LLM eval in venv_llm).

Run it::

    uvicorn api_endpoint:app --host 0.0.0.0 --port 8000
"""

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

CLASSIFIER_ID = "ser_speechcuellm_v1"
MODEL_VERSION = "meta-llama/Meta-Llama-3-8B-Instruct+lora"

SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_SCRIPT = SCRIPT_DIR / "run_inference.sh"
SUPPORTED_DATASETS = ("iemocap", "msp")
DEFAULT_DATASET = "iemocap"  # used when the caller omits `dataset` (only iemocap in use for now)

# The pipeline is single-flight: fixed CUDA device + DeepSpeed master port.
_pipeline_lock = threading.Lock()


class EmotionPrediction(BaseModel):
    index: int
    prediction: str
    llm_input: Optional[str] = None


class SerResult(BaseModel):
    audio_file_id: str
    classifier_id: str
    model_version: str
    run_id: str
    event_type: str
    dataset: str
    with_history: bool
    predictions: List[EmotionPrediction]
    created_at: str


def _parse_preds_file(text_path: Path) -> List[EmotionPrediction]:
    """Extract the predictions out of a ``preds_for_eval_*.text`` file.

    The file written by LLM_code/main.py is not valid JSON as a whole: it
    starts with a ``score`` JSON object, then a text confusion matrix, a
    ``confuse_case:`` line, and finally the ``preds_for_eval`` JSON array. We
    only care about that trailing array (each item is
    ``{index, input, output, target}``); the leading metrics are meaningless
    for unlabeled single-clip inference.
    """
    raw = text_path.read_text(encoding="utf-8")

    # The predictions array is the last top-level JSON array in the file.
    # Find the last '[' that begins a well-formed array through end-of-string.
    last_open = raw.rfind("\n[")
    candidate = raw[last_open + 1:] if last_open != -1 else raw
    try:
        items = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"could not parse predictions array from {text_path.name}: {exc}",
        ) from exc

    if not isinstance(items, list):
        raise HTTPException(
            status_code=500,
            detail=f"expected a predictions array in {text_path.name}, got {type(items).__name__}",
        )

    predictions = []
    for item in items:
        predictions.append(
            EmotionPrediction(
                index=item.get("index", -1),
                prediction=str(item.get("output", "")).strip(),
                llm_input=item.get("input"),
            )
        )
    return predictions


def _locate_preds_file(final_dir: Path) -> Path:
    matches = sorted(glob.glob(str(final_dir / "preds_for_eval_*.text")))
    if not matches:
        raise HTTPException(
            status_code=500,
            detail=f"no preds_for_eval_*.text produced under {final_dir}",
        )
    # If multiple epochs were written, take the highest epoch number.
    def _epoch(path: str) -> int:
        m = re.search(r"preds_for_eval_(\d+)\.text$", path)
        return int(m.group(1)) if m else -1

    return Path(max(matches, key=_epoch))


app = FastAPI(title="Speech Emotion Recognition API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect", response_model=SerResult)
def detect(
    files: UploadFile = File(
        ...,
        description="audio file (single WAV). Field name is `files` to match the "
        "ots-vad / ots-lid / ots-sed multipart contract used by ots-pipeline.",
    ),
    dataset: Optional[str] = Query(
        None,
        description=(
            "which trained model/feature set/label space to use (iemocap or msp); "
            f"optional, defaults to {DEFAULT_DATASET!r} when omitted"
        ),
    ),
    with_history: bool = Query(
        False,
        description="include historical dialogue context in the LLM prompt",
    ),
    checkpoint_dir: Optional[str] = Query(
        None,
        description="override the checkpoint directory (defaults to ../checkpoints/<dataset>_checkpoints)",
    ),
):
    if dataset is None:
        dataset = DEFAULT_DATASET
    elif dataset not in SUPPORTED_DATASETS:
        raise HTTPException(
            status_code=422,
            detail=f"dataset must be one of {SUPPORTED_DATASETS}, got {dataset!r}",
        )
    if not INFERENCE_SCRIPT.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"inference script not found at {INFERENCE_SCRIPT}",
        )

    # The pipeline can only run one at a time (GPU 0 + fixed DeepSpeed port).
    if not _pipeline_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="pipeline is busy with another request; retry later",
        )

    run_id = f"{CLASSIFIER_ID}_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:3]}"
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    audio_file_id = files.filename or f"audio_{uuid.uuid4().hex[:8]}"

    work_dir = Path(tempfile.mkdtemp(prefix="ser_detect_"))
    suffix = os.path.splitext(files.filename)[1].lower() if files.filename else ".wav"
    if suffix != ".wav":
        # prepare_manifest only accepts .wav (or a .csv manifest); reject early.
        _pipeline_lock.release()
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(
            status_code=422,
            detail=f"only .wav uploads are supported, got {suffix!r}",
        )

    wav_path = work_dir / f"input{suffix}"
    output_dir = work_dir / "inference_outputs"

    try:
        with open(wav_path, "wb") as out_wav:
            out_wav.write(files.file.read())

        ckpt = checkpoint_dir or str(SCRIPT_DIR.parent / "checkpoints" / f"{dataset}_checkpoints")

        cmd = [
            "bash",
            str(INFERENCE_SCRIPT),
            dataset,
            str(wav_path),
            "True" if with_history else "False",
            ckpt,
            str(output_dir),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            raise HTTPException(
                status_code=500,
                detail=f"inference pipeline failed (exit {proc.returncode}):\n{tail}",
            )

        preds_file = _locate_preds_file(output_dir / "final_predictions")
        predictions = _parse_preds_file(preds_file)

        return SerResult(
            audio_file_id=audio_file_id,
            classifier_id=CLASSIFIER_ID,
            model_version=MODEL_VERSION,
            run_id=run_id,
            event_type="speech emotion recognition",
            dataset=dataset,
            with_history=with_history,
            predictions=predictions,
            created_at=created_at,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        _pipeline_lock.release()

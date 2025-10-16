import os
import json
import asyncio
import shutil
import uuid
import time
import wave
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel


def pick_device(device_arg: str) -> str:
    if device_arg and device_arg != "auto":
        return device_arg
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def pick_compute_type(compute_arg: str, device: str) -> str:
    if compute_arg and compute_arg != "auto":
        return compute_arg
    return "float16" if device == "cuda" else "int8"


MODEL_NAME = os.environ.get("ASR_MODEL", "small")
DEVICE = pick_device(os.environ.get("ASR_DEVICE", "auto"))
COMPUTE_TYPE = pick_compute_type(os.environ.get("ASR_COMPUTE_TYPE", "auto"), DEVICE)

app = FastAPI(title="ASR Streaming Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static web UI under /web to avoid intercepting /ws
if os.path.isdir("web"):
    app.mount("/web", StaticFiles(directory="web", html=True), name="web")


@app.get("/")
async def index():
    if os.path.isfile("web/index.html"):
        return FileResponse("web/index.html")
    return {"message": "ASR Streaming Service", "web": "/web/", "ws": "/ws"}


# Load model once
MODEL: Optional[WhisperModel] = None


@app.on_event("startup")
async def startup_event():
    global MODEL
    MODEL = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)


def transcribe_buffer(audio_float32: np.ndarray, task: str = "transcribe", language: Optional[str] = None) -> List[Dict[str, Any]]:
    segments, info = MODEL.transcribe(
        audio_float32,
        task=task,
        language=language,
        vad_filter=False,
        beam_size=5,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    out = []
    for i, s in enumerate(segments):
        out.append({
            "id": i,
            "start": float(s.start or 0.0),
            "end": float(s.end or 0.0),
            "text": (s.text or "").strip(),
        })
    return out


@app.websocket("/ws")
async def websocket_stream(ws: WebSocket):
    await ws.accept()
    # session & logging
    session_id = uuid.uuid4().hex
    log_dir = os.environ.get("ASR_LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{session_id}.jsonl")
    log_fp = open(log_path, "a", encoding="utf-8")
    # input audio saving setup
    input_dir = os.environ.get("ASR_INPUT_DIR", "inputs")
    os.makedirs(input_dir, exist_ok=True)
    input_path = os.path.join(input_dir, f"{session_id}.wav")
    wave_writer = None

    def log_event(ev: Dict[str, Any]):
        try:
            ev = dict(ev)
            ev.setdefault("session_id", session_id)
            ev.setdefault("ts", time.time())
            log_fp.write(json.dumps(ev, ensure_ascii=False) + "\n")
            log_fp.flush()
        except Exception:
            pass
    # session state
    sample_rate = 16000
    task = "transcribe"
    language: Optional[str] = None
    min_chunk_sec = 3.0
    max_buffer_sec = 30.0
    context_sec = 1.0
    buffer = bytearray()
    buffer_start_offset_samples = 0  # trimmed head in samples
    last_sent_end_sec = 0.0

    await ws.send_json({
        "type": "ready",
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "sample_rate": sample_rate,
        "session_id": session_id,
        "log_path": log_path,
        "input_path": input_path,
    })
    # write ready event
    client = getattr(ws, "client", None)
    client_host = getattr(client, "host", None)
    client_port = getattr(client, "port", None)
    log_event({
        "type": "ready",
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "sample_rate": sample_rate,
        "client_host": client_host,
        "client_port": client_port,
        "input_path": input_path,
    })

    try:
        while True:
            # Receive any message (text control or binary audio)
            try:
                message = await asyncio.wait_for(ws.receive(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

            # Starlette websocket messages
            mtype = message.get("type")
            if mtype == "websocket.receive":
                if "text" in message and message["text"] is not None:
                    # control message
                    try:
                        msg = json.loads(message["text"])
                    except Exception:
                        msg = {"type": "text", "data": message["text"]}
                    if isinstance(msg, dict) and msg.get("type") == "start":
                        sample_rate = int(msg.get("sample_rate", sample_rate))
                        task = msg.get("task", task)
                        language = msg.get("language")
                        await ws.send_json({
                            "type": "ack",
                            "message": "start received",
                            "sample_rate": sample_rate,
                            "task": task,
                            "language": language,
                            "input_path": input_path,
                        })
                        log_event({
                            "type": "start",
                            "sample_rate": sample_rate,
                            "task": task,
                            "language": language,
                        })
                        # open wav writer for input audio
                        if wave_writer is None:
                            try:
                                ww = wave.open(input_path, "wb")
                                ww.setnchannels(1)
                                ww.setsampwidth(2)
                                ww.setframerate(sample_rate)
                                wave_writer = ww
                            except Exception:
                                wave_writer = None
                        continue
                    elif isinstance(msg, dict) and msg.get("type") == "stop":
                        # final decode
                        if len(buffer) >= 2:
                            audio = np.frombuffer(buffer, dtype=np.int16).copy()
                            audio_f = audio.astype(np.float32) / 32768.0
                            segs = transcribe_buffer(audio_f, task=task, language=language)
                            # add offset
                            offset_sec = buffer_start_offset_samples / sample_rate
                            for s in segs:
                                s["start"] += offset_sec
                                s["end"] += offset_sec
                            # send only new segments
                            new_segs = [s for s in segs if s["end"] > last_sent_end_sec + 1e-3]
                            if new_segs:
                                await ws.send_json({"type": "segments", "segments": new_segs, "final": True})
                                log_event({"type": "segments", "segments": new_segs, "final": True})
                        await ws.send_json({"type": "done"})
                        log_event({"type": "done"})
                        await ws.close()
                        return
                    else:
                        # ignore unknown text
                        pass
                elif "bytes" in message and message["bytes"] is not None:
                    # binary audio chunk (PCM16LE)
                    chunk = message["bytes"]
                    # write PCM16 chunks into session wav
                    if wave_writer is None:
                        try:
                            ww = wave.open(input_path, "wb")
                            ww.setnchannels(1)
                            ww.setsampwidth(2)
                            ww.setframerate(sample_rate)
                            wave_writer = ww
                        except Exception:
                            wave_writer = None
                    try:
                        if wave_writer is not None:
                            wave_writer.writeframes(chunk)
                    except Exception:
                        pass
                    buffer.extend(chunk)

                    # when enough audio accumulated, run decode
                    samples = len(buffer) // 2
                    if samples >= int(min_chunk_sec * sample_rate):
                        audio = np.frombuffer(buffer, dtype=np.int16).copy()
                        audio_f = audio.astype(np.float32) / 32768.0
                        segs = transcribe_buffer(audio_f, task=task, language=language)

                        offset_sec = buffer_start_offset_samples / sample_rate
                        # shift by offset
                        for s in segs:
                            s["start"] += offset_sec
                            s["end"] += offset_sec

                        # send only new segments
                        new_segs = [s for s in segs if s["end"] > last_sent_end_sec + 1e-3]
                        if new_segs:
                            await ws.send_json({"type": "segments", "segments": new_segs})
                            last_sent_end_sec = max(last_sent_end_sec, max(s["end"] for s in new_segs))
                            log_event({"type": "segments", "segments": new_segs})

                        # trim buffer to keep tail for context
                        keep_from_sec = max(last_sent_end_sec - context_sec, 0.0)
                        keep_from_samples = int(keep_from_sec * sample_rate)
                        if keep_from_sec > 0:
                            # compute relative index within buffer
                            rel_index = max(keep_from_samples - buffer_start_offset_samples, 0)
                            rel_index_bytes = rel_index * 2
                            buffer = buffer[rel_index_bytes:]
                            buffer_start_offset_samples = buffer_start_offset_samples + rel_index

                        # avoid very large buffer
                        total_sec = (len(buffer) // 2) / sample_rate
                        if total_sec > max_buffer_sec:
                            cut_samples = (len(buffer) // 2) - int(max_buffer_sec * sample_rate)
                            if cut_samples > 0:
                                buffer = buffer[cut_samples * 2:]
                                buffer_start_offset_samples += cut_samples
                else:
                    # other receive message without text/bytes
                    pass
            elif mtype == "websocket.disconnect":
                log_event({"type": "disconnect"})
                try:
                    if wave_writer is not None:
                        wave_writer.close()
                        wave_writer = None
                except Exception:
                    pass
                break
            else:
                # ping/pong etc.
                pass
    except WebSocketDisconnect:
        log_event({"type": "disconnect"})
        try:
            if wave_writer is not None:
                wave_writer.close()
                wave_writer = None
        except Exception:
            pass
        return
    finally:
        try:
            if wave_writer is not None:
                wave_writer.close()
        except Exception:
            pass
        try:
            log_fp.close()
        except Exception:
            pass


@app.post("/api/transcribe")
async def transcribe_file(file: UploadFile = File(...), task: str = "transcribe", language: Optional[str] = None):
    suffix = os.path.splitext(file.filename or "audio")[1]
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        segments, info = MODEL.transcribe(tmp_path, task=task, language=language, vad_filter=True)
        segs = [{
            "id": i,
            "start": float(s.start or 0.0),
            "end": float(s.end or 0.0),
            "text": (s.text or "").strip(),
        } for i, s in enumerate(segments)]
        return {
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
            "segments": segs,
            "model": MODEL_NAME,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
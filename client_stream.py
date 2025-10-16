#!/usr/bin/env python3
import argparse
import asyncio
import json
import wave
from pathlib import Path

import numpy as np
import websockets


def load_wav_resample_to_16k_pcm(path: str, target_sr: int = 16000) -> bytes:
    with wave.open(path, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        n_frames = w.getnframes()
        frames = w.readframes(n_frames)

    if sampwidth != 2:
        raise RuntimeError(f"仅支持 16-bit PCM WAV，检测到 sampwidth={sampwidth}")

    pcm16 = np.frombuffer(frames, dtype=np.int16)
    if n_channels > 1:
        pcm16 = pcm16.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

    # Convert to float32 and resample if needed
    audio = pcm16.astype(np.float32) / 32768.0
    if framerate != target_sr:
        # Linear interpolation resample
        duration = audio.shape[0] / framerate
        x_old = np.linspace(0, duration, num=audio.shape[0], endpoint=False)
        new_length = int(duration * target_sr)
        x_new = np.linspace(0, duration, num=new_length, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)

    # back to PCM16 bytes
    audio = np.clip(audio, -1.0, 1.0)
    pcm16_out = (audio * 32767.0).astype(np.int16).tobytes()
    return pcm16_out


async def stream_file(ws_url: str, wav_path: str, chunk_ms: int = 200):
    pcm = load_wav_resample_to_16k_pcm(wav_path)
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "sample_rate": 16000}))
        bytes_per_sample = 2
        samples_total = len(pcm) // bytes_per_sample
        samples_per_chunk = int(16000 * (chunk_ms / 1000.0))
        for i in range(0, samples_total, samples_per_chunk):
            chunk = pcm[i * bytes_per_sample:(i + samples_per_chunk) * bytes_per_sample]
            await ws.send(chunk)
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.05)
                print("[recv]", msg)
            except asyncio.TimeoutError:
                pass
        await ws.send(json.dumps({"type": "stop"}))
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                print("[recv]", msg)
        except asyncio.TimeoutError:
            pass


def main():
    ap = argparse.ArgumentParser(description="WebSocket 文件流式客户端")
    ap.add_argument("--url", default="ws://localhost:8000/ws", help="WebSocket 服务地址")
    ap.add_argument("--file", required=True, help="任意采样率的 PCM16 WAV 文件路径（自动重采样至16kHz）")
    ap.add_argument("--chunk-ms", type=int, default=200, help="发送分片大小，毫秒")
    args = ap.parse_args()

    if not Path(args.file).exists():
        raise SystemExit(f"文件不存在: {args.file}")

    asyncio.run(stream_file(args.url, args.file, args.chunk_ms))


if __name__ == "__main__":
    main()
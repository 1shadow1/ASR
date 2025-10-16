#!/usr/bin/env python3
import os
import sys
import json
import argparse
import shutil
from datetime import timedelta
from typing import List, Dict, Any

from tqdm import tqdm
from faster_whisper import WhisperModel


AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".mp4", ".webm"
}


def is_audio_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in AUDIO_EXTS


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def format_timestamp(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000))
    td = timedelta(seconds=int(seconds))
    # timedelta prints like HH:MM:SS
    hh, mm, ss = str(td).split(":")
    return f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d},{ms:03d}"


def write_srt(segments, out_path: str) -> None:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg.start)
        end = format_timestamp(seg.end)
        text = (seg.text or "").strip()
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_json(audio_path: str, info, segments, meta: Dict[str, Any], out_path: str) -> None:
    data = {
        "audio_path": audio_path,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "task": meta.get("task"),
        "model": meta.get("model"),
        "device": meta.get("device"),
        "compute_type": meta.get("compute_type"),
        "segments": [
            {
                "id": idx,
                "start": seg.start,
                "end": seg.end,
                "text": (seg.text or "").strip(),
            }
            for idx, seg in enumerate(segments)
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def collect_audio_files(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        files = []
        for root, _, names in os.walk(input_path):
            for n in names:
                p = os.path.join(root, n)
                if is_audio_file(p):
                    files.append(p)
        return sorted(files)
    elif is_audio_file(input_path):
        return [input_path]
    else:
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR CLI 基于 faster-whisper")
    parser.add_argument("input", help="音频文件路径，或包含音频文件的目录")
    parser.add_argument("--model", default="small", help="模型名称或路径，如 tiny/base/small/medium/large-v3")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="任务：转写或翻译为英文")
    parser.add_argument("--language", default=None, help="指定源语言（如 'zh'、'en'），留空自动检测")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="运行设备，默认自动检测")
    parser.add_argument("--compute-type", default="auto", help="计算类型，如 float16/int8/int8_float32，默认按设备自动选择")
    parser.add_argument("--output-dir", default="outputs", help="输出目录，默认 outputs/")
    parser.add_argument("--srt", action="store_true", help="导出 SRT 字幕")
    parser.add_argument("--json", action="store_true", help="导出 JSON 结果")
    parser.add_argument("--vad", action="store_true", help="启用 VAD 过滤提升分段质量")
    parser.add_argument("--chunk-length", type=int, default=30, help="分片长度（秒，整数），默认 30")
    return parser.parse_args()


def pick_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def pick_compute_type(compute_arg: str, device: str) -> str:
    if compute_arg != "auto":
        return compute_arg
    return "float16" if device == "cuda" else "int8"


def transcribe_one(model: WhisperModel, audio_path: str, args: argparse.Namespace, device: str, compute_type: str) -> Dict[str, Any]:
    segments, info = model.transcribe(
        audio_path,
        task=args.task,
        language=args.language,
        vad_filter=bool(args.vad),
        chunk_length=args.chunk_length,
    )

    base = os.path.splitext(os.path.basename(audio_path))[0]
    ensure_dir(args.output_dir)
    meta = {
        "task": args.task,
        "model": args.model,
        "device": device,
        "compute_type": compute_type,
    }

    srt_path = os.path.join(args.output_dir, f"{base}.srt")
    json_path = os.path.join(args.output_dir, f"{base}.json")

    seg_list = list(segments)
    if args.srt or (not args.srt and not args.json):
        write_srt(seg_list, srt_path)
    if args.json or (not args.srt and not args.json):
        write_json(audio_path, info, seg_list, meta, json_path)

    return {
        "audio": audio_path,
        "srt": srt_path if os.path.exists(srt_path) else None,
        "json": json_path if os.path.exists(json_path) else None,
        "language": getattr(info, "language", None),
        "prob": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
    }


def main() -> None:
    args = parse_args()

    files = collect_audio_files(args.input)
    if not files:
        print(f"未找到音频文件：{args.input}", file=sys.stderr)
        sys.exit(2)

    if not has_ffmpeg():
        print("警告：未检测到 ffmpeg，部分格式可能无法读取。请安装 ffmpeg。", file=sys.stderr)

    device = pick_device(args.device)
    compute_type = pick_compute_type(args.compute_type, device)

    print(f"加载模型：{args.model} | 设备：{device} | 计算：{compute_type}")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    results = []
    for f in tqdm(files, desc="转写中", unit="file"):
        try:
            res = transcribe_one(model, f, args, device, compute_type)
            results.append(res)
        except Exception as e:
            print(f"处理失败: {f} -> {e}", file=sys.stderr)

    # 汇总写一个索引文件
    ensure_dir(args.output_dir)
    index_path = os.path.join(args.output_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    print(f"完成。输出目录：{args.output_dir}")


if __name__ == "__main__":
    main()
# ASR 转写脚手架（faster-whisper）

一个通用的语音转写/英译命令行工具，基于 `faster-whisper`，支持将音频文件或目录批量转写，并导出 `SRT` 字幕与 `JSON` 结果。

## 安装

1. 确保系统安装了 `ffmpeg`（读取多种音频格式时需要）。
2. 创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> GPU（NVIDIA）建议使用 `compute_type=float16`；CPU 建议 `int8` 或 `int8_float32`。

## 使用

基本用法（转写）：

```bash
python asr_cli.py <音频文件或目录> \
  --model small \
  --task transcribe \
  --vad \
  --output-dir outputs
```

翻译为英文：

```bash
python asr_cli.py <音频路径> --task translate
```

常用参数：
- `--model`：模型（如 `tiny`、`base`、`small`、`medium`、`large-v3`）。
- `--device`：`auto`/`cpu`/`cuda`；默认自动检测。
- `--compute-type`：`float16`（GPU 推荐）、`int8`/`int8_float32`（CPU 推荐）。
- `--language`：源语言（如 `zh`、`en`），留空自动检测。
- `--vad`：启用 VAD 过滤，分段更干净。
- `--chunk-length`：分片长度（秒），默认 30。
- `--srt`/`--json`：控制导出格式；不指定则同时导出。

输出会在 `outputs/` 目录生成同名 `.srt` 和 `.json` 文件，并汇总 `index.json`。

## 流式服务与客户端

### 启动服务（FastAPI + WebSocket）

```bash
source .venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

- Web UI: 打开 `http://localhost:8000/`，允许麦克风，实时看到识别输出。
- WebSocket: `ws://localhost:8000/ws` 接收 `PCM16LE @ 16kHz` 二进制分片；控制消息：
  - `{"type":"start","sample_rate":16000}` 初始化；
  - 连续发送二进制音频；
  - `{"type":"stop"}` 完成并返回最终片段。

### Python 客户端（文件流式发送）

```bash
python client_stream.py --file samples/osr.wav --url ws://localhost:8000/ws --chunk-ms 200
```

注意：当前客户端示例要求 WAV 为 `16kHz/mono/PCM16`，其他格式请先转换。

## 提示与建议

- 首次运行会自动下载所选模型文件，时间取决于网络与模型大小。
- 若需要词级时间戳或说话人分离，可后续集成 `WhisperX` 与 `pyannote.audio`。
- 较长音频的吞吐取决于设备与模型大小；GPU 显存 ≥8GB 建议 `large-v3`，CPU 选 `small/medium` 或量化。
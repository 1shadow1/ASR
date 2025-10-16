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
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

- Web UI: 打开 `http://localhost:8001/web/`，允许麦克风，实时看到识别输出。
- WebSocket: `ws://localhost:8001/ws` 接收 `PCM16LE @ 16kHz` 二进制分片；控制消息：
  - `{"type":"start","sample_rate":16000}` 初始化；
  - 连续发送二进制音频；
  - `{"type":"stop"}` 完成并返回最终片段。

提示：某些环境下 `localhost` 可能握手超时，改用 `127.0.0.1` 可避免解析/IPv6问题。

### Python 客户端（文件流式发送）

客户端会自动重采样任意采样率的 `PCM16 WAV` 到 `16kHz`，并支持立体声转单声道：

```bash
python client_stream.py --file /path/to/audio.wav --url ws://127.0.0.1:8001/ws --chunk-ms 200
```

### REST 文件转写接口

服务同时提供 REST 上传并转写的接口：

```bash
curl -F "file=@/path/to/audio.wav" "http://localhost:8001/api/transcribe?task=transcribe&language=zh"
```

返回示例：

```json
{
  "language": "zh",
  "duration": 12.34,
  "segments": [{"id":0,"start":0.0,"end":2.1,"text":"..."}],
  "model": "small",
  "device": "cpu",
  "compute_type": "int8"
}
```

### 环境变量配置（模型与设备）

- `ASR_MODEL`：默认 `small`，可选 `tiny`/`base`/`medium`/`large-v3` 等。
- `ASR_DEVICE`：`auto`/`cpu`/`cuda`，默认自动检测。
- `ASR_COMPUTE_TYPE`：默认根据设备选择，GPU 推荐 `float16`，CPU 推荐 `int8` 或 `int8_float32`。

示例：

```bash
ASR_MODEL=large-v3 ASR_DEVICE=cuda ASR_COMPUTE_TYPE=float16 \
  uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### 常见问题与排查

- WebSocket 403 或握手失败：不要将静态资源挂载在根路径 `/`；当前页面挂载在 `/web/`，WebSocket 在 `/ws`。若 `localhost` 握手超时，改用 `ws://127.0.0.1:8001/ws`。
- BufferError: 若服务端日志出现 `cannot resize buffer`，已在 `server.py` 中对 `np.frombuffer(...).copy()` 进行修复以避免视图持有导致扩容失败。
- 识别没有输出：确保输入为人声；可将 `min_chunk_sec` 降低（如 1.0），或在客户端降低 `--chunk-ms`（如 100）提高解码频率。
- 依赖缺失：若报 `python-multipart` 未安装，执行 `pip install python-multipart`。
- 首次运行下载模型较慢：可预先拉取或使用较小模型（如 `base/small`）。

### 日志（JSONL）

实时识别的会话日志会写入 `logs/<session_id>.jsonl`（可通过环境变量 `ASR_LOG_DIR` 修改目录）。日志包括：

- `ready`：服务端返回模型、设备、`session_id` 与 `log_path`。
- `start`：收到客户端的开始控制消息与采样率、任务类型等。
- `segments`：逐批次识别结果（包含 `start/end/text`）。
- `done`：会话结束并发送最终分段。
- `disconnect`：连接关闭（正常或异常）。

示例：

```jsonl
{"type":"ready","model":"small","device":"cpu","compute_type":"int8","sample_rate":16000,"session_id":"...","log_path":"logs/....jsonl"}
{"type":"start","sample_rate":16000,"task":"transcribe","language":null}
{"type":"segments","segments":[{"start":0.0,"end":1.2,"text":"..."}]}
{"type":"done"}
{"type":"disconnect"}
```

### 输入音频保存（WAV）

服务端会在每次 WebSocket 会话期间保存输入的原始音频为 `WAV` 文件，默认路径：`inputs/<session_id>.wav`。该目录可通过环境变量 `ASR_INPUT_DIR` 配置。

- 返回字段：在 `ready` 与 `ack` 消息中会包含 `input_path`，例如：

```json
{
  "type": "ready",
  "sample_rate": 16000,
  "session_id": "69f1510d9f4649c3b436eb3ef55c833a",
  "log_path": "logs/69f1510d9f4649c3b436eb3ef55c833a.jsonl",
  "input_path": "inputs/69f1510d9f4649c3b436eb3ef55c833a.wav"
}
```

- 文件格式：`mono` 单声道、`PCM16`、采样率为客户端声明的 `sample_rate`（网页与示例客户端均为 `16kHz`）。
- 生命周期：在收到 `stop`/`disconnect` 后会关闭并完成写入；异常断开也会在服务端进行资源清理。
- 用途：便于后期复核与模型对比评估，可直接用 `ffplay` 或任意播放器打开：

```bash
ffplay -nodisp -autoexit inputs/<session_id>.wav
```

环境变量示例：

```bash
ASR_INPUT_DIR=/srv/ASR/inputs ASR_LOG_DIR=/srv/ASR/logs \
  uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

## 提示与建议

- 首次运行会自动下载所选模型文件，时间取决于网络与模型大小。
- 若需要词级时间戳或说话人分离，可后续集成 `WhisperX` 与 `pyannote.audio`。
- 较长音频的吞吐取决于设备与模型大小；GPU 显存 ≥8GB 建议 `large-v3`，CPU 选 `small/medium` 或量化。
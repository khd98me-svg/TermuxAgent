#!/usr/bin/env python3
"""CloudVision Kaggle worker: controlled video enhancement over a GitHub command queue."""
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import requests

OWNER = "khd98me-svg"
REPO = "TermuxAgent"
BRANCH = "cloudvision-kaggle-bridge"
COMMAND_PATH = "cloudvision/bridge-command.json"
RESULT_DIR = "cloudvision/bridge-results"
API = f"https://api.github.com/repos/{OWNER}/{REPO}"
TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
if not TOKEN:
    raise SystemExit("Missing GITHUB_TOKEN. Add it through Kaggle Secrets; never paste it into notebook text or chat.")

SESSION = requests.Session()
SESSION.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
})
PYTHON = sys.executable
WORK = Path("/kaggle/working/cloudvision")
WORK.mkdir(parents=True, exist_ok=True)

def github_get(path):
    response = SESSION.get(f"{API}/{path}", params={"ref": BRANCH}, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()

def get_command():
    item = github_get(f"contents/{COMMAND_PATH}")
    if not item:
        return None
    raw = base64.b64decode(item["content"]).decode("utf-8")
    return json.loads(raw)

def result_already_published(command_id):
    return github_get(f"contents/{RESULT_DIR}/{command_id}.json") is not None

def publish_result(command_id, payload):
    path = f"{RESULT_DIR}/{command_id}.json"
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    encoded = base64.b64encode(body.encode("utf-8")).decode("ascii")
    response = SESSION.put(
        f"{API}/contents/{path}",
        json={"message": f"CloudVision result {command_id}", "content": encoded, "branch": BRANCH},
        timeout=60,
    )
    response.raise_for_status()
    print(f"[bridge] result published: {path}", flush=True)

def run_cmd(args, **kwargs):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.run(args, check=True, **kwargs)

def require_ffmpeg():
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            raise RuntimeError(f"{executable} is unavailable in this Kaggle image.")

def cuda_info():
    import torch
    available = torch.cuda.is_available()
    return {
        "torch": torch.__version__,
        "cuda_available": available,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if available else None,
        "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if available else None,
    }

def install_realesrgan():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select GPU T4 in Kaggle and restart the session.")
    require_ffmpeg()
    # Keep Kaggle's preinstalled Torch, TorchVision, NumPy and OpenCV. Install only
    # the Python dependencies required by the pinned Real-ESRGAN source versions.
    run_cmd([PYTHON, "-m", "pip", "install", "-q", "addict", "future", "lmdb", "Pillow",
             "PyYAML", "requests", "scikit-image", "scipy", "tqdm", "yapf", "cython", "tensorboard"])
    for path, url, tag in [
        ("/kaggle/working/BasicSR", "https://github.com/XPixelGroup/BasicSR.git", "v1.4.2"),
        ("/kaggle/working/Real-ESRGAN", "https://github.com/xinntao/Real-ESRGAN.git", "v0.3.0"),
    ]:
        if not os.path.isdir(path):
            run_cmd(["git", "clone", "-q", "--depth", "1", "--branch", tag, url, path])
    # Fix legacy setup.py version lookup under current Python.
    for setup_path in ("/kaggle/working/BasicSR/setup.py", "/kaggle/working/Real-ESRGAN/setup.py"):
        with open(setup_path, "r", encoding="utf-8") as handle:
            source = handle.read()
        old = """def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']"""
        new = """def get_version():
    namespace = {}
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'), namespace)
    return namespace['__version__']"""
        if old in source:
            source = source.replace(old, new)
            with open(setup_path, "w", encoding="utf-8") as handle:
                handle.write(source)
        elif "namespace = {}" not in source:
            raise RuntimeError(f"Unexpected get_version() implementation in {setup_path}")
    run_cmd([PYTHON, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "/kaggle/working/BasicSR"])
    run_cmd([PYTHON, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "/kaggle/working/Real-ESRGAN"])

def load_upscaler(tile):
    import sys
    import types
    import torch
    try:
        import torchvision.transforms.functional_tensor
    except ModuleNotFoundError:
        import torchvision.transforms.functional as TF
        shim = types.ModuleType("torchvision.transforms.functional_tensor")
        shim.rgb_to_grayscale = TF.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = shim
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    weights = WORK / "RealESRGAN_x4plus.pth"
    if not weights.exists():
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        print("[cloudvision] downloading official RealESRGAN_x4plus weights", flush=True)
        response = requests.get(url, timeout=240)
        response.raise_for_status()
        weights.write_bytes(response.content)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(scale=4, model_path=str(weights), model=model, tile=tile,
                        tile_pad=10, pre_pad=0, half=True, device=torch.device("cuda:0")), weights

def validate_video_path(raw_path):
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Provide input_path, for example /kaggle/input/my-video/video.mp4")
    path = Path(raw_path).expanduser().resolve()
    allowed_roots = [Path("/kaggle/input").resolve(), Path("/kaggle/working").resolve()]
    if not any(path == root or root in path.parents for root in allowed_roots):
        raise ValueError("input_path must be inside /kaggle/input or /kaggle/working.")
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    return path

def probe_video(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True
    )
    data = json.loads(result.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video:
        raise RuntimeError("No video stream found in input file.")
    fps_text = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    fps = float(Fraction(fps_text))
    if not (0 < fps <= 240):
        raise RuntimeError(f"Unsupported or invalid video frame rate: {fps_text}")
    return {
        "width": int(video["width"]), "height": int(video["height"]), "fps": fps,
        "fps_text": fps_text, "duration": float(data.get("format", {}).get("duration", 0) or 0),
        "has_audio": any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
        "codec": video.get("codec_name"),
    }

def enhance_video(command):
    import numpy as np
    import torch

    allowed_keys = {"id", "action", "input_path", "scale", "tile", "max_frames", "crf"}
    extra = set(command) - allowed_keys
    if extra:
        raise ValueError(f"Unsupported command fields: {sorted(extra)}")
    source = validate_video_path(command.get("input_path"))
    scale = command.get("scale", 2)
    tile = command.get("tile", 256)
    max_frames = command.get("max_frames")
    crf = command.get("crf", 18)
    if scale not in (2, 4):
        raise ValueError("scale must be 2 or 4.")
    if tile not in (128, 256, 384, 512, 0):
        raise ValueError("tile must be 128, 256, 384, 512, or 0 (no tiling).")
    if max_frames is not None and (not isinstance(max_frames, int) or isinstance(max_frames, bool) or not 1 <= max_frames <= 100000):
        raise ValueError("max_frames must be an integer from 1 to 100000.")
    if not isinstance(crf, int) or isinstance(crf, bool) or not 15 <= crf <= 28:
        raise ValueError("crf must be an integer from 15 to 28.")
    require_ffmpeg()
    info = probe_video(source)
    out_w, out_h = info["width"] * scale, info["height"] * scale
    if out_w * out_h > 3840 * 2160:
        raise ValueError(f"Requested output {out_w}x{out_h} exceeds the 4K pixel budget. Use scale=2 or a smaller source.")
    # Validate the command and source before installing the AI stack.
    install_realesrgan()
    upsampler, weights = load_upscaler(tile)
    command_id = command["id"]
    output = WORK / f"{command_id}_x{scale}.mp4"
    silent = WORK / f"{command_id}_video_only.mp4"
    decode_args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source), "-map", "0:v:0"]
    if max_frames is not None:
        decode_args += ["-frames:v", str(max_frames)]
    decode_args += ["-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    decode = subprocess.Popen(
        decode_args,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    encode = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", info["fps_text"],
         "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", str(crf), "-pix_fmt", "yuv420p", str(silent)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )
    frame_bytes = info["width"] * info["height"] * 3
    processed = 0
    started = time.time()
    try:
        while True:
            raw = decode.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError("Incomplete decoded frame received from FFmpeg.")
            if max_frames is not None and processed >= max_frames:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(info["height"], info["width"], 3)
            enhanced, _ = upsampler.enhance(frame, outscale=scale)
            if enhanced.shape[1] != out_w or enhanced.shape[0] != out_h:
                raise RuntimeError(f"Unexpected enhanced frame dimensions: {enhanced.shape}")
            encode.stdin.write(enhanced.tobytes())
            processed += 1
            if processed % 50 == 0:
                print(f"[cloudvision] processed {processed} frames; elapsed {time.time()-started:.1f}s", flush=True)
        encode.stdin.close()
        decode.stdout.close()
        encode.wait()
        decode.wait()
        if encode.returncode != 0:
            raise RuntimeError("FFmpeg video encoder failed: " + encode.stderr.read().decode("utf-8", "replace")[-3000:])
        if decode.returncode != 0:
            raise RuntimeError("FFmpeg video decoder failed: " + decode.stderr.read().decode("utf-8", "replace")[-3000:])
    except Exception:
        for process in (decode, encode):
            if process.poll() is None:
                process.kill()
        raise
    if processed == 0:
        raise RuntimeError("No frames were processed.")
    # Mux the original audio back without re-encoding the enhanced video.
    mux_args = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(silent), "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart"
    ]
    # A frame-limited run must trim the original audio to the shorter video.
    if max_frames is not None:
        mux_args.append("-shortest")
    mux_args.append(str(output))
    mux = subprocess.run(mux_args, capture_output=True, text=True)
    if mux.returncode != 0:
        raise RuntimeError("Audio mux failed: " + mux.stderr[-3000:])
    silent.unlink(missing_ok=True)
    torch.cuda.synchronize()
    return {
        **cuda_info(),
        "input_file": str(source), "output_file": str(output),
        "input_bytes": source.stat().st_size, "output_bytes": output.stat().st_size,
        "input_dimensions": [info["width"], info["height"]],
        "output_dimensions": [out_w, out_h], "fps": info["fps"],
        "source_duration_seconds": info["duration"], "frames_processed": processed,
        "scale": scale, "tile": tile, "audio_preserved": info["has_audio"],
        "processing_seconds": round(time.time() - started, 2),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "result": "PASS",
        "note": "AI-generated enhancement; details are inferred, not recovered from missing source data."
    }

def image_test():
    install_realesrgan()
    import numpy as np
    from PIL import Image, ImageDraw
    upsampler, weights = load_upscaler(256)
    image = Image.new("RGB", (128, 128), (35, 48, 65))
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 111, 111), outline=(240, 190, 70), width=3)
    draw.ellipse((38, 38, 90, 90), fill=(70, 160, 210), outline=(245, 245, 245), width=2)
    frame = np.array(image)[:, :, ::-1].copy()
    start = time.time()
    output, _ = upsampler.enhance(frame, outscale=4)
    result_path = WORK / "bridge_image_test_x4.png"
    Image.fromarray(output[:, :, ::-1]).save(result_path)
    return {**cuda_info(), "input_size": [128, 128], "output_size": [int(output.shape[1]), int(output.shape[0])],
            "inference_seconds": round(time.time()-start, 3), "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "output_file": str(result_path), "result": "PASS"}

ACTIONS = {
    "cuda_info": cuda_info,
    "realesrgan_image_test": image_test,
    "video_upscale": enhance_video,
}

def main():
    print("[bridge] online; waiting for CloudVision jobs from GitHub.", flush=True)
    print("[bridge] allowed actions: cuda_info, realesrgan_image_test, video_upscale", flush=True)
    print("[bridge] stop this cell with Kaggle's interrupt button.", flush=True)
    last_id = None
    while True:
        try:
            command = get_command()
            if command:
                if not isinstance(command, dict):
                    raise ValueError("Command must be a JSON object.")
                command_id = str(command.get("id", ""))
                action = str(command.get("action", ""))
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", command_id):
                    raise ValueError("Invalid command id")
                if command_id != last_id:
                    if result_already_published(command_id):
                        print(f"[bridge] {command_id} already has a result; skipping duplicate.", flush=True)
                        last_id = command_id
                        time.sleep(15)
                        continue
                    print(f"[bridge] received {command_id}: {action}", flush=True)
                    started = datetime.now(timezone.utc).isoformat()
                    try:
                        if action not in ACTIONS:
                            raise ValueError(f"Action not allowed: {action}")
                        data = ACTIONS[action](command) if action == "video_upscale" else ACTIONS[action]()
                        result = {"id": command_id, "action": action, "status": "success",
                                  "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                                  "data": data}
                    except Exception as exc:
                        result = {"id": command_id, "action": action, "status": "error",
                                  "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                                  "error": str(exc), "traceback": traceback.format_exc()[-12000:]}
                    # Do not mark a command complete until its result is safely committed.
                    publish_result(command_id, result)
                    last_id = command_id
            time.sleep(15)
        except KeyboardInterrupt:
            print("[bridge] stopped by user.", flush=True)
            break
        except Exception as exc:
            print(f"[bridge] polling/publish error: {exc}", flush=True)
            time.sleep(30)

if __name__ == "__main__":
    main()

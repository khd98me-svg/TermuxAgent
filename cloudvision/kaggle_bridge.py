#!/usr/bin/env python3
"""CloudVision Kaggle bridge. Only fixed, allow-listed actions are executable."""
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

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

def publish_result(command_id, payload):
    path = f"{RESULT_DIR}/{command_id}.json"
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    encoded = base64.b64encode(body.encode("utf-8")).decode("ascii")
    response = SESSION.put(
        f"{API}/contents/{path}",
        json={
            "message": f"CloudVision bridge result {command_id}",
            "content": encoded,
            "branch": BRANCH,
        },
        timeout=30,
    )
    response.raise_for_status()
    print(f"[bridge] result published: {path}", flush=True)

def run_cmd(args):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)

def cuda_info():
    import torch
    return {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if torch.cuda.is_available() else None,
    }

def install_realesrgan():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Set Kaggle Accelerator to GPU T4 and restart the session.")
    run_cmd([PYTHON, "-m", "pip", "install", "-q", "numpy==2.3.5", "opencv-python-headless==4.11.0.86",
             "addict", "future", "lmdb", "Pillow", "PyYAML", "requests", "scikit-image", "scipy",
             "tqdm", "yapf", "cython", "tensorboard"])
    for path, url in [
        ("/kaggle/working/BasicSR", "https://github.com/XPixelGroup/BasicSR.git"),
        ("/kaggle/working/Real-ESRGAN", "https://github.com/xinntao/Real-ESRGAN.git"),
    ]:
        if not os.path.isdir(path):
            tag = "v1.4.2" if path.endswith("BasicSR") else "v0.3.0"
            run_cmd(["git", "clone", "-q", "--depth", "1", "--branch", tag, url, path])

    # Patch the legacy version loader for modern Python.
    for path in ["/kaggle/working/BasicSR/setup.py", "/kaggle/working/Real-ESRGAN/setup.py"]:
        with open(path, "r", encoding="utf-8") as handle:
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
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)
        elif "namespace = {}" not in source:
            raise RuntimeError(f"Unexpected get_version() implementation in {path}")

    run_cmd([PYTHON, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "/kaggle/working/BasicSR"])
    run_cmd([PYTHON, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "/kaggle/working/Real-ESRGAN"])

def realesrgan_image_test():
    install_realesrgan()
    import types
    import numpy as np
    import torch
    try:
        import torchvision.transforms.functional_tensor
    except ModuleNotFoundError:
        import torchvision.transforms.functional as TF
        shim = types.ModuleType("torchvision.transforms.functional_tensor")
        shim.rgb_to_grayscale = TF.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = shim

    from PIL import Image, ImageDraw
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    image = Image.new("RGB", (128, 128), (35, 48, 65))
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 111, 111), outline=(240, 190, 70), width=3)
    draw.ellipse((38, 38, 90, 90), fill=(70, 160, 210), outline=(245, 245, 245), width=2)
    draw.line((0, 127, 127, 0), fill=(230, 90, 100), width=3)

    weights = "/kaggle/working/RealESRGAN_x4plus.pth"
    if not os.path.exists(weights):
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        with open(weights, "wb") as handle:
            handle.write(response.content)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=weights, model=model, tile=256, tile_pad=10,
                             pre_pad=0, half=True, device=torch.device("cuda:0"))
    input_bgr = np.array(image)[:, :, ::-1].copy()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    output_bgr, _ = upsampler.enhance(input_bgr, outscale=4)
    elapsed = time.time() - start
    output_path = "/kaggle/working/cloudvision_bridge_x4.png"
    Image.fromarray(output_bgr[:, :, ::-1]).save(output_path)
    if output_bgr.shape[:2] != (512, 512):
        raise RuntimeError(f"Unexpected output dimensions: {output_bgr.shape}")
    return {
        **cuda_info(),
        "input_size": [128, 128],
        "output_size": [int(output_bgr.shape[1]), int(output_bgr.shape[0])],
        "scale": 4,
        "inference_seconds": round(elapsed, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "weights_sha256": hashlib.sha256(open(weights, "rb").read()).hexdigest(),
        "output_file": output_path,
        "result": "PASS",
    }

ACTIONS = {
    "cuda_info": cuda_info,
    "realesrgan_image_test": realesrgan_image_test,
}

def main():
    print("[bridge] online; polling GitHub every 20 seconds.", flush=True)
    print("[bridge] stop this cell with Kaggle's interrupt button.", flush=True)
    last_id = None
    while True:
        try:
            command = get_command()
            if command:
                command_id = str(command.get("id", ""))
                action = str(command.get("action", ""))
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", command_id):
                    raise ValueError("Invalid command id")
                if command_id != last_id:
                    last_id = command_id
                    print(f"[bridge] received {command_id}: {action}", flush=True)
                    started = datetime.now(timezone.utc).isoformat()
                    try:
                        if action not in ACTIONS:
                            raise ValueError(f"Action not allowed: {action}")
                        data = ACTIONS[action]()
                        result = {"id": command_id, "action": action, "status": "success",
                                  "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                                  "data": data}
                    except Exception as exc:
                        result = {"id": command_id, "action": action, "status": "error",
                                  "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                                  "error": str(exc), "traceback": traceback.format_exc()[-12000:]}
                    publish_result(command_id, result)
            time.sleep(20)
        except KeyboardInterrupt:
            print("[bridge] stopped by user.", flush=True)
            break
        except Exception as exc:
            print(f"[bridge] polling error: {exc}", flush=True)
            time.sleep(30)

if __name__ == "__main__":
    main()

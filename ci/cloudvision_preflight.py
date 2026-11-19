import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

REPORT = {"python": sys.version, "platform": platform.platform(), "stages": {}}
OUT = Path("cloudvision-output")
OUT.mkdir(exist_ok=True)

def stage(name, status, **info):
    REPORT["stages"][name] = {"status": status, **info}
    print(f"[{status}] {name}: {info}")

# Verify the exact framework versions expected by the user's Kaggle runtime.
import torch
import torchvision
import numpy as np
import cv2

REPORT["versions"] = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": np.__version__,
    "opencv": cv2.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_build": torch.version.cuda,
}
assert torch.__version__.split("+")[0] == "2.11.0", REPORT["versions"]
assert torchvision.__version__.split("+")[0] == "0.26.0", REPORT["versions"]
stage("framework_versions", "PASS", **REPORT["versions"])

# Compatibility shim for BasicSR's import of the removed TorchVision module.
try:
    import torchvision.transforms
    import torchvision.transforms.functional as tv_functional
    sys.modules["torchvision.transforms.functional_tensor"] = tv_functional
    torchvision.transforms.functional_tensor = tv_functional

    import basicsr
    import realesrgan
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    stage(
        "package_imports", "PASS",
        basicsr=getattr(basicsr, "__version__", "unknown"),
        realesrgan=getattr(realesrgan, "__version__", "unknown"),
    )
except Exception as exc:
    stage("package_imports", "FAIL", error=repr(exc), traceback=traceback.format_exc())
    raise

# Download official x4plus weights and run a bounded CPU inference.
try:
    weights = Path("RealESRGAN_x4plus.pth")
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    if not weights.exists():
        urllib.request.urlretrieve(url, weights)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=4
    )
    upsampler = RealESRGANer(
        scale=4, model_path=str(weights), model=model,
        tile=0, tile_pad=10, pre_pad=0, half=False,
        device=torch.device("cpu"),
    )

    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(32, dtype=np.uint8)[None, :] * 7
    image[:, :, 1] = np.arange(24, dtype=np.uint8)[:, None] * 10
    image[5:19, 7:25] = (40, 180, 230)

    start = time.perf_counter()
    result, _ = upsampler.enhance(image, outscale=4)
    elapsed = time.perf_counter() - start
    assert result.shape == (96, 128, 3), result.shape

    image_path = OUT / "realesrgan_cpu_smoke.png"
    assert cv2.imwrite(str(image_path), result)
    stage(
        "realesrgan_cpu_inference", "PASS",
        input_shape=[24, 32, 3], output_shape=list(result.shape),
        elapsed_seconds=round(elapsed, 3), weights_sha256=digest,
        output=str(image_path),
    )
except Exception as exc:
    stage("realesrgan_cpu_inference", "FAIL", error=repr(exc), traceback=traceback.format_exc())
    raise

# Validate the FFmpeg 4K encode path. This is NOT AI-upscaled video.
try:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    assert ffmpeg and ffprobe
    video = OUT / "ffmpeg_4k_encode_smoke.mp4"
    subprocess.run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=2:duration=1",
        "-vf", "scale=3840:2160:flags=lanczos", "-frames:v", "2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(video)
    ], check=True)
    probe = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,nb_frames",
        "-of", "json", str(video)
    ], check=True, text=True, capture_output=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["width"] == 3840 and stream["height"] == 2160, stream
    stage("ffmpeg_4k_encode", "PASS", stream=stream, output=str(video))
except Exception as exc:
    stage("ffmpeg_4k_encode", "FAIL", error=repr(exc), traceback=traceback.format_exc())
    raise

REPORT["overall"] = "PASS" if all(x["status"] == "PASS" for x in REPORT["stages"].values()) else "FAIL"
Path("cloudvision-report.json").write_text(json.dumps(REPORT, indent=2, ensure_ascii=False))
print(json.dumps(REPORT, indent=2, ensure_ascii=False))
if REPORT["overall"] != "PASS":
    raise SystemExit(1)

# CloudVision — Kaggle GPU execution bridge

This branch contains the first working implementation path for CloudVision video enhancement. GitHub is the job queue and report store; the Kaggle notebook is the GPU worker. This is not a permanent server: the Kaggle session must remain running while a job is processed.

## Components
- `kaggle_github_bridge.ipynb`: launch the worker inside a Kaggle GPU notebook.
- `kaggle_bridge.py`: controlled command listener and Real-ESRGAN video pipeline.
- `bridge-command.json`: one queued job at a time.
- `bridge-results/<id>.json`: result or error report for each job.
- Processed MP4 files are saved under `/kaggle/working/cloudvision/` in the Kaggle session. Video files are deliberately not committed to GitHub.

## One-time setup
1. Open a Kaggle notebook and select **GPU T4**.
2. Add the source video to the notebook as a Kaggle Dataset (or upload it to the notebook session). Dataset files are normally under `/kaggle/input/<dataset-name>/...`; uploaded working files are under `/kaggle/working/...`.
3. Create a GitHub fine-grained token scoped only to `khd98me-svg/TermuxAgent`, with **Contents: Read and write**.
4. In Kaggle, open **Add-ons → Secrets** and save the token under the exact name `GITHUB_TOKEN`.
5. Run the launcher cell from `kaggle_github_bridge.ipynb`. Keep it running while jobs are being submitted.

Never paste tokens, cookies, session IDs, or Kaggle credentials into ChatGPT or commit them to this repository.

## Submit a video enhancement job

Create or update `cloudvision/bridge-command.json` on branch `cloudvision-kaggle-bridge` with a unique ID and an input path that exists inside `/kaggle/input` or `/kaggle/working`.

Example:
```json
{
  "id": "video-job-001",
  "action": "video_upscale",
  "input_path": "/kaggle/input/my-video/source.mp4",
  "scale": 2,
  "tile": 256,
  "crf": 18
}
```

Parameters:
- `scale`: 2 or 4. Start with 2 for a practical 4K output from 1080p input.
- `tile`: 128, 256, 384, 512, or 0 (no tiling). Smaller tiles use less VRAM but can be slower.
- `crf`: 15–28; lower means higher encoded quality and larger files. Default is 18.
- `max_frames`: optional integer for a short diagnostic run only. Omit it to process the full video.

The worker:
1. Checks the video path and probes dimensions, frame rate, duration, and audio.
2. Uses the official RealESRGAN_x4plus model on CUDA, processing frames sequentially to avoid loading the entire video into RAM.
3. Encodes the enhanced video with FFmpeg and muxes the original audio back.
4. Saves the MP4 under `/kaggle/working/cloudvision/<job-id>_x<scale>.mp4`.
5. Commits a JSON report to `cloudvision/bridge-results/<job-id>.json`.

The report includes the output path, dimensions, frame count, processing time, GPU/VRAM information, file sizes, and model-weight hash. Download the MP4 from the Kaggle notebook's Files/Output panel; the MP4 itself is not pushed to GitHub.

## Other supported actions
```json
{"id":"gpu-check-001","action":"cuda_info"}
```
```json
{"id":"image-check-001","action":"realesrgan_image_test"}
```

Only these three fixed actions are allowed: `cuda_info`, `realesrgan_image_test`, and `video_upscale`. The worker does not execute arbitrary shell commands or arbitrary Python source. Input files are restricted to Kaggle input/working directories.

## Important limitations
- Kaggle may stop sessions due to quotas, idle limits, or platform policies. It is a batch GPU worker, not 24/7 hosting.
- A 4× upscale can exceed the 4K pixel budget for some source sizes. The worker rejects outputs above 4K total dimensions.
- Real-ESRGAN synthesizes plausible detail; it cannot recover information that was never present in the source.
- This is the first end-to-end processing backend. A persistent job manager, browser upload/download UI, retry/resume across interrupted sessions, and multi-GPU scheduling remain subsequent product work.
- The worker has been committed to GitHub but must still be run in a real Kaggle T4 session before its runtime compatibility and output quality can be claimed as verified.

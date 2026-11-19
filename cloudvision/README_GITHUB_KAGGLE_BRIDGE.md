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
2. For recorded YouTube videos, submit a supported HTTPS YouTube URL in `source_url`; the worker downloads it into the Kaggle session automatically. Alternatively, provide a local file under `/kaggle/input` or `/kaggle/working`.
3. Create a GitHub fine-grained token scoped only to `khd98me-svg/TermuxAgent`, with **Contents: Read and write**.
4. In Kaggle, open **Add-ons → Secrets** and save the token under the exact name `GITHUB_TOKEN`.
5. Run the launcher cell from `kaggle_github_bridge.ipynb`. Keep it running while jobs are being submitted.

Never paste tokens, cookies, session IDs, or Kaggle credentials into ChatGPT or commit them to this repository.

## Automated launch from GitHub Actions

The workflow `.github/workflows/cloudvision-kaggle-remote.yml` can queue a recorded YouTube URL, update the existing Kaggle notebook, launch it through the Kaggle CLI, wait for completion, and upload Kaggle outputs as a GitHub Actions artifact.

One-time repository secrets required in **GitHub → Settings → Secrets and variables → Actions**:
- `KAGGLE_USERNAME`: your Kaggle account username.
- `KAGGLE_KEY`: the API key from your Kaggle account settings. Keep it in GitHub Secrets; never paste it into chat or commit it.
- `KAGGLE_KERNEL_ID`: the full ID of the existing Kaggle notebook in `username/notebook-slug` form.

Important: `KAGGLE_KERNEL_ID` must identify the same Kaggle notebook where the `GITHUB_TOKEN` Kaggle Secret is already attached. The workflow updates and runs that notebook rather than creating a new one. Kaggle GPU quota and availability still apply.

After the three GitHub repository secrets are configured, use **Actions → CloudVision remote Kaggle GPU job → Run workflow**. The default diagnostic run processes 24 frames. Leave `max_frames` blank only when ready to process the entire video. The output video and reports are downloaded from the completed workflow's artifact.

## Submit a video enhancement job

Create or update `cloudvision/bridge-command.json` on branch `cloudvision-kaggle-bridge` with a unique ID. For a recorded YouTube video, use `source_url`:

```json
{
  "id": "youtube-job-001",
  "action": "video_upscale",
  "source_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "scale": 2,
  "tile": 256,
  "crf": 18
}
```

Or use `input_path` for a file already present in Kaggle. Supply exactly one of `source_url` or `input_path`. YouTube playlists are disabled; only HTTPS URLs on youtube.com or youtu.be are accepted. The worker checks that the URL is not a currently-live or upcoming stream before downloading. Access restrictions, age gates, or videos unavailable to yt-dlp may prevent download.

Parameters:
- `scale`: 1, 2, or 4. Use 1 to preserve the source dimensions (useful for 4K sources), 2 for 2× output, or 4 for 4× output.
- `tile`: 128, 256, 384, 512, or 0 (no tiling). Smaller tiles use less VRAM but can be slower.
- `crf`: 15–28; lower means higher encoded quality and larger files. Default is 18.
- `max_frames`: optional integer for a short diagnostic run only. Omit it to process the full video.

The worker can run in listener mode for manual sessions or one-shot mode for automated execution. In one-shot mode it waits up to 900 seconds for one new job, publishes the result, and exits.

The worker:
1. Downloads a recorded YouTube video when `source_url` is supplied, or validates a local Kaggle path, then probes dimensions, frame rate, duration, and audio.
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
- This version supports recorded YouTube videos, not real-time live-stream enhancement. Live capture and low-latency continuous processing require a separate streaming pipeline.
- Use YouTube downloads only for content you own or are authorized to download, and comply with YouTube's terms and applicable law.
- Kaggle may stop sessions due to quotas, idle limits, or platform policies. It is a batch GPU worker, not 24/7 hosting.
- A 4× upscale can exceed the 4K pixel budget for some source sizes. The worker rejects outputs above 4K total dimensions.
- Real-ESRGAN synthesizes plausible detail; it cannot recover information that was never present in the source.
- This is the first end-to-end processing backend. A persistent job manager, browser upload/download UI, retry/resume across interrupted sessions, and multi-GPU scheduling remain subsequent product work.
- The worker has been committed to GitHub but must still be run in a real Kaggle T4 session before its runtime compatibility and output quality can be claimed as verified.

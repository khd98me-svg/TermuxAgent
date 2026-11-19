# CloudVision GitHub ↔ Kaggle bridge

This is a lightweight command bridge for an already-running Kaggle notebook session. Kaggle polls a single command file on this branch; it executes only two allow-listed actions and writes JSON results back to this repository.

## Files
- `kaggle_github_bridge.ipynb`: import into Kaggle and run its launcher cell.
- `kaggle_bridge.py`: polling worker; no arbitrary shell command support.
- `bridge-command.json`: created only when an operator submits a command.
- `bridge-results/<id>.json`: execution report files.

## One-time Kaggle setup
1. Open the notebook in Kaggle and select **GPU T4**.
2. Create a GitHub fine-grained personal access token scoped only to `khd98me-svg/TermuxAgent`, with **Contents: Read and write**. Do not grant workflow, administration, or unrelated permissions.
3. In Kaggle, open Add-ons / Secrets and save the token under the exact name `GITHUB_TOKEN`.
4. Run the launcher cell and leave it running. The cell prints that the bridge is online.

Never paste tokens, cookies, session IDs, or Kaggle credentials into ChatGPT or commit them to this repository.

## Allowed commands
Command JSON has exactly this small shape:
```json
{"id":"unique-run-id","action":"cuda_info"}
```
or
```json
{"id":"unique-run-id","action":"realesrgan_image_test"}
```

No free-form command, shell string, Python source, URL, or path is accepted as an action. The worker validates the ID and dispatches only to its built-in allow-list.

## Results
The worker commits a JSON report to:
`cloudvision/bridge-results/<id>.json`

The image inference action installs the pinned Real-ESRGAN dependencies without replacing the Kaggle-provided PyTorch build, downloads the official x4plus weights, runs a 128x128 → 512x512 inference on CUDA, and records runtime, peak VRAM, and weight SHA-256.

## Operational limits
- The notebook session must remain alive and the listener cell must keep running.
- Kaggle may stop sessions due to quotas, idle limits, or platform policies. This is not a permanent GPU server.
- The bridge is for controlled experiments, not production hosting.

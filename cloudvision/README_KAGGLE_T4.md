# CloudVision — Kaggle T4 CUDA smoke test

## What this validates
- Keeps the PyTorch and TorchVision builds already provided by Kaggle.
- Requires a CUDA GPU and prints the detected GPU and VRAM.
- Installs pinned BasicSR/Real-ESRGAN sources with the Python 3.13 version-reader compatibility patch.
- Downloads the official RealESRGAN_x4plus weights.
- Runs actual x4 inference on CUDA and asserts a 128x128 input becomes 512x512.
- Reports inference time and peak allocated VRAM.

## Run it
1. Download `kaggle_t4_cuda_test.ipynb` from this folder.
2. In Kaggle, create/import a notebook and upload the downloaded `.ipynb`.
3. Open Notebook options / Session options and set Accelerator to **GPU T4**.
4. Run all cells in order. The first cell stops immediately if CUDA is unavailable.
5. Send back the complete output from the final cell, especially the lines beginning `GPU:`, `Output:`, `Inference seconds:`, `Peak VRAM GiB:`, and `RESULT:`.

## Important
This is a GPU image-inference smoke test, not yet a video-upscaling benchmark. Do not change the notebook's Torch version. Video processing, audio preservation, 4K throughput, and long-clip memory behavior are the next validation stage after this passes.

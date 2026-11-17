# LTX-2.3 AI2V — Kaggle Dual-T4
- Run `LTX23_DualGPU_Kaggle.ipynb` in Kaggle with GPU T4 x2.
- Add the original `video_ltx2_3_ia2v_Dual_gpu.json` to a Kaggle Dataset before Cell 2.
- Cell 2 downloads the checkpoint, two LoRAs, Gemma FP4 encoder, and latent upscaler, then writes the modified workflow.
- DisTorch2 distributes checkpoint blocks over cuda:0 (7 GB), cuda:1 (7 GB), and CPU RAM. This is model placement/offload, not tensor-parallel compute.
- The original AV conditioning and two-stage graph are preserved. Stock tiled VAE decode is retained.
- Start with 768x512 / 4 seconds. The 22B FP8 checkpoint remains large; this profile is lower-memory, not a smaller-parameter model.
- The Kaggle MultiGPU fork README reports testing on ComfyUI 0.37.0 / PyTorch 2.10+cu130; your Kaggle runtime may differ. Check node registration and startup logs before a full render.
- No ComfyUI core, comfy_aimdo, or DisTorch2 source files are edited.

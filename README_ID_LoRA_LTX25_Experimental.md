# Experimental LTX-2.5 ID-LoRA dual-T4 port

This repository addition contains a Kaggle installer/model downloader and an explicitly experimental UI graph sketch.

## Files
- `ID_LoRA_LTX25_Kaggle_Dual_T4_Experimental.ipynb`: 3 code cells: ComfyUI+Manager installation, model download, launch.
- `ID_LoRA_LTX25_Experimental_DualT4.json`: graph/parameter sketch based on the official two-stage ID-LoRA node graph.

## Critical compatibility status
Public TalkVid/CelebV-HQ ID-LoRA adapters are trained against LTX-2.3. The legacy ID-LoRA custom node's `IDLoraTwoStageModelLoader` is specifically an LTX-2.3 loader. Therefore the JSON is **not executable as an LTX-2.5 graph** merely by changing model filenames. A proper port must validate the state-dict key names/shapes and port the stage-1 sampler and stage-2 distilled-LoRA/upscaler path to the LTX-2.5 architecture.

The notebook installs current ComfyUI where ID-LoRA reference-audio conditioning is native (`LTXVReferenceAudio`), and downloads the LTX-2.5 INT8 ConvRot model family plus the LTX-2.3 TalkVid adapter. It does not install the deprecated custom ID-LoRA node, since that node has hard-coded LTX-2.3 pipeline assumptions.

## Dual T4
Two T4 GPUs do not pool into one 30-GB device. The notebook exposes both GPUs and uses low-VRAM mode, but this alone does not shard the transformer across them. Multi-GPU sharding needs a compatible, tested loader/patch; do not assume `CUDA_VISIBLE_DEVICES=0,1` splits model weights. This avoids silently claiming a working DisTorch2 dual-GPU setup without validating that the selected loader works with the current ComfyUI and LTX-2.5.

## Sources
- https://github.com/ID-LoRA/ID-LoRA-LTX2.3-ComfyUI
- https://github.com/ID-LoRA/ID-LoRA-LTX2.3-ComfyUI/blob/main/example_workflows/id_lora_two_stage.json
- https://docs-dev.ltx.io/open-source-model/integration-tools/comfy-ui

**Status: experimental research scaffold, not a tested LTX-2.5 two-stage ID-LoRA workflow.**

#!/usr/bin/env python3
"""Patch the uploaded LTX-2.3 IA2V ComfyUI workflow for a conservative Kaggle 2x-T4 profile.
Run: python patch_ltx23_dualt4.py input.json output.json
"""
import json, sys
from pathlib import Path
src, dst = map(Path, sys.argv[1:3])
wf = json.loads(src.read_text(encoding="utf-8"))
sg = wf["definitions"]["subgraphs"][0]
outer = next(n for n in wf["nodes"] if n.get("id") == 340)
outer["widgets_values"][1:4] = [False, 768, 512]
outer["widgets_values"][5] = 4
loader = next(n for n in sg["nodes"] if n.get("id") == 317)
loader["type"] = "CheckpointLoaderSimpleDisTorch2MultiGPU"
loader.setdefault("properties", {})["Node name for S&R"] = loader["type"]
loader["properties"]["cnr_id"] = "ComfyUI-MultiGPU-Kaggle"
loader["inputs"] = [i for i in loader.get("inputs", []) if i.get("name") == "ckpt_name"]
for name, typ in [("compute_device","COMBO"),("virtual_vram_gb","FLOAT"),("donor_device","COMBO"),("expert_mode_allocations","STRING"),("eject_models","BOOLEAN")]:
    loader["inputs"].append({"localized_name":name,"name":name,"type":typ,"widget":{"name":name},"link":None})
loader["widgets_values"] = ["ltx-2.3-22b-dev-fp8.safetensors","cuda:0",0.0,"cpu","cuda:0,7gb;cuda:1,7gb;cpu,*",True]
loader["widgets_values_named"] = {"ckpt_name":"ltx-2.3-22b-dev-fp8.safetensors","compute_device":"cuda:0","virtual_vram_gb":0.0,"donor_device":"cpu","expert_mode_allocations":"cuda:0,7gb;cuda:1,7gb;cpu,*","eject_models":True}
dst.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote", dst)

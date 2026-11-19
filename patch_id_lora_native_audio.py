#!/usr/bin/env python3
"""Patch the native LTX-2.3 ID-LoRA workflow so final audio comes from Stage 1.

Usage:
  python patch_id_lora_native_audio.py input_workflow.json
  python patch_id_lora_native_audio.py input_workflow.json -o fixed_workflow.json

The patch is deliberately strict: it verifies the expected node IDs and old
connection before changing anything. It does not alter the model, prompt,
samplers, image path, or Stage 2 video path.
"""
import argparse
import json
from pathlib import Path

OLD_LINK = [63, 43, 1, 45, 0, "LATENT"]
NEW_LINK = [69, 33, 1, 45, 0, "LATENT"]
STAGE1_AUDIO_LINK = [48, 33, 1, 36, 1, "LATENT"]


def node_by_id(workflow, node_id):
    matches = [n for n in workflow.get("nodes", []) if n.get("id") == node_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one node id={node_id}; found {len(matches)}")
    return matches[0]


def patch(workflow):
    if workflow.get("id") != "id-lora-native-kaggle-t4x2":
        raise ValueError("This does not look like the supplied id-lora-native-kaggle-t4x2 workflow.")

    n33 = node_by_id(workflow, 33)
    n43 = node_by_id(workflow, 43)
    n45 = node_by_id(workflow, 45)

    links = workflow.get("links", [])
    if OLD_LINK not in links:
        if NEW_LINK in links and n45.get("inputs", [])[0].get("link") == 69:
            raise ValueError("Workflow already appears to contain the Stage-1 audio fix.")
        raise ValueError("Expected old link [63,43,1,45,0,'LATENT'] was not found; refusing to guess.")

    samples_inputs = [i for i in n45.get("inputs", []) if i.get("name") == "samples"]
    if len(samples_inputs) != 1 or samples_inputs[0].get("link") != 63:
        raise ValueError("Node 45 samples input is not connected to link 63 as expected.")

    audio_output = n33.get("outputs", [])[1]
    if audio_output.get("name") != "audio_latent" or 48 not in (audio_output.get("links") or []):
        raise ValueError("Node 33 audio output does not match the expected Stage-1 graph.")

    stage2_audio_output = n43.get("outputs", [])[1]
    if stage2_audio_output.get("name") != "audio_latent" or 63 not in (stage2_audio_output.get("links") or []):
        raise ValueError("Node 43 audio output does not match the expected Stage-2 graph.")

    # Remove Stage-2 audio from the final decoder.
    links.remove(OLD_LINK)
    stage2_audio_output["links"] = [
        link_id for link_id in (stage2_audio_output.get("links") or []) if link_id != 63
    ]

    # Feed the Stage-1 audio latent directly to the final audio decoder.
    audio_output["links"] = list(audio_output.get("links") or [])
    if 69 not in audio_output["links"]:
        audio_output["links"].append(69)
    samples_inputs[0]["link"] = 69
    links.append(NEW_LINK)

    workflow["last_link_id"] = max(int(workflow.get("last_link_id", 0)), 69)
    return workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Original ComfyUI workflow JSON")
    parser.add_argument("-o", "--output", type=Path, help="Output path (default: <input>_AUDIOFIX.json)")
    args = parser.parse_args()

    source = args.input
    target = args.output or source.with_name(source.stem + "_AUDIOFIX" + source.suffix)
    data = json.loads(source.read_text(encoding="utf-8"))
    patched = patch(data)
    target.write_text(json.dumps(patched, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Patched workflow written to: {target}")
    print("Audio path: Node 33 output 1 -> Node 45 samples")
    print("Stage-2 video path is unchanged: Node 43 output 0 -> Node 44")


if __name__ == "__main__":
    main()

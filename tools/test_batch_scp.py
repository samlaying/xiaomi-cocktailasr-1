#!/usr/bin/env python3
"""
Batch inference for Xiaomi-CocktailASR-1 SpeechLLM directly over a 5-column scp_text_ref.all.

Each line is a TSV row ``utt_id, wav, text, ref_wav, ref_id``. This tool loads the
model once via the standard HuggingFace ``trust_remote_code`` interface and calls
``model(wav, ref)`` per row -- i.e. it runs the exact same high-level path as::

    from transformers import AutoModel
    model = AutoModel.from_pretrained(hf_model_dir, trust_remote_code=True)
    text  = model(wav, ref)

so the ``ref + 1s silence + target`` concatenation happens inside the model
(``MicAsrFeatureExtractor.prepare_audio``). There is no separate pre-concatenation
step -- tools/prepare_from_scp.py is only needed if you want the concatenated wavs
materialized on disk.

Reproducibility note: rows are processed strictly in file order (no sorting). The
reference-window random crop in ``prepare_audio`` draws from a single global
``random.seed(42)`` stream in call order, so this iteration order is what makes a
run reproducible -- do NOT reorder the input.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/test_batch_scp.py \
        --hf_model_dir ./models/Xiaomi-CocktailASR-1 \
        --input_scp /path/to/scp_text_ref.all \
        --output_file /path/to/result.txt
"""

import argparse
import os
import time

import torch
from transformers import AutoModel


def read_scp_text_ref(path):
    """Read a 5-column scp_text_ref.all.

    Returns a list of dicts ``{out_id, wav, ref_wav, ref_text}`` in file order.
    ``out_id`` is ``utt_id+ref_id`` to match tools/prepare_from_scp.py naming so
    downstream scoring lines up. Malformed lines (<5 columns) are skipped.
    """
    entries = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=9)
            if len(parts) < 5:
                print(f"  Skipping malformed line: {line[:80]}")
                skipped += 1
                continue
            uttid, wav, text, ref_wav, refid = parts[:5]
            entries.append(
                {"out_id": f"{uttid}+{refid}", "wav": wav, "ref_wav": ref_wav, "ref_text": text}
            )
    return entries, skipped


def get_args():
    parser = argparse.ArgumentParser(description="Batch inference for Xiaomi-CocktailASR-1 SpeechLLM")
    parser.add_argument("--hf_model_dir", required=True, help="Path to Xiaomi-CocktailASR-1 model directory")
    parser.add_argument("--input_scp", required=True,
                        help="Path to scp_text_ref.all (TSV: utt_id, wav, text, ref_wav, ref_id)")
    parser.add_argument("--output_file", required=True, help="Output hypothesis file")
    parser.add_argument("--ref_file", default=None,
                        help="Optional path to write reference transcripts (default: <output_dir>/text)")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought mode")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max tokens (default 256, 512 for CoT)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    return parser.parse_args()


def main():
    args = get_args()
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    print(f"[PID {os.getpid()}] Device: cuda, dtype: {args.dtype}")
    print(f"  HF model:  {args.hf_model_dir}")
    print(f"  input_scp: {args.input_scp}")
    print(f"  Output:    {args.output_file}")

    model = AutoModel.from_pretrained(
        args.hf_model_dir, trust_remote_code=True, torch_dtype=dtype
    ).cuda().eval()

    entries, skipped = read_scp_text_ref(args.input_scp)
    print(f"  Total utterances: {len(entries)} ({skipped} skipped)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    hyps = {}
    start_time = time.time()

    for i, e in enumerate(entries, 1):
        try:
            # High-level HF interface: concatenation (ref + silence + target)
            # happens inside the model via prepare_audio.
            hyps[e["out_id"]] = model(
                e["wav"], e["ref_wav"], cot=args.cot, max_new_tokens=args.max_new_tokens
            )
        except Exception as exc:
            import traceback
            print(f"  ERROR on {e['out_id']}: {exc}")
            traceback.print_exc()
            hyps[e["out_id"]] = ""

        if i % 50 == 0 or i == len(entries):
            elapsed = time.time() - start_time
            print(f"  [{i}/{len(entries)}] {elapsed:.1f}s, {i / elapsed:.1f} utt/s")

    with open(args.output_file, "w", encoding="utf-8") as f:
        for e in entries:
            text = hyps.get(e["out_id"], "").replace("\n", "\\n")
            f.write(f"{e['out_id']} {text}\n")

    ref_file = args.ref_file or os.path.join(os.path.dirname(os.path.abspath(args.output_file)), "text")
    with open(ref_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(f"{e['out_id']} {e['ref_text']}\n")

    elapsed = time.time() - start_time
    print(f"\n  Done. {len(entries)} utterances in {elapsed:.1f}s ({len(entries) / max(elapsed, 1e-9):.1f} utt/s)")
    print(f"  Hypotheses: {args.output_file}")
    print(f"  References: {ref_file}")


if __name__ == "__main__":
    main()

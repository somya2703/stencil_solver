#!/usr/bin/env python3
"""
apply_d2.py -- apply checklist item D2 to BUILD_AND_RUN.md in place.

Usage (from the repo root):
    python3 apply_d2.py BUILD_AND_RUN.md
    git diff BUILD_AND_RUN.md        # review before committing

Every edit asserts that its anchor text was found, so a mismatch stops the
script with a clear message instead of silently skipping. The file is only
written if all edits succeed.
"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "BUILD_AND_RUN.md"
text = open(path, encoding="utf-8").read()


def sub(pattern, repl, flags=0, count=0, label=""):
    global text
    new, n = re.subn(pattern, lambda m: repl, text, count=count, flags=flags)
    assert n > 0, f"anchor not found: {label or pattern[:60]}"
    text = new
    print(f"ok ({n}x): {label}")


def rep(old, new, count=-1, label=""):
    global text
    assert old in text, f"anchor not found: {label or old[:60]}"
    text = text.replace(old, new) if count < 0 else text.replace(old, new, count)
    print(f"ok: {label}")


# 1. Step 3 explanation: drop the "mistuned A100 defaults" story, state the
#    measured best config instead (resolves the Step 3 vs 8B contradiction).
sub(
    r"Also pass the tiled-kernel tuning knobs\..*?(?=```bash)",
    "Also pass the tiled-kernel tuning knobs. The best configuration measured\n"
    "on this GPU is `TILE_Y=8, PENCIL_Z=12`: tiled runs at about 0.71x of naive\n"
    "at 512^3, so it is still *slower* than naive (see Section 5B and\n"
    "`docs/CASE_STUDY.md` section 3.2 for why). The previous recommendation of\n"
    "`TILE_Y=4, PENCIL_Z=8` measures worse (0.58x at 512^3), because a smaller\n"
    "`TILE_Y` shrinks the useful fraction of each block (Section 8B).\n"
    "`CMakeLists.txt` exposes the knobs:\n\n",
    flags=re.DOTALL, count=1, label="Step 3 explanation paragraph",
)

# 2. Remove the "see the mistuned behaviour first" paragraph.
sub(
    r"If you'd rather see the \*mistuned\* A100-defaults behavior first.*?not a bug in the kernel itself\.\n\n",
    "",
    flags=re.DOTALL, count=1, label="mistuned-defaults paragraph",
)

# 3. Configure flags: 4/8 -> 8/12 in cmake commands (multi-line and quick-ref).
rep("-DSTENCIL_TILE_Y=4 \\\n    -DSTENCIL_PENCIL_Z=8",
    "-DSTENCIL_TILE_Y=8 \\\n    -DSTENCIL_PENCIL_Z=12",
    label="cmake flags (Step 3, Section 9)")
rep('-DCMAKE_CUDA_ARCHITECTURES="89" \\\n    -DSTENCIL_TILE_Y=4 -DSTENCIL_PENCIL_Z=8',
    '-DCMAKE_CUDA_ARCHITECTURES="89" \\\n    -DSTENCIL_TILE_Y=8 -DSTENCIL_PENCIL_Z=12',
    label="cmake flags (Quick reference)")
rep("`-DSTENCIL_TILE_Y=4 -DSTENCIL_PENCIL_Z=8` flags",
    "`-DSTENCIL_TILE_Y=8 -DSTENCIL_PENCIL_Z=12` flags",
    label="Section 1B flags mention")

# 4. Expected configure banner.
rep("║  Tile Y       : 4\n║  Pencil Z     : 8",
    "║  Tile Y       : 8\n║  Pencil Z     : 12",
    label="configure banner")

# 5. Warmup guidance in 5A.
rep("If you *do* want the 128³/256³/512³ sweep on purpose",
    "**Use `--warmup 100` or more for steady-state numbers.** A cold process\n"
    "with `--warmup 10` can read up to ~12% slower than the steady state\n"
    "(256³: ~1.57 ms at warmup 10 vs 1.40 ms at warmup 300). The command above\n"
    "keeps `--warmup 10` only so the output matches the block shown.\n\n"
    "If you *do* want the 128³/256³/512³ sweep on purpose",
    count=1, label="warmup guidance")

# 6. Output blocks in 5B/6B were captured at 4/8; say so once, shorten labels.
old_label = "**Measured output (reference hardware, `-DSTENCIL_TILE_Y=4 -DSTENCIL_PENCIL_Z=8`):**"
assert old_label in text, "anchor not found: 4/8 output label"
note = (
    "> **Note on tile configuration.** The 5B, 6B and 6C output blocks below\n"
    "> were captured with `TILE_Y=4, PENCIL_Z=8`, the guide's earlier Step 3\n"
    "> setting. Step 3 now builds `8/12`, which is faster for the tiled kernel\n"
    "> (512^3: 16.03 ms vs 19.91 ms, 0.72x vs 0.58x of naive). With the new\n"
    "> build expect the banner to read `Tile dims : 32 x 8 (SMEM block: 40 x 16)`\n"
    "> and `Z pencil depth : 12`, the tiled rows to be roughly 20% lower, and the\n"
    "> naive rows unchanged. The other grid sizes have not been re-measured at\n"
    "> 8/12 yet; re-run them and replace these blocks when convenient.\n\n"
)
short_label = "**Measured output (reference hardware, captured at `TILE_Y=4, PENCIL_Z=8`):**"
text = text.replace(old_label, note + short_label, 1)
text = text.replace(old_label, short_label)
print("ok: 4/8 capture note + labels")

# 7. Section 7 peak-bw: verify it already uses the real card values.
assert "--peak-bw 192" in text, "Section 7 no longer uses --peak-bw 192"
print("ok: Section 7 already uses --peak-bw 192 (no change needed)")

# 8. Changelog entry.
text = text.rstrip("\n") + (
    "\n\n**Checklist item D2:**\n\n"
    "24. **Step 3 vs Section 8B contradiction resolved.** Step 3 no longer\n"
    "    recommends `TILE_Y=4, PENCIL_Z=8` (which Section 8B and the\n"
    "    measurements show is the slowest valid configuration, 0.58x); it now\n"
    "    configures the best measured `TILE_Y=8, PENCIL_Z=12` (about 0.71x).\n"
    "    The \"mistuned A100 defaults\" explanation was removed, since tiled is\n"
    "    slower than naive at every configuration tested. Added the\n"
    "    `--warmup >= 100` steady-state advice to Section 5A. The 5B/6B/6C\n"
    "    output blocks are still the 4/8 captures and are labelled as such\n"
    "    pending a re-run at 8/12. Section 7 already used `--peak-bw 192`.\n"
)

open(path, "w", encoding="utf-8").write(text)
print("written:", path)

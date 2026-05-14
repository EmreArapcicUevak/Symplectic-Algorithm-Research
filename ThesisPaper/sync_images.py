#!/usr/bin/env python3
"""
Resolve every image referenced by Thesis.tex (direct \\includegraphics + the
custom macros \\fbplot, \\HsweepFig, \\UsweepFig, \\algoAnalysis, \\plotAnalysis),
then copy any missing ones from a source pool into ThesisPaper/Images/.

Run with no args to do a normal sync. Useful flags:
  --dry-run         show what would be copied / removed, change nothing
  --remove-orphans  delete files in ThesisPaper/Images/ that nothing references
  --source DIR ...  one or more pool directories to search recursively
                    (default: Results/Figures and Images relative to repo root)
"""
from __future__ import annotations
import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THESIS_TEX = REPO / "ThesisPaper" / "Thesis.tex"
DST_DIR = REPO / "ThesisPaper" / "Images"
DEFAULT_SOURCES = [REPO / "Results" / "Figures", REPO / "Images"]
IMAGE_EXTS = (".pdf", ".png", ".jpg", ".jpeg")


# --------------------------------------------------------------------------
# Reference extraction. Each function returns a set of basenames (no ext).
# --------------------------------------------------------------------------

def _direct_includegraphics(text: str) -> set[str]:
    """Match \\includegraphics[...]{Images/<stem>} (with optional ./).
    Stems containing a backslash are macro-template placeholders inside a
    \\newcommand body (e.g. \\AlgoName, \\kval) — skip them."""
    out = set()
    for m in re.finditer(
        r"\\includegraphics(?:\[[^\]]*\])?\{(?:\./)?Images/([^}]+)\}", text
    ):
        stem = m.group(1)
        if "\\" in stem:
            continue
        stem = re.sub(r"\.(pdf|png|jpe?g)$", "", stem, flags=re.IGNORECASE)
        out.add(stem)
    return out


def _fbplot(text: str) -> set[str]:
    """\\HsweepFig{α}{x01}{x02} and \\UsweepFig{α}{x01}{x02} expand to four
    \\fbplot{type}{N}{x01}{x02}{α} calls, N ∈ {100,200,400,800}."""
    out = set()
    for cmd, plot_type in [("HsweepFig", "hamiltonian_plot"),
                           ("UsweepFig", "control_plot")]:
        for m in re.finditer(
            rf"\\{cmd}\{{([^}}]+)\}}\{{([^}}]+)\}}\{{([^}}]+)\}}", text
        ):
            a, x01, x02 = m.group(1), m.group(2), m.group(3)
            for N in (100, 200, 400, 800):
                out.add(
                    f"{plot_type}_N={N},x₀=[{x01}, {x02}],"
                    f"x_d=[0.0, 10.0],m=1.0,k=1.0,α={a},l₀=9.0"
                )
    return out


_ALGO_PARAM_RE = re.compile(r"(\w+)\s*=\s*(\{[^}]*\}|[^,\n]+)")


def _algo_analysis(text: str) -> set[str]:
    """\\algoAnalysis[method=…, k=…, h=…, N=…, m=…, xA=…, xB=…, xC=…] produces
    a fixed pair of iteration plots, plus a (control, projection-comparison)
    triple per non-empty xA/xB/xC."""
    out = set()
    for m in re.finditer(r"\\algoAnalysis\[([^\]]+)\]", text):
        p = {}
        for km in _ALGO_PARAM_RE.finditer(m.group(1)):
            key = km.group(1).strip()
            val = km.group(2).strip().rstrip(",").strip()
            if val.startswith("{") and val.endswith("}"):
                val = val[1:-1].strip()
            p[key] = val
        try:
            method, k, h, N, mass = p["method"], p["k"], p["h"], p["N"], p["m"]
        except KeyError:
            continue
        out.add(f"IterationPlot_Modified {method}_k={k}_h={h}_N={N}_m={mass}")
        out.add(f"IterationPlot_{method}_k={k}_h={h}_N={N}_m={mass}")
        for xkey in ("xA", "xB", "xC"):
            if p.get(xkey):
                x = p[xkey]
                out.add(f"ControlPlot_Modified {method}_k={k}_h={h}_N={N}_m={mass}_x0={x}")
                out.add(f"ControlPlot_{method}_k={k}_h={h}_N={N}_m={mass}_x0={x}")
                out.add(f"Projection_Comparison_{method}_k={k}_h={h}_N={N}_m={mass}_x0={x}")
    return out


EXTRACTORS = [_direct_includegraphics, _fbplot, _algo_analysis]


def collect_referenced(tex_text: str) -> set[str]:
    """Union of every extractor's output."""
    refs = set()
    for fn in EXTRACTORS:
        refs |= fn(tex_text)
    return refs


# --------------------------------------------------------------------------
# Source pool indexing & sync
# --------------------------------------------------------------------------

def index_sources(roots: list[Path]) -> dict[str, Path]:
    """stem -> first matching file across all source roots (depth-first)."""
    index: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                # First match wins; if you want different precedence, reorder roots.
                index.setdefault(p.stem, p)
    return index


def sync(refs: set[str], sources: dict[str, Path], dry_run: bool,
         remove_orphans: bool) -> int:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    have = {p.stem: p for p in DST_DIR.iterdir() if p.is_file()}
    missing_in_dst = refs - have.keys()
    orphans = have.keys() - refs

    copied, not_found = [], []
    for stem in sorted(missing_in_dst):
        src = sources.get(stem)
        if src is None:
            not_found.append(stem)
            continue
        dst = DST_DIR / src.name
        if dry_run:
            copied.append((src, dst))
        else:
            shutil.copy2(src, dst)
            copied.append((src, dst))

    removed = []
    if remove_orphans:
        for stem in sorted(orphans):
            p = have[stem]
            if not dry_run:
                p.unlink()
            removed.append(p)

    # ---- report ----
    print(f"referenced by Thesis.tex : {len(refs)}")
    print(f"present in {DST_DIR.name}/   : {len(have)}")
    print(f"copied{' (dry-run)' if dry_run else ''}     : {len(copied)}")
    for src, dst in copied:
        print(f"  + {dst.name}    <- {src.relative_to(REPO)}")
    if not_found:
        print(f"\nMISSING IN SOURCE POOL: {len(not_found)}")
        for stem in not_found:
            print(f"  ? {stem}")
    if orphans and not remove_orphans:
        print(f"\nORPHANS (in {DST_DIR.name}/ but not referenced): {len(orphans)}")
        print("  (rerun with --remove-orphans to delete them)")
        for stem in sorted(orphans):
            print(f"  - {stem}")
    elif removed:
        print(f"\nremoved orphans{' (dry-run)' if dry_run else ''}: {len(removed)}")
        for p in removed:
            print(f"  - {p.name}")

    # nonzero exit if anything's wrong with the source pool
    return 1 if not_found else 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--remove-orphans", action="store_true")
    ap.add_argument(
        "--source", nargs="+", type=Path, default=DEFAULT_SOURCES,
        help="source pool directory/ies, searched recursively",
    )
    args = ap.parse_args()

    text = THESIS_TEX.read_text()
    refs = collect_referenced(text)
    sources = index_sources(args.source)
    print(f"source pool indexed: {len(sources)} files across {len(args.source)} root(s)")
    return sync(refs, sources, args.dry_run, args.remove_orphans)


if __name__ == "__main__":
    sys.exit(main())

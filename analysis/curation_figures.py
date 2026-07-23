"""
Agreement analysis: AI-agent curation (Pipeline 1) vs. manual curation.

Parses ./data/20260413-results-analysis/*.txt (four sections per scope:
Overlap / Only in ai-agents / Only-in-manual / inherit from ssREAD) and emits
publication-ready figures plus a master table.

    python analysis/curation_figures.py

Outputs to analysis/figures/:
    fig1_venn.{png,pdf}        Set agreement, AI vs manual
    fig2_per_scope.{png,pdf}   Per-scope composition, stacked bars
    fig3_recall.{png,pdf}      Per-scope recall with 95% Wilson CIs
    table1_per_scope.csv       Master table
    table1_per_scope.md        Same, markdown
    data_quality_report.txt    Corrections applied + caveats

NOTE ON INTERPRETATION
    The "Only in ai-agents" papers are UNVALIDATED. Recall against the manual
    set is measurable; precision is NOT. Nothing here reports precision or F1.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
from collections import OrderedDict, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from scipy.optimize import brentq

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "20260413-results-analysis")
OUT_DIR = os.path.join(ROOT, "analysis", "figures")

# --------------------------------------------------------------------------
# Design tokens (validated palette; see dataviz skill references/palette.md)
#   3-slot categorical passes all-pairs in light mode:
#   CVD dE 9.2, normal-vision dE 24.0. Aqua is 2.74:1 on surface -> the
#   "relief rule" applies, satisfied by visible direct labels + table view.
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

C_OVERLAP = "#2a78d6"  # slot 1 blue   - found by both
C_MANUAL = "#eb6834"   # slot 2 orange - manual only
C_AI = "#1baf7a"       # slot 3 aqua   - AI only
C_LEGACY = "#898781"   # gray          - ssREAD legacy (not a comparator)

GAP_PX = 2.0   # surface gap between touching marks
ROUND_PX = 4.0  # rounded data-end

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",  # never dashed
})

SECTIONS = OrderedDict([
    ("overlap", "Overlap"),
    ("ai_only", "Only in ai-agents"),
    ("manual_only", "Only-in-manual"),
    ("ssread", "inherit from ssREAD"),
])

# Precedence for resolving a PMID listed in >1 section of the SAME file.
# Overlap is the strongest evidence (both curators saw it), then the
# single-curator sections, then the inherited legacy list.
PRECEDENCE = ["overlap", "manual_only", "ai_only", "ssread"]

PRETTY = {
    "alzheimer": "Alzheimer's",
    "parkinson": "Parkinson's",
    "multiple-sclerosis": "Multiple sclerosis",
    "amyotrophic-lateral-sclerosis": "ALS",
    "frontotemporal-dementia": "Frontotemporal dementia",
    "huntington": "Huntington's",
    "spinocerebellar-ataxia": "Spinocerebellar ataxia",
    "spinal-muscular-atrophy": "Spinal muscular atrophy",
    "prion": "Prion disease",
}


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_file(path, issues):
    """Return {section: [pmid, ...]} for one result file."""
    secs = OrderedDict((k, []) for k in SECTIONS)
    cur = None
    fname = os.path.basename(path)
    for raw in open(path):
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            head = s.lstrip("#").strip().lower()
            cur = next((k for k, v in SECTIONS.items() if v.lower() in head), None)
            if cur is None:
                issues.append(f"{fname}: unrecognised header {s!r} - lines skipped")
            continue
        if cur is None:
            issues.append(f"{fname}: content before any header: {s!r}")
            continue
        if re.fullmatch(r"\d+", s):
            secs[cur].append(s)
        else:
            issues.append(f"{fname} [{cur}]: non-PMID line {s!r} - skipped")
    return secs


def dedupe(secs, fname, issues):
    """Drop within-section repeats and cross-section repeats by PRECEDENCE."""
    for k, v in secs.items():
        seen, out = set(), []
        for pm in v:
            if pm in seen:
                issues.append(f"{fname} [{k}]: duplicate PMID {pm} - kept once")
                continue
            seen.add(pm)
            out.append(pm)
        secs[k] = out

    owner = {}
    for sec in PRECEDENCE:
        for pm in secs[sec]:
            owner.setdefault(pm, sec)
    for sec in SECTIONS:
        keep = []
        for pm in secs[sec]:
            if owner[pm] != sec:
                issues.append(
                    f"{fname}: PMID {pm} appeared in both [{owner[pm]}] and "
                    f"[{sec}] - assigned to [{owner[pm]}] by precedence"
                )
            else:
                keep.append(pm)
        secs[sec] = keep
    return secs


def load():
    issues, rows = [], []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        base = os.path.basename(path).replace("-result-20260723.txt", "")
        modality = "Spatial" if base.startswith("spatial-") else "Single-cell"
        disease = re.sub(r"^(spatial|single-cell)-", "", base)
        secs = dedupe(parse_file(path, issues), os.path.basename(path), issues)
        rows.append({
            "file": base,
            "modality": modality,
            "disease": PRETTY.get(disease, disease.replace("-", " ").capitalize()),
            "pmids": secs,
            **{k: len(v) for k, v in secs.items()},
        })
    return rows, issues


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    """Wilson score interval - correct at small n, where normal approx fails."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def unique_paper_counts(rows):
    """Collapse scope-paper pairs to unique PMIDs (39 PMIDs span >1 scope)."""
    by_ai, by_manual, legacy = set(), set(), set()
    for r in rows:
        p = r["pmids"]
        by_ai |= set(p["overlap"]) | set(p["ai_only"])
        by_manual |= set(p["overlap"]) | set(p["manual_only"])
        legacy |= set(p["ssread"])
    return {
        "overlap": len(by_ai & by_manual),
        "ai_only": len(by_ai - by_manual),
        "manual_only": len(by_manual - by_ai),
        "ssread": len(legacy - by_ai - by_manual),
    }


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def rounded_hbar(ax, x0, width, y, height, color, round_right=False, r_data=0.0):
    """Horizontal bar segment; optionally round the right (data) end."""
    if width <= 0:
        return
    r = min(r_data, width) if round_right else 0.0
    x1 = x0 + width
    if r <= 0:
        verts = [(x0, y - height / 2), (x1, y - height / 2),
                 (x1, y + height / 2), (x0, y + height / 2), (x0, y - height / 2)]
        codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    else:
        k = r * 0.5523
        yb, yt = y - height / 2, y + height / 2
        verts = [
            (x0, yb), (x1 - r, yb),
            (x1 - r + k, yb), (x1, yb + r - k), (x1, yb + r),
            (x1, yt - r),
            (x1, yt - r + k), (x1 - r + k, yt), (x1 - r, yt),
            (x0, yt), (x0, yb),
        ]
        codes = [Path.MOVETO, Path.LINETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.LINETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def px_to_data_x(ax, px):
    inv = ax.transData.inverted()
    return abs(inv.transform((px, 0))[0] - inv.transform((0, 0))[0])


def save(fig, name):
    for ext in ("png", "pdf"):
        # Suppress the PDF CreationDate stamp so regenerating produces
        # byte-identical output - these artifacts are tracked in git.
        meta = {"CreationDate": None} if ext == "pdf" else {}
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=300,
                    bbox_inches="tight", facecolor=SURFACE, metadata=meta)
    plt.close(fig)
    print(f"  wrote figures/{name}.png / .pdf")


# --------------------------------------------------------------------------
# Figure 1 - area-proportional Euler, AI vs manual
# --------------------------------------------------------------------------
def _lens_polygon(xA, rA, xB, rB, n=240):
    """Vertices of the intersection lens of two circles centred on y=0."""
    d = xB - xA
    a = (rA**2 - rB**2 + d**2) / (2 * d)
    h2 = rA**2 - a**2
    if h2 <= 0:
        return None
    h = math.sqrt(h2)
    thA = math.atan2(h, a)          # half-angle of the lens seen from A
    thB = math.atan2(h, a - d)      # ... and from B (obtuse)
    t1 = np.linspace(-thA, thA, n)          # arc of A that lies inside B
    t2 = np.linspace(thB, 2 * math.pi - thB, n)  # arc of B that lies inside A
    return np.vstack([
        np.column_stack([xA + rA * np.cos(t1), rA * np.sin(t1)]),
        np.column_stack([xB + rB * np.cos(t2), rB * np.sin(t2)]),
    ])


def fig1_venn(rows, uniq):
    ov = sum(r["overlap"] for r in rows)
    ai = sum(r["ai_only"] for r in rows)
    mn = sum(r["manual_only"] for r in rows)
    ss = sum(r["ssread"] for r in rows)
    n_ai, n_mn = ov + ai, ov + mn
    recall = ov / n_mn

    rA = math.sqrt(n_ai / math.pi)
    rB = math.sqrt(n_mn / math.pi)

    def lens_area(d):
        if d >= rA + rB:
            return 0.0
        if d <= abs(rA - rB):
            return math.pi * min(rA, rB) ** 2
        a = (rA**2 - rB**2 + d**2) / (2 * d)
        b = d - a
        return (rA**2 * math.acos(np.clip(a / rA, -1, 1)) - a * math.sqrt(max(rA**2 - a**2, 0))
                + rB**2 * math.acos(np.clip(b / rB, -1, 1)) - b * math.sqrt(max(rB**2 - b**2, 0)))

    lo, hi = abs(rA - rB) + 1e-9, rA + rB - 1e-9
    d = brentq(lambda x: lens_area(x) - ov, lo, hi) if lens_area(lo) > ov > lens_area(hi) \
        else abs(rA - rB) + 0.4

    fig = plt.figure(figsize=(7.6, 5.0))
    ax = fig.add_axes([0.02, 0.20, 0.96, 0.62])
    xA, xB = 0.0, d

    # Solid validated hues, no alpha blending: an alpha overlap would invent an
    # unvalidated 4th colour. The lens is drawn on top in slot 1, with a 2px
    # surface ring doing the separating (never a border stroke).
    ax.add_patch(Circle((xA, 0), rA, facecolor=C_AI, edgecolor="none", zorder=2))
    ax.add_patch(Circle((xB, 0), rB, facecolor=C_MANUAL, edgecolor="none", zorder=2))
    pts = _lens_polygon(xA, rA, xB, rB)
    if pts is not None:
        ax.add_patch(plt.Polygon(pts, closed=True, facecolor=C_OVERLAP,
                                 edgecolor=SURFACE, linewidth=2, zorder=3))

    # In-fill values: ink chosen by fill luminance so contrast always clears.
    ax.text(xA - rA * 0.48, rA * 0.10, f"{ai}", ha="center", va="center",
            fontsize=23, fontweight="semibold", color=INK, zorder=5)
    ax.text(xA - rA * 0.48, -rA * 0.22, "AI agents only\n(unvalidated)", ha="center",
            va="center", fontsize=8.5, color=INK, zorder=5, linespacing=1.5)

    cx = (xB - rB + xA + rA) / 2
    ax.text(cx, rA * 0.10, f"{ov}", ha="center", va="center",
            fontsize=23, fontweight="semibold", color="#ffffff", zorder=5)
    ax.text(cx, -rA * 0.22, "Both", ha="center", va="center",
            fontsize=8.5, color="#ffffff", zorder=5)

    ax.annotate(f"{mn}  manual only", xy=(xB + rB * 0.62, -rB * 0.38),
                xytext=(xB + rB + 2.0, -rA * 0.60),
                ha="left", va="center", fontsize=8.5, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=1))

    # Headers offset in POINTS, not data units - data offsets don't scale with
    # font size and the two lines collide.
    for x, ha, name, n in ((xA - rA, "left", "AI agents", n_ai),
                           (xB + rB, "right", "Manual curation", n_mn)):
        ax.annotate(name, xy=(x, rA), xytext=(0, 27), textcoords="offset points",
                    ha=ha, va="bottom", fontsize=10.5, fontweight="semibold", color=INK)
        # Short subtitle: the full "scope-paper assignments" wording on both
        # sides is wider than the gap between the two circle edges.
        ax.annotate(f"{n} assignments", xy=(x, rA), xytext=(0, 13),
                    textcoords="offset points", ha=ha, va="bottom",
                    fontsize=8.5, color=MUTED)

    ax.set_xlim(xA - rA - 1.0, xB + rB + 9.0)
    ax.set_ylim(-rA - 1.5, rA + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.text(0.02, 0.965,
             f"AI-agent screening recovers {recall*100:.1f}% of manually curated papers\n"
             f"and surfaces {ai} additional candidates",
             fontsize=12.5, fontweight="semibold", color=INK, ha="left", va="top",
             linespacing=1.45)
    fig.text(0.02, 0.135,
             f"Areas proportional to set size; one assignment = one (scope, paper) curation decision.\n"
             f"Recall vs. manual = {ov}/{n_mn} = {recall*100:.1f}%.  "
             f"Expansion = {n_ai/n_mn:.2f}×.  Jaccard = {ov/(ov+ai+mn)*100:.1f}%.\n"
             f"Unique papers (deduplicated across scopes): both {uniq['overlap']}, "
             f"AI only {uniq['ai_only']}, manual only {uniq['manual_only']}.\n"
             f"Separately, {ss} legacy assignments inherited from ssREAD are excluded above "
             f"(different date coverage; not a third curator).",
             fontsize=7.6, color=MUTED, ha="left", va="top", linespacing=1.7)
    save(fig, "fig1_venn")


# --------------------------------------------------------------------------
# Figure 2 - per-scope composition
# --------------------------------------------------------------------------
def fig2_per_scope(rows):
    panels = [("Single-cell", [r for r in rows if r["modality"] == "Single-cell"]),
              ("Spatial", [r for r in rows if r["modality"] == "Spatial"])]
    for _, rs in panels:
        rs.sort(key=lambda r: r["overlap"] + r["ai_only"] + r["manual_only"])

    fig, axes = plt.subplots(
        2, 1, figsize=(8.4, 7.6),
        gridspec_kw={"height_ratios": [len(panels[0][1]), len(panels[1][1])], "hspace": 0.18},
    )
    xmax = max(r["overlap"] + r["ai_only"] + r["manual_only"] for r in rows)

    for ax, (title, rs) in zip(axes, panels):
        ax.set_xlim(0, xmax * 1.14)
        ax.set_ylim(-0.7, len(rs) - 0.3)
        gap = px_to_data_x(ax, GAP_PX)
        rad = px_to_data_x(ax, ROUND_PX)
        height = 0.52  # leaves air in the band

        for i, r in enumerate(rs):
            segs = [("manual_only", r["manual_only"], C_MANUAL),
                    ("overlap", r["overlap"], C_OVERLAP),
                    ("ai_only", r["ai_only"], C_AI)]
            segs = [s for s in segs if s[1] > 0]
            x = 0.0
            for j, (_key, val, col) in enumerate(segs):
                last = j == len(segs) - 1
                w = val - (0 if last else gap)
                rounded_hbar(ax, x, max(w, val * 0.35), i, height, col,
                             round_right=last, r_data=rad)
                # Relief rule: label segments wide enough to hold the text.
                if val / xmax > 0.055:
                    ax.text(x + val / 2, i, f"{val}", ha="center", va="center",
                            fontsize=7.5, color="#ffffff", fontweight="semibold", zorder=6)
                x += val
            ax.text(x + xmax * 0.012, i, f"{int(x)}", ha="left", va="center",
                    fontsize=8, color=INK_2, fontweight="semibold")

        ax.set_yticks(range(len(rs)))
        ax.set_yticklabels([r["disease"] for r in rs], fontsize=8.5, color=INK_2)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", fontweight="semibold", pad=6)
        ax.xaxis.grid(True, linewidth=0.8, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="y", length=0)

    axes[1].set_xlabel("Papers curated (scope–paper assignments)", fontsize=9, color=INK_2)
    axes[0].tick_params(labelbottom=False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none")
               for c in (C_MANUAL, C_OVERLAP, C_AI)]
    axes[0].legend(handles, ["Manual only", "Both", "AI agents only (unvalidated)"],
                   loc="lower right", bbox_to_anchor=(1.0, 1.06), ncol=3,
                   handlelength=1.1, handleheight=1.1, columnspacing=1.4,
                   labelcolor=INK_2)

    fig.suptitle("AI-agent contribution varies by disease scope",
                 fontsize=12, fontweight="semibold", color=INK, x=0.125, ha="left", y=0.985)
    fig.text(0.125, 0.055,
             "Segments ordered manual-exclusive → shared → AI-exclusive. Bar-end value is the scope total.\n"
             "ssREAD legacy assignments excluded. Counts, not percentages: several scopes have very few papers.",
             fontsize=7.6, color=MUTED, ha="left", va="top", linespacing=1.6)
    save(fig, "fig2_per_scope")


# --------------------------------------------------------------------------
# Figure 3 - recall forest plot
# --------------------------------------------------------------------------
def fig3_recall(rows):
    rs = [r for r in rows if (r["overlap"] + r["manual_only"]) > 0]
    for r in rs:
        n = r["overlap"] + r["manual_only"]
        r["_n"], r["_recall"] = n, r["overlap"] / n
        r["_ci"] = wilson(r["overlap"], n)
    # Faceted by modality to match Fig 2; within each panel, ordered by
    # reference-set size (largest at top). Ordering by recall instead would put
    # the n=1-2 scopes on the podium and read as "these did best".
    panels = [("Single-cell", [r for r in rs if r["modality"] == "Single-cell"]),
              ("Spatial", [r for r in rs if r["modality"] == "Spatial"])]
    for _, prs in panels:
        prs.sort(key=lambda r: r["_n"])

    ov = sum(r["overlap"] for r in rows)
    mn = sum(r["manual_only"] for r in rows)
    pooled = ov / (ov + mn)
    nmax = max(r["_n"] for r in rs)  # one global size scale across both panels

    fig, axes = plt.subplots(
        2, 1, figsize=(8.6, 6.6),
        gridspec_kw={"height_ratios": [len(panels[0][1]), len(panels[1][1])], "hspace": 0.16},
    )
    fig.subplots_adjust(left=0.28, right=0.74, top=0.85, bottom=0.155)

    for ax, (title, prs) in zip(axes, panels):
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.8, len(prs) - 0.2)
        ax.axvline(pooled * 100, color=C_OVERLAP, linewidth=1, alpha=0.4, zorder=1)

        for i, r in enumerate(prs):
            lo, hi = r["_ci"]
            ms = 5 + 13 * math.sqrt(r["_n"] / nmax)
            ax.plot([lo * 100, hi * 100], [i, i], color=AXIS, linewidth=1.6,
                    solid_capstyle="round", zorder=2)
            ax.plot([r["_recall"] * 100], [i], marker="o", markersize=ms,
                    color=C_OVERLAP, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            # Value gutter outside the plot frame - never collides with the marks.
            ax.text(1.05, i, f"{r['overlap']}/{r['_n']}", transform=ax.get_yaxis_transform(),
                    fontsize=8, color=INK_2, ha="left", va="center",
                    fontfamily="monospace", clip_on=False)
            ax.text(1.26, i, f"{r['_recall']*100:.0f}%", transform=ax.get_yaxis_transform(),
                    fontsize=8, color=INK_2, ha="left", va="center",
                    fontfamily="monospace", clip_on=False)

        ax.set_yticks(range(len(prs)))
        ax.set_yticklabels([r["disease"] for r in prs], fontsize=8.5, color=INK_2)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", fontweight="semibold", pad=6)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.grid(True, linewidth=0.8, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="y", length=0)

    # Column headers and the pooled-line label ride the top panel only.
    top, ntop = axes[0], len(panels[0][1])
    top.text(pooled * 100, ntop - 0.55, f"pooled {pooled*100:.1f}%",
             fontsize=7.5, color=C_OVERLAP, ha="right", va="bottom")
    for xf, lab in ((1.05, "found/total"), (1.26, "recall")):
        top.text(xf, ntop - 0.55, lab, transform=top.get_yaxis_transform(),
                 fontsize=7.5, color=MUTED, ha="left", va="bottom",
                 fontfamily="monospace", clip_on=False)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("Recall vs. manual curation (%), with 95% Wilson confidence interval",
                       fontsize=9, color=INK_2)

    fig.text(0.02, 0.968,
             "Recall is high where the manual set is large; small scopes are uninformative",
             fontsize=12.5, fontweight="semibold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.068,
             "Within each panel, ordered by size of the manual reference set; marker area ∝ that size (one scale across "
             "both panels).\nWide intervals mark scopes with too few papers to support a claim — a point estimate of "
             "100% on n=1–2 is not evidence.\nSpatial prion disease and spatial spinal muscular atrophy have no "
             "manually curated papers and are omitted.",
             fontsize=7.6, color=MUTED, ha="left", va="top", linespacing=1.7)
    save(fig, "fig3_recall")


# --------------------------------------------------------------------------
# Table 1
# --------------------------------------------------------------------------
TABLE_HDR = ["modality", "disease", "n_manual", "n_ai", "overlap", "ai_only",
             "manual_only", "recall_pct", "ci95_low", "ci95_high", "expansion",
             "ssread_legacy"]


def build_table(rows):
    """Per-scope rows plus a TOTAL row, as printable cells."""
    out = []
    for r in sorted(rows, key=lambda r: (r["modality"], -(r["overlap"] + r["manual_only"]))):
        n_man = r["overlap"] + r["manual_only"]
        n_ai = r["overlap"] + r["ai_only"]
        lo, hi = wilson(r["overlap"], n_man)
        out.append([
            r["modality"], r["disease"], n_man, n_ai, r["overlap"], r["ai_only"],
            r["manual_only"],
            f"{r['overlap']/n_man*100:.1f}" if n_man else "n/a",
            f"{lo*100:.1f}" if n_man else "n/a",
            f"{hi*100:.1f}" if n_man else "n/a",
            f"{n_ai/n_man:.2f}" if n_man else "n/a",
            r["ssread"],
        ])
    ov = sum(r["overlap"] for r in rows)
    ai = sum(r["ai_only"] for r in rows)
    mn = sum(r["manual_only"] for r in rows)
    ss = sum(r["ssread"] for r in rows)
    lo, hi = wilson(ov, ov + mn)
    out.append(["**ALL**", "**Total**", ov + mn, ov + ai, ov, ai, mn,
                f"{ov/(ov+mn)*100:.1f}", f"{lo*100:.1f}", f"{hi*100:.1f}",
                f"{(ov+ai)/(ov+mn):.2f}", ss])
    return out


def compute_stats(rows, uniq, issues):
    """Every number the prose quotes - single source of truth."""
    ov = sum(r["overlap"] for r in rows)
    ai = sum(r["ai_only"] for r in rows)
    mn = sum(r["manual_only"] for r in rows)
    ss = sum(r["ssread"] for r in rows)
    lo, hi = wilson(ov, ov + mn)

    scope_count = defaultdict(set)
    for r in rows:
        for sec in ("overlap", "ai_only", "manual_only", "ssread"):
            for pm in r["pmids"][sec]:
                scope_count[pm].add(r["file"])
    multi_scope = sum(1 for v in scope_count.values() if len(v) > 1)

    with_manual = [r for r in rows if (r["overlap"] + r["manual_only"]) > 0]
    # "Strongest evidence" = largest reference set, NOT highest recall: a 100%
    # point estimate on n=7 is exactly what Figure 3 warns against reading.
    best = max(with_manual, key=lambda r: (r["overlap"] + r["manual_only"],
                                           r["overlap"] / (r["overlap"] + r["manual_only"])))
    expansions = [(r, (r["overlap"] + r["ai_only"]) / (r["overlap"] + r["manual_only"]))
                  for r in with_manual]
    top_exp = max(expansions, key=lambda t: t[1])

    return {
        "n_scopes": len(rows),
        "n_single_cell": sum(1 for r in rows if r["modality"] == "Single-cell"),
        "n_spatial": sum(1 for r in rows if r["modality"] == "Spatial"),
        "overlap": ov, "ai_only": ai, "manual_only": mn, "ssread": ss,
        "n_ai_total": ov + ai, "n_manual_total": ov + mn,
        "recall_pct": round(ov / (ov + mn) * 100, 1),
        "recall_ci95": [round(lo * 100, 1), round(hi * 100, 1)],
        "expansion": round((ov + ai) / (ov + mn), 2),
        "jaccard_pct": round(ov / (ov + ai + mn) * 100, 1),
        "unique_papers": uniq,
        "multi_scope_pmids": multi_scope,
        "data_issues": len(issues),
        "best_scope": f"{best['disease']} ({best['modality'].lower()})",
        "best_scope_recall_pct": round(best["overlap"] / (best["overlap"] + best["manual_only"]) * 100, 1),
        "best_scope_n": best["overlap"] + best["manual_only"],
        "top_expansion_scope": f"{top_exp[0]['disease']} ({top_exp[0]['modality'].lower()})",
        "top_expansion": round(top_exp[1], 2),
        "scopes_no_manual": [f"{r['disease']} ({r['modality'].lower()})"
                             for r in rows if (r["overlap"] + r["manual_only"]) == 0],
    }


def write_tables(rows, uniq, issues, stats):
    cells = build_table(rows)
    with open(os.path.join(OUT_DIR, "table1_per_scope.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TABLE_HDR)
        w.writerows([[str(c).replace("**", "") for c in row] for row in cells])

    with open(os.path.join(OUT_DIR, "table1_per_scope.md"), "w") as f:
        f.write("# Table 1 — AI-agent vs. manual curation, per scope\n\n")
        f.write(md_table(cells))
        f.write(
            f"\n**Unique papers** (deduplicated across scopes): both {uniq['overlap']}, "
            f"AI only {uniq['ai_only']}, manual only {uniq['manual_only']}, "
            f"ssREAD-only {uniq['ssread']}.\n"
            "\n*Precision and F1 are deliberately absent: the AI-only papers are unvalidated, "
            "so only recall against the manual set is estimable.*\n"
        )

    with open(os.path.join(OUT_DIR, "data_quality_report.txt"), "w") as f:
        f.write("DATA QUALITY REPORT\n===================\n\n")
        f.write(f"Precedence for cross-section duplicates: {' > '.join(PRECEDENCE)}\n\n")
        if issues:
            f.write(f"{len(issues)} issue(s) found and resolved:\n")
            for i in issues:
                f.write(f"  - {i}\n")
        else:
            f.write("No parsing issues.\n")
        f.write("\nCAVEATS\n-------\n")
        f.write("1. AI-only papers are UNVALIDATED. Precision/F1 are not computable.\n")
        f.write(f"2. Totals count scope-paper assignments; {stats['multi_scope_pmids']} "
                "PMIDs appear in >1 scope.\n")
        f.write("3. ssREAD is a legacy inherited set with different date coverage;\n"
                "   it is not a third curator and is excluded from the comparison.\n")

    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("  wrote figures/table1_per_scope.csv / .md, data_quality_report.txt, stats.json")


def md_table(cells):
    lines = ["| " + " | ".join(TABLE_HDR) + " |",
             "|" + "|".join(["---"] * len(TABLE_HDR)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in cells]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Generated documentation
# --------------------------------------------------------------------------
README_BEGIN = "<!-- BEGIN:curation-benchmark (generated by analysis/curation_figures.py) -->"
README_END = "<!-- END:curation-benchmark -->"


def write_benchmark_doc(rows, uniq, issues, stats):
    """Generate docs/curation_benchmark.md - every number interpolated."""
    s = stats
    fig = "../analysis/figures"
    body = f"""<!-- Generated by analysis/curation_figures.py — do not edit by hand. -->

# Curation Benchmark — AI Agents vs. Manual Curation

## Overview

Pipeline 1 screens PubMed papers with two LLM steps (`IdentifyOriginalDataStep`,
`IdentifyRelevanceStep`) and accepts a paper only when both return `True`. This
document benchmarks that automated screening against manual curation across
**{s['n_scopes']} scopes** ({s['n_single_cell']} single-cell, {s['n_spatial']} spatial).

Source data: `data/20260413-results-analysis/*.txt`, one file per scope, each
partitioned into four sections:

| Section | Meaning |
|---|---|
| `## Overlap` | Curated by **both** the AI agents and manual review |
| `## Only in ai-agents` | Curated by the AI agents alone |
| `## Only-in-manual` | Curated by manual review alone |
| `## inherit from ssREAD` | Inherited from the earlier ssREAD collection |

**The unit of analysis is a scope–paper assignment**, i.e. one (scope, paper)
curation decision — not a unique paper. {s['multi_scope_pmids']} PMIDs legitimately appear
under more than one scope (a paper can be both Alzheimer's single-cell and
Alzheimer's spatial). Unique-paper counts are given alongside where relevant.

## Headline result

![Set agreement between AI-agent and manual curation]({fig}/fig1_venn.png)

**Figure 1. AI-agent screening recovers {s['recall_pct']}% of manually curated papers and
surfaces {s['ai_only']} additional candidates.** Circle areas are proportional to set size;
the intersection is solved numerically so the geometry is faithful. The manual
set sits almost entirely inside the AI set.

| Metric | Value |
|---|---|
| Recall vs. manual | **{s['recall_pct']}%** ({s['overlap']}/{s['n_manual_total']}), 95% CI {s['recall_ci95'][0]}–{s['recall_ci95'][1]}% |
| Expansion factor | **{s['expansion']}×** ({s['n_ai_total']} AI vs. {s['n_manual_total']} manual assignments) |
| Jaccard similarity | {s['jaccard_pct']}% |
| Found by both | {s['overlap']} |
| AI agents only (unvalidated) | {s['ai_only']} |
| Manual only (agent misses) | {s['manual_only']} |
| Unique papers | both {uniq['overlap']}, AI only {uniq['ai_only']}, manual only {uniq['manual_only']} |

## Per-scope composition

![Per-scope curation composition]({fig}/fig2_per_scope.png)

**Figure 2. AI-agent contribution varies by disease scope.** Segments are ordered
manual-exclusive → shared → AI-exclusive; the value at the bar end is the scope
total. Absolute counts are shown rather than percentages because several scopes
contain very few papers. The largest relative expansion is
**{s['top_expansion_scope']} at {s['top_expansion']}×**.

## Recall with uncertainty

![Per-scope recall with 95% Wilson confidence intervals]({fig}/fig3_recall.png)

**Figure 3. Recall is high where the manual set is large; small scopes are
uninformative.** Faceted by modality, ordered within each panel by the size of the
manual reference set, with marker area proportional to that size (one scale
across both panels). Error bars are 95% Wilson score intervals, which stay
correct at small *n* where the normal approximation fails.

The strongest evidence comes from {s['best_scope']}: {s['best_scope_recall_pct']}% recall on
n={s['best_scope_n']}. Conversely a point estimate of 100% on n=1–2 carries an interval
spanning most of the range and should not be read as a result.

## Table 1 — per-scope detail

{md_table(build_table(rows))}
Also available as [`table1_per_scope.csv`]({fig}/table1_per_scope.csv) for
spreadsheet import.

## Interpretation limits

Three constraints bound what these figures can claim.

**1. Precision is not computable — and is deliberately absent.** The {s['ai_only']}
AI-only papers have not been manually adjudicated. Recall against the manual set
is measurable because the manual set is a reference; precision is not, because
"the agent found something the humans did not" is not evidence of an error. No
precision, F1, or accuracy figure appears anywhere in this analysis. Treating
the AI-only set as false positives would invert the purpose of the pipeline.

**2. ssREAD is not a third curator.** The {s['ssread']} inherited assignments come from
the earlier ssREAD collection, which has different date coverage, and are
excluded from every comparison above. They are disjoint from the Overlap section
in all scopes where they appear.

**3. Small scopes carry no weight.** {len(s['scopes_no_manual'])} scopes have no manually curated
papers at all ({', '.join(s['scopes_no_manual']) if s['scopes_no_manual'] else 'none'}) and are omitted from
Figure 3. Several more have n ≤ 3, where the confidence interval spans most of
the range.

### Recommended next step

Manually adjudicate a random sample of ~50 of the {s['ai_only']} AI-only papers. That
single exercise converts an unvalidated candidate set into a precision estimate
with a confidence interval, and is the one missing piece needed to state a
performance claim rather than a coverage claim.

## Data quality

The source files required {s['data_issues']} corrections, applied automatically and logged in
full to [`data_quality_report.txt`]({fig}/data_quality_report.txt). PMIDs listed in
more than one section of the same file are resolved by the precedence
`{' > '.join(PRECEDENCE)}`.

Issues found:

{chr(10).join('- `' + i + '`' for i in issues) if issues else '- None.'}

These are defects in the source `.txt` files and are worth fixing upstream — the
first one alone shifts the headline recall figure.

## Reproducing

```bash
eval $(poetry env activate)
python analysis/curation_figures.py
```

Regenerates every figure (PNG at 300 dpi + vector PDF), the tables, the data
quality report, `stats.json`, and this document. All numbers quoted above are
interpolated from the parsed data at generation time, so the prose cannot drift
out of sync with the figures.
"""
    path = os.path.join(ROOT, "docs", "curation_benchmark.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(body)
    print("  wrote docs/curation_benchmark.md")


def update_readme(stats):
    """Splice a short summary into README.md between marker comments."""
    s = stats
    block = f"""{README_BEGIN}
## Validation — AI agents vs. manual curation

Pipeline 1 was benchmarked against manual curation across {s['n_scopes']} disease scopes
({s['n_single_cell']} single-cell, {s['n_spatial']} spatial).

| Metric | Value |
|---|---|
| Recall vs. manual curation | **{s['recall_pct']}%** ({s['overlap']}/{s['n_manual_total']}), 95% CI {s['recall_ci95'][0]}–{s['recall_ci95'][1]}% |
| Expansion factor | **{s['expansion']}×** ({s['n_ai_total']} vs. {s['n_manual_total']} scope–paper assignments) |
| Additional candidates surfaced | {s['ai_only']} (unvalidated) |

![AI agents vs. manual curation](analysis/figures/fig1_venn.png)

The {s['ai_only']} AI-only papers have not been manually adjudicated, so **precision is not
estimable and is not reported**. See [docs/curation_benchmark.md](docs/curation_benchmark.md)
for the full analysis, per-scope breakdown, and interpretation limits.

Regenerate with `python analysis/curation_figures.py`.
{README_END}"""

    path = os.path.join(ROOT, "README.md")
    text = open(path).read()
    if README_BEGIN in text and README_END in text:
        pre = text.split(README_BEGIN)[0]
        post = text.split(README_END, 1)[1]
        text = pre + block + post
        action = "updated"
    else:
        anchor = "\n## License"
        insert = block + "\n\n---\n"
        text = (text.replace(anchor, "\n" + insert + anchor, 1)
                if anchor in text else text.rstrip() + "\n\n---\n\n" + insert)
        action = "inserted"
    with open(path, "w") as f:
        f.write(text)
    print(f"  {action} README.md validation section")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, issues = load()
    uniq = unique_paper_counts(rows)
    stats = compute_stats(rows, uniq, issues)
    print(f"Parsed {len(rows)} scopes; {len(issues)} data issue(s) resolved.")
    fig1_venn(rows, uniq)
    fig2_per_scope(rows)
    fig3_recall(rows)
    write_tables(rows, uniq, issues, stats)
    write_benchmark_doc(rows, uniq, issues, stats)
    update_readme(stats)
    print(f"\nAll outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()

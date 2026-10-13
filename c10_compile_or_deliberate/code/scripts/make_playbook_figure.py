"""Figure 4 of the paper: an annotated, verbatim excerpt of the seed-1 playbook (25k-token checkpoint).

Reads results/synth/main_qwen3-coder-30b_s1/playbook_B25000.py and writes paper/figures/playbook_excerpt.tex (TikZ; needs xcolor, tikz and
fontawesome5). The code is reproduced character for character: selected rule blocks are shown with their original line
numbers, omitted rules are marked, long lines are soft-wrapped (marked with a hook arrow in the line-number gutter),
and each annotated rule gets a numbered callout. The callouts restate findings reported in Section 6.4.
  python code/scripts/make_playbook_figure.py
"""
from __future__ import annotations

import os
import re

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(PROJ, "results", "synth", "main_qwen3-coder-30b_s1", "playbook_B25000.py")
OUT = os.path.join(os.environ.get("COD_FIG", os.path.join(PROJ, "paper", "figures")), "playbook_excerpt.tex")

SHOW = [14, 1, 11, 12, 16]  # rule numbers shown, in file order
CALLOUTS = {  # rule -> (colour, icon, title, text)
    14: ("pOff", "sliders-h", "Calibrated to the top",
         "Moved to 1st place by CPU calibration; it, not rule 16, rescues composition upsets."),
    1: ("pSis", "power-off", "Shutdown near a limit",
        "Shuts the plant down when a trip limit is under 0.25\\,h away, which the endpoint scores as a failure."),
    11: ("pOn", "arrow-down", "Production cut",
         "Lowers the production-rate setpoint by 2\\,m$^3$/h at every rate-deviation alarm."),
    12: ("pTx", "exchange-alt", "Substitution, re-issued",
         "Swaps the pressure transmitter for its soft sensor; the memory flag it tests is never set."),
    16: ("pDead", "ban", "Never fires",
         "Needs a $-10$\\,mol\\% feed-composition deviation, which never occurs in operation."),
}
OMIT_NOTE = {  # omitted run of rules after a shown rule -> description
    1: "rules 2--10: alarm responses (high-high, high and level alarms, grade deviation)",
    12: "rules 13 and 15: temperature-transmitter substitution, stuck cooling-water valve",
    16: "rules 17--19: feed loss, temperature drift, default no action",
}
WRAP = 74            # characters per code line before a soft wrap
LS = 7.4             # line spacing (pt)
CW = 9.85            # width of the code panel (cm)
GUT = 0.62           # line-number gutter (cm)
CH = 0.525 * 6.4 / 28.4527  # width of one typewriter character at 6.4 pt (cm)

KW = r"\b(def|if|and|or|not|in|return)\b"
BI = r"\b(min|max|abs)\b"
TOKEN = re.compile(rf"(?P<com>#.*$)|(?P<str>\"[^\"]*\")|(?P<kw>{KW})|(?P<bi>{BI})|(?P<num>(?<![\w.])-?\d+(\.\d+)?)")
STYLE = {"com": "\\color{pCom}\\itshape ", "str": "\\color{pStr}", "kw": "\\color{pKw}", "bi": "\\color{pBi}",
         "num": "\\color{pNum}", "txt": ""}


def esc(s: str) -> str:
    out = []
    for ch in s:
        out.append({"\\": "\\textbackslash{}", "{": "\\{", "}": "\\}", "_": "\\_", "#": "\\#", "%": "\\%",
                    "&": "\\&", "$": "\\$", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}", " ": "\\ ",
                    "-": "-{}"}.get(ch, ch))
    return "".join(out)


def segments(line: str) -> list[tuple[str, str]]:
    """(style, text) runs of one source line."""
    segs, pos = [], 0
    for m in TOKEN.finditer(line):
        if m.start() > pos:
            segs.append(("txt", line[pos:m.start()]))
        segs.append((m.lastgroup if m.lastgroup in STYLE else "txt", m.group(0)))
        pos = m.end()
    if pos < len(line):
        segs.append(("txt", line[pos:]))
    return segs


def wrap(line: str) -> list[tuple[int, int]]:
    """Character spans of the display lines of one source line (soft wrap at spaces)."""
    indent = len(line) - len(line.lstrip(" "))
    spans, start, width = [], 0, WRAP
    while len(line) - start > width:
        cut = line.rfind(" ", start + 1, start + width)
        cut = start + width if cut <= start else cut + 1
        spans.append((start, cut))
        start, width = cut, WRAP - indent - 4
    spans.append((start, len(line)))
    return spans


def render(line: str, a: int, b: int) -> str:
    out, pos = [], 0
    for style, text in segments(line):
        lo, hi = max(pos, a), min(pos + len(text), b)
        if lo < hi:
            piece = esc(line[lo:hi])
            out.append("{" + STYLE[style] + piece + "}" if STYLE[style] else piece)
        pos += len(text)
    return "".join(out)


def main():
    src = open(SRC).read().split("\n")
    starts = {int(m.group(1)): i for i, l in enumerate(src) if (m := re.match(r"\s*# Rule (\d+):", l))}
    order = sorted(starts, key=starts.get)
    rows = [("code", 3, src[2], 0, len(src[2]), None)]  # def line
    blocks = {}
    for r in SHOW:
        i = starts[r]
        j = i + 1
        while j < len(src) and src[j].strip() and not src[j].lstrip().startswith("# Rule"):
            j += 1
        first = len(rows)
        for k in range(i, j):
            for n, (a, b) in enumerate(wrap(src[k])):
                rows.append(("code", k + 1 if n == 0 else None, src[k], a, b, r))
        blocks[r] = (first, len(rows) - 1)
        if r in OMIT_NOTE:
            rows.append(("omit", None, OMIT_NOTE[r], 0, 0, None))
    L = []
    L.append("%% GENERATED by code/scripts/make_playbook_figure.py from results/synth/main_qwen3-coder-30b_s1/playbook_B25000.py -- do not edit")
    L.append("\\definecolor{pOff}{HTML}{2A78D6}\\definecolor{pOn}{HTML}{EB6834}\\definecolor{pSis}{HTML}{C0392B}"
             "\\definecolor{pTx}{HTML}{7E57C2}\\definecolor{pDead}{HTML}{7A7A7A}")
    L.append("\\definecolor{pKw}{HTML}{1F4E9E}\\definecolor{pBi}{HTML}{0F7C7C}\\definecolor{pStr}{HTML}{A23C1C}"
             "\\definecolor{pNum}{HTML}{8A4D00}\\definecolor{pCom}{HTML}{4F7A45}\\definecolor{pInk}{HTML}{1A1A1A}")
    L.append("\\begin{tikzpicture}[x=1cm, y=1pt, font=\\sffamily,"
             " code/.style={anchor=base west, inner sep=0pt, font=\\ttfamily\\fontsize{6.4}{7.4}\\selectfont, text=pInk},"
             " num/.style={anchor=base east, inner sep=0pt, font=\\ttfamily\\fontsize{5.6}{6}\\selectfont, text=pInk!45}]")
    n = len(rows)
    top, bot = 12.0, -(n - 1) * LS - 5.0
    L.append(f"\\fill[pInk!3, rounded corners=4pt] (0,{bot:.1f}) rectangle ({CW},{top + 11:.1f});")
    L.append(f"\\fill[pInk!14, rounded corners=4pt] (0,{top:.1f}) rectangle ({CW},{top + 11:.1f});"
             f"\\fill[pInk!14] (0,{top:.1f}) rectangle ({CW},{top + 5:.1f});")
    L.append(f"\\fill[pInk!7] (0,{bot + 3:.1f}) rectangle ({GUT},{top:.1f});")
    L.append(f"\\draw[pInk!35, line width=0.5pt, rounded corners=4pt] (0,{bot:.1f}) rectangle ({CW},{top + 11:.1f});")
    L.append(f"\\node[anchor=west, font=\\sffamily\\fontsize{{6.8}}{{7.5}}\\selectfont\\bfseries, text=pInk!80] at (0.15,{top + 5.5:.1f})"
             " {\\faIcon{file-code}\\,\\,Seed-1 playbook, checkpoint at 25k tokens (excerpt)};")
    L.append(f"\\node[anchor=east, font=\\sffamily\\fontsize{{6}}{{7}}\\selectfont, text=pInk!65] at ({CW - 0.15},{top + 5.5:.1f})"
             " {Python, first-match rules};")
    # block backgrounds
    for r, (a, b) in blocks.items():
        col = CALLOUTS[r][0]
        y0, y1 = -a * LS + 6.3, -b * LS - 2.4
        L.append(f"\\fill[{col}!8] ({GUT + 0.02},{y1:.1f}) rectangle ({CW - 0.02},{y0:.1f});")
        L.append(f"\\draw[{col}, line width=1.4pt] ({GUT + 0.04},{y1:.1f}) -- ({GUT + 0.04},{y0:.1f});")
    for k, (kind, lno, text, a, b, r) in enumerate(rows):
        y = -k * LS
        if kind == "omit":
            L.append(f"\\node[num, font=\\fontsize{{7}}{{7}}\\selectfont] at ({GUT - 0.12},{y:.1f}) {{$\\vdots$}};")
            L.append(f"\\node[anchor=base west, inner sep=0pt, font=\\sffamily\\itshape\\fontsize{{6.3}}{{7}}\\selectfont, text=pInk!55]"
                     f" at ({GUT + 0.18},{y:.1f}) {{{text} (omitted)}};")
            continue
        indent = len(text) - len(text.lstrip(" "))
        if lno is not None:
            L.append(f"\\node[num] at ({GUT - 0.12},{y:.1f}) {{{lno}}};")
            x = GUT + 0.15
            body = render(text, a, b)
        else:  # soft-wrapped continuation: indent + 4, hook in the gutter
            L.append(f"\\node[num, text=pInk!40] at ({GUT - 0.1},{y + 0.5:.1f}) {{$\\hookrightarrow$}};")
            x = GUT + 0.15 + (indent + 4) * CH
            a += len(text[a:b]) - len(text[a:b].lstrip(" "))  # a continuation never starts with a space
            body = render(text, a, b)
        L.append(f"\\node[code] at ({x:.3f},{y:.1f}) {{{body}}};")
    # callouts: badge on the right border of the code panel, card in the right column. Each card starts level with its
    # badge unless the previous card reaches below that point; TikZ measures the previous card (let ... in).
    cx = CW + 0.38
    for idx, r in enumerate(SHOW, start=1):
        col, icon, title, text = CALLOUTS[r]
        a, b = blocks[r]
        ybadge = -a * LS + 2.2
        L.append(f"\\node[circle, fill={col}, text=white, inner sep=0pt, minimum size=3.6mm, font=\\sffamily\\bfseries\\fontsize{{6.2}}{{6.2}}\\selectfont]"
                 f" (b{idx}) at ({CW},{ybadge:.1f}) {{{idx}}};")
        ypos = f"{ybadge + 6.0:.1f}" if idx == 1 else f"{{min({ybadge + 6.0:.1f}, \\y1 - 4)}}"
        head = "\\path " if idx == 1 else f"\\path let \\p1 = (c{idx - 1}.south) in "
        L.append(head + f"node[anchor=north west, draw={col}!80!black, fill={col}!7, rounded corners=3pt, line width=0.6pt,"
                 f" text width={13.7 - CW - 0.38 - 0.22:.2f}cm, inner sep=3pt, align=flush left, font=\\sffamily\\fontsize{{6.3}}{{7.3}}\\selectfont, text=pInk!90]"
                 f" (c{idx}) at ({cx},{ypos}) {{{{\\color{{{col}}}\\fontsize{{8}}{{8}}\\selectfont\\faIcon{{{icon}}}}}\\,\\,"
                 f"{{\\bfseries\\fontsize{{6.8}}{{7.6}}\\selectfont {idx}\\;{title}}}\\\\[1pt]{text}}};")
        L.append(f"\\draw[{col}, line width=0.9pt] (b{idx}.east) -- ++(0.1,0) |- ([yshift=-7pt]c{idx}.north west);")
    L.append("\\end{tikzpicture}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}: {n} display lines")


if __name__ == "__main__":
    main()

# QRQ Copy Performance Analyzer

Analyze your QRQ (Morse code) summary logs and produce both a detailed text report and a polished, presentation‑ready HTML dashboard — no external Python packages required.

## What It Does

- Parses QRQ summary `*.txt` logs (as exported by the QRQ app).
- Computes overall accuracy, speeds, operator breakdowns, and session extremes.
- Shows daily trends (accuracy and average WPM) to visualize progress over time.
- Breaks down errors by type (substitution, omission, insertion) with examples.
- Ranks characters by error rate and shows per‑character error vs. speed.
- Aggregates accuracy into speed bands with a training recommendation.
- Renders a polished HTML dashboard with SVG charts and executive summary cards.

## Quick Start

1) Place `qrq_report.py` with your QRQ logs (on my computer this is /Users/username/.qrq/Summary), or pass a directory.

- Default run in current directory:

```bash
python3 qrq_report.py
```

- Specify logs directory and output locations:

```bash
python3 qrq_report.py /path/to/logs \
  -o /path/to/qrq_analysis_report.txt \
  --html-output /path/to/html/qrq_analysis_report.html
```

- Skip HTML (text report only):

```bash
python3 qrq_report.py --no-html
```

The script writes a text summary (`qrq_analysis_report.txt`) and, by default, a full HTML dashboard to `html/qrq_analysis_report.html` (creating the `html/` folder if needed).

## Requirements

- Python 3.8+ (standard library only). No third‑party dependencies.
- QRQ summary text logs (the exported table format with header line like `QRQ attempt by <CALL>.`).

## Input Format (Expected)

The script expects the QRQ summary layout, e.g.:

```
QRQ attempt by S53ZO.

Sent call        Input            Difference       CpM WpM Score F6
--------------------------------------------------------------------
G3XJS            G3XJH            G3XJh            190  38   380 *
...
--------------------------------------------------------------------
Score: 64064, Max. speed (CpM/WpM): 220 / 44
Saved at: 20250926_1033
```

Notes:
- Asterisk (`*`) marks F6 (assist) on that row — the script counts these.
- Parsers tolerate spacing variations and handle the trailing `*` next to `Score` values.

## Outputs

- Text report: `qrq_analysis_report.txt`
  - Overall metrics, operator breakdown
  - Best/worst sessions
  - Daily trend table
  - Error taxonomy and examples
  - Character error rates and per‑character vs. speed
  - Speed‑band accuracy and recommendation

- HTML report: `html/qrq_analysis_report.html`
  - Executive summary cards
  - Daily Accuracy and Daily Average WPM line charts (SVG)
  - Accuracy by Speed Band bar chart (SVG)
  - Top Character Error Rates (horizontal bar chart, SVG)
  - Tables for speed bands and error taxonomy

## How It Works (Brief)

- Reads every `*.txt` in the target directory.
- Parses table rows, normalizes case, and extracts CpM, WpM, Score, and F6.
- Uses Levenshtein alignment to categorize character‑level operations (match, substitution, deletion, insertion) for per‑character error statistics.
- Aggregates metrics per day, per operator, per speed band, and per character.
- Renders a plain‑text summary and a styled HTML dashboard (SVG charts built with the standard library).

## Tips

- Keep accuracy ≥60% in a speed band before advancing; the report highlights where accuracy drops.
- Use the character vs. speed section to pick a few high‑impact letters/digits (e.g., S, B, 7, V, 2, `/`) for targeted drills.

## Troubleshooting

- Report is empty or some rows missing: ensure the files are QRQ summary exports and contain the six data columns (Sent, Input, Difference, CpM, WpM, Score). Rows with malformed numbers are skipped.
- HTML not generated: remove `--no-html` or provide a valid `--html-output` path. The script creates the folder if needed.

## Contributing

Issues and PRs to improve parsing, add exporters (CSV), or extend visuals are welcome.


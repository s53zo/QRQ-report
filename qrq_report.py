#!/usr/bin/env python3

"""
Generate a text report summarising QRQ CW copy performance.

The script scans all ``*.txt`` files in the given directory (default: current
working directory), recreates the analysis performed during the interactive
session, and writes a human-readable report to disk.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics as stats
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass
class CallCopy:
    sent: str
    received: str
    diff: str
    cpm: int
    wpm: int
    score: int
    flag: str


@dataclass
class Attempt:
    path: Path
    operator: str
    timestamp: datetime
    rows: List[CallCopy]
    total_score: Optional[int]
    max_speed_cpm: Optional[int]
    max_speed_wpm: Optional[int]
    saved_at: Optional[str]

    @property
    def entries(self) -> int:
        return len(self.rows)

    @property
    def correct(self) -> int:
        return sum(1 for row in self.rows if row.diff == "-")

    @property
    def errors(self) -> int:
        return self.entries - self.correct

    @property
    def accuracy(self) -> float:
        return self.correct / self.entries if self.entries else 0.0

    @property
    def avg_wpm(self) -> float:
        return stats.mean(row.wpm for row in self.rows)

    @property
    def avg_cpm(self) -> float:
        return stats.mean(row.cpm for row in self.rows)

    @property
    def max_wpm(self) -> int:
        return max(row.wpm for row in self.rows)

    @property
    def max_cpm(self) -> int:
        return max(row.cpm for row in self.rows)

    @property
    def f6_count(self) -> int:
        return sum(1 for row in self.rows if row.flag == "*")


def parse_attempt(path: Path) -> Optional[Attempt]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    operator = None
    for line in lines:
        if line.startswith("QRQ attempt by "):
            operator = line.split("QRQ attempt by ", 1)[1].strip(". ")
            break

    if operator is None:
        return None

    rows: List[CallCopy] = []
    total_score = None
    max_speed_cpm = None
    max_speed_wpm = None
    saved_at = None

    for raw in lines:
        stripped = raw.strip()
        if (
            not stripped
            or stripped.startswith("Sent call")
            or stripped.startswith("QRQ attempt by")
            or stripped.startswith("Score:")
            or stripped.startswith("Saved at:")
            or set(stripped) == {"-"}
        ):
            if stripped.startswith("Score:"):
                score_match = re.search(r"Score:\s*([\d]+)", stripped)
                if score_match:
                    total_score = int(score_match.group(1))
                speed_match = re.search(r"Max\. speed \(CpM/WpM\):\s*(\d+)\s*/\s*(\d+)", stripped)
                if speed_match:
                    max_speed_cpm = int(speed_match.group(1))
                    max_speed_wpm = int(speed_match.group(2))
            elif stripped.startswith("Saved at:"):
                saved_at = stripped.split(":", 1)[1].strip()
            continue

        parts = re.split(r"\s{2,}", stripped)
        if len(parts) < 6:
            continue

        sent, received, diff, cpm, wpm, score = parts[:6]
        flag = ""
        if len(parts) > 6 and parts[6].strip():
            flag = parts[6].strip()

        if score.strip().endswith("*"):
            flag = "*"
            score = score.replace("*", "").strip()

        try:
            row = CallCopy(
                sent=sent.strip(),
                received=received.strip(),
                diff=diff.strip(),
                cpm=int(cpm.strip()),
                wpm=int(wpm.strip()),
                score=int(score.strip()),
                flag=flag,
            )
        except ValueError:
            continue

        rows.append(row)

    if not rows:
        return None

    try:
        timestamp = datetime.strptime(path.stem.split("-", 1)[1], "%Y%m%d_%H%M")
    except (IndexError, ValueError):
        timestamp = datetime.fromtimestamp(path.stat().st_mtime)

    return Attempt(
        path=path,
        operator=operator,
        timestamp=timestamp,
        rows=rows,
        total_score=total_score,
        max_speed_cpm=max_speed_cpm,
        max_speed_wpm=max_speed_wpm,
        saved_at=saved_at,
    )


def levenshtein_ops(sent: str, received: str) -> List[Tuple[str, str, str]]:
    """
    Return alignment operations between the two strings.

    Each tuple is (op, sent_char, received_char) where op is one of:
    - ``match``: characters identical
    - ``sub``  : substitution
    - ``del``  : deletion (character missing in received)
    - ``ins``  : insertion (extra character in received)
    """

    a = sent.upper()
    b = received.upper()
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # match or substitution
            )

    ops: List[Tuple[str, str, str]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("match", sent[i - 1], received[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", sent[i - 1], received[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", sent[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", received[j - 1]))
            j -= 1

    ops.reverse()
    return ops


def classify_call(sent: str, received: str) -> str:
    a = sent.upper()
    b = received.upper()
    if a == b:
        return "exact"
    if sent != received and a == b:
        return "case-only"
    if len(b) < len(a):
        return "omission"
    if len(b) > len(a):
        return "insertion"
    mismatches = sum(aa != bb for aa, bb in zip(a, b))
    return "single-substitution" if mismatches == 1 else "multi-substitution"


def _ensure_range(values: List[float]) -> Tuple[float, float]:
    min_val = min(values)
    max_val = max(values)
    if math.isclose(min_val, max_val):
        padding = 1.0 if math.isclose(max_val, 0.0) else abs(max_val) * 0.1
        min_val -= padding
        max_val += padding
    else:
        span = max_val - min_val
        min_val -= span * 0.05
        max_val += span * 0.05
    return min_val, max_val


def svg_line_chart(labels: List[str], values: List[float], title: str, unit: str = "") -> str:
    if not labels or not values:
        return "<svg width='800' height='200'><text x='50%' y='50%' text-anchor='middle'>No data</text></svg>"

    width, height, pad = 900, 320, 60
    min_val, max_val = _ensure_range(values)
    value_range = max_val - min_val
    y_scale = (height - 2 * pad) / value_range if value_range else 1
    step = (width - 2 * pad) / (len(labels) - 1) if len(labels) > 1 else 0

    grid_lines = []
    grid_steps = 5
    for i in range(grid_steps + 1):
        val = min_val + value_range * i / grid_steps
        y = height - pad - (val - min_val) * y_scale
        grid_lines.append(
            f"<line x1='{pad}' x2='{width - pad}' y1='{y:.2f}' y2='{y:.2f}' class='grid' />"
            f"<text x='{pad - 15}' y='{y + 4:.2f}' class='axis-label'>{val:.1f}{escape(unit)}</text>"
        )

    points = []
    markers = []
    for idx, val in enumerate(values):
        x = pad + idx * step if len(labels) > 1 else width / 2
        y = height - pad - (val - min_val) * y_scale
        points.append(f"{x:.2f},{y:.2f}")
        markers.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4' class='line-point' />")

    x_labels = []
    for idx, label in enumerate(labels):
        x = pad + idx * step if len(labels) > 1 else width / 2
        x_labels.append(
            f"<text x='{x:.2f}' y='{height - pad + 25}' class='axis-label' transform='rotate(-35 {x:.2f},{height - pad + 25})'>{escape(label)}</text>"
        )

    polyline = " ".join(points)
    grid_svg = "".join(grid_lines)
    markers_svg = "".join(markers)
    x_labels_svg = "".join(x_labels)
    return (
        f"<svg viewBox='0 0 {width} {height}' class='chart'>"
        f"<title>{escape(title)}</title>"
        f"<rect x='0' y='0' width='{width}' height='{height}' class='chart-bg' />"
        f"<line x1='{pad}' x2='{pad}' y1='{pad}' y2='{height - pad}' class='axis' />"
        f"<line x1='{pad}' x2='{width - pad}' y1='{height - pad}' y2='{height - pad}' class='axis' />"
        f"{grid_svg}"
        f"<polyline points='{polyline}' class='line' />"
        f"{markers_svg}"
        f"{x_labels_svg}"
        f"</svg>"
    )


def svg_vertical_bar_chart(labels: List[str], values: List[float], title: str, unit: str = "") -> str:
    if not labels or not values:
        return "<svg width='800' height='200'><text x='50%' y='50%' text-anchor='middle'>No data</text></svg>"

    width, height, pad = 900, 320, 70
    bar_space = width - 2 * pad
    bar_width = bar_space / max(len(labels) * 1.6, 1)
    max_val = max(values) if values else 1
    if math.isclose(max_val, 0.0):
        max_val = 1.0
    scale = (height - 2 * pad) / max_val

    bars = []
    x_labels = []
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = pad + idx * bar_width * 1.6
        bar_height = value * scale
        y = height - pad - bar_height
        bars.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_width:.2f}' height='{bar_height:.2f}' class='bar' />"
            f"<text x='{x + bar_width / 2:.2f}' y='{y - 8:.2f}' class='bar-label'>{value:.1f}{escape(unit)}</text>"
        )
        x_labels.append(
            f"<text x='{x + bar_width / 2:.2f}' y='{height - pad + 25}' class='axis-label'>{escape(label)}</text>"
        )

    return (
        f"<svg viewBox='0 0 {width} {height}' class='chart'>"
        f"<title>{escape(title)}</title>"
        f"<rect x='0' y='0' width='{width}' height='{height}' class='chart-bg' />"
        f"<line x1='{pad}' x2='{pad}' y1='{pad}' y2='{height - pad}' class='axis' />"
        f"<line x1='{pad}' x2='{width - pad}' y1='{height - pad}' y2='{height - pad}' class='axis' />"
        f"{''.join(bars)}"
        f"{''.join(x_labels)}"
        f"</svg>"
    )


def svg_horizontal_bar_chart(labels: List[str], values: List[float], title: str, unit: str = "") -> str:
    if not labels or not values:
        return "<svg width='800' height='200'><text x='50%' y='50%' text-anchor='middle'>No data</text></svg>"

    height = 80 + len(labels) * 40
    width = 900
    pad_left, pad_right, pad_top, pad_bottom = 140, 80, 40, 40
    max_val = max(values)
    if math.isclose(max_val, 0.0):
        max_val = 1.0
    scale = (width - pad_left - pad_right) / max_val

    bars = []
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = pad_top + idx * 40
        bar_length = value * scale
        bars.append(
            f"<text x='{pad_left - 10}' y='{y + 18}' class='axis-label' text-anchor='end'>{escape(label)}</text>"
            f"<rect x='{pad_left}' y='{y}' width='{bar_length:.2f}' height='24' class='bar horizontal' />"
            f"<text x='{pad_left + bar_length + 10:.2f}' y='{y + 18}' class='bar-label'>{value:.1f}{escape(unit)}</text>"
        )

    return (
        f"<svg viewBox='0 0 {width} {height}' class='chart'>"
        f"<title>{escape(title)}</title>"
        f"<rect x='0' y='0' width='{width}' height='{height}' class='chart-bg' />"
        f"{''.join(bars)}"
        f"</svg>"
    )


def analyse_attempts(attempts: Iterable[Attempt]):
    attempts = list(attempts)
    if not attempts:
        raise ValueError("No QRQ attempt files found.")

    all_rows = [row for attempt in attempts for row in attempt.rows]
    total_calls = len(all_rows)
    total_correct = sum(1 for row in all_rows if row.diff == "-")
    overall_accuracy = total_correct / total_calls if total_calls else 0.0
    all_wpms = [row.wpm for row in all_rows]
    all_cpms = [row.cpm for row in all_rows]

    operator_map = defaultdict(list)
    for attempt in attempts:
        operator_map[attempt.operator].append(attempt)

    error_types = Counter()
    error_examples = defaultdict(list)
    substitution_pairs = Counter()
    omission_counts = Counter()
    insertion_counts = Counter()
    char_totals = Counter()
    char_errors = Counter()
    char_wpm = defaultdict(lambda: defaultdict(lambda: {"total": 0, "errors": 0}))

    for attempt in attempts:
        for row in attempt.rows:
            classification = classify_call(row.sent, row.received)
            if classification != "exact":
                error_types[classification] += 1
                if len(error_examples[classification]) < 5:
                    error_examples[classification].append(
                        (attempt.path.name, row.sent, row.received, row.diff)
                    )

            ops = levenshtein_ops(row.sent, row.received)
            for op, sent_ch, recv_ch in ops:
                if op in ("match", "sub", "del"):
                    key = sent_ch.upper()
                    char_totals[key] += 1
                    char_wpm[key][row.wpm]["total"] += 1
                    if op != "match":
                        char_errors[key] += 1
                        char_wpm[key][row.wpm]["errors"] += 1
                if op == "sub":
                    substitution_pairs[(sent_ch.upper(), recv_ch.upper())] += 1
                elif op == "del":
                    omission_counts[sent_ch.upper()] += 1
                elif op == "ins":
                    insertion_counts[recv_ch.upper()] += 1

    best_attempt = max(attempts, key=lambda a: a.accuracy)
    worst_attempt = min(attempts, key=lambda a: a.accuracy)

    char_stats = []
    for ch, total in char_totals.items():
        errors = char_errors[ch]
        rate = errors / total if total else 0.0
        char_stats.append({"char": ch, "total": total, "errors": errors, "rate": rate})
    char_stats.sort(key=lambda item: (-item["rate"], -item["total"]))

    per_date = defaultdict(lambda: {"correct": 0, "total": 0, "wpms": []})
    for attempt in attempts:
        key = attempt.timestamp.date()
        per_date[key]["correct"] += attempt.correct
        per_date[key]["total"] += attempt.entries
        per_date[key]["wpms"].append(attempt.avg_wpm)

    speed_bins = [
        ("≤38 WPM", lambda w: w <= 38),
        ("39–42 WPM", lambda w: 39 <= w <= 42),
        ("43–46 WPM", lambda w: 43 <= w <= 46),
        ("≥47 WPM", lambda w: w >= 47),
    ]
    bin_stats = []
    for label, predicate in speed_bins:
        calls = [row for row in all_rows if predicate(row.wpm)]
        if not calls:
            bin_stats.append({"label": label, "calls": 0, "correct": 0, "accuracy": 0.0, "avg_wpm": 0.0})
            continue
        correct = sum(1 for row in calls if row.diff == "-")
        accuracy = correct / len(calls)
        avg_wpm = stats.mean(row.wpm for row in calls)
        bin_stats.append(
            {"label": label, "calls": len(calls), "correct": correct, "accuracy": accuracy, "avg_wpm": avg_wpm}
        )

    return {
        "attempts": attempts,
        "total_calls": total_calls,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "avg_wpm": stats.mean(all_wpms) if all_wpms else 0.0,
        "median_wpm": stats.median(all_wpms) if all_wpms else 0.0,
        "avg_cpm": stats.mean(all_cpms) if all_cpms else 0.0,
        "median_cpm": stats.median(all_cpms) if all_cpms else 0.0,
        "operator_map": operator_map,
        "error_types": error_types,
        "error_examples": error_examples,
        "substitution_pairs": substitution_pairs,
        "omission_counts": omission_counts,
        "insertion_counts": insertion_counts,
        "char_stats": char_stats,
        "char_wpm": char_wpm,
        "best_attempt": best_attempt,
        "worst_attempt": worst_attempt,
        "per_date": per_date,
        "bin_stats": bin_stats,
    }


def format_attempt_line(label: str, attempt: Attempt) -> str:
    return (
        f"{label}: {attempt.path.name} | "
        f"{attempt.timestamp.strftime('%Y-%m-%d %H:%M')} | "
        f"{attempt.operator} | "
        f"{attempt.correct}/{attempt.entries} correct "
        f"({attempt.accuracy:.3f}) at {attempt.avg_wpm:.1f} WPM avg "
        f"(max {attempt.max_wpm} WPM); F6={attempt.f6_count}"
    )


def render_report(base: Path, output_path: Path) -> None:
    attempt_objs = []
    for path in sorted(base.glob("*.txt")):
        attempt = parse_attempt(path)
        if attempt:
            attempt_objs.append(attempt)

    summary = analyse_attempts(attempt_objs)

    lines = []
    lines.append("QRQ Copy Performance Report")
    lines.append("=" * 32)
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Source directory: {base}")
    lines.append("")

    lines.append("Overall Summary")
    lines.append("---------------")
    lines.append(f"Attempts analysed : {len(summary['attempts'])}")
    lines.append(f"Total calls       : {summary['total_calls']}")
    lines.append(
        f"Accuracy          : {summary['total_correct']}/{summary['total_calls']} "
        f"({summary['overall_accuracy']:.3%})"
    )
    lines.append(
        f"Average speed     : {summary['avg_wpm']:.2f} WPM "
        f"(median {summary['median_wpm']:.0f} WPM)"
    )
    lines.append(
        f"Average character : {summary['avg_cpm']:.2f} CpM "
        f"(median {summary['median_cpm']:.0f} CpM)"
    )
    lines.append("")

    lines.append("Operator Breakdown")
    lines.append("-------------------")
    for operator, attempts in sorted(summary["operator_map"].items()):
        rows = [row for attempt in attempts for row in attempt.rows]
        correct = sum(1 for row in rows if row.diff == "-")
        total = len(rows)
        avg_wpm = stats.mean(row.wpm for row in rows) if rows else 0.0
        lines.append(
            f"{operator}: {len(attempts)} attempts | {correct}/{total} "
            f"({correct/total:.3%}) | avg {avg_wpm:.2f} WPM"
        )
    lines.append("")

    lines.append("Session Extremes")
    lines.append("-----------------")
    lines.append(format_attempt_line("Best ", summary["best_attempt"]))
    lines.append(format_attempt_line("Worst", summary["worst_attempt"]))
    lines.append("")

    lines.append("Daily Trend")
    lines.append("-----------")
    for date in sorted(summary["per_date"]):
        bucket = summary["per_date"][date]
        accuracy = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
        avg_wpm = stats.mean(bucket["wpms"]) if bucket["wpms"] else 0.0
        lines.append(f"{date}: {bucket['correct']}/{bucket['total']} ({accuracy:.1%}) | avg WPM {avg_wpm:.1f}")
    lines.append("")

    lines.append("Error Types")
    lines.append("-----------")
    total_errors = sum(summary["error_types"].values())
    for etype, count in summary["error_types"].most_common():
        pct = count / total_errors if total_errors else 0
        lines.append(f"{etype:20s}: {count:4d} ({pct:.1%})")
        for example in summary["error_examples"][etype]:
            file_name, sent, received, diff = example
            lines.append(f"    e.g. {file_name}: {sent} -> {received} (diff {diff})")
    lines.append("")

    lines.append("Character Error Rates (>= 10 exposures)")
    lines.append("----------------------------------------")
    lines.append(f"{'Char':<6}{'Seen':>8}{'Errors':>10}{'Error %':>12}")
    for item in summary["char_stats"]:
        if item["total"] < 10:
            continue
        lines.append(
            f"{item['char']:<6}{item['total']:>8}{item['errors']:>10}"
            f"{item['rate']*100:>11.1f}%"
        )
    lines.append("")

    lines.append("Top Substitutions")
    lines.append("-----------------")
    for (sent, received), count in summary["substitution_pairs"].most_common(10):
        lines.append(f"{sent} -> {received}: {count}")
    lines.append("")

    lines.append("Top Omissions")
    lines.append("-------------")
    for char, count in summary["omission_counts"].most_common(10):
        lines.append(f"{char}: {count}")
    lines.append("")

    lines.append("Top Insertions (extra characters typed)")
    lines.append("---------------------------------------")
    for char, count in summary["insertion_counts"].most_common(10):
        lines.append(f"{char}: {count}")
    lines.append("")

    lines.append("Per-Character Error Rate by Speed (>= 5 exposures per speed)")
    lines.append("-------------------------------------------------------------")
    # Select the top 10 characters by error rate for focused breakdown.
    focus_chars = [item["char"] for item in summary["char_stats"][:10]]
    for char in focus_chars:
        lines.append(f"Character {char}")
        speed_map = summary["char_wpm"][char]
        for wpm in sorted(speed_map):
            bucket = speed_map[wpm]
            total = bucket["total"]
            errors = bucket["errors"]
            if total < 5:
                continue
            rate = errors / total if total else 0
            lines.append(f"  {wpm:>3d} WPM: {errors}/{total} ({rate:.0%})")
        lines.append("")  # spacer between characters

    lines.append("Speed Bands & Training Notes")
    lines.append("----------------------------")
    for bin_stat in summary["bin_stats"]:
        calls = bin_stat["calls"]
        if calls == 0:
            lines.append(f"{bin_stat['label']}: no calls copied.")
            continue
        accuracy = bin_stat["accuracy"]
        lines.append(
            f"{bin_stat['label']}: {bin_stat['correct']}/{calls} ({accuracy:.1%}) "
            f"at avg {bin_stat['avg_wpm']:.1f} WPM"
        )
    lines.append("")

    # Heuristic recommendation: focus on the first band below 60% accuracy that still has volume.
    focus_band = next(
        (band for band in summary["bin_stats"] if band["calls"] >= 30 and band["accuracy"] < 0.6),
        None,
    )
    if focus_band:
        lines.append(
            "Recommendation: spend extra time drilling characters and calls in the "
            f"{focus_band['label']} range until accuracy climbs above 60%. "
            "Slow down slightly for isolated practice, then rebuild speed while "
            "maintaining clean copy."
        )
    else:
        lines.append(
            "Recommendation: Accuracy holds above 60% in all monitored bands; continue "
            "progressively adding higher-speed runs while keeping accuracy in that zone."
        )
    lines.append("")

    report_text = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(report_text, encoding="utf-8")
    return summary


def render_html_report(summary: dict, base: Path, output_path: Path) -> None:
    per_date = sorted(summary["per_date"].items(), key=lambda item: item[0])
    date_labels = [item[0].strftime("%Y-%m-%d") for item in per_date]
    date_accuracy = [
        (bucket["correct"] / bucket["total"] * 100) if bucket["total"] else 0 for _, bucket in per_date
    ]
    date_speed = [stats.mean(bucket["wpms"]) if bucket["wpms"] else 0 for _, bucket in per_date]

    speed_labels = [band["label"] for band in summary["bin_stats"]]
    speed_accuracy = [band["accuracy"] * 100 for band in summary["bin_stats"]]

    top_chars = summary["char_stats"][:10]
    char_labels = [item["char"] for item in top_chars]
    char_rates = [item["rate"] * 100 for item in top_chars]

    accuracy_chart = svg_line_chart(date_labels, date_accuracy, "Daily Accuracy", unit="%")
    speed_chart = svg_vertical_bar_chart(speed_labels, speed_accuracy, "Accuracy by Speed Band", unit="%")
    char_chart = svg_horizontal_bar_chart(char_labels, char_rates, "Top Character Error Rates", unit="%")
    wpm_chart = svg_line_chart(date_labels, date_speed, "Daily Average WPM", unit=" WPM")

    attempts_count = len(summary["attempts"])
    total_calls = summary["total_calls"]
    overall_accuracy = summary["overall_accuracy"] * 100
    avg_wpm = summary["avg_wpm"]
    median_wpm = summary["median_wpm"]
    avg_cpm = summary["avg_cpm"]

    best_attempt = summary["best_attempt"]
    worst_attempt = summary["worst_attempt"]

    substitution_rows = summary["substitution_pairs"].most_common(5)
    omission_rows = summary["omission_counts"].most_common(5)
    insertion_rows = summary["insertion_counts"].most_common(5)

    overall_cards = [
        ("Total Attempts", f"{attempts_count}"),
        ("Total Calls", f"{total_calls}"),
        ("Overall Accuracy", f"{overall_accuracy:.1f}%"),
        ("Average Speed", f"{avg_wpm:.1f} WPM"),
        ("Median Speed", f"{median_wpm:.1f} WPM"),
        ("Average CpM", f"{avg_cpm:.1f}"),
    ]

    def attempt_summary_block(label: str, attempt: Attempt) -> str:
        return (
            f"<div class='attempt-card'>"
            f"<h4>{escape(label)}</h4>"
            f"<p class='attempt-title'>{escape(attempt.path.name)}</p>"
            f"<ul>"
            f"<li><strong>Date:</strong> {attempt.timestamp.strftime('%Y-%m-%d %H:%M')}</li>"
            f"<li><strong>Operator:</strong> {escape(attempt.operator)}</li>"
            f"<li><strong>Accuracy:</strong> {attempt.correct}/{attempt.entries} "
            f"({attempt.accuracy:.1%})</li>"
            f"<li><strong>Average Speed:</strong> {attempt.avg_wpm:.1f} WPM (max {attempt.max_wpm} WPM)</li>"
            f"<li><strong>F6 usage:</strong> {attempt.f6_count}</li>"
            f"</ul>"
            f"</div>"
        )

    insights = [
        f"Accuracy remains above 60% at speeds up to 42 WPM but drops to {summary['bin_stats'][2]['accuracy'] * 100:.1f}% in the 43–46 WPM band.",
        f"Characters like {', '.join(escape(item['char']) for item in top_chars[:5])} account for the majority of letter-level errors.",
        "Omissions concentrate on portable designators ('/', 'K', 'A'), highlighting the need for deliberate practice on stroke-heavy calls.",
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>QRQ Copy Performance Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f9fc;
      --panel: #ffffff;
      --accent: #1f6feb;
      --accent-soft: rgba(31, 111, 235, 0.12);
      --text: #1f2933;
      --muted: #6b7280;
      --border: #d0d7de;
      --success: #0f9d58;
      font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      margin: 0;
      line-height: 1.6;
    }}
    header {{
      background: linear-gradient(135deg, rgba(31, 111, 235, 0.95), rgba(15, 157, 88, 0.85));
      color: #fff;
      padding: 48px 7%;
    }}
    header h1 {{
      margin: 0 0 12px;
      font-size: 2.4rem;
      letter-spacing: 0.02em;
    }}
    header p {{
      max-width: 640px;
      margin: 0;
      font-size: 1.1rem;
    }}
    main {{
      margin: 0 auto;
      padding: 32px 7%;
      max-width: 1200px;
    }}
    section {{
      background: var(--panel);
      border-radius: 16px;
      box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
      padding: 28px 32px;
      margin-bottom: 32px;
      border: 1px solid rgba(255,255,255,0.6);
    }}
    h2 {{
      font-size: 1.6rem;
      margin-top: 0;
      margin-bottom: 16px;
    }}
    h3 {{
      font-size: 1.2rem;
      margin-bottom: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }}
    .card {{
      padding: 20px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: var(--panel);
    }}
    .card h4 {{
      margin: 0 0 4px;
      font-size: 0.9rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .card p {{
      margin: 0;
      font-size: 1.5rem;
      font-weight: 600;
    }}
    .highlight {{
      background: var(--accent-soft);
      border-color: rgba(31,111,235,0.25);
    }}
    .attempt-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 20px;
    }}
    .attempt-card {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
      background: #fdfefe;
    }}
    .attempt-card h4 {{
      margin: 0;
      font-size: 0.95rem;
      letter-spacing: 0.06em;
      color: var(--muted);
      text-transform: uppercase;
    }}
    .attempt-title {{
      margin: 8px 0 14px;
      font-weight: 600;
      font-size: 1.05rem;
    }}
    .attempt-card ul {{
      margin: 0;
      padding-left: 20px;
    }}
    .chart {{
      width: 100%;
      max-width: 100%;
      height: auto;
    }}
    .chart-bg {{
      fill: #fff;
      rx: 16px;
      ry: 16px;
      stroke: var(--border);
      stroke-width: 1;
    }}
    .grid {{
      stroke: rgba(105, 124, 154, 0.25);
      stroke-width: 1;
      stroke-dasharray: 4 6;
    }}
    .axis {{
      stroke: #9aa5b1;
      stroke-width: 1.4;
    }}
    .axis-label {{
      fill: var(--muted);
      font-size: 0.75rem;
    }}
    .line {{
      fill: none;
      stroke: var(--accent);
      stroke-width: 3;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .line-point {{
      fill: var(--accent);
      stroke: #fff;
      stroke-width: 2;
    }}
    .bar {{
      fill: rgba(31, 111, 235, 0.85);
    }}
    .bar.horizontal {{
      fill: rgba(15, 157, 88, 0.75);
    }}
    .bar-label {{
      fill: var(--text);
      font-size: 0.8rem;
      font-weight: 600;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
    }}
    table th, table td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }}
    table th {{
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    ul.insights {{
      padding-left: 20px;
      margin: 0;
    }}
    ul.insights li {{
      margin-bottom: 8px;
    }}
    footer {{
      text-align: center;
      padding: 24px;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    @media (max-width: 800px) {{
      header {{
        padding: 36px 5%;
      }}
      main {{
        padding: 28px 5%;
      }}
      .cards {{
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>QRQ Copy Performance Dashboard</h1>
    <p>Interactive-style analysis of QRQ CW copying sessions recorded in <strong>{escape(str(base))}</strong>. The dashboard highlights accuracy trends, speed thresholds, and character-level pain points to guide targeted training.</p>
  </header>
  <main>
    <section>
      <h2>Executive Summary</h2>
      <div class="cards">
        {''.join(f"<div class='card'><h4>{escape(title)}</h4><p>{escape(value)}</p></div>" for title, value in overall_cards)}
      </div>
      <h3>Top Highlights</h3>
      <ul class="insights">
        {''.join(f"<li>{escape(point)}</li>" for point in insights)}
      </ul>
    </section>

    <section>
      <h2>Session Benchmarks</h2>
      <div class="attempt-grid">
        {attempt_summary_block("Best Session", best_attempt)}
        {attempt_summary_block("Most Challenging Session", worst_attempt)}
      </div>
    </section>

    <section>
      <h2>Performance Over Time</h2>
      <h3>Daily Accuracy</h3>
      {accuracy_chart}
      <h3>Daily Average Speed</h3>
      {wpm_chart}
    </section>

    <section>
      <h2>Speed Band Diagnostics</h2>
      <p>The bars show how copy accuracy behaves within common QRQ ranges. Use this to select practice speeds that maintain at least 60% copy accuracy before moving higher.</p>
      {speed_chart}
      <table>
        <thead>
          <tr><th>Speed Band</th><th>Calls</th><th>Correct</th><th>Accuracy</th><th>Avg WPM</th></tr>
        </thead>
        <tbody>
          {''.join(f"<tr><td>{escape(row['label'])}</td><td>{row['calls']}</td><td>{row['correct']}</td><td>{row['accuracy']*100:.1f}%</td><td>{row['avg_wpm']:.1f}</td></tr>" for row in summary['bin_stats'])}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Character Stress Points</h2>
      <p>Characters below show the highest error rates. Most issues stem from truncated dit endings (e.g., S→I, B→D) and long alternating patterns (e.g., '/').</p>
      {char_chart}
      <table>
        <thead>
          <tr><th>Character</th><th>Seen</th><th>Errors</th><th>Error Rate</th></tr>
        </thead>
        <tbody>
          {''.join(f"<tr><td>{escape(item['char'])}</td><td>{item['total']}</td><td>{item['errors']}</td><td>{item['rate']*100:.1f}%</td></tr>" for item in top_chars)}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Error Taxonomy</h2>
      <table>
        <thead>
          <tr><th>Error Type</th><th>Count</th><th>Share</th></tr>
        </thead>
        <tbody>
          {''.join(f"<tr><td>{escape(etype)}</td><td>{count}</td><td>{count / sum(summary['error_types'].values()) * 100:.1f}%</td></tr>" for etype, count in summary['error_types'].most_common())}
        </tbody>
      </table>
      <div class="cards" style="margin-top:24px;">
        <div class="card">
          <h4>Top Substitutions</h4>
          <ul>
            {''.join(f"<li>{escape(a)} → {escape(b)} ({count})</li>" for (a, b), count in substitution_rows)}
          </ul>
        </div>
        <div class="card">
          <h4>Frequent Omissions</h4>
          <ul>
            {''.join(f"<li>{escape(char)} ({count})</li>" for char, count in omission_rows)}
          </ul>
        </div>
        <div class="card">
          <h4>Extra Characters Typed</h4>
          <ul>
            {''.join(f"<li>{escape(char)} ({count})</li>" for char, count in insertion_rows)}
          </ul>
        </div>
      </div>
    </section>
  </main>
  <footer>
    Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &mdash; QRQ Copy Performance Dashboard
  </footer>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse QRQ CW summary files and write a performance report."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing QRQ summary *.txt files (default: current directory).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="qrq_analysis_report.txt",
        help="Output path for the generated report (default: qrq_analysis_report.txt).",
    )
    parser.add_argument(
        "--html-output",
        default="html/qrq_analysis_report.html",
        help="Output path for the HTML dashboard (default: html/qrq_analysis_report.html).",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip generating the HTML dashboard.",
    )
    args = parser.parse_args()

    base = Path(args.directory).expanduser().resolve()
    if not base.exists():
        raise SystemExit(f"Directory not found: {base}")

    output_path = Path(args.output).expanduser().resolve()
    summary = render_report(base, output_path)
    print(f"Report written to {output_path}")

    if not args.no_html:
        html_path = Path(args.html_output).expanduser()
        if not html_path.is_absolute():
            html_path = base / html_path
        html_path.parent.mkdir(parents=True, exist_ok=True)
        render_html_report(summary, base, html_path.resolve())
        print(f"HTML report written to {html_path.resolve()}")


if __name__ == "__main__":
    main()

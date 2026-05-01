#!/usr/bin/env python3
"""Convert fallback summary.json logs into TensorBoard event files."""

import argparse
import json
import math
import time
from pathlib import Path

try:
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
except Exception as exc:
    raise SystemExit(
        "TensorBoard Python package is required to write tfevents. "
        "Run this script with a Python interpreter that can import `tensorboard`."
    ) from exc


def _is_scalar(value):
    return isinstance(value, (int, float, bool)) and not isinstance(value, bool) or isinstance(value, bool)


def _flatten_scalars(tag, value):
    if _is_scalar(value):
        yield tag, float(value)
        return

    if isinstance(value, dict):
        for child_key, child_value in value.items():
            child_key = str(child_key)
            if child_key == tag or "/" in child_key:
                next_tag = child_key
            elif tag:
                next_tag = f"{tag}/{child_key}"
            else:
                next_tag = child_key
            yield from _flatten_scalars(next_tag, child_value)


def _sorted_steps(summary_data):
    def sort_key(item):
        key = item[0]
        try:
            return (0, int(key))
        except Exception:
            return (1, str(key))

    return sorted(summary_data.items(), key=sort_key)


def convert(summary_path, output_dir):
    summary_path = Path(summary_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    writer = EventFileWriter(str(output_dir))

    scalar_count = 0
    skipped_count = 0
    step_count = 0

    try:
        for step_key, step_payload in _sorted_steps(summary_data):
            try:
                step = int(step_key)
            except Exception:
                skipped_count += 1
                continue

            if not isinstance(step_payload, dict):
                skipped_count += 1
                continue

            step_count += 1
            wall_time = time.time()

            for tag, value in step_payload.items():
                wrote_any = False
                for scalar_tag, scalar_value in _flatten_scalars(str(tag), value):
                    if not math.isfinite(scalar_value):
                        skipped_count += 1
                        continue

                    summary = summary_pb2.Summary(
                        value=[
                            summary_pb2.Summary.Value(
                                tag=scalar_tag,
                                simple_value=scalar_value,
                            )
                        ]
                    )
                    event = event_pb2.Event(
                        wall_time=wall_time,
                        step=step,
                        summary=summary,
                    )
                    writer.add_event(event)
                    scalar_count += 1
                    wrote_any = True

                if not wrote_any:
                    skipped_count += 1
    finally:
        writer.flush()
        writer.close()

    return step_count, scalar_count, skipped_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert fallback summary.json logs into TensorBoard tfevents."
    )
    parser.add_argument(
        "summary_json",
        help="Path to summary.json produced by the fallback SummaryWriter.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the TensorBoard event file will be written. Defaults to summary.json parent directory.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent

    step_count, scalar_count, skipped_count = convert(summary_path, output_dir)
    print(f"summary_json={summary_path}")
    print(f"output_dir={output_dir}")
    print(f"steps_written={step_count}")
    print(f"scalars_written={scalar_count}")
    print(f"items_skipped={skipped_count}")


if __name__ == "__main__":
    main()

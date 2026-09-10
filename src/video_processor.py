"""Run the underwater enhancement pipeline over a video file.

Examples
--------
Play an enhanced video in a window::

    python src/video_processor.py --input orig_foot.mp4

Write the enhanced video to disk without opening a window::

    python src/video_processor.py --input orig_foot.mp4 \\
        --output results/enhanced.mp4 --no-display

Tweak CLAHE strength::

    python src/video_processor.py --input orig_foot.mp4 --clip-limit 3.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from enhancement import enhance_frame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance an underwater video with normalisation + CLAHE.",
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Optional path to write the enhanced video (e.g. results/out.mp4).",
    )
    parser.add_argument(
        "--clip-limit", type=float, default=2.0,
        help="CLAHE clip limit (default: 2.0).",
    )
    parser.add_argument(
        "--tile-grid", type=int, nargs=2, metavar=("W", "H"), default=(10, 50),
        help="CLAHE tile grid size as two ints (default: 10 50).",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Do not open a preview window (useful on headless machines).",
    )
    parser.add_argument(
        "--show-fps", action="store_true",
        help="Overlay live processing FPS on the preview.",
    )
    return parser.parse_args(argv)


def process_video(args: argparse.Namespace) -> int:
    if not args.input.exists():
        print(f"Error: input video not found: {args.input}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"Error: could not open video: {args.input}", file=sys.stderr)
        return 1

    writer = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(args.output), fourcc, source_fps, (width, height)
        )

    tile_grid = tuple(args.tile_grid)
    frame_count = 0
    total_processing_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.perf_counter()
            enhanced = enhance_frame(
                frame,
                clip_limit=args.clip_limit,
                tile_grid_size=tile_grid,
            )
            elapsed = time.perf_counter() - start
            total_processing_time += elapsed
            frame_count += 1

            if args.show_fps and elapsed > 0:
                cv2.putText(
                    enhanced,
                    f"FPS: {1.0 / elapsed:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            if writer is not None:
                writer.write(enhanced)

            if not args.no_display:
                cv2.imshow("Enhanced Video", enhanced)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if frame_count:
        avg_fps = frame_count / total_processing_time if total_processing_time else 0.0
        print(
            f"Processed {frame_count} frames "
            f"(avg processing speed: {avg_fps:.2f} FPS)"
        )
    else:
        print("No frames were read from the input.")

    return 0


def main(argv: list[str] | None = None) -> int:
    return process_video(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

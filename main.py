"""Application entry point — CLI and FastAPI server.

Usage (CLI mode — image file):
    python main.py --image photo.jpg --music song.mp3
    python main.py --image photo.jpg --music song.mp3 --output-dir my_outputs

Usage (CLI mode — webcam capture):
    python main.py --camera --music song.mp3
    python main.py --camera --music song.mp3 --output-dir my_outputs

Usage (server mode):
    python main.py --server
    python main.py --server --port 8080
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.core.config import config


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Face Music Matcher — Compare facial geometry with musical structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the FastAPI server instead of running CLI comparison.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the API server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API server (default: 8000).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Path to the face image (JPEG/PNG).",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Capture face from webcam instead of using --image.",
    )
    parser.add_argument(
        "--music",
        type=Path,
        help="Path to the music file (WAV/MP3).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for visualisation outputs (default: outputs/).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the result as JSON (CLI mode only).",
    )

    return parser


def capture_from_webcam(output_dir: Path) -> Path:
    """Open the webcam, show a live preview, and capture a photo on SPACE.

    Press **SPACE** to capture, **ESC** to quit without capturing.

    Args:
        output_dir: Directory to save the captured image.

    Returns:
        Path to the saved JPEG image.

    Raises:
        SystemExit: If the webcam cannot be opened or user quits.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.", file=sys.stderr)
        sys.exit(1)

    print("📷 Webcam opened!")
    print("   Press SPACE to capture a photo")
    print("   Press ESC   to quit")
    print()

    saved_path: Path | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.", file=sys.stderr)
            break

        # Mirror the frame for a natural selfie view.
        frame = cv2.flip(frame, 1)

        # Draw instructions on the frame.
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(
            frame, "SPACE = Capture  |  ESC = Quit",
            (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        # Draw a face guide ellipse.
        cx, cy = w // 2, h // 2
        cv2.ellipse(frame, (cx, cy), (100, 130), 0, 0, 360, (0, 255, 0), 2)

        cv2.imshow("Face Music Matcher — Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"webcam_{uuid.uuid4().hex[:8]}.jpg"
            saved_path = output_dir / filename
            cv2.imwrite(str(saved_path), frame)
            print(f"✅ Photo captured: {saved_path}")
            break
        elif key == 27:  # ESC
            print("❌ Capture cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if saved_path is None:
        print("No photo was captured.", file=sys.stderr)
        sys.exit(1)

    return saved_path


def run_cli(args: argparse.Namespace) -> None:
    """Execute the comparison via CLI.

    Args:
        args: Parsed command-line arguments.
    """
    from app.infrastructure.audio.librosa_extractor import LibrosaMusicExtractor
    from app.infrastructure.matching.cosine_matcher import CosineEuclideanMatcher
    from app.infrastructure.storage.plot_generator import MatplotlibPlotGenerator
    from app.infrastructure.vision.mediapipe_extractor import MediaPipeFaceExtractor
    from app.use_cases.comparison import CompareFaceAndMusicUseCase

    # -- Determine image source ------------------------------------------
    if args.camera:
        config.ensure_directories()
        image_path = capture_from_webcam(config.UPLOAD_DIR)
    elif args.image:
        image_path = args.image
    else:
        print(
            "Error: You must provide either --image or --camera.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.music:
        print("Error: --music is required.", file=sys.stderr)
        sys.exit(1)

    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    if not args.music.exists():
        print(f"Error: Music file not found: {args.music}", file=sys.stderr)
        sys.exit(1)

    # Override output directory if specified.
    if args.output_dir != Path("outputs"):
        object.__setattr__(config, "OUTPUT_DIR", args.output_dir)

    # Wire dependencies.
    from app.infrastructure.audio.music_signature_builder import MusicSignatureBuilder
    from app.infrastructure.matching.hybrid_matcher import HybridMatcher
    from app.infrastructure.storage.plot_generator import MatplotlibPlotGenerator
    from app.infrastructure.vision.face_signature_builder import FaceSignatureBuilder
    from app.use_cases.comparison import CompareFaceAndMusicUseCase

    use_case = CompareFaceAndMusicUseCase(
        face_builder=FaceSignatureBuilder(),
        music_builder=MusicSignatureBuilder(),
        matcher=HybridMatcher(),
        plot_generator=MatplotlibPlotGenerator(),
    )

    print(f"🔍 Analysing face: {image_path}")
    print(f"🎵 Analysing music: {args.music}")
    print("─" * 50)

    result = use_case.execute(image_path, args.music)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n  Compatibility: {result.compatibility:.2f}%")
        print(f"\n  Component scores:")
        for name, score in result.component_scores.items():
            print(f"    {name:20s}  {score:.2f}%")
        print(f"\n  Plots saved to: {config.OUTPUT_DIR.resolve()}")
        for plot_name, plot_path in result.plot_paths.items():
            print(f"    {plot_name}: {plot_path}")

    print("\n✅ Comparison complete.")


def run_server(args: argparse.Namespace) -> None:
    """Start the FastAPI server.

    Args:
        args: Parsed command-line arguments.
    """
    import uvicorn

    print(f"🚀 Starting Face Music Matcher API on http://{args.host}:{args.port}")
    print(f"   Docs: http://{args.host}:{args.port}/docs")
    print("─" * 50)

    uvicorn.run(
        "app.api.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


def main() -> None:
    """Parse arguments and dispatch to CLI or server mode."""
    parser = build_parser()
    args = parser.parse_args()

    if args.server:
        run_server(args)
    else:
        run_cli(args)


if __name__ == "__main__":
    main()

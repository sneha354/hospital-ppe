"""
Unified CLI Entrypoint for Hospital Medical PPE Compliance Monitoring System.
Usage:
    python main.py run-server        # Start the FastAPI Web Dashboard & Video Stream
    python main.py run-streamlit     # Launch Streamlit Interactive Control Center
    python main.py demo-simulation   # Run simulation in an OpenCV window or verification mode
    python main.py inspect-video     # Process a surveillance video file offline
"""

import sys
import argparse
import os


def run_fastapi_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Starts the FastAPI Web Dashboard and MJPEG stream server."""
    import uvicorn
    print(f"[Hospital PPE AI] Starting FastAPI Web Server at http://{host}:{port}")
    uvicorn.run("web.server:app", host=host, port=port, reload=reload)


def run_streamlit_app(port: int = 8501):
    """Launches the interactive Streamlit Control Center."""
    import subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", str(port)]
    print(f"[Hospital PPE AI] Launching Streamlit App on port {port}...")
    subprocess.run(cmd)


def run_demo_simulation(frames: int = 200, show_window: bool = False):
    """Runs the full pipeline on synthetic hospital simulation to verify all modules."""
    import cv2
    from core.pipeline import PPECompliancePipeline

    print("[Hospital PPE AI] Initializing simulation verification pipeline...")
    pipeline = PPECompliancePipeline(source="simulation", backend="simulation")

    for i in range(1, frames + 1):
        success, frame, results, incidents = pipeline.process_next_frame()
        if not success:
            print(f"Failed to process frame {i}")
            break

        if incidents:
            for inc in incidents:
                print(f"[ALERT] Frame {i}: {inc.severity} in {inc.zone_name} - Missing: {inc.missing_items}")

        if show_window:
            cv2.imshow("Hospital PPE Monitor - Live Simulation", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        if i % 50 == 0:
            stats = pipeline.alert_manager.get_analytics_stats()
            print(f"Processed {i}/{frames} frames. Compliance: {stats['compliance_rate_percent']}% | Incidents: {stats['total_incidents']}")

    if show_window:
        cv2.destroyAllWindows()

    print("[Hospital PPE AI] Simulation verification finished successfully.")


def inspect_video_file(video_path: str, output_path: str = "data/output_annotated.mp4"):
    """Processes a surveillance video file offline and outputs annotated video."""
    import cv2
    from core.pipeline import PPECompliancePipeline

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    print(f"[Hospital PPE AI] Ingesting video: {video_path}")
    pipeline = PPECompliancePipeline(source=video_path, backend="heuristic")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    cap.release()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        success, frame, results, incidents = pipeline.process_next_frame()
        if not success or frame is None:
            break

        out.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames...")

    out.release()
    print(f"[Hospital PPE AI] Finished processing. Annotated video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Hospital Medical PPE & Face Mask Compliance Monitoring System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: run-server
    server_parser = subparsers.add_parser("run-server", help="Start FastAPI Web Dashboard")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    server_parser.add_argument("--port", type=int, default=8000, help="Port number")
    server_parser.add_argument("--reload", action="store_true", help="Enable hot reload")

    # Command: run-streamlit
    st_parser = subparsers.add_parser("run-streamlit", help="Start Streamlit Control Center")
    st_parser.add_argument("--port", type=int, default=8501, help="Streamlit port")

    # Command: demo-simulation
    demo_parser = subparsers.add_parser("demo-simulation", help="Run simulation test")
    demo_parser.add_argument("--frames", type=int, default=150, help="Number of frames to process")
    demo_parser.add_argument("--gui", action="store_true", help="Display OpenCV GUI window")

    # Command: inspect-video
    video_parser = subparsers.add_parser("inspect-video", help="Inspect a surveillance video file")
    video_parser.add_argument("--video", required=True, help="Path to input video file")
    video_parser.add_argument("--output", default="data/output_annotated.mp4", help="Path to save annotated video")

    args = parser.parse_args()

    if args.command == "run-server":
        run_fastapi_server(host=args.host, port=args.port, reload=args.reload)
    elif args.command == "run-streamlit":
        run_streamlit_app(port=args.port)
    elif args.command == "demo-simulation":
        run_demo_simulation(frames=args.frames, show_window=args.gui)
    elif args.command == "inspect-video":
        inspect_video_file(args.video, args.output)
    else:
        # Default behavior: start server
        run_fastapi_server()


if __name__ == "__main__":
    main()


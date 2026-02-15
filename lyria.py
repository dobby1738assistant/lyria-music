#!/usr/bin/env python3
"""
Lyria Music — Clean daemon with instant control via named pipe.

Usage:
    lyria.py play "ambient techno" [--bpm 120] [--density 0.5] [--brightness 0.5]
    lyria.py morph "new prompt" [--density 0.3] [--brightness 0.7]
    lyria.py set --bpm 90 --density 0.4
    lyria.py stop
    lyria.py status
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import queue
import time

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: pip install 'google-genai>=1.52.0'")
    sys.exit(1)

FIFO_PATH = "/tmp/lyria.fifo"
PID_FILE = "/tmp/lyria.pid"
STATE_FILE = "/tmp/lyria.state"
LOG_FILE = "/tmp/lyria.log"

DEFAULT_DEVICE = "plughw:2,0"


# ── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def write_state(state: dict):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass


def read_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def send_command(cmd: dict):
    """Send a command to the running daemon via FIFO."""
    if not os.path.exists(FIFO_PATH):
        print("Error: No Lyria stream running.", file=sys.stderr)
        sys.exit(1)
    try:
        with open(FIFO_PATH, "w") as f:
            f.write(json.dumps(cmd) + "\n")
    except Exception as e:
        print(f"Error sending command: {e}", file=sys.stderr)
        sys.exit(1)


def is_running() -> int | None:
    """Return PID if daemon is running, else None."""
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # Check if alive
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None


def cleanup_files():
    for f in [FIFO_PATH, PID_FILE, STATE_FILE]:
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass


# ── Audio Writer Thread ──────────────────────────────────────────────────────

class AudioWriter:
    def __init__(self, device: str):
        self.device = device
        self.q = queue.Queue(maxsize=50)
        self.proc = None
        self.thread = None
        self.running = False

    def start(self):
        self.proc = subprocess.Popen(
            ["aplay", "-D", self.device, "-f", "S16_LE", "-r", "48000",
             "-c", "2", "-t", "raw", "--buffer-size", "192000"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self.running = True
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()

    def _writer_loop(self):
        while self.running:
            try:
                data = self.q.get(timeout=1)
                if data is None:
                    break
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
            except queue.Empty:
                continue
            except (BrokenPipeError, OSError):
                break

    def feed(self, data: bytes):
        try:
            self.q.put_nowait(data)
        except queue.Full:
            pass  # Drop frame rather than block

    def stop(self):
        self.running = False
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        if self.thread:
            self.thread.join(timeout=3)
        if self.proc:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


# ── FIFO Command Reader Thread ───────────────────────────────────────────────

class CommandReader:
    """Reads commands from FIFO in a thread, puts them on an asyncio queue."""
    def __init__(self, loop: asyncio.AbstractEventLoop, cmd_queue: asyncio.Queue):
        self.loop = loop
        self.cmd_queue = cmd_queue
        self.running = False
        self.thread = None

    def start(self):
        # Create FIFO
        if os.path.exists(FIFO_PATH):
            os.unlink(FIFO_PATH)
        os.mkfifo(FIFO_PATH)

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        while self.running:
            try:
                # open blocks until a writer connects
                with open(FIFO_PATH, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            cmd = json.loads(line)
                            asyncio.run_coroutine_threadsafe(
                                self.cmd_queue.put(cmd), self.loop
                            )
                        except json.JSONDecodeError:
                            pass
            except OSError:
                if self.running:
                    time.sleep(0.1)

    def stop(self):
        self.running = False
        # Unblock the FIFO read by writing to it
        try:
            with open(FIFO_PATH, "w") as f:
                f.write('{"action":"_shutdown"}\n')
        except Exception:
            pass
        if self.thread:
            self.thread.join(timeout=2)


# ── Main Daemon ──────────────────────────────────────────────────────────────

async def run_daemon(args):
    pid = is_running()
    if pid:
        print(f"Error: Lyria already running (PID {pid}). Use 'stop' first.", file=sys.stderr)
        sys.exit(1)

    key = args.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("ERROR: Set GOOGLE_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    # Write PID
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    # Truncate log
    open(LOG_FILE, "w").close()

    client = genai.Client(api_key=key, http_options={"api_version": "v1alpha"})
    audio = AudioWriter(args.device)
    audio.start()

    loop = asyncio.get_event_loop()
    cmd_queue = asyncio.Queue()
    reader = CommandReader(loop, cmd_queue)
    reader.start()

    shutdown = False

    def handle_signal(sig, frame):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    state = {
        "prompt": args.prompt, "bpm": args.bpm,
        "density": args.density, "brightness": args.brightness,
    }

    log(f"♪ Playing: \"{state['prompt']}\" @ {state['bpm']} BPM")
    write_state({"status": "playing", **state})

    try:
        async with client.aio.live.music.connect(model="models/lyria-realtime-exp") as session:
            await session.set_weighted_prompts(
                prompts=[types.WeightedPrompt(text=state["prompt"], weight=1.0)]
            )

            config = {"bpm": state["bpm"], "temperature": args.temperature}
            if state["density"] is not None:
                config["density"] = state["density"]
            if state["brightness"] is not None:
                config["brightness"] = state["brightness"]

            await session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(**config)
            )
            await session.play()

            async for message in session.receive():
                if shutdown:
                    break

                # Process any pending commands (non-blocking)
                while not cmd_queue.empty():
                    try:
                        cmd = cmd_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    action = cmd.get("action", "morph")

                    if action in ("stop", "_shutdown"):
                        shutdown = True
                        break

                    # Update prompt
                    if "prompt" in cmd:
                        state["prompt"] = cmd["prompt"]
                        log(f"♪ → \"{state['prompt']}\"")

                        if "prompts" in cmd:
                            weighted = [types.WeightedPrompt(text=p["text"], weight=p.get("weight", 1.0))
                                        for p in cmd["prompts"]]
                        else:
                            weighted = [types.WeightedPrompt(
                                text=state["prompt"], weight=cmd.get("weight", 1.0)
                            )]
                        await session.set_weighted_prompts(prompts=weighted)

                    # Update config
                    needs_reset = False
                    cfg_update = {}
                    for k in ("density", "brightness"):
                        if k in cmd:
                            state[k] = cmd[k]
                            cfg_update[k] = cmd[k]
                    if "bpm" in cmd:
                        state["bpm"] = cmd["bpm"]
                        needs_reset = True
                        log(f"♪ BPM → {state['bpm']}")

                    if cfg_update or needs_reset:
                        c = {"bpm": state["bpm"], "temperature": args.temperature}
                        if state["density"] is not None:
                            c["density"] = state["density"]
                        if state["brightness"] is not None:
                            c["brightness"] = state["brightness"]
                        await session.set_music_generation_config(
                            config=types.LiveMusicGenerationConfig(**c)
                        )
                        if needs_reset:
                            await session.reset_context()

                    write_state({"status": "playing", **state})

                if shutdown:
                    break

                # Feed audio
                if hasattr(message, "server_content") and message.server_content:
                    chunks = getattr(message.server_content, "audio_chunks", None)
                    if chunks:
                        for chunk in chunks:
                            if chunk.data:
                                audio.feed(chunk.data)

    except Exception as e:
        log(f"Error: {e}")
    finally:
        log("⏹ Stopped.")
        reader.stop()
        audio.stop()
        write_state({"status": "stopped"})
        cleanup_files()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lyria Music Streamer")
    sub = parser.add_subparsers(dest="command", required=True)

    # play
    p = sub.add_parser("play", help="Start streaming")
    p.add_argument("prompt", help="Musical prompt")
    p.add_argument("--bpm", type=int, default=120)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--density", type=float, default=None)
    p.add_argument("--brightness", type=float, default=None)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--api-key", default=None)

    # morph
    m = sub.add_parser("morph", help="Change the music")
    m.add_argument("prompt", help="New musical prompt")
    m.add_argument("--weight", type=float, default=1.0)
    m.add_argument("--density", type=float, default=None)
    m.add_argument("--brightness", type=float, default=None)
    m.add_argument("--bpm", type=int, default=None)

    # set
    s = sub.add_parser("set", help="Adjust parameters without changing prompt")
    s.add_argument("--density", type=float, default=None)
    s.add_argument("--brightness", type=float, default=None)
    s.add_argument("--bpm", type=int, default=None)

    # stop
    sub.add_parser("stop", help="Stop streaming")

    # status
    sub.add_parser("status", help="Show current state")

    args = parser.parse_args()

    if args.command == "play":
        asyncio.run(run_daemon(args))

    elif args.command == "morph":
        cmd = {"prompt": args.prompt, "weight": args.weight}
        if args.density is not None:
            cmd["density"] = args.density
        if args.brightness is not None:
            cmd["brightness"] = args.brightness
        if args.bpm is not None:
            cmd["bpm"] = args.bpm
        send_command(cmd)
        print(f"♪ Morphing to: \"{args.prompt}\"")

    elif args.command == "set":
        cmd = {}
        if args.density is not None:
            cmd["density"] = args.density
        if args.brightness is not None:
            cmd["brightness"] = args.brightness
        if args.bpm is not None:
            cmd["bpm"] = args.bpm
        if not cmd:
            print("Nothing to set. Use --density, --brightness, or --bpm")
            sys.exit(1)
        send_command(cmd)
        print(f"♪ Updated: {cmd}")

    elif args.command == "stop":
        pid = is_running()
        if not pid:
            print("No Lyria stream running.")
            return
        send_command({"action": "stop"})
        # Wait for clean exit
        for _ in range(20):
            time.sleep(0.25)
            if not is_running():
                print("⏹ Stopped.")
                return
        # Force kill
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
        cleanup_files()
        print("⏹ Force stopped.")

    elif args.command == "status":
        pid = is_running()
        if not pid:
            print("No Lyria stream running.")
            return
        state = read_state()
        print(f"♪ Playing (PID {pid})")
        print(f"  Prompt: \"{state.get('prompt', '?')}\"")
        print(f"  BPM: {state.get('bpm', '?')}")
        print(f"  Density: {state.get('density', 'auto')}")
        print(f"  Brightness: {state.get('brightness', 'auto')}")


if __name__ == "__main__":
    main()

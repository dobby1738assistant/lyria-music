#!/usr/bin/env python3
"""Stream live AI-generated music via Google Lyria RealTime to an audio device."""

import argparse
import asyncio
import subprocess
import sys
import os

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai SDK not installed. Run: pip install 'google-genai>=1.52.0'")
    sys.exit(1)


async def stream_music(
    prompt: str,
    duration: float = 10.0,
    bpm: int = 120,
    temperature: float = 1.0,
    density: float | None = None,
    brightness: float | None = None,
    device: str = "plughw:3,0",
    api_key: str | None = None,
):
    """Connect to Lyria RealTime and stream audio to an ALSA device."""
    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY, or pass --api-key")
        sys.exit(1)

    client = genai.Client(api_key=key, http_options={"api_version": "v1alpha"})

    target_bytes = int(duration * 48000 * 2 * 2)  # duration * rate * channels * bytes_per_sample
    audio_buffer = bytearray()

    # Phase 1: Collect audio from WebSocket
    async with client.aio.live.music.connect(model="models/lyria-realtime-exp") as session:
        await session.set_weighted_prompts(
            prompts=[types.WeightedPrompt(text=prompt, weight=1.0)]
        )

        config_kwargs = {"bpm": bpm, "temperature": temperature}
        if density is not None:
            config_kwargs["density"] = density
        if brightness is not None:
            config_kwargs["brightness"] = brightness

        await session.set_music_generation_config(
            config=types.LiveMusicGenerationConfig(**config_kwargs)
        )

        await session.play()
        print(f"Generating {duration}s of music...", flush=True)

        async for message in session.receive():
            if hasattr(message, "server_content") and message.server_content:
                audio_chunks = getattr(message.server_content, "audio_chunks", None)
                if audio_chunks:
                    for chunk in audio_chunks:
                        if chunk.data:
                            audio_buffer.extend(chunk.data)
                    if len(audio_buffer) >= target_bytes:
                        break

    # Trim to exact duration
    audio_buffer = audio_buffer[:target_bytes]
    print(f"Playing {len(audio_buffer) / (48000*2*2):.1f}s of audio...", flush=True)

    # Phase 2: Write WAV and play
    import wave, tempfile
    wav_path = tempfile.mktemp(suffix=".wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(bytes(audio_buffer))

    print(f"Playing via {device}...", flush=True)
    subprocess.run(["aplay", "-D", device, wav_path], stderr=subprocess.DEVNULL)
    os.unlink(wav_path)
    print("Done.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Stream AI music via Google Lyria RealTime")
    parser.add_argument("prompt", help="Musical prompt (e.g. 'ambient techno')")
    parser.add_argument("-d", "--duration", type=float, default=10.0, help="Duration in seconds (default: 10)")
    parser.add_argument("--bpm", type=int, default=120, help="Beats per minute (60-200, default: 120)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Randomness (default: 1.0)")
    parser.add_argument("--density", type=float, default=None, help="Busyness 0.0-1.0")
    parser.add_argument("--brightness", type=float, default=None, help="Brightness 0.0-1.0")
    parser.add_argument("--device", default="plughw:3,0", help="ALSA device (default: plughw:3,0)")
    parser.add_argument("--api-key", default=None, help="Google API key (or set GOOGLE_API_KEY env)")
    args = parser.parse_args()

    asyncio.run(stream_music(
        prompt=args.prompt,
        duration=args.duration,
        bpm=args.bpm,
        temperature=args.temperature,
        density=args.density,
        brightness=args.brightness,
        device=args.device,
        api_key=args.api_key,
    ))


if __name__ == "__main__":
    main()

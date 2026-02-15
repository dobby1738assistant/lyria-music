# lyria-music 🎵

Stream AI-generated music in real-time using Google DeepMind's **Lyria RealTime** model.

Lyria RealTime generates a continuous 48kHz stereo audio stream via WebSocket. You control genre, mood, instruments, BPM, density, and brightness with text prompts — and the music plays live through your speakers.

## Quick Start

```bash
pip install "google-genai>=1.52.0"
export GOOGLE_API_KEY="your-key-here"

# Stream 10 seconds of ambient techno
python3 lyria_stream.py "ambient techno" -d 10

# Chill lo-fi at 85 BPM
python3 lyria_stream.py "lo-fi hip hop chill beats" -d 30 --bpm 85 --density 0.3

# Bright jazz piano
python3 lyria_stream.py "jazz piano trio" --bpm 140 --brightness 0.8 -d 20
```

## Requirements

- Python 3.11+
- `google-genai>=1.52.0`
- `aplay` (ALSA utils — pre-installed on most Linux systems)
- A Google API key with Gemini API access ([get one free](https://aistudio.google.com/apikey))

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `prompt` | (required) | Musical prompt — genre, instruments, mood |
| `-d, --duration` | 10 | Duration in seconds |
| `--bpm` | 120 | Beats per minute (60-200) |
| `--temperature` | 1.0 | Randomness / creativity |
| `--density` | auto | Busyness (0.0 sparse → 1.0 chaotic) |
| `--brightness` | auto | Tone (0.0 muffled → 1.0 crisp) |
| `--device` | plughw:3,0 | ALSA output device |
| `--api-key` | env var | Google API key (or set `GOOGLE_API_KEY`) |

## How It Works

1. Opens a WebSocket to Google's `lyria-realtime-exp` model
2. Sends your text prompt as a weighted musical directive
3. Receives 2-second chunks of raw 16-bit PCM audio at 48kHz stereo
4. Pipes chunks directly to `aplay` — zero latency, no file conversion
5. Stops after the requested duration

## Prompt Tips

- **Combine elements:** `"dark ambient piano with heavy reverb"`
- **Musical terms work:** `"staccato jazz drums"`, `"legato orchestral strings"`
- **Blend genres:** `"lo-fi hip hop meets classical orchestra"`
- **Mood descriptors:** `"melancholic"`, `"euphoric"`, `"meditative"`, `"aggressive"`

## API Details

- **Model:** `models/lyria-realtime-exp` (experimental, currently free)
- **API version:** `v1alpha` (Gemini API)
- **Audio format:** 16-bit signed LE PCM, 48000 Hz, 2 channels
- **Chunk size:** ~2 seconds of audio per message

## License

MIT

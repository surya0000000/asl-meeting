# Virtual Microphone Routing Guide

This guide explains how to route generated speech audio from ASL Meeting Copilot into Zoom or Google Meet.

## macOS (BlackHole)

1. Install BlackHole:
   - https://existential.audio/blackhole/
2. Open **Audio MIDI Setup**.
3. Create a **Multi-Output Device** that includes:
   - Your headphones/speakers
   - BlackHole (2ch)
4. In your audio player or system output, select the Multi-Output Device.
5. In Zoom or Google Meet microphone settings, choose **BlackHole 2ch** as input.
6. Verify level meters move when ASL Copilot plays synthesized speech.

## Windows (VB-CABLE)

1. Install VB-CABLE:
   - https://vb-audio.com/Cable/
2. Set your application output (or system default output) to **CABLE Input**.
3. In Zoom or Google Meet microphone settings, choose **CABLE Output**.
4. Optionally monitor locally using **Listen to this device** in Sound Control Panel.

## Linux (PulseAudio / PipeWire)

Use a virtual sink/source pair:

```bash
pactl load-module module-null-sink sink_name=asl_sink
pactl load-module module-remap-source master=asl_sink.monitor source_name=asl_mic
```

Then set meeting app microphone to `asl_mic`.

## Notes

- Keep TTS volume moderate to prevent clipping.
- For low-latency meetings, prefer local/offline TTS (pyttsx3).
- If speech feedback loops occur, disable “original sound”/echo cancellation overrides in conferencing apps.


# libsimplewebm

A small, focused WebM demuxer & decoder library written in C++.

`libsimplewebm` provides WebM/Matroska demuxing, VP8/VP9 video decoding, Opus/Vorbis audio decoding & support for WebM video with alpha data.

The project uses Google's [`libwebm`](https://chromium.googlesource.com/webm/libwebm) for container parsing, [`libvpx`](https://chromium.googlesource.com/webm/libvpx) for VP8/VP9 video decoding & [`libopus`](https://opus-codec.org/) / [`libvorbis`](https://xiph.org/vorbis/) for audio decoding.

The goal is to keep the library small, dependency-focused & practical for applications that need WebM playback without dragging an entire multimedia framework along for the ride.

## Features

* WebM / Matroska demuxing
* VP8 video decoding
* VP9 video decoding
* VP8/VP9 alpha data support
* WebM `BlockAdditional` / alpha block handling
* Opus audio decoding
* Vorbis audio decoding
* Video track selection
* Audio track selection
* Incremental parsing through a reader abstraction

## Alpha Support

Alpha support is an important part of `libsimplewebm`.

WebM can store an additional alpha stream alongside video frames using Matroska's `BlockGroup` / `BlockAdditions` / `BlockMore` mechanism. `libsimplewebm` preserves this data instead of silently discarding it.

The demuxer exposes the alpha data separately from the primary video frame:

```cpp
WebMFrame frame;

if (demuxer.readFrame(&frame, nullptr))
{
    const unsigned char *color = frame.buffer;
    const long colorSize = frame.bufferSize;

    if (frame.hasAlpha())
    {
        const unsigned char *alpha = frame.alphaBuffer;
        const long alphaSize = frame.alphaBufferSize;

        // Decode/use the alpha bitstream here.
    }
}
```

This is especially useful for applications & engines that need transparent WebM video.

## Dependencies

### Required

* CMake 4.0+
* C++11 or newer compiler
* libvpx
* libopus
* libvorbis
* pkg-config

### Container parser

The project currently uses the required portions of Google's `libwebm`.

> [!NOTE]
> Required `libwebm` sources are included in the repository so users do not need to install a separate system copy of `libwebm`.

Parts of `libwebm` may be modified or replaced over time as `libsimplewebm` evolves toward a smaller, more focused container parser.

## Building

Configure the project with CMake:

```sh
cmake -S . -B build
```

Build it:

```sh
cmake --build build
```

## Example

The example application takes a WebM file and decodes its video & audio streams:

```sh
./build/libsimplewebm_example video.webm
```

## Architecture

The current architecture is intentionally split into container parsing, demuxing & codec decoding.

```text
                 WebM / Matroska file
                           │
                           ▼
                    ┌─────────────┐
                    │   libwebm   │
                    │   parser    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ WebMDemuxer │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
       ┌────────────┐ ┌──────────┐ ┌───────────────┐
       │ VPXDecoder │ │   Alpha  │ │ Opus/Vorbis   │
       │            │ │   Data   │ │   Decoder     │
       └─────┬──────┘ └────┬─────┘ └───────┬───────┘
             │             │               │
             ▼             ▼               ▼
          VP8/VP9       Alpha data      PCM audio
           frames
```

The long-term goal is to keep this architecture lean:

```text
                 WebM / Matroska
                       │
                       ▼
              ┌─────────────────┐
              │ Container Parser│
              └────────┬────────┘
                       │
                       ▼
                ┌──────────────┐
                │   Demuxer    │
                └──────┬───────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
          Video                Audio
             │                   │
        ┌────┴────┐         ┌────┴────┐
        ▼         ▼         ▼         ▼
       VP8       VP9      Opus     Vorbis
        │         │
        └────┬────┘
             ▼
       Alpha / transparency
```

## Why libsimplewebm?

WebM playback does not necessarily require a massive multimedia stack.

`libsimplewebm` is intended for projects that want:

* A small WebM-focused API
* Direct access to compressed video/audio frames
* VP8/VP9 decoding
* Opus/Vorbis decoding
* Alpha-channel support
* Minimal container-level overhead
* A relatively simple integration path

The library is particularly useful for game engines, media tools, applications & other projects where pulling in a complete multimedia framework would be unnecessary.

## Project Status

`libsimplewebm` is actively being developed.

The project is currently undergoing work to reduce the amount of unnecessary `libwebm` code required by the library while retaining the WebM features that `libsimplewebm` actually needs.

This means the container layer may change substantially as the project progresses.

The objective is simple:

> **Keep the functionality. Cut the fat.**

## License

See [`LICENSE`](LICENSE) for licensing information.

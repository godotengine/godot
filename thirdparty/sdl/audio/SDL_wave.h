/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

// RIFF WAVE files are little-endian

/*******************************************/
// Define values for Microsoft WAVE format
/*******************************************/
// FOURCC
#define RIFF 0x46464952 // "RIFF"
#define WAVE 0x45564157 // "WAVE"
#define FACT 0x74636166 // "fact"
#define LIST 0x5453494c // "LIST"
#define BEXT 0x74786562 // "bext"
#define JUNK 0x4B4E554A // "JUNK"
#define FMT  0x20746D66 // "fmt "
#define DATA 0x61746164 // "data"
// Format tags
#define UNKNOWN_CODE    0x0000
#define PCM_CODE        0x0001
#define MS_ADPCM_CODE   0x0002
#define IEEE_FLOAT_CODE 0x0003
#define ALAW_CODE       0x0006
#define MULAW_CODE      0x0007
#define IMA_ADPCM_CODE  0x0011
#define MPEG_CODE       0x0050
#define MPEGLAYER3_CODE 0x0055
#define EXTENSIBLE_CODE 0xFFFE

// Stores the WAVE format information.
typedef struct WaveFormat
{
    Uint16 formattag;     // Raw value of the first field in the fmt chunk data.
    Uint16 encoding;      // Actual encoding, possibly from the extensible header.
    Uint16 channels;      // Number of channels.
    Uint32 frequency;     // Sampling rate in Hz.
    Uint32 byterate;      // Average bytes per second.
    Uint16 blockalign;    // Bytes per block.
    Uint16 bitspersample; // Currently supported are 8, 16, 24, 32, and 4 for ADPCM.

    /* Extra information size. Number of extra bytes starting at byte 18 in the
     * fmt chunk data. This is at least 22 for the extensible header.
     */
    Uint16 extsize;

    // Extensible WAVE header fields
    Uint16 validsamplebits;
    Uint32 samplesperblock; // For compressed formats. Can be zero. Actually 16 bits in the header.
    Uint32 channelmask;
    Uint8 subformat[16]; // A format GUID.
} WaveFormat;

// Stores information on the fact chunk.
typedef struct WaveFact
{
    /* Represents the state of the fact chunk in the WAVE file.
     * Set to -1 if the fact chunk is invalid.
     * Set to 0 if the fact chunk is not present
     * Set to 1 if the fact chunk is present and valid.
     * Set to 2 if samplelength is going to be used as the number of sample frames.
     */
    Sint32 status;

    /* Version 1 of the RIFF specification calls the field in the fact chunk
     * dwFileSize. The Standards Update then calls it dwSampleLength and specifies
     * that it is 'the length of the data in samples'. WAVE files from Windows
     * with this chunk have it set to the samples per channel (sample frames).
     * This is useful to truncate compressed audio to a specific sample count
     * because a compressed block is usually decoded to a fixed number of
     * sample frames.
     */
    Uint32 samplelength; // Raw sample length value from the fact chunk.
} WaveFact;

// Generic struct for the chunks in the WAVE file.
typedef struct WaveChunk
{
    Uint32 fourcc;   // FOURCC of the chunk.
    Uint32 length;   // Size of the chunk data.
    Sint64 position; // Position of the data in the stream.
    Uint8 *data;     // When allocated, this points to the chunk data. length is used for the memory allocation size.
    size_t size;     // Number of bytes in data that could be read from the stream. Can be smaller than length.
} WaveChunk;

// Controls how the size of the RIFF chunk affects the loading of a WAVE file.
typedef enum WaveRiffSizeHint
{
    RiffSizeNoHint,
    RiffSizeForce,
    RiffSizeIgnoreZero,
    RiffSizeIgnore,
    RiffSizeMaximum
} WaveRiffSizeHint;

// Controls how a truncated WAVE file is handled.
typedef enum WaveTruncationHint
{
    TruncNoHint,
    TruncVeryStrict,
    TruncStrict,
    TruncDropFrame,
    TruncDropBlock
} WaveTruncationHint;

// Controls how the fact chunk affects the loading of a WAVE file.
typedef enum WaveFactChunkHint
{
    FactNoHint,
    FactTruncate,
    FactStrict,
    FactIgnoreZero,
    FactIgnore
} WaveFactChunkHint;

typedef struct WaveFile
{
    WaveChunk chunk;
    WaveFormat format;
    WaveFact fact;

    /* Number of sample frames that will be decoded. Calculated either with the
     * size of the data chunk or, if the appropriate hint is enabled, with the
     * sample length value from the fact chunk.
     */
    Sint64 sampleframes;

    void *decoderdata; // Some decoders require extra data for a state.

    WaveRiffSizeHint riffhint;
    WaveTruncationHint trunchint;
    WaveFactChunkHint facthint;
} WaveFile;

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

#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif
#ifndef INT_MAX
SDL_COMPILE_TIME_ASSERT(int_size, sizeof(int) == sizeof(Sint32));
#define INT_MAX SDL_MAX_SINT32
#endif
#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

// Microsoft WAVE file loading routines

#include "SDL_wave.h"
#include "SDL_sysaudio.h"

/* Reads the value stored at the location of the f1 pointer, multiplies it
 * with the second argument and then stores the result to f1.
 * Returns 0 on success, or -1 if the multiplication overflows, in which case f1
 * does not get modified.
 */
static int SafeMult(size_t *f1, size_t f2)
{
    if (*f1 > 0 && SIZE_MAX / *f1 <= f2) {
        return -1;
    }
    *f1 *= f2;
    return 0;
}

typedef struct ADPCM_DecoderState
{
    Uint32 channels;        // Number of channels.
    size_t blocksize;       // Size of an ADPCM block in bytes.
    size_t blockheadersize; // Size of an ADPCM block header in bytes.
    size_t samplesperblock; // Number of samples per channel in an ADPCM block.
    size_t framesize;       // Size of a sample frame (16-bit PCM) in bytes.
    Sint64 framestotal;     // Total number of sample frames.
    Sint64 framesleft;      // Number of sample frames still to be decoded.
    void *ddata;            // Decoder data from initialization.
    void *cstate;           // Decoding state for each channel.

    // ADPCM data.
    struct
    {
        Uint8 *data;
        size_t size;
        size_t pos;
    } input;

    // Current ADPCM block in the ADPCM data above.
    struct
    {
        Uint8 *data;
        size_t size;
        size_t pos;
    } block;

    // Decoded 16-bit PCM data.
    struct
    {
        Sint16 *data;
        size_t size;
        size_t pos;
    } output;
} ADPCM_DecoderState;

typedef struct MS_ADPCM_CoeffData
{
    Uint16 coeffcount;
    Sint16 *coeff;
    Sint16 aligndummy; // Has to be last member.
} MS_ADPCM_CoeffData;

typedef struct MS_ADPCM_ChannelState
{
    Uint16 delta;
    Sint16 coeff1;
    Sint16 coeff2;
} MS_ADPCM_ChannelState;

#ifdef SDL_WAVE_DEBUG_LOG_FORMAT
static void WaveDebugLogFormat(WaveFile *file)
{
    WaveFormat *format = &file->format;
    const char *fmtstr = "WAVE file: %s, %u Hz, %s, %u bits, %u %s/s";
    const char *waveformat, *wavechannel, *wavebpsunit = "B";
    Uint32 wavebps = format->byterate;
    char channelstr[64];

    SDL_zeroa(channelstr);

    switch (format->encoding) {
    case PCM_CODE:
        waveformat = "PCM";
        break;
    case IEEE_FLOAT_CODE:
        waveformat = "IEEE Float";
        break;
    case ALAW_CODE:
        waveformat = "A-law";
        break;
    case MULAW_CODE:
        waveformat = "\xc2\xb5-law";
        break;
    case MS_ADPCM_CODE:
        waveformat = "MS ADPCM";
        break;
    case IMA_ADPCM_CODE:
        waveformat = "IMA ADPCM";
        break;
    default:
        waveformat = "Unknown";
        break;
    }

#define SDL_WAVE_DEBUG_CHANNELCFG(STR, CODE) \
    case CODE:                               \
        wavechannel = STR;                   \
        break;
#define SDL_WAVE_DEBUG_CHANNELSTR(STR, CODE)                                        \
    if (format->channelmask & CODE) {                                               \
        SDL_strlcat(channelstr, channelstr[0] ? "-" STR : STR, sizeof(channelstr)); \
    }

    if (format->formattag == EXTENSIBLE_CODE && format->channelmask > 0) {
        switch (format->channelmask) {
            SDL_WAVE_DEBUG_CHANNELCFG("1.0 Mono", 0x4)
            SDL_WAVE_DEBUG_CHANNELCFG("1.1 Mono", 0xc)
            SDL_WAVE_DEBUG_CHANNELCFG("2.0 Stereo", 0x3)
            SDL_WAVE_DEBUG_CHANNELCFG("2.1 Stereo", 0xb)
            SDL_WAVE_DEBUG_CHANNELCFG("3.0 Stereo", 0x7)
            SDL_WAVE_DEBUG_CHANNELCFG("3.1 Stereo", 0xf)
            SDL_WAVE_DEBUG_CHANNELCFG("3.0 Surround", 0x103)
            SDL_WAVE_DEBUG_CHANNELCFG("3.1 Surround", 0x10b)
            SDL_WAVE_DEBUG_CHANNELCFG("4.0 Quad", 0x33)
            SDL_WAVE_DEBUG_CHANNELCFG("4.1 Quad", 0x3b)
            SDL_WAVE_DEBUG_CHANNELCFG("4.0 Surround", 0x107)
            SDL_WAVE_DEBUG_CHANNELCFG("4.1 Surround", 0x10f)
            SDL_WAVE_DEBUG_CHANNELCFG("5.0", 0x37)
            SDL_WAVE_DEBUG_CHANNELCFG("5.1", 0x3f)
            SDL_WAVE_DEBUG_CHANNELCFG("5.0 Side", 0x607)
            SDL_WAVE_DEBUG_CHANNELCFG("5.1 Side", 0x60f)
            SDL_WAVE_DEBUG_CHANNELCFG("6.0", 0x137)
            SDL_WAVE_DEBUG_CHANNELCFG("6.1", 0x13f)
            SDL_WAVE_DEBUG_CHANNELCFG("6.0 Side", 0x707)
            SDL_WAVE_DEBUG_CHANNELCFG("6.1 Side", 0x70f)
            SDL_WAVE_DEBUG_CHANNELCFG("7.0", 0xf7)
            SDL_WAVE_DEBUG_CHANNELCFG("7.1", 0xff)
            SDL_WAVE_DEBUG_CHANNELCFG("7.0 Side", 0x6c7)
            SDL_WAVE_DEBUG_CHANNELCFG("7.1 Side", 0x6cf)
            SDL_WAVE_DEBUG_CHANNELCFG("7.0 Surround", 0x637)
            SDL_WAVE_DEBUG_CHANNELCFG("7.1 Surround", 0x63f)
            SDL_WAVE_DEBUG_CHANNELCFG("9.0 Surround", 0x5637)
            SDL_WAVE_DEBUG_CHANNELCFG("9.1 Surround", 0x563f)
            SDL_WAVE_DEBUG_CHANNELCFG("11.0 Surround", 0x56f7)
            SDL_WAVE_DEBUG_CHANNELCFG("11.1 Surround", 0x56ff)
        default:
            SDL_WAVE_DEBUG_CHANNELSTR("FL", 0x1)
            SDL_WAVE_DEBUG_CHANNELSTR("FR", 0x2)
            SDL_WAVE_DEBUG_CHANNELSTR("FC", 0x4)
            SDL_WAVE_DEBUG_CHANNELSTR("LF", 0x8)
            SDL_WAVE_DEBUG_CHANNELSTR("BL", 0x10)
            SDL_WAVE_DEBUG_CHANNELSTR("BR", 0x20)
            SDL_WAVE_DEBUG_CHANNELSTR("FLC", 0x40)
            SDL_WAVE_DEBUG_CHANNELSTR("FRC", 0x80)
            SDL_WAVE_DEBUG_CHANNELSTR("BC", 0x100)
            SDL_WAVE_DEBUG_CHANNELSTR("SL", 0x200)
            SDL_WAVE_DEBUG_CHANNELSTR("SR", 0x400)
            SDL_WAVE_DEBUG_CHANNELSTR("TC", 0x800)
            SDL_WAVE_DEBUG_CHANNELSTR("TFL", 0x1000)
            SDL_WAVE_DEBUG_CHANNELSTR("TFC", 0x2000)
            SDL_WAVE_DEBUG_CHANNELSTR("TFR", 0x4000)
            SDL_WAVE_DEBUG_CHANNELSTR("TBL", 0x8000)
            SDL_WAVE_DEBUG_CHANNELSTR("TBC", 0x10000)
            SDL_WAVE_DEBUG_CHANNELSTR("TBR", 0x20000)
            break;
        }
    } else {
        switch (format->channels) {
        default:
            if (SDL_snprintf(channelstr, sizeof(channelstr), "%u channels", format->channels) >= 0) {
                wavechannel = channelstr;
                break;
            }
        case 0:
            wavechannel = "Unknown";
            break;
        case 1:
            wavechannel = "Mono";
            break;
        case 2:
            wavechannel = "Setero";
            break;
        }
    }

#undef SDL_WAVE_DEBUG_CHANNELCFG
#undef SDL_WAVE_DEBUG_CHANNELSTR

    if (wavebps >= 1024) {
        wavebpsunit = "KiB";
        wavebps = wavebps / 1024 + (wavebps & 0x3ff ? 1 : 0);
    }

    SDL_LogDebug(SDL_LOG_CATEGORY_AUDIO, fmtstr, waveformat, format->frequency, wavechannel, format->bitspersample, wavebps, wavebpsunit);
}
#endif

#ifdef SDL_WAVE_DEBUG_DUMP_FORMAT
static void WaveDebugDumpFormat(WaveFile *file, Uint32 rifflen, Uint32 fmtlen, Uint32 datalen)
{
    WaveFormat *format = &file->format;
    const char *fmtstr1 = "WAVE chunk dump:\n"
                          "-------------------------------------------\n"
                          "RIFF                            %11u\n"
                          "-------------------------------------------\n"
                          "    fmt                         %11u\n"
                          "        wFormatTag                   0x%04x\n"
                          "        nChannels               %11u\n"
                          "        nSamplesPerSec          %11u\n"
                          "        nAvgBytesPerSec         %11u\n"
                          "        nBlockAlign             %11u\n";
    const char *fmtstr2 = "        wBitsPerSample          %11u\n";
    const char *fmtstr3 = "        cbSize                  %11u\n";
    const char *fmtstr4a = "        wValidBitsPerSample     %11u\n";
    const char *fmtstr4b = "        wSamplesPerBlock        %11u\n";
    const char *fmtstr5 = "        dwChannelMask            0x%08x\n"
                          "        SubFormat\n"
                          "        %08x-%04x-%04x-%02x%02x%02x%02x%02x%02x%02x%02x\n";
    const char *fmtstr6 = "-------------------------------------------\n"
                          " fact\n"
                          "  dwSampleLength                %11u\n";
    const char *fmtstr7 = "-------------------------------------------\n"
                          " data                           %11u\n"
                          "-------------------------------------------\n";
    char *dumpstr;
    size_t dumppos = 0;
    const size_t bufsize = 1024;
    int res;

    dumpstr = SDL_malloc(bufsize);
    if (!dumpstr) {
        return;
    }
    dumpstr[0] = 0;

    res = SDL_snprintf(dumpstr, bufsize, fmtstr1, rifflen, fmtlen, format->formattag, format->channels, format->frequency, format->byterate, format->blockalign);
    dumppos += res > 0 ? res : 0;
    if (fmtlen >= 16) {
        res = SDL_snprintf(dumpstr + dumppos, bufsize - dumppos, fmtstr2, format->bitspersample);
        dumppos += res > 0 ? res : 0;
    }
    if (fmtlen >= 18) {
        res = SDL_snprintf(dumpstr + dumppos, bufsize - dumppos, fmtstr3, format->extsize);
        dumppos += res > 0 ? res : 0;
    }
    if (format->formattag == EXTENSIBLE_CODE && fmtlen >= 40 && format->extsize >= 22) {
        const Uint8 *g = format->subformat;
        const Uint32 g1 = g[0] | ((Uint32)g[1] << 8) | ((Uint32)g[2] << 16) | ((Uint32)g[3] << 24);
        const Uint32 g2 = g[4] | ((Uint32)g[5] << 8);
        const Uint32 g3 = g[6] | ((Uint32)g[7] << 8);

        switch (format->encoding) {
        default:
            res = SDL_snprintf(dumpstr + dumppos, bufsize - dumppos, fmtstr4a, format->validsamplebits);
            dumppos += res > 0 ? res : 0;
            break;
        case MS_ADPCM_CODE:
        case IMA_ADPCM_CODE:
            res = SDL_snprintf(dumpstr + dumppos, bufsize - dumppos, fmtstr4b, format->samplesperblock);
            dumppos += res > 0 ? res : 0;
            break;
        }
        res = SDL_snprintf(dumpstr + dumppos, bufsize - dumppos, fmtstr5, format->channelmask, g1, g2, g3, g[8], g[9], g[10], g[11], g[12], g[13], g[14], g[15]);
        dumppos += res > 0 ? res : 0;
    } else {
        switch (format->encoding) {
        case MS_ADPCM_CODE:
        case IMA_ADPCM_CODE:
            if (fmtlen >= 20 && format->extsize >= 2) {
                res = SDL_snprintf(dumpstr + dumppos, bufsize - dumppos, fmtstr4b, format->samplesperblock);
                dumppos += res > 0 ? res : 0;
            }
            break;
        }
    }
    if (file->fact.status >= 1) {
        res = SDL_snprintf(dumpstr + dumppos, bufsize - dumppos, fmtstr6, file->fact.samplelength);
        dumppos += res > 0 ? res : 0;
    }
    res = SDL_snprintf(dumpstr + dumppos, bufsize - dumppos, fmtstr7, datalen);
    dumppos += res > 0 ? res : 0;

    SDL_LogDebug(SDL_LOG_CATEGORY_AUDIO, "%s", dumpstr);

    SDL_free(dumpstr);
}
#endif

static Sint64 WaveAdjustToFactValue(WaveFile *file, Sint64 sampleframes)
{
    if (file->fact.status == 2) {
        if (file->facthint == FactStrict && sampleframes < file->fact.samplelength) {
            SDL_SetError("Invalid number of sample frames in WAVE fact chunk (too many)");
            return -1;
        } else if (sampleframes > file->fact.samplelength) {
            return file->fact.samplelength;
        }
    }

    return sampleframes;
}

static bool MS_ADPCM_CalculateSampleFrames(WaveFile *file, size_t datalength)
{
    WaveFormat *format = &file->format;
    const size_t blockheadersize = (size_t)file->format.channels * 7;
    const size_t availableblocks = datalength / file->format.blockalign;
    const size_t blockframebitsize = (size_t)file->format.bitspersample * file->format.channels;
    const size_t trailingdata = datalength % file->format.blockalign;

    if (file->trunchint == TruncVeryStrict || file->trunchint == TruncStrict) {
        // The size of the data chunk must be a multiple of the block size.
        if (datalength < blockheadersize || trailingdata > 0) {
            return SDL_SetError("Truncated MS ADPCM block");
        }
    }

    // Calculate number of sample frames that will be decoded.
    file->sampleframes = (Sint64)availableblocks * format->samplesperblock;
    if (trailingdata > 0) {
        // The last block is truncated. Check if we can get any samples out of it.
        if (file->trunchint == TruncDropFrame) {
            // Drop incomplete sample frame.
            if (trailingdata >= blockheadersize) {
                size_t trailingsamples = 2 + (trailingdata - blockheadersize) * 8 / blockframebitsize;
                if (trailingsamples > format->samplesperblock) {
                    trailingsamples = format->samplesperblock;
                }
                file->sampleframes += trailingsamples;
            }
        }
    }

    file->sampleframes = WaveAdjustToFactValue(file, file->sampleframes);
    if (file->sampleframes < 0) {
        return false;
    }

    return true;
}

static bool MS_ADPCM_Init(WaveFile *file, size_t datalength)
{
    WaveFormat *format = &file->format;
    WaveChunk *chunk = &file->chunk;
    const size_t blockheadersize = (size_t)format->channels * 7;
    const size_t blockdatasize = (size_t)format->blockalign - blockheadersize;
    const size_t blockframebitsize = (size_t)format->bitspersample * format->channels;
    const size_t blockdatasamples = (blockdatasize * 8) / blockframebitsize;
    const Sint16 presetcoeffs[14] = { 256, 0, 512, -256, 0, 0, 192, 64, 240, 0, 460, -208, 392, -232 };
    size_t i, coeffcount;
    MS_ADPCM_CoeffData *coeffdata;

    // Sanity checks.

    /* While it's clear how IMA ADPCM handles more than two channels, the nibble
     * order of MS ADPCM makes it awkward. The Standards Update does not talk
     * about supporting more than stereo anyway.
     */
    if (format->channels > 2) {
        return SDL_SetError("Invalid number of channels");
    }

    if (format->bitspersample != 4) {
        return SDL_SetError("Invalid MS ADPCM bits per sample of %u", (unsigned int)format->bitspersample);
    }

    // The block size must be big enough to contain the block header.
    if (format->blockalign < blockheadersize) {
        return SDL_SetError("Invalid MS ADPCM block size (nBlockAlign)");
    }

    if (format->formattag == EXTENSIBLE_CODE) {
        /* Does have a GUID (like all format tags), but there's no specification
         * for how the data is packed into the extensible header. Making
         * assumptions here could lead to new formats nobody wants to support.
         */
        return SDL_SetError("MS ADPCM with the extensible header is not supported");
    }

    /* There are wSamplesPerBlock, wNumCoef, and at least 7 coefficient pairs in
     * the extended part of the header.
     */
    if (chunk->size < 22) {
        return SDL_SetError("Could not read MS ADPCM format header");
    }

    format->samplesperblock = chunk->data[18] | ((Uint16)chunk->data[19] << 8);
    // Number of coefficient pairs. A pair has two 16-bit integers.
    coeffcount = chunk->data[20] | ((size_t)chunk->data[21] << 8);
    /* bPredictor, the integer offset into the coefficients array, is only
     * 8 bits. It can only address the first 256 coefficients. Let's limit
     * the count number here.
     */
    if (coeffcount > 256) {
        coeffcount = 256;
    }

    if (chunk->size < 22 + coeffcount * 4) {
        return SDL_SetError("Could not read custom coefficients in MS ADPCM format header");
    } else if (format->extsize < 4 + coeffcount * 4) {
        return SDL_SetError("Invalid MS ADPCM format header (too small)");
    } else if (coeffcount < 7) {
        return SDL_SetError("Missing required coefficients in MS ADPCM format header");
    }

    coeffdata = (MS_ADPCM_CoeffData *)SDL_malloc(sizeof(MS_ADPCM_CoeffData) + coeffcount * 4);
    file->decoderdata = coeffdata; // Freed in cleanup.
    if (!coeffdata) {
        return false;
    }
    coeffdata->coeff = &coeffdata->aligndummy;
    coeffdata->coeffcount = (Uint16)coeffcount;

    // Copy the 16-bit pairs.
    for (i = 0; i < coeffcount * 2; i++) {
        Sint32 c = chunk->data[22 + i * 2] | ((Sint32)chunk->data[23 + i * 2] << 8);
        if (c >= 0x8000) {
            c -= 0x10000;
        }
        if (i < 14 && c != presetcoeffs[i]) {
            return SDL_SetError("Wrong preset coefficients in MS ADPCM format header");
        }
        coeffdata->coeff[i] = (Sint16)c;
    }

    /* Technically, wSamplesPerBlock is required, but we have all the
     * information in the other fields to calculate it, if it's zero.
     */
    if (format->samplesperblock == 0) {
        /* Let's be nice to the encoders that didn't know how to fill this.
         * The Standards Update calculates it this way:
         *
         *   x = Block size (in bits) minus header size (in bits)
         *   y = Bit depth multiplied by channel count
         *   z = Number of samples per channel in block header
         *   wSamplesPerBlock = x / y + z
         */
        format->samplesperblock = (Uint32)blockdatasamples + 2;
    }

    /* nBlockAlign can be in conflict with wSamplesPerBlock. For example, if
     * the number of samples doesn't fit into the block. The Standards Update
     * also describes wSamplesPerBlock with a formula that makes it necessary to
     * always fill the block with the maximum amount of samples, but this is not
     * enforced here as there are no compatibility issues.
     * A truncated block header with just one sample is not supported.
     */
    if (format->samplesperblock == 1 || blockdatasamples < format->samplesperblock - 2) {
        return SDL_SetError("Invalid number of samples per MS ADPCM block (wSamplesPerBlock)");
    }

    if (!MS_ADPCM_CalculateSampleFrames(file, datalength)) {
        return false;
    }

    return true;
}

static Sint16 MS_ADPCM_ProcessNibble(MS_ADPCM_ChannelState *cstate, Sint32 sample1, Sint32 sample2, Uint8 nybble)
{
    const Sint32 max_audioval = 32767;
    const Sint32 min_audioval = -32768;
    const Uint16 max_deltaval = 65535;
    const Uint16 adaptive[] = {
        230, 230, 230, 230, 307, 409, 512, 614,
        768, 614, 512, 409, 307, 230, 230, 230
    };
    Sint32 new_sample;
    Sint32 errordelta;
    Uint32 delta = cstate->delta;

    new_sample = (sample1 * cstate->coeff1 + sample2 * cstate->coeff2) / 256;
    // The nibble is a signed 4-bit error delta.
    errordelta = (Sint32)nybble - (nybble >= 0x08 ? 0x10 : 0);
    new_sample += (Sint32)delta * errordelta;
    if (new_sample < min_audioval) {
        new_sample = min_audioval;
    } else if (new_sample > max_audioval) {
        new_sample = max_audioval;
    }
    delta = (delta * adaptive[nybble]) / 256;
    if (delta < 16) {
        delta = 16;
    } else if (delta > max_deltaval) {
        /* This issue is not described in the Standards Update and therefore
         * undefined. It seems sensible to prevent overflows with a limit.
         */
        delta = max_deltaval;
    }

    cstate->delta = (Uint16)delta;
    return (Sint16)new_sample;
}

static bool MS_ADPCM_DecodeBlockHeader(ADPCM_DecoderState *state)
{
    Uint8 coeffindex;
    const Uint32 channels = state->channels;
    Sint32 sample;
    Uint32 c;
    MS_ADPCM_ChannelState *cstate = (MS_ADPCM_ChannelState *)state->cstate;
    MS_ADPCM_CoeffData *ddata = (MS_ADPCM_CoeffData *)state->ddata;

    for (c = 0; c < channels; c++) {
        size_t o = c;

        // Load the coefficient pair into the channel state.
        coeffindex = state->block.data[o];
        if (coeffindex > ddata->coeffcount) {
            return SDL_SetError("Invalid MS ADPCM coefficient index in block header");
        }
        cstate[c].coeff1 = ddata->coeff[coeffindex * 2];
        cstate[c].coeff2 = ddata->coeff[coeffindex * 2 + 1];

        // Initial delta value.
        o = (size_t)channels + c * 2;
        cstate[c].delta = state->block.data[o] | ((Uint16)state->block.data[o + 1] << 8);

        /* Load the samples from the header. Interestingly, the sample later in
         * the output stream comes first.
         */
        o = (size_t)channels * 3 + c * 2;
        sample = state->block.data[o] | ((Sint32)state->block.data[o + 1] << 8);
        if (sample >= 0x8000) {
            sample -= 0x10000;
        }
        state->output.data[state->output.pos + channels] = (Sint16)sample;

        o = (size_t)channels * 5 + c * 2;
        sample = state->block.data[o] | ((Sint32)state->block.data[o + 1] << 8);
        if (sample >= 0x8000) {
            sample -= 0x10000;
        }
        state->output.data[state->output.pos] = (Sint16)sample;

        state->output.pos++;
    }

    state->block.pos += state->blockheadersize;

    // Skip second sample frame that came from the header.
    state->output.pos += state->channels;

    // Header provided two sample frames.
    state->framesleft -= 2;

    return true;
}

/* Decodes the data of the MS ADPCM block. Decoding will stop if a block is too
 * short, returning with none or partially decoded data. The partial data
 * will always contain full sample frames (same sample count for each channel).
 * Incomplete sample frames are discarded.
 */
static bool MS_ADPCM_DecodeBlockData(ADPCM_DecoderState *state)
{
    Uint16 nybble = 0;
    Sint16 sample1, sample2;
    const Uint32 channels = state->channels;
    Uint32 c;
    MS_ADPCM_ChannelState *cstate = (MS_ADPCM_ChannelState *)state->cstate;

    size_t blockpos = state->block.pos;
    size_t blocksize = state->block.size;

    size_t outpos = state->output.pos;

    Sint64 blockframesleft = state->samplesperblock - 2;
    if (blockframesleft > state->framesleft) {
        blockframesleft = state->framesleft;
    }

    while (blockframesleft > 0) {
        for (c = 0; c < channels; c++) {
            if (nybble & 0x4000) {
                nybble <<= 4;
            } else if (blockpos < blocksize) {
                nybble = state->block.data[blockpos++] | 0x4000;
            } else {
                // Out of input data. Drop the incomplete frame and return.
                state->output.pos = outpos - c;
                return false;
            }

            // Load previous samples which may come from the block header.
            sample1 = state->output.data[outpos - channels];
            sample2 = state->output.data[outpos - channels * 2];

            sample1 = MS_ADPCM_ProcessNibble(cstate + c, sample1, sample2, (nybble >> 4) & 0x0f);
            state->output.data[outpos++] = sample1;
        }

        state->framesleft--;
        blockframesleft--;
    }

    state->output.pos = outpos;

    return true;
}

static bool MS_ADPCM_Decode(WaveFile *file, Uint8 **audio_buf, Uint32 *audio_len)
{
    bool result;
    size_t bytesleft, outputsize;
    WaveChunk *chunk = &file->chunk;
    ADPCM_DecoderState state;
    MS_ADPCM_ChannelState cstate[2];

    SDL_zero(state);
    SDL_zeroa(cstate);

    if (chunk->size != chunk->length) {
        // Could not read everything. Recalculate number of sample frames.
        if (!MS_ADPCM_CalculateSampleFrames(file, chunk->size)) {
            return false;
        }
    }

    // Nothing to decode, nothing to return.
    if (file->sampleframes == 0) {
        *audio_buf = NULL;
        *audio_len = 0;
        return true;
    }

    state.blocksize = file->format.blockalign;
    state.channels = file->format.channels;
    state.blockheadersize = (size_t)state.channels * 7;
    state.samplesperblock = file->format.samplesperblock;
    state.framesize = state.channels * sizeof(Sint16);
    state.ddata = file->decoderdata;
    state.framestotal = file->sampleframes;
    state.framesleft = state.framestotal;

    state.input.data = chunk->data;
    state.input.size = chunk->size;
    state.input.pos = 0;

    // The output size in bytes. May get modified if data is truncated.
    outputsize = (size_t)state.framestotal;
    if (SafeMult(&outputsize, state.framesize)) {
        return SDL_SetError("WAVE file too big");
    } else if (outputsize > SDL_MAX_UINT32 || state.framestotal > SIZE_MAX) {
        return SDL_SetError("WAVE file too big");
    }

    state.output.pos = 0;
    state.output.size = outputsize / sizeof(Sint16);
    state.output.data = (Sint16 *)SDL_calloc(1, outputsize);
    if (!state.output.data) {
        return false;
    }

    state.cstate = cstate;

    // Decode block by block. A truncated block will stop the decoding.
    bytesleft = state.input.size - state.input.pos;
    while (state.framesleft > 0 && bytesleft >= state.blockheadersize) {
        state.block.data = state.input.data + state.input.pos;
        state.block.size = bytesleft < state.blocksize ? bytesleft : state.blocksize;
        state.block.pos = 0;

        if (state.output.size - state.output.pos < (Uint64)state.framesleft * state.channels) {
            // Somehow didn't allocate enough space for the output.
            SDL_free(state.output.data);
            return SDL_SetError("Unexpected overflow in MS ADPCM decoder");
        }

        // Initialize decoder with the values from the block header.
        result = MS_ADPCM_DecodeBlockHeader(&state);
        if (!result) {
            SDL_free(state.output.data);
            return false;
        }

        // Decode the block data. It stores the samples directly in the output.
        result = MS_ADPCM_DecodeBlockData(&state);
        if (!result) {
            // Unexpected end. Stop decoding and return partial data if necessary.
            if (file->trunchint == TruncVeryStrict || file->trunchint == TruncStrict) {
                SDL_free(state.output.data);
                return SDL_SetError("Truncated data chunk");
            } else if (file->trunchint != TruncDropFrame) {
                state.output.pos -= state.output.pos % (state.samplesperblock * state.channels);
            }
            outputsize = state.output.pos * sizeof(Sint16); // Can't overflow, is always smaller.
            break;
        }

        state.input.pos += state.block.size;
        bytesleft = state.input.size - state.input.pos;
    }

    *audio_buf = (Uint8 *)state.output.data;
    *audio_len = (Uint32)outputsize;

    return true;
}

static bool IMA_ADPCM_CalculateSampleFrames(WaveFile *file, size_t datalength)
{
    WaveFormat *format = &file->format;
    const size_t blockheadersize = (size_t)format->channels * 4;
    const size_t subblockframesize = (size_t)format->channels * 4;
    const size_t availableblocks = datalength / format->blockalign;
    const size_t trailingdata = datalength % format->blockalign;

    if (file->trunchint == TruncVeryStrict || file->trunchint == TruncStrict) {
        // The size of the data chunk must be a multiple of the block size.
        if (datalength < blockheadersize || trailingdata > 0) {
            return SDL_SetError("Truncated IMA ADPCM block");
        }
    }

    // Calculate number of sample frames that will be decoded.
    file->sampleframes = (Uint64)availableblocks * format->samplesperblock;
    if (trailingdata > 0) {
        // The last block is truncated. Check if we can get any samples out of it.
        if (file->trunchint == TruncDropFrame && trailingdata > blockheadersize - 2) {
            /* The sample frame in the header of the truncated block is present.
             * Drop incomplete sample frames.
             */
            size_t trailingsamples = 1;

            if (trailingdata > blockheadersize) {
                // More data following after the header.
                const size_t trailingblockdata = trailingdata - blockheadersize;
                const size_t trailingsubblockdata = trailingblockdata % subblockframesize;
                trailingsamples += (trailingblockdata / subblockframesize) * 8;
                /* Due to the interleaved sub-blocks, the last 4 bytes determine
                 * how many samples of the truncated sub-block are lost.
                 */
                if (trailingsubblockdata > subblockframesize - 4) {
                    trailingsamples += (trailingsubblockdata % 4) * 2;
                }
            }

            if (trailingsamples > format->samplesperblock) {
                trailingsamples = format->samplesperblock;
            }
            file->sampleframes += trailingsamples;
        }
    }

    file->sampleframes = WaveAdjustToFactValue(file, file->sampleframes);
    if (file->sampleframes < 0) {
        return false;
    }

    return true;
}

static bool IMA_ADPCM_Init(WaveFile *file, size_t datalength)
{
    WaveFormat *format = &file->format;
    WaveChunk *chunk = &file->chunk;
    const size_t blockheadersize = (size_t)format->channels * 4;
    const size_t blockdatasize = (size_t)format->blockalign - blockheadersize;
    const size_t blockframebitsize = (size_t)format->bitspersample * format->channels;
    const size_t blockdatasamples = (blockdatasize * 8) / blockframebitsize;

    // Sanity checks.

    // IMA ADPCM can also have 3-bit samples, but it's not supported by SDL at this time.
    if (format->bitspersample == 3) {
        return SDL_SetError("3-bit IMA ADPCM currently not supported");
    } else if (format->bitspersample != 4) {
        return SDL_SetError("Invalid IMA ADPCM bits per sample of %u", (unsigned int)format->bitspersample);
    }

    /* The block size is required to be a multiple of 4 and it must be able to
     * hold a block header.
     */
    if (format->blockalign < blockheadersize || format->blockalign % 4) {
        return SDL_SetError("Invalid IMA ADPCM block size (nBlockAlign)");
    }

    if (format->formattag == EXTENSIBLE_CODE) {
        /* There's no specification for this, but it's basically the same
         * format because the extensible header has wSampePerBlocks too.
         */
    } else {
        // The Standards Update says there 'should' be 2 bytes for wSamplesPerBlock.
        if (chunk->size >= 20 && format->extsize >= 2) {
            format->samplesperblock = chunk->data[18] | ((Uint16)chunk->data[19] << 8);
        }
    }

    if (format->samplesperblock == 0) {
        /* Field zero? No problem. We just assume the encoder packed the block.
         * The specification calculates it this way:
         *
         *   x = Block size (in bits) minus header size (in bits)
         *   y = Bit depth multiplied by channel count
         *   z = Number of samples per channel in header
         *   wSamplesPerBlock = x / y + z
         */
        format->samplesperblock = (Uint32)blockdatasamples + 1;
    }

    /* nBlockAlign can be in conflict with wSamplesPerBlock. For example, if
     * the number of samples doesn't fit into the block. The Standards Update
     * also describes wSamplesPerBlock with a formula that makes it necessary
     * to always fill the block with the maximum amount of samples, but this is
     * not enforced here as there are no compatibility issues.
     */
    if (blockdatasamples < format->samplesperblock - 1) {
        return SDL_SetError("Invalid number of samples per IMA ADPCM block (wSamplesPerBlock)");
    }

    if (!IMA_ADPCM_CalculateSampleFrames(file, datalength)) {
        return false;
    }

    return true;
}

static Sint16 IMA_ADPCM_ProcessNibble(Sint8 *cindex, Sint16 lastsample, Uint8 nybble)
{
    const Sint32 max_audioval = 32767;
    const Sint32 min_audioval = -32768;
    const Sint8 index_table_4b[16] = {
        -1, -1, -1, -1,
        2, 4, 6, 8,
        -1, -1, -1, -1,
        2, 4, 6, 8
    };
    const Uint16 step_table[89] = {
        7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31,
        34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130,
        143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408,
        449, 494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282,
        1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327,
        3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630,
        9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350,
        22385, 24623, 27086, 29794, 32767
    };
    Uint32 step;
    Sint32 sample, delta;
    Sint8 index = *cindex;

    // Clamp index into valid range.
    if (index > 88) {
        index = 88;
    } else if (index < 0) {
        index = 0;
    }

    // explicit cast to avoid gcc warning about using 'char' as array index
    step = step_table[(size_t)index];

    // Update index value
    *cindex = index + index_table_4b[nybble];

    /* This calculation uses shifts and additions because multiplications were
     * much slower back then. Sadly, this can't just be replaced with an actual
     * multiplication now as the old algorithm drops some bits. The closest
     * approximation I could find is something like this:
     * (nybble & 0x8 ? -1 : 1) * ((nybble & 0x7) * step / 4 + step / 8)
     */
    delta = step >> 3;
    if (nybble & 0x04) {
        delta += step;
    }
    if (nybble & 0x02) {
        delta += step >> 1;
    }
    if (nybble & 0x01) {
        delta += step >> 2;
    }
    if (nybble & 0x08) {
        delta = -delta;
    }

    sample = lastsample + delta;

    // Clamp output sample
    if (sample > max_audioval) {
        sample = max_audioval;
    } else if (sample < min_audioval) {
        sample = min_audioval;
    }

    return (Sint16)sample;
}

static bool IMA_ADPCM_DecodeBlockHeader(ADPCM_DecoderState *state)
{
    Sint16 step;
    Uint32 c;
    Uint8 *cstate = (Uint8 *)state->cstate;

    for (c = 0; c < state->channels; c++) {
        size_t o = state->block.pos + c * 4;

        // Extract the sample from the header.
        Sint32 sample = state->block.data[o] | ((Sint32)state->block.data[o + 1] << 8);
        if (sample >= 0x8000) {
            sample -= 0x10000;
        }
        state->output.data[state->output.pos++] = (Sint16)sample;

        // Channel step index.
        step = (Sint16)state->block.data[o + 2];
        cstate[c] = (Sint8)(step > 0x80 ? step - 0x100 : step);

        // Reserved byte in block header, should be 0.
        if (state->block.data[o + 3] != 0) {
            /* Uh oh, corrupt data?  Buggy code? */;
        }
    }

    state->block.pos += state->blockheadersize;

    // Header provided one sample frame.
    state->framesleft--;

    return true;
}

/* Decodes the data of the IMA ADPCM block. Decoding will stop if a block is too
 * short, returning with none or partially decoded data. The partial data always
 * contains full sample frames (same sample count for each channel).
 * Incomplete sample frames are discarded.
 */
static bool IMA_ADPCM_DecodeBlockData(ADPCM_DecoderState *state)
{
    size_t i;
    const Uint32 channels = state->channels;
    const size_t subblockframesize = (size_t)channels * 4;
    Uint64 bytesrequired;
    Uint32 c;
    bool result = true;

    size_t blockpos = state->block.pos;
    size_t blocksize = state->block.size;
    size_t blockleft = blocksize - blockpos;

    size_t outpos = state->output.pos;

    Sint64 blockframesleft = state->samplesperblock - 1;
    if (blockframesleft > state->framesleft) {
        blockframesleft = state->framesleft;
    }

    bytesrequired = (blockframesleft + 7) / 8 * subblockframesize;
    if (blockleft < bytesrequired) {
        // Data truncated. Calculate how many samples we can get out if it.
        const size_t guaranteedframes = blockleft / subblockframesize;
        const size_t remainingbytes = blockleft % subblockframesize;
        blockframesleft = guaranteedframes;
        if (remainingbytes > subblockframesize - 4) {
            blockframesleft += (Sint64)(remainingbytes % 4) * 2;
        }
        // Signal the truncation.
        result = false;
    }

    /* Each channel has their nibbles packed into 32-bit blocks. These blocks
     * are interleaved and make up the data part of the ADPCM block. This loop
     * decodes the samples as they come from the input data and puts them at
     * the appropriate places in the output data.
     */
    while (blockframesleft > 0) {
        const size_t subblocksamples = blockframesleft < 8 ? (size_t)blockframesleft : 8;

        for (c = 0; c < channels; c++) {
            Uint8 nybble = 0;
            // Load previous sample which may come from the block header.
            Sint16 sample = state->output.data[outpos + c - channels];

            for (i = 0; i < subblocksamples; i++) {
                if (i & 1) {
                    nybble >>= 4;
                } else {
                    nybble = state->block.data[blockpos++];
                }

                sample = IMA_ADPCM_ProcessNibble((Sint8 *)state->cstate + c, sample, nybble & 0x0f);
                state->output.data[outpos + c + i * channels] = sample;
            }
        }

        outpos += channels * subblocksamples;
        state->framesleft -= subblocksamples;
        blockframesleft -= subblocksamples;
    }

    state->block.pos = blockpos;
    state->output.pos = outpos;

    return result;
}

static bool IMA_ADPCM_Decode(WaveFile *file, Uint8 **audio_buf, Uint32 *audio_len)
{
    bool result;
    size_t bytesleft, outputsize;
    WaveChunk *chunk = &file->chunk;
    ADPCM_DecoderState state;
    Sint8 *cstate;

    if (chunk->size != chunk->length) {
        // Could not read everything. Recalculate number of sample frames.
        if (!IMA_ADPCM_CalculateSampleFrames(file, chunk->size)) {
            return false;
        }
    }

    // Nothing to decode, nothing to return.
    if (file->sampleframes == 0) {
        *audio_buf = NULL;
        *audio_len = 0;
        return true;
    }

    SDL_zero(state);
    state.channels = file->format.channels;
    state.blocksize = file->format.blockalign;
    state.blockheadersize = (size_t)state.channels * 4;
    state.samplesperblock = file->format.samplesperblock;
    state.framesize = state.channels * sizeof(Sint16);
    state.framestotal = file->sampleframes;
    state.framesleft = state.framestotal;

    state.input.data = chunk->data;
    state.input.size = chunk->size;
    state.input.pos = 0;

    // The output size in bytes. May get modified if data is truncated.
    outputsize = (size_t)state.framestotal;
    if (SafeMult(&outputsize, state.framesize)) {
        return SDL_SetError("WAVE file too big");
    } else if (outputsize > SDL_MAX_UINT32 || state.framestotal > SIZE_MAX) {
        return SDL_SetError("WAVE file too big");
    }

    state.output.pos = 0;
    state.output.size = outputsize / sizeof(Sint16);
    state.output.data = (Sint16 *)SDL_malloc(outputsize);
    if (!state.output.data) {
        return false;
    }

    cstate = (Sint8 *)SDL_calloc(state.channels, sizeof(Sint8));
    if (!cstate) {
        SDL_free(state.output.data);
        return false;
    }
    state.cstate = cstate;

    // Decode block by block. A truncated block will stop the decoding.
    bytesleft = state.input.size - state.input.pos;
    while (state.framesleft > 0 && bytesleft >= state.blockheadersize) {
        state.block.data = state.input.data + state.input.pos;
        state.block.size = bytesleft < state.blocksize ? bytesleft : state.blocksize;
        state.block.pos = 0;

        if (state.output.size - state.output.pos < (Uint64)state.framesleft * state.channels) {
            // Somehow didn't allocate enough space for the output.
            SDL_free(state.output.data);
            SDL_free(cstate);
            return SDL_SetError("Unexpected overflow in IMA ADPCM decoder");
        }

        // Initialize decoder with the values from the block header.
        result = IMA_ADPCM_DecodeBlockHeader(&state);
        if (result) {
            // Decode the block data. It stores the samples directly in the output.
            result = IMA_ADPCM_DecodeBlockData(&state);
        }

        if (!result) {
            // Unexpected end. Stop decoding and return partial data if necessary.
            if (file->trunchint == TruncVeryStrict || file->trunchint == TruncStrict) {
                SDL_free(state.output.data);
                SDL_free(cstate);
                return SDL_SetError("Truncated data chunk");
            } else if (file->trunchint != TruncDropFrame) {
                state.output.pos -= state.output.pos % (state.samplesperblock * state.channels);
            }
            outputsize = state.output.pos * sizeof(Sint16); // Can't overflow, is always smaller.
            break;
        }

        state.input.pos += state.block.size;
        bytesleft = state.input.size - state.input.pos;
    }

    *audio_buf = (Uint8 *)state.output.data;
    *audio_len = (Uint32)outputsize;

    SDL_free(cstate);

    return true;
}

static bool LAW_Init(WaveFile *file, size_t datalength)
{
    WaveFormat *format = &file->format;

    // Standards Update requires this to be 8.
    if (format->bitspersample != 8) {
        return SDL_SetError("Invalid companded bits per sample of %u", (unsigned int)format->bitspersample);
    }

    // Not going to bother with weird padding.
    if (format->blockalign != format->channels) {
        return SDL_SetError("Unsupported block alignment");
    }

    if ((file->trunchint == TruncVeryStrict || file->trunchint == TruncStrict)) {
        if (format->blockalign > 1 && datalength % format->blockalign) {
            return SDL_SetError("Truncated data chunk in WAVE file");
        }
    }

    file->sampleframes = WaveAdjustToFactValue(file, datalength / format->blockalign);
    if (file->sampleframes < 0) {
        return false;
    }

    return true;
}

static bool LAW_Decode(WaveFile *file, Uint8 **audio_buf, Uint32 *audio_len)
{
#ifdef SDL_WAVE_LAW_LUT
    const Sint16 alaw_lut[256] = {
        -5504, -5248, -6016, -5760, -4480, -4224, -4992, -4736, -7552, -7296, -8064, -7808, -6528, -6272, -7040, -6784, -2752,
        -2624, -3008, -2880, -2240, -2112, -2496, -2368, -3776, -3648, -4032, -3904, -3264, -3136, -3520, -3392, -22016,
        -20992, -24064, -23040, -17920, -16896, -19968, -18944, -30208, -29184, -32256, -31232, -26112, -25088, -28160, -27136, -11008,
        -10496, -12032, -11520, -8960, -8448, -9984, -9472, -15104, -14592, -16128, -15616, -13056, -12544, -14080, -13568, -344,
        -328, -376, -360, -280, -264, -312, -296, -472, -456, -504, -488, -408, -392, -440, -424, -88,
        -72, -120, -104, -24, -8, -56, -40, -216, -200, -248, -232, -152, -136, -184, -168, -1376,
        -1312, -1504, -1440, -1120, -1056, -1248, -1184, -1888, -1824, -2016, -1952, -1632, -1568, -1760, -1696, -688,
        -656, -752, -720, -560, -528, -624, -592, -944, -912, -1008, -976, -816, -784, -880, -848, 5504,
        5248, 6016, 5760, 4480, 4224, 4992, 4736, 7552, 7296, 8064, 7808, 6528, 6272, 7040, 6784, 2752,
        2624, 3008, 2880, 2240, 2112, 2496, 2368, 3776, 3648, 4032, 3904, 3264, 3136, 3520, 3392, 22016,
        20992, 24064, 23040, 17920, 16896, 19968, 18944, 30208, 29184, 32256, 31232, 26112, 25088, 28160, 27136, 11008,
        10496, 12032, 11520, 8960, 8448, 9984, 9472, 15104, 14592, 16128, 15616, 13056, 12544, 14080, 13568, 344,
        328, 376, 360, 280, 264, 312, 296, 472, 456, 504, 488, 408, 392, 440, 424, 88,
        72, 120, 104, 24, 8, 56, 40, 216, 200, 248, 232, 152, 136, 184, 168, 1376,
        1312, 1504, 1440, 1120, 1056, 1248, 1184, 1888, 1824, 2016, 1952, 1632, 1568, 1760, 1696, 688,
        656, 752, 720, 560, 528, 624, 592, 944, 912, 1008, 976, 816, 784, 880, 848
    };
    const Sint16 mulaw_lut[256] = {
        -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956, -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764, -15996,
        -15484, -14972, -14460, -13948, -13436, -12924, -12412, -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316, -7932,
        -7676, -7420, -7164, -6908, -6652, -6396, -6140, -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092, -3900,
        -3772, -3644, -3516, -3388, -3260, -3132, -3004, -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980, -1884,
        -1820, -1756, -1692, -1628, -1564, -1500, -1436, -1372, -1308, -1244, -1180, -1116, -1052, -988, -924, -876,
        -844, -812, -780, -748, -716, -684, -652, -620, -588, -556, -524, -492, -460, -428, -396, -372,
        -356, -340, -324, -308, -292, -276, -260, -244, -228, -212, -196, -180, -164, -148, -132, -120,
        -112, -104, -96, -88, -80, -72, -64, -56, -48, -40, -32, -24, -16, -8, 0, 32124,
        31100, 30076, 29052, 28028, 27004, 25980, 24956, 23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764, 15996,
        15484, 14972, 14460, 13948, 13436, 12924, 12412, 11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316, 7932,
        7676, 7420, 7164, 6908, 6652, 6396, 6140, 5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092, 3900,
        3772, 3644, 3516, 3388, 3260, 3132, 3004, 2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980, 1884,
        1820, 1756, 1692, 1628, 1564, 1500, 1436, 1372, 1308, 1244, 1180, 1116, 1052, 988, 924, 876,
        844, 812, 780, 748, 716, 684, 652, 620, 588, 556, 524, 492, 460, 428, 396, 372,
        356, 340, 324, 308, 292, 276, 260, 244, 228, 212, 196, 180, 164, 148, 132, 120,
        112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0
    };
#endif

    WaveFormat *format = &file->format;
    WaveChunk *chunk = &file->chunk;
    size_t i, sample_count, expanded_len;
    Uint8 *src;
    Sint16 *dst;

    if (chunk->length != chunk->size) {
        file->sampleframes = WaveAdjustToFactValue(file, chunk->size / format->blockalign);
        if (file->sampleframes < 0) {
            return false;
        }
    }

    // Nothing to decode, nothing to return.
    if (file->sampleframes == 0) {
        *audio_buf = NULL;
        *audio_len = 0;
        return true;
    }

    sample_count = (size_t)file->sampleframes;
    if (SafeMult(&sample_count, format->channels)) {
        return SDL_SetError("WAVE file too big");
    }

    expanded_len = sample_count;
    if (SafeMult(&expanded_len, sizeof(Sint16))) {
        return SDL_SetError("WAVE file too big");
    } else if (expanded_len > SDL_MAX_UINT32 || file->sampleframes > SIZE_MAX) {
        return SDL_SetError("WAVE file too big");
    }

    // 1 to avoid allocating zero bytes, to keep static analysis happy.
    src = (Uint8 *)SDL_realloc(chunk->data, expanded_len ? expanded_len : 1);
    if (!src) {
        return false;
    }
    chunk->data = NULL;
    chunk->size = 0;

    dst = (Sint16 *)src;

    /* Work backwards, since we're expanding in-place. `format` will
     * inform the caller about the byte order.
     */
    i = sample_count;
    switch (file->format.encoding) {
#ifdef SDL_WAVE_LAW_LUT
    case ALAW_CODE:
        while (i--) {
            dst[i] = alaw_lut[src[i]];
        }
        break;
    case MULAW_CODE:
        while (i--) {
            dst[i] = mulaw_lut[src[i]];
        }
        break;
#else
    case ALAW_CODE:
        while (i--) {
            Uint8 nibble = src[i];
            Uint8 exponent = (nibble & 0x7f) ^ 0x55;
            Sint16 mantissa = exponent & 0xf;

            exponent >>= 4;
            if (exponent > 0) {
                mantissa |= 0x10;
            }
            mantissa = (mantissa << 4) | 0x8;
            if (exponent > 1) {
                mantissa <<= exponent - 1;
            }

            dst[i] = nibble & 0x80 ? mantissa : -mantissa;
        }
        break;
    case MULAW_CODE:
        while (i--) {
            Uint8 nibble = ~src[i];
            Sint16 mantissa = nibble & 0xf;
            Uint8 exponent = (nibble >> 4) & 0x7;
            Sint16 step = 4 << (exponent + 1);

            mantissa = (0x80 << exponent) + step * mantissa + step / 2 - 132;

            dst[i] = nibble & 0x80 ? -mantissa : mantissa;
        }
        break;
#endif
    default:
        SDL_free(src);
        return SDL_SetError("Unknown companded encoding");
    }

    *audio_buf = src;
    *audio_len = (Uint32)expanded_len;

    return true;
}

static bool PCM_Init(WaveFile *file, size_t datalength)
{
    WaveFormat *format = &file->format;

    if (format->encoding == PCM_CODE) {
        switch (format->bitspersample) {
        case 8:
        case 16:
        case 24:
        case 32:
            // These are supported.
            break;
        default:
            return SDL_SetError("%u-bit PCM format not supported", (unsigned int)format->bitspersample);
        }
    } else if (format->encoding == IEEE_FLOAT_CODE) {
        if (format->bitspersample != 32) {
            return SDL_SetError("%u-bit IEEE floating-point format not supported", (unsigned int)format->bitspersample);
        }
    }

    /* It wouldn't be that hard to support more exotic block sizes, but
     * the most common formats should do for now.
     */
    // Make sure we're a multiple of the blockalign, at least.
    if ((format->channels * format->bitspersample) % (format->blockalign * 8)) {
        return SDL_SetError("Unsupported block alignment");
    }

    if ((file->trunchint == TruncVeryStrict || file->trunchint == TruncStrict)) {
        if (format->blockalign > 1 && datalength % format->blockalign) {
            return SDL_SetError("Truncated data chunk in WAVE file");
        }
    }

    file->sampleframes = WaveAdjustToFactValue(file, datalength / format->blockalign);
    if (file->sampleframes < 0) {
        return false;
    }

    return true;
}

static bool PCM_ConvertSint24ToSint32(WaveFile *file, Uint8 **audio_buf, Uint32 *audio_len)
{
    WaveFormat *format = &file->format;
    WaveChunk *chunk = &file->chunk;
    size_t i, expanded_len, sample_count;
    Uint8 *ptr;

    sample_count = (size_t)file->sampleframes;
    if (SafeMult(&sample_count, format->channels)) {
        return SDL_SetError("WAVE file too big");
    }

    expanded_len = sample_count;
    if (SafeMult(&expanded_len, sizeof(Sint32))) {
        return SDL_SetError("WAVE file too big");
    } else if (expanded_len > SDL_MAX_UINT32 || file->sampleframes > SIZE_MAX) {
        return SDL_SetError("WAVE file too big");
    }

    // 1 to avoid allocating zero bytes, to keep static analysis happy.
    ptr = (Uint8 *)SDL_realloc(chunk->data, expanded_len ? expanded_len : 1);
    if (!ptr) {
        return false;
    }

    // This pointer is now invalid.
    chunk->data = NULL;
    chunk->size = 0;

    *audio_buf = ptr;
    *audio_len = (Uint32)expanded_len;

    // work from end to start, since we're expanding in-place.
    for (i = sample_count; i > 0; i--) {
        const size_t o = i - 1;
        uint8_t b[4];

        b[0] = 0;
        b[1] = ptr[o * 3];
        b[2] = ptr[o * 3 + 1];
        b[3] = ptr[o * 3 + 2];

        ptr[o * 4 + 0] = b[0];
        ptr[o * 4 + 1] = b[1];
        ptr[o * 4 + 2] = b[2];
        ptr[o * 4 + 3] = b[3];
    }

    return true;
}

static bool PCM_Decode(WaveFile *file, Uint8 **audio_buf, Uint32 *audio_len)
{
    WaveFormat *format = &file->format;
    WaveChunk *chunk = &file->chunk;
    size_t outputsize;

    if (chunk->length != chunk->size) {
        file->sampleframes = WaveAdjustToFactValue(file, chunk->size / format->blockalign);
        if (file->sampleframes < 0) {
            return false;
        }
    }

    // Nothing to decode, nothing to return.
    if (file->sampleframes == 0) {
        *audio_buf = NULL;
        *audio_len = 0;
        return true;
    }

    // 24-bit samples get shifted to 32 bits.
    if (format->encoding == PCM_CODE && format->bitspersample == 24) {
        return PCM_ConvertSint24ToSint32(file, audio_buf, audio_len);
    }

    outputsize = (size_t)file->sampleframes;
    if (SafeMult(&outputsize, format->blockalign)) {
        return SDL_SetError("WAVE file too big");
    } else if (outputsize > SDL_MAX_UINT32 || file->sampleframes > SIZE_MAX) {
        return SDL_SetError("WAVE file too big");
    }

    *audio_buf = chunk->data;
    *audio_len = (Uint32)outputsize;

    // This pointer is going to be returned to the caller. Prevent free in cleanup.
    chunk->data = NULL;
    chunk->size = 0;

    return true;
}

static WaveRiffSizeHint WaveGetRiffSizeHint(void)
{
    const char *hint = SDL_GetHint(SDL_HINT_WAVE_RIFF_CHUNK_SIZE);

    if (hint) {
        if (SDL_strcmp(hint, "force") == 0) {
            return RiffSizeForce;
        } else if (SDL_strcmp(hint, "ignore") == 0) {
            return RiffSizeIgnore;
        } else if (SDL_strcmp(hint, "ignorezero") == 0) {
            return RiffSizeIgnoreZero;
        } else if (SDL_strcmp(hint, "maximum") == 0) {
            return RiffSizeMaximum;
        }
    }

    return RiffSizeNoHint;
}

static WaveTruncationHint WaveGetTruncationHint(void)
{
    const char *hint = SDL_GetHint(SDL_HINT_WAVE_TRUNCATION);

    if (hint) {
        if (SDL_strcmp(hint, "verystrict") == 0) {
            return TruncVeryStrict;
        } else if (SDL_strcmp(hint, "strict") == 0) {
            return TruncStrict;
        } else if (SDL_strcmp(hint, "dropframe") == 0) {
            return TruncDropFrame;
        } else if (SDL_strcmp(hint, "dropblock") == 0) {
            return TruncDropBlock;
        }
    }

    return TruncNoHint;
}

static WaveFactChunkHint WaveGetFactChunkHint(void)
{
    const char *hint = SDL_GetHint(SDL_HINT_WAVE_FACT_CHUNK);

    if (hint) {
        if (SDL_strcmp(hint, "truncate") == 0) {
            return FactTruncate;
        } else if (SDL_strcmp(hint, "strict") == 0) {
            return FactStrict;
        } else if (SDL_strcmp(hint, "ignorezero") == 0) {
            return FactIgnoreZero;
        } else if (SDL_strcmp(hint, "ignore") == 0) {
            return FactIgnore;
        }
    }

    return FactNoHint;
}

static void WaveFreeChunkData(WaveChunk *chunk)
{
    if (chunk->data) {
        SDL_free(chunk->data);
        chunk->data = NULL;
    }
    chunk->size = 0;
}

static int WaveNextChunk(SDL_IOStream *src, WaveChunk *chunk)
{
    Uint32 chunkheader[2];
    Sint64 nextposition = chunk->position + chunk->length;

    // Data is no longer valid after this function returns.
    WaveFreeChunkData(chunk);

    // Error on overflows.
    if (SDL_MAX_SINT64 - chunk->length < chunk->position || SDL_MAX_SINT64 - 8 < nextposition) {
        return -1;
    }

    // RIFF chunks have a 2-byte alignment. Skip padding byte.
    if (chunk->length & 1) {
        nextposition++;
    }

    if (SDL_SeekIO(src, nextposition, SDL_IO_SEEK_SET) != nextposition) {
        // Not sure how we ended up here. Just abort.
        return -2;
    } else if (SDL_ReadIO(src, chunkheader, sizeof(Uint32) * 2) != (sizeof(Uint32) * 2)) {
        return -1;
    }

    chunk->fourcc = SDL_Swap32LE(chunkheader[0]);
    chunk->length = SDL_Swap32LE(chunkheader[1]);
    chunk->position = nextposition + 8;

    return 0;
}

static int WaveReadPartialChunkData(SDL_IOStream *src, WaveChunk *chunk, size_t length)
{
    WaveFreeChunkData(chunk);

    if (length > chunk->length) {
        length = chunk->length;
    }

    if (length > 0) {
        chunk->data = (Uint8 *)SDL_malloc(length);
        if (!chunk->data) {
            return -1;
        }

        if (SDL_SeekIO(src, chunk->position, SDL_IO_SEEK_SET) != chunk->position) {
            // Not sure how we ended up here. Just abort.
            return -2;
        }

        chunk->size = SDL_ReadIO(src, chunk->data, length);
        if (chunk->size != length) {
            // Expected to be handled by the caller.
        }
    }

    return 0;
}

static int WaveReadChunkData(SDL_IOStream *src, WaveChunk *chunk)
{
    return WaveReadPartialChunkData(src, chunk, chunk->length);
}

typedef struct WaveExtensibleGUID
{
    Uint16 encoding;
    Uint8 guid[16];
} WaveExtensibleGUID;

// Some of the GUIDs that are used by WAVEFORMATEXTENSIBLE.
#define WAVE_FORMATTAG_GUID(tag)                                                     \
    {                                                                                \
        (tag) & 0xff, (tag) >> 8, 0, 0, 0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113 \
    }
static WaveExtensibleGUID extensible_guids[] = {
    { PCM_CODE, WAVE_FORMATTAG_GUID(PCM_CODE) },
    { MS_ADPCM_CODE, WAVE_FORMATTAG_GUID(MS_ADPCM_CODE) },
    { IEEE_FLOAT_CODE, WAVE_FORMATTAG_GUID(IEEE_FLOAT_CODE) },
    { ALAW_CODE, WAVE_FORMATTAG_GUID(ALAW_CODE) },
    { MULAW_CODE, WAVE_FORMATTAG_GUID(MULAW_CODE) },
    { IMA_ADPCM_CODE, WAVE_FORMATTAG_GUID(IMA_ADPCM_CODE) }
};

static Uint16 WaveGetFormatGUIDEncoding(WaveFormat *format)
{
    size_t i;
    for (i = 0; i < SDL_arraysize(extensible_guids); i++) {
        if (SDL_memcmp(format->subformat, extensible_guids[i].guid, 16) == 0) {
            return extensible_guids[i].encoding;
        }
    }
    return UNKNOWN_CODE;
}

static bool WaveReadFormat(WaveFile *file)
{
    WaveChunk *chunk = &file->chunk;
    WaveFormat *format = &file->format;
    SDL_IOStream *fmtsrc;
    size_t fmtlen = chunk->size;

    if (fmtlen > SDL_MAX_SINT32) {
        // Limit given by SDL_IOFromConstMem.
        return SDL_SetError("Data of WAVE fmt chunk too big");
    }
    fmtsrc = SDL_IOFromConstMem(chunk->data, (int)chunk->size);
    if (!fmtsrc) {
        return false;
    }

    if (!SDL_ReadU16LE(fmtsrc, &format->formattag) ||
        !SDL_ReadU16LE(fmtsrc, &format->channels) ||
        !SDL_ReadU32LE(fmtsrc, &format->frequency) ||
        !SDL_ReadU32LE(fmtsrc, &format->byterate) ||
        !SDL_ReadU16LE(fmtsrc, &format->blockalign)) {
        return false;
    }
    format->encoding = format->formattag;

    // This is PCM specific in the first version of the specification.
    if (fmtlen >= 16) {
        if (!SDL_ReadU16LE(fmtsrc, &format->bitspersample)) {
            return false;
        }
    } else if (format->encoding == PCM_CODE) {
        SDL_CloseIO(fmtsrc);
        return SDL_SetError("Missing wBitsPerSample field in WAVE fmt chunk");
    }

    // The earlier versions also don't have this field.
    if (fmtlen >= 18) {
        if (!SDL_ReadU16LE(fmtsrc, &format->extsize)) {
            return false;
        }
    }

    if (format->formattag == EXTENSIBLE_CODE) {
        /* note that this ignores channel masks, smaller valid bit counts
         * inside a larger container, and most subtypes. This is just enough
         * to get things that didn't really _need_ WAVE_FORMAT_EXTENSIBLE
         * to be useful working when they use this format flag.
         */

        // Extensible header must be at least 22 bytes.
        if (fmtlen < 40 || format->extsize < 22) {
            SDL_CloseIO(fmtsrc);
            return SDL_SetError("Extensible WAVE header too small");
        }

        if (!SDL_ReadU16LE(fmtsrc, &format->validsamplebits) ||
            !SDL_ReadU32LE(fmtsrc, &format->channelmask) ||
            SDL_ReadIO(fmtsrc, format->subformat, 16) != 16) {
        }
        format->samplesperblock = format->validsamplebits;
        format->encoding = WaveGetFormatGUIDEncoding(format);
    }

    SDL_CloseIO(fmtsrc);

    return true;
}

static bool WaveCheckFormat(WaveFile *file, size_t datalength)
{
    WaveFormat *format = &file->format;

    // Check for some obvious issues.

    if (format->channels == 0) {
        return SDL_SetError("Invalid number of channels");
    }

    if (format->frequency == 0) {
        return SDL_SetError("Invalid sample rate");
    } else if (format->frequency > INT_MAX) {
        return SDL_SetError("Sample rate exceeds limit of %d", INT_MAX);
    }

    // Reject invalid fact chunks in strict mode.
    if (file->facthint == FactStrict && file->fact.status == -1) {
        return SDL_SetError("Invalid fact chunk in WAVE file");
    }

    /* Check for issues common to all encodings. Some unsupported formats set
     * the bits per sample to zero. These fall through to the 'unsupported
     * format' error.
     */
    switch (format->encoding) {
    case IEEE_FLOAT_CODE:
    case ALAW_CODE:
    case MULAW_CODE:
    case MS_ADPCM_CODE:
    case IMA_ADPCM_CODE:
        // These formats require a fact chunk.
        if (file->facthint == FactStrict && file->fact.status <= 0) {
            return SDL_SetError("Missing fact chunk in WAVE file");
        }
        SDL_FALLTHROUGH;
    case PCM_CODE:
        // All supported formats require a non-zero bit depth.
        if (file->chunk.size < 16) {
            return SDL_SetError("Missing wBitsPerSample field in WAVE fmt chunk");
        } else if (format->bitspersample == 0) {
            return SDL_SetError("Invalid bits per sample");
        }

        // All supported formats must have a proper block size.
        if (format->blockalign == 0) {
            format->blockalign = 1;  // force it to 1 if it was unset.
        }

        /* If the fact chunk is valid and the appropriate hint is set, the
         * decoders will use the number of sample frames from the fact chunk.
         */
        if (file->fact.status == 1) {
            WaveFactChunkHint hint = file->facthint;
            Uint32 samples = file->fact.samplelength;
            if (hint == FactTruncate || hint == FactStrict || (hint == FactIgnoreZero && samples > 0)) {
                file->fact.status = 2;
            }
        }
    }

    // Check the format for encoding specific issues and initialize decoders.
    switch (format->encoding) {
    case PCM_CODE:
    case IEEE_FLOAT_CODE:
        if (!PCM_Init(file, datalength)) {
            return false;
        }
        break;
    case ALAW_CODE:
    case MULAW_CODE:
        if (!LAW_Init(file, datalength)) {
            return false;
        }
        break;
    case MS_ADPCM_CODE:
        if (!MS_ADPCM_Init(file, datalength)) {
            return false;
        }
        break;
    case IMA_ADPCM_CODE:
        if (!IMA_ADPCM_Init(file, datalength)) {
            return false;
        }
        break;
    case MPEG_CODE:
    case MPEGLAYER3_CODE:
        return SDL_SetError("MPEG formats not supported");
    default:
        if (format->formattag == EXTENSIBLE_CODE) {
            const char *errstr = "Unknown WAVE format GUID: %08x-%04x-%04x-%02x%02x%02x%02x%02x%02x%02x%02x";
            const Uint8 *g = format->subformat;
            const Uint32 g1 = g[0] | ((Uint32)g[1] << 8) | ((Uint32)g[2] << 16) | ((Uint32)g[3] << 24);
            const Uint32 g2 = g[4] | ((Uint32)g[5] << 8);
            const Uint32 g3 = g[6] | ((Uint32)g[7] << 8);
            return SDL_SetError(errstr, g1, g2, g3, g[8], g[9], g[10], g[11], g[12], g[13], g[14], g[15]);
        }
        return SDL_SetError("Unknown WAVE format tag: 0x%04x", (unsigned int)format->encoding);
    }

    return true;
}

static bool WaveLoad(SDL_IOStream *src, WaveFile *file, SDL_AudioSpec *spec, Uint8 **audio_buf, Uint32 *audio_len)
{
    int result;
    Uint32 chunkcount = 0;
    Uint32 chunkcountlimit = 10000;
    const char *hint;
    Sint64 RIFFstart, RIFFend, lastchunkpos;
    bool RIFFlengthknown = false;
    WaveFormat *format = &file->format;
    WaveChunk *chunk = &file->chunk;
    WaveChunk RIFFchunk;
    WaveChunk fmtchunk;
    WaveChunk datachunk;

    SDL_zero(RIFFchunk);
    SDL_zero(fmtchunk);
    SDL_zero(datachunk);

    hint = SDL_GetHint(SDL_HINT_WAVE_CHUNK_LIMIT);
    if (hint) {
        unsigned int count;
        if (SDL_sscanf(hint, "%u", &count) == 1) {
            chunkcountlimit = count <= SDL_MAX_UINT32 ? count : SDL_MAX_UINT32;
        }
    }

    RIFFstart = SDL_TellIO(src);
    if (RIFFstart < 0) {
        return SDL_SetError("Could not seek in file");
    }

    RIFFchunk.position = RIFFstart;
    if (WaveNextChunk(src, &RIFFchunk) < 0) {
        return SDL_SetError("Could not read RIFF header");
    }

    // Check main WAVE file identifiers.
    if (RIFFchunk.fourcc == RIFF) {
        Uint32 formtype;
        // Read the form type. "WAVE" expected.
        if (!SDL_ReadU32LE(src, &formtype)) {
            return SDL_SetError("Could not read RIFF form type");
        } else if (formtype != WAVE) {
            return SDL_SetError("RIFF form type is not WAVE (not a Waveform file)");
        }
    } else if (RIFFchunk.fourcc == WAVE) {
        // RIFF chunk missing or skipped. Length unknown.
        RIFFchunk.position = 0;
        RIFFchunk.length = 0;
    } else {
        return SDL_SetError("Could not find RIFF or WAVE identifiers (not a Waveform file)");
    }

    // The 4-byte form type is immediately followed by the first chunk.
    chunk->position = RIFFchunk.position + 4;

    /* Use the RIFF chunk size to limit the search for the chunks. This is not
     * always reliable and the hint can be used to tune the behavior. By
     * default, it will never search past 4 GiB.
     */
    switch (file->riffhint) {
    case RiffSizeIgnore:
        RIFFend = RIFFchunk.position + SDL_MAX_UINT32;
        break;
    default:
    case RiffSizeIgnoreZero:
        if (RIFFchunk.length == 0) {
            RIFFend = RIFFchunk.position + SDL_MAX_UINT32;
            break;
        }
        SDL_FALLTHROUGH;
    case RiffSizeForce:
        RIFFend = RIFFchunk.position + RIFFchunk.length;
        RIFFlengthknown = true;
        break;
    case RiffSizeMaximum:
        RIFFend = SDL_MAX_SINT64;
        break;
    }

    /* Step through all chunks and save information on the fmt, data, and fact
     * chunks. Ignore the chunks we don't know as per specification. This
     * currently also ignores cue, list, and slnt chunks.
     */
    while ((Uint64)RIFFend > (Uint64)chunk->position + chunk->length + (chunk->length & 1)) {
        // Abort after too many chunks or else corrupt files may waste time.
        if (chunkcount++ >= chunkcountlimit) {
            return SDL_SetError("Chunk count in WAVE file exceeds limit of %" SDL_PRIu32, chunkcountlimit);
        }

        result = WaveNextChunk(src, chunk);
        if (result < 0) {
            // Unexpected EOF. Corrupt file or I/O issues.
            if (file->trunchint == TruncVeryStrict) {
                return SDL_SetError("Unexpected end of WAVE file");
            }
            // Let the checks after this loop sort this issue out.
            break;
        } else if (result == -2) {
            return SDL_SetError("Could not seek to WAVE chunk header");
        }

        if (chunk->fourcc == FMT) {
            if (fmtchunk.fourcc == FMT) {
                // Multiple fmt chunks. Ignore or error?
            } else {
                // The fmt chunk must occur before the data chunk.
                if (datachunk.fourcc == DATA) {
                    return SDL_SetError("fmt chunk after data chunk in WAVE file");
                }
                fmtchunk = *chunk;
            }
        } else if (chunk->fourcc == DATA) {
            /* Only use the first data chunk. Handling the wavl list madness
             * may require a different approach.
             */
            if (datachunk.fourcc != DATA) {
                datachunk = *chunk;
            }
        } else if (chunk->fourcc == FACT) {
            /* The fact chunk data must be at least 4 bytes for the
             * dwSampleLength field. Ignore all fact chunks after the first one.
             */
            if (file->fact.status == 0) {
                if (chunk->length < 4) {
                    file->fact.status = -1;
                } else {
                    // Let's use src directly, it's just too convenient.
                    Sint64 position = SDL_SeekIO(src, chunk->position, SDL_IO_SEEK_SET);
                    if (position == chunk->position && SDL_ReadU32LE(src, &file->fact.samplelength)) {
                        file->fact.status = 1;
                    } else {
                        file->fact.status = -1;
                    }
                }
            }
        }

        /* Go through all chunks in verystrict mode or stop the search early if
         * all required chunks were found.
         */
        if (file->trunchint == TruncVeryStrict) {
            if ((Uint64)RIFFend < (Uint64)chunk->position + chunk->length) {
                return SDL_SetError("RIFF size truncates chunk");
            }
        } else if (fmtchunk.fourcc == FMT && datachunk.fourcc == DATA) {
            if (file->fact.status == 1 || file->facthint == FactIgnore || file->facthint == FactNoHint) {
                break;
            }
        }
    }

    /* Save the position after the last chunk. This position will be used if the
     * RIFF length is unknown.
     */
    lastchunkpos = chunk->position + chunk->length;

    // The fmt chunk is mandatory.
    if (fmtchunk.fourcc != FMT) {
        return SDL_SetError("Missing fmt chunk in WAVE file");
    }
    // A data chunk must be present.
    if (datachunk.fourcc != DATA) {
        return SDL_SetError("Missing data chunk in WAVE file");
    }
    // Check if the last chunk has all of its data in verystrict mode.
    if (file->trunchint == TruncVeryStrict) {
        // data chunk is handled later.
        if (chunk->fourcc != DATA && chunk->length > 0) {
            Uint8 tmp;
            Uint64 position = (Uint64)chunk->position + chunk->length - 1;
            if (position > SDL_MAX_SINT64 || SDL_SeekIO(src, (Sint64)position, SDL_IO_SEEK_SET) != (Sint64)position) {
                return SDL_SetError("Could not seek to WAVE chunk data");
            } else if (!SDL_ReadU8(src, &tmp)) {
                return SDL_SetError("RIFF size truncates chunk");
            }
        }
    }

    // Process fmt chunk.
    *chunk = fmtchunk;

    /* No need to read more than 1046 bytes of the fmt chunk data with the
     * formats that are currently supported. (1046 because of MS ADPCM coefficients)
     */
    if (WaveReadPartialChunkData(src, chunk, 1046) < 0) {
        return SDL_SetError("Could not read data of WAVE fmt chunk");
    }

    /* The fmt chunk data must be at least 14 bytes to include all common fields.
     * It usually is 16 and larger depending on the header and encoding.
     */
    if (chunk->length < 14) {
        return SDL_SetError("Invalid WAVE fmt chunk length (too small)");
    } else if (chunk->size < 14) {
        return SDL_SetError("Could not read data of WAVE fmt chunk");
    } else if (!WaveReadFormat(file)) {
        return false;
    } else if (!WaveCheckFormat(file, (size_t)datachunk.length)) {
        return false;
    }

#ifdef SDL_WAVE_DEBUG_LOG_FORMAT
    WaveDebugLogFormat(file);
#endif
#ifdef SDL_WAVE_DEBUG_DUMP_FORMAT
    WaveDebugDumpFormat(file, RIFFchunk.length, fmtchunk.length, datachunk.length);
#endif

    WaveFreeChunkData(chunk);

    // Process data chunk.
    *chunk = datachunk;

    if (chunk->length > 0) {
        result = WaveReadChunkData(src, chunk);
        if (result < 0) {
            return false;
        } else if (result == -2) {
            return SDL_SetError("Could not seek data of WAVE data chunk");
        }
    }

    if (chunk->length != chunk->size) {
        // I/O issues or corrupt file.
        if (file->trunchint == TruncVeryStrict || file->trunchint == TruncStrict) {
            return SDL_SetError("Could not read data of WAVE data chunk");
        }
        // The decoders handle this truncation.
    }

    // Decode or convert the data if necessary.
    switch (format->encoding) {
    case PCM_CODE:
    case IEEE_FLOAT_CODE:
        if (!PCM_Decode(file, audio_buf, audio_len)) {
            return false;
        }
        break;
    case ALAW_CODE:
    case MULAW_CODE:
        if (!LAW_Decode(file, audio_buf, audio_len)) {
            return false;
        }
        break;
    case MS_ADPCM_CODE:
        if (!MS_ADPCM_Decode(file, audio_buf, audio_len)) {
            return false;
        }
        break;
    case IMA_ADPCM_CODE:
        if (!IMA_ADPCM_Decode(file, audio_buf, audio_len)) {
            return false;
        }
        break;
    }

    /* Setting up the specs. All unsupported formats were filtered out
     * by checks earlier in this function.
     */
    spec->freq = format->frequency;
    spec->channels = (Uint8)format->channels;
    spec->format = SDL_AUDIO_UNKNOWN;

    switch (format->encoding) {
    case MS_ADPCM_CODE:
    case IMA_ADPCM_CODE:
    case ALAW_CODE:
    case MULAW_CODE:
        // These can be easily stored in the byte order of the system.
        spec->format = SDL_AUDIO_S16;
        break;
    case IEEE_FLOAT_CODE:
        spec->format = SDL_AUDIO_F32LE;
        break;
    case PCM_CODE:
        switch (format->bitspersample) {
        case 8:
            spec->format = SDL_AUDIO_U8;
            break;
        case 16:
            spec->format = SDL_AUDIO_S16LE;
            break;
        case 24: // Has been shifted to 32 bits.
        case 32:
            spec->format = SDL_AUDIO_S32LE;
            break;
        default:
            // Just in case something unexpected happened in the checks.
            return SDL_SetError("Unexpected %u-bit PCM data format", (unsigned int)format->bitspersample);
        }
        break;
    default:
        return SDL_SetError("Unexpected data format");
    }

    // Report the end position back to the cleanup code.
    if (RIFFlengthknown) {
        chunk->position = RIFFend;
    } else {
        chunk->position = lastchunkpos;
    }

    return true;
}

bool SDL_LoadWAV_IO(SDL_IOStream *src, bool closeio, SDL_AudioSpec *spec, Uint8 **audio_buf, Uint32 *audio_len)
{
    bool result = false;
    WaveFile file;

    if (spec) {
        SDL_zerop(spec);
    }
    if (audio_buf) {
        *audio_buf = NULL;
    }
    if (audio_len) {
        *audio_len = 0;
    }

    // Make sure we are passed a valid data source
    if (!src) {
        SDL_InvalidParamError("src");
        goto done;
    } else if (!spec) {
        SDL_InvalidParamError("spec");
        goto done;
    } else if (!audio_buf) {
        SDL_InvalidParamError("audio_buf");
        goto done;
    } else if (!audio_len) {
        SDL_InvalidParamError("audio_len");
        goto done;
    }

    SDL_zero(file);
    file.riffhint = WaveGetRiffSizeHint();
    file.trunchint = WaveGetTruncationHint();
    file.facthint = WaveGetFactChunkHint();

    result = WaveLoad(src, &file, spec, audio_buf, audio_len);
    if (!result) {
        SDL_free(*audio_buf);
        audio_buf = NULL;
        audio_len = 0;
    }

    // Cleanup
    if (!closeio) {
        SDL_SeekIO(src, file.chunk.position, SDL_IO_SEEK_SET);
    }
    WaveFreeChunkData(&file.chunk);
    SDL_free(file.decoderdata);
done:
    if (closeio && src) {
        SDL_CloseIO(src);
    }
    return result;
}

bool SDL_LoadWAV(const char *path, SDL_AudioSpec *spec, Uint8 **audio_buf, Uint32 *audio_len)
{
    SDL_IOStream *stream = SDL_IOFromFile(path, "rb");
    if (!stream) {
        if (spec) {
            SDL_zerop(spec);
        }
        if (audio_buf) {
            *audio_buf = NULL;
        }
        if (audio_len) {
            *audio_len = 0;
        }
        return false;
    }
    return SDL_LoadWAV_IO(stream, true, spec, audio_buf, audio_len);
}


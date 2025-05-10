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

// This provides the default mixing callback for the SDL audio routines

#include "SDL_sysaudio.h"

/* This table is used to add two sound values together and pin
 * the value to avoid overflow.  (used with permission from ARDI)
 */
static const Uint8 mix8[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03,
    0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
    0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19,
    0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24,
    0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A,
    0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50,
    0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x5B,
    0x5C, 0x5D, 0x5E, 0x5F, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66,
    0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70, 0x71,
    0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x7B, 0x7C,
    0x7D, 0x7E, 0x7F, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F, 0x90, 0x91, 0x92,
    0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D,
    0x9E, 0x9F, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8,
    0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF, 0xB0, 0xB1, 0xB2, 0xB3,
    0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD, 0xBE,
    0xBF, 0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
    0xCA, 0xCB, 0xCC, 0xCD, 0xCE, 0xCF, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4,
    0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF,
    0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
    0xEB, 0xEC, 0xED, 0xEE, 0xEF, 0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5,
    0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
};

// The volume ranges from 0 - 128
#define MIX_MAXVOLUME 128
#define ADJUST_VOLUME(type, s, v) ((s) = (type)(((s) * (v)) / MIX_MAXVOLUME))
#define ADJUST_VOLUME_U8(s, v)    ((s) = (Uint8)(((((s) - 128) * (v)) / MIX_MAXVOLUME) + 128))

// !!! FIXME: This needs some SIMD magic.
// !!! FIXME: Add fast-path for volume = 1
// !!! FIXME: Use larger scales for 16-bit/32-bit integers

bool SDL_MixAudio(Uint8 *dst, const Uint8 *src, SDL_AudioFormat format, Uint32 len, float fvolume)
{
    int volume = (int)SDL_roundf(fvolume * MIX_MAXVOLUME);

    if (volume == 0) {
        return true;
    }

    switch (format) {

    case SDL_AUDIO_U8:
    {
        Uint8 src_sample;

        while (len--) {
            src_sample = *src;
            ADJUST_VOLUME_U8(src_sample, volume);
            *dst = mix8[*dst + src_sample];
            ++dst;
            ++src;
        }
    } break;

    case SDL_AUDIO_S8:
    {
        Sint8 *dst8, *src8;
        Sint8 src_sample;
        int dst_sample;
        const int max_audioval = SDL_MAX_SINT8;
        const int min_audioval = SDL_MIN_SINT8;

        src8 = (Sint8 *)src;
        dst8 = (Sint8 *)dst;
        while (len--) {
            src_sample = *src8;
            ADJUST_VOLUME(Sint8, src_sample, volume);
            dst_sample = *dst8 + src_sample;
            if (dst_sample > max_audioval) {
                dst_sample = max_audioval;
            } else if (dst_sample < min_audioval) {
                dst_sample = min_audioval;
            }
            *dst8 = (Sint8)dst_sample;
            ++dst8;
            ++src8;
        }
    } break;

    case SDL_AUDIO_S16LE:
    {
        Sint16 src1, src2;
        int dst_sample;
        const int max_audioval = SDL_MAX_SINT16;
        const int min_audioval = SDL_MIN_SINT16;

        len /= 2;
        while (len--) {
            src1 = SDL_Swap16LE(*(Sint16 *)src);
            ADJUST_VOLUME(Sint16, src1, volume);
            src2 = SDL_Swap16LE(*(Sint16 *)dst);
            src += 2;
            dst_sample = src1 + src2;
            if (dst_sample > max_audioval) {
                dst_sample = max_audioval;
            } else if (dst_sample < min_audioval) {
                dst_sample = min_audioval;
            }
            *(Sint16 *)dst = SDL_Swap16LE((Sint16)dst_sample);
            dst += 2;
        }
    } break;

    case SDL_AUDIO_S16BE:
    {
        Sint16 src1, src2;
        int dst_sample;
        const int max_audioval = SDL_MAX_SINT16;
        const int min_audioval = SDL_MIN_SINT16;

        len /= 2;
        while (len--) {
            src1 = SDL_Swap16BE(*(Sint16 *)src);
            ADJUST_VOLUME(Sint16, src1, volume);
            src2 = SDL_Swap16BE(*(Sint16 *)dst);
            src += 2;
            dst_sample = src1 + src2;
            if (dst_sample > max_audioval) {
                dst_sample = max_audioval;
            } else if (dst_sample < min_audioval) {
                dst_sample = min_audioval;
            }
            *(Sint16 *)dst = SDL_Swap16BE((Sint16)dst_sample);
            dst += 2;
        }
    } break;

    case SDL_AUDIO_S32LE:
    {
        const Uint32 *src32 = (Uint32 *)src;
        Uint32 *dst32 = (Uint32 *)dst;
        Sint64 src1, src2;
        Sint64 dst_sample;
        const Sint64 max_audioval = SDL_MAX_SINT32;
        const Sint64 min_audioval = SDL_MIN_SINT32;

        len /= 4;
        while (len--) {
            src1 = (Sint64)((Sint32)SDL_Swap32LE(*src32));
            src32++;
            ADJUST_VOLUME(Sint64, src1, volume);
            src2 = (Sint64)((Sint32)SDL_Swap32LE(*dst32));
            dst_sample = src1 + src2;
            if (dst_sample > max_audioval) {
                dst_sample = max_audioval;
            } else if (dst_sample < min_audioval) {
                dst_sample = min_audioval;
            }
            *(dst32++) = SDL_Swap32LE((Uint32)((Sint32)dst_sample));
        }
    } break;

    case SDL_AUDIO_S32BE:
    {
        const Uint32 *src32 = (Uint32 *)src;
        Uint32 *dst32 = (Uint32 *)dst;
        Sint64 src1, src2;
        Sint64 dst_sample;
        const Sint64 max_audioval = SDL_MAX_SINT32;
        const Sint64 min_audioval = SDL_MIN_SINT32;

        len /= 4;
        while (len--) {
            src1 = (Sint64)((Sint32)SDL_Swap32BE(*src32));
            src32++;
            ADJUST_VOLUME(Sint64, src1, volume);
            src2 = (Sint64)((Sint32)SDL_Swap32BE(*dst32));
            dst_sample = src1 + src2;
            if (dst_sample > max_audioval) {
                dst_sample = max_audioval;
            } else if (dst_sample < min_audioval) {
                dst_sample = min_audioval;
            }
            *(dst32++) = SDL_Swap32BE((Uint32)((Sint32)dst_sample));
        }
    } break;

    case SDL_AUDIO_F32LE:
    {
        const float *src32 = (float *)src;
        float *dst32 = (float *)dst;
        float src1, src2;
        float dst_sample;
        const float max_audioval = 1.0f;
        const float min_audioval = -1.0f;

        len /= 4;
        while (len--) {
            src1 = SDL_SwapFloatLE(*src32) * fvolume;
            src2 = SDL_SwapFloatLE(*dst32);
            src32++;

            dst_sample = src1 + src2;
            if (dst_sample > max_audioval) {
                dst_sample = max_audioval;
            } else if (dst_sample < min_audioval) {
                dst_sample = min_audioval;
            }
            *(dst32++) = SDL_SwapFloatLE(dst_sample);
        }
    } break;

    case SDL_AUDIO_F32BE:
    {
        const float *src32 = (float *)src;
        float *dst32 = (float *)dst;
        float src1, src2;
        float dst_sample;
        const float max_audioval = 1.0f;
        const float min_audioval = -1.0f;

        len /= 4;
        while (len--) {
            src1 = SDL_SwapFloatBE(*src32) * fvolume;
            src2 = SDL_SwapFloatBE(*dst32);
            src32++;

            dst_sample = src1 + src2;
            if (dst_sample > max_audioval) {
                dst_sample = max_audioval;
            } else if (dst_sample < min_audioval) {
                dst_sample = min_audioval;
            }
            *(dst32++) = SDL_SwapFloatBE(dst_sample);
        }
    } break;

    default: // If this happens... FIXME!
        return SDL_SetError("SDL_MixAudio(): unknown audio format");
    }

    return true;
}

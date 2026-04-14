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

// Public domain murmur3 32-bit hash algorithm
//
// Adapted from: https://en.wikipedia.org/wiki/MurmurHash

static SDL_INLINE Uint32 murmur_32_scramble(Uint32 k)
{
    k *= 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    return k;
}

Uint32 SDLCALL SDL_murmur3_32(const void *data, size_t len, Uint32 seed)
{
    const Uint8 *bytes = (const Uint8 *)data;
    Uint32 hash = seed;
    Uint32 k;

    // Read in groups of 4.
    if ((((uintptr_t)bytes) & 3) == 0) {
        // We can do aligned 32-bit reads
        for (size_t i = len >> 2; i--; ) {
            k = *(const Uint32 *)bytes;
            k = SDL_Swap32LE(k);
            bytes += sizeof(Uint32);
            hash ^= murmur_32_scramble(k);
            hash = (hash << 13) | (hash >> 19);
            hash = hash * 5 + 0xe6546b64;
        }
    } else {
        for (size_t i = len >> 2; i--; ) {
            SDL_memcpy(&k, bytes, sizeof(Uint32));
            k = SDL_Swap32LE(k);
            bytes += sizeof(Uint32);
            hash ^= murmur_32_scramble(k);
            hash = (hash << 13) | (hash >> 19);
            hash = hash * 5 + 0xe6546b64;
        }
    }

    // Read the rest.
    size_t left = (len & 3);
    if (left) {
        k = 0;
        for (size_t i = left; i--; ) {
            k <<= 8;
            k |= bytes[i];
        }

        // A swap is *not* necessary here because the preceding loop already
        // places the low bytes in the low places according to whatever endianness
        // we use. Swaps only apply when the memory is copied in a chunk.
        hash ^= murmur_32_scramble(k);
    }

    /* Finalize. */
    hash ^= len;
    hash ^= hash >> 16;
    hash *= 0x85ebca6b;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35;
    hash ^= hash >> 16;

    return hash;
}

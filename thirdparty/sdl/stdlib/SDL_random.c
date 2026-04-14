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

// This file contains portable random functions for SDL

static Uint64 SDL_rand_state;
static bool SDL_rand_initialized = false;

void SDL_srand(Uint64 seed)
{
    if (!seed) {
        seed = SDL_GetPerformanceCounter();
    }
    SDL_rand_state = seed;
    SDL_rand_initialized = true;
}

Sint32 SDL_rand(Sint32 n)
{
    if (!SDL_rand_initialized) {
        SDL_srand(0);
    }

    return SDL_rand_r(&SDL_rand_state, n);
}

float SDL_randf(void)
{
    if (!SDL_rand_initialized) {
        SDL_srand(0);
    }

    return SDL_randf_r(&SDL_rand_state);
}

Uint32 SDL_rand_bits(void)
{
    if (!SDL_rand_initialized) {
        SDL_srand(0);
    }

    return SDL_rand_bits_r(&SDL_rand_state);
}

Uint32 SDL_rand_bits_r(Uint64 *state)
{
    if (!state) {
        return 0;
    }

    // The C and A parameters of this LCG have been chosen based on hundreds
    // of core-hours of testing with PractRand and TestU01's Crush.
    // Using a 32-bit A improves performance on 32-bit architectures.
    // C can be any odd number, but < 256 generates smaller code on ARM32
    // These values perform as well as a full 64-bit implementation against
    // Crush and PractRand. Plus, their worst-case performance is better
    // than common 64-bit constants when tested against PractRand using seeds
    // with only a single bit set.

    // We tested all 32-bit and 33-bit A with all C < 256 from a v2 of:
    // Steele GL, Vigna S. Computationally easy, spectrally good multipliers
    // for congruential pseudorandom number generators.
    // Softw Pract Exper. 2022;52(2):443-458. doi: 10.1002/spe.3030
    // https://arxiv.org/abs/2001.05304v2

    *state = *state * 0xff1cd035ul + 0x05;

    // Only return top 32 bits because they have a longer period
    return (Uint32)(*state >> 32);
}

Sint32 SDL_rand_r(Uint64 *state, Sint32 n)
{
    // Algorithm: get 32 bits from SDL_rand_bits() and treat it as a 0.32 bit
    // fixed point number. Multiply by the 31.0 bit n to get a 31.32 bit
    // result. Shift right by 32 to get the 31 bit integer that we want.

    if (n < 0) {
        // The algorithm looks like it works for numbers < 0 but it has an
        // infinitesimal chance of returning a value out of range.
        // Returning -SDL_rand(abs(n)) blows up at INT_MIN instead.
        // It's easier to just say no.
        return 0;
    }

    // On 32-bit arch, the compiler will optimize to a single 32-bit multiply
    Uint64 val = (Uint64)SDL_rand_bits_r(state) * n;
    return (Sint32)(val >> 32);
}

float SDL_randf_r(Uint64 *state)
{
    // Note: its using 24 bits because float has 23 bits significand + 1 implicit bit
    return (SDL_rand_bits_r(state) >> (32 - 24)) * 0x1p-24f;
}


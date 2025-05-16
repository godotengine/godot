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

/*

 Based on automated SDL_Surface tests originally written by Edgar Simo 'bobbens'.

 Rewritten for test lib by Andreas Schiffler.

*/
#include <SDL3/SDL_test.h>

#define FILENAME_SIZE 128

/* Counter for _CompareSurface calls; used for filename creation when comparisons fail */
static int _CompareSurfaceCount = 0;

/* Compare surfaces */
int SDLTest_CompareSurfaces(SDL_Surface *surface, SDL_Surface *referenceSurface, int allowable_error)
{
    int ret;
    int i, j;
    int dist;
    int sampleErrorX = 0, sampleErrorY = 0, sampleDist = 0;
    SDL_Color sampleReference = { 0, 0, 0, 0 };
    SDL_Color sampleActual = { 0, 0, 0, 0 };
    Uint8 R, G, B, A;
    Uint8 Rd, Gd, Bd, Ad;
    char imageFilename[FILENAME_SIZE];
    char referenceFilename[FILENAME_SIZE];

    /* Validate input surfaces */
    if (!surface) {
        SDLTest_LogError("Cannot compare NULL surface");
        return -1;
    }

    if (!referenceSurface) {
        SDLTest_LogError("Cannot compare NULL reference surface");
        return -1;
    }

    /* Make sure surface size is the same. */
    if ((surface->w != referenceSurface->w) || (surface->h != referenceSurface->h)) {
        SDLTest_LogError("Expected %dx%d surface, got %dx%d", referenceSurface->w, referenceSurface->h, surface->w, surface->h);
        return -2;
    }

    /* Sanitize input value */
    if (allowable_error < 0) {
        allowable_error = 0;
    }

    SDL_LockSurface(surface);
    SDL_LockSurface(referenceSurface);

    ret = 0;
    /* Compare image - should be same format. */
    for (j = 0; j < surface->h; j++) {
        for (i = 0; i < surface->w; i++) {
            int temp;

            temp = SDL_ReadSurfacePixel(surface, i, j, &R, &G, &B, &A);
            if (!temp) {
                SDLTest_LogError("Failed to retrieve pixel (%d,%d): %s", i, j, SDL_GetError());
                ret++;
                continue;
            }

            temp = SDL_ReadSurfacePixel(referenceSurface, i, j, &Rd, &Gd, &Bd, &Ad);
            if (!temp) {
                SDLTest_LogError("Failed to retrieve reference pixel (%d,%d): %s", i, j, SDL_GetError());
                ret++;
                continue;
            }

            dist = 0;
            dist += (R - Rd) * (R - Rd);
            dist += (G - Gd) * (G - Gd);
            dist += (B - Bd) * (B - Bd);

            /* Allow some difference in blending accuracy */
            if (dist > allowable_error) {
                ret++;
                if (ret == 1) {
                    sampleErrorX = i;
                    sampleErrorY = j;
                    sampleDist = dist;
                    sampleReference.r = Rd;
                    sampleReference.g = Gd;
                    sampleReference.b = Bd;
                    sampleReference.a = Ad;
                    sampleActual.r = R;
                    sampleActual.g = G;
                    sampleActual.b = B;
                    sampleActual.a = A;
                }
            }
        }
    }

    SDL_UnlockSurface(surface);
    SDL_UnlockSurface(referenceSurface);

    /* Save test image and reference for analysis on failures */
    _CompareSurfaceCount++;
    if (ret != 0) {
        SDLTest_LogError("Comparison of pixels with allowable error of %i failed %i times.", allowable_error, ret);
        SDLTest_LogError("Reference surface format: %s", SDL_GetPixelFormatName(referenceSurface->format));
        SDLTest_LogError("Actual surface format: %s", SDL_GetPixelFormatName(surface->format));
        SDLTest_LogError("First detected occurrence at position %i,%i with a squared RGB-difference of %i.", sampleErrorX, sampleErrorY, sampleDist);
        SDLTest_LogError("Reference pixel: R=%u G=%u B=%u A=%u", sampleReference.r, sampleReference.g, sampleReference.b, sampleReference.a);
        SDLTest_LogError("Actual pixel   : R=%u G=%u B=%u A=%u", sampleActual.r, sampleActual.g, sampleActual.b, sampleActual.a);
        (void)SDL_snprintf(imageFilename, FILENAME_SIZE - 1, "CompareSurfaces%04d_TestOutput.bmp", _CompareSurfaceCount);
        SDL_SaveBMP(surface, imageFilename);
        (void)SDL_snprintf(referenceFilename, FILENAME_SIZE - 1, "CompareSurfaces%04d_Reference.bmp", _CompareSurfaceCount);
        SDL_SaveBMP(referenceSurface, referenceFilename);
        SDLTest_LogError("Surfaces from failed comparison saved as '%s' and '%s'", imageFilename, referenceFilename);
    }

    return ret;
}

int SDLTest_CompareMemory(const void *actual, size_t size_actual, const void *reference, size_t size_reference) {
#define WIDTH 16

    const size_t size_max = SDL_max(size_actual, size_reference);
    size_t i;
    struct {
        const char *header;
        const Uint8 *data;
        size_t size;
    } columns[] = {
        {
            "actual",
            actual,
            size_actual,
        },
        {
            "reference",
            reference,
            size_reference,
        },
    };
    char line_buffer[16 + SDL_arraysize(columns) * (4 * WIDTH + 1) + (SDL_arraysize(columns) - 1) * 2 + 1];

    SDLTest_AssertCheck(size_actual == size_reference, "Sizes of memory blocks must be equal (actual=%" SDL_PRIu64 " expected=%" SDL_PRIu64 ")", (Uint64)size_actual, (Uint64)size_reference);
    if (size_actual == size_reference) {
        int equals;
        equals = SDL_memcmp(actual, reference, size_max) == 0;
        SDLTest_AssertCheck(equals, "Memory blocks contain the same data");
        if (equals) {
            return 0;
        }
    }

    SDL_memset(line_buffer, ' ', sizeof(line_buffer));
    line_buffer[sizeof(line_buffer) - 1] = '\0';
    for (i = 0; i < SDL_arraysize(columns); i++) {
        SDL_memcpy(line_buffer + 16 + 1 + i * (4 * WIDTH + 3), columns[i].header, SDL_strlen(columns[i].header));
    }
    SDLTest_LogError("%s", line_buffer);

    for (i = 0; i < size_max; i += WIDTH) {
        size_t pos = 0;
        size_t col;

        pos += SDL_snprintf(line_buffer + pos, SDL_arraysize(line_buffer) - pos, "%016" SDL_PRIx64 , (Uint64)i);

        for (col = 0; col < SDL_arraysize(columns); col++) {
            size_t j;

            for (j = 0; j < WIDTH; j++) {
                if (i + j < columns[col].size) {
                    pos += SDL_snprintf(line_buffer + pos, SDL_arraysize(line_buffer) - pos, " %02x", columns[col].data[i + j]);
                } else {
                    pos += SDL_snprintf(line_buffer + pos, SDL_arraysize(line_buffer) - pos, "   ");
                }
            }
            pos += SDL_snprintf(line_buffer + pos, SDL_arraysize(line_buffer) - pos, " ");
            for (j = 0; j < WIDTH; j++) {
                char c = ' ';
                if (i + j < columns[col].size) {
                    c = columns[col].data[i + j];
                    if (!SDL_isprint(c)) {
                        c = '.';
                    }
                }
                pos += SDL_snprintf(line_buffer + pos, SDL_arraysize(line_buffer) - pos, "%c", c);
            }
            if (col < SDL_arraysize(columns) - 1) {
                pos += SDL_snprintf(line_buffer + pos, SDL_arraysize(line_buffer), " |");
            }
        }
        SDLTest_LogError("%s", line_buffer);
        SDL_assert(pos == SDL_arraysize(line_buffer) - 1);
    }
#undef WIDTH
    return 1;
}

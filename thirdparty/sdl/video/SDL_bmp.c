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

/*
   Code to load and save surfaces in Windows BMP format.

   Why support BMP format?  Well, it's a native format for Windows, and
   most image processing programs can read and write it.  It would be nice
   to be able to have at least one image format that we can natively load
   and save, and since PNG is so complex that it would bloat the library,
   BMP is a good alternative.

   This code currently supports Win32 DIBs in uncompressed 8 and 24 bpp.
*/

#include "SDL_pixels_c.h"
#include "SDL_surface_c.h"

#define SAVE_32BIT_BMP

// Compression encodings for BMP files
#ifndef BI_RGB
#define BI_RGB       0
#define BI_RLE8      1
#define BI_RLE4      2
#define BI_BITFIELDS 3
#endif

// Logical color space values for BMP files
// https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-wmf/eb4bbd50-b3ce-4917-895c-be31f214797f
#ifndef LCS_WINDOWS_COLOR_SPACE
// 0x57696E20 == "Win "
#define LCS_WINDOWS_COLOR_SPACE 0x57696E20
#endif

#ifndef LCS_sRGB
// 0x73524742 == "sRGB"
#define LCS_sRGB 0x73524742
#endif

// Logical/physical color relationship
// https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-wmf/9fec0834-607d-427d-abd5-ab240fb0db38
#ifndef LCS_GM_GRAPHICS
#define LCS_GM_GRAPHICS 0x00000002
#endif

static bool readRlePixels(SDL_Surface *surface, SDL_IOStream *src, int isRle8)
{
    /*
    | Sets the surface pixels from src.  A bmp image is upside down.
    */
    int pitch = surface->pitch;
    int height = surface->h;
    Uint8 *start = (Uint8 *)surface->pixels;
    Uint8 *end = start + (height * pitch);
    Uint8 *bits = end - pitch, *spot;
    int ofs = 0;
    Uint8 ch;
    Uint8 needsPad;
    const int pixels_per_byte = (isRle8 ? 1 : 2);

#define COPY_PIXEL(x)                \
    spot = &bits[ofs++];             \
    if (spot >= start && spot < end) \
        *spot = (x)

    for (;;) {
        if (!SDL_ReadU8(src, &ch)) {
            return false;
        }
        /*
        | encoded mode starts with a run length, and then a byte
        | with two colour indexes to alternate between for the run
        */
        if (ch) {
            Uint8 pixelvalue;
            if (!SDL_ReadU8(src, &pixelvalue)) {
                return false;
            }
            ch /= pixels_per_byte;
            do {
                COPY_PIXEL(pixelvalue);
            } while (--ch);
        } else {
            /*
            | A leading zero is an escape; it may signal the end of the bitmap,
            | a cursor move, or some absolute data.
            | zero tag may be absolute mode or an escape
            */
            if (!SDL_ReadU8(src, &ch)) {
                return false;
            }
            switch (ch) {
            case 0: // end of line
                ofs = 0;
                bits -= pitch; // go to previous
                break;
            case 1:               // end of bitmap
                return true; // success!
            case 2:               // delta
                if (!SDL_ReadU8(src, &ch)) {
                    return false;
                }
                ofs += ch / pixels_per_byte;

                if (!SDL_ReadU8(src, &ch)) {
                    return false;
                }
                bits -= ((ch / pixels_per_byte) * pitch);
                break;
            default: // no compression
                ch /= pixels_per_byte;
                needsPad = (ch & 1);
                do {
                    Uint8 pixelvalue;
                    if (!SDL_ReadU8(src, &pixelvalue)) {
                        return false;
                    }
                    COPY_PIXEL(pixelvalue);
                } while (--ch);

                // pad at even boundary
                if (needsPad && !SDL_ReadU8(src, &ch)) {
                    return false;
                }
                break;
            }
        }
    }
}

static void CorrectAlphaChannel(SDL_Surface *surface)
{
    // Check to see if there is any alpha channel data
    bool hasAlpha = false;
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
    int alphaChannelOffset = 0;
#else
    int alphaChannelOffset = 3;
#endif
    Uint8 *alpha = ((Uint8 *)surface->pixels) + alphaChannelOffset;
    Uint8 *end = alpha + surface->h * surface->pitch;

    while (alpha < end) {
        if (*alpha != 0) {
            hasAlpha = true;
            break;
        }
        alpha += 4;
    }

    if (!hasAlpha) {
        alpha = ((Uint8 *)surface->pixels) + alphaChannelOffset;
        while (alpha < end) {
            *alpha = SDL_ALPHA_OPAQUE;
            alpha += 4;
        }
    }
}

SDL_Surface *SDL_LoadBMP_IO(SDL_IOStream *src, bool closeio)
{
    bool was_error = true;
    Sint64 fp_offset = 0;
    int i, pad;
    SDL_Surface *surface;
    Uint32 Rmask = 0;
    Uint32 Gmask = 0;
    Uint32 Bmask = 0;
    Uint32 Amask = 0;
    Uint8 *bits;
    Uint8 *top, *end;
    bool topDown;
    bool haveRGBMasks = false;
    bool haveAlphaMask = false;
    bool correctAlpha = false;

    // The Win32 BMP file header (14 bytes)
    char magic[2];
    // Uint32 bfSize;
    // Uint16 bfReserved1;
    // Uint16 bfReserved2;
    Uint32 bfOffBits;

    // The Win32 BITMAPINFOHEADER struct (40 bytes)
    Uint32 biSize;
    Sint32 biWidth = 0;
    Sint32 biHeight = 0;
    // Uint16 biPlanes;
    Uint16 biBitCount = 0;
    Uint32 biCompression = 0;
    // Uint32 biSizeImage;
    // Sint32 biXPelsPerMeter;
    // Sint32 biYPelsPerMeter;
    Uint32 biClrUsed = 0;
    // Uint32 biClrImportant;

    // Make sure we are passed a valid data source
    surface = NULL;
    if (!src) {
        SDL_InvalidParamError("src");
        goto done;
    }

    // Read in the BMP file header
    fp_offset = SDL_TellIO(src);
    if (fp_offset < 0) {
        goto done;
    }
    SDL_ClearError();
    if (SDL_ReadIO(src, magic, 2) != 2) {
        goto done;
    }
    if (SDL_strncmp(magic, "BM", 2) != 0) {
        SDL_SetError("File is not a Windows BMP file");
        goto done;
    }
    if (!SDL_ReadU32LE(src, NULL /* bfSize */) ||
        !SDL_ReadU16LE(src, NULL /* bfReserved1 */) ||
        !SDL_ReadU16LE(src, NULL /* bfReserved2 */) ||
        !SDL_ReadU32LE(src, &bfOffBits)) {
        goto done;
    }

    // Read the Win32 BITMAPINFOHEADER
    if (!SDL_ReadU32LE(src, &biSize)) {
        goto done;
    }
    if (biSize == 12) { // really old BITMAPCOREHEADER
        Uint16 biWidth16, biHeight16;
        if (!SDL_ReadU16LE(src, &biWidth16) ||
            !SDL_ReadU16LE(src, &biHeight16) ||
            !SDL_ReadU16LE(src, NULL /* biPlanes */) ||
            !SDL_ReadU16LE(src, &biBitCount)) {
            goto done;
        }
        biWidth = biWidth16;
        biHeight = biHeight16;
        biCompression = BI_RGB;
        // biSizeImage = 0;
        // biXPelsPerMeter = 0;
        // biYPelsPerMeter = 0;
        biClrUsed = 0;
        // biClrImportant = 0;
    } else if (biSize >= 40) { // some version of BITMAPINFOHEADER
        Uint32 headerSize;
        if (!SDL_ReadS32LE(src, &biWidth) ||
            !SDL_ReadS32LE(src, &biHeight) ||
            !SDL_ReadU16LE(src, NULL /* biPlanes */) ||
            !SDL_ReadU16LE(src, &biBitCount) ||
            !SDL_ReadU32LE(src, &biCompression) ||
            !SDL_ReadU32LE(src, NULL /* biSizeImage */) ||
            !SDL_ReadU32LE(src, NULL /* biXPelsPerMeter */) ||
            !SDL_ReadU32LE(src, NULL /* biYPelsPerMeter */) ||
            !SDL_ReadU32LE(src, &biClrUsed) ||
            !SDL_ReadU32LE(src, NULL /* biClrImportant */)) {
            goto done;
        }

        // 64 == BITMAPCOREHEADER2, an incompatible OS/2 2.x extension. Skip this stuff for now.
        if (biSize != 64) {
            /* This is complicated. If compression is BI_BITFIELDS, then
               we have 3 DWORDS that specify the RGB masks. This is either
               stored here in an BITMAPV2INFOHEADER (which only differs in
               that it adds these RGB masks) and biSize >= 52, or we've got
               these masks stored in the exact same place, but strictly
               speaking, this is the bmiColors field in BITMAPINFO immediately
               following the legacy v1 info header, just past biSize. */
            if (biCompression == BI_BITFIELDS) {
                haveRGBMasks = true;
                if (!SDL_ReadU32LE(src, &Rmask) ||
                    !SDL_ReadU32LE(src, &Gmask) ||
                    !SDL_ReadU32LE(src, &Bmask)) {
                    goto done;
                }

                // ...v3 adds an alpha mask.
                if (biSize >= 56) { // BITMAPV3INFOHEADER; adds alpha mask
                    haveAlphaMask = true;
                    if (!SDL_ReadU32LE(src, &Amask)) {
                        goto done;
                    }
                }
            } else {
                // the mask fields are ignored for v2+ headers if not BI_BITFIELD.
                if (biSize >= 52) { // BITMAPV2INFOHEADER; adds RGB masks
                    if (!SDL_ReadU32LE(src, NULL /* Rmask */) ||
                        !SDL_ReadU32LE(src, NULL /* Gmask */) ||
                        !SDL_ReadU32LE(src, NULL /* Bmask */)) {
                        goto done;
                    }
                }
                if (biSize >= 56) { // BITMAPV3INFOHEADER; adds alpha mask
                    if (!SDL_ReadU32LE(src, NULL /* Amask */)) {
                        goto done;
                    }
                }
            }

            /* Insert other fields here; Wikipedia and MSDN say we're up to
               v5 of this header, but we ignore those for now (they add gamma,
               color spaces, etc). Ignoring the weird OS/2 2.x format, we
               currently parse up to v3 correctly (hopefully!). */
        }

        // skip any header bytes we didn't handle...
        headerSize = (Uint32)(SDL_TellIO(src) - (fp_offset + 14));
        if (biSize > headerSize) {
            if (SDL_SeekIO(src, (biSize - headerSize), SDL_IO_SEEK_CUR) < 0) {
                goto done;
            }
        }
    }
    if (biWidth <= 0 || biHeight == 0) {
        SDL_SetError("BMP file with bad dimensions (%" SDL_PRIs32 "x%" SDL_PRIs32 ")", biWidth, biHeight);
        goto done;
    }
    if (biHeight < 0) {
        topDown = true;
        biHeight = -biHeight;
    } else {
        topDown = false;
    }

    // Check for read error
    if (SDL_strcmp(SDL_GetError(), "") != 0) {
        goto done;
    }

    // Reject invalid bit depths
    switch (biBitCount) {
    case 0:
    case 3:
    case 5:
    case 6:
    case 7:
        SDL_SetError("%u bpp BMP images are not supported", biBitCount);
        goto done;
    default:
        break;
    }

    // RLE4 and RLE8 BMP compression is supported
    switch (biCompression) {
    case BI_RGB:
        // If there are no masks, use the defaults
        SDL_assert(!haveRGBMasks);
        SDL_assert(!haveAlphaMask);
        // Default values for the BMP format
        switch (biBitCount) {
        case 15:
        case 16:
            // SDL_PIXELFORMAT_XRGB1555 or SDL_PIXELFORMAT_ARGB1555 if Amask
            Rmask = 0x7C00;
            Gmask = 0x03E0;
            Bmask = 0x001F;
            break;
        case 24:
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
            // SDL_PIXELFORMAT_RGB24
            Rmask = 0x000000FF;
            Gmask = 0x0000FF00;
            Bmask = 0x00FF0000;
#else
            // SDL_PIXELFORMAT_BGR24
            Rmask = 0x00FF0000;
            Gmask = 0x0000FF00;
            Bmask = 0x000000FF;
#endif
            break;
        case 32:
            // We don't know if this has alpha channel or not
            correctAlpha = true;
            // SDL_PIXELFORMAT_RGBA8888
            Amask = 0xFF000000;
            Rmask = 0x00FF0000;
            Gmask = 0x0000FF00;
            Bmask = 0x000000FF;
            break;
        default:
            break;
        }
        break;

    case BI_BITFIELDS:
        break; // we handled this in the info header.

    default:
        break;
    }

    // Create a compatible surface, note that the colors are RGB ordered
    {
        SDL_PixelFormat format;

        // Get the pixel format
        format = SDL_GetPixelFormatForMasks(biBitCount, Rmask, Gmask, Bmask, Amask);
        surface = SDL_CreateSurface(biWidth, biHeight, format);

        if (!surface) {
            goto done;
        }
    }

    // Load the palette, if any
    if (SDL_ISPIXELFORMAT_INDEXED(surface->format)) {
        SDL_Palette *palette = SDL_CreateSurfacePalette(surface);
        if (!palette) {
            goto done;
        }

        if (SDL_SeekIO(src, fp_offset + 14 + biSize, SDL_IO_SEEK_SET) < 0) {
            SDL_SetError("Error seeking in datastream");
            goto done;
        }

        if (biBitCount >= 32) { // we shift biClrUsed by this value later.
            SDL_SetError("Unsupported or incorrect biBitCount field");
            goto done;
        }

        if (biClrUsed == 0) {
            biClrUsed = 1 << biBitCount;
        }

        if (biClrUsed > (Uint32)palette->ncolors) {
            biClrUsed = 1 << biBitCount; // try forcing it?
            if (biClrUsed > (Uint32)palette->ncolors) {
                SDL_SetError("Unsupported or incorrect biClrUsed field");
                goto done;
            }
        }
        palette->ncolors = biClrUsed;

        if (biSize == 12) {
            for (i = 0; i < palette->ncolors; ++i) {
                if (!SDL_ReadU8(src, &palette->colors[i].b) ||
                    !SDL_ReadU8(src, &palette->colors[i].g) ||
                    !SDL_ReadU8(src, &palette->colors[i].r)) {
                    goto done;
                }
                palette->colors[i].a = SDL_ALPHA_OPAQUE;
            }
        } else {
            for (i = 0; i < palette->ncolors; ++i) {
                if (!SDL_ReadU8(src, &palette->colors[i].b) ||
                    !SDL_ReadU8(src, &palette->colors[i].g) ||
                    !SDL_ReadU8(src, &palette->colors[i].r) ||
                    !SDL_ReadU8(src, &palette->colors[i].a)) {
                    goto done;
                }

                /* According to Microsoft documentation, the fourth element
                   is reserved and must be zero, so we shouldn't treat it as
                   alpha.
                */
                palette->colors[i].a = SDL_ALPHA_OPAQUE;
            }
        }
    }

    // Read the surface pixels.  Note that the bmp image is upside down
    if (SDL_SeekIO(src, fp_offset + bfOffBits, SDL_IO_SEEK_SET) < 0) {
        SDL_SetError("Error seeking in datastream");
        goto done;
    }
    if ((biCompression == BI_RLE4) || (biCompression == BI_RLE8)) {
        if (!readRlePixels(surface, src, biCompression == BI_RLE8)) {
            SDL_SetError("Error reading from datastream");
            goto done;
        }

        // Success!
        was_error = false;
        goto done;
    }
    top = (Uint8 *)surface->pixels;
    end = (Uint8 *)surface->pixels + (surface->h * surface->pitch);
    pad = ((surface->pitch % 4) ? (4 - (surface->pitch % 4)) : 0);
    if (topDown) {
        bits = top;
    } else {
        bits = end - surface->pitch;
    }
    while (bits >= top && bits < end) {
        if (SDL_ReadIO(src, bits, surface->pitch) != (size_t)surface->pitch) {
            goto done;
        }
        if (biBitCount == 8 && surface->palette && biClrUsed < (1u << biBitCount)) {
            for (i = 0; i < surface->w; ++i) {
                if (bits[i] >= biClrUsed) {
                    SDL_SetError("A BMP image contains a pixel with a color out of the palette");
                    goto done;
                }
            }
        }
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
        /* Byte-swap the pixels if needed. Note that the 24bpp
           case has already been taken care of above. */
        switch (biBitCount) {
        case 15:
        case 16:
        {
            Uint16 *pix = (Uint16 *)bits;
            for (i = 0; i < surface->w; i++) {
                pix[i] = SDL_Swap16(pix[i]);
            }
            break;
        }

        case 32:
        {
            Uint32 *pix = (Uint32 *)bits;
            for (i = 0; i < surface->w; i++) {
                pix[i] = SDL_Swap32(pix[i]);
            }
            break;
        }
        }
#endif

        // Skip padding bytes, ugh
        if (pad) {
            Uint8 padbyte;
            for (i = 0; i < pad; ++i) {
                if (!SDL_ReadU8(src, &padbyte)) {
                    goto done;
                }
            }
        }
        if (topDown) {
            bits += surface->pitch;
        } else {
            bits -= surface->pitch;
        }
    }
    if (correctAlpha) {
        CorrectAlphaChannel(surface);
    }

    was_error = false;

done:
    if (was_error) {
        if (src) {
            SDL_SeekIO(src, fp_offset, SDL_IO_SEEK_SET);
        }
        SDL_DestroySurface(surface);
        surface = NULL;
    }
    if (closeio && src) {
        SDL_CloseIO(src);
    }
    return surface;
}

SDL_Surface *SDL_LoadBMP(const char *file)
{
    SDL_IOStream *stream = SDL_IOFromFile(file, "rb");
    if (!stream) {
        return NULL;
    }
    return SDL_LoadBMP_IO(stream, true);
}

bool SDL_SaveBMP_IO(SDL_Surface *surface, SDL_IOStream *dst, bool closeio)
{
    bool was_error = true;
    Sint64 fp_offset, new_offset;
    int i, pad;
    SDL_Surface *intermediate_surface = NULL;
    Uint8 *bits;
    bool save32bit = false;
    bool saveLegacyBMP = false;

    // The Win32 BMP file header (14 bytes)
    char magic[2] = { 'B', 'M' };
    Uint32 bfSize;
    Uint16 bfReserved1;
    Uint16 bfReserved2;
    Uint32 bfOffBits;

    // The Win32 BITMAPINFOHEADER struct (40 bytes)
    Uint32 biSize;
    Sint32 biWidth;
    Sint32 biHeight;
    Uint16 biPlanes;
    Uint16 biBitCount;
    Uint32 biCompression;
    Uint32 biSizeImage;
    Sint32 biXPelsPerMeter;
    Sint32 biYPelsPerMeter;
    Uint32 biClrUsed;
    Uint32 biClrImportant;

    // The additional header members from the Win32 BITMAPV4HEADER struct (108 bytes in total)
    Uint32 bV4RedMask = 0;
    Uint32 bV4GreenMask = 0;
    Uint32 bV4BlueMask = 0;
    Uint32 bV4AlphaMask = 0;
    Uint32 bV4CSType = 0;
    Sint32 bV4Endpoints[3 * 3] = { 0 };
    Uint32 bV4GammaRed = 0;
    Uint32 bV4GammaGreen = 0;
    Uint32 bV4GammaBlue = 0;

    // The additional header members from the Win32 BITMAPV5HEADER struct (124 bytes in total)
    Uint32 bV5Intent = 0;
    Uint32 bV5ProfileData = 0;
    Uint32 bV5ProfileSize = 0;
    Uint32 bV5Reserved = 0;

    // Make sure we have somewhere to save
    if (!SDL_SurfaceValid(surface)) {
        SDL_InvalidParamError("surface");
        goto done;
    }
    if (!dst) {
        SDL_InvalidParamError("dst");
        goto done;
    }

#ifdef SAVE_32BIT_BMP
    // We can save alpha information in a 32-bit BMP
    if (SDL_BITSPERPIXEL(surface->format) >= 8 &&
        (SDL_ISPIXELFORMAT_ALPHA(surface->format) ||
         surface->map.info.flags & SDL_COPY_COLORKEY)) {
        save32bit = true;
    }
#endif // SAVE_32BIT_BMP

    if (surface->palette && !save32bit) {
        if (SDL_BITSPERPIXEL(surface->format) == 8) {
            intermediate_surface = surface;
        } else {
            SDL_SetError("%u bpp BMP files not supported",
                         SDL_BITSPERPIXEL(surface->format));
            goto done;
        }
    } else if ((surface->format == SDL_PIXELFORMAT_BGR24 && !save32bit) ||
               (surface->format == SDL_PIXELFORMAT_BGRA32 && save32bit)) {
        intermediate_surface = surface;
    } else {
        SDL_PixelFormat pixel_format;

        /* If the surface has a colorkey or alpha channel we'll save a
           32-bit BMP with alpha channel, otherwise save a 24-bit BMP. */
        if (save32bit) {
            pixel_format = SDL_PIXELFORMAT_BGRA32;
        } else {
            pixel_format = SDL_PIXELFORMAT_BGR24;
        }
        intermediate_surface = SDL_ConvertSurface(surface, pixel_format);
        if (!intermediate_surface) {
            SDL_SetError("Couldn't convert image to %d bpp",
                         (int)SDL_BITSPERPIXEL(pixel_format));
            goto done;
        }
    }

    if (save32bit) {
        saveLegacyBMP = SDL_GetHintBoolean(SDL_HINT_BMP_SAVE_LEGACY_FORMAT, false);
    }

    if (SDL_LockSurface(intermediate_surface)) {
        const size_t bw = intermediate_surface->w * intermediate_surface->fmt->bytes_per_pixel;

        // Set the BMP file header values
        bfSize = 0; // We'll write this when we're done
        bfReserved1 = 0;
        bfReserved2 = 0;
        bfOffBits = 0; // We'll write this when we're done

        // Write the BMP file header values
        fp_offset = SDL_TellIO(dst);
        if (fp_offset < 0) {
            goto done;
        }
        if (SDL_WriteIO(dst, magic, 2) != 2 ||
            !SDL_WriteU32LE(dst, bfSize) ||
            !SDL_WriteU16LE(dst, bfReserved1) ||
            !SDL_WriteU16LE(dst, bfReserved2) ||
            !SDL_WriteU32LE(dst, bfOffBits)) {
            goto done;
        }

        // Set the BMP info values
        biSize = 40;
        biWidth = intermediate_surface->w;
        biHeight = intermediate_surface->h;
        biPlanes = 1;
        biBitCount = intermediate_surface->fmt->bits_per_pixel;
        biCompression = BI_RGB;
        biSizeImage = intermediate_surface->h * intermediate_surface->pitch;
        biXPelsPerMeter = 0;
        biYPelsPerMeter = 0;
        if (intermediate_surface->palette) {
            biClrUsed = intermediate_surface->palette->ncolors;
        } else {
            biClrUsed = 0;
        }
        biClrImportant = 0;

        // Set the BMP info values
        if (save32bit && !saveLegacyBMP) {
            biSize = 124;
            // Version 4 values
            biCompression = BI_BITFIELDS;
            // The BMP format is always little endian, these masks stay the same
            bV4RedMask = 0x00ff0000;
            bV4GreenMask = 0x0000ff00;
            bV4BlueMask = 0x000000ff;
            bV4AlphaMask = 0xff000000;
            bV4CSType = LCS_sRGB;
            bV4GammaRed = 0;
            bV4GammaGreen = 0;
            bV4GammaBlue = 0;
            // Version 5 values
            bV5Intent = LCS_GM_GRAPHICS;
            bV5ProfileData = 0;
            bV5ProfileSize = 0;
            bV5Reserved = 0;
        }

        // Write the BMP info values
        if (!SDL_WriteU32LE(dst, biSize) ||
            !SDL_WriteS32LE(dst, biWidth) ||
            !SDL_WriteS32LE(dst, biHeight) ||
            !SDL_WriteU16LE(dst, biPlanes) ||
            !SDL_WriteU16LE(dst, biBitCount) ||
            !SDL_WriteU32LE(dst, biCompression) ||
            !SDL_WriteU32LE(dst, biSizeImage) ||
            !SDL_WriteU32LE(dst, biXPelsPerMeter) ||
            !SDL_WriteU32LE(dst, biYPelsPerMeter) ||
            !SDL_WriteU32LE(dst, biClrUsed) ||
            !SDL_WriteU32LE(dst, biClrImportant)) {
            goto done;
        }

        // Write the BMP info values
        if (save32bit && !saveLegacyBMP) {
            // Version 4 values
            if (!SDL_WriteU32LE(dst, bV4RedMask) ||
                !SDL_WriteU32LE(dst, bV4GreenMask) ||
                !SDL_WriteU32LE(dst, bV4BlueMask) ||
                !SDL_WriteU32LE(dst, bV4AlphaMask) ||
                !SDL_WriteU32LE(dst, bV4CSType)) {
                goto done;
            }
            for (i = 0; i < 3 * 3; i++) {
                if (!SDL_WriteU32LE(dst, bV4Endpoints[i])) {
                    goto done;
                }
            }
            if (!SDL_WriteU32LE(dst, bV4GammaRed) ||
                !SDL_WriteU32LE(dst, bV4GammaGreen) ||
                !SDL_WriteU32LE(dst, bV4GammaBlue)) {
                goto done;
            }
            // Version 5 values
            if (!SDL_WriteU32LE(dst, bV5Intent) ||
                !SDL_WriteU32LE(dst, bV5ProfileData) ||
                !SDL_WriteU32LE(dst, bV5ProfileSize) ||
                !SDL_WriteU32LE(dst, bV5Reserved)) {
                goto done;
            }
        }

        // Write the palette (in BGR color order)
        if (intermediate_surface->palette) {
            SDL_Color *colors;
            int ncolors;

            colors = intermediate_surface->palette->colors;
            ncolors = intermediate_surface->palette->ncolors;
            for (i = 0; i < ncolors; ++i) {
                if (!SDL_WriteU8(dst, colors[i].b) ||
                    !SDL_WriteU8(dst, colors[i].g) ||
                    !SDL_WriteU8(dst, colors[i].r) ||
                    !SDL_WriteU8(dst, colors[i].a)) {
                    goto done;
                }
            }
        }

        // Write the bitmap offset
        bfOffBits = (Uint32)(SDL_TellIO(dst) - fp_offset);
        if (SDL_SeekIO(dst, fp_offset + 10, SDL_IO_SEEK_SET) < 0) {
            goto done;
        }
        if (!SDL_WriteU32LE(dst, bfOffBits)) {
            goto done;
        }
        if (SDL_SeekIO(dst, fp_offset + bfOffBits, SDL_IO_SEEK_SET) < 0) {
            goto done;
        }

        // Write the bitmap image upside down
        bits = (Uint8 *)intermediate_surface->pixels + (intermediate_surface->h * intermediate_surface->pitch);
        pad = ((bw % 4) ? (4 - (bw % 4)) : 0);
        while (bits > (Uint8 *)intermediate_surface->pixels) {
            bits -= intermediate_surface->pitch;
            if (SDL_WriteIO(dst, bits, bw) != bw) {
                goto done;
            }
            if (pad) {
                const Uint8 padbyte = 0;
                for (i = 0; i < pad; ++i) {
                    if (!SDL_WriteU8(dst, padbyte)) {
                        goto done;
                    }
                }
            }
        }

        // Write the BMP file size
        new_offset = SDL_TellIO(dst);
        if (new_offset < 0) {
            goto done;
        }
        bfSize = (Uint32)(new_offset - fp_offset);
        if (SDL_SeekIO(dst, fp_offset + 2, SDL_IO_SEEK_SET) < 0) {
            goto done;
        }
        if (!SDL_WriteU32LE(dst, bfSize)) {
            goto done;
        }
        if (SDL_SeekIO(dst, fp_offset + bfSize, SDL_IO_SEEK_SET) < 0) {
            goto done;
        }

        // Close it up..
        SDL_UnlockSurface(intermediate_surface);

        was_error = false;
    }

done:
    if (intermediate_surface && intermediate_surface != surface) {
        SDL_DestroySurface(intermediate_surface);
    }
    if (closeio && dst) {
        if (!SDL_CloseIO(dst)) {
            was_error = true;
        }
    }
    if (was_error) {
        return false;
    }
    return true;
}

bool SDL_SaveBMP(SDL_Surface *surface, const char *file)
{
    SDL_IOStream *stream = SDL_IOFromFile(file, "wb");
    if (!stream) {
        return false;
    }
    return SDL_SaveBMP_IO(surface, stream, true);
}

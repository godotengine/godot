/*
 * Copyright (c) 2020-2021 Samsung Electronics Co., Ltd. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*
  LodePNG version 20200306

  Copyright (c) 2005-2020 Lode Vandevenne

  This software is provided 'as-is', without any express or implied
  warranty. In no event will the authors be held liable for any damages
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

      3. This notice may not be removed or altered from any source
      distribution.
*/

#ifndef _TVG_LODEPNG_H_
#define _TVG_LODEPNG_H_

#include <stddef.h>

/*The PNG color types (also used for raw image).*/
enum LodePNGColorType
{
    LCT_GREY = 0, /*grayscale: 1,2,4,8,16 bit*/
    LCT_RGB = 2, /*RGB: 8,16 bit*/
    LCT_PALETTE = 3, /*palette: 1,2,4,8 bit*/
    LCT_GREY_ALPHA = 4, /*grayscale with alpha: 8,16 bit*/
    LCT_RGBA = 6, /*RGB with alpha: 8,16 bit*/
    /*LCT_MAX_OCTET_VALUE lets the compiler allow this enum to represent any invalid
    byte value from 0 to 255 that could be present in an invalid PNG file header. Do
    not use, compare with or set the name LCT_MAX_OCTET_VALUE, instead either use
    the valid color type names above, or numeric values like 1 or 7 when checking for
    particular disallowed color type byte values, or cast to integer to print it.*/
    LCT_MAX_OCTET_VALUE = 255
};

/*Settings for zlib decompression*/
struct LodePNGDecompressSettings
{
    /* Check LodePNGDecoderSettings for more ignorable errors such as ignore_crc */
    unsigned ignore_adler32; /*if 1, continue and don't give an error message if the Adler32 checksum is corrupted*/
    unsigned ignore_nlen; /*ignore complement of len checksum in uncompressed blocks*/

    /*use custom zlib decoder instead of built in one (default: null)*/
    unsigned (*custom_zlib)(unsigned char**, size_t*, const unsigned char*, size_t, const LodePNGDecompressSettings*);
    /*use custom deflate decoder instead of built in one (default: null) if custom_zlib is not null, custom_inflate is ignored (the zlib format uses deflate)*/
    unsigned (*custom_inflate)(unsigned char**, size_t*, const unsigned char*, size_t, const LodePNGDecompressSettings*);

    const void* custom_context; /*optional custom settings for custom functions*/
};

/*
  Color mode of an image. Contains all information required to decode the pixel
  bits to RGBA colors. This information is the same as used in the PNG file
  format, and is used both for PNG and raw image data in LodePNG.
*/
struct LodePNGColorMode
{
    /*header (IHDR)*/
    LodePNGColorType colortype; /*color type, see PNG standard or documentation further in this header file*/
    unsigned bitdepth;  /*bits per sample, see PNG standard or documentation further in this header file*/

    /*
      palette (PLTE and tRNS)

      Dynamically allocated with the colors of the palette, including alpha.
      This field may not be allocated directly, use lodepng_color_mode_init first,
      then lodepng_palette_add per color to correctly initialize it (to ensure size
      of exactly 1024 bytes).

      The alpha channels must be set as well, set them to 255 for opaque images.

      When decoding, by default you can ignore this palette, since LodePNG already
      fills the palette colors in the pixels of the raw RGBA output.

      The palette is only supported for color type 3.
    */
    unsigned char* palette; /*palette in RGBARGBA... order. Must be either 0, or when allocated must have 1024 bytes*/
    size_t palettesize; /*palette size in number of colors (amount of used bytes is 4 * palettesize)*/

    /*
      transparent color key (tRNS)

      This color uses the same bit depth as the bitdepth value in this struct, which can be 1-bit to 16-bit.
      For grayscale PNGs, r, g and b will all 3 be set to the same.

      When decoding, by default you can ignore this information, since LodePNG sets
      pixels with this key to transparent already in the raw RGBA output.

      The color key is only supported for color types 0 and 2.
    */
    unsigned key_defined; /*is a transparent color key given? 0 = false, 1 = true*/
    unsigned key_r;       /*red/grayscale component of color key*/
    unsigned key_g;       /*green component of color key*/
    unsigned key_b;       /*blue component of color key*/
};

/*Information about the PNG image, except pixels, width and height.*/
struct LodePNGInfo
{
    /*header (IHDR), palette (PLTE) and transparency (tRNS) chunks*/
    unsigned compression_method;/*compression method of the original file. Always 0.*/
    unsigned filter_method;     /*filter method of the original file*/
    unsigned interlace_method;  /*interlace method of the original file: 0=none, 1=Adam7*/
    LodePNGColorMode color;     /*color type and bits, palette and transparency of the PNG file*/
};

/*
  Settings for the decoder. This contains settings for the PNG and the Zlib
  decoder, but not the Info settings from the Info structs.
*/
struct LodePNGDecoderSettings
{
    LodePNGDecompressSettings zlibsettings; /*in here is the setting to ignore Adler32 checksums*/

    /* Check LodePNGDecompressSettings for more ignorable errors such as ignore_adler32 */
    unsigned ignore_crc; /*ignore CRC checksums*/
    unsigned ignore_critical; /*ignore unknown critical chunks*/
    unsigned ignore_end; /*ignore issues at end of file if possible (missing IEND chunk, too large chunk, ...)*/
    /* TODO: make a system involving warnings with levels and a strict mode instead. Other potentially recoverable
       errors: srgb rendering intent value, size of content of ancillary chunks, more than 79 characters for some
       strings, placement/combination rules for ancillary chunks, crc of unknown chunks, allowed characters
       in string keys, etc... */

    unsigned color_convert; /*whether to convert the PNG to the color type you want. Default: yes*/
};

/*The settings, state and information for extended encoding and decoding.*/
struct LodePNGState
{
    LodePNGDecoderSettings decoder; /*the decoding settings*/
    LodePNGColorMode info_raw; /*specifies the format in which you would like to get the raw pixel buffer*/
    LodePNGInfo info_png; /*info of the PNG image obtained after decoding*/
    unsigned error;
};

void lodepng_state_init(LodePNGState* state);
void lodepng_state_cleanup(LodePNGState* state);
unsigned lodepng_decode(unsigned char** out, unsigned* w, unsigned* h, LodePNGState* state, const unsigned char* in, size_t insize);
unsigned lodepng_inspect(unsigned* w, unsigned* h, LodePNGState* state, const unsigned char* in, size_t insize);

#endif //_TVG_LODEPNG_H_
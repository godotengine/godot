/****************************************************************************
 *
 * emj1shim.c
 *
 * Apple EMJC decoder, based on https://github.com/PoomSmart/EmojiFonts/blob/main/emjc.py
 */


#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/tttags.h>
#include FT_CONFIG_STANDARD_LIBRARY_H


#if defined( TT_CONFIG_OPTION_EMBEDDED_BITMAPS ) && \
    defined( FT_CONFIG_OPTION_USE_EMJC )

#include <lzfse.h>
#include "emj1shim.h"

#include "sferrors.h"


  FT_Int32
  convertToDiff( FT_Int32 value,
                 FT_Int32 offset )
  {
    return (value & 1) ? (-(value >> 1) - offset) : ((value >> 1) + offset);
  }


  FT_Int32
  filterValue( FT_Int32 left,
               FT_Int32 upper )
  {
    FT_Int32 value = left + upper + 1;
    return (value < 0) ? -((-value) / 2) : (value / 2);
  }


  FT_Int32
  multiplyAlpha( FT_Int32  alpha,
                 FT_Int32  color )
  {
    FT_Int32  temp = alpha * color + 0x80;


    return ( temp + ( temp >> 8 ) ) >> 8;
  }


  FT_LOCAL_DEF( FT_Error )
  Load_SBit_Emj1( FT_GlyphSlot     slot,
                 FT_Int           x_offset,
                 FT_Int           y_offset,
                 FT_Int           pix_bits,
                 TT_SBit_Metrics  metrics,
                 FT_Memory        memory,
                 FT_Byte*         data,
                 FT_UInt          data_len,
                 FT_Bool          populate_map_and_metrics,
                 FT_Bool          metrics_only )
  {
    FT_Bitmap    *map   = &slot->bitmap;
    FT_Error      error = FT_Err_Ok;

    if ( x_offset < 0 ||
         y_offset < 0 )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( !populate_map_and_metrics                            &&
         ( (FT_UInt)x_offset + metrics->width  > map->width ||
           (FT_UInt)y_offset + metrics->height > map->rows  ||
           pix_bits != 32                                   ||
           map->pixel_mode != FT_PIXEL_MODE_BGRA            ) )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( data[0] != 'e' || data[1] != 'm' || data[2] != 'j' || data[3] != '1' ) {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    //FT_UInt version = (FT_UInt)data[4] | ((FT_UInt)data[5] << 8); // Always 0x0100
    //FT_UInt ??? = (FT_UInt)data[6] | ((FT_UInt)data[7] << 8); // Always 0xa101
    FT_UInt imgWidth = (FT_UInt)data[8] | ((FT_UInt)data[9] << 8);
    FT_UInt imgHeight = (FT_UInt)data[10] | ((FT_UInt)data[11] << 8);
    FT_UInt appendixLength = (FT_UInt)data[12] | ((FT_UInt)data[13] << 8);
    //FT_UInt ??? = (FT_UInt)data[14] | ((FT_UInt)data[15] << 8); // Always 0x0000

    FT_UInt filterLength = imgHeight;
    FT_UInt pixels = imgHeight * imgWidth;
    FT_UInt colors = pixels * 3;
    FT_ULong dstLength = pixels + filterLength + pixels * 3 + appendixLength;

    if ( error                                        ||
         ( !populate_map_and_metrics                &&
           ( (FT_Int)imgWidth  != metrics->width  ||
             (FT_Int)imgHeight != metrics->height ) ) )
      goto Exit;

    if ( populate_map_and_metrics )
    {
      /* reject too large bitmaps similarly to the rasterizer */
      if ( imgHeight > 0x7FFF || imgWidth > 0x7FFF )
      {
        error = FT_THROW( Array_Too_Large );
        goto Exit;
      }

      metrics->width  = (FT_UShort)imgWidth;
      metrics->height = (FT_UShort)imgHeight;

      map->width      = metrics->width;
      map->rows       = metrics->height;
      map->pixel_mode = FT_PIXEL_MODE_BGRA;
      map->pitch      = (int)( map->width * 4 );
      map->num_grays  = 256;
    }

    if ( metrics_only )
      goto Exit;


    if ( populate_map_and_metrics )
    {
      /* this doesn't overflow: 0x7FFF * 0x7FFF * 4 < 2^32 */
      FT_ULong  size = map->rows * (FT_ULong)map->pitch;


      error = ft_glyphslot_alloc_bitmap( slot, size );
      if ( error )
        goto Exit;
    }

    FT_Byte* decompressedData = NULL;
    if ( FT_QNEW_ARRAY( decompressedData, dstLength ) ) {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    FT_ULong decLength = lzfse_decode_buffer( decompressedData, dstLength, data + 16, data_len - 16, NULL );
    if ( dstLength != decLength ) {
      FT_FREE( decompressedData );
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    FT_Byte* alpha = decompressedData;
    FT_Byte* filters = decompressedData + pixels;
    FT_Byte* rgb = decompressedData + pixels + filterLength;
    FT_Byte* appendix = decompressedData + pixels + filterLength + colors;
    FT_Int32* colorData = NULL;
    if ( FT_QNEW_ARRAY( colorData, colors * sizeof(FT_Int32) ) ) {
      FT_FREE( decompressedData );
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }
    FT_MEM_ZERO( colorData, colors * sizeof(FT_Int32) );

    FT_Int offset = 0;
    for (FT_UInt i = 0; i < appendixLength; i++) {
        offset += appendix[i] / 4;
        if (offset >= colors) {
            break;
        }
        colorData[offset] = 128 * (appendix[i] % 4);
        offset += 1;
    }

    FT_UInt buffOffset = y_offset * map->pitch + x_offset * 4;
    FT_UInt i = 0;
    for (FT_UInt y = 0; y < imgHeight; y++) {
        FT_UInt filter = filters[y];
        for (FT_UInt x = 0; x < imgWidth; x++) {
            colorData[i * 3 + 0] = convertToDiff(rgb[i * 3 + 0], colorData[i * 3 + 0]);
            colorData[i * 3 + 1] = convertToDiff(rgb[i * 3 + 1], colorData[i * 3 + 1]);
            colorData[i * 3 + 2] = convertToDiff(rgb[i * 3 + 2], colorData[i * 3 + 2]);
            if (filter == 1) {
                if (x > 0 && y > 0) {
                    FT_Int32 left = colorData[(i - 1) * 3 + 0];
                    FT_Int32 upper = colorData[(i - imgWidth) * 3 + 0];
                    FT_Int32 leftUpper = colorData[(i - imgWidth - 1) * 3 + 0];
                    if (abs(left - leftUpper) < abs(upper - leftUpper)) {
                        colorData[i * 3 + 0] += upper;
                        colorData[i * 3 + 1] += colorData[(i - imgWidth) * 3 + 1];
                        colorData[i * 3 + 2] += colorData[(i - imgWidth) * 3 + 2];
                    } else {
                        colorData[i * 3 + 0] += left;
                        colorData[i * 3 + 1] += colorData[(i - 1) * 3 + 1];
                        colorData[i * 3 + 2] += colorData[(i - 1) * 3 + 2];
                    }
                } else if (x > 0) {
                    colorData[i * 3 + 0] += colorData[(i - 1) * 3 + 0];
                    colorData[i * 3 + 1] += colorData[(i - 1) * 3 + 1];
                    colorData[i * 3 + 2] += colorData[(i - 1) * 3 + 2];
                } else if (y > 0) {
                    colorData[i * 3 + 0] += colorData[(i - imgWidth) * 3 + 0];
                    colorData[i * 3 + 1] += colorData[(i - imgWidth) * 3 + 1];
                    colorData[i * 3 + 2] += colorData[(i - imgWidth) * 3 + 2];
                }
            } else if (filter == 2) {
                if (x > 0) {
                    colorData[i * 3 + 0] += colorData[(i - 1) * 3 + 0];
                    colorData[i * 3 + 1] += colorData[(i - 1) * 3 + 1];
                    colorData[i * 3 + 2] += colorData[(i - 1) * 3 + 2];
                }
            } else if (filter == 3) {
                if (y > 0) {
                    colorData[i * 3 + 0] += colorData[(i - imgWidth) * 3 + 0];
                    colorData[i * 3 + 1] += colorData[(i - imgWidth) * 3 + 1];
                    colorData[i * 3 + 2] += colorData[(i - imgWidth) * 3 + 2];
                }
            } else if (filter == 4) {
                if (x > 0 && y > 0) {
                    colorData[i * 3 + 0] += filterValue(colorData[(i - 1) * 3 + 0], colorData[(i - imgWidth) * 3 + 0]);
                    colorData[i * 3 + 1] += filterValue(colorData[(i - 1) * 3 + 1], colorData[(i - imgWidth) * 3 + 1]);
                    colorData[i * 3 + 2] += filterValue(colorData[(i - 1) * 3 + 2], colorData[(i - imgWidth) * 3 + 2]);
                } else if (x > 0) {
                    colorData[i * 3 + 0] += colorData[(i - 1) * 3 + 0];
                    colorData[i * 3 + 1] += colorData[(i - 1) * 3 + 1];
                    colorData[i * 3 + 2] += colorData[(i - 1) * 3 + 2];
                } else if (y > 0) {
                    colorData[i * 3 + 0] += colorData[(i - imgWidth) * 3 + 0];
                    colorData[i * 3 + 1] += colorData[(i - imgWidth) * 3 + 1];
                    colorData[i * 3 + 2] += colorData[(i - imgWidth) * 3 + 2];
                }
            }

            FT_Int32 base = colorData[i * 3 + 0];
            FT_Int32 p = colorData[i * 3 + 1];
            FT_Int32 q = colorData[i * 3 + 2];
            FT_Int32 r = 0;
            FT_Int32 g = 0;
            FT_Int32 b = 0;
            if (p < 0 && q < 0) {
                r = base + p / 2 - (q + 1) / 2;
                g = base + q / 2;
                b = base - (p + 1) / 2 - (q + 1) / 2;
            } else if (p < 0) {
                r = base + p / 2 - q / 2;
                g = base + (q + 1) / 2;
                b = base - (p + 1) / 2 - q / 2;
            } else if (q < 0) {
                r = base + (p + 1) / 2 - (q + 1) / 2;
                g = base + q / 2;
                b = base - p / 2 - (q + 1) / 2;
            } else {
                r = base + (p + 1) / 2 - q / 2;
                g = base + (q + 1) / 2;
                b = base - p / 2 - q / 2;
            }

            map->buffer[buffOffset + i * 4 + 0] = multiplyAlpha( alpha[i], (b < 0) ? ((b % 257) + 257) : (b % 257));
            map->buffer[buffOffset + i * 4 + 1] = multiplyAlpha( alpha[i], (g < 0) ? ((g % 257) + 257) : (g % 257));
            map->buffer[buffOffset + i * 4 + 2] = multiplyAlpha( alpha[i], (r < 0) ? ((r % 257) + 257) : (r % 257));
            map->buffer[buffOffset + i * 4 + 3] = alpha[i];
            i++;
        }
    }

    FT_FREE( colorData );
    FT_FREE( decompressedData );

  Exit:
    return error;
  }

#else /* !(TT_CONFIG_OPTION_EMBEDDED_BITMAPS && FT_CONFIG_OPTION_USE_EMJC) */

  /* ANSI C doesn't like empty source files */
  typedef int  _emj1shim_dummy;

#endif /* !(TT_CONFIG_OPTION_EMBEDDED_BITMAPS && FT_CONFIG_OPTION_USE_EMJC) */


/* END */

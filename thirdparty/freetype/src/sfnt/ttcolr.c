/****************************************************************************
 *
 * ttcolr.c
 *
 *   TrueType and OpenType colored glyph layer support (body).
 *
 * Copyright (C) 2018-2019 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * Originally written by Shao Yu Zhang <shaozhang@fb.com>.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


  /**************************************************************************
   *
   * `COLR' table specification:
   *
   *   https://www.microsoft.com/typography/otspec/colr.htm
   *
   */


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_STREAM_H
#include FT_TRUETYPE_TAGS_H
#include FT_COLOR_H


#ifdef TT_CONFIG_OPTION_COLOR_LAYERS

#include "ttcolr.h"


  /* NOTE: These are the table sizes calculated through the specs. */
#define BASE_GLYPH_SIZE            6
#define LAYER_SIZE                 4
#define COLR_HEADER_SIZE          14


  typedef struct BaseGlyphRecord_
  {
    FT_UShort  gid;
    FT_UShort  first_layer_index;
    FT_UShort  num_layers;

  } BaseGlyphRecord;


  typedef struct Colr_
  {
    FT_UShort  version;
    FT_UShort  num_base_glyphs;
    FT_UShort  num_layers;

    FT_Byte*  base_glyphs;
    FT_Byte*  layers;

    /* The memory which backs up the `COLR' table. */
    void*     table;
    FT_ULong  table_size;

  } Colr;


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttcolr


  FT_LOCAL_DEF( FT_Error )
  tt_face_load_colr( TT_Face    face,
                     FT_Stream  stream )
  {
    FT_Error   error;
    FT_Memory  memory = face->root.memory;

    FT_Byte*  table = NULL;
    FT_Byte*  p     = NULL;

    Colr*  colr = NULL;

    FT_ULong  base_glyph_offset, layer_offset;
    FT_ULong  table_size;


    /* `COLR' always needs `CPAL' */
    if ( !face->cpal )
      return FT_THROW( Invalid_File_Format );

    error = face->goto_table( face, TTAG_COLR, stream, &table_size );
    if ( error )
      goto NoColr;

    if ( table_size < COLR_HEADER_SIZE )
      goto InvalidTable;

    if ( FT_FRAME_EXTRACT( table_size, table ) )
      goto NoColr;

    p = table;

    if ( FT_NEW( colr ) )
      goto NoColr;

    colr->version = FT_NEXT_USHORT( p );
    if ( colr->version != 0 )
      goto InvalidTable;

    colr->num_base_glyphs = FT_NEXT_USHORT( p );
    base_glyph_offset     = FT_NEXT_ULONG( p );

    if ( base_glyph_offset >= table_size )
      goto InvalidTable;
    if ( colr->num_base_glyphs * BASE_GLYPH_SIZE >
           table_size - base_glyph_offset )
      goto InvalidTable;

    layer_offset     = FT_NEXT_ULONG( p );
    colr->num_layers = FT_NEXT_USHORT( p );

    if ( layer_offset >= table_size )
      goto InvalidTable;
    if ( colr->num_layers * LAYER_SIZE > table_size - layer_offset )
      goto InvalidTable;

    colr->base_glyphs = (FT_Byte*)( table + base_glyph_offset );
    colr->layers      = (FT_Byte*)( table + layer_offset      );
    colr->table       = table;
    colr->table_size  = table_size;

    face->colr = colr;

    return FT_Err_Ok;

  InvalidTable:
    error = FT_THROW( Invalid_Table );

  NoColr:
    FT_FRAME_RELEASE( table );
    FT_FREE( colr );

    return error;
  }


  FT_LOCAL_DEF( void )
  tt_face_free_colr( TT_Face  face )
  {
    FT_Stream  stream = face->root.stream;
    FT_Memory  memory = face->root.memory;

    Colr*  colr = (Colr*)face->colr;


    if ( colr )
    {
      FT_FRAME_RELEASE( colr->table );
      FT_FREE( colr );
    }
  }


  static FT_Bool
  find_base_glyph_record( FT_Byte*          base_glyph_begin,
                          FT_Int            num_base_glyph,
                          FT_UInt           glyph_id,
                          BaseGlyphRecord*  record )
  {
    FT_Int  min = 0;
    FT_Int  max = num_base_glyph - 1;


    while ( min <= max )
    {
      FT_Int    mid = min + ( max - min ) / 2;
      FT_Byte*  p   = base_glyph_begin + mid * BASE_GLYPH_SIZE;

      FT_UShort  gid = FT_NEXT_USHORT( p );


      if ( gid < glyph_id )
        min = mid + 1;
      else if (gid > glyph_id )
        max = mid - 1;
      else
      {
        record->gid               = gid;
        record->first_layer_index = FT_NEXT_USHORT( p );
        record->num_layers        = FT_NEXT_USHORT( p );

        return 1;
      }
    }

    return 0;
  }


  FT_LOCAL_DEF( FT_Bool )
  tt_face_get_colr_layer( TT_Face            face,
                          FT_UInt            base_glyph,
                          FT_UInt           *aglyph_index,
                          FT_UInt           *acolor_index,
                          FT_LayerIterator*  iterator )
  {
    Colr*            colr = (Colr*)face->colr;
    BaseGlyphRecord  glyph_record;


    if ( !colr )
      return 0;

    if ( !iterator->p )
    {
      FT_ULong  offset;


      /* first call to function */
      iterator->layer = 0;

      if ( !find_base_glyph_record( colr->base_glyphs,
                                    colr->num_base_glyphs,
                                    base_glyph,
                                    &glyph_record ) )
        return 0;

      if ( glyph_record.num_layers )
        iterator->num_layers = glyph_record.num_layers;
      else
        return 0;

      offset = LAYER_SIZE * glyph_record.first_layer_index;
      if ( offset + LAYER_SIZE * glyph_record.num_layers > colr->table_size )
        return 0;

      iterator->p = colr->layers + offset;
    }

    if ( iterator->layer >= iterator->num_layers )
      return 0;

    *aglyph_index = FT_NEXT_USHORT( iterator->p );
    *acolor_index = FT_NEXT_USHORT( iterator->p );

    if ( *aglyph_index >= (FT_UInt)( FT_FACE( face )->num_glyphs )   ||
         ( *acolor_index != 0xFFFF                                 &&
           *acolor_index >= face->palette_data.num_palette_entries ) )
      return 0;

    iterator->layer++;

    return 1;
  }


  FT_LOCAL_DEF( FT_Error )
  tt_face_colr_blend_layer( TT_Face       face,
                            FT_UInt       color_index,
                            FT_GlyphSlot  dstSlot,
                            FT_GlyphSlot  srcSlot )
  {
    FT_Error  error;

    FT_UInt  x, y;
    FT_Byte  b, g, r, alpha;

    FT_ULong  size;
    FT_Byte*  src;
    FT_Byte*  dst;


    if ( !dstSlot->bitmap.buffer )
    {
      /* Initialize destination of color bitmap */
      /* with the size of first component.      */
      dstSlot->bitmap_left = srcSlot->bitmap_left;
      dstSlot->bitmap_top  = srcSlot->bitmap_top;

      dstSlot->bitmap.width      = srcSlot->bitmap.width;
      dstSlot->bitmap.rows       = srcSlot->bitmap.rows;
      dstSlot->bitmap.pixel_mode = FT_PIXEL_MODE_BGRA;
      dstSlot->bitmap.pitch      = (int)dstSlot->bitmap.width * 4;
      dstSlot->bitmap.num_grays  = 256;

      size = dstSlot->bitmap.rows * (unsigned int)dstSlot->bitmap.pitch;

      error = ft_glyphslot_alloc_bitmap( dstSlot, size );
      if ( error )
        return error;

      FT_MEM_ZERO( dstSlot->bitmap.buffer, size );
    }
    else
    {
      /* Resize destination if needed such that new component fits. */
      FT_Int  x_min, x_max, y_min, y_max;


      x_min = FT_MIN( dstSlot->bitmap_left, srcSlot->bitmap_left );
      x_max = FT_MAX( dstSlot->bitmap_left + (FT_Int)dstSlot->bitmap.width,
                      srcSlot->bitmap_left + (FT_Int)srcSlot->bitmap.width );

      y_min = FT_MIN( dstSlot->bitmap_top - (FT_Int)dstSlot->bitmap.rows,
                      srcSlot->bitmap_top - (FT_Int)srcSlot->bitmap.rows );
      y_max = FT_MAX( dstSlot->bitmap_top, srcSlot->bitmap_top );

      if ( x_min != dstSlot->bitmap_left                                 ||
           x_max != dstSlot->bitmap_left + (FT_Int)dstSlot->bitmap.width ||
           y_min != dstSlot->bitmap_top - (FT_Int)dstSlot->bitmap.rows   ||
           y_max != dstSlot->bitmap_top                                  )
      {
        FT_Memory  memory = face->root.memory;

        FT_UInt  width = (FT_UInt)( x_max - x_min );
        FT_UInt  rows  = (FT_UInt)( y_max - y_min );
        FT_UInt  pitch = width * 4;

        FT_Byte*  buf = NULL;
        FT_Byte*  p;
        FT_Byte*  q;


        size  = rows * pitch;
        if ( FT_ALLOC( buf, size ) )
          return error;

        p = dstSlot->bitmap.buffer;
        q = buf +
            (int)pitch * ( y_max - dstSlot->bitmap_top ) +
            4 * ( dstSlot->bitmap_left - x_min );

        for ( y = 0; y < dstSlot->bitmap.rows; y++ )
        {
          FT_MEM_COPY( q, p, dstSlot->bitmap.width * 4 );

          p += dstSlot->bitmap.pitch;
          q += pitch;
        }

        ft_glyphslot_set_bitmap( dstSlot, buf );

        dstSlot->bitmap_top  = y_max;
        dstSlot->bitmap_left = x_min;

        dstSlot->bitmap.width = width;
        dstSlot->bitmap.rows  = rows;
        dstSlot->bitmap.pitch = (int)pitch;

        dstSlot->internal->flags |= FT_GLYPH_OWN_BITMAP;
        dstSlot->format           = FT_GLYPH_FORMAT_BITMAP;
      }
    }

    if ( color_index == 0xFFFF )
    {
      if ( face->have_foreground_color )
      {
        b     = face->foreground_color.blue;
        g     = face->foreground_color.green;
        r     = face->foreground_color.red;
        alpha = face->foreground_color.alpha;
      }
      else
      {
        if ( face->palette_data.palette_flags                          &&
             ( face->palette_data.palette_flags[face->palette_index] &
                 FT_PALETTE_FOR_DARK_BACKGROUND                      ) )
        {
          /* white opaque */
          b     = 0xFF;
          g     = 0xFF;
          r     = 0xFF;
          alpha = 0xFF;
        }
        else
        {
          /* black opaque */
          b     = 0x00;
          g     = 0x00;
          r     = 0x00;
          alpha = 0xFF;
        }
      }
    }
    else
    {
      b     = face->palette[color_index].blue;
      g     = face->palette[color_index].green;
      r     = face->palette[color_index].red;
      alpha = face->palette[color_index].alpha;
    }

    /* XXX Convert if srcSlot.bitmap is not grey? */
    src = srcSlot->bitmap.buffer;
    dst = dstSlot->bitmap.buffer +
          dstSlot->bitmap.pitch * ( dstSlot->bitmap_top - srcSlot->bitmap_top ) +
          4 * ( srcSlot->bitmap_left - dstSlot->bitmap_left );

    for ( y = 0; y < srcSlot->bitmap.rows; y++ )
    {
      for ( x = 0; x < srcSlot->bitmap.width; x++ )
      {
        int  aa = src[x];
        int  fa = alpha * aa / 255;

        int  fb = b * fa / 255;
        int  fg = g * fa / 255;
        int  fr = r * fa / 255;

        int  ba2 = 255 - fa;

        int  bb = dst[4 * x + 0];
        int  bg = dst[4 * x + 1];
        int  br = dst[4 * x + 2];
        int  ba = dst[4 * x + 3];


        dst[4 * x + 0] = (FT_Byte)( bb * ba2 / 255 + fb );
        dst[4 * x + 1] = (FT_Byte)( bg * ba2 / 255 + fg );
        dst[4 * x + 2] = (FT_Byte)( br * ba2 / 255 + fr );
        dst[4 * x + 3] = (FT_Byte)( ba * ba2 / 255 + fa );
      }

      src += srcSlot->bitmap.pitch;
      dst += dstSlot->bitmap.pitch;
    }

    return FT_Err_Ok;
  }

#else /* !TT_CONFIG_OPTION_COLOR_LAYERS */

  /* ANSI C doesn't like empty source files */
  typedef int  _tt_colr_dummy;

#endif /* !TT_CONFIG_OPTION_COLOR_LAYERS */

/* EOF */

/****************************************************************************
 *
 * ttcpal.c
 *
 *   TrueType and OpenType color palette support (body).
 *
 * Copyright (C) 2018-2022 by
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
   * `CPAL' table specification:
   *
   *   https://www.microsoft.com/typography/otspec/cpal.htm
   *
   */


#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/tttags.h>
#include <freetype/ftcolor.h>


#ifdef TT_CONFIG_OPTION_COLOR_LAYERS

#include "ttcpal.h"


  /* NOTE: These are the table sizes calculated through the specs. */
#define CPAL_V0_HEADER_BASE_SIZE  12U
#define COLOR_SIZE                 4U


  /* all data from `CPAL' not covered in FT_Palette_Data */
  typedef struct Cpal_
  {
    FT_UShort  version;        /* Table version number (0 or 1 supported). */
    FT_UShort  num_colors;               /* Total number of color records, */
                                         /* combined for all palettes.     */
    FT_Byte*  colors;                              /* RGBA array of colors */
    FT_Byte*  color_indices; /* Index of each palette's first color record */
                             /* in the combined color record array.        */

    /* The memory which backs up the `CPAL' table. */
    void*     table;
    FT_ULong  table_size;

  } Cpal;


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttcpal


  FT_LOCAL_DEF( FT_Error )
  tt_face_load_cpal( TT_Face    face,
                     FT_Stream  stream )
  {
    FT_Error   error;
    FT_Memory  memory = face->root.memory;

    FT_Byte*  table = NULL;
    FT_Byte*  p     = NULL;

    Cpal*  cpal = NULL;

    FT_ULong  colors_offset;
    FT_ULong  table_size;


    error = face->goto_table( face, TTAG_CPAL, stream, &table_size );
    if ( error )
      goto NoCpal;

    if ( table_size < CPAL_V0_HEADER_BASE_SIZE )
      goto InvalidTable;

    if ( FT_FRAME_EXTRACT( table_size, table ) )
      goto NoCpal;

    p = table;

    if ( FT_NEW( cpal ) )
      goto NoCpal;

    cpal->version = FT_NEXT_USHORT( p );
    if ( cpal->version > 1 )
      goto InvalidTable;

    face->palette_data.num_palette_entries = FT_NEXT_USHORT( p );
    face->palette_data.num_palettes        = FT_NEXT_USHORT( p );

    cpal->num_colors = FT_NEXT_USHORT( p );
    colors_offset    = FT_NEXT_ULONG( p );

    if ( CPAL_V0_HEADER_BASE_SIZE             +
         face->palette_data.num_palettes * 2U > table_size )
      goto InvalidTable;

    if ( colors_offset >= table_size )
      goto InvalidTable;
    if ( cpal->num_colors * COLOR_SIZE > table_size - colors_offset )
      goto InvalidTable;

    if ( face->palette_data.num_palette_entries > cpal->num_colors )
      goto InvalidTable;

    cpal->color_indices = p;
    cpal->colors        = (FT_Byte*)( table + colors_offset );

    if ( cpal->version == 1 )
    {
      FT_ULong    type_offset, label_offset, entry_label_offset;
      FT_UShort*  array = NULL;
      FT_UShort*  limit;
      FT_UShort*  q;


      if ( CPAL_V0_HEADER_BASE_SIZE             +
           face->palette_data.num_palettes * 2U +
           3U * 4                               > table_size )
        goto InvalidTable;

      p += face->palette_data.num_palettes * 2U;

      type_offset        = FT_NEXT_ULONG( p );
      label_offset       = FT_NEXT_ULONG( p );
      entry_label_offset = FT_NEXT_ULONG( p );

      if ( type_offset )
      {
        if ( type_offset >= table_size )
          goto InvalidTable;
        if ( face->palette_data.num_palettes * 2U >
               table_size - type_offset )
          goto InvalidTable;

        if ( FT_QNEW_ARRAY( array, face->palette_data.num_palettes ) )
          goto NoCpal;

        p     = table + type_offset;
        q     = array;
        limit = q + face->palette_data.num_palettes;

        while ( q < limit )
          *q++ = FT_NEXT_USHORT( p );

        face->palette_data.palette_flags = array;
      }

      if ( label_offset )
      {
        if ( label_offset >= table_size )
          goto InvalidTable;
        if ( face->palette_data.num_palettes * 2U >
               table_size - label_offset )
          goto InvalidTable;

        if ( FT_QNEW_ARRAY( array, face->palette_data.num_palettes ) )
          goto NoCpal;

        p     = table + label_offset;
        q     = array;
        limit = q + face->palette_data.num_palettes;

        while ( q < limit )
          *q++ = FT_NEXT_USHORT( p );

        face->palette_data.palette_name_ids = array;
      }

      if ( entry_label_offset )
      {
        if ( entry_label_offset >= table_size )
          goto InvalidTable;
        if ( face->palette_data.num_palette_entries * 2U >
               table_size - entry_label_offset )
          goto InvalidTable;

        if ( FT_QNEW_ARRAY( array, face->palette_data.num_palette_entries ) )
          goto NoCpal;

        p     = table + entry_label_offset;
        q     = array;
        limit = q + face->palette_data.num_palette_entries;

        while ( q < limit )
          *q++ = FT_NEXT_USHORT( p );

        face->palette_data.palette_entry_name_ids = array;
      }
    }

    cpal->table      = table;
    cpal->table_size = table_size;

    face->cpal = cpal;

    /* set up default palette */
    if ( FT_NEW_ARRAY( face->palette,
                       face->palette_data.num_palette_entries ) )
      goto NoCpal;

    if ( tt_face_palette_set( face, 0 ) )
      goto InvalidTable;

    return FT_Err_Ok;

  InvalidTable:
    error = FT_THROW( Invalid_Table );

  NoCpal:
    FT_FRAME_RELEASE( table );
    FT_FREE( cpal );

    face->cpal = NULL;

    /* arrays in `face->palette_data' and `face->palette' */
    /* are freed in `sfnt_done_face'                      */

    return error;
  }


  FT_LOCAL_DEF( void )
  tt_face_free_cpal( TT_Face  face )
  {
    FT_Stream  stream = face->root.stream;
    FT_Memory  memory = face->root.memory;

    Cpal*  cpal = (Cpal*)face->cpal;


    if ( cpal )
    {
      FT_FRAME_RELEASE( cpal->table );
      FT_FREE( cpal );
    }
  }


  FT_LOCAL_DEF( FT_Error )
  tt_face_palette_set( TT_Face  face,
                       FT_UInt  palette_index )
  {
    Cpal*  cpal = (Cpal*)face->cpal;

    FT_Byte*   offset;
    FT_Byte*   p;

    FT_Color*  q;
    FT_Color*  limit;

    FT_UShort  color_index;


    if ( !cpal || palette_index >= face->palette_data.num_palettes )
      return FT_THROW( Invalid_Argument );

    offset      = cpal->color_indices + 2 * palette_index;
    color_index = FT_PEEK_USHORT( offset );

    if ( color_index + face->palette_data.num_palette_entries >
           cpal->num_colors )
      return FT_THROW( Invalid_Table );

    p     = cpal->colors + COLOR_SIZE * color_index;
    q     = face->palette;
    limit = q + face->palette_data.num_palette_entries;

    while ( q < limit )
    {
      q->blue  = FT_NEXT_BYTE( p );
      q->green = FT_NEXT_BYTE( p );
      q->red   = FT_NEXT_BYTE( p );
      q->alpha = FT_NEXT_BYTE( p );

      q++;
    }

    return FT_Err_Ok;
  }


#else /* !TT_CONFIG_OPTION_COLOR_LAYERS */

  /* ANSI C doesn't like empty source files */
  typedef int  _tt_cpal_dummy;

#endif /* !TT_CONFIG_OPTION_COLOR_LAYERS */

/* EOF */

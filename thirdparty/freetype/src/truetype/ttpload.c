/****************************************************************************
 *
 * ttpload.c
 *
 *   TrueType-specific tables loader (body).
 *
 * Copyright (C) 1996-2019 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_STREAM_H
#include FT_TRUETYPE_TAGS_H

#include "ttpload.h"

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include "ttgxvar.h"
#endif

#include "tterrors.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttpload


  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_loca
   *
   * @Description:
   *   Load the locations table.
   *
   * @InOut:
   *   face ::
   *     A handle to the target face object.
   *
   * @Input:
   *   stream ::
   *     The input stream.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_load_loca( TT_Face    face,
                     FT_Stream  stream )
  {
    FT_Error  error;
    FT_ULong  table_len;
    FT_Int    shift;


    /* we need the size of the `glyf' table for malformed `loca' tables */
    error = face->goto_table( face, TTAG_glyf, stream, &face->glyf_len );

    /* it is possible that a font doesn't have a glyf table at all */
    /* or its size is zero                                         */
    if ( FT_ERR_EQ( error, Table_Missing ) )
    {
      face->glyf_len    = 0;
      face->glyf_offset = 0;
    }
    else if ( error )
      goto Exit;
    else
    {
#ifdef FT_CONFIG_OPTION_INCREMENTAL
      if ( face->root.internal->incremental_interface )
        face->glyf_offset = 0;
      else
#endif
        face->glyf_offset = FT_STREAM_POS();
    }

    FT_TRACE2(( "Locations " ));
    error = face->goto_table( face, TTAG_loca, stream, &table_len );
    if ( error )
    {
      error = FT_THROW( Locations_Missing );
      goto Exit;
    }

    if ( face->header.Index_To_Loc_Format != 0 )
    {
      shift = 2;

      if ( table_len >= 0x40000L )
      {
        FT_TRACE2(( "table too large\n" ));
        table_len = 0x3FFFFL;
      }
      face->num_locations = table_len >> shift;
    }
    else
    {
      shift = 1;

      if ( table_len >= 0x20000L )
      {
        FT_TRACE2(( "table too large\n" ));
        table_len = 0x1FFFFL;
      }
      face->num_locations = table_len >> shift;
    }

    if ( face->num_locations != (FT_ULong)face->root.num_glyphs + 1 )
    {
      FT_TRACE2(( "glyph count mismatch!  loca: %d, maxp: %d\n",
                  face->num_locations - 1, face->root.num_glyphs ));

      /* we only handle the case where `maxp' gives a larger value */
      if ( face->num_locations <= (FT_ULong)face->root.num_glyphs )
      {
        FT_ULong  new_loca_len =
                    ( (FT_ULong)face->root.num_glyphs + 1 ) << shift;

        TT_Table  entry = face->dir_tables;
        TT_Table  limit = entry + face->num_tables;

        FT_Long  pos   = (FT_Long)FT_STREAM_POS();
        FT_Long  dist  = 0x7FFFFFFFL;
        FT_Bool  found = 0;


        /* compute the distance to next table in font file */
        for ( ; entry < limit; entry++ )
        {
          FT_Long  diff = (FT_Long)entry->Offset - pos;


          if ( diff > 0 && diff < dist )
          {
            dist  = diff;
            found = 1;
          }
        }

        if ( !found )
        {
          /* `loca' is the last table */
          dist = (FT_Long)stream->size - pos;
        }

        if ( new_loca_len <= (FT_ULong)dist )
        {
          face->num_locations = (FT_ULong)face->root.num_glyphs + 1;
          table_len           = new_loca_len;

          FT_TRACE2(( "adjusting num_locations to %d\n",
                      face->num_locations ));
        }
        else
        {
          face->root.num_glyphs = face->num_locations
                                    ? (FT_Long)face->num_locations - 1 : 0;

          FT_TRACE2(( "adjusting num_glyphs to %d\n",
                      face->root.num_glyphs ));
        }
      }
    }

    /*
     * Extract the frame.  We don't need to decompress it since
     * we are able to parse it directly.
     */
    if ( FT_FRAME_EXTRACT( table_len, face->glyph_locations ) )
      goto Exit;

    FT_TRACE2(( "loaded\n" ));

  Exit:
    return error;
  }


  FT_LOCAL_DEF( FT_ULong )
  tt_face_get_location( TT_Face   face,
                        FT_UInt   gindex,
                        FT_UInt  *asize )
  {
    FT_ULong  pos1, pos2;
    FT_Byte*  p;
    FT_Byte*  p_limit;


    pos1 = pos2 = 0;

    if ( gindex < face->num_locations )
    {
      if ( face->header.Index_To_Loc_Format != 0 )
      {
        p       = face->glyph_locations + gindex * 4;
        p_limit = face->glyph_locations + face->num_locations * 4;

        pos1 = FT_NEXT_ULONG( p );
        pos2 = pos1;

        if ( p + 4 <= p_limit )
          pos2 = FT_NEXT_ULONG( p );
      }
      else
      {
        p       = face->glyph_locations + gindex * 2;
        p_limit = face->glyph_locations + face->num_locations * 2;

        pos1 = FT_NEXT_USHORT( p );
        pos2 = pos1;

        if ( p + 2 <= p_limit )
          pos2 = FT_NEXT_USHORT( p );

        pos1 <<= 1;
        pos2 <<= 1;
      }
    }

    /* Check broken location data. */
    if ( pos1 > face->glyf_len )
    {
      FT_TRACE1(( "tt_face_get_location:"
                  " too large offset (0x%08lx) found for glyph index %ld,\n"
                  "                     "
                  " exceeding the end of `glyf' table (0x%08lx)\n",
                  pos1, gindex, face->glyf_len ));
      *asize = 0;
      return 0;
    }

    if ( pos2 > face->glyf_len )
    {
      /* We try to sanitize the last `loca' entry. */
      if ( gindex == face->num_locations - 2 )
      {
        FT_TRACE1(( "tt_face_get_location:"
                    " too large size (%ld bytes) found for glyph index %ld,\n"
                    "                     "
                    " truncating at the end of `glyf' table to %ld bytes\n",
                    pos2 - pos1, gindex, face->glyf_len - pos1 ));
        pos2 = face->glyf_len;
      }
      else
      {
        FT_TRACE1(( "tt_face_get_location:"
                    " too large offset (0x%08lx) found for glyph index %ld,\n"
                    "                     "
                    " exceeding the end of `glyf' table (0x%08lx)\n",
                    pos2, gindex + 1, face->glyf_len ));
        *asize = 0;
        return 0;
      }
    }

    /* The `loca' table must be ordered; it refers to the length of */
    /* an entry as the difference between the current and the next  */
    /* position.  However, there do exist (malformed) fonts which   */
    /* don't obey this rule, so we are only able to provide an      */
    /* upper bound for the size.                                    */
    /*                                                              */
    /* We get (intentionally) a wrong, non-zero result in case the  */
    /* `glyf' table is missing.                                     */
    if ( pos2 >= pos1 )
      *asize = (FT_UInt)( pos2 - pos1 );
    else
      *asize = (FT_UInt)( face->glyf_len - pos1 );

    return pos1;
  }


  FT_LOCAL_DEF( void )
  tt_face_done_loca( TT_Face  face )
  {
    FT_Stream  stream = face->root.stream;


    FT_FRAME_RELEASE( face->glyph_locations );
    face->num_locations = 0;
  }



  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_cvt
   *
   * @Description:
   *   Load the control value table into a face object.
   *
   * @InOut:
   *   face ::
   *     A handle to the target face object.
   *
   * @Input:
   *   stream ::
   *     A handle to the input stream.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_load_cvt( TT_Face    face,
                    FT_Stream  stream )
  {
#ifdef TT_USE_BYTECODE_INTERPRETER

    FT_Error   error;
    FT_Memory  memory = stream->memory;
    FT_ULong   table_len;


    FT_TRACE2(( "CVT " ));

    error = face->goto_table( face, TTAG_cvt, stream, &table_len );
    if ( error )
    {
      FT_TRACE2(( "is missing\n" ));

      face->cvt_size = 0;
      face->cvt      = NULL;
      error          = FT_Err_Ok;

      goto Exit;
    }

    face->cvt_size = table_len / 2;

    if ( FT_NEW_ARRAY( face->cvt, face->cvt_size ) )
      goto Exit;

    if ( FT_FRAME_ENTER( face->cvt_size * 2L ) )
      goto Exit;

    {
      FT_Short*  cur   = face->cvt;
      FT_Short*  limit = cur + face->cvt_size;


      for ( ; cur < limit; cur++ )
        *cur = FT_GET_SHORT();
    }

    FT_FRAME_EXIT();
    FT_TRACE2(( "loaded\n" ));

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    if ( face->doblend )
      error = tt_face_vary_cvt( face, stream );
#endif

  Exit:
    return error;

#else /* !TT_USE_BYTECODE_INTERPRETER */

    FT_UNUSED( face   );
    FT_UNUSED( stream );

    return FT_Err_Ok;

#endif
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_fpgm
   *
   * @Description:
   *   Load the font program.
   *
   * @InOut:
   *   face ::
   *     A handle to the target face object.
   *
   * @Input:
   *   stream ::
   *     A handle to the input stream.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_load_fpgm( TT_Face    face,
                     FT_Stream  stream )
  {
#ifdef TT_USE_BYTECODE_INTERPRETER

    FT_Error  error;
    FT_ULong  table_len;


    FT_TRACE2(( "Font program " ));

    /* The font program is optional */
    error = face->goto_table( face, TTAG_fpgm, stream, &table_len );
    if ( error )
    {
      face->font_program      = NULL;
      face->font_program_size = 0;
      error                   = FT_Err_Ok;

      FT_TRACE2(( "is missing\n" ));
    }
    else
    {
      face->font_program_size = table_len;
      if ( FT_FRAME_EXTRACT( table_len, face->font_program ) )
        goto Exit;

      FT_TRACE2(( "loaded, %12d bytes\n", face->font_program_size ));
    }

  Exit:
    return error;

#else /* !TT_USE_BYTECODE_INTERPRETER */

    FT_UNUSED( face   );
    FT_UNUSED( stream );

    return FT_Err_Ok;

#endif
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_prep
   *
   * @Description:
   *   Load the cvt program.
   *
   * @InOut:
   *   face ::
   *     A handle to the target face object.
   *
   * @Input:
   *   stream ::
   *     A handle to the input stream.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_load_prep( TT_Face    face,
                     FT_Stream  stream )
  {
#ifdef TT_USE_BYTECODE_INTERPRETER

    FT_Error  error;
    FT_ULong  table_len;


    FT_TRACE2(( "Prep program " ));

    error = face->goto_table( face, TTAG_prep, stream, &table_len );
    if ( error )
    {
      face->cvt_program      = NULL;
      face->cvt_program_size = 0;
      error                  = FT_Err_Ok;

      FT_TRACE2(( "is missing\n" ));
    }
    else
    {
      face->cvt_program_size = table_len;
      if ( FT_FRAME_EXTRACT( table_len, face->cvt_program ) )
        goto Exit;

      FT_TRACE2(( "loaded, %12d bytes\n", face->cvt_program_size ));
    }

  Exit:
    return error;

#else /* !TT_USE_BYTECODE_INTERPRETER */

    FT_UNUSED( face   );
    FT_UNUSED( stream );

    return FT_Err_Ok;

#endif
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_hdmx
   *
   * @Description:
   *   Load the `hdmx' table into the face object.
   *
   * @Input:
   *   face ::
   *     A handle to the target face object.
   *
   *   stream ::
   *     A handle to the input stream.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */

  FT_LOCAL_DEF( FT_Error )
  tt_face_load_hdmx( TT_Face    face,
                     FT_Stream  stream )
  {
    FT_Error   error;
    FT_Memory  memory = stream->memory;
    FT_UInt    nn, num_records;
    FT_ULong   table_size, record_size;
    FT_Byte*   p;
    FT_Byte*   limit;


    /* this table is optional */
    error = face->goto_table( face, TTAG_hdmx, stream, &table_size );
    if ( error || table_size < 8 )
      return FT_Err_Ok;

    if ( FT_FRAME_EXTRACT( table_size, face->hdmx_table ) )
      goto Exit;

    p     = face->hdmx_table;
    limit = p + table_size;

    /* Given that `hdmx' tables are losing its importance (for example, */
    /* variation fonts introduced in OpenType 1.8 must not have this    */
    /* table) we no longer test for a correct `version' field.          */
    p          += 2;
    num_records = FT_NEXT_USHORT( p );
    record_size = FT_NEXT_ULONG( p );

    /* The maximum number of bytes in an hdmx device record is the */
    /* maximum number of glyphs + 2; this is 0xFFFF + 2, thus      */
    /* explaining why `record_size' is a long (which we read as    */
    /* unsigned long for convenience).  In practice, two bytes are */
    /* sufficient to hold the size value.                          */
    /*                                                             */
    /* There are at least two fonts, HANNOM-A and HANNOM-B version */
    /* 2.0 (2005), which get this wrong: The upper two bytes of    */
    /* the size value are set to 0xFF instead of 0x00.  We catch   */
    /* and fix this.                                               */

    if ( record_size >= 0xFFFF0000UL )
      record_size &= 0xFFFFU;

    /* The limit for `num_records' is a heuristic value. */
    if ( num_records > 255              ||
         ( num_records > 0            &&
           ( record_size > 0x10001L ||
             record_size < 4        ) ) )
    {
      error = FT_THROW( Invalid_File_Format );
      goto Fail;
    }

    if ( FT_NEW_ARRAY( face->hdmx_record_sizes, num_records ) )
      goto Fail;

    for ( nn = 0; nn < num_records; nn++ )
    {
      if ( p + record_size > limit )
        break;

      face->hdmx_record_sizes[nn] = p[0];
      p                          += record_size;
    }

    face->hdmx_record_count = nn;
    face->hdmx_table_size   = table_size;
    face->hdmx_record_size  = record_size;

  Exit:
    return error;

  Fail:
    FT_FRAME_RELEASE( face->hdmx_table );
    face->hdmx_table_size = 0;
    goto Exit;
  }


  FT_LOCAL_DEF( void )
  tt_face_free_hdmx( TT_Face  face )
  {
    FT_Stream  stream = face->root.stream;
    FT_Memory  memory = stream->memory;


    FT_FREE( face->hdmx_record_sizes );
    FT_FRAME_RELEASE( face->hdmx_table );
  }


  /**************************************************************************
   *
   * Return the advance width table for a given pixel size if it is found
   * in the font's `hdmx' table (if any).
   */
  FT_LOCAL_DEF( FT_Byte* )
  tt_face_get_device_metrics( TT_Face  face,
                              FT_UInt  ppem,
                              FT_UInt  gindex )
  {
    FT_UInt   nn;
    FT_Byte*  result      = NULL;
    FT_ULong  record_size = face->hdmx_record_size;
    FT_Byte*  record      = face->hdmx_table + 8;


    for ( nn = 0; nn < face->hdmx_record_count; nn++ )
      if ( face->hdmx_record_sizes[nn] == ppem )
      {
        gindex += 2;
        if ( gindex < record_size )
          result = record + nn * record_size + gindex;
        break;
      }

    return result;
  }


/* END */

/****************************************************************************
 *
 * cidgload.c
 *
 *   CID-keyed Type1 Glyph Loader (body).
 *
 * Copyright (C) 1996-2025 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include "cidload.h"
#include "cidgload.h"
#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/ftoutln.h>
#include <freetype/internal/ftcalc.h>

#include <freetype/internal/psaux.h>
#include <freetype/internal/cfftypes.h>
#include <freetype/ftdriver.h>

#include "ciderrs.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  cidgload


  /*
   * A helper function to compute FD number (`fd_select`), the offset to the
   * head of the glyph data (`off1`), and the offset to the and of the glyph
   * data (`off2`).
   *
   * The number how many times `cid_get_offset` is invoked can be controlled
   * by the number of non-NULL arguments.  If `fd_select` is non-NULL but
   * `off1` and `off2` are NULL, `cid_get_offset` is invoked only for
   * `fd_select`; `off1` and `off2` are not validated.
   *
   */
  FT_LOCAL_DEF( FT_Error )
  cid_compute_fd_and_offsets( CID_Face   face,
                              FT_UInt    glyph_index,
                              FT_ULong*  fd_select_p,
                              FT_ULong*  off1_p,
                              FT_ULong*  off2_p )
  {
    FT_Error  error = FT_Err_Ok;

    CID_FaceInfo  cid       = &face->cid;
    FT_Stream     stream    =  face->cid_stream;
    FT_UInt       entry_len = cid->fd_bytes + cid->gd_bytes;

    FT_Byte*  p;
    FT_Bool   need_frame_exit = 0;
    FT_ULong  fd_select, off1, off2;


    /* For ordinary fonts, read the CID font dictionary index */
    /* and charstring offset from the CIDMap.                 */

    if ( FT_STREAM_SEEK( cid->data_offset + cid->cidmap_offset +
                         glyph_index * entry_len )               ||
         FT_FRAME_ENTER( 2 * entry_len )                         )
      goto Exit;

    need_frame_exit = 1;

    p         = (FT_Byte*)stream->cursor;
    fd_select = cid_get_offset( &p, cid->fd_bytes );
    off1      = cid_get_offset( &p, cid->gd_bytes );

    p    += cid->fd_bytes;
    off2  = cid_get_offset( &p, cid->gd_bytes );

    if ( fd_select_p )
      *fd_select_p = fd_select;
    if ( off1_p )
      *off1_p = off1;
    if ( off2_p )
      *off2_p = off2;

    if ( fd_select >= cid->num_dicts )
    {
      /*
       * fd_select == 0xFF is often used to indicate that the CID
       * has no charstring to be rendered, similar to GID = 0xFFFF
       * in TrueType fonts.
       */
      if ( ( cid->fd_bytes == 1 && fd_select == 0xFFU   ) ||
           ( cid->fd_bytes == 2 && fd_select == 0xFFFFU ) )
      {
        FT_TRACE1(( "cid_load_glyph: fail for glyph index %u:\n",
                    glyph_index ));
        FT_TRACE1(( "                FD number %lu is the maximum\n",
                    fd_select ));
        FT_TRACE1(( "                integer fitting into %u byte%s\n",
                    cid->fd_bytes, cid->fd_bytes == 1 ? "" : "s" ));
      }
      else
      {
        FT_TRACE0(( "cid_load_glyph: fail for glyph index %u:\n",
                    glyph_index ));
        FT_TRACE0(( "                FD number %lu is larger\n",
                    fd_select ));
        FT_TRACE0(( "                than number of dictionaries (%u)\n",
                    cid->num_dicts ));
      }

      error = FT_THROW( Invalid_Offset );
      goto Exit;
    }
    else if ( off2 > stream->size )
    {
      FT_TRACE0(( "cid_load_glyph: fail for glyph index %u:\n",
                  glyph_index ));
      FT_TRACE0(( "               end of the glyph data\n" ));
      FT_TRACE0(( "               is beyond the data stream\n" ));

      error = FT_THROW( Invalid_Offset );
      goto Exit;
    }
    else if ( off1 > off2 )
    {
      FT_TRACE0(( "cid_load_glyph: fail for glyph index %u:\n",
                  glyph_index ));
      FT_TRACE0(( "                the end position of glyph data\n" ));
      FT_TRACE0(( "                is set before the start position\n" ));

      error = FT_THROW( Invalid_Offset );
    }

    Exit:
      if ( need_frame_exit )
        FT_FRAME_EXIT();

    return error;
  }


  FT_CALLBACK_DEF( FT_Error )
  cid_load_glyph( T1_Decoder  decoder,
                  FT_UInt     glyph_index )
  {
    CID_Face       face = (CID_Face)decoder->builder.face;
    CID_FaceInfo   cid  = &face->cid;
    FT_Byte*       p;
    FT_ULong       fd_select;
    FT_Stream      stream       = face->cid_stream;
    FT_Error       error        = FT_Err_Ok;
    FT_Byte*       charstring   = NULL;
    FT_Memory      memory       = face->root.memory;
    FT_ULong       glyph_length = 0;
    PSAux_Service  psaux        = (PSAux_Service)face->psaux;

    FT_Bool  force_scaling = FALSE;

#ifdef FT_CONFIG_OPTION_INCREMENTAL
    FT_Incremental_InterfaceRec  *inc =
                                   face->root.internal->incremental_interface;
#endif


    FT_TRACE1(( "cid_load_glyph: glyph index %u\n", glyph_index ));

#ifdef FT_CONFIG_OPTION_INCREMENTAL

    /* For incremental fonts get the character data using */
    /* the callback function.                             */
    if ( inc )
    {
      FT_Data  glyph_data;


      error = inc->funcs->get_glyph_data( inc->object,
                                          glyph_index, &glyph_data );
      if ( error || glyph_data.length < cid->fd_bytes )
        goto Exit;

      p         = (FT_Byte*)glyph_data.pointer;
      fd_select = cid_get_offset( &p, cid->fd_bytes );

      glyph_length = glyph_data.length - cid->fd_bytes;

      if ( !FT_QALLOC( charstring, glyph_length ) )
        FT_MEM_COPY( charstring, glyph_data.pointer + cid->fd_bytes,
                     glyph_length );

      inc->funcs->free_glyph_data( inc->object, &glyph_data );

      if ( error )
        goto Exit;
    }

    else

#endif /* FT_CONFIG_OPTION_INCREMENTAL */
    {
      FT_ULong  off1, off2;


      error = cid_compute_fd_and_offsets( face, glyph_index,
                                          &fd_select, &off1, &off2 );
      if ( error )
        goto Exit;

      glyph_length = off2 - off1;

      if ( glyph_length == 0                             ||
           FT_QALLOC( charstring, glyph_length )         ||
           FT_STREAM_READ_AT( cid->data_offset + off1,
                              charstring, glyph_length ) )
        goto Exit;
    }

    /* Now set up the subrs array and parse the charstrings. */
    {
      CID_FaceDict  dict;
      CID_Subrs     cid_subrs = face->subrs + fd_select;
      FT_UInt       cs_offset;


      /* Set up subrs */
      decoder->num_subrs  = cid_subrs->num_subrs;
      decoder->subrs      = cid_subrs->code;
      decoder->subrs_len  = 0;
      decoder->subrs_hash = NULL;

      /* Set up font matrix */
      dict                 = cid->font_dicts + fd_select;

      decoder->font_matrix = dict->font_matrix;
      decoder->font_offset = dict->font_offset;
      decoder->lenIV       = dict->private_dict.lenIV;

      /* Decode the charstring. */

      /* Adjustment for seed bytes. */
      cs_offset = decoder->lenIV >= 0 ? (FT_UInt)decoder->lenIV : 0;
      if ( cs_offset > glyph_length )
      {
        FT_TRACE0(( "cid_load_glyph: fail for glyph_index=%u,"
                    " offset to the charstring is beyond glyph length\n",
                    glyph_index ));
        error = FT_THROW( Invalid_Offset );
        goto Exit;
      }

      /* Decrypt only if lenIV >= 0. */
      if ( decoder->lenIV >= 0 )
        psaux->t1_decrypt( charstring, glyph_length, 4330 );

      /* choose which renderer to use */
#ifdef T1_CONFIG_OPTION_OLD_ENGINE
      if ( ( (PS_Driver)FT_FACE_DRIVER( face ) )->hinting_engine ==
               FT_HINTING_FREETYPE                                  ||
           decoder->builder.metrics_only                            )
        error = psaux->t1_decoder_funcs->parse_charstrings_old(
                  decoder,
                  charstring + cs_offset,
                  glyph_length - cs_offset );
#else
      if ( decoder->builder.metrics_only )
        error = psaux->t1_decoder_funcs->parse_metrics(
                  decoder,
                  charstring + cs_offset,
                  glyph_length - cs_offset );
#endif
      else
      {
        PS_Decoder      psdecoder;
        CFF_SubFontRec  subfont;


        psaux->ps_decoder_init( &psdecoder, decoder, TRUE );

        psaux->t1_make_subfont( FT_FACE( face ),
                                &dict->private_dict,
                                &subfont );
        psdecoder.current_subfont = &subfont;

        error = psaux->t1_decoder_funcs->parse_charstrings(
                  &psdecoder,
                  charstring + cs_offset,
                  glyph_length - cs_offset );

        /* Adobe's engine uses 16.16 numbers everywhere;              */
        /* as a consequence, glyphs larger than 2000ppem get rejected */
        if ( FT_ERR_EQ( error, Glyph_Too_Big ) )
        {
          /* this time, we retry unhinted and scale up the glyph later on */
          /* (the engine uses and sets the hardcoded value 0x10000 / 64 = */
          /* 0x400 for both `x_scale' and `y_scale' in this case)         */
          ((CID_GlyphSlot)decoder->builder.glyph)->hint = FALSE;

          force_scaling = TRUE;

          error = psaux->t1_decoder_funcs->parse_charstrings(
                    &psdecoder,
                    charstring + cs_offset,
                    glyph_length - cs_offset );
        }
      }
    }

#ifdef FT_CONFIG_OPTION_INCREMENTAL

    /* Incremental fonts can optionally override the metrics. */
    if ( !error && inc && inc->funcs->get_glyph_metrics )
    {
      FT_Incremental_MetricsRec  metrics;


      metrics.bearing_x = FIXED_TO_INT( decoder->builder.left_bearing.x );
      metrics.bearing_y = 0;
      metrics.advance   = FIXED_TO_INT( decoder->builder.advance.x );
      metrics.advance_v = FIXED_TO_INT( decoder->builder.advance.y );

      error = inc->funcs->get_glyph_metrics( inc->object,
                                             glyph_index, FALSE, &metrics );

      decoder->builder.left_bearing.x = INT_TO_FIXED( metrics.bearing_x );
      decoder->builder.advance.x      = INT_TO_FIXED( metrics.advance );
      decoder->builder.advance.y      = INT_TO_FIXED( metrics.advance_v );
    }

#endif /* FT_CONFIG_OPTION_INCREMENTAL */

  Exit:
    FT_FREE( charstring );

    ((CID_GlyphSlot)decoder->builder.glyph)->scaled = force_scaling;

    return error;
  }


#if 0


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /**********                                                      *********/
  /**********                                                      *********/
  /**********            COMPUTE THE MAXIMUM ADVANCE WIDTH         *********/
  /**********                                                      *********/
  /**********    The following code is in charge of computing      *********/
  /**********    the maximum advance width of the font.  It        *********/
  /**********    quickly processes each glyph charstring to        *********/
  /**********    extract the value from either a `sbw' or `seac'   *********/
  /**********    operator.                                         *********/
  /**********                                                      *********/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


  FT_LOCAL_DEF( FT_Error )
  cid_face_compute_max_advance( CID_Face  face,
                                FT_Int*   max_advance )
  {
    FT_Error       error;
    T1_DecoderRec  decoder;
    FT_Int         glyph_index;

    PSAux_Service  psaux = (PSAux_Service)face->psaux;


    *max_advance = 0;

    /* Initialize load decoder */
    error = psaux->t1_decoder_funcs->init( &decoder,
                                           (FT_Face)face,
                                           0, /* size       */
                                           0, /* glyph slot */
                                           0, /* glyph names! XXX */
                                           0, /* blend == 0 */
                                           0, /* hinting == 0 */
                                           cid_load_glyph );
    if ( error )
      return error;

    /* TODO: initialize decoder.len_buildchar and decoder.buildchar */
    /*       if we ever support CID-keyed multiple master fonts     */

    decoder.builder.metrics_only = 1;
    decoder.builder.load_points  = 0;

    /* for each glyph, parse the glyph charstring and extract */
    /* the advance width                                      */
    for ( glyph_index = 0; glyph_index < face->root.num_glyphs;
          glyph_index++ )
    {
      /* now get load the unscaled outline */
      error = cid_load_glyph( &decoder, glyph_index );
      /* ignore the error if one occurred - skip to next glyph */
    }

    *max_advance = FIXED_TO_INT( decoder.builder.advance.x );

    psaux->t1_decoder_funcs->done( &decoder );

    return FT_Err_Ok;
  }


#endif /* 0 */


  FT_LOCAL_DEF( FT_Error )
  cid_slot_load_glyph( FT_GlyphSlot  cidglyph,      /* CID_GlyphSlot */
                       FT_Size       cidsize,       /* CID_Size      */
                       FT_UInt       glyph_index,
                       FT_Int32      load_flags )
  {
    CID_GlyphSlot  glyph = (CID_GlyphSlot)cidglyph;
    FT_Error       error;
    T1_DecoderRec  decoder;
    CID_Face       face = (CID_Face)cidglyph->face;
    FT_Bool        hinting;
    FT_Bool        scaled;

    PSAux_Service  psaux = (PSAux_Service)face->psaux;
    FT_Matrix      font_matrix;
    FT_Vector      font_offset;
    FT_Bool        must_finish_decoder = FALSE;


    if ( glyph_index >= (FT_UInt)face->root.num_glyphs )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( load_flags & FT_LOAD_NO_RECURSE )
      load_flags |= FT_LOAD_NO_SCALE | FT_LOAD_NO_HINTING;

    glyph->x_scale = cidsize->metrics.x_scale;
    glyph->y_scale = cidsize->metrics.y_scale;

    hinting = FT_BOOL( ( load_flags & FT_LOAD_NO_SCALE   ) == 0 &&
                       ( load_flags & FT_LOAD_NO_HINTING ) == 0 );
    scaled  = FT_BOOL( ( load_flags & FT_LOAD_NO_SCALE   ) == 0 );

    glyph->hint      = hinting;
    glyph->scaled    = scaled;

    error = psaux->t1_decoder_funcs->init( &decoder,
                                           cidglyph->face,
                                           cidsize,
                                           cidglyph,
                                           0, /* glyph names -- XXX */
                                           0, /* blend == 0 */
                                           hinting,
                                           FT_LOAD_TARGET_MODE( load_flags ),
                                           cid_load_glyph );
    if ( error )
      goto Exit;

    /* TODO: initialize decoder.len_buildchar and decoder.buildchar */
    /*       if we ever support CID-keyed multiple master fonts     */

    must_finish_decoder = TRUE;

    /* set up the decoder */
    decoder.builder.no_recurse = FT_BOOL( load_flags & FT_LOAD_NO_RECURSE );

    error = cid_load_glyph( &decoder, glyph_index );
    if ( error )
      goto Exit;

    /* copy flags back for forced scaling */
    hinting = glyph->hint;
    scaled  = glyph->scaled;

    font_matrix = decoder.font_matrix;
    font_offset = decoder.font_offset;

    /* save new glyph tables */
    psaux->t1_decoder_funcs->done( &decoder );

    must_finish_decoder = FALSE;

    /* now set the metrics -- this is rather simple, as    */
    /* the left side bearing is the xMin, and the top side */
    /* bearing the yMax; for composite glyphs, return only */
    /* left side bearing and advance width                 */
    if ( load_flags & FT_LOAD_NO_RECURSE )
    {
      FT_Slot_Internal  internal = cidglyph->internal;


      cidglyph->metrics.horiBearingX =
        FIXED_TO_INT( decoder.builder.left_bearing.x );
      cidglyph->metrics.horiAdvance =
        FIXED_TO_INT( decoder.builder.advance.x );

      internal->glyph_matrix      = font_matrix;
      internal->glyph_delta       = font_offset;
      internal->glyph_transformed = 1;
    }
    else
    {
      FT_BBox            cbox;
      FT_Glyph_Metrics*  metrics = &cidglyph->metrics;


      cidglyph->format = FT_GLYPH_FORMAT_OUTLINE;

      cidglyph->outline.flags &= FT_OUTLINE_OWNER;
      cidglyph->outline.flags |= FT_OUTLINE_REVERSE_FILL;
      if ( cidsize->metrics.y_ppem < 24 )
        cidglyph->outline.flags |= FT_OUTLINE_HIGH_PRECISION;

      /* copy the _unscaled_ advance width */
      metrics->horiAdvance =
        FIXED_TO_INT( decoder.builder.advance.x );
      cidglyph->linearHoriAdvance =
        FIXED_TO_INT( decoder.builder.advance.x );
      cidglyph->internal->glyph_transformed = 0;

      /* make up vertical ones */
      metrics->vertAdvance        = ( face->cid.font_bbox.yMax -
                                      face->cid.font_bbox.yMin ) >> 16;
      cidglyph->linearVertAdvance = metrics->vertAdvance;

      /* apply the font matrix, if any */
      if ( font_matrix.xx != 0x10000L || font_matrix.yy != 0x10000L ||
           font_matrix.xy != 0        || font_matrix.yx != 0        )
      {
        FT_Outline_Transform( &cidglyph->outline, &font_matrix );

        metrics->horiAdvance = FT_MulFix( metrics->horiAdvance,
                                          font_matrix.xx );
        metrics->vertAdvance = FT_MulFix( metrics->vertAdvance,
                                          font_matrix.yy );
      }

      if ( font_offset.x || font_offset.y )
      {
        FT_Outline_Translate( &cidglyph->outline,
                              font_offset.x,
                              font_offset.y );

        metrics->horiAdvance += font_offset.x;
        metrics->vertAdvance += font_offset.y;
      }

      if ( ( load_flags & FT_LOAD_NO_SCALE ) == 0 || scaled )
      {
        /* scale the outline and the metrics */
        FT_Int       n;
        FT_Outline*  cur = decoder.builder.base;
        FT_Vector*   vec = cur->points;
        FT_Fixed     x_scale = glyph->x_scale;
        FT_Fixed     y_scale = glyph->y_scale;


        /* First of all, scale the points */
        if ( !hinting || !decoder.builder.hints_funcs )
          for ( n = cur->n_points; n > 0; n--, vec++ )
          {
            vec->x = FT_MulFix( vec->x, x_scale );
            vec->y = FT_MulFix( vec->y, y_scale );
          }

        /* Then scale the metrics */
        metrics->horiAdvance = FT_MulFix( metrics->horiAdvance, x_scale );
        metrics->vertAdvance = FT_MulFix( metrics->vertAdvance, y_scale );
      }

      /* compute the other metrics */
      FT_Outline_Get_CBox( &cidglyph->outline, &cbox );

      metrics->width  = cbox.xMax - cbox.xMin;
      metrics->height = cbox.yMax - cbox.yMin;

      metrics->horiBearingX = cbox.xMin;
      metrics->horiBearingY = cbox.yMax;

      if ( load_flags & FT_LOAD_VERTICAL_LAYOUT )
      {
        /* make up vertical ones */
        ft_synthesize_vertical_metrics( metrics,
                                        metrics->vertAdvance );
      }
    }

  Exit:

    if ( must_finish_decoder )
      psaux->t1_decoder_funcs->done( &decoder );

    return error;
  }


/* END */

/****************************************************************************
 *
 * t1gload.c
 *
 *   Type 1 Glyph Loader (body).
 *
 * Copyright (C) 1996-2020 by
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
#include "t1gload.h"
#include FT_INTERNAL_CALC_H
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_STREAM_H
#include FT_OUTLINE_H
#include FT_INTERNAL_POSTSCRIPT_AUX_H
#include FT_INTERNAL_CFF_TYPES_H
#include FT_DRIVER_H

#include "t1errors.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  t1gload


  static FT_Error
  T1_Parse_Glyph_And_Get_Char_String( T1_Decoder  decoder,
                                      FT_UInt     glyph_index,
                                      FT_Data*    char_string,
                                      FT_Bool*    force_scaling )
  {
    T1_Face   face  = (T1_Face)decoder->builder.face;
    T1_Font   type1 = &face->type1;
    FT_Error  error = FT_Err_Ok;

    PSAux_Service           psaux         = (PSAux_Service)face->psaux;
    const T1_Decoder_Funcs  decoder_funcs = psaux->t1_decoder_funcs;
    PS_Decoder              psdecoder;

#ifdef FT_CONFIG_OPTION_INCREMENTAL
    FT_Incremental_InterfaceRec *inc =
                      face->root.internal->incremental_interface;
#endif

#ifdef T1_CONFIG_OPTION_OLD_ENGINE
    PS_Driver  driver = (PS_Driver)FT_FACE_DRIVER( face );
#endif


    decoder->font_matrix = type1->font_matrix;
    decoder->font_offset = type1->font_offset;

#ifdef FT_CONFIG_OPTION_INCREMENTAL

    /* For incremental fonts get the character data using the */
    /* callback function.                                     */
    if ( inc )
      error = inc->funcs->get_glyph_data( inc->object,
                                          glyph_index, char_string );
    else

#endif /* FT_CONFIG_OPTION_INCREMENTAL */

    /* For ordinary fonts get the character data stored in the face record. */
    {
      char_string->pointer = type1->charstrings[glyph_index];
      char_string->length  = (FT_Int)type1->charstrings_len[glyph_index];
    }

    if ( !error )
    {
      /* choose which renderer to use */
#ifdef T1_CONFIG_OPTION_OLD_ENGINE
      if ( driver->hinting_engine == FT_HINTING_FREETYPE ||
           decoder->builder.metrics_only                 )
        error = decoder_funcs->parse_charstrings_old(
                  decoder,
                  (FT_Byte*)char_string->pointer,
                  (FT_UInt)char_string->length );
#else
      if ( decoder->builder.metrics_only )
        error = decoder_funcs->parse_metrics(
                  decoder,
                  (FT_Byte*)char_string->pointer,
                  (FT_UInt)char_string->length );
#endif
      else
      {
        CFF_SubFontRec  subfont;


        psaux->ps_decoder_init( &psdecoder, decoder, TRUE );

        psaux->t1_make_subfont( FT_FACE( face ),
                                &face->type1.private_dict, &subfont );
        psdecoder.current_subfont = &subfont;

        error = decoder_funcs->parse_charstrings(
                  &psdecoder,
                  (FT_Byte*)char_string->pointer,
                  (FT_ULong)char_string->length );

        /* Adobe's engine uses 16.16 numbers everywhere;              */
        /* as a consequence, glyphs larger than 2000ppem get rejected */
        if ( FT_ERR_EQ( error, Glyph_Too_Big ) )
        {
          /* this time, we retry unhinted and scale up the glyph later on */
          /* (the engine uses and sets the hardcoded value 0x10000 / 64 = */
          /* 0x400 for both `x_scale' and `y_scale' in this case)         */
          ((T1_GlyphSlot)decoder->builder.glyph)->hint = FALSE;

          *force_scaling = TRUE;

          error = decoder_funcs->parse_charstrings(
                    &psdecoder,
                    (FT_Byte*)char_string->pointer,
                    (FT_ULong)char_string->length );
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

    return error;
  }


  FT_CALLBACK_DEF( FT_Error )
  T1_Parse_Glyph( T1_Decoder  decoder,
                  FT_UInt     glyph_index )
  {
    FT_Data   glyph_data;
    FT_Bool   force_scaling = FALSE;
    FT_Error  error         = T1_Parse_Glyph_And_Get_Char_String(
                                decoder, glyph_index, &glyph_data,
                                &force_scaling );


#ifdef FT_CONFIG_OPTION_INCREMENTAL

    if ( !error )
    {
      T1_Face  face = (T1_Face)decoder->builder.face;


      if ( face->root.internal->incremental_interface )
        face->root.internal->incremental_interface->funcs->free_glyph_data(
          face->root.internal->incremental_interface->object,
          &glyph_data );
    }

#endif /* FT_CONFIG_OPTION_INCREMENTAL */

    return error;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
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
  T1_Compute_Max_Advance( T1_Face  face,
                          FT_Pos*  max_advance )
  {
    FT_Error       error;
    T1_DecoderRec  decoder;
    FT_Int         glyph_index;
    T1_Font        type1 = &face->type1;
    PSAux_Service  psaux = (PSAux_Service)face->psaux;


    FT_ASSERT( ( face->len_buildchar == 0 ) == ( face->buildchar == NULL ) );

    *max_advance = 0;

    /* initialize load decoder */
    error = psaux->t1_decoder_funcs->init( &decoder,
                                           (FT_Face)face,
                                           0, /* size       */
                                           0, /* glyph slot */
                                           (FT_Byte**)type1->glyph_names,
                                           face->blend,
                                           0,
                                           FT_RENDER_MODE_NORMAL,
                                           T1_Parse_Glyph );
    if ( error )
      return error;

    decoder.builder.metrics_only = 1;
    decoder.builder.load_points  = 0;

    decoder.num_subrs     = type1->num_subrs;
    decoder.subrs         = type1->subrs;
    decoder.subrs_len     = type1->subrs_len;
    decoder.subrs_hash    = type1->subrs_hash;

    decoder.buildchar     = face->buildchar;
    decoder.len_buildchar = face->len_buildchar;

    *max_advance = 0;

    FT_TRACE6(( "T1_Compute_Max_Advance:\n" ));

    /* for each glyph, parse the glyph charstring and extract */
    /* the advance width                                      */
    for ( glyph_index = 0; glyph_index < type1->num_glyphs; glyph_index++ )
    {
      /* now get load the unscaled outline */
      (void)T1_Parse_Glyph( &decoder, (FT_UInt)glyph_index );
      if ( glyph_index == 0 || decoder.builder.advance.x > *max_advance )
        *max_advance = decoder.builder.advance.x;

      /* ignore the error if one occurred - skip to next glyph */
    }

    FT_TRACE6(( "T1_Compute_Max_Advance: max advance: %f\n",
                *max_advance / 65536.0 ));

    psaux->t1_decoder_funcs->done( &decoder );

    return FT_Err_Ok;
  }


  FT_LOCAL_DEF( FT_Error )
  T1_Get_Advances( FT_Face    t1face,        /* T1_Face */
                   FT_UInt    first,
                   FT_UInt    count,
                   FT_Int32   load_flags,
                   FT_Fixed*  advances )
  {
    T1_Face        face  = (T1_Face)t1face;
    T1_DecoderRec  decoder;
    T1_Font        type1 = &face->type1;
    PSAux_Service  psaux = (PSAux_Service)face->psaux;
    FT_UInt        nn;
    FT_Error       error;


    FT_TRACE5(( "T1_Get_Advances:\n" ));

    if ( load_flags & FT_LOAD_VERTICAL_LAYOUT )
    {
      for ( nn = 0; nn < count; nn++ )
      {
        advances[nn] = 0;

        FT_TRACE5(( "  idx %d: advance height 0 font units\n",
                    first + nn ));
      }

      return FT_Err_Ok;
    }

    error = psaux->t1_decoder_funcs->init( &decoder,
                                           (FT_Face)face,
                                           0, /* size       */
                                           0, /* glyph slot */
                                           (FT_Byte**)type1->glyph_names,
                                           face->blend,
                                           0,
                                           FT_RENDER_MODE_NORMAL,
                                           T1_Parse_Glyph );
    if ( error )
      return error;

    decoder.builder.metrics_only = 1;
    decoder.builder.load_points  = 0;

    decoder.num_subrs  = type1->num_subrs;
    decoder.subrs      = type1->subrs;
    decoder.subrs_len  = type1->subrs_len;
    decoder.subrs_hash = type1->subrs_hash;

    decoder.buildchar     = face->buildchar;
    decoder.len_buildchar = face->len_buildchar;

    for ( nn = 0; nn < count; nn++ )
    {
      error = T1_Parse_Glyph( &decoder, first + nn );
      if ( !error )
        advances[nn] = FIXED_TO_INT( decoder.builder.advance.x );
      else
        advances[nn] = 0;

      FT_TRACE5(( "  idx %d: advance width %d font unit%s\n",
                  first + nn,
                  advances[nn],
                  advances[nn] == 1 ? "" : "s" ));
    }

    return FT_Err_Ok;
  }


  FT_LOCAL_DEF( FT_Error )
  T1_Load_Glyph( FT_GlyphSlot  t1glyph,          /* T1_GlyphSlot */
                 FT_Size       t1size,           /* T1_Size      */
                 FT_UInt       glyph_index,
                 FT_Int32      load_flags )
  {
    T1_GlyphSlot            glyph = (T1_GlyphSlot)t1glyph;
    FT_Error                error;
    T1_DecoderRec           decoder;
    T1_Face                 face = (T1_Face)t1glyph->face;
    FT_Bool                 hinting;
    FT_Bool                 scaled;
    FT_Bool                 force_scaling = FALSE;
    T1_Font                 type1         = &face->type1;
    PSAux_Service           psaux         = (PSAux_Service)face->psaux;
    const T1_Decoder_Funcs  decoder_funcs = psaux->t1_decoder_funcs;

    FT_Matrix               font_matrix;
    FT_Vector               font_offset;
    FT_Data                 glyph_data;
    FT_Bool                 must_finish_decoder = FALSE;
#ifdef FT_CONFIG_OPTION_INCREMENTAL
    FT_Bool                 glyph_data_loaded = 0;
#endif


#ifdef FT_CONFIG_OPTION_INCREMENTAL
    if ( glyph_index >= (FT_UInt)face->root.num_glyphs &&
         !face->root.internal->incremental_interface   )
#else
    if ( glyph_index >= (FT_UInt)face->root.num_glyphs )
#endif /* FT_CONFIG_OPTION_INCREMENTAL */
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    FT_TRACE1(( "T1_Load_Glyph: glyph index %d\n", glyph_index ));

    FT_ASSERT( ( face->len_buildchar == 0 ) == ( face->buildchar == NULL ) );

    if ( load_flags & FT_LOAD_NO_RECURSE )
      load_flags |= FT_LOAD_NO_SCALE | FT_LOAD_NO_HINTING;

    if ( t1size )
    {
      glyph->x_scale = t1size->metrics.x_scale;
      glyph->y_scale = t1size->metrics.y_scale;
    }
    else
    {
      glyph->x_scale = 0x10000L;
      glyph->y_scale = 0x10000L;
    }

    t1glyph->outline.n_points   = 0;
    t1glyph->outline.n_contours = 0;

    hinting = FT_BOOL( !( load_flags & FT_LOAD_NO_SCALE   ) &&
                       !( load_flags & FT_LOAD_NO_HINTING ) );
    scaled  = FT_BOOL( !( load_flags & FT_LOAD_NO_SCALE   ) );

    glyph->hint     = hinting;
    glyph->scaled   = scaled;
    t1glyph->format = FT_GLYPH_FORMAT_OUTLINE;

    error = decoder_funcs->init( &decoder,
                                 t1glyph->face,
                                 t1size,
                                 t1glyph,
                                 (FT_Byte**)type1->glyph_names,
                                 face->blend,
                                 hinting,
                                 FT_LOAD_TARGET_MODE( load_flags ),
                                 T1_Parse_Glyph );
    if ( error )
      goto Exit;

    must_finish_decoder = TRUE;

    decoder.builder.no_recurse = FT_BOOL( load_flags & FT_LOAD_NO_RECURSE );

    decoder.num_subrs     = type1->num_subrs;
    decoder.subrs         = type1->subrs;
    decoder.subrs_len     = type1->subrs_len;
    decoder.subrs_hash    = type1->subrs_hash;

    decoder.buildchar     = face->buildchar;
    decoder.len_buildchar = face->len_buildchar;

    /* now load the unscaled outline */
    error = T1_Parse_Glyph_And_Get_Char_String( &decoder, glyph_index,
                                                &glyph_data,
                                                &force_scaling );
    if ( error )
      goto Exit;
#ifdef FT_CONFIG_OPTION_INCREMENTAL
    glyph_data_loaded = 1;
#endif

    hinting     = glyph->hint;
    font_matrix = decoder.font_matrix;
    font_offset = decoder.font_offset;

    /* save new glyph tables */
    decoder_funcs->done( &decoder );

    must_finish_decoder = FALSE;

    /* now, set the metrics -- this is rather simple, as   */
    /* the left side bearing is the xMin, and the top side */
    /* bearing the yMax                                    */
    if ( !error )
    {
      t1glyph->outline.flags &= FT_OUTLINE_OWNER;
      t1glyph->outline.flags |= FT_OUTLINE_REVERSE_FILL;

      /* for composite glyphs, return only left side bearing and */
      /* advance width                                           */
      if ( load_flags & FT_LOAD_NO_RECURSE )
      {
        FT_Slot_Internal  internal = t1glyph->internal;


        t1glyph->metrics.horiBearingX =
          FIXED_TO_INT( decoder.builder.left_bearing.x );
        t1glyph->metrics.horiAdvance  =
          FIXED_TO_INT( decoder.builder.advance.x );

        internal->glyph_matrix      = font_matrix;
        internal->glyph_delta       = font_offset;
        internal->glyph_transformed = 1;
      }
      else
      {
        FT_BBox            cbox;
        FT_Glyph_Metrics*  metrics = &t1glyph->metrics;


        /* copy the _unscaled_ advance width */
        metrics->horiAdvance =
          FIXED_TO_INT( decoder.builder.advance.x );
        t1glyph->linearHoriAdvance =
          FIXED_TO_INT( decoder.builder.advance.x );
        t1glyph->internal->glyph_transformed = 0;

        if ( load_flags & FT_LOAD_VERTICAL_LAYOUT )
        {
          /* make up vertical ones */
          metrics->vertAdvance = ( face->type1.font_bbox.yMax -
                                   face->type1.font_bbox.yMin ) >> 16;
          t1glyph->linearVertAdvance = metrics->vertAdvance;
        }
        else
        {
          metrics->vertAdvance =
            FIXED_TO_INT( decoder.builder.advance.y );
          t1glyph->linearVertAdvance =
            FIXED_TO_INT( decoder.builder.advance.y );
        }

        t1glyph->format = FT_GLYPH_FORMAT_OUTLINE;

        if ( t1size && t1size->metrics.y_ppem < 24 )
          t1glyph->outline.flags |= FT_OUTLINE_HIGH_PRECISION;

#if 1
        /* apply the font matrix, if any */
        if ( font_matrix.xx != 0x10000L || font_matrix.yy != 0x10000L ||
             font_matrix.xy != 0        || font_matrix.yx != 0        )
        {
          FT_Outline_Transform( &t1glyph->outline, &font_matrix );

          metrics->horiAdvance = FT_MulFix( metrics->horiAdvance,
                                            font_matrix.xx );
          metrics->vertAdvance = FT_MulFix( metrics->vertAdvance,
                                            font_matrix.yy );
        }

        if ( font_offset.x || font_offset.y )
        {
          FT_Outline_Translate( &t1glyph->outline,
                                font_offset.x,
                                font_offset.y );

          metrics->horiAdvance += font_offset.x;
          metrics->vertAdvance += font_offset.y;
        }
#endif

        if ( ( load_flags & FT_LOAD_NO_SCALE ) == 0 || force_scaling )
        {
          /* scale the outline and the metrics */
          FT_Int       n;
          FT_Outline*  cur = decoder.builder.base;
          FT_Vector*   vec = cur->points;
          FT_Fixed     x_scale = glyph->x_scale;
          FT_Fixed     y_scale = glyph->y_scale;


          /* First of all, scale the points, if we are not hinting */
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
        FT_Outline_Get_CBox( &t1glyph->outline, &cbox );

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

      /* Set control data to the glyph charstrings.  Note that this is */
      /* _not_ zero-terminated.                                        */
      t1glyph->control_data = (FT_Byte*)glyph_data.pointer;
      t1glyph->control_len  = glyph_data.length;
    }


  Exit:

#ifdef FT_CONFIG_OPTION_INCREMENTAL
    if ( glyph_data_loaded && face->root.internal->incremental_interface )
    {
      face->root.internal->incremental_interface->funcs->free_glyph_data(
        face->root.internal->incremental_interface->object,
        &glyph_data );

      /* Set the control data to null - it is no longer available if   */
      /* loaded incrementally.                                         */
      t1glyph->control_data = NULL;
      t1glyph->control_len  = 0;
    }
#endif

    if ( must_finish_decoder )
      decoder_funcs->done( &decoder );

    return error;
  }


/* END */

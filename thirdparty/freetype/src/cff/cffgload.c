/****************************************************************************
 *
 * cffgload.c
 *
 *   OpenType Glyph Loader (body).
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


#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/sfnt.h>
#include <freetype/internal/ftcalc.h>
#include <freetype/internal/psaux.h>
#include <freetype/ftoutln.h>
#include <freetype/ftdriver.h>

#include "cffload.h"
#include "cffgload.h"

#include "cfferrs.h"

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#define IS_DEFAULT_INSTANCE( _face )             \
          ( !( FT_IS_NAMED_INSTANCE( _face ) ||  \
               FT_IS_VARIATION( _face )      ) )
#else
#define IS_DEFAULT_INSTANCE( _face )  1
#endif


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  cffgload


  FT_LOCAL_DEF( FT_Error )
  cff_get_glyph_data( TT_Face    face,
                      FT_UInt    glyph_index,
                      FT_Byte**  pointer,
                      FT_ULong*  length )
  {
#ifdef FT_CONFIG_OPTION_INCREMENTAL
    /* For incremental fonts get the character data using the */
    /* callback function.                                     */
    if ( face->root.internal->incremental_interface )
    {
      FT_Data   data;
      FT_Error  error =
                  face->root.internal->incremental_interface->funcs->get_glyph_data(
                    face->root.internal->incremental_interface->object,
                    glyph_index, &data );


      *pointer = (FT_Byte*)data.pointer;
      *length  = data.length;

      return error;
    }
    else
#endif /* FT_CONFIG_OPTION_INCREMENTAL */

    {
      CFF_Font  cff = (CFF_Font)( face->extra.data );


      return cff_index_access_element( &cff->charstrings_index, glyph_index,
                                       pointer, length );
    }
  }


  FT_LOCAL_DEF( void )
  cff_free_glyph_data( TT_Face    face,
                       FT_Byte**  pointer,
                       FT_ULong   length )
  {
#ifndef FT_CONFIG_OPTION_INCREMENTAL
    FT_UNUSED( length );
#endif

#ifdef FT_CONFIG_OPTION_INCREMENTAL
    /* For incremental fonts get the character data using the */
    /* callback function.                                     */
    if ( face->root.internal->incremental_interface )
    {
      FT_Data  data;


      data.pointer = *pointer;
      data.length  = (FT_UInt)length;

      face->root.internal->incremental_interface->funcs->free_glyph_data(
        face->root.internal->incremental_interface->object, &data );
    }
    else
#endif /* FT_CONFIG_OPTION_INCREMENTAL */

    {
      CFF_Font  cff = (CFF_Font)( face->extra.data );


      cff_index_forget_element( &cff->charstrings_index, pointer );
    }
  }


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


#if 0 /* unused until we support pure CFF fonts */


  FT_LOCAL_DEF( FT_Error )
  cff_compute_max_advance( TT_Face  face,
                           FT_Int*  max_advance )
  {
    FT_Error     error = FT_Err_Ok;
    CFF_Decoder  decoder;
    FT_Int       glyph_index;
    CFF_Font     cff = (CFF_Font)face->other;

    PSAux_Service            psaux         = (PSAux_Service)face->psaux;
    const CFF_Decoder_Funcs  decoder_funcs = psaux->cff_decoder_funcs;


    *max_advance = 0;

    /* Initialize load decoder */
    decoder_funcs->init( &decoder, face, 0, 0, 0, 0, 0, 0 );

    decoder.builder.metrics_only = 1;
    decoder.builder.load_points  = 0;

    /* For each glyph, parse the glyph charstring and extract */
    /* the advance width.                                     */
    for ( glyph_index = 0; glyph_index < face->root.num_glyphs;
          glyph_index++ )
    {
      FT_Byte*  charstring;
      FT_ULong  charstring_len;


      /* now get load the unscaled outline */
      error = cff_get_glyph_data( face, glyph_index,
                                  &charstring, &charstring_len );
      if ( !error )
      {
        error = decoder_funcs->prepare( &decoder, size, glyph_index );
        if ( !error )
          error = decoder_funcs->parse_charstrings_old( &decoder,
                                                        charstring,
                                                        charstring_len,
                                                        0 );

        cff_free_glyph_data( face, &charstring, &charstring_len );
      }

      /* ignore the error if one has occurred -- skip to next glyph */
      error = FT_Err_Ok;
    }

    *max_advance = decoder.builder.advance.x;

    return FT_Err_Ok;
  }


#endif /* 0 */


  FT_LOCAL_DEF( FT_Error )
  cff_slot_load( CFF_GlyphSlot  glyph,
                 CFF_Size       size,
                 FT_UInt        glyph_index,
                 FT_Int32       load_flags )
  {
    FT_Error     error;
    CFF_Decoder  decoder;
    PS_Decoder   psdecoder;
    TT_Face      face = (TT_Face)glyph->root.face;
    FT_Bool      hinting, scaled, force_scaling;
    CFF_Font     cff  = (CFF_Font)face->extra.data;

    PSAux_Service            psaux         = (PSAux_Service)face->psaux;
    const CFF_Decoder_Funcs  decoder_funcs = psaux->cff_decoder_funcs;

    FT_Matrix  font_matrix;
    FT_Vector  font_offset;


    force_scaling = FALSE;

    /* in a CID-keyed font, consider `glyph_index' as a CID and map */
    /* it immediately to the real glyph_index -- if it isn't a      */
    /* subsetted font, glyph_indices and CIDs are identical, though */
    if ( cff->top_font.font_dict.cid_registry != 0xFFFFU &&
         cff->charset.cids                               )
    {
      /* don't handle CID 0 (.notdef) which is directly mapped to GID 0 */
      if ( glyph_index != 0 )
      {
        glyph_index = cff_charset_cid_to_gindex( &cff->charset,
                                                 glyph_index );
        if ( glyph_index == 0 )
          return FT_THROW( Invalid_Argument );
      }
    }
    else if ( glyph_index >= cff->num_glyphs )
      return FT_THROW( Invalid_Argument );

#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS

    /* try to load embedded bitmap if any              */
    /*                                                 */
    /* XXX: The convention should be emphasized in     */
    /*      the documents because it can be confusing. */
    {
      CFF_Face      cff_face = (CFF_Face)size->root.face;
      SFNT_Service  sfnt     = (SFNT_Service)cff_face->sfnt;
      FT_Stream     stream   = cff_face->root.stream;


      if ( size->strike_index != 0xFFFFFFFFUL      &&
           ( load_flags & FT_LOAD_NO_BITMAP ) == 0 &&
           IS_DEFAULT_INSTANCE( size->root.face )  )
      {
        TT_SBit_MetricsRec  metrics;


        error = sfnt->load_sbit_image( face,
                                       size->strike_index,
                                       glyph_index,
                                       (FT_UInt)load_flags,
                                       stream,
                                       &glyph->root.bitmap,
                                       &metrics );

        if ( !error )
        {
          FT_Bool    has_vertical_info;
          FT_UShort  advance;
          FT_Short   dummy;


          glyph->root.metrics.width  = (FT_Pos)metrics.width  * 64;
          glyph->root.metrics.height = (FT_Pos)metrics.height * 64;

          glyph->root.metrics.horiBearingX = (FT_Pos)metrics.horiBearingX * 64;
          glyph->root.metrics.horiBearingY = (FT_Pos)metrics.horiBearingY * 64;
          glyph->root.metrics.horiAdvance  = (FT_Pos)metrics.horiAdvance  * 64;

          glyph->root.metrics.vertBearingX = (FT_Pos)metrics.vertBearingX * 64;
          glyph->root.metrics.vertBearingY = (FT_Pos)metrics.vertBearingY * 64;
          glyph->root.metrics.vertAdvance  = (FT_Pos)metrics.vertAdvance  * 64;

          glyph->root.format = FT_GLYPH_FORMAT_BITMAP;

          if ( load_flags & FT_LOAD_VERTICAL_LAYOUT )
          {
            glyph->root.bitmap_left = metrics.vertBearingX;
            glyph->root.bitmap_top  = metrics.vertBearingY;
          }
          else
          {
            glyph->root.bitmap_left = metrics.horiBearingX;
            glyph->root.bitmap_top  = metrics.horiBearingY;
          }

          /* compute linear advance widths */

          (void)( (SFNT_Service)face->sfnt )->get_metrics( face, 0,
                                                           glyph_index,
                                                           &dummy,
                                                           &advance );
          glyph->root.linearHoriAdvance = advance;

          has_vertical_info = FT_BOOL(
                                face->vertical_info                   &&
                                face->vertical.number_Of_VMetrics > 0 );

          /* get the vertical metrics from the vmtx table if we have one */
          if ( has_vertical_info )
          {
            (void)( (SFNT_Service)face->sfnt )->get_metrics( face, 1,
                                                             glyph_index,
                                                             &dummy,
                                                             &advance );
            glyph->root.linearVertAdvance = advance;
          }
          else
          {
            /* make up vertical ones */
            if ( face->os2.version != 0xFFFFU )
              glyph->root.linearVertAdvance = (FT_Pos)
                ( face->os2.sTypoAscender - face->os2.sTypoDescender );
            else
              glyph->root.linearVertAdvance = (FT_Pos)
                ( face->horizontal.Ascender - face->horizontal.Descender );
          }

          return error;
        }
      }
    }

#endif /* TT_CONFIG_OPTION_EMBEDDED_BITMAPS */

    /* return immediately if we only want the embedded bitmaps */
    if ( load_flags & FT_LOAD_SBITS_ONLY )
      return FT_THROW( Invalid_Argument );

#ifdef FT_CONFIG_OPTION_SVG
    /* check for OT-SVG */
    if ( ( load_flags & FT_LOAD_NO_SVG ) == 0 &&
         ( load_flags & FT_LOAD_COLOR )       &&
         face->svg                            )
    {
      /*
       * We load the SVG document and try to grab the advances from the
       * table.  For the bearings we rely on the presetting hook to do that.
       */

      SFNT_Service  sfnt = (SFNT_Service)face->sfnt;


      if ( size && (size->root.metrics.x_ppem < 1 ||
                    size->root.metrics.y_ppem < 1 ) )
      {
        error = FT_THROW( Invalid_Size_Handle );
        return error;
      }

      FT_TRACE3(( "Trying to load SVG glyph\n" ));

      error = sfnt->load_svg_doc( (FT_GlyphSlot)glyph, glyph_index );
      if ( !error )
      {
        FT_Fixed  x_scale = size->root.metrics.x_scale;
        FT_Fixed  y_scale = size->root.metrics.y_scale;

        FT_Short   dummy;
        FT_UShort  advanceX;
        FT_UShort  advanceY;


        FT_TRACE3(( "Successfully loaded SVG glyph\n" ));

        glyph->root.format = FT_GLYPH_FORMAT_SVG;

        /*
         * If horizontal or vertical advances are not present in the table,
         * this is a problem with the font since the standard requires them.
         * However, we are graceful and calculate the values by ourselves
         * for the vertical case.
         */
        sfnt->get_metrics( face,
                           FALSE,
                           glyph_index,
                           &dummy,
                           &advanceX );
        sfnt->get_metrics( face,
                           TRUE,
                           glyph_index,
                           &dummy,
                           &advanceY );

        glyph->root.linearHoriAdvance = advanceX;
        glyph->root.linearVertAdvance = advanceY;

        glyph->root.metrics.horiAdvance = FT_MulFix( advanceX, x_scale );
        glyph->root.metrics.vertAdvance = FT_MulFix( advanceY, y_scale );

        return error;
      }

      FT_TRACE3(( "Failed to load SVG glyph\n" ));
    }

#endif /* FT_CONFIG_OPTION_SVG */

    /* top-level code ensures that FT_LOAD_NO_HINTING is set */
    /* if FT_LOAD_NO_SCALE is active                         */
    hinting = FT_BOOL( ( load_flags & FT_LOAD_NO_HINTING ) == 0 );
    scaled  = FT_BOOL( ( load_flags & FT_LOAD_NO_SCALE   ) == 0 );

    glyph->hint        = hinting;
    glyph->scaled      = scaled;

    if ( scaled )
    {
      glyph->x_scale = size->root.metrics.x_scale;
      glyph->y_scale = size->root.metrics.y_scale;
    }
    else
    {
      glyph->x_scale = 0x10000L;
      glyph->y_scale = 0x10000L;
    }

    /* if we have a CID subfont, use its matrix (which has already */
    /* been multiplied with the root matrix)                       */

    /* this scaling is only relevant if the PS hinter isn't active */
    if ( cff->num_subfonts )
    {
      FT_Long  top_upm, sub_upm;
      FT_Byte  fd_index = cff_fd_select_get( &cff->fd_select,
                                             glyph_index );


      if ( fd_index >= cff->num_subfonts )
        fd_index = (FT_Byte)( cff->num_subfonts - 1 );

      top_upm = (FT_Long)cff->top_font.font_dict.units_per_em;
      sub_upm = (FT_Long)cff->subfonts[fd_index]->font_dict.units_per_em;

      font_matrix = cff->subfonts[fd_index]->font_dict.font_matrix;
      font_offset = cff->subfonts[fd_index]->font_dict.font_offset;

      if ( top_upm != sub_upm )
      {
        glyph->x_scale = FT_MulDiv( glyph->x_scale, top_upm, sub_upm );
        glyph->y_scale = FT_MulDiv( glyph->y_scale, top_upm, sub_upm );

        force_scaling = TRUE;
      }
    }
    else
    {
      font_matrix = cff->top_font.font_dict.font_matrix;
      font_offset = cff->top_font.font_dict.font_offset;
    }

    {
#ifdef CFF_CONFIG_OPTION_OLD_ENGINE
      PS_Driver  driver = (PS_Driver)FT_FACE_DRIVER( face );
#endif

      FT_Byte*  charstring;
      FT_ULong  charstring_len;


      decoder_funcs->init( &decoder, face, size, glyph, hinting,
                           FT_LOAD_TARGET_MODE( load_flags ),
                           cff_get_glyph_data,
                           cff_free_glyph_data );

      /* this is for pure CFFs */
      if ( load_flags & FT_LOAD_ADVANCE_ONLY )
        decoder.width_only = TRUE;

      decoder.builder.no_recurse =
        FT_BOOL( load_flags & FT_LOAD_NO_RECURSE );

      /* this function also checks for a valid subfont index */
      error = decoder_funcs->prepare( &decoder, size, glyph_index );
      if ( error )
        goto Glyph_Build_Finished;

      /* now load the unscaled outline */
      error = cff_get_glyph_data( face, glyph_index,
                                  &charstring, &charstring_len );
      if ( error )
        goto Glyph_Build_Finished;

#ifdef CFF_CONFIG_OPTION_OLD_ENGINE
      /* choose which CFF renderer to use */
      if ( driver->hinting_engine == FT_HINTING_FREETYPE )
        error = decoder_funcs->parse_charstrings_old( &decoder,
                                                      charstring,
                                                      charstring_len,
                                                      0 );
      else
#endif
      {
        psaux->ps_decoder_init( &psdecoder, &decoder, FALSE );

        error = decoder_funcs->parse_charstrings( &psdecoder,
                                                  charstring,
                                                  charstring_len );

        /* Adobe's engine uses 16.16 numbers everywhere;              */
        /* as a consequence, glyphs larger than 2000ppem get rejected */
        if ( FT_ERR_EQ( error, Glyph_Too_Big ) )
        {
          /* this time, we retry unhinted and scale up the glyph later on */
          /* (the engine uses and sets the hardcoded value 0x10000 / 64 = */
          /* 0x400 for both `x_scale' and `y_scale' in this case)         */
          hinting       = FALSE;
          force_scaling = TRUE;
          glyph->hint   = hinting;

          error = decoder_funcs->parse_charstrings( &psdecoder,
                                                    charstring,
                                                    charstring_len );
        }
      }

      cff_free_glyph_data( face, &charstring, charstring_len );

      if ( error )
        goto Glyph_Build_Finished;

#ifdef FT_CONFIG_OPTION_INCREMENTAL
      /* Control data and length may not be available for incremental */
      /* fonts.                                                       */
      if ( face->root.internal->incremental_interface )
      {
        glyph->root.control_data = NULL;
        glyph->root.control_len = 0;
      }
      else
#endif /* FT_CONFIG_OPTION_INCREMENTAL */

      /* We set control_data and control_len if charstrings is loaded. */
      /* See how charstring loads at cff_index_access_element() in     */
      /* cffload.c.                                                    */
      {
        CFF_Index  csindex = &cff->charstrings_index;


        if ( csindex->offsets )
        {
          glyph->root.control_data = csindex->bytes +
                                     csindex->offsets[glyph_index] - 1;
          glyph->root.control_len  = (FT_Long)charstring_len;
        }
      }

  Glyph_Build_Finished:
      /* save new glyph tables, if no error */
      if ( !error )
        decoder.builder.funcs.done( &decoder.builder );
      /* XXX: anything to do for broken glyph entry? */
    }

#ifdef FT_CONFIG_OPTION_INCREMENTAL

    /* Incremental fonts can optionally override the metrics. */
    if ( !error                                                               &&
         face->root.internal->incremental_interface                           &&
         face->root.internal->incremental_interface->funcs->get_glyph_metrics )
    {
      FT_Incremental_MetricsRec  metrics;


      metrics.bearing_x = decoder.builder.left_bearing.x;
      metrics.bearing_y = 0;
      metrics.advance   = decoder.builder.advance.x;
      metrics.advance_v = decoder.builder.advance.y;

      error = face->root.internal->incremental_interface->funcs->get_glyph_metrics(
                face->root.internal->incremental_interface->object,
                glyph_index, FALSE, &metrics );

      decoder.builder.left_bearing.x = metrics.bearing_x;
      decoder.builder.advance.x      = metrics.advance;
      decoder.builder.advance.y      = metrics.advance_v;
    }

#endif /* FT_CONFIG_OPTION_INCREMENTAL */

    if ( !error )
    {
      /* Now, set the metrics -- this is rather simple, as   */
      /* the left side bearing is the xMin, and the top side */
      /* bearing the yMax. For composite glyphs, return only */
      /* left side bearing and advance width.                */
      if ( load_flags & FT_LOAD_NO_RECURSE )
      {
        FT_Slot_Internal  internal = glyph->root.internal;


        glyph->root.metrics.horiBearingX = decoder.builder.left_bearing.x;
        glyph->root.metrics.horiAdvance  = decoder.glyph_width;
        internal->glyph_matrix           = font_matrix;
        internal->glyph_delta            = font_offset;
        internal->glyph_transformed      = 1;
      }
      else
      {
        FT_BBox            cbox;
        FT_Glyph_Metrics*  metrics = &glyph->root.metrics;
        FT_Bool            has_vertical_info;


        glyph->root.format = FT_GLYPH_FORMAT_OUTLINE;

        glyph->root.outline.flags = FT_OUTLINE_REVERSE_FILL;
        if ( size && size->root.metrics.y_ppem < 24 )
          glyph->root.outline.flags |= FT_OUTLINE_HIGH_PRECISION;

        if ( face->horizontal.number_Of_HMetrics )
        {
          FT_Short   horiBearingX = 0;
          FT_UShort  horiAdvance  = 0;


          ( (SFNT_Service)face->sfnt )->get_metrics( face, 0,
                                                     glyph_index,
                                                     &horiBearingX,
                                                     &horiAdvance );
          metrics->horiAdvance          = horiAdvance;
          metrics->horiBearingX         = horiBearingX;
          glyph->root.linearHoriAdvance = horiAdvance;
        }
        else
        {
          /* copy the _unscaled_ advance width */
          metrics->horiAdvance          = decoder.glyph_width;
          glyph->root.linearHoriAdvance = decoder.glyph_width;
        }

        glyph->root.internal->glyph_transformed = 0;

        has_vertical_info = FT_BOOL( face->vertical_info                   &&
                                     face->vertical.number_Of_VMetrics > 0 );

        /* get the vertical metrics from the vmtx table if we have one */
        if ( has_vertical_info )
        {
          FT_Short   vertBearingY = 0;
          FT_UShort  vertAdvance  = 0;


          ( (SFNT_Service)face->sfnt )->get_metrics( face, 1,
                                                     glyph_index,
                                                     &vertBearingY,
                                                     &vertAdvance );
          metrics->vertBearingY = vertBearingY;
          metrics->vertAdvance  = vertAdvance;
        }
        else
        {
          /* make up vertical ones */
          if ( face->os2.version != 0xFFFFU )
            metrics->vertAdvance = (FT_Pos)( face->os2.sTypoAscender -
                                             face->os2.sTypoDescender );
          else
            metrics->vertAdvance = (FT_Pos)( face->horizontal.Ascender -
                                             face->horizontal.Descender );
        }

        glyph->root.linearVertAdvance = metrics->vertAdvance;

        /* apply the font matrix, if any */
        if ( font_matrix.xx != 0x10000L || font_matrix.yy != 0x10000L ||
             font_matrix.xy != 0        || font_matrix.yx != 0        )
        {
          FT_Outline_Transform( &glyph->root.outline, &font_matrix );

          metrics->horiAdvance = FT_MulFix( metrics->horiAdvance,
                                            font_matrix.xx );
          metrics->vertAdvance = FT_MulFix( metrics->vertAdvance,
                                            font_matrix.yy );
        }

        if ( font_offset.x || font_offset.y )
        {
          FT_Outline_Translate( &glyph->root.outline,
                                font_offset.x,
                                font_offset.y );

          metrics->horiAdvance += font_offset.x;
          metrics->vertAdvance += font_offset.y;
        }

        if ( scaled || force_scaling )
        {
          /* scale the outline and the metrics */
          FT_Int       n;
          FT_Outline*  cur     = &glyph->root.outline;
          FT_Vector*   vec     = cur->points;
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
        FT_Outline_Get_CBox( &glyph->root.outline, &cbox );

        metrics->width  = cbox.xMax - cbox.xMin;
        metrics->height = cbox.yMax - cbox.yMin;

        metrics->horiBearingX = cbox.xMin;
        metrics->horiBearingY = cbox.yMax;

        if ( has_vertical_info )
        {
          metrics->vertBearingX = metrics->horiBearingX -
                                    metrics->horiAdvance / 2;
          metrics->vertBearingY = FT_MulFix( metrics->vertBearingY,
                                             glyph->y_scale );
        }
        else
        {
          if ( load_flags & FT_LOAD_VERTICAL_LAYOUT )
            ft_synthesize_vertical_metrics( metrics,
                                            metrics->vertAdvance );
        }
      }
    }

    return error;
  }


/* END */

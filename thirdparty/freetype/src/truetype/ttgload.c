/****************************************************************************
 *
 * ttgload.c
 *
 *   TrueType Glyph Loader (body).
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


#include <ft2build.h>
#include <freetype/internal/ftdebug.h>
#include FT_CONFIG_CONFIG_H
#include <freetype/internal/ftcalc.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/sfnt.h>
#include <freetype/tttags.h>
#include <freetype/ftoutln.h>
#include <freetype/ftdriver.h>
#include <freetype/ftlist.h>

#include "ttgload.h"
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
#define FT_COMPONENT  ttgload


  /**************************************************************************
   *
   * Simple glyph flags.
   */
#define ON_CURVE_POINT  0x01  /* same value as FT_CURVE_TAG_ON            */
#define X_SHORT_VECTOR  0x02
#define Y_SHORT_VECTOR  0x04
#define REPEAT_FLAG     0x08
#define X_POSITIVE      0x10  /* two meanings depending on X_SHORT_VECTOR */
#define SAME_X          0x10
#define Y_POSITIVE      0x20  /* two meanings depending on Y_SHORT_VECTOR */
#define SAME_Y          0x20
#define OVERLAP_SIMPLE  0x40  /* retained as FT_OUTLINE_OVERLAP           */


  /**************************************************************************
   *
   * Composite glyph flags.
   */
#define ARGS_ARE_WORDS             0x0001
#define ARGS_ARE_XY_VALUES         0x0002
#define ROUND_XY_TO_GRID           0x0004
#define WE_HAVE_A_SCALE            0x0008
/* reserved                        0x0010 */
#define MORE_COMPONENTS            0x0020
#define WE_HAVE_AN_XY_SCALE        0x0040
#define WE_HAVE_A_2X2              0x0080
#define WE_HAVE_INSTR              0x0100
#define USE_MY_METRICS             0x0200
#define OVERLAP_COMPOUND           0x0400  /* retained as FT_OUTLINE_OVERLAP */
#define SCALED_COMPONENT_OFFSET    0x0800
#define UNSCALED_COMPONENT_OFFSET  0x1000


#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#define IS_DEFAULT_INSTANCE( _face )             \
          ( !( FT_IS_NAMED_INSTANCE( _face ) ||  \
               FT_IS_VARIATION( _face )      ) )
#else
#define IS_DEFAULT_INSTANCE( _face )  1
#endif


  /**************************************************************************
   *
   * Return the horizontal metrics in font units for a given glyph.
   */
  FT_LOCAL_DEF( void )
  TT_Get_HMetrics( TT_Face     face,
                   FT_UInt     idx,
                   FT_Short*   lsb,
                   FT_UShort*  aw )
  {
    ( (SFNT_Service)face->sfnt )->get_metrics( face, 0, idx, lsb, aw );

    FT_TRACE5(( "  advance width (font units): %d\n", *aw ));
    FT_TRACE5(( "  left side bearing (font units): %d\n", *lsb ));
  }


  /**************************************************************************
   *
   * Return the vertical metrics in font units for a given glyph.
   * See function `tt_loader_set_pp' below for explanations.
   */
  FT_LOCAL_DEF( void )
  TT_Get_VMetrics( TT_Face     face,
                   FT_UInt     idx,
                   FT_Pos      yMax,
                   FT_Short*   tsb,
                   FT_UShort*  ah )
  {
    if ( face->vertical_info )
      ( (SFNT_Service)face->sfnt )->get_metrics( face, 1, idx, tsb, ah );

    else if ( face->os2.version != 0xFFFFU )
    {
      *tsb = (FT_Short)( face->os2.sTypoAscender - yMax );
      *ah  = (FT_UShort)FT_ABS( face->os2.sTypoAscender -
                                face->os2.sTypoDescender );
    }

    else
    {
      *tsb = (FT_Short)( face->horizontal.Ascender - yMax );
      *ah  = (FT_UShort)FT_ABS( face->horizontal.Ascender -
                                face->horizontal.Descender );
    }

#ifdef FT_DEBUG_LEVEL_TRACE
    if ( !face->vertical_info )
      FT_TRACE5(( "  [vertical metrics missing, computing values]\n" ));
#endif

    FT_TRACE5(( "  advance height (font units): %d\n", *ah ));
    FT_TRACE5(( "  top side bearing (font units): %d\n", *tsb ));
  }


  static FT_Error
  tt_get_metrics( TT_Loader  loader,
                  FT_UInt    glyph_index )
  {
    TT_Face    face   = loader->face;

    FT_Error   error;
    FT_Stream  stream = loader->stream;

    FT_Short   left_bearing = 0, top_bearing = 0;
    FT_UShort  advance_width = 0, advance_height = 0;

    /* we must preserve the stream position          */
    /* (which gets altered by the metrics functions) */
    FT_ULong  pos = FT_STREAM_POS();


    TT_Get_HMetrics( face, glyph_index,
                     &left_bearing,
                     &advance_width );
    TT_Get_VMetrics( face, glyph_index,
                     loader->bbox.yMax,
                     &top_bearing,
                     &advance_height );

    if ( FT_STREAM_SEEK( pos ) )
      return error;

    loader->left_bearing = left_bearing;
    loader->advance      = advance_width;
    loader->top_bearing  = top_bearing;
    loader->vadvance     = advance_height;

#ifdef FT_CONFIG_OPTION_INCREMENTAL
    /* With the incremental interface, these values are set by  */
    /* a call to `tt_get_metrics_incremental'.                  */
    if ( face->root.internal->incremental_interface == NULL )
#endif
    {
      if ( !loader->linear_def )
      {
        loader->linear_def = 1;
        loader->linear     = advance_width;
      }
    }

    return FT_Err_Ok;
  }


#ifdef FT_CONFIG_OPTION_INCREMENTAL

  static void
  tt_get_metrics_incremental( TT_Loader  loader,
                              FT_UInt    glyph_index )
  {
    TT_Face  face = loader->face;

    FT_Short   left_bearing = 0, top_bearing = 0;
    FT_UShort  advance_width = 0, advance_height = 0;


    /* If this is an incrementally loaded font check whether there are */
    /* overriding metrics for this glyph.                              */
    if ( face->root.internal->incremental_interface                           &&
         face->root.internal->incremental_interface->funcs->get_glyph_metrics )
    {
      FT_Incremental_MetricsRec  incr_metrics;
      FT_Error                   error;


      incr_metrics.bearing_x = loader->left_bearing;
      incr_metrics.bearing_y = 0;
      incr_metrics.advance   = loader->advance;
      incr_metrics.advance_v = 0;

      error = face->root.internal->incremental_interface->funcs->get_glyph_metrics(
                face->root.internal->incremental_interface->object,
                glyph_index, FALSE, &incr_metrics );
      if ( error )
        goto Exit;

      left_bearing  = (FT_Short)incr_metrics.bearing_x;
      advance_width = (FT_UShort)incr_metrics.advance;

#if 0

      /* GWW: Do I do the same for vertical metrics? */
      incr_metrics.bearing_x = 0;
      incr_metrics.bearing_y = loader->top_bearing;
      incr_metrics.advance   = loader->vadvance;

      error = face->root.internal->incremental_interface->funcs->get_glyph_metrics(
                face->root.internal->incremental_interface->object,
                glyph_index, TRUE, &incr_metrics );
      if ( error )
        goto Exit;

      top_bearing    = (FT_Short)incr_metrics.bearing_y;
      advance_height = (FT_UShort)incr_metrics.advance;

#endif /* 0 */

      loader->left_bearing = left_bearing;
      loader->advance      = advance_width;
      loader->top_bearing  = top_bearing;
      loader->vadvance     = advance_height;

      if ( !loader->linear_def )
      {
        loader->linear_def = 1;
        loader->linear     = advance_width;
      }
    }

  Exit:
    return;
  }

#endif /* FT_CONFIG_OPTION_INCREMENTAL */


  /**************************************************************************
   *
   * The following functions are used by default with TrueType fonts.
   * However, they can be replaced by alternatives if we need to support
   * TrueType-compressed formats (like MicroType) in the future.
   *
   */

  FT_CALLBACK_DEF( FT_Error )
  TT_Access_Glyph_Frame( TT_Loader  loader,
                         FT_UInt    glyph_index,
                         FT_ULong   offset,
                         FT_UInt    byte_count )
  {
    FT_Error   error;
    FT_Stream  stream = loader->stream;

    FT_UNUSED( glyph_index );


    /* the following line sets the `error' variable through macros! */
    if ( FT_STREAM_SEEK( offset ) || FT_FRAME_ENTER( byte_count ) )
      return error;

    loader->cursor = stream->cursor;
    loader->limit  = stream->limit;

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( void )
  TT_Forget_Glyph_Frame( TT_Loader  loader )
  {
    FT_Stream  stream = loader->stream;


    FT_FRAME_EXIT();
  }


  FT_CALLBACK_DEF( FT_Error )
  TT_Load_Glyph_Header( TT_Loader  loader )
  {
    FT_Byte*  p     = loader->cursor;
    FT_Byte*  limit = loader->limit;


    if ( p + 10 > limit )
      return FT_THROW( Invalid_Outline );

    loader->n_contours = FT_NEXT_SHORT( p );

    loader->bbox.xMin = FT_NEXT_SHORT( p );
    loader->bbox.yMin = FT_NEXT_SHORT( p );
    loader->bbox.xMax = FT_NEXT_SHORT( p );
    loader->bbox.yMax = FT_NEXT_SHORT( p );

    FT_TRACE5(( "  # of contours: %d\n", loader->n_contours ));
    FT_TRACE5(( "  xMin: %4ld  xMax: %4ld\n", loader->bbox.xMin,
                                            loader->bbox.xMax ));
    FT_TRACE5(( "  yMin: %4ld  yMax: %4ld\n", loader->bbox.yMin,
                                            loader->bbox.yMax ));
    loader->cursor = p;

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( FT_Error )
  TT_Load_Simple_Glyph( TT_Loader  load )
  {
    FT_Error        error;
    FT_Byte*        p          = load->cursor;
    FT_Byte*        limit      = load->limit;
    FT_GlyphLoader  gloader    = load->gloader;
    FT_Outline*     outline    = &gloader->current.outline;
    FT_Int          n_contours = load->n_contours;
    FT_Int          n_points;
    FT_UShort       n_ins;

    FT_Byte         *flag, *flag_limit;
    FT_Byte         c, count;
    FT_Vector       *vec, *vec_limit;
    FT_Pos          x, y;
    FT_UShort       *cont, *cont_limit;
    FT_Int          last;


    /* check that we can add the contours to the glyph */
    error = FT_GLYPHLOADER_CHECK_POINTS( gloader, 0, n_contours );
    if ( error )
      goto Fail;

    /* check space for contours array + instructions count */
    if ( n_contours >= 0xFFF || p + 2 * n_contours + 2 > limit )
      goto Invalid_Outline;

    /* reading the contours' endpoints & number of points */
    cont       = outline->contours;
    cont_limit = cont + n_contours;

    last = -1;
    for ( ; cont < cont_limit; cont++ )
    {
      *cont = FT_NEXT_USHORT( p );

      if ( *cont <= last )
        goto Invalid_Outline;

      last = *cont;
    }

    n_points = last + 1;

    FT_TRACE5(( "  # of points: %d\n", n_points ));

    /* note that we will add four phantom points later */
    error = FT_GLYPHLOADER_CHECK_POINTS( gloader, n_points + 4, 0 );
    if ( error )
      goto Fail;

    /* space checked above */
    n_ins = FT_NEXT_USHORT( p );

    FT_TRACE5(( "  Instructions size: %u\n", n_ins ));

    /* check instructions size */
    if ( p + n_ins > limit )
    {
      FT_TRACE1(( "TT_Load_Simple_Glyph: excessive instruction count\n" ));
      error = FT_THROW( Too_Many_Hints );
      goto Fail;
    }

#ifdef TT_USE_BYTECODE_INTERPRETER

    if ( IS_HINTED( load->load_flags ) )
    {
      TT_ExecContext  exec = load->exec;
      FT_Memory       memory = exec->memory;


      if ( exec->glyphSize )
        FT_FREE( exec->glyphIns );
      exec->glyphSize = 0;

      /* we don't trust `maxSizeOfInstructions' in the `maxp' table */
      /* and thus allocate the bytecode array size by ourselves     */
      if ( n_ins )
      {
        if ( FT_DUP( exec->glyphIns, p, n_ins ) )
          return error;

        exec->glyphSize  = n_ins;
      }
    }

#endif /* TT_USE_BYTECODE_INTERPRETER */

    p += n_ins;

    /* reading the point tags */
    flag       = outline->tags;
    flag_limit = flag + n_points;

    FT_ASSERT( flag );

    while ( flag < flag_limit )
    {
      if ( p + 1 > limit )
        goto Invalid_Outline;

      *flag++ = c = FT_NEXT_BYTE( p );
      if ( c & REPEAT_FLAG )
      {
        if ( p + 1 > limit )
          goto Invalid_Outline;

        count = FT_NEXT_BYTE( p );
        if ( flag + (FT_Int)count > flag_limit )
          goto Invalid_Outline;

        for ( ; count > 0; count-- )
          *flag++ = c;
      }
    }

    /* retain the overlap flag */
    if ( n_points && outline->tags[0] & OVERLAP_SIMPLE )
      gloader->base.outline.flags |= FT_OUTLINE_OVERLAP;

    /* reading the X coordinates */

    vec       = outline->points;
    vec_limit = vec + n_points;
    flag      = outline->tags;
    x         = 0;

    for ( ; vec < vec_limit; vec++, flag++ )
    {
      FT_Pos   delta = 0;
      FT_Byte  f     = *flag;


      if ( f & X_SHORT_VECTOR )
      {
        if ( p + 1 > limit )
          goto Invalid_Outline;

        delta = (FT_Pos)FT_NEXT_BYTE( p );
        if ( !( f & X_POSITIVE ) )
          delta = -delta;
      }
      else if ( !( f & SAME_X ) )
      {
        if ( p + 2 > limit )
          goto Invalid_Outline;

        delta = (FT_Pos)FT_NEXT_SHORT( p );
      }

      x     += delta;
      vec->x = x;
    }

    /* reading the Y coordinates */

    vec       = outline->points;
    vec_limit = vec + n_points;
    flag      = outline->tags;
    y         = 0;

    for ( ; vec < vec_limit; vec++, flag++ )
    {
      FT_Pos   delta = 0;
      FT_Byte  f     = *flag;


      if ( f & Y_SHORT_VECTOR )
      {
        if ( p + 1 > limit )
          goto Invalid_Outline;

        delta = (FT_Pos)FT_NEXT_BYTE( p );
        if ( !( f & Y_POSITIVE ) )
          delta = -delta;
      }
      else if ( !( f & SAME_Y ) )
      {
        if ( p + 2 > limit )
          goto Invalid_Outline;

        delta = (FT_Pos)FT_NEXT_SHORT( p );
      }

      y     += delta;
      vec->y = y;

      /* the cast is for stupid compilers */
      *flag  = (FT_Byte)( f & ON_CURVE_POINT );
    }

    outline->n_points   = (FT_UShort)n_points;
    outline->n_contours = (FT_UShort)n_contours;

    load->cursor = p;

  Fail:
    return error;

  Invalid_Outline:
    error = FT_THROW( Invalid_Outline );
    goto Fail;
  }


  FT_CALLBACK_DEF( FT_Error )
  TT_Load_Composite_Glyph( TT_Loader  loader )
  {
    FT_Error        error;
    FT_Byte*        p          = loader->cursor;
    FT_Byte*        limit      = loader->limit;
    FT_GlyphLoader  gloader    = loader->gloader;
    FT_Long         num_glyphs = loader->face->root.num_glyphs;
    FT_SubGlyph     subglyph;
    FT_UInt         num_subglyphs;


    num_subglyphs = 0;

    do
    {
      FT_Fixed  xx, xy, yy, yx;
      FT_UInt   count;


      /* check that we can load a new subglyph */
      error = FT_GlyphLoader_CheckSubGlyphs( gloader, num_subglyphs + 1 );
      if ( error )
        goto Fail;

      /* check space */
      if ( p + 4 > limit )
        goto Invalid_Composite;

      subglyph = gloader->current.subglyphs + num_subglyphs;

      subglyph->arg1 = subglyph->arg2 = 0;

      subglyph->flags = FT_NEXT_USHORT( p );
      subglyph->index = FT_NEXT_USHORT( p );

      /* we reject composites that have components */
      /* with invalid glyph indices                */
      if ( subglyph->index >= num_glyphs )
        goto Invalid_Composite;

      /* check space */
      count = 2;
      if ( subglyph->flags & ARGS_ARE_WORDS )
        count += 2;
      if ( subglyph->flags & WE_HAVE_A_SCALE )
        count += 2;
      else if ( subglyph->flags & WE_HAVE_AN_XY_SCALE )
        count += 4;
      else if ( subglyph->flags & WE_HAVE_A_2X2 )
        count += 8;

      if ( p + count > limit )
        goto Invalid_Composite;

      /* read arguments */
      if ( subglyph->flags & ARGS_ARE_XY_VALUES )
      {
        if ( subglyph->flags & ARGS_ARE_WORDS )
        {
          subglyph->arg1 = FT_NEXT_SHORT( p );
          subglyph->arg2 = FT_NEXT_SHORT( p );
        }
        else
        {
          subglyph->arg1 = FT_NEXT_CHAR( p );
          subglyph->arg2 = FT_NEXT_CHAR( p );
        }
      }
      else
      {
        if ( subglyph->flags & ARGS_ARE_WORDS )
        {
          subglyph->arg1 = (FT_Int)FT_NEXT_USHORT( p );
          subglyph->arg2 = (FT_Int)FT_NEXT_USHORT( p );
        }
        else
        {
          subglyph->arg1 = (FT_Int)FT_NEXT_BYTE( p );
          subglyph->arg2 = (FT_Int)FT_NEXT_BYTE( p );
        }
      }

      /* read transform */
      xx = yy = 0x10000L;
      xy = yx = 0;

      if ( subglyph->flags & WE_HAVE_A_SCALE )
      {
        xx = (FT_Fixed)FT_NEXT_SHORT( p ) * 4;
        yy = xx;
      }
      else if ( subglyph->flags & WE_HAVE_AN_XY_SCALE )
      {
        xx = (FT_Fixed)FT_NEXT_SHORT( p ) * 4;
        yy = (FT_Fixed)FT_NEXT_SHORT( p ) * 4;
      }
      else if ( subglyph->flags & WE_HAVE_A_2X2 )
      {
        xx = (FT_Fixed)FT_NEXT_SHORT( p ) * 4;
        yx = (FT_Fixed)FT_NEXT_SHORT( p ) * 4;
        xy = (FT_Fixed)FT_NEXT_SHORT( p ) * 4;
        yy = (FT_Fixed)FT_NEXT_SHORT( p ) * 4;
      }

      subglyph->transform.xx = xx;
      subglyph->transform.xy = xy;
      subglyph->transform.yx = yx;
      subglyph->transform.yy = yy;

      num_subglyphs++;

    } while ( subglyph->flags & MORE_COMPONENTS );

    gloader->current.num_subglyphs = num_subglyphs;
    FT_TRACE5(( "  %u component%s\n",
                num_subglyphs,
                num_subglyphs > 1 ? "s" : "" ));

#ifdef FT_DEBUG_LEVEL_TRACE
    {
      FT_UInt  i;


      subglyph = gloader->current.subglyphs;

      for ( i = 0; i < num_subglyphs; i++ )
      {
        if ( num_subglyphs > 1 )
          FT_TRACE7(( "    subglyph %u:\n", i ));

        FT_TRACE7(( "      glyph index: %d\n", subglyph->index ));

        if ( subglyph->flags & ARGS_ARE_XY_VALUES )
          FT_TRACE7(( "      offset: x=%d, y=%d\n",
                      subglyph->arg1,
                      subglyph->arg2 ));
        else
          FT_TRACE7(( "      matching points: base=%d, component=%d\n",
                      subglyph->arg1,
                      subglyph->arg2 ));

        if ( subglyph->flags & WE_HAVE_A_SCALE )
          FT_TRACE7(( "      scaling: %f\n",
                      (double)subglyph->transform.xx / 65536 ));
        else if ( subglyph->flags & WE_HAVE_AN_XY_SCALE )
          FT_TRACE7(( "      scaling: x=%f, y=%f\n",
                      (double)subglyph->transform.xx / 65536,
                      (double)subglyph->transform.yy / 65536 ));
        else if ( subglyph->flags & WE_HAVE_A_2X2 )
        {
          FT_TRACE7(( "      scaling: xx=%f, yx=%f\n",
                      (double)subglyph->transform.xx / 65536,
                      (double)subglyph->transform.yx / 65536 ));
          FT_TRACE7(( "               xy=%f, yy=%f\n",
                      (double)subglyph->transform.xy / 65536,
                      (double)subglyph->transform.yy / 65536 ));
        }

        subglyph++;
      }
    }
#endif /* FT_DEBUG_LEVEL_TRACE */

#ifdef TT_USE_BYTECODE_INTERPRETER

    {
      FT_Stream  stream = loader->stream;


      /* we must undo the FT_FRAME_ENTER in order to point */
      /* to the composite instructions, if we find some.   */
      /* We will process them later.                       */
      /*                                                   */
      loader->ins_pos = (FT_ULong)( FT_STREAM_POS() +
                                    p - limit );
    }

#endif

    loader->cursor = p;

  Fail:
    return error;

  Invalid_Composite:
    error = FT_THROW( Invalid_Composite );
    goto Fail;
  }


  FT_LOCAL_DEF( void )
  TT_Init_Glyph_Loading( TT_Face  face )
  {
    face->access_glyph_frame   = TT_Access_Glyph_Frame;
    face->read_glyph_header    = TT_Load_Glyph_Header;
    face->read_simple_glyph    = TT_Load_Simple_Glyph;
    face->read_composite_glyph = TT_Load_Composite_Glyph;
    face->forget_glyph_frame   = TT_Forget_Glyph_Frame;
  }


  static void
  tt_prepare_zone( TT_GlyphZone  zone,
                   FT_GlyphLoad  load,
                   FT_UInt       start_point,
                   FT_UInt       start_contour )
  {
    zone->n_points    = load->outline.n_points + 4 - (FT_UShort)start_point;
    zone->n_contours  = load->outline.n_contours - (FT_UShort)start_contour;
    zone->org         = load->extra_points + start_point;
    zone->cur         = load->outline.points + start_point;
    zone->orus        = load->extra_points2 + start_point;
    zone->tags        = load->outline.tags + start_point;
    zone->contours    = load->outline.contours + start_contour;
    zone->first_point = (FT_UShort)start_point;
  }


  /**************************************************************************
   *
   * @Function:
   *   TT_Hint_Glyph
   *
   * @Description:
   *   Hint the glyph using the zone prepared by the caller.  Note that
   *   the zone is supposed to include four phantom points.
   */
  static FT_Error
  TT_Hint_Glyph( TT_Loader  loader,
                 FT_Bool    is_composite )
  {
    TT_GlyphZone  zone = &loader->zone;

#ifdef TT_USE_BYTECODE_INTERPRETER
    TT_ExecContext  exec  = loader->exec;
    TT_Size         size  = loader->size;
    FT_Long         n_ins = exec->glyphSize;
#else
    FT_UNUSED( is_composite );
#endif


#ifdef TT_USE_BYTECODE_INTERPRETER
    /* save original point positions in `org' array */
    if ( n_ins > 0 )
      FT_ARRAY_COPY( zone->org, zone->cur, zone->n_points );

    /* XXX: UNDOCUMENTED! Hinting instructions of a composite glyph */
    /*      completely refer to the (already) hinted subglyphs.     */
    if ( is_composite )
    {
      exec->metrics.x_scale = 1 << 16;
      exec->metrics.y_scale = 1 << 16;

      FT_ARRAY_COPY( zone->orus, zone->cur, zone->n_points );
    }
    else
    {
      exec->metrics.x_scale = size->metrics->x_scale;
      exec->metrics.y_scale = size->metrics->y_scale;
    }
#endif

    /* round phantom points */
    zone->cur[zone->n_points - 4].x =
      FT_PIX_ROUND( zone->cur[zone->n_points - 4].x );
    zone->cur[zone->n_points - 3].x =
      FT_PIX_ROUND( zone->cur[zone->n_points - 3].x );
    zone->cur[zone->n_points - 2].y =
      FT_PIX_ROUND( zone->cur[zone->n_points - 2].y );
    zone->cur[zone->n_points - 1].y =
      FT_PIX_ROUND( zone->cur[zone->n_points - 1].y );

#ifdef TT_USE_BYTECODE_INTERPRETER

    if ( n_ins > 0 )
    {
      FT_Error  error;


      TT_Set_CodeRange( exec, tt_coderange_glyph, exec->glyphIns, n_ins );

      exec->is_composite = is_composite;
      exec->pts          = *zone;

      error = TT_Run_Context( exec, size );
      if ( error && exec->pedantic_hinting )
        return error;

      /* store drop-out mode in bits 5-7; set bit 2 also as a marker */
      loader->gloader->current.outline.tags[0] |=
        ( exec->GS.scan_type << 5 ) | FT_CURVE_TAG_HAS_SCANMODE;
    }

#endif

    /* Save possibly modified glyph phantom points unless in v40 backward  */
    /* compatibility mode, where no movement on the x axis means no reason */
    /* to change bearings or advance widths.                               */

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
    if ( exec->backward_compatibility )
      return FT_Err_Ok;
#endif

    loader->pp1 = zone->cur[zone->n_points - 4];
    loader->pp2 = zone->cur[zone->n_points - 3];
    loader->pp3 = zone->cur[zone->n_points - 2];
    loader->pp4 = zone->cur[zone->n_points - 1];

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   TT_Process_Simple_Glyph
   *
   * @Description:
   *   Once a simple glyph has been loaded, it needs to be processed.
   *   Usually, this means scaling and hinting through bytecode
   *   interpretation.
   */
  static FT_Error
  TT_Process_Simple_Glyph( TT_Loader  loader )
  {
    FT_Error        error    = FT_Err_Ok;
    FT_GlyphLoader  gloader  = loader->gloader;
    FT_Outline*     outline  = &gloader->current.outline;
    FT_Int          n_points = outline->n_points;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    FT_Memory   memory    = loader->face->root.memory;
    FT_Vector*  unrounded = NULL;
#endif


    /* set phantom points */
    outline->points[n_points    ] = loader->pp1;
    outline->points[n_points + 1] = loader->pp2;
    outline->points[n_points + 2] = loader->pp3;
    outline->points[n_points + 3] = loader->pp4;

    n_points += 4;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

    if ( !IS_DEFAULT_INSTANCE( FT_FACE( loader->face ) ) )
    {
      if ( FT_QNEW_ARRAY( unrounded, n_points ) )
        goto Exit;

      /* Deltas apply to the unscaled data. */
      error = TT_Vary_Apply_Glyph_Deltas( loader,
                                          outline,
                                          unrounded );
      if ( error )
        goto Exit;
    }

#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */

    if ( IS_HINTED( loader->load_flags ) )
    {
      tt_prepare_zone( &loader->zone, &gloader->current, 0, 0 );

      FT_ARRAY_COPY( loader->zone.orus, loader->zone.cur,
                     loader->zone.n_points );
    }

    {
      FT_Vector*  vec   = outline->points;
      FT_Vector*  limit = outline->points + n_points;

      FT_Fixed  x_scale = 0; /* pacify compiler */
      FT_Fixed  y_scale = 0;

      FT_Bool  do_scale = FALSE;


      {
        /* scale the glyph */
        if ( ( loader->load_flags & FT_LOAD_NO_SCALE ) == 0 )
        {
          x_scale = loader->size->metrics->x_scale;
          y_scale = loader->size->metrics->y_scale;

          do_scale = TRUE;
        }
      }

      if ( do_scale )
      {
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
        if ( !IS_DEFAULT_INSTANCE( FT_FACE( loader->face ) ) )
        {
          FT_Vector*  u = unrounded;


          for ( ; vec < limit; vec++, u++ )
          {
            vec->x = ADD_LONG( FT_MulFix( u->x, x_scale ), 32 ) >> 6;
            vec->y = ADD_LONG( FT_MulFix( u->y, y_scale ), 32 ) >> 6;
          }
        }
        else
#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */
        {
          for ( ; vec < limit; vec++ )
          {
            vec->x = FT_MulFix( vec->x, x_scale );
            vec->y = FT_MulFix( vec->y, y_scale );
          }
        }
      }

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      /* if we have a HVAR table, `pp1' and/or `pp2' */
      /* are already adjusted but unscaled           */
      if ( ( loader->face->variation_support & TT_FACE_FLAG_VAR_HADVANCE ) &&
           IS_HINTED( loader->load_flags )                                 )
      {
        loader->pp1.x = FT_MulFix( loader->pp1.x, x_scale );
        loader->pp2.x = FT_MulFix( loader->pp2.x, x_scale );
        /* pp1.y and pp2.y are always zero */
      }
      else
#endif
      {
        loader->pp1 = outline->points[n_points - 4];
        loader->pp2 = outline->points[n_points - 3];
      }

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      /* if we have a VVAR table, `pp3' and/or `pp4' */
      /* are already adjusted but unscaled           */
      if ( ( loader->face->variation_support & TT_FACE_FLAG_VAR_VADVANCE ) &&
           IS_HINTED( loader->load_flags )                                 )
      {
        loader->pp3.x = FT_MulFix( loader->pp3.x, x_scale );
        loader->pp3.y = FT_MulFix( loader->pp3.y, y_scale );
        loader->pp4.x = FT_MulFix( loader->pp4.x, x_scale );
        loader->pp4.y = FT_MulFix( loader->pp4.y, y_scale );
      }
      else
#endif
      {
        loader->pp3 = outline->points[n_points - 2];
        loader->pp4 = outline->points[n_points - 1];
      }
    }

    if ( IS_HINTED( loader->load_flags ) )
      error = TT_Hint_Glyph( loader, 0 );

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
  Exit:
    FT_FREE( unrounded );
#endif

    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   TT_Process_Composite_Component
   *
   * @Description:
   *   Once a composite component has been loaded, it needs to be
   *   processed.  Usually, this means transforming and translating.
   */
  static FT_Error
  TT_Process_Composite_Component( TT_Loader    loader,
                                  FT_SubGlyph  subglyph,
                                  FT_UInt      start_point,
                                  FT_UInt      num_base_points )
  {
    FT_GlyphLoader  gloader = loader->gloader;
    FT_Outline      current;
    FT_Bool         have_scale;
    FT_Pos          x, y;


    current.points   = gloader->base.outline.points +
                         num_base_points;
    current.n_points = gloader->base.outline.n_points -
                         (FT_UShort)num_base_points;

    have_scale = FT_BOOL( subglyph->flags & ( WE_HAVE_A_SCALE     |
                                              WE_HAVE_AN_XY_SCALE |
                                              WE_HAVE_A_2X2       ) );

    /* perform the transform required for this subglyph */
    if ( have_scale )
      FT_Outline_Transform( &current, &subglyph->transform );

    /* get offset */
    if ( !( subglyph->flags & ARGS_ARE_XY_VALUES ) )
    {
      FT_UInt     num_points = gloader->base.outline.n_points;
      FT_UInt     k = (FT_UInt)subglyph->arg1;
      FT_UInt     l = (FT_UInt)subglyph->arg2;
      FT_Vector*  p1;
      FT_Vector*  p2;


      /* match l-th point of the newly loaded component to the k-th point */
      /* of the previously loaded components.                             */

      /* change to the point numbers used by our outline */
      k += start_point;
      l += num_base_points;
      if ( k >= num_base_points ||
           l >= num_points      )
        return FT_THROW( Invalid_Composite );

      p1 = gloader->base.outline.points + k;
      p2 = gloader->base.outline.points + l;

      x = SUB_LONG( p1->x, p2->x );
      y = SUB_LONG( p1->y, p2->y );
    }
    else
    {
      x = subglyph->arg1;
      y = subglyph->arg2;

      if ( !x && !y )
        return FT_Err_Ok;

      /* Use a default value dependent on                                  */
      /* TT_CONFIG_OPTION_COMPONENT_OFFSET_SCALED.  This is useful for old */
      /* TT fonts which don't set the xxx_COMPONENT_OFFSET bit.            */

      if ( have_scale &&
#ifdef TT_CONFIG_OPTION_COMPONENT_OFFSET_SCALED
           !( subglyph->flags & UNSCALED_COMPONENT_OFFSET ) )
#else
            ( subglyph->flags & SCALED_COMPONENT_OFFSET ) )
#endif
      {

#if 0

        /********************************************************************
         *
         * This algorithm is what Apple documents.  But it doesn't work.
         */
        int  a = subglyph->transform.xx > 0 ?  subglyph->transform.xx
                                            : -subglyph->transform.xx;
        int  b = subglyph->transform.yx > 0 ?  subglyph->transform.yx
                                            : -subglyph->transform.yx;
        int  c = subglyph->transform.xy > 0 ?  subglyph->transform.xy
                                            : -subglyph->transform.xy;
        int  d = subglyph->transform.yy > 0 ?  subglyph->transform.yy
                                            : -subglyph->transform.yy;
        int  m = a > b ? a : b;
        int  n = c > d ? c : d;


        if ( a - b <= 33 && a - b >= -33 )
          m *= 2;
        if ( c - d <= 33 && c - d >= -33 )
          n *= 2;
        x = FT_MulFix( x, m );
        y = FT_MulFix( y, n );

#else /* 1 */

        /********************************************************************
         *
         * This algorithm is a guess and works much better than the above.
         */
        FT_Fixed  mac_xscale = FT_Hypot( subglyph->transform.xx,
                                         subglyph->transform.xy );
        FT_Fixed  mac_yscale = FT_Hypot( subglyph->transform.yy,
                                         subglyph->transform.yx );


        x = FT_MulFix( x, mac_xscale );
        y = FT_MulFix( y, mac_yscale );

#endif /* 1 */

      }

      if ( !( loader->load_flags & FT_LOAD_NO_SCALE ) )
      {
        FT_Fixed  x_scale = loader->size->metrics->x_scale;
        FT_Fixed  y_scale = loader->size->metrics->y_scale;


        x = FT_MulFix( x, x_scale );
        y = FT_MulFix( y, y_scale );

        if ( subglyph->flags & ROUND_XY_TO_GRID &&
             IS_HINTED( loader->load_flags )    )
        {
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
          if ( !loader->exec->backward_compatibility )
#endif
            x = FT_PIX_ROUND( x );

          y = FT_PIX_ROUND( y );
        }
      }
    }

    if ( x || y )
      FT_Outline_Translate( &current, x, y );

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   TT_Process_Composite_Glyph
   *
   * @Description:
   *   This is slightly different from TT_Process_Simple_Glyph, in that
   *   its sole purpose is to hint the glyph.  Thus this function is
   *   only available when bytecode interpreter is enabled.
   */
  static FT_Error
  TT_Process_Composite_Glyph( TT_Loader  loader,
                              FT_UInt    start_point,
                              FT_UInt    start_contour )
  {
    FT_Error     error;
    FT_Outline*  outline = &loader->gloader->base.outline;
    FT_UInt      i;


    /* make room for phantom points */
    error = FT_GLYPHLOADER_CHECK_POINTS( loader->gloader,
                                         outline->n_points + 4,
                                         0 );
    if ( error )
      return error;

    outline->points[outline->n_points    ] = loader->pp1;
    outline->points[outline->n_points + 1] = loader->pp2;
    outline->points[outline->n_points + 2] = loader->pp3;
    outline->points[outline->n_points + 3] = loader->pp4;

#ifdef TT_USE_BYTECODE_INTERPRETER

    {
      TT_ExecContext  exec   = loader->exec;
      FT_Memory       memory = exec->memory;
      FT_Stream       stream = loader->stream;
      FT_UShort       n_ins;


      if ( exec->glyphSize )
        FT_FREE( exec->glyphIns );
      exec->glyphSize = 0;

      /* TT_Load_Composite_Glyph only gives us the offset of instructions */
      /* so we read them here                                             */
      if ( FT_STREAM_SEEK( loader->ins_pos ) ||
           FT_READ_USHORT( n_ins )           )
        return error;

      FT_TRACE5(( "  Instructions size = %hu\n", n_ins ));

      if ( !n_ins )
        return FT_Err_Ok;

      /* don't trust `maxSizeOfInstructions'; */
      /* only do a rough safety check         */
      if ( n_ins > loader->byte_len )
      {
        FT_TRACE1(( "TT_Process_Composite_Glyph:"
                    " too many instructions (%hu) for glyph with length %u\n",
                    n_ins, loader->byte_len ));
        return FT_THROW( Too_Many_Hints );
      }

      if ( FT_QNEW_ARRAY( exec->glyphIns, n_ins )  ||
           FT_STREAM_READ( exec->glyphIns, n_ins ) )
        return error;

      exec->glyphSize = n_ins;
    }

#endif

    tt_prepare_zone( &loader->zone, &loader->gloader->base,
                     start_point, start_contour );

    /* Some points are likely touched during execution of  */
    /* instructions on components.  So let's untouch them. */
    for ( i = 0; i < loader->zone.n_points - 4U; i++ )
      loader->zone.tags[i] &= ~FT_CURVE_TAG_TOUCH_BOTH;

    return TT_Hint_Glyph( loader, 1 );
  }


  /*
   * Calculate the phantom points
   *
   * Defining the right side bearing (rsb) as
   *
   *   rsb = aw - (lsb + xmax - xmin)
   *
   * (with `aw' the advance width, `lsb' the left side bearing, and `xmin'
   * and `xmax' the glyph's minimum and maximum x value), the OpenType
   * specification defines the initial position of horizontal phantom points
   * as
   *
   *   pp1 = (round(xmin - lsb), 0)      ,
   *   pp2 = (round(pp1 + aw), 0)        .
   *
   * Note that the rounding to the grid (in the device space) is not
   * documented currently in the specification.
   *
   * However, the specification lacks the precise definition of vertical
   * phantom points.  Greg Hitchcock provided the following explanation.
   *
   * - a `vmtx' table is present
   *
   *   For any glyph, the minimum and maximum y values (`ymin' and `ymax')
   *   are given in the `glyf' table, the top side bearing (tsb) and advance
   *   height (ah) are given in the `vmtx' table.  The bottom side bearing
   *   (bsb) is then calculated as
   *
   *     bsb = ah - (tsb + ymax - ymin)       ,
   *
   *   and the initial position of vertical phantom points is
   *
   *     pp3 = (x, round(ymax + tsb))       ,
   *     pp4 = (x, round(pp3 - ah))         .
   *
   *   See below for value `x'.
   *
   * - no `vmtx' table in the font
   *
   *   If there is an `OS/2' table, we set
   *
   *     DefaultAscender = sTypoAscender       ,
   *     DefaultDescender = sTypoDescender     ,
   *
   *   otherwise we use data from the `hhea' table:
   *
   *     DefaultAscender = Ascender         ,
   *     DefaultDescender = Descender       .
   *
   *   With these two variables we can now set
   *
   *     ah = DefaultAscender - sDefaultDescender    ,
   *     tsb = DefaultAscender - yMax                ,
   *
   *   and proceed as if a `vmtx' table was present.
   *
   * Usually we have
   *
   *   x = aw / 2      ,                                                (1)
   *
   * but there is one compatibility case where it can be set to
   *
   *   x = -DefaultDescender -
   *         ((DefaultAscender - DefaultDescender - aw) / 2)     .      (2)
   *
   * and another one with
   *
   *   x = 0     .                                                      (3)
   *
   * In Windows, the history of those values is quite complicated,
   * depending on the hinting engine (that is, the graphics framework).
   *
   *   framework        from                 to       formula
   *  ----------------------------------------------------------
   *    GDI       Windows 98               current      (1)
   *              (Windows 2000 for NT)
   *    GDI+      Windows XP               Windows 7    (2)
   *    GDI+      Windows 8                current      (3)
   *    DWrite    Windows 7                current      (3)
   *
   * For simplicity, FreeType uses (1) for grayscale subpixel hinting and
   * (3) for everything else.
   *
   */
  static void
  tt_loader_set_pp( TT_Loader  loader )
  {
    loader->pp1.x = loader->bbox.xMin - loader->left_bearing;
    loader->pp1.y = 0;
    loader->pp2.x = loader->pp1.x + loader->advance;
    loader->pp2.y = 0;

    loader->pp3.x = 0;
    loader->pp3.y = loader->bbox.yMax + loader->top_bearing;
    loader->pp4.x = 0;
    loader->pp4.y = loader->pp3.y - loader->vadvance;

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
    {
      TT_Driver driver = (TT_Driver)FT_FACE_DRIVER( loader->face );


      if ( driver->interpreter_version == TT_INTERPRETER_VERSION_40 &&
           loader->exec                                             &&
           loader->exec->mode != FT_RENDER_MODE_MONO                &&
           loader->exec->mode != FT_RENDER_MODE_LCD                 &&
           loader->exec->mode != FT_RENDER_MODE_LCD_V               )
      {
        loader->pp3.x = loader->advance / 2;
        loader->pp4.x = loader->advance / 2;
      }
    }
#endif
  }


  /* a utility function to retrieve i-th node from given FT_List */
  static FT_ListNode
  ft_list_get_node_at( FT_List  list,
                       FT_UInt  idx )
  {
    FT_ListNode  cur;


    if ( !list )
      return NULL;

    for ( cur = list->head; cur; cur = cur->next )
    {
      if ( !idx )
        return cur;

      idx--;
    }

    return NULL;
  }


  /**************************************************************************
   *
   * @Function:
   *   load_truetype_glyph
   *
   * @Description:
   *   Loads a given truetype glyph.  Handles composites and uses a
   *   TT_Loader object.
   */
  static FT_Error
  load_truetype_glyph( TT_Loader  loader,
                       FT_UInt    glyph_index,
                       FT_UInt    recurse_count,
                       FT_Bool    header_only )
  {
    FT_Error        error   = FT_Err_Ok;
    FT_Fixed        x_scale, y_scale;
    FT_ULong        offset;
    TT_Face         face    = loader->face;
    FT_GlyphLoader  gloader = loader->gloader;

    FT_Bool  opened_frame = 0;

#ifdef FT_CONFIG_OPTION_INCREMENTAL
    FT_StreamRec    inc_stream;
    FT_Data         glyph_data;
    FT_Bool         glyph_data_loaded = 0;
#endif


#ifdef FT_DEBUG_LEVEL_TRACE
    if ( recurse_count )
      FT_TRACE5(( "  nesting level: %u\n", recurse_count ));
#endif

    /* some fonts have an incorrect value of `maxComponentDepth' */
    if ( recurse_count > face->max_profile.maxComponentDepth )
    {
      FT_TRACE1(( "load_truetype_glyph: maxComponentDepth set to %u\n",
                  recurse_count ));
      face->max_profile.maxComponentDepth = (FT_UShort)recurse_count;
    }

#ifndef FT_CONFIG_OPTION_INCREMENTAL
    /* check glyph index */
    if ( glyph_index >= (FT_UInt)face->root.num_glyphs )
    {
      error = FT_THROW( Invalid_Glyph_Index );
      goto Exit;
    }
#endif

    loader->glyph_index = glyph_index;

    if ( loader->load_flags & FT_LOAD_NO_SCALE )
    {
      x_scale = 0x10000L;
      y_scale = 0x10000L;
    }
    else
    {
      x_scale = loader->size->metrics->x_scale;
      y_scale = loader->size->metrics->y_scale;
    }

    /* Set `offset' to the start of the glyph relative to the start of */
    /* the `glyf' table, and `byte_len' to the length of the glyph in  */
    /* bytes.                                                          */

#ifdef FT_CONFIG_OPTION_INCREMENTAL

    /* If we are loading glyph data via the incremental interface, set */
    /* the loader stream to a memory stream reading the data returned  */
    /* by the interface.                                               */
    if ( face->root.internal->incremental_interface )
    {
      error = face->root.internal->incremental_interface->funcs->get_glyph_data(
                face->root.internal->incremental_interface->object,
                glyph_index, &glyph_data );
      if ( error )
        goto Exit;

      glyph_data_loaded = 1;
      offset            = 0;
      loader->byte_len  = glyph_data.length;

      FT_ZERO( &inc_stream );
      FT_Stream_OpenMemory( &inc_stream,
                            glyph_data.pointer,
                            glyph_data.length );

      loader->stream = &inc_stream;
    }
    else

#endif /* FT_CONFIG_OPTION_INCREMENTAL */
    {
      FT_ULong  len;


      offset = tt_face_get_location( FT_FACE( face ), glyph_index, &len );

      loader->byte_len = (FT_UInt)len;
    }

    if ( loader->byte_len > 0 )
    {
#ifdef FT_CONFIG_OPTION_INCREMENTAL
      /* for the incremental interface, `glyf_offset' is always zero */
      if ( !face->glyf_offset                          &&
           !face->root.internal->incremental_interface )
#else
      if ( !face->glyf_offset )
#endif /* FT_CONFIG_OPTION_INCREMENTAL */
      {
        FT_TRACE2(( "no `glyf' table but non-zero `loca' entry\n" ));
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      error = face->access_glyph_frame( loader, glyph_index,
                                        face->glyf_offset + offset,
                                        loader->byte_len );
      if ( error )
        goto Exit;

      /* read glyph header first */
      error = face->read_glyph_header( loader );

      face->forget_glyph_frame( loader );

      if ( error )
        goto Exit;
    }

    /* a space glyph */
    if ( loader->byte_len == 0 || loader->n_contours == 0 )
    {
      loader->bbox.xMin = 0;
      loader->bbox.xMax = 0;
      loader->bbox.yMin = 0;
      loader->bbox.yMax = 0;
    }

    /* the metrics must be computed after loading the glyph header */
    /* since we need the glyph's `yMax' value in case the vertical */
    /* metrics must be emulated                                    */
    error = tt_get_metrics( loader, glyph_index );
    if ( error )
      goto Exit;

    if ( header_only )
      goto Exit;

#ifdef FT_CONFIG_OPTION_INCREMENTAL
    tt_get_metrics_incremental( loader, glyph_index );
#endif
    tt_loader_set_pp( loader );

    /* shortcut for empty glyphs */
    if ( loader->byte_len == 0 || loader->n_contours == 0 )
    {

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

      if ( !IS_DEFAULT_INSTANCE( FT_FACE( face ) ) )
      {
        /* a small outline structure with four elements for */
        /* communication with `TT_Vary_Apply_Glyph_Deltas'  */
        FT_Vector   points[4];
        FT_Outline  outline;

        /* unrounded values */
        FT_Vector  unrounded[4] = { {0, 0}, {0, 0}, {0, 0}, {0, 0} };


        points[0] = loader->pp1;
        points[1] = loader->pp2;
        points[2] = loader->pp3;
        points[3] = loader->pp4;

        outline.n_points   = 0;
        outline.n_contours = 0;
        outline.points     = points;
        outline.tags       = NULL;
        outline.contours   = NULL;

        /* this must be done before scaling */
        error = TT_Vary_Apply_Glyph_Deltas( loader,
                                            &outline,
                                            unrounded );
        if ( error )
          goto Exit;
      }

#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */

      /* scale phantom points, if necessary; */
      /* they get rounded in `TT_Hint_Glyph' */
      if ( ( loader->load_flags & FT_LOAD_NO_SCALE ) == 0 )
      {
        loader->pp1.x = FT_MulFix( loader->pp1.x, x_scale );
        loader->pp2.x = FT_MulFix( loader->pp2.x, x_scale );
        /* pp1.y and pp2.y are always zero */

        loader->pp3.x = FT_MulFix( loader->pp3.x, x_scale );
        loader->pp3.y = FT_MulFix( loader->pp3.y, y_scale );
        loader->pp4.x = FT_MulFix( loader->pp4.x, x_scale );
        loader->pp4.y = FT_MulFix( loader->pp4.y, y_scale );
      }

      error = FT_Err_Ok;
      goto Exit;
    }


    /***********************************************************************/
    /***********************************************************************/
    /***********************************************************************/

    /* we now open a frame again, right after the glyph header */
    /* (which consists of 10 bytes)                            */
    error = face->access_glyph_frame( loader, glyph_index,
                                      face->glyf_offset + offset + 10,
                                      loader->byte_len - 10 );
    if ( error )
      goto Exit;

    opened_frame = 1;

    /* if it is a simple glyph, load it */

    if ( loader->n_contours > 0 )
    {
      error = face->read_simple_glyph( loader );
      if ( error )
        goto Exit;

      /* all data have been read */
      face->forget_glyph_frame( loader );
      opened_frame = 0;

      error = TT_Process_Simple_Glyph( loader );
      if ( error )
        goto Exit;

      FT_GlyphLoader_Add( gloader );
    }

    /***********************************************************************/
    /***********************************************************************/
    /***********************************************************************/

    /* otherwise, load a composite! */
    else if ( loader->n_contours < 0 )
    {
      FT_Memory  memory = face->root.memory;

      FT_UInt   start_point;
      FT_UInt   start_contour;
      FT_ULong  ins_pos;  /* position of composite instructions, if any */

      FT_ListNode  node, node2;


      /* normalize the `n_contours' value */
      loader->n_contours = -1;

      /*
       * We store the glyph index directly in the `node->data' pointer,
       * following the glib solution (cf. macro `GUINT_TO_POINTER') with a
       * double cast to make this portable.  Note, however, that this needs
       * pointers with a width of at least 32 bits.
       */

      /* clear the nodes filled by sibling chains */
      node = ft_list_get_node_at( &loader->composites, recurse_count );
      for ( node2 = node; node2; node2 = node2->next )
        node2->data = (void*)-1;

      /* check whether we already have a composite glyph with this index */
      if ( FT_List_Find( &loader->composites,
                         FT_UINT_TO_POINTER( glyph_index ) ) )
      {
        FT_TRACE1(( "TT_Load_Composite_Glyph:"
                    " infinite recursion detected\n" ));
        error = FT_THROW( Invalid_Composite );
        goto Exit;
      }

      else if ( node )
        node->data = FT_UINT_TO_POINTER( glyph_index );

      else
      {
        if ( FT_QNEW( node ) )
          goto Exit;
        node->data = FT_UINT_TO_POINTER( glyph_index );
        FT_List_Add( &loader->composites, node );
      }

      start_point   = gloader->base.outline.n_points;
      start_contour = gloader->base.outline.n_contours;

      /* for each subglyph, read composite header */
      error = face->read_composite_glyph( loader );
      if ( error )
        goto Exit;

      /* store the offset of instructions */
      ins_pos = loader->ins_pos;

      /* all data we need are read */
      face->forget_glyph_frame( loader );
      opened_frame = 0;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

      if ( !IS_DEFAULT_INSTANCE( FT_FACE( face ) ) )
      {
        FT_UShort    i, limit;
        FT_SubGlyph  subglyph;

        FT_Outline  outline = { 0, 0, NULL, NULL, NULL, 0 };
        FT_Vector*  unrounded = NULL;


        limit = (FT_UShort)gloader->current.num_subglyphs;

        /* construct an outline structure for              */
        /* communication with `TT_Vary_Apply_Glyph_Deltas' */
        if ( FT_QNEW_ARRAY( outline.points, limit + 4 ) ||
             FT_QNEW_ARRAY( outline.tags, limit )       ||
             FT_QNEW_ARRAY( outline.contours, limit )   ||
             FT_QNEW_ARRAY( unrounded, limit + 4 )      )
          goto Exit1;

        outline.n_contours = outline.n_points = limit;

        subglyph = gloader->current.subglyphs;

        for ( i = 0; i < limit; i++, subglyph++ )
        {
          /* applying deltas for anchor points doesn't make sense, */
          /* but we don't have to specially check this since       */
          /* unused delta values are zero anyways                  */
          outline.points[i].x = subglyph->arg1;
          outline.points[i].y = subglyph->arg2;
          outline.tags[i]     = ON_CURVE_POINT;
          outline.contours[i] = i;
        }

        outline.points[i++] = loader->pp1;
        outline.points[i++] = loader->pp2;
        outline.points[i++] = loader->pp3;
        outline.points[i  ] = loader->pp4;

        /* this call provides additional offsets */
        /* for each component's translation      */
        if ( FT_SET_ERROR( TT_Vary_Apply_Glyph_Deltas( loader,
                                                       &outline,
                                                       unrounded ) ) )
          goto Exit1;

        subglyph = gloader->current.subglyphs;

        for ( i = 0; i < limit; i++, subglyph++ )
        {
          if ( subglyph->flags & ARGS_ARE_XY_VALUES )
          {
            subglyph->arg1 = (FT_Int16)outline.points[i].x;
            subglyph->arg2 = (FT_Int16)outline.points[i].y;
          }
        }

      Exit1:
        FT_FREE( outline.points );
        FT_FREE( outline.tags );
        FT_FREE( outline.contours );
        FT_FREE( unrounded );

        if ( error )
          goto Exit;
      }

#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */

      /* scale phantom points, if necessary; */
      /* they get rounded in `TT_Hint_Glyph' */
      if ( ( loader->load_flags & FT_LOAD_NO_SCALE ) == 0 )
      {
        loader->pp1.x = FT_MulFix( loader->pp1.x, x_scale );
        loader->pp2.x = FT_MulFix( loader->pp2.x, x_scale );
        /* pp1.y and pp2.y are always zero */

        loader->pp3.x = FT_MulFix( loader->pp3.x, x_scale );
        loader->pp3.y = FT_MulFix( loader->pp3.y, y_scale );
        loader->pp4.x = FT_MulFix( loader->pp4.x, x_scale );
        loader->pp4.y = FT_MulFix( loader->pp4.y, y_scale );
      }

      /* if the flag FT_LOAD_NO_RECURSE is set, we return the subglyph */
      /* `as is' in the glyph slot (the client application will be     */
      /* responsible for interpreting these data)...                   */
      if ( loader->load_flags & FT_LOAD_NO_RECURSE )
      {
        FT_GlyphLoader_Add( gloader );
        loader->glyph->format = FT_GLYPH_FORMAT_COMPOSITE;

        goto Exit;
      }

      /*********************************************************************/
      /*********************************************************************/
      /*********************************************************************/

      {
        FT_UInt      n, num_base_points;
        FT_SubGlyph  subglyph       = NULL;

        FT_UInt      num_points     = start_point;
        FT_UInt      num_subglyphs  = gloader->current.num_subglyphs;
        FT_UInt      num_base_subgs = gloader->base.num_subglyphs;

        FT_Stream    old_stream     = loader->stream;
        FT_UInt      old_byte_len   = loader->byte_len;


        FT_GlyphLoader_Add( gloader );

        /* read each subglyph independently */
        for ( n = 0; n < num_subglyphs; n++ )
        {
          FT_Vector  pp[4];

          FT_Int  linear_hadvance;
          FT_Int  linear_vadvance;


          /* Each time we call `load_truetype_glyph' in this loop, the */
          /* value of `gloader.base.subglyphs' can change due to table */
          /* reallocations.  We thus need to recompute the subglyph    */
          /* pointer on each iteration.                                */
          subglyph = gloader->base.subglyphs + num_base_subgs + n;

          pp[0] = loader->pp1;
          pp[1] = loader->pp2;
          pp[2] = loader->pp3;
          pp[3] = loader->pp4;

          linear_hadvance = loader->linear;
          linear_vadvance = loader->vadvance;

          num_base_points = gloader->base.outline.n_points;

          error = load_truetype_glyph( loader,
                                       (FT_UInt)subglyph->index,
                                       recurse_count + 1,
                                       FALSE );
          if ( error )
            goto Exit;

          /* restore subglyph pointer */
          subglyph = gloader->base.subglyphs + num_base_subgs + n;

          /* restore phantom points if necessary */
          if ( !( subglyph->flags & USE_MY_METRICS ) )
          {
            loader->pp1 = pp[0];
            loader->pp2 = pp[1];
            loader->pp3 = pp[2];
            loader->pp4 = pp[3];

            loader->linear   = linear_hadvance;
            loader->vadvance = linear_vadvance;
          }

          num_points = gloader->base.outline.n_points;

          if ( num_points == num_base_points )
            continue;

          /* gloader->base.outline consists of three parts:           */
          /*                                                          */
          /* 0 ----> start_point ----> num_base_points ----> n_points */
          /*    (1)               (2)                   (3)           */
          /*                                                          */
          /* (1) points that exist from the beginning                 */
          /* (2) component points that have been loaded so far        */
          /* (3) points of the newly loaded component                 */
          error = TT_Process_Composite_Component( loader,
                                                  subglyph,
                                                  start_point,
                                                  num_base_points );
          if ( error )
            goto Exit;
        }

        loader->stream   = old_stream;
        loader->byte_len = old_byte_len;

        /* process the glyph */
        loader->ins_pos = ins_pos;
        if ( IS_HINTED( loader->load_flags ) &&
#ifdef TT_USE_BYTECODE_INTERPRETER
             subglyph                        &&
             subglyph->flags & WE_HAVE_INSTR &&
#endif
             num_points > start_point )
        {
          error = TT_Process_Composite_Glyph( loader,
                                              start_point,
                                              start_contour );
          if ( error )
            goto Exit;
        }
      }

      /* retain the overlap flag */
      if ( gloader->base.num_subglyphs                         &&
           gloader->base.subglyphs[0].flags & OVERLAP_COMPOUND )
        gloader->base.outline.flags |= FT_OUTLINE_OVERLAP;
    }

    /***********************************************************************/
    /***********************************************************************/
    /***********************************************************************/

  Exit:

    if ( opened_frame )
      face->forget_glyph_frame( loader );

#ifdef FT_CONFIG_OPTION_INCREMENTAL

    /* restore the original stream */
    loader->stream = face->root.stream;

    if ( glyph_data_loaded )
      face->root.internal->incremental_interface->funcs->free_glyph_data(
        face->root.internal->incremental_interface->object,
        &glyph_data );

#endif

    return error;
  }


  static FT_Error
  compute_glyph_metrics( TT_Loader  loader,
                         FT_UInt    glyph_index )
  {
    TT_Face       face  = loader->face;
    TT_Size       size  = loader->size;
    TT_GlyphSlot  glyph = loader->glyph;
    FT_BBox       bbox;
    FT_Fixed      y_scale;


    y_scale = 0x10000L;
    if ( ( loader->load_flags & FT_LOAD_NO_SCALE ) == 0 )
      y_scale = size->metrics->y_scale;

    if ( glyph->format != FT_GLYPH_FORMAT_COMPOSITE )
      FT_Outline_Get_CBox( &glyph->outline, &bbox );
    else
      bbox = loader->bbox;

    /* get the device-independent horizontal advance; it is scaled later */
    /* by the base layer.                                                */
    glyph->linearHoriAdvance = loader->linear;

    glyph->metrics.horiBearingX = bbox.xMin;
    glyph->metrics.horiBearingY = bbox.yMax;
    if ( loader->widthp )
      glyph->metrics.horiAdvance = loader->widthp[glyph_index] * 64;
    else
      glyph->metrics.horiAdvance = SUB_LONG( loader->pp2.x, loader->pp1.x );

    /* set glyph dimensions */
    glyph->metrics.width  = SUB_LONG( bbox.xMax, bbox.xMin );
    glyph->metrics.height = SUB_LONG( bbox.yMax, bbox.yMin );

    /* Now take care of vertical metrics.  In the case where there is */
    /* no vertical information within the font (relatively common),   */
    /* create some metrics manually                                   */
    {
      FT_Pos  top;      /* scaled vertical top side bearing  */
      FT_Pos  advance;  /* scaled vertical advance height    */


      /* Get the unscaled top bearing and advance height. */
      if ( face->vertical_info                   &&
           face->vertical.number_Of_VMetrics > 0 )
      {
        top = (FT_Short)FT_DivFix( SUB_LONG( loader->pp3.y, bbox.yMax ),
                                   y_scale );

        if ( loader->pp3.y <= loader->pp4.y )
          advance = 0;
        else
          advance = (FT_UShort)FT_DivFix( SUB_LONG( loader->pp3.y,
                                                    loader->pp4.y ),
                                          y_scale );
      }
      else
      {
        FT_Pos  height;


        /* XXX Compute top side bearing and advance height in  */
        /*     Get_VMetrics instead of here.                   */

        /* NOTE: The OS/2 values are the only `portable' ones, */
        /*       which is why we use them, if there is an OS/2 */
        /*       table in the font.  Otherwise, we use the     */
        /*       values defined in the horizontal header.      */

        height = (FT_Short)FT_DivFix( SUB_LONG( bbox.yMax,
                                                bbox.yMin ),
                                      y_scale );
        if ( face->os2.version != 0xFFFFU )
          advance = (FT_Pos)( face->os2.sTypoAscender -
                              face->os2.sTypoDescender );
        else
          advance = (FT_Pos)( face->horizontal.Ascender -
                              face->horizontal.Descender );

        top = ( advance - height ) / 2;
      }

#ifdef FT_CONFIG_OPTION_INCREMENTAL
      {
        FT_Incremental_InterfaceRec*  incr;
        FT_Incremental_MetricsRec     incr_metrics;
        FT_Error                      error;


        incr = face->root.internal->incremental_interface;

        /* If this is an incrementally loaded font see if there are */
        /* overriding metrics for this glyph.                       */
        if ( incr && incr->funcs->get_glyph_metrics )
        {
          incr_metrics.bearing_x = 0;
          incr_metrics.bearing_y = top;
          incr_metrics.advance   = advance;

          error = incr->funcs->get_glyph_metrics( incr->object,
                                                  glyph_index,
                                                  TRUE,
                                                  &incr_metrics );
          if ( error )
            return error;

          top     = incr_metrics.bearing_y;
          advance = incr_metrics.advance;
        }
      }

      /* GWW: Do vertical metrics get loaded incrementally too? */

#endif /* FT_CONFIG_OPTION_INCREMENTAL */

      glyph->linearVertAdvance = advance;

      /* scale the metrics */
      if ( !( loader->load_flags & FT_LOAD_NO_SCALE ) )
      {
        top     = FT_MulFix( top,     y_scale );
        advance = FT_MulFix( advance, y_scale );
      }

      /* XXX: for now, we have no better algorithm for the lsb, but it */
      /*      should work fine.                                        */
      /*                                                               */
      glyph->metrics.vertBearingX = SUB_LONG( glyph->metrics.horiBearingX,
                                              glyph->metrics.horiAdvance / 2 );
      glyph->metrics.vertBearingY = top;
      glyph->metrics.vertAdvance  = advance;
    }

    return FT_Err_Ok;
  }


#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS

  static FT_Error
  load_sbit_image( TT_Size       size,
                   TT_GlyphSlot  glyph,
                   FT_UInt       glyph_index,
                   FT_Int32      load_flags )
  {
    TT_Face             face   = (TT_Face)glyph->face;
    SFNT_Service        sfnt   = (SFNT_Service)face->sfnt;
    FT_Error            error;
    TT_SBit_MetricsRec  sbit_metrics;


    error = sfnt->load_sbit_image( face,
                                   size->strike_index,
                                   glyph_index,
                                   (FT_UInt)load_flags,
                                   face->root.stream,
                                   &glyph->bitmap,
                                   &sbit_metrics );
    if ( !error )
    {
      glyph->metrics.width  = (FT_Pos)sbit_metrics.width  * 64;
      glyph->metrics.height = (FT_Pos)sbit_metrics.height * 64;

      glyph->metrics.horiBearingX = (FT_Pos)sbit_metrics.horiBearingX * 64;
      glyph->metrics.horiBearingY = (FT_Pos)sbit_metrics.horiBearingY * 64;
      glyph->metrics.horiAdvance  = (FT_Pos)sbit_metrics.horiAdvance  * 64;

      glyph->metrics.vertBearingX = (FT_Pos)sbit_metrics.vertBearingX * 64;
      glyph->metrics.vertBearingY = (FT_Pos)sbit_metrics.vertBearingY * 64;
      glyph->metrics.vertAdvance  = (FT_Pos)sbit_metrics.vertAdvance  * 64;

      glyph->format = FT_GLYPH_FORMAT_BITMAP;

      if ( load_flags & FT_LOAD_VERTICAL_LAYOUT )
      {
        glyph->bitmap_left = sbit_metrics.vertBearingX;
        glyph->bitmap_top  = sbit_metrics.vertBearingY;
      }
      else
      {
        glyph->bitmap_left = sbit_metrics.horiBearingX;
        glyph->bitmap_top  = sbit_metrics.horiBearingY;
      }
    }
    /* a missing glyph in a bitmap-only font is assumed whitespace */
    /* that needs to be constructed using metrics data from `hmtx' */
    /* and, optionally, `vmtx' tables                              */
    else if ( FT_ERR_EQ( error, Missing_Bitmap ) &&
              !FT_IS_SCALABLE( glyph->face )     &&
              face->horz_metrics_size            )
    {
      FT_Fixed  x_scale = size->root.metrics.x_scale;
      FT_Fixed  y_scale = size->root.metrics.y_scale;

      FT_Short  left_bearing = 0;
      FT_Short  top_bearing  = 0;

      FT_UShort  advance_width  = 0;
      FT_UShort  advance_height = 0;


      TT_Get_HMetrics( face, glyph_index,
                       &left_bearing,
                       &advance_width );
      TT_Get_VMetrics( face, glyph_index,
                       0,
                       &top_bearing,
                       &advance_height );

      glyph->metrics.width  = 0;
      glyph->metrics.height = 0;

      glyph->metrics.horiBearingX = FT_MulFix( left_bearing, x_scale );
      glyph->metrics.horiBearingY = 0;
      glyph->metrics.horiAdvance  = FT_MulFix( advance_width, x_scale );

      glyph->metrics.vertBearingX = 0;
      glyph->metrics.vertBearingY = FT_MulFix( top_bearing, y_scale );
      glyph->metrics.vertAdvance  = FT_MulFix( advance_height, y_scale );

      glyph->format            = FT_GLYPH_FORMAT_BITMAP;
      glyph->bitmap.pixel_mode = FT_PIXEL_MODE_MONO;

      glyph->bitmap_left = 0;
      glyph->bitmap_top  = 0;

      error = FT_Err_Ok;
    }

    return error;
  }

#endif /* TT_CONFIG_OPTION_EMBEDDED_BITMAPS */


  static FT_Error
  tt_loader_init( TT_Loader     loader,
                  TT_Size       size,
                  TT_GlyphSlot  glyph,
                  FT_Int32      load_flags,
                  FT_Bool       glyf_table_only )
  {
    TT_Face    face   = (TT_Face)glyph->face;


    FT_ZERO( loader );

#ifdef TT_USE_BYTECODE_INTERPRETER

    /* load execution context */
    if ( IS_HINTED( load_flags ) && !glyf_table_only )
    {
      FT_Error        error;
      TT_ExecContext  exec;
      FT_Render_Mode  mode      = FT_LOAD_TARGET_MODE( load_flags );
      FT_Bool         grayscale = FT_BOOL( mode != FT_RENDER_MODE_MONO );
      FT_Bool         reexecute = FALSE;
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
      TT_Driver       driver    = (TT_Driver)FT_FACE_DRIVER( glyph->face );
#endif


      if ( size->bytecode_ready > 0 )
        return size->bytecode_ready;
      if ( size->bytecode_ready < 0 )
      {
        FT_Bool  pedantic = FT_BOOL( load_flags & FT_LOAD_PEDANTIC );


        error = tt_size_init_bytecode( size, pedantic );
        if ( error )
          return error;
      }

      exec = size->context;

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
      if ( driver->interpreter_version == TT_INTERPRETER_VERSION_40 )
      {
        grayscale = FALSE;

        /* any mode change requires a re-execution of the CVT program */
        if ( mode != exec->mode )
        {
          FT_TRACE4(( "tt_loader_init: render mode change,"
                      " re-executing `prep' table\n" ));

          exec->mode = mode;
          reexecute  = TRUE;
        }
      }
#endif

      /* a change from mono to grayscale rendering (and vice versa) */
      /* requires a re-execution of the CVT program                 */
      if ( grayscale != exec->grayscale )
      {
        FT_TRACE4(( "tt_loader_init: grayscale hinting change,"
                    " re-executing `prep' table\n" ));

        exec->grayscale = grayscale;
        reexecute       = TRUE;
      }

      if ( size->cvt_ready > 0 )
        return size->cvt_ready;
      if ( size->cvt_ready < 0 || reexecute )
      {
        error = tt_size_run_prep( size );
        if ( error )
          return error;
      }

      error = TT_Load_Context( exec, face, size );
      if ( error )
        return error;

      /* check whether the cvt program has disabled hinting */
      if ( size->GS.instruct_control & 1 )
        load_flags |= FT_LOAD_NO_HINTING;

      /* check whether GS modifications should be reverted */
      if ( size->GS.instruct_control & 2 )
        size->GS = tt_default_graphics_state;

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
      /*
       * Toggle backward compatibility according to what font wants, except
       * when
       *
       * 1) we have a `tricky' font that heavily relies on the interpreter to
       *    render glyphs correctly, for example DFKai-SB, or
       * 2) FT_RENDER_MODE_MONO (i.e, monochrome rendering) is requested.
       *
       * In those cases, backward compatibility needs to be turned off to get
       * correct rendering.  The rendering is then completely up to the
       * font's programming.
       *
       */
      if ( driver->interpreter_version == TT_INTERPRETER_VERSION_40 &&
           mode != FT_RENDER_MODE_MONO                              &&
           !FT_IS_TRICKY( glyph->face )                             )
        exec->backward_compatibility = ( size->GS.instruct_control & 4 ) ^ 4;
      else
        exec->backward_compatibility = 0;
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL */

      loader->exec = exec;

      /* Use the hdmx table if any unless FT_LOAD_COMPUTE_METRICS */
      /* is set or backward compatibility mode of the v38 or v40  */
      /* interpreters is active.  See `ttinterp.h' for details on */
      /* backward compatibility mode.                             */
      if ( IS_HINTED( load_flags )                   &&
           !( load_flags & FT_LOAD_COMPUTE_METRICS ) &&
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
           !exec->backward_compatibility             &&
#endif
           !face->postscript.isFixedPitch            )
      {
        loader->widthp = size->widthp;
      }
      else
        loader->widthp = NULL;
    }

#endif /* TT_USE_BYTECODE_INTERPRETER */

    /* get face's glyph loader */
    if ( !glyf_table_only )
    {
      FT_GlyphLoader  gloader = glyph->internal->loader;


      FT_GlyphLoader_Rewind( gloader );
      loader->gloader = gloader;
    }

    loader->load_flags = (FT_ULong)load_flags;

    loader->face   = face;
    loader->size   = size;
    loader->glyph  = (FT_GlyphSlot)glyph;
    loader->stream = face->root.stream;

    loader->composites.head = NULL;
    loader->composites.tail = NULL;

    return FT_Err_Ok;
  }


  static void
  tt_loader_done( TT_Loader  loader )
  {
    FT_List_Finalize( &loader->composites,
                      NULL,
                      loader->face->root.memory,
                      NULL );
  }


  /**************************************************************************
   *
   * @Function:
   *   TT_Load_Glyph
   *
   * @Description:
   *   A function used to load a single glyph within a given glyph slot,
   *   for a given size.
   *
   * @InOut:
   *   glyph ::
   *     A handle to a target slot object where the glyph
   *     will be loaded.
   *
   * @Input:
   *   size ::
   *     A handle to the source face size at which the glyph
   *     must be scaled/loaded.
   *
   *   glyph_index ::
   *     The index of the glyph in the font file.
   *
   *   load_flags ::
   *     A flag indicating what to load for this glyph.  The
   *     FT_LOAD_XXX constants can be used to control the
   *     glyph loading process (e.g., whether the outline
   *     should be scaled, whether to load bitmaps or not,
   *     whether to hint the outline, etc).
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  TT_Load_Glyph( TT_Size       size,
                 TT_GlyphSlot  glyph,
                 FT_UInt       glyph_index,
                 FT_Int32      load_flags )
  {
    TT_Face       face = (TT_Face)glyph->face;
    FT_Error      error;
    TT_LoaderRec  loader;


    FT_TRACE1(( "TT_Load_Glyph: glyph index %u\n", glyph_index ));

#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS

    /* try to load embedded bitmap (if any) */
    if ( size->strike_index != 0xFFFFFFFFUL  &&
         !( load_flags & FT_LOAD_NO_BITMAP &&
            FT_IS_SCALABLE( glyph->face )  ) &&
         IS_DEFAULT_INSTANCE( glyph->face )  )
    {
      error = load_sbit_image( size, glyph, glyph_index, load_flags );
      if ( !error )
      {
        if ( FT_IS_SCALABLE( glyph->face ) ||
             FT_HAS_SBIX( glyph->face )    )
        {
          FT_Fixed  x_scale = size->root.metrics.x_scale;
          FT_Fixed  y_scale = size->root.metrics.y_scale;


          /* for the bbox we need the header only */
          (void)tt_loader_init( &loader, size, glyph, load_flags, TRUE );
          (void)load_truetype_glyph( &loader, glyph_index, 0, TRUE );
          tt_loader_done( &loader );
          glyph->linearHoriAdvance = loader.linear;
          glyph->linearVertAdvance = loader.vadvance;

          /* Bitmaps from the 'sbix' table need special treatment:  */
          /* if there is a glyph contour, the bitmap origin must be */
          /* shifted to be relative to the lower left corner of the */
          /* glyph bounding box, also taking the left-side bearing  */
          /* (or top bearing) into account.                         */
          if ( face->sbit_table_type == TT_SBIT_TABLE_TYPE_SBIX &&
               loader.n_contours > 0                            )
          {
            FT_Int  bitmap_left;
            FT_Int  bitmap_top;


            if ( load_flags & FT_LOAD_VERTICAL_LAYOUT )
            {
              /* This is a guess, since Apple's CoreText engine doesn't */
              /* really do vertical typesetting.                        */
              bitmap_left = loader.bbox.xMin;
              bitmap_top  = loader.top_bearing;
            }
            else
            {
              bitmap_left = loader.left_bearing;
              bitmap_top  = loader.bbox.yMin;
            }

            glyph->bitmap_left += FT_MulFix( bitmap_left, x_scale ) >> 6;
            glyph->bitmap_top  += FT_MulFix( bitmap_top,  y_scale ) >> 6;
          }

          /* sanity checks: if `xxxAdvance' in the sbit metric */
          /* structure isn't set, use `linearXXXAdvance'      */
          if ( !glyph->metrics.horiAdvance && glyph->linearHoriAdvance )
            glyph->metrics.horiAdvance = FT_MulFix( glyph->linearHoriAdvance,
                                                    x_scale );
          if ( !glyph->metrics.vertAdvance && glyph->linearVertAdvance )
            glyph->metrics.vertAdvance = FT_MulFix( glyph->linearVertAdvance,
                                                    y_scale );
        }

        goto Exit;
      }
      else if ( !FT_IS_SCALABLE( glyph->face ) )
        goto Exit;
    }

    if ( load_flags & FT_LOAD_SBITS_ONLY )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

#endif /* TT_CONFIG_OPTION_EMBEDDED_BITMAPS */

    /* if FT_LOAD_NO_SCALE is not set, `ttmetrics' must be valid */
    if ( !( load_flags & FT_LOAD_NO_SCALE ) && !size->ttmetrics.ppem )
    {
      error = FT_THROW( Invalid_Size_Handle );
      goto Exit;
    }

#ifdef FT_CONFIG_OPTION_SVG

    /* check for OT-SVG */
    if ( ( load_flags & FT_LOAD_NO_SVG ) == 0 &&
         ( load_flags & FT_LOAD_COLOR )       &&
         face->svg                            )
    {
      SFNT_Service  sfnt = (SFNT_Service)face->sfnt;


      FT_TRACE3(( "Trying to load SVG glyph\n" ));

      error = sfnt->load_svg_doc( glyph, glyph_index );
      if ( !error )
      {
        FT_Fixed  x_scale = size->root.metrics.x_scale;
        FT_Fixed  y_scale = size->root.metrics.y_scale;

        FT_Short   leftBearing;
        FT_Short   topBearing;
        FT_UShort  advanceX;
        FT_UShort  advanceY;


        FT_TRACE3(( "Successfully loaded SVG glyph\n" ));

        glyph->format = FT_GLYPH_FORMAT_SVG;

        sfnt->get_metrics( face,
                           FALSE,
                           glyph_index,
                           &leftBearing,
                           &advanceX );
        sfnt->get_metrics( face,
                           TRUE,
                           glyph_index,
                           &topBearing,
                           &advanceY );

        glyph->linearHoriAdvance = advanceX;
        glyph->linearVertAdvance = advanceY;

        glyph->metrics.horiAdvance = FT_MulFix( advanceX, x_scale );
        glyph->metrics.vertAdvance = FT_MulFix( advanceY, y_scale );

        goto Exit;
      }

      FT_TRACE3(( "Failed to load SVG glyph\n" ));
    }

    /* return immediately if we only want SVG glyphs */
    if ( load_flags & FT_LOAD_SVG_ONLY )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

#endif /* FT_CONFIG_OPTION_SVG */

    error = tt_loader_init( &loader, size, glyph, load_flags, FALSE );
    if ( error )
      goto Exit;

    /* done if we are only interested in the `hdmx` advance */
    if ( load_flags & FT_LOAD_ADVANCE_ONLY         &&
         !( load_flags & FT_LOAD_VERTICAL_LAYOUT ) &&
         loader.widthp                             )
    {
      glyph->metrics.horiAdvance = loader.widthp[glyph_index] * 64;
      goto Done;
    }

    /* main loading loop */
    error = load_truetype_glyph( &loader, glyph_index, 0, FALSE );
    if ( !error )
    {
      if ( glyph->format == FT_GLYPH_FORMAT_COMPOSITE )
      {
        glyph->num_subglyphs = loader.gloader->base.num_subglyphs;
        glyph->subglyphs     = loader.gloader->base.subglyphs;
      }
      else
      {
        glyph->format         = FT_GLYPH_FORMAT_OUTLINE;

        glyph->outline        = loader.gloader->base.outline;
        glyph->outline.flags &= ~FT_OUTLINE_SINGLE_PASS;

        /* Set the `high precision' bit flag.  This is _critical_ to   */
        /* get correct output for monochrome TrueType glyphs at all    */
        /* sizes using the bytecode interpreter.                       */
        if ( !( load_flags & FT_LOAD_NO_SCALE ) &&
             size->metrics->y_ppem < 24         )
          glyph->outline.flags |= FT_OUTLINE_HIGH_PRECISION;

        /* Translate array so that (0,0) is the glyph's origin.  Note  */
        /* that this behaviour is independent on the value of bit 1 of */
        /* the `flags' field in the `head' table -- at least major     */
        /* applications like Acroread indicate that.                   */
        if ( loader.pp1.x )
          FT_Outline_Translate( &glyph->outline, -loader.pp1.x, 0 );
      }

#ifdef TT_USE_BYTECODE_INTERPRETER

      if ( IS_HINTED( load_flags ) )
      {
        glyph->control_data = loader.exec->glyphIns;
        glyph->control_len  = loader.exec->glyphSize;

        if ( loader.exec->GS.scan_control )
        {
          /* convert scan conversion mode to FT_OUTLINE_XXX flags */
          switch ( loader.exec->GS.scan_type )
          {
          case 0: /* simple drop-outs including stubs */
            glyph->outline.flags |= FT_OUTLINE_INCLUDE_STUBS;
            break;
          case 1: /* simple drop-outs excluding stubs */
            /* nothing; it's the default rendering mode */
            break;
          case 4: /* smart drop-outs including stubs */
            glyph->outline.flags |= FT_OUTLINE_SMART_DROPOUTS |
                                    FT_OUTLINE_INCLUDE_STUBS;
            break;
          case 5: /* smart drop-outs excluding stubs  */
            glyph->outline.flags |= FT_OUTLINE_SMART_DROPOUTS;
            break;

          default: /* no drop-out control */
            glyph->outline.flags |= FT_OUTLINE_IGNORE_DROPOUTS;
            break;
          }
        }
        else
          glyph->outline.flags |= FT_OUTLINE_IGNORE_DROPOUTS;
      }

#endif /* TT_USE_BYTECODE_INTERPRETER */

      error = compute_glyph_metrics( &loader, glyph_index );
    }

    FT_TRACE1(( "  subglyphs = %u, contours = %hu, points = %hu,"
                " flags = 0x%.3x\n",
                loader.gloader->base.num_subglyphs,
                glyph->outline.n_contours,
                glyph->outline.n_points,
                glyph->outline.flags ));

  Done:
    tt_loader_done( &loader );

  Exit:
    FT_TRACE1(( error ? "  failed (error code 0x%x)\n" : "",
                error ));

    return error;
  }


/* END */

/****************************************************************************
 *
 * pfrgload.c
 *
 *   FreeType PFR glyph loader (body).
 *
 * Copyright (C) 2002-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include "pfrgload.h"
#include "pfrsbit.h"
#include "pfrload.h"            /* for macro definitions */
#include <freetype/internal/ftdebug.h>

#include "pfrerror.h"

#undef  FT_COMPONENT
#define FT_COMPONENT  pfr


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      PFR GLYPH BUILDER                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  FT_LOCAL_DEF( void )
  pfr_glyph_init( PFR_Glyph       glyph,
                  FT_GlyphLoader  loader )
  {
    FT_ZERO( glyph );

    glyph->loader     = loader;
    glyph->path_begun = 0;

    FT_GlyphLoader_Rewind( loader );
  }


  FT_LOCAL_DEF( void )
  pfr_glyph_done( PFR_Glyph  glyph )
  {
    FT_Memory  memory = glyph->loader->memory;


    FT_FREE( glyph->x_control );
    glyph->y_control = NULL;

    glyph->max_xy_control = 0;
#if 0
    glyph->num_x_control  = 0;
    glyph->num_y_control  = 0;
#endif

    FT_FREE( glyph->subs );

    glyph->max_subs = 0;
    glyph->num_subs = 0;

    glyph->loader     = NULL;
    glyph->path_begun = 0;
  }


  /* close current contour, if any */
  static void
  pfr_glyph_close_contour( PFR_Glyph  glyph )
  {
    FT_GlyphLoader  loader  = glyph->loader;
    FT_Outline*     outline = &loader->current.outline;
    FT_Int          last, first;


    if ( !glyph->path_begun )
      return;

    /* compute first and last point indices in current glyph outline */
    last  = outline->n_points - 1;
    first = 0;
    if ( outline->n_contours > 0 )
      first = outline->contours[outline->n_contours - 1];

    /* if the last point falls on the same location as the first one */
    /* we need to delete it                                          */
    if ( last > first )
    {
      FT_Vector*  p1 = outline->points + first;
      FT_Vector*  p2 = outline->points + last;


      if ( p1->x == p2->x && p1->y == p2->y )
      {
        outline->n_points--;
        last--;
      }
    }

    /* don't add empty contours */
    if ( last >= first )
      outline->contours[outline->n_contours++] = (short)last;

    glyph->path_begun = 0;
  }


  /* reset glyph to start the loading of a new glyph */
  static void
  pfr_glyph_start( PFR_Glyph  glyph )
  {
    glyph->path_begun = 0;
  }


  static FT_Error
  pfr_glyph_line_to( PFR_Glyph   glyph,
                     FT_Vector*  to )
  {
    FT_GlyphLoader  loader  = glyph->loader;
    FT_Outline*     outline = &loader->current.outline;
    FT_Error        error;


    /* check that we have begun a new path */
    if ( !glyph->path_begun )
    {
      error = FT_THROW( Invalid_Table );
      FT_ERROR(( "pfr_glyph_line_to: invalid glyph data\n" ));
      goto Exit;
    }

    error = FT_GLYPHLOADER_CHECK_POINTS( loader, 1, 0 );
    if ( !error )
    {
      FT_Int  n = outline->n_points;


      outline->points[n] = *to;
      outline->tags  [n] = FT_CURVE_TAG_ON;

      outline->n_points++;
    }

  Exit:
    return error;
  }


  static FT_Error
  pfr_glyph_curve_to( PFR_Glyph   glyph,
                      FT_Vector*  control1,
                      FT_Vector*  control2,
                      FT_Vector*  to )
  {
    FT_GlyphLoader  loader  = glyph->loader;
    FT_Outline*     outline = &loader->current.outline;
    FT_Error        error;


    /* check that we have begun a new path */
    if ( !glyph->path_begun )
    {
      error = FT_THROW( Invalid_Table );
      FT_ERROR(( "pfr_glyph_line_to: invalid glyph data\n" ));
      goto Exit;
    }

    error = FT_GLYPHLOADER_CHECK_POINTS( loader, 3, 0 );
    if ( !error )
    {
      FT_Vector*  vec = outline->points         + outline->n_points;
      FT_Byte*    tag = (FT_Byte*)outline->tags + outline->n_points;


      vec[0] = *control1;
      vec[1] = *control2;
      vec[2] = *to;
      tag[0] = FT_CURVE_TAG_CUBIC;
      tag[1] = FT_CURVE_TAG_CUBIC;
      tag[2] = FT_CURVE_TAG_ON;

      outline->n_points = (FT_Short)( outline->n_points + 3 );
    }

  Exit:
    return error;
  }


  static FT_Error
  pfr_glyph_move_to( PFR_Glyph   glyph,
                     FT_Vector*  to )
  {
    FT_GlyphLoader  loader  = glyph->loader;
    FT_Error        error;


    /* close current contour if any */
    pfr_glyph_close_contour( glyph );

    /* indicate that a new contour has started */
    glyph->path_begun = 1;

    /* check that there is space for a new contour and a new point */
    error = FT_GLYPHLOADER_CHECK_POINTS( loader, 1, 1 );
    if ( !error )
    {
      /* add new start point */
      error = pfr_glyph_line_to( glyph, to );
    }

    return error;
  }


  static void
  pfr_glyph_end( PFR_Glyph  glyph )
  {
    /* close current contour if any */
    pfr_glyph_close_contour( glyph );

    /* merge the current glyph into the stack */
    FT_GlyphLoader_Add( glyph->loader );
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      PFR GLYPH LOADER                         *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  /* load a simple glyph */
  static FT_Error
  pfr_glyph_load_simple( PFR_Glyph  glyph,
                         FT_Byte*   p,
                         FT_Byte*   limit )
  {
    FT_Error   error  = FT_Err_Ok;
    FT_Memory  memory = glyph->loader->memory;
    FT_UInt    flags, x_count, y_count, i, count, mask;
    FT_Int     x;


    PFR_CHECK( 1 );
    flags = PFR_NEXT_BYTE( p );

    /* test for composite glyphs */
    if ( flags & PFR_GLYPH_IS_COMPOUND )
      goto Failure;

    x_count = 0;
    y_count = 0;

    if ( flags & PFR_GLYPH_1BYTE_XYCOUNT )
    {
      PFR_CHECK( 1 );
      count   = PFR_NEXT_BYTE( p );
      x_count = count & 15;
      y_count = count >> 4;
    }
    else
    {
      if ( flags & PFR_GLYPH_XCOUNT )
      {
        PFR_CHECK( 1 );
        x_count = PFR_NEXT_BYTE( p );
      }

      if ( flags & PFR_GLYPH_YCOUNT )
      {
        PFR_CHECK( 1 );
        y_count = PFR_NEXT_BYTE( p );
      }
    }

    count = x_count + y_count;

    /* re-allocate array when necessary */
    if ( count > glyph->max_xy_control )
    {
      FT_UInt  new_max = FT_PAD_CEIL( count, 8 );


      if ( FT_RENEW_ARRAY( glyph->x_control,
                           glyph->max_xy_control,
                           new_max ) )
        goto Exit;

      glyph->max_xy_control = new_max;
    }

    glyph->y_control = glyph->x_control + x_count;

    mask = 0;
    x    = 0;

    for ( i = 0; i < count; i++ )
    {
      if ( ( i & 7 ) == 0 )
      {
        PFR_CHECK( 1 );
        mask = PFR_NEXT_BYTE( p );
      }

      if ( mask & 1 )
      {
        PFR_CHECK( 2 );
        x = PFR_NEXT_SHORT( p );
      }
      else
      {
        PFR_CHECK( 1 );
        x += PFR_NEXT_BYTE( p );
      }

      glyph->x_control[i] = x;

      mask >>= 1;
    }

    /* XXX: we ignore the secondary stroke and edge definitions */
    /*      since we don't support native PFR hinting           */
    /*                                                          */
    if ( flags & PFR_GLYPH_SINGLE_EXTRA_ITEMS )
    {
      error = pfr_extra_items_skip( &p, limit );
      if ( error )
        goto Exit;
    }

    pfr_glyph_start( glyph );

    /* now load a simple glyph */
    {
      FT_Vector   pos[4];
      FT_Vector*  cur;


      pos[0].x = pos[0].y = 0;
      pos[3]   = pos[0];

      for (;;)
      {
        FT_UInt  format, format_low, args_format = 0, args_count, n;


        /****************************************************************
         * read instruction
         */
        PFR_CHECK( 1 );
        format     = PFR_NEXT_BYTE( p );
        format_low = format & 15;

        switch ( format >> 4 )
        {
        case 0:                                               /* end glyph */
          FT_TRACE6(( "- end glyph" ));
          args_count = 0;
          break;

        case 1:                                  /* general line operation */
          FT_TRACE6(( "- general line" ));
          goto Line1;

        case 4:                                 /* move to inside contour  */
          FT_TRACE6(( "- move to inside" ));
          goto Line1;

        case 5:                                 /* move to outside contour */
          FT_TRACE6(( "- move to outside" ));
        Line1:
          args_format = format_low;
          args_count  = 1;
          break;

        case 2:                                      /* horizontal line to */
          FT_TRACE6(( "- horizontal line to cx.%d", format_low ));
          if ( format_low >= x_count )
            goto Failure;
          pos[0].x   = glyph->x_control[format_low];
          pos[0].y   = pos[3].y;
          pos[3]     = pos[0];
          args_count = 0;
          break;

        case 3:                                        /* vertical line to */
          FT_TRACE6(( "- vertical line to cy.%d", format_low ));
          if ( format_low >= y_count )
            goto Failure;
          pos[0].x   = pos[3].x;
          pos[0].y   = glyph->y_control[format_low];
          pos[3]     = pos[0];
          args_count = 0;
          break;

        case 6:                            /* horizontal to vertical curve */
          FT_TRACE6(( "- hv curve " ));
          args_format = 0xB8E;
          args_count  = 3;
          break;

        case 7:                            /* vertical to horizontal curve */
          FT_TRACE6(( "- vh curve" ));
          args_format = 0xE2B;
          args_count  = 3;
          break;

        default:                                       /* general curve to */
          FT_TRACE6(( "- general curve" ));
          args_count  = 4;
          args_format = format_low;
        }

        /************************************************************
         * now read arguments
         */
        cur = pos;
        for ( n = 0; n < args_count; n++ )
        {
          FT_UInt  idx;
          FT_Int   delta;


          /* read the X argument */
          switch ( args_format & 3 )
          {
          case 0:                           /* 8-bit index */
            PFR_CHECK( 1 );
            idx = PFR_NEXT_BYTE( p );
            if ( idx >= x_count )
              goto Failure;
            cur->x = glyph->x_control[idx];
            FT_TRACE7(( " cx#%d", idx ));
            break;

          case 1:                           /* 16-bit absolute value */
            PFR_CHECK( 2 );
            cur->x = PFR_NEXT_SHORT( p );
            FT_TRACE7(( " x.%ld", cur->x ));
            break;

          case 2:                           /* 8-bit delta */
            PFR_CHECK( 1 );
            delta  = PFR_NEXT_INT8( p );
            cur->x = pos[3].x + delta;
            FT_TRACE7(( " dx.%d", delta ));
            break;

          default:
            FT_TRACE7(( " |" ));
            cur->x = pos[3].x;
          }

          /* read the Y argument */
          switch ( ( args_format >> 2 ) & 3 )
          {
          case 0:                           /* 8-bit index */
            PFR_CHECK( 1 );
            idx  = PFR_NEXT_BYTE( p );
            if ( idx >= y_count )
              goto Failure;
            cur->y = glyph->y_control[idx];
            FT_TRACE7(( " cy#%d", idx ));
            break;

          case 1:                           /* 16-bit absolute value */
            PFR_CHECK( 2 );
            cur->y = PFR_NEXT_SHORT( p );
            FT_TRACE7(( " y.%ld", cur->y ));
            break;

          case 2:                           /* 8-bit delta */
            PFR_CHECK( 1 );
            delta  = PFR_NEXT_INT8( p );
            cur->y = pos[3].y + delta;
            FT_TRACE7(( " dy.%d", delta ));
            break;

          default:
            FT_TRACE7(( " -" ));
            cur->y = pos[3].y;
          }

          /* read the additional format flag for the general curve */
          if ( n == 0 && args_count == 4 )
          {
            PFR_CHECK( 1 );
            args_format = PFR_NEXT_BYTE( p );
            args_count--;
          }
          else
            args_format >>= 4;

          /* save the previous point */
          pos[3] = cur[0];
          cur++;
        }

        FT_TRACE7(( "\n" ));

        /************************************************************
         * finally, execute instruction
         */
        switch ( format >> 4 )
        {
        case 0:                                       /* end glyph => EXIT */
          pfr_glyph_end( glyph );
          goto Exit;

        case 1:                                         /* line operations */
        case 2:
        case 3:
          error = pfr_glyph_line_to( glyph, pos );
          goto Test_Error;

        case 4:                                 /* move to inside contour  */
        case 5:                                 /* move to outside contour */
          error = pfr_glyph_move_to( glyph, pos );
          goto Test_Error;

        default:                                       /* curve operations */
          error = pfr_glyph_curve_to( glyph, pos, pos + 1, pos + 2 );

        Test_Error:  /* test error condition */
          if ( error )
            goto Exit;
        }
      } /* for (;;) */
    }

  Exit:
    return error;

  Failure:
  Too_Short:
    error = FT_THROW( Invalid_Table );
    FT_ERROR(( "pfr_glyph_load_simple: invalid glyph data\n" ));
    goto Exit;
  }


  /* load a composite/compound glyph */
  static FT_Error
  pfr_glyph_load_compound( PFR_Glyph  glyph,
                           FT_Byte*   p,
                           FT_Byte*   limit )
  {
    FT_Error        error  = FT_Err_Ok;
    FT_GlyphLoader  loader = glyph->loader;
    FT_Memory       memory = loader->memory;
    PFR_SubGlyph    subglyph;
    FT_UInt         flags, i, count, org_count;
    FT_Int          x_pos, y_pos;


    PFR_CHECK( 1 );
    flags = PFR_NEXT_BYTE( p );

    /* test for composite glyphs */
    if ( !( flags & PFR_GLYPH_IS_COMPOUND ) )
      goto Failure;

    count = flags & 0x3F;

    /* ignore extra items when present */
    /*                                 */
    if ( flags & PFR_GLYPH_COMPOUND_EXTRA_ITEMS )
    {
      error = pfr_extra_items_skip( &p, limit );
      if ( error )
        goto Exit;
    }

    /* we can't rely on the FT_GlyphLoader to load sub-glyphs, because   */
    /* the PFR format is dumb, using direct file offsets to point to the */
    /* sub-glyphs (instead of glyph indices).  Sigh.                     */
    /*                                                                   */
    /* For now, we load the list of sub-glyphs into a different array    */
    /* but this will prevent us from using the auto-hinter at its best   */
    /* quality.                                                          */
    /*                                                                   */
    org_count = glyph->num_subs;

    if ( org_count + count > glyph->max_subs )
    {
      FT_UInt  new_max = ( org_count + count + 3 ) & (FT_UInt)-4;


      /* we arbitrarily limit the number of subglyphs */
      /* to avoid endless recursion                   */
      if ( new_max > 64 )
      {
        error = FT_THROW( Invalid_Table );
        FT_ERROR(( "pfr_glyph_load_compound:"
                   " too many compound glyphs components\n" ));
        goto Exit;
      }

      if ( FT_RENEW_ARRAY( glyph->subs, glyph->max_subs, new_max ) )
        goto Exit;

      glyph->max_subs = new_max;
    }

    subglyph = glyph->subs + org_count;

    for ( i = 0; i < count; i++, subglyph++ )
    {
      FT_UInt  format;


      x_pos = 0;
      y_pos = 0;

      PFR_CHECK( 1 );
      format = PFR_NEXT_BYTE( p );

      /* read scale when available */
      subglyph->x_scale = 0x10000L;
      if ( format & PFR_SUBGLYPH_XSCALE )
      {
        PFR_CHECK( 2 );
        subglyph->x_scale = PFR_NEXT_SHORT( p ) * 16;
      }

      subglyph->y_scale = 0x10000L;
      if ( format & PFR_SUBGLYPH_YSCALE )
      {
        PFR_CHECK( 2 );
        subglyph->y_scale = PFR_NEXT_SHORT( p ) * 16;
      }

      /* read offset */
      switch ( format & 3 )
      {
      case 1:
        PFR_CHECK( 2 );
        x_pos = PFR_NEXT_SHORT( p );
        break;

      case 2:
        PFR_CHECK( 1 );
        x_pos += PFR_NEXT_INT8( p );
        break;

      default:
        ;
      }

      switch ( ( format >> 2 ) & 3 )
      {
      case 1:
        PFR_CHECK( 2 );
        y_pos = PFR_NEXT_SHORT( p );
        break;

      case 2:
        PFR_CHECK( 1 );
        y_pos += PFR_NEXT_INT8( p );
        break;

      default:
        ;
      }

      subglyph->x_delta = x_pos;
      subglyph->y_delta = y_pos;

      /* read glyph position and size now */
      if ( format & PFR_SUBGLYPH_2BYTE_SIZE )
      {
        PFR_CHECK( 2 );
        subglyph->gps_size = PFR_NEXT_USHORT( p );
      }
      else
      {
        PFR_CHECK( 1 );
        subglyph->gps_size = PFR_NEXT_BYTE( p );
      }

      if ( format & PFR_SUBGLYPH_3BYTE_OFFSET )
      {
        PFR_CHECK( 3 );
        subglyph->gps_offset = PFR_NEXT_ULONG( p );
      }
      else
      {
        PFR_CHECK( 2 );
        subglyph->gps_offset = PFR_NEXT_USHORT( p );
      }

      glyph->num_subs++;
    }

  Exit:
    return error;

  Failure:
  Too_Short:
    error = FT_THROW( Invalid_Table );
    FT_ERROR(( "pfr_glyph_load_compound: invalid glyph data\n" ));
    goto Exit;
  }


  static FT_Error
  pfr_glyph_load_rec( PFR_Glyph  glyph,
                      FT_Stream  stream,
                      FT_ULong   gps_offset,
                      FT_ULong   offset,
                      FT_ULong   size )
  {
    FT_Error  error;
    FT_Byte*  p;
    FT_Byte*  limit;


    if ( FT_STREAM_SEEK( gps_offset + offset ) ||
         FT_FRAME_ENTER( size )                )
      goto Exit;

    p     = (FT_Byte*)stream->cursor;
    limit = p + size;

    if ( size > 0 && *p & PFR_GLYPH_IS_COMPOUND )
    {
      FT_UInt         n, old_count, count;
      FT_GlyphLoader  loader = glyph->loader;
      FT_Outline*     base   = &loader->base.outline;


      old_count = glyph->num_subs;

      /* this is a compound glyph - load it */
      error = pfr_glyph_load_compound( glyph, p, limit );

      FT_FRAME_EXIT();

      if ( error )
        goto Exit;

      count = glyph->num_subs - old_count;

      FT_TRACE4(( "compound glyph with %d element%s (offset %lu):\n",
                  count,
                  count == 1 ? "" : "s",
                  offset ));

      /* now, load each individual glyph */
      for ( n = 0; n < count; n++ )
      {
        FT_Int        i, old_points, num_points;
        PFR_SubGlyph  subglyph;


        FT_TRACE4(( "  subglyph %d:\n", n ));

        subglyph   = glyph->subs + old_count + n;
        old_points = base->n_points;

        error = pfr_glyph_load_rec( glyph, stream, gps_offset,
                                    subglyph->gps_offset,
                                    subglyph->gps_size );
        if ( error )
          break;

        /* note that `glyph->subs' might have been re-allocated */
        subglyph   = glyph->subs + old_count + n;
        num_points = base->n_points - old_points;

        /* translate and eventually scale the new glyph points */
        if ( subglyph->x_scale != 0x10000L || subglyph->y_scale != 0x10000L )
        {
          FT_Vector*  vec = base->points + old_points;


          for ( i = 0; i < num_points; i++, vec++ )
          {
            vec->x = FT_MulFix( vec->x, subglyph->x_scale ) +
                       subglyph->x_delta;
            vec->y = FT_MulFix( vec->y, subglyph->y_scale ) +
                       subglyph->y_delta;
          }
        }
        else
        {
          FT_Vector*  vec = loader->base.outline.points + old_points;


          for ( i = 0; i < num_points; i++, vec++ )
          {
            vec->x += subglyph->x_delta;
            vec->y += subglyph->y_delta;
          }
        }

        /* proceed to next sub-glyph */
      }

      FT_TRACE4(( "end compound glyph with %d element%s\n",
                  count,
                  count == 1 ? "" : "s" ));
    }
    else
    {
      FT_TRACE4(( "simple glyph (offset %lu)\n", offset ));

      /* load a simple glyph */
      error = pfr_glyph_load_simple( glyph, p, limit );

      FT_FRAME_EXIT();
    }

  Exit:
    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  pfr_glyph_load( PFR_Glyph  glyph,
                  FT_Stream  stream,
                  FT_ULong   gps_offset,
                  FT_ULong   offset,
                  FT_ULong   size )
  {
    /* initialize glyph loader */
    FT_GlyphLoader_Rewind( glyph->loader );

    glyph->num_subs = 0;

    /* load the glyph, recursively when needed */
    return pfr_glyph_load_rec( glyph, stream, gps_offset, offset, size );
  }


/* END */

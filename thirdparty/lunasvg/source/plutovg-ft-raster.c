/***************************************************************************/
/*                                                                         */
/*  ftgrays.c                                                              */
/*                                                                         */
/*    A new `perfect' anti-aliasing renderer (body).                       */
/*                                                                         */
/*  Copyright 2000-2003, 2005-2014 by                                      */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, FTL.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

/*************************************************************************/
/*                                                                       */
/* This is a new anti-aliasing scan-converter for FreeType 2.  The       */
/* algorithm used here is _very_ different from the one in the standard  */
/* `ftraster' module.  Actually, `ftgrays' computes the _exact_          */
/* coverage of the outline on each pixel cell.                           */
/*                                                                       */
/* It is based on ideas that I initially found in Raph Levien's          */
/* excellent LibArt graphics library (see http://www.levien.com/libart   */
/* for more information, though the web pages do not tell anything       */
/* about the renderer; you'll have to dive into the source code to       */
/* understand how it works).                                             */
/*                                                                       */
/* Note, however, that this is a _very_ different implementation         */
/* compared to Raph's.  Coverage information is stored in a very         */
/* different way, and I don't use sorted vector paths.  Also, it doesn't */
/* use floating point values.                                            */
/*                                                                       */
/* This renderer has the following advantages:                           */
/*                                                                       */
/* - It doesn't need an intermediate bitmap.  Instead, one can supply a  */
/*   callback function that will be called by the renderer to draw gray  */
/*   spans on any target surface.  You can thus do direct composition on */
/*   any kind of bitmap, provided that you give the renderer the right   */
/*   callback.                                                           */
/*                                                                       */
/* - A perfect anti-aliaser, i.e., it computes the _exact_ coverage on   */
/*   each pixel cell.                                                    */
/*                                                                       */
/* - It performs a single pass on the outline (the `standard' FT2        */
/*   renderer makes two passes).                                         */
/*                                                                       */
/* - It can easily be modified to render to _any_ number of gray levels  */
/*   cheaply.                                                            */
/*                                                                       */
/* - For small (< 20) pixel sizes, it is faster than the standard        */
/*   renderer.                                                           */
/*                                                                       */
/*************************************************************************/

#include "plutovg-ft-raster.h"
#include "plutovg-ft-math.h"

#define PVG_FT_BEGIN_STMNT  do {
#define PVG_FT_END_STMNT    } while ( 0 )

#include <setjmp.h>

#define pvg_ft_setjmp   setjmp
#define pvg_ft_longjmp  longjmp
#define pvg_ft_jmp_buf  jmp_buf

#include <stddef.h>

typedef ptrdiff_t  PVG_FT_PtrDist;

#define ErrRaster_Invalid_Mode      -2
#define ErrRaster_Invalid_Outline   -1
#define ErrRaster_Invalid_Argument  -3
#define ErrRaster_Memory_Overflow   -4
#define ErrRaster_OutOfMemory       -6

#include <stdlib.h>
#include <limits.h>

#define PVG_FT_MINIMUM_POOL_SIZE 8192

#define RAS_ARG   PWorker  worker
#define RAS_ARG_  PWorker  worker,

#define RAS_VAR   worker
#define RAS_VAR_  worker,

#define ras       (*worker)

  /* must be at least 6 bits! */
#define PIXEL_BITS  8

#define ONE_PIXEL       ( 1L << PIXEL_BITS )
#define TRUNC( x )      (TCoord)( (x) >> PIXEL_BITS )
#define FRACT( x )      (TCoord)( (x) & ( ONE_PIXEL - 1 ) )

#if PIXEL_BITS >= 6
#define UPSCALE( x )    ( (x) * ( ONE_PIXEL >> 6 ) )
#define DOWNSCALE( x )  ( (x) >> ( PIXEL_BITS - 6 ) )
#else
#define UPSCALE( x )    ( (x) >> ( 6 - PIXEL_BITS ) )
#define DOWNSCALE( x )  ( (x) * ( 64 >> PIXEL_BITS ) )
#endif

/* Compute `dividend / divisor' and return both its quotient and     */
/* remainder, cast to a specific type.  This macro also ensures that */
/* the remainder is always positive.                                 */
#define PVG_FT_DIV_MOD( type, dividend, divisor, quotient, remainder ) \
PVG_FT_BEGIN_STMNT                                                   \
  (quotient)  = (type)( (dividend) / (divisor) );                \
  (remainder) = (type)( (dividend) % (divisor) );                \
  if ( (remainder) < 0 )                                         \
  {                                                              \
    (quotient)--;                                                \
    (remainder) += (type)(divisor);                              \
  }                                                              \
PVG_FT_END_STMNT

  /* These macros speed up repetitive divisions by replacing them */
  /* with multiplications and right shifts.                       */
#define PVG_FT_UDIVPREP( b )                                       \
  long  b ## _r = (long)( ULONG_MAX >> PIXEL_BITS ) / ( b )
#define PVG_FT_UDIV( a, b )                                        \
  ( ( (unsigned long)( a ) * (unsigned long)( b ## _r ) ) >>   \
    ( sizeof( long ) * CHAR_BIT - PIXEL_BITS ) )


  /*************************************************************************/
  /*                                                                       */
  /*   TYPE DEFINITIONS                                                    */
  /*                                                                       */

  /* don't change the following types to PVG_FT_Int or PVG_FT_Pos, since we might */
  /* need to define them to "float" or "double" when experimenting with   */
  /* new algorithms                                                       */

  typedef long   TCoord;   /* integer scanline/pixel coordinate */
  typedef long   TPos;     /* sub-pixel coordinate              */
  typedef long   TArea ;   /* cell areas, coordinate products   */

  /* maximal number of gray spans in a call to the span callback */
#define PVG_FT_MAX_GRAY_SPANS  256


  typedef struct TCell_*  PCell;

  typedef struct  TCell_
  {
    int    x;
    int    cover;
    TArea  area;
    PCell  next;

  } TCell;


  typedef struct  TWorker_
  {
    TCoord  ex, ey;
    TPos    min_ex, max_ex;
    TPos    min_ey, max_ey;
    TPos    count_ex, count_ey;

    TArea   area;
    int     cover;
    int     invalid;

    PCell   cells;
    PVG_FT_PtrDist     max_cells;
    PVG_FT_PtrDist     num_cells;

    TPos    x,  y;

    PVG_FT_Outline  outline;
    PVG_FT_BBox     clip_box;

    PVG_FT_Span     gray_spans[PVG_FT_MAX_GRAY_SPANS];
    int         num_gray_spans;
    int         skip_spans;

    PVG_FT_Raster_Span_Func  render_span;
    void*                render_span_data;

    int  band_size;
    int  band_shoot;

    pvg_ft_jmp_buf  jump_buffer;

    void*       buffer;
    long        buffer_size;

    PCell*     ycells;
    TPos       ycount;
  } TWorker, *PWorker;


  /*************************************************************************/
  /*                                                                       */
  /* Initialize the cells table.                                           */
  /*                                                                       */
  static void
  gray_init_cells( RAS_ARG_ void*  buffer,
                            long   byte_size )
  {
    ras.buffer      = buffer;
    ras.buffer_size = byte_size;

    ras.ycells      = (PCell*) buffer;
    ras.cells       = NULL;
    ras.max_cells   = 0;
    ras.num_cells   = 0;
    ras.area        = 0;
    ras.cover       = 0;
    ras.invalid     = 1;
  }


  /*************************************************************************/
  /*                                                                       */
  /* Compute the outline bounding box.                                     */
  /*                                                                       */
  static void
  gray_compute_cbox( RAS_ARG )
  {
    PVG_FT_Outline*  outline = &ras.outline;
    PVG_FT_Vector*   vec     = outline->points;
    PVG_FT_Vector*   limit   = vec + outline->n_points;


    if ( outline->n_points <= 0 )
    {
      ras.min_ex = ras.max_ex = 0;
      ras.min_ey = ras.max_ey = 0;
      return;
    }

    ras.min_ex = ras.max_ex = vec->x;
    ras.min_ey = ras.max_ey = vec->y;

    vec++;

    for ( ; vec < limit; vec++ )
    {
      TPos  x = vec->x;
      TPos  y = vec->y;


      if ( x < ras.min_ex ) ras.min_ex = x;
      if ( x > ras.max_ex ) ras.max_ex = x;
      if ( y < ras.min_ey ) ras.min_ey = y;
      if ( y > ras.max_ey ) ras.max_ey = y;
    }

    /* truncate the bounding box to integer pixels */
    ras.min_ex = ras.min_ex >> 6;
    ras.min_ey = ras.min_ey >> 6;
    ras.max_ex = ( ras.max_ex + 63 ) >> 6;
    ras.max_ey = ( ras.max_ey + 63 ) >> 6;
  }


  /*************************************************************************/
  /*                                                                       */
  /* Record the current cell in the table.                                 */
  /*                                                                       */
  static PCell
  gray_find_cell( RAS_ARG )
  {
    PCell  *pcell, cell;
    TPos    x = ras.ex;


    if ( x > ras.count_ex )
      x = ras.count_ex;

    pcell = &ras.ycells[ras.ey];
    for (;;)
    {
      cell = *pcell;
      if ( cell == NULL || cell->x > x )
        break;

      if ( cell->x == x )
        goto Exit;

      pcell = &cell->next;
    }

    if ( ras.num_cells >= ras.max_cells )
      pvg_ft_longjmp( ras.jump_buffer, 1 );

    cell        = ras.cells + ras.num_cells++;
    cell->x     = x;
    cell->area  = 0;
    cell->cover = 0;

    cell->next  = *pcell;
    *pcell      = cell;

  Exit:
    return cell;
  }


  static void
  gray_record_cell( RAS_ARG )
  {
    if ( ras.area | ras.cover )
    {
      PCell  cell = gray_find_cell( RAS_VAR );


      cell->area  += ras.area;
      cell->cover += ras.cover;
    }
  }


  /*************************************************************************/
  /*                                                                       */
  /* Set the current cell to a new position.                               */
  /*                                                                       */
  static void
  gray_set_cell( RAS_ARG_ TCoord  ex,
                          TCoord  ey )
  {
    /* Move the cell pointer to a new position.  We set the `invalid'      */
    /* flag to indicate that the cell isn't part of those we're interested */
    /* in during the render phase.  This means that:                       */
    /*                                                                     */
    /* . the new vertical position must be within min_ey..max_ey-1.        */
    /* . the new horizontal position must be strictly less than max_ex     */
    /*                                                                     */
    /* Note that if a cell is to the left of the clipping region, it is    */
    /* actually set to the (min_ex-1) horizontal position.                 */

    /* All cells that are on the left of the clipping region go to the */
    /* min_ex - 1 horizontal position.                                 */
    ey -= ras.min_ey;

    if ( ex > ras.max_ex )
      ex = ras.max_ex;

    ex -= ras.min_ex;
    if ( ex < 0 )
      ex = -1;

    /* are we moving to a different cell ? */
    if ( ex != ras.ex || ey != ras.ey )
    {
      /* record the current one if it is valid */
      if ( !ras.invalid )
        gray_record_cell( RAS_VAR );

      ras.area  = 0;
      ras.cover = 0;
      ras.ex    = ex;
      ras.ey    = ey;
    }

    ras.invalid = ( (unsigned int)ey >= (unsigned int)ras.count_ey ||
                                  ex >= ras.count_ex           );
  }


  /*************************************************************************/
  /*                                                                       */
  /* Start a new contour at a given cell.                                  */
  /*                                                                       */
  static void
  gray_start_cell( RAS_ARG_ TCoord  ex,
                            TCoord  ey )
  {
    if ( ex > ras.max_ex )
      ex = (TCoord)( ras.max_ex );

    if ( ex < ras.min_ex )
      ex = (TCoord)( ras.min_ex - 1 );

    ras.area    = 0;
    ras.cover   = 0;
    ras.ex      = ex - ras.min_ex;
    ras.ey      = ey - ras.min_ey;
    ras.invalid = 0;

    gray_set_cell( RAS_VAR_ ex, ey );
  }

// The new render-line implementation is not yet used
#if 1

  /*************************************************************************/
  /*                                                                       */
  /* Render a scanline as one or more cells.                               */
  /*                                                                       */
  static void
  gray_render_scanline( RAS_ARG_ TCoord  ey,
                                 TPos    x1,
                                 TCoord  y1,
                                 TPos    x2,
                                 TCoord  y2 )
  {
    TCoord  ex1, ex2, fx1, fx2, first, dy, delta, mod;
    TPos    p, dx;
    int     incr;


    ex1 = TRUNC( x1 );
    ex2 = TRUNC( x2 );

    /* trivial case.  Happens often */
    if ( y1 == y2 )
    {
      gray_set_cell( RAS_VAR_ ex2, ey );
      return;
    }

    fx1   = FRACT( x1 );
    fx2   = FRACT( x2 );

    /* everything is located in a single cell.  That is easy! */
    /*                                                        */
    if ( ex1 == ex2 )
      goto End;

    /* ok, we'll have to render a run of adjacent cells on the same */
    /* scanline...                                                  */
    /*                                                              */
    dx = x2 - x1;
    dy = y2 - y1;

    if ( dx > 0 )
    {
      p     = ( ONE_PIXEL - fx1 ) * dy;
      first = ONE_PIXEL;
      incr  = 1;
    } else {
      p     = fx1 * dy;
      first = 0;
      incr  = -1;
      dx    = -dx;
    }

    PVG_FT_DIV_MOD( TCoord, p, dx, delta, mod );

    ras.area  += (TArea)( fx1 + first ) * delta;
    ras.cover += delta;
    y1        += delta;
    ex1       += incr;
    gray_set_cell( RAS_VAR_ ex1, ey );

    if ( ex1 != ex2 )
    {
      TCoord  lift, rem;


      p = ONE_PIXEL * dy;
      PVG_FT_DIV_MOD( TCoord, p, dx, lift, rem );

      do
      {
        delta = lift;
        mod  += rem;
        if ( mod >= (TCoord)dx )
        {
          mod -= (TCoord)dx;
          delta++;
        }

        ras.area  += (TArea)( ONE_PIXEL * delta );
        ras.cover += delta;
        y1        += delta;
        ex1       += incr;
        gray_set_cell( RAS_VAR_ ex1, ey );
      } while ( ex1 != ex2 );
    }
    fx1 = ONE_PIXEL - first;

  End:
    dy = y2 - y1;

    ras.area  += (TArea)( ( fx1 + fx2 ) * dy );
    ras.cover += dy;
  }


  /*************************************************************************/
  /*                                                                       */
  /* Render a given line as a series of scanlines.                         */
  /*                                                                       */
  static void
  gray_render_line( RAS_ARG_ TPos  to_x,
                             TPos  to_y )
  {
    TCoord  ey1, ey2, fy1, fy2, first, delta, mod;
    TPos    p, dx, dy, x, x2;
    int     incr;

    ey1 = TRUNC( ras.y );
    ey2 = TRUNC( to_y );     /* if (ey2 >= ras.max_ey) ey2 = ras.max_ey-1; */

    /* perform vertical clipping */
    if ( ( ey1 >= ras.max_ey && ey2 >= ras.max_ey ) ||
         ( ey1 <  ras.min_ey && ey2 <  ras.min_ey ) )
      goto End;

    fy1 = FRACT( ras.y );
    fy2 = FRACT( to_y );

    /* everything is on a single scanline */
    if ( ey1 == ey2 )
    {
      gray_render_scanline( RAS_VAR_ ey1, ras.x, fy1, to_x, fy2 );
      goto End;
    }

    dx = to_x - ras.x;
    dy = to_y - ras.y;

    /* vertical line - avoid calling gray_render_scanline */
    if ( dx == 0 )
    {
      TCoord  ex     = TRUNC( ras.x );
      TCoord  two_fx = FRACT( ras.x ) << 1;
      TPos    area, max_ey1;


      if ( dy > 0)
      {
        first = ONE_PIXEL;
      }
      else
      {
        first = 0;
      }

      delta      = first - fy1;
      ras.area  += (TArea)two_fx * delta;
      ras.cover += delta;

      delta = first + first - ONE_PIXEL;
      area  = (TArea)two_fx * delta;
      max_ey1 = ras.count_ey + ras.min_ey;
      if (dy < 0) {
        if (ey1 > max_ey1) {
          ey1 = (max_ey1 > ey2) ? max_ey1 : ey2;
          gray_set_cell( &ras, ex, ey1 );
        } else {
          ey1--;
          gray_set_cell( &ras, ex, ey1 );
        }
        while ( ey1 > ey2 && ey1 >= ras.min_ey)
        {
          ras.area  += area;
          ras.cover += delta;
          ey1--;

          gray_set_cell( &ras, ex, ey1 );
        }
        if (ey1 != ey2) {
          ey1 = ey2;
          gray_set_cell( &ras, ex, ey1 );
        }
      } else {
        if (ey1 < ras.min_ey) {
          ey1 = (ras.min_ey < ey2) ? ras.min_ey : ey2;
          gray_set_cell( &ras, ex, ey1 );
        } else {
          ey1++;
          gray_set_cell( &ras, ex, ey1 );
        }
        while ( ey1 < ey2 && ey1 < max_ey1)
        {
          ras.area  += area;
          ras.cover += delta;
          ey1++;

          gray_set_cell( &ras, ex, ey1 );
        }
        if (ey1 != ey2) {
          ey1 = ey2;
          gray_set_cell( &ras, ex, ey1 );
        }
      }

      delta      = (int)( fy2 - ONE_PIXEL + first );
      ras.area  += (TArea)two_fx * delta;
      ras.cover += delta;

      goto End;
    }

    /* ok, we have to render several scanlines */
    if ( dy > 0)
    {
      p     = ( ONE_PIXEL - fy1 ) * dx;
      first = ONE_PIXEL;
      incr  = 1;
    }
    else
    {
      p     = fy1 * dx;
      first = 0;
      incr  = -1;
      dy    = -dy;
    }

    /* the fractional part of x-delta is mod/dy. It is essential to */
    /* keep track of its accumulation for accurate rendering.       */
    PVG_FT_DIV_MOD( TCoord, p, dy, delta, mod );

    x = ras.x + delta;
    gray_render_scanline( RAS_VAR_ ey1, ras.x, fy1, x, (TCoord)first );

    ey1 += incr;
    gray_set_cell( RAS_VAR_ TRUNC( x ), ey1 );

    if ( ey1 != ey2 )
    {
      TCoord  lift, rem;


      p    = ONE_PIXEL * dx;
      PVG_FT_DIV_MOD( TCoord, p, dy, lift, rem );

      do
      {
        delta = lift;
        mod  += rem;
        if ( mod >= (TCoord)dy )
        {
          mod -= (TCoord)dy;
          delta++;
        }

        x2 = x + delta;
        gray_render_scanline( RAS_VAR_ ey1,
                                       x, ONE_PIXEL - first,
                                       x2, first );
        x = x2;

        ey1 += incr;
        gray_set_cell( RAS_VAR_ TRUNC( x ), ey1 );
      } while ( ey1 != ey2 );
    }

    gray_render_scanline( RAS_VAR_ ey1,
                                   x, ONE_PIXEL - first,
                                   to_x, fy2 );

  End:
    ras.x       = to_x;
    ras.y       = to_y;
  }


#else

  /*************************************************************************/
  /*                                                                       */
  /* Render a straight line across multiple cells in any direction.        */
  /*                                                                       */
  static void
  gray_render_line( RAS_ARG_ TPos  to_x,
                             TPos  to_y )
  {
    TPos    dx, dy, fx1, fy1, fx2, fy2;
    TCoord  ex1, ex2, ey1, ey2;


    ex1 = TRUNC( ras.x );
    ex2 = TRUNC( to_x );
    ey1 = TRUNC( ras.y );
    ey2 = TRUNC( to_y );

    /* perform vertical clipping */
    if ( ( ey1 >= ras.max_ey && ey2 >= ras.max_ey ) ||
         ( ey1 <  ras.min_ey && ey2 <  ras.min_ey ) )
      goto End;

    dx = to_x - ras.x;
    dy = to_y - ras.y;

    fx1 = FRACT( ras.x );
    fy1 = FRACT( ras.y );

    if ( ex1 == ex2 && ey1 == ey2 )       /* inside one cell */
      ;
    else if ( dy == 0 ) /* ex1 != ex2 */  /* any horizontal line */
    {
      ex1 = ex2;
      gray_set_cell( RAS_VAR_ ex1, ey1 );
    }
    else if ( dx == 0 )
    {
      if ( dy > 0 )                       /* vertical line up */
        do
        {
          fy2 = ONE_PIXEL;
          ras.cover += ( fy2 - fy1 );
          ras.area  += ( fy2 - fy1 ) * fx1 * 2;
          fy1 = 0;
          ey1++;
          gray_set_cell( RAS_VAR_ ex1, ey1 );
        } while ( ey1 != ey2 );
      else                                /* vertical line down */
        do
        {
          fy2 = 0;
          ras.cover += ( fy2 - fy1 );
          ras.area  += ( fy2 - fy1 ) * fx1 * 2;
          fy1 = ONE_PIXEL;
          ey1--;
          gray_set_cell( RAS_VAR_ ex1, ey1 );
        } while ( ey1 != ey2 );
    }
    else                                  /* any other line */
    {
      TArea  prod = dx * fy1 - dy * fx1;
      PVG_FT_UDIVPREP( dx );
      PVG_FT_UDIVPREP( dy );


      /* The fundamental value `prod' determines which side and the  */
      /* exact coordinate where the line exits current cell.  It is  */
      /* also easily updated when moving from one cell to the next.  */
      do
      {
        if      ( prod                                   <= 0 &&
                  prod - dx * ONE_PIXEL                  >  0 ) /* left */
        {
          fx2 = 0;
          fy2 = (TPos)PVG_FT_UDIV( -prod, -dx );
          prod -= dy * ONE_PIXEL;
          ras.cover += ( fy2 - fy1 );
          ras.area  += ( fy2 - fy1 ) * ( fx1 + fx2 );
          fx1 = ONE_PIXEL;
          fy1 = fy2;
          ex1--;
        }
        else if ( prod - dx * ONE_PIXEL                  <= 0 &&
                  prod - dx * ONE_PIXEL + dy * ONE_PIXEL >  0 ) /* up */
        {
          prod -= dx * ONE_PIXEL;
          fx2 = (TPos)PVG_FT_UDIV( -prod, dy );
          fy2 = ONE_PIXEL;
          ras.cover += ( fy2 - fy1 );
          ras.area  += ( fy2 - fy1 ) * ( fx1 + fx2 );
          fx1 = fx2;
          fy1 = 0;
          ey1++;
        }
        else if ( prod - dx * ONE_PIXEL + dy * ONE_PIXEL <= 0 &&
                  prod                  + dy * ONE_PIXEL >= 0 ) /* right */
        {
          prod += dy * ONE_PIXEL;
          fx2 = ONE_PIXEL;
          fy2 = (TPos)PVG_FT_UDIV( prod, dx );
          ras.cover += ( fy2 - fy1 );
          ras.area  += ( fy2 - fy1 ) * ( fx1 + fx2 );
          fx1 = 0;
          fy1 = fy2;
          ex1++;
        }
        else /* ( prod                  + dy * ONE_PIXEL <  0 &&
                  prod                                   >  0 )    down */
        {
          fx2 = (TPos)PVG_FT_UDIV( prod, -dy );
          fy2 = 0;
          prod += dx * ONE_PIXEL;
          ras.cover += ( fy2 - fy1 );
          ras.area  += ( fy2 - fy1 ) * ( fx1 + fx2 );
          fx1 = fx2;
          fy1 = ONE_PIXEL;
          ey1--;
        }

        gray_set_cell( RAS_VAR_ ex1, ey1 );
      } while ( ex1 != ex2 || ey1 != ey2 );
    }

    fx2 = FRACT( to_x );
    fy2 = FRACT( to_y );

    ras.cover += ( fy2 - fy1 );
    ras.area  += ( fy2 - fy1 ) * ( fx1 + fx2 );

  End:
    ras.x       = to_x;
    ras.y       = to_y;
  }

#endif

  static void
  gray_split_conic( PVG_FT_Vector*  base )
  {
    TPos  a, b;


    base[4].x = base[2].x;
    b = base[1].x;
    a = base[3].x = ( base[2].x + b ) / 2;
    b = base[1].x = ( base[0].x + b ) / 2;
    base[2].x = ( a + b ) / 2;

    base[4].y = base[2].y;
    b = base[1].y;
    a = base[3].y = ( base[2].y + b ) / 2;
    b = base[1].y = ( base[0].y + b ) / 2;
    base[2].y = ( a + b ) / 2;
  }


  static void
  gray_render_conic( RAS_ARG_ const PVG_FT_Vector*  control,
                              const PVG_FT_Vector*  to )
  {
    PVG_FT_Vector   bez_stack[16 * 2 + 1];  /* enough to accommodate bisections */
    PVG_FT_Vector*  arc = bez_stack;
    TPos        dx, dy;
    int         draw, split;


    arc[0].x = UPSCALE( to->x );
    arc[0].y = UPSCALE( to->y );
    arc[1].x = UPSCALE( control->x );
    arc[1].y = UPSCALE( control->y );
    arc[2].x = ras.x;
    arc[2].y = ras.y;

    /* short-cut the arc that crosses the current band */
    if ( ( TRUNC( arc[0].y ) >= ras.max_ey &&
           TRUNC( arc[1].y ) >= ras.max_ey &&
           TRUNC( arc[2].y ) >= ras.max_ey ) ||
         ( TRUNC( arc[0].y ) <  ras.min_ey &&
           TRUNC( arc[1].y ) <  ras.min_ey &&
           TRUNC( arc[2].y ) <  ras.min_ey ) )
    {
      ras.x = arc[0].x;
      ras.y = arc[0].y;
      return;
    }

    dx = PVG_FT_ABS( arc[2].x + arc[0].x - 2 * arc[1].x );
    dy = PVG_FT_ABS( arc[2].y + arc[0].y - 2 * arc[1].y );
    if ( dx < dy )
      dx = dy;

    /* We can calculate the number of necessary bisections because  */
    /* each bisection predictably reduces deviation exactly 4-fold. */
    /* Even 32-bit deviation would vanish after 16 bisections.      */
    draw = 1;
    while ( dx > ONE_PIXEL / 4 )
    {
      dx >>= 2;
      draw <<= 1;
    }

    /* We use decrement counter to count the total number of segments */
    /* to draw starting from 2^level. Before each draw we split as    */
    /* many times as there are trailing zeros in the counter.         */
    do
    {
      split = 1;
      while ( ( draw & split ) == 0 )
      {
        gray_split_conic( arc );
        arc += 2;
        split <<= 1;
      }

      gray_render_line( RAS_VAR_ arc[0].x, arc[0].y );
      arc -= 2;

    } while ( --draw );
  }


  static void
  gray_split_cubic( PVG_FT_Vector*  base )
  {
    TPos  a, b, c, d;


    base[6].x = base[3].x;
    c = base[1].x;
    d = base[2].x;
    base[1].x = a = ( base[0].x + c ) / 2;
    base[5].x = b = ( base[3].x + d ) / 2;
    c = ( c + d ) / 2;
    base[2].x = a = ( a + c ) / 2;
    base[4].x = b = ( b + c ) / 2;
    base[3].x = ( a + b ) / 2;

    base[6].y = base[3].y;
    c = base[1].y;
    d = base[2].y;
    base[1].y = a = ( base[0].y + c ) / 2;
    base[5].y = b = ( base[3].y + d ) / 2;
    c = ( c + d ) / 2;
    base[2].y = a = ( a + c ) / 2;
    base[4].y = b = ( b + c ) / 2;
    base[3].y = ( a + b ) / 2;
  }


  static void
  gray_render_cubic( RAS_ARG_ const PVG_FT_Vector*  control1,
                              const PVG_FT_Vector*  control2,
                              const PVG_FT_Vector*  to )
  {
    PVG_FT_Vector   bez_stack[16 * 3 + 1];  /* enough to accommodate bisections */
    PVG_FT_Vector*  arc = bez_stack;
    PVG_FT_Vector*  limit = bez_stack + 45;
    TPos        dx, dy, dx_, dy_;
    TPos        dx1, dy1, dx2, dy2;
    TPos        L, s, s_limit;


    arc[0].x = UPSCALE( to->x );
    arc[0].y = UPSCALE( to->y );
    arc[1].x = UPSCALE( control2->x );
    arc[1].y = UPSCALE( control2->y );
    arc[2].x = UPSCALE( control1->x );
    arc[2].y = UPSCALE( control1->y );
    arc[3].x = ras.x;
    arc[3].y = ras.y;

    /* short-cut the arc that crosses the current band */
    if ( ( TRUNC( arc[0].y ) >= ras.max_ey &&
           TRUNC( arc[1].y ) >= ras.max_ey &&
           TRUNC( arc[2].y ) >= ras.max_ey &&
           TRUNC( arc[3].y ) >= ras.max_ey ) ||
         ( TRUNC( arc[0].y ) <  ras.min_ey &&
           TRUNC( arc[1].y ) <  ras.min_ey &&
           TRUNC( arc[2].y ) <  ras.min_ey &&
           TRUNC( arc[3].y ) <  ras.min_ey ) )
    {
      ras.x = arc[0].x;
      ras.y = arc[0].y;
      return;
    }

    for (;;)
    {
      /* Decide whether to split or draw. See `Rapid Termination          */
      /* Evaluation for Recursive Subdivision of Bezier Curves' by Thomas */
      /* F. Hain, at                                                      */
      /* http://www.cis.southalabama.edu/~hain/general/Publications/Bezier/Camera-ready%20CISST02%202.pdf */


      /* dx and dy are x and y components of the P0-P3 chord vector. */
      dx = dx_ = arc[3].x - arc[0].x;
      dy = dy_ = arc[3].y - arc[0].y;

      L = PVG_FT_HYPOT( dx_, dy_ );

      /* Avoid possible arithmetic overflow below by splitting. */
      if ( L >= (1 << 23) )
        goto Split;

      /* Max deviation may be as much as (s/L) * 3/4 (if Hain's v = 1). */
      s_limit = L * (TPos)( ONE_PIXEL / 6 );

      /* s is L * the perpendicular distance from P1 to the line P0-P3. */
      dx1 = arc[1].x - arc[0].x;
      dy1 = arc[1].y - arc[0].y;
      s = PVG_FT_ABS( dy * dx1 - dx * dy1 );

      if ( s > s_limit )
        goto Split;

      /* s is L * the perpendicular distance from P2 to the line P0-P3. */
      dx2 = arc[2].x - arc[0].x;
      dy2 = arc[2].y - arc[0].y;
      s = PVG_FT_ABS( dy * dx2 - dx * dy2 );

      if ( s > s_limit )
        goto Split;

      /* Split super curvy segments where the off points are so far
         from the chord that the angles P0-P1-P3 or P0-P2-P3 become
         acute as detected by appropriate dot products. */
      if ( dx1 * ( dx1 - dx ) + dy1 * ( dy1 - dy ) > 0 ||
           dx2 * ( dx2 - dx ) + dy2 * ( dy2 - dy ) > 0 )
        goto Split;

      gray_render_line( RAS_VAR_ arc[0].x, arc[0].y );

      if ( arc == bez_stack )
        return;

      arc -= 3;
      continue;

    Split:
      if( arc == limit )
        return;
      gray_split_cubic( arc );
      arc += 3;
    }
  }



  static int
  gray_move_to( const PVG_FT_Vector*  to,
                PWorker           worker )
  {
    TPos  x, y;


    /* record current cell, if any */
    if ( !ras.invalid )
      gray_record_cell( worker );

    /* start to a new position */
    x = UPSCALE( to->x );
    y = UPSCALE( to->y );

    gray_start_cell( worker, TRUNC( x ), TRUNC( y ) );

    ras.x = x;
    ras.y = y;
    return 0;
  }


  static void
  gray_hline( RAS_ARG_ TCoord  x,
                       TCoord  y,
                       TPos    area,
                       int     acount )
  {
    int coverage;


    /* compute the coverage line's coverage, depending on the    */
    /* outline fill rule                                         */
    /*                                                           */
    /* the coverage percentage is area/(PIXEL_BITS*PIXEL_BITS*2) */
    /*                                                           */
    coverage = (int)( area >> ( PIXEL_BITS * 2 + 1 - 8 ) );
                                                    /* use range 0..256 */
    if ( coverage < 0 )
      coverage = -coverage;

    if ( ras.outline.flags & PVG_FT_OUTLINE_EVEN_ODD_FILL )
    {
      coverage &= 511;

      if ( coverage > 256 )
        coverage = 512 - coverage;
      else if ( coverage == 256 )
        coverage = 255;
    }
    else
    {
      /* normal non-zero winding rule */
      if ( coverage >= 256 )
        coverage = 255;
    }

    y += (TCoord)ras.min_ey;
    x += (TCoord)ras.min_ex;

    /* PVG_FT_Span.x is an int, so limit our coordinates appropriately */
    if ( x >= (1 << 23) )
      x = (1 << 23) - 1;

    /* PVG_FT_Span.y is an int, so limit our coordinates appropriately */
    if ( y >= (1 << 23) )
      y = (1 << 23) - 1;

    if ( coverage )
    {
      PVG_FT_Span*  span;
      int       count;
      int       skip;

      /* see whether we can add this span to the current list */
      count = ras.num_gray_spans;
      span  = ras.gray_spans + count - 1;
      if ( count > 0                          &&
           span->y == y                       &&
           span->x + span->len == x           &&
           span->coverage == coverage         )
      {
        span->len = span->len + acount;
        return;
      }

      if ( count >= PVG_FT_MAX_GRAY_SPANS )
      {
        if ( ras.render_span && count > ras.skip_spans )
        {
          skip = ras.skip_spans > 0 ? ras.skip_spans : 0;
          ras.render_span( ras.num_gray_spans - skip,
                           ras.gray_spans + skip,
                           ras.render_span_data );
        }

        ras.skip_spans -= ras.num_gray_spans;
        /* ras.render_span( span->y, ras.gray_spans, count ); */
        ras.num_gray_spans = 0;

        span  = ras.gray_spans;
      }
      else
        span++;

      /* add a gray span to the current list */
      span->x        = x;
      span->len      = acount;
      span->y        = y;
      span->coverage = (unsigned char)coverage;

      ras.num_gray_spans++;
    }
  }



  static void
  gray_sweep( RAS_ARG)
  {
    int  yindex;

    if ( ras.num_cells == 0 )
      return;

    for ( yindex = 0; yindex < ras.ycount; yindex++ )
    {
      PCell   cell  = ras.ycells[yindex];
      TCoord  cover = 0;
      TCoord  x     = 0;


      for ( ; cell != NULL; cell = cell->next )
      {
        TArea  area;


        if ( cell->x > x && cover != 0 )
          gray_hline( RAS_VAR_ x, yindex, cover * ( ONE_PIXEL * 2 ),
                      cell->x - x );

        cover += cell->cover;
        area   = cover * ( ONE_PIXEL * 2 ) - cell->area;

        if ( area != 0 && cell->x >= 0 )
          gray_hline( RAS_VAR_ cell->x, yindex, area, 1 );

        x = cell->x + 1;
      }

      if ( ras.count_ex > x && cover != 0 )
        gray_hline( RAS_VAR_ x, yindex, cover * ( ONE_PIXEL * 2 ),
                    ras.count_ex - x );
    }
  }

PVG_FT_Error PVG_FT_Outline_Check(PVG_FT_Outline* outline)
{
    if (outline) {
        PVG_FT_Int n_points = outline->n_points;
        PVG_FT_Int n_contours = outline->n_contours;
        PVG_FT_Int end0, end;
        PVG_FT_Int n;

        /* empty glyph? */
        if (n_points == 0 && n_contours == 0) return 0;

        /* check point and contour counts */
        if (n_points <= 0 || n_contours <= 0) goto Bad;

        end0 = end = -1;
        for (n = 0; n < n_contours; n++) {
            end = outline->contours[n];

            /* note that we don't accept empty contours */
            if (end <= end0 || end >= n_points) goto Bad;

            end0 = end;
        }

        if (end != n_points - 1) goto Bad;

        /* XXX: check the tags array */
        return 0;
    }

Bad:
    return ErrRaster_Invalid_Outline;
}

void PVG_FT_Outline_Get_CBox(const PVG_FT_Outline* outline, PVG_FT_BBox* acbox)
{
    PVG_FT_Pos xMin, yMin, xMax, yMax;

    if (outline && acbox) {
        if (outline->n_points == 0) {
            xMin = 0;
            yMin = 0;
            xMax = 0;
            yMax = 0;
        } else {
            PVG_FT_Vector* vec = outline->points;
            PVG_FT_Vector* limit = vec + outline->n_points;

            xMin = xMax = vec->x;
            yMin = yMax = vec->y;
            vec++;

            for (; vec < limit; vec++) {
                PVG_FT_Pos x, y;

                x = vec->x;
                if (x < xMin) xMin = x;
                if (x > xMax) xMax = x;

                y = vec->y;
                if (y < yMin) yMin = y;
                if (y > yMax) yMax = y;
            }
        }
        acbox->xMin = xMin;
        acbox->xMax = xMax;
        acbox->yMin = yMin;
        acbox->yMax = yMax;
    }
}

  /*************************************************************************/
  /*                                                                       */
  /*  The following function should only compile in stand_alone mode,      */
  /*  i.e., when building this component without the rest of FreeType.     */
  /*                                                                       */
  /*************************************************************************/

  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    PVG_FT_Outline_Decompose                                               */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Walks over an outline's structure to decompose it into individual  */
  /*    segments and Bezier arcs.  This function is also able to emit      */
  /*    `move to' and `close to' operations to indicate the start and end  */
  /*    of new contours in the outline.                                    */
  /*                                                                       */
  /* <Input>                                                               */
  /*    outline        :: A pointer to the source target.                  */
  /*                                                                       */
  /*    user           :: A typeless pointer which is passed to each       */
  /*                      emitter during the decomposition.  It can be     */
  /*                      used to store the state during the               */
  /*                      decomposition.                                   */
  /*                                                                       */
  /* <Return>                                                              */
  /*    Error code.  0 means success.                                      */
  /*                                                                       */
  static
  int  PVG_FT_Outline_Decompose( const PVG_FT_Outline*        outline,
                                void*                       user )
  {
#undef SCALED
#define SCALED( x )  (x)

    PVG_FT_Vector   v_last;
    PVG_FT_Vector   v_control;
    PVG_FT_Vector   v_start;

    PVG_FT_Vector*  point;
    PVG_FT_Vector*  limit;
    char*       tags;

    int   n;         /* index of contour in outline     */
    int   first;     /* index of first point in contour */
    int   error;
    char  tag;       /* current point's state           */

    if ( !outline )
      return ErrRaster_Invalid_Outline;

    first = 0;

    for ( n = 0; n < outline->n_contours; n++ )
    {
      int  last;  /* index of last point in contour */


      last  = outline->contours[n];
      if ( last < 0 )
        goto Invalid_Outline;
      limit = outline->points + last;

      v_start   = outline->points[first];
      v_start.x = SCALED( v_start.x );
      v_start.y = SCALED( v_start.y );

      v_last   = outline->points[last];
      v_last.x = SCALED( v_last.x );
      v_last.y = SCALED( v_last.y );

      v_control = v_start;

      point = outline->points + first;
      tags  = outline->tags  + first;
      tag   = PVG_FT_CURVE_TAG( tags[0] );

      /* A contour cannot start with a cubic control point! */
      if ( tag == PVG_FT_CURVE_TAG_CUBIC )
        goto Invalid_Outline;

      /* check first point to determine origin */
      if ( tag == PVG_FT_CURVE_TAG_CONIC )
      {
        /* first point is conic control.  Yes, this happens. */
        if ( PVG_FT_CURVE_TAG( outline->tags[last] ) == PVG_FT_CURVE_TAG_ON )
        {
          /* start at last point if it is on the curve */
          v_start = v_last;
          limit--;
        }
        else
        {
          /* if both first and last points are conic,         */
          /* start at their middle and record its position    */
          /* for closure                                      */
          v_start.x = ( v_start.x + v_last.x ) / 2;
          v_start.y = ( v_start.y + v_last.y ) / 2;

          v_last = v_start;
        }
        point--;
        tags--;
      }

      error = gray_move_to( &v_start, user );
      if ( error )
        goto Exit;

      while ( point < limit )
      {
        point++;
        tags++;

        tag = PVG_FT_CURVE_TAG( tags[0] );
        switch ( tag )
        {
        case PVG_FT_CURVE_TAG_ON:  /* emit a single line_to */
          {
            PVG_FT_Vector  vec;


            vec.x = SCALED( point->x );
            vec.y = SCALED( point->y );

            gray_render_line(user, UPSCALE(vec.x), UPSCALE(vec.y));
            continue;
          }

        case PVG_FT_CURVE_TAG_CONIC:  /* consume conic arcs */
          {
            v_control.x = SCALED( point->x );
            v_control.y = SCALED( point->y );

          Do_Conic:
            if ( point < limit )
            {
              PVG_FT_Vector  vec;
              PVG_FT_Vector  v_middle;


              point++;
              tags++;
              tag = PVG_FT_CURVE_TAG( tags[0] );

              vec.x = SCALED( point->x );
              vec.y = SCALED( point->y );

              if ( tag == PVG_FT_CURVE_TAG_ON )
              {
                gray_render_conic(user, &v_control, &vec);
                continue;
              }

              if ( tag != PVG_FT_CURVE_TAG_CONIC )
                goto Invalid_Outline;

              v_middle.x = ( v_control.x + vec.x ) / 2;
              v_middle.y = ( v_control.y + vec.y ) / 2;

              gray_render_conic(user, &v_control, &v_middle);

              v_control = vec;
              goto Do_Conic;
            }

            gray_render_conic(user, &v_control, &v_start);
            goto Close;
          }

        default:  /* PVG_FT_CURVE_TAG_CUBIC */
          {
            PVG_FT_Vector  vec1, vec2;


            if ( point + 1 > limit                             ||
                 PVG_FT_CURVE_TAG( tags[1] ) != PVG_FT_CURVE_TAG_CUBIC )
              goto Invalid_Outline;

            point += 2;
            tags  += 2;

            vec1.x = SCALED( point[-2].x );
            vec1.y = SCALED( point[-2].y );

            vec2.x = SCALED( point[-1].x );
            vec2.y = SCALED( point[-1].y );

            if ( point <= limit )
            {
              PVG_FT_Vector  vec;


              vec.x = SCALED( point->x );
              vec.y = SCALED( point->y );

              gray_render_cubic(user, &vec1, &vec2, &vec);
              continue;
            }

            gray_render_cubic(user, &vec1, &vec2, &v_start);
            goto Close;
          }
        }
      }

      /* close the contour with a line segment */
      gray_render_line(user, UPSCALE(v_start.x), UPSCALE(v_start.y));

   Close:
      first = last + 1;
    }

    return 0;

  Exit:
    return error;

  Invalid_Outline:
    return ErrRaster_Invalid_Outline;
  }

  typedef struct  TBand_
  {
    TPos  min, max;

  } TBand;

  static int
  gray_convert_glyph_inner( RAS_ARG )
  {
    volatile int  error = 0;

    if ( pvg_ft_setjmp( ras.jump_buffer ) == 0 )
    {
      error = PVG_FT_Outline_Decompose( &ras.outline, &ras );
      if ( !ras.invalid )
        gray_record_cell( RAS_VAR );
    }
    else
    {
      error = ErrRaster_Memory_Overflow;
    }

    return error;
  }


  static int
  gray_convert_glyph( RAS_ARG )
  {
    TBand            bands[40];
    TBand* volatile  band;
    int volatile     n, num_bands;
    TPos volatile    min, max, max_y;
    PVG_FT_BBox*      clip;
    int              skip;

    ras.num_gray_spans = 0;

    /* Set up state in the raster object */
    gray_compute_cbox( RAS_VAR );

    /* clip to target bitmap, exit if nothing to do */
    clip = &ras.clip_box;

    if ( ras.max_ex <= clip->xMin || ras.min_ex >= clip->xMax ||
         ras.max_ey <= clip->yMin || ras.min_ey >= clip->yMax )
      return 0;

    if ( ras.min_ex < clip->xMin ) ras.min_ex = clip->xMin;
    if ( ras.min_ey < clip->yMin ) ras.min_ey = clip->yMin;

    if ( ras.max_ex > clip->xMax ) ras.max_ex = clip->xMax;
    if ( ras.max_ey > clip->yMax ) ras.max_ey = clip->yMax;

    ras.count_ex = ras.max_ex - ras.min_ex;
    ras.count_ey = ras.max_ey - ras.min_ey;

    /* set up vertical bands */
    num_bands = (int)( ( ras.max_ey - ras.min_ey ) / ras.band_size );
    if ( num_bands == 0 )
      num_bands = 1;
    if ( num_bands >= 39 )
      num_bands = 39;

    ras.band_shoot = 0;

    min   = ras.min_ey;
    max_y = ras.max_ey;

    for ( n = 0; n < num_bands; n++, min = max )
    {
      max = min + ras.band_size;
      if ( n == num_bands - 1 || max > max_y )
        max = max_y;

      bands[0].min = min;
      bands[0].max = max;
      band         = bands;

      while ( band >= bands )
      {
        TPos  bottom, top, middle;
        int   error;

        {
          PCell  cells_max;
          int    yindex;
          int    cell_start, cell_end, cell_mod;


          ras.ycells = (PCell*)ras.buffer;
          ras.ycount = band->max - band->min;

          cell_start = sizeof ( PCell ) * ras.ycount;
          cell_mod   = cell_start % sizeof ( TCell );
          if ( cell_mod > 0 )
            cell_start += sizeof ( TCell ) - cell_mod;

          cell_end  = ras.buffer_size;
          cell_end -= cell_end % sizeof( TCell );

          cells_max = (PCell)( (char*)ras.buffer + cell_end );
          ras.cells = (PCell)( (char*)ras.buffer + cell_start );
          if ( ras.cells >= cells_max )
            goto ReduceBands;

          ras.max_cells = (int)(cells_max - ras.cells);
          if ( ras.max_cells < 2 )
            goto ReduceBands;

          for ( yindex = 0; yindex < ras.ycount; yindex++ )
            ras.ycells[yindex] = NULL;
        }

        ras.num_cells = 0;
        ras.invalid   = 1;
        ras.min_ey    = band->min;
        ras.max_ey    = band->max;
        ras.count_ey  = band->max - band->min;

        error = gray_convert_glyph_inner( RAS_VAR );

        if ( !error )
        {
          gray_sweep( RAS_VAR);
          band--;
          continue;
        }
        else if ( error != ErrRaster_Memory_Overflow )
          return 1;

      ReduceBands:
        /* render pool overflow; we will reduce the render band by half */
        bottom = band->min;
        top    = band->max;
        middle = bottom + ( ( top - bottom ) >> 1 );

        /* This is too complex for a single scanline; there must */
        /* be some problems.                                     */
        if ( middle == bottom )
        {
          return ErrRaster_OutOfMemory;
        }

        if ( bottom-top >= ras.band_size )
          ras.band_shoot++;

        band[1].min = bottom;
        band[1].max = middle;
        band[0].min = middle;
        band[0].max = top;
        band++;
      }
    }

    if ( ras.render_span && ras.num_gray_spans > ras.skip_spans )
    {
        skip = ras.skip_spans > 0 ? ras.skip_spans : 0;
        ras.render_span( ras.num_gray_spans - skip,
                         ras.gray_spans + skip,
                         ras.render_span_data );
    }

    ras.skip_spans -= ras.num_gray_spans;

    if ( ras.band_shoot > 8 && ras.band_size > 16 )
      ras.band_size = ras.band_size / 2;

    return 0;
  }


  static int
  gray_raster_render( RAS_ARG_ void* buffer, long buffer_size,
                      const PVG_FT_Raster_Params*  params )
  {
    const PVG_FT_Outline*  outline    = (const PVG_FT_Outline*)params->source;
    if ( outline == NULL )
      return ErrRaster_Invalid_Outline;

    /* return immediately if the outline is empty */
    if ( outline->n_points == 0 || outline->n_contours <= 0 )
      return 0;

    if ( !outline->contours || !outline->points )
      return ErrRaster_Invalid_Outline;

    if ( outline->n_points !=
           outline->contours[outline->n_contours - 1] + 1 )
      return ErrRaster_Invalid_Outline;

    /* this version does not support monochrome rendering */
    if ( !( params->flags & PVG_FT_RASTER_FLAG_AA ) )
      return ErrRaster_Invalid_Mode;

    if ( !( params->flags & PVG_FT_RASTER_FLAG_DIRECT ) )
      return ErrRaster_Invalid_Mode;

    /* compute clipping box */
    if ( params->flags & PVG_FT_RASTER_FLAG_CLIP )
    {
      ras.clip_box = params->clip_box;
    }
    else
    {
      ras.clip_box.xMin = -(1 << 23);
      ras.clip_box.yMin = -(1 << 23);
      ras.clip_box.xMax =  (1 << 23) - 1;
      ras.clip_box.yMax =  (1 << 23) - 1;
    }

    gray_init_cells( RAS_VAR_ buffer, buffer_size );

    ras.outline   = *outline;
    ras.num_cells = 0;
    ras.invalid   = 1;
    ras.band_size = (int)(buffer_size / (long)(sizeof(TCell) * 8));

    ras.render_span      = (PVG_FT_Raster_Span_Func)params->gray_spans;
    ras.render_span_data = params->user;

    return gray_convert_glyph( RAS_VAR );
  }

  void
  PVG_FT_Raster_Render(const PVG_FT_Raster_Params *params)
  {
      char stack[PVG_FT_MINIMUM_POOL_SIZE];
      size_t length = PVG_FT_MINIMUM_POOL_SIZE;

      TWorker worker;
      worker.skip_spans = 0;
      int rendered_spans = 0;
      int error = gray_raster_render(&worker, stack, length, params);
      while(error == ErrRaster_OutOfMemory) {
          if(worker.skip_spans < 0)
              rendered_spans += -worker.skip_spans;
          worker.skip_spans = rendered_spans;
          length *= 2;
          void* heap = malloc(length);
          error = gray_raster_render(&worker, heap, length, params);
          free(heap);
      }
  }

/* END */

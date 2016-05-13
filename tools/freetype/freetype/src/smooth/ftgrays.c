/***************************************************************************/
/*                                                                         */
/*  ftgrays.c                                                              */
/*                                                                         */
/*    A new `perfect' anti-aliasing renderer (body).                       */
/*                                                                         */
/*  Copyright 2000-2003, 2005-2013 by                                      */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

  /*************************************************************************/
  /*                                                                       */
  /* This file can be compiled without the rest of the FreeType engine, by */
  /* defining the _STANDALONE_ macro when compiling it.  You also need to  */
  /* put the files `ftgrays.h' and `ftimage.h' into the current            */
  /* compilation directory.  Typically, you could do something like        */
  /*                                                                       */
  /* - copy `src/smooth/ftgrays.c' (this file) to your current directory   */
  /*                                                                       */
  /* - copy `include/freetype/ftimage.h' and `src/smooth/ftgrays.h' to the */
  /*   same directory                                                      */
  /*                                                                       */
  /* - compile `ftgrays' with the _STANDALONE_ macro defined, as in        */
  /*                                                                       */
  /*     cc -c -D_STANDALONE_ ftgrays.c                                    */
  /*                                                                       */
  /* The renderer can be initialized with a call to                        */
  /* `ft_gray_raster.raster_new'; an anti-aliased bitmap can be generated  */
  /* with a call to `ft_gray_raster.raster_render'.                        */
  /*                                                                       */
  /* See the comments and documentation in the file `ftimage.h' for more   */
  /* details on how the raster works.                                      */
  /*                                                                       */
  /*************************************************************************/

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


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_smooth


#ifdef _STANDALONE_


  /* Auxiliary macros for token concatenation. */
#define FT_ERR_XCAT( x, y )  x ## y
#define FT_ERR_CAT( x, y )   FT_ERR_XCAT( x, y )


  /* define this to dump debugging information */
/* #define FT_DEBUG_LEVEL_TRACE */


#ifdef FT_DEBUG_LEVEL_TRACE
#include <stdio.h>
#include <stdarg.h>
#endif

#include <stddef.h>
#include <string.h>
#include <setjmp.h>
#include <limits.h>
#define FT_UINT_MAX  UINT_MAX
#define FT_INT_MAX   INT_MAX

#define ft_memset   memset

#define ft_setjmp   setjmp
#define ft_longjmp  longjmp
#define ft_jmp_buf  jmp_buf

typedef ptrdiff_t  FT_PtrDist;


#define ErrRaster_Invalid_Mode      -2
#define ErrRaster_Invalid_Outline   -1
#define ErrRaster_Invalid_Argument  -3
#define ErrRaster_Memory_Overflow   -4

#define FT_BEGIN_HEADER
#define FT_END_HEADER

#include "ftimage.h"
#include "ftgrays.h"


  /* This macro is used to indicate that a function parameter is unused. */
  /* Its purpose is simply to reduce compiler warnings.  Note also that  */
  /* simply defining it as `(void)x' doesn't avoid warnings with certain */
  /* ANSI compilers (e.g. LCC).                                          */
#define FT_UNUSED( x )  (x) = (x)


  /* we only use level 5 & 7 tracing messages; cf. ftdebug.h */

#ifdef FT_DEBUG_LEVEL_TRACE

  void
  FT_Message( const char*  fmt,
              ... )
  {
    va_list  ap;


    va_start( ap, fmt );
    vfprintf( stderr, fmt, ap );
    va_end( ap );
  }


  /* empty function useful for setting a breakpoint to catch errors */
  int
  FT_Throw( int          error,
            int          line,
            const char*  file )
  {
    FT_UNUSED( error );
    FT_UNUSED( line );
    FT_UNUSED( file );

    return 0;
  }


  /* we don't handle tracing levels in stand-alone mode; */
#ifndef FT_TRACE5
#define FT_TRACE5( varformat )  FT_Message varformat
#endif
#ifndef FT_TRACE7
#define FT_TRACE7( varformat )  FT_Message varformat
#endif
#ifndef FT_ERROR
#define FT_ERROR( varformat )   FT_Message varformat
#endif

#define FT_THROW( e )                               \
          ( FT_Throw( FT_ERR_CAT( ErrRaster, e ),   \
                      __LINE__,                     \
                      __FILE__ )                  | \
            FT_ERR_CAT( ErrRaster, e )            )

#else /* !FT_DEBUG_LEVEL_TRACE */

#define FT_TRACE5( x )  do { } while ( 0 )     /* nothing */
#define FT_TRACE7( x )  do { } while ( 0 )     /* nothing */
#define FT_ERROR( x )   do { } while ( 0 )     /* nothing */
#define FT_THROW( e )   FT_ERR_CAT( ErrRaster_, e )


#endif /* !FT_DEBUG_LEVEL_TRACE */


#define FT_DEFINE_OUTLINE_FUNCS( class_,               \
                                 move_to_, line_to_,   \
                                 conic_to_, cubic_to_, \
                                 shift_, delta_ )      \
          static const FT_Outline_Funcs class_ =       \
          {                                            \
            move_to_,                                  \
            line_to_,                                  \
            conic_to_,                                 \
            cubic_to_,                                 \
            shift_,                                    \
            delta_                                     \
         };

#define FT_DEFINE_RASTER_FUNCS( class_, glyph_format_,            \
                                raster_new_, raster_reset_,       \
                                raster_set_mode_, raster_render_, \
                                raster_done_ )                    \
          const FT_Raster_Funcs class_ =                          \
          {                                                       \
            glyph_format_,                                        \
            raster_new_,                                          \
            raster_reset_,                                        \
            raster_set_mode_,                                     \
            raster_render_,                                       \
            raster_done_                                          \
         };


#else /* !_STANDALONE_ */


#include <ft2build.h>
#include "ftgrays.h"
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_DEBUG_H
#include FT_OUTLINE_H

#include "ftsmerrs.h"

#include "ftspic.h"

#define Smooth_Err_Invalid_Mode     Smooth_Err_Cannot_Render_Glyph
#define Smooth_Err_Memory_Overflow  Smooth_Err_Out_Of_Memory
#define ErrRaster_Memory_Overflow   Smooth_Err_Out_Of_Memory


#endif /* !_STANDALONE_ */


#ifndef FT_MEM_SET
#define FT_MEM_SET( d, s, c )  ft_memset( d, s, c )
#endif

#ifndef FT_MEM_ZERO
#define FT_MEM_ZERO( dest, count )  FT_MEM_SET( dest, 0, count )
#endif

  /* as usual, for the speed hungry :-) */

#undef RAS_ARG
#undef RAS_ARG_
#undef RAS_VAR
#undef RAS_VAR_

#ifndef FT_STATIC_RASTER

#define RAS_ARG   gray_PWorker  worker
#define RAS_ARG_  gray_PWorker  worker,

#define RAS_VAR   worker
#define RAS_VAR_  worker,

#else /* FT_STATIC_RASTER */

#define RAS_ARG   /* empty */
#define RAS_ARG_  /* empty */
#define RAS_VAR   /* empty */
#define RAS_VAR_  /* empty */

#endif /* FT_STATIC_RASTER */


  /* must be at least 6 bits! */
#define PIXEL_BITS  8

#undef FLOOR
#undef CEILING
#undef TRUNC
#undef SCALED

#define ONE_PIXEL       ( 1L << PIXEL_BITS )
#define PIXEL_MASK      ( -1L << PIXEL_BITS )
#define TRUNC( x )      ( (TCoord)( (x) >> PIXEL_BITS ) )
#define SUBPIXELS( x )  ( (TPos)(x) << PIXEL_BITS )
#define FLOOR( x )      ( (x) & -ONE_PIXEL )
#define CEILING( x )    ( ( (x) + ONE_PIXEL - 1 ) & -ONE_PIXEL )
#define ROUND( x )      ( ( (x) + ONE_PIXEL / 2 ) & -ONE_PIXEL )

#if PIXEL_BITS >= 6
#define UPSCALE( x )    ( (x) << ( PIXEL_BITS - 6 ) )
#define DOWNSCALE( x )  ( (x) >> ( PIXEL_BITS - 6 ) )
#else
#define UPSCALE( x )    ( (x) >> ( 6 - PIXEL_BITS ) )
#define DOWNSCALE( x )  ( (x) << ( 6 - PIXEL_BITS ) )
#endif


  /*************************************************************************/
  /*                                                                       */
  /*   TYPE DEFINITIONS                                                    */
  /*                                                                       */

  /* don't change the following types to FT_Int or FT_Pos, since we might */
  /* need to define them to "float" or "double" when experimenting with   */
  /* new algorithms                                                       */

  typedef long  TCoord;   /* integer scanline/pixel coordinate */
  typedef long  TPos;     /* sub-pixel coordinate              */

  /* determine the type used to store cell areas.  This normally takes at */
  /* least PIXEL_BITS*2 + 1 bits.  On 16-bit systems, we need to use      */
  /* `long' instead of `int', otherwise bad things happen                 */

#if PIXEL_BITS <= 7

  typedef int  TArea;

#else /* PIXEL_BITS >= 8 */

  /* approximately determine the size of integers using an ANSI-C header */
#if FT_UINT_MAX == 0xFFFFU
  typedef long  TArea;
#else
  typedef int   TArea;
#endif

#endif /* PIXEL_BITS >= 8 */


  /* maximum number of gray spans in a call to the span callback */
#define FT_MAX_GRAY_SPANS  32


  typedef struct TCell_*  PCell;

  typedef struct  TCell_
  {
    TPos    x;     /* same with gray_TWorker.ex    */
    TCoord  cover; /* same with gray_TWorker.cover */
    TArea   area;
    PCell   next;

  } TCell;


#if defined( _MSC_VER )      /* Visual C++ (and Intel C++) */
  /* We disable the warning `structure was padded due to   */
  /* __declspec(align())' in order to compile cleanly with */
  /* the maximum level of warnings.                        */
#pragma warning( push )
#pragma warning( disable : 4324 )
#endif /* _MSC_VER */

  typedef struct  gray_TWorker_
  {
    TCoord  ex, ey;
    TPos    min_ex, max_ex;
    TPos    min_ey, max_ey;
    TPos    count_ex, count_ey;

    TArea   area;
    TCoord  cover;
    int     invalid;

    PCell       cells;
    FT_PtrDist  max_cells;
    FT_PtrDist  num_cells;

    TCoord  cx, cy;
    TPos    x,  y;

    TPos    last_ey;

    FT_Vector   bez_stack[32 * 3 + 1];
    int         lev_stack[32];

    FT_Outline  outline;
    FT_Bitmap   target;
    FT_BBox     clip_box;

    FT_Span     gray_spans[FT_MAX_GRAY_SPANS];
    int         num_gray_spans;

    FT_Raster_Span_Func  render_span;
    void*                render_span_data;
    int                  span_y;

    int  band_size;
    int  band_shoot;

    ft_jmp_buf  jump_buffer;

    void*       buffer;
    long        buffer_size;

    PCell*     ycells;
    TPos       ycount;

  } gray_TWorker, *gray_PWorker;

#if defined( _MSC_VER )
#pragma warning( pop )
#endif


#ifndef FT_STATIC_RASTER
#define ras  (*worker)
#else
  static gray_TWorker  ras;
#endif


  typedef struct gray_TRaster_
  {
    void*         buffer;
    long          buffer_size;
    int           band_size;
    void*         memory;
    gray_PWorker  worker;

  } gray_TRaster, *gray_PRaster;



  /*************************************************************************/
  /*                                                                       */
  /* Initialize the cells table.                                           */
  /*                                                                       */
  static void
  gray_init_cells( RAS_ARG_ void*  buffer,
                   long            byte_size )
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
    FT_Outline*  outline = &ras.outline;
    FT_Vector*   vec     = outline->points;
    FT_Vector*   limit   = vec + outline->n_points;


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
      ft_longjmp( ras.jump_buffer, 1 );

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
    if ( !ras.invalid && ( ras.area | ras.cover ) )
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
    }

    ras.ex      = ex;
    ras.ey      = ey;
    ras.invalid = ( (unsigned)ey >= (unsigned)ras.count_ey ||
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
    ras.last_ey = SUBPIXELS( ey );
    ras.invalid = 0;

    gray_set_cell( RAS_VAR_ ex, ey );
  }


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
    TCoord  ex1, ex2, fx1, fx2, delta, mod;
    long    p, first, dx;
    int     incr;


    dx = x2 - x1;

    ex1 = TRUNC( x1 );
    ex2 = TRUNC( x2 );
    fx1 = (TCoord)( x1 - SUBPIXELS( ex1 ) );
    fx2 = (TCoord)( x2 - SUBPIXELS( ex2 ) );

    /* trivial case.  Happens often */
    if ( y1 == y2 )
    {
      gray_set_cell( RAS_VAR_ ex2, ey );
      return;
    }

    /* everything is located in a single cell.  That is easy! */
    /*                                                        */
    if ( ex1 == ex2 )
    {
      delta      = y2 - y1;
      ras.area  += (TArea)(( fx1 + fx2 ) * delta);
      ras.cover += delta;
      return;
    }

    /* ok, we'll have to render a run of adjacent cells on the same */
    /* scanline...                                                  */
    /*                                                              */
    p     = ( ONE_PIXEL - fx1 ) * ( y2 - y1 );
    first = ONE_PIXEL;
    incr  = 1;

    if ( dx < 0 )
    {
      p     = fx1 * ( y2 - y1 );
      first = 0;
      incr  = -1;
      dx    = -dx;
    }

    delta = (TCoord)( p / dx );
    mod   = (TCoord)( p % dx );
    if ( mod < 0 )
    {
      delta--;
      mod += (TCoord)dx;
    }

    ras.area  += (TArea)(( fx1 + first ) * delta);
    ras.cover += delta;

    ex1 += incr;
    gray_set_cell( RAS_VAR_ ex1, ey );
    y1  += delta;

    if ( ex1 != ex2 )
    {
      TCoord  lift, rem;


      p    = ONE_PIXEL * ( y2 - y1 + delta );
      lift = (TCoord)( p / dx );
      rem  = (TCoord)( p % dx );
      if ( rem < 0 )
      {
        lift--;
        rem += (TCoord)dx;
      }

      mod -= (int)dx;

      while ( ex1 != ex2 )
      {
        delta = lift;
        mod  += rem;
        if ( mod >= 0 )
        {
          mod -= (TCoord)dx;
          delta++;
        }

        ras.area  += (TArea)(ONE_PIXEL * delta);
        ras.cover += delta;
        y1        += delta;
        ex1       += incr;
        gray_set_cell( RAS_VAR_ ex1, ey );
      }
    }

    delta      = y2 - y1;
    ras.area  += (TArea)(( fx2 + ONE_PIXEL - first ) * delta);
    ras.cover += delta;
  }


  /*************************************************************************/
  /*                                                                       */
  /* Render a given line as a series of scanlines.                         */
  /*                                                                       */
  static void
  gray_render_line( RAS_ARG_ TPos  to_x,
                             TPos  to_y )
  {
    TCoord  ey1, ey2, fy1, fy2, mod;
    TPos    dx, dy, x, x2;
    long    p, first;
    int     delta, rem, lift, incr;


    ey1 = TRUNC( ras.last_ey );
    ey2 = TRUNC( to_y );     /* if (ey2 >= ras.max_ey) ey2 = ras.max_ey-1; */
    fy1 = (TCoord)( ras.y - ras.last_ey );
    fy2 = (TCoord)( to_y - SUBPIXELS( ey2 ) );

    dx = to_x - ras.x;
    dy = to_y - ras.y;

    /* XXX: we should do something about the trivial case where dx == 0, */
    /*      as it happens very often!                                    */

    /* perform vertical clipping */
    {
      TCoord  min, max;


      min = ey1;
      max = ey2;
      if ( ey1 > ey2 )
      {
        min = ey2;
        max = ey1;
      }
      if ( min >= ras.max_ey || max < ras.min_ey )
        goto End;
    }

    /* everything is on a single scanline */
    if ( ey1 == ey2 )
    {
      gray_render_scanline( RAS_VAR_ ey1, ras.x, fy1, to_x, fy2 );
      goto End;
    }

    /* vertical line - avoid calling gray_render_scanline */
    incr = 1;

    if ( dx == 0 )
    {
      TCoord  ex     = TRUNC( ras.x );
      TCoord  two_fx = (TCoord)( ( ras.x - SUBPIXELS( ex ) ) << 1 );
      TArea   area;


      first = ONE_PIXEL;
      if ( dy < 0 )
      {
        first = 0;
        incr  = -1;
      }

      delta      = (int)( first - fy1 );
      ras.area  += (TArea)two_fx * delta;
      ras.cover += delta;
      ey1       += incr;

      gray_set_cell( RAS_VAR_ ex, ey1 );

      delta = (int)( first + first - ONE_PIXEL );
      area  = (TArea)two_fx * delta;
      while ( ey1 != ey2 )
      {
        ras.area  += area;
        ras.cover += delta;
        ey1       += incr;

        gray_set_cell( RAS_VAR_ ex, ey1 );
      }

      delta      = (int)( fy2 - ONE_PIXEL + first );
      ras.area  += (TArea)two_fx * delta;
      ras.cover += delta;

      goto End;
    }

    /* ok, we have to render several scanlines */
    p     = ( ONE_PIXEL - fy1 ) * dx;
    first = ONE_PIXEL;
    incr  = 1;

    if ( dy < 0 )
    {
      p     = fy1 * dx;
      first = 0;
      incr  = -1;
      dy    = -dy;
    }

    delta = (int)( p / dy );
    mod   = (int)( p % dy );
    if ( mod < 0 )
    {
      delta--;
      mod += (TCoord)dy;
    }

    x = ras.x + delta;
    gray_render_scanline( RAS_VAR_ ey1, ras.x, fy1, x, (TCoord)first );

    ey1 += incr;
    gray_set_cell( RAS_VAR_ TRUNC( x ), ey1 );

    if ( ey1 != ey2 )
    {
      p     = ONE_PIXEL * dx;
      lift  = (int)( p / dy );
      rem   = (int)( p % dy );
      if ( rem < 0 )
      {
        lift--;
        rem += (int)dy;
      }
      mod -= (int)dy;

      while ( ey1 != ey2 )
      {
        delta = lift;
        mod  += rem;
        if ( mod >= 0 )
        {
          mod -= (int)dy;
          delta++;
        }

        x2 = x + delta;
        gray_render_scanline( RAS_VAR_ ey1, x,
                                       (TCoord)( ONE_PIXEL - first ), x2,
                                       (TCoord)first );
        x = x2;

        ey1 += incr;
        gray_set_cell( RAS_VAR_ TRUNC( x ), ey1 );
      }
    }

    gray_render_scanline( RAS_VAR_ ey1, x,
                                   (TCoord)( ONE_PIXEL - first ), to_x,
                                   fy2 );

  End:
    ras.x       = to_x;
    ras.y       = to_y;
    ras.last_ey = SUBPIXELS( ey2 );
  }


  static void
  gray_split_conic( FT_Vector*  base )
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
  gray_render_conic( RAS_ARG_ const FT_Vector*  control,
                              const FT_Vector*  to )
  {
    TPos        dx, dy;
    TPos        min, max, y;
    int         top, level;
    int*        levels;
    FT_Vector*  arc;


    levels = ras.lev_stack;

    arc      = ras.bez_stack;
    arc[0].x = UPSCALE( to->x );
    arc[0].y = UPSCALE( to->y );
    arc[1].x = UPSCALE( control->x );
    arc[1].y = UPSCALE( control->y );
    arc[2].x = ras.x;
    arc[2].y = ras.y;
    top      = 0;

    dx = FT_ABS( arc[2].x + arc[0].x - 2 * arc[1].x );
    dy = FT_ABS( arc[2].y + arc[0].y - 2 * arc[1].y );
    if ( dx < dy )
      dx = dy;

    if ( dx < ONE_PIXEL / 4 )
      goto Draw;

    /* short-cut the arc that crosses the current band */
    min = max = arc[0].y;

    y = arc[1].y;
    if ( y < min ) min = y;
    if ( y > max ) max = y;

    y = arc[2].y;
    if ( y < min ) min = y;
    if ( y > max ) max = y;

    if ( TRUNC( min ) >= ras.max_ey || TRUNC( max ) < ras.min_ey )
      goto Draw;

    level = 0;
    do
    {
      dx >>= 2;
      level++;
    } while ( dx > ONE_PIXEL / 4 );

    levels[0] = level;

    do
    {
      level = levels[top];
      if ( level > 0 )
      {
        gray_split_conic( arc );
        arc += 2;
        top++;
        levels[top] = levels[top - 1] = level - 1;
        continue;
      }

    Draw:
      gray_render_line( RAS_VAR_ arc[0].x, arc[0].y );
      top--;
      arc -= 2;

    } while ( top >= 0 );
  }


  static void
  gray_split_cubic( FT_Vector*  base )
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
  gray_render_cubic( RAS_ARG_ const FT_Vector*  control1,
                              const FT_Vector*  control2,
                              const FT_Vector*  to )
  {
    FT_Vector*  arc;
    TPos        min, max, y;


    arc      = ras.bez_stack;
    arc[0].x = UPSCALE( to->x );
    arc[0].y = UPSCALE( to->y );
    arc[1].x = UPSCALE( control2->x );
    arc[1].y = UPSCALE( control2->y );
    arc[2].x = UPSCALE( control1->x );
    arc[2].y = UPSCALE( control1->y );
    arc[3].x = ras.x;
    arc[3].y = ras.y;

    /* Short-cut the arc that crosses the current band. */
    min = max = arc[0].y;

    y = arc[1].y;
    if ( y < min )
      min = y;
    if ( y > max )
      max = y;

    y = arc[2].y;
    if ( y < min )
      min = y;
    if ( y > max )
      max = y;

    y = arc[3].y;
    if ( y < min )
      min = y;
    if ( y > max )
      max = y;

    if ( TRUNC( min ) >= ras.max_ey || TRUNC( max ) < ras.min_ey )
      goto Draw;

    for (;;)
    {
      /* Decide whether to split or draw. See `Rapid Termination          */
      /* Evaluation for Recursive Subdivision of Bezier Curves' by Thomas */
      /* F. Hain, at                                                      */
      /* http://www.cis.southalabama.edu/~hain/general/Publications/Bezier/Camera-ready%20CISST02%202.pdf */

      {
        TPos  dx, dy, dx_, dy_;
        TPos  dx1, dy1, dx2, dy2;
        TPos  L, s, s_limit;


        /* dx and dy are x and y components of the P0-P3 chord vector. */
        dx = arc[3].x - arc[0].x;
        dy = arc[3].y - arc[0].y;

        /* L is an (under)estimate of the Euclidean distance P0-P3.       */
        /*                                                                */
        /* If dx >= dy, then r = sqrt(dx^2 + dy^2) can be overestimated   */
        /* with least maximum error by                                    */
        /*                                                                */
        /*   r_upperbound = dx + (sqrt(2) - 1) * dy  ,                    */
        /*                                                                */
        /* where sqrt(2) - 1 can be (over)estimated by 107/256, giving an */
        /* error of no more than 8.4%.                                    */
        /*                                                                */
        /* Similarly, some elementary calculus shows that r can be        */
        /* underestimated with least maximum error by                     */
        /*                                                                */
        /*   r_lowerbound = sqrt(2 + sqrt(2)) / 2 * dx                    */
        /*                  + sqrt(2 - sqrt(2)) / 2 * dy  .               */
        /*                                                                */
        /* 236/256 and 97/256 are (under)estimates of the two algebraic   */
        /* numbers, giving an error of no more than 8.1%.                 */

        dx_ = FT_ABS( dx );
        dy_ = FT_ABS( dy );

        /* This is the same as                     */
        /*                                         */
        /*   L = ( 236 * FT_MAX( dx_, dy_ )        */
        /*       + 97 * FT_MIN( dx_, dy_ ) ) >> 8; */
        L = ( dx_ > dy_ ? 236 * dx_ +  97 * dy_
                        :  97 * dx_ + 236 * dy_ ) >> 8;

        /* Avoid possible arithmetic overflow below by splitting. */
        if ( L > 32767 )
          goto Split;

        /* Max deviation may be as much as (s/L) * 3/4 (if Hain's v = 1). */
        s_limit = L * (TPos)( ONE_PIXEL / 6 );

        /* s is L * the perpendicular distance from P1 to the line P0-P3. */
        dx1 = arc[1].x - arc[0].x;
        dy1 = arc[1].y - arc[0].y;
        s = FT_ABS( dy * dx1 - dx * dy1 );

        if ( s > s_limit )
          goto Split;

        /* s is L * the perpendicular distance from P2 to the line P0-P3. */
        dx2 = arc[2].x - arc[0].x;
        dy2 = arc[2].y - arc[0].y;
        s = FT_ABS( dy * dx2 - dx * dy2 );

        if ( s > s_limit )
          goto Split;

        /* Split super curvy segments where the off points are so far
           from the chord that the angles P0-P1-P3 or P0-P2-P3 become
           acute as detected by appropriate dot products. */
        if ( dx1 * ( dx1 - dx ) + dy1 * ( dy1 - dy ) > 0 ||
             dx2 * ( dx2 - dx ) + dy2 * ( dy2 - dy ) > 0 )
          goto Split;

        /* No reason to split. */
        goto Draw;
      }

    Split:
      gray_split_cubic( arc );
      arc += 3;
      continue;

    Draw:
      gray_render_line( RAS_VAR_ arc[0].x, arc[0].y );

      if ( arc == ras.bez_stack )
        return;

      arc -= 3;
    }
  }


  static int
  gray_move_to( const FT_Vector*  to,
                gray_PWorker      worker )
  {
    TPos  x, y;


    /* record current cell, if any */
    gray_record_cell( RAS_VAR );

    /* start to a new position */
    x = UPSCALE( to->x );
    y = UPSCALE( to->y );

    gray_start_cell( RAS_VAR_ TRUNC( x ), TRUNC( y ) );

    worker->x = x;
    worker->y = y;
    return 0;
  }


  static int
  gray_line_to( const FT_Vector*  to,
                gray_PWorker      worker )
  {
    gray_render_line( RAS_VAR_ UPSCALE( to->x ), UPSCALE( to->y ) );
    return 0;
  }


  static int
  gray_conic_to( const FT_Vector*  control,
                 const FT_Vector*  to,
                 gray_PWorker      worker )
  {
    gray_render_conic( RAS_VAR_ control, to );
    return 0;
  }


  static int
  gray_cubic_to( const FT_Vector*  control1,
                 const FT_Vector*  control2,
                 const FT_Vector*  to,
                 gray_PWorker      worker )
  {
    gray_render_cubic( RAS_VAR_ control1, control2, to );
    return 0;
  }


  static void
  gray_render_span( int             y,
                    int             count,
                    const FT_Span*  spans,
                    gray_PWorker    worker )
  {
    unsigned char*  p;
    FT_Bitmap*      map = &worker->target;


    /* first of all, compute the scanline offset */
    p = (unsigned char*)map->buffer - y * map->pitch;
    if ( map->pitch >= 0 )
      p += (unsigned)( ( map->rows - 1 ) * map->pitch );

    for ( ; count > 0; count--, spans++ )
    {
      unsigned char  coverage = spans->coverage;


      if ( coverage )
      {
        /* For small-spans it is faster to do it by ourselves than
         * calling `memset'.  This is mainly due to the cost of the
         * function call.
         */
        if ( spans->len >= 8 )
          FT_MEM_SET( p + spans->x, (unsigned char)coverage, spans->len );
        else
        {
          unsigned char*  q = p + spans->x;


          switch ( spans->len )
          {
          case 7: *q++ = (unsigned char)coverage;
          case 6: *q++ = (unsigned char)coverage;
          case 5: *q++ = (unsigned char)coverage;
          case 4: *q++ = (unsigned char)coverage;
          case 3: *q++ = (unsigned char)coverage;
          case 2: *q++ = (unsigned char)coverage;
          case 1: *q   = (unsigned char)coverage;
          default:
            ;
          }
        }
      }
    }
  }


  static void
  gray_hline( RAS_ARG_ TCoord  x,
                       TCoord  y,
                       TPos    area,
                       TCoord  acount )
  {
    int  coverage;


    /* compute the coverage line's coverage, depending on the    */
    /* outline fill rule                                         */
    /*                                                           */
    /* the coverage percentage is area/(PIXEL_BITS*PIXEL_BITS*2) */
    /*                                                           */
    coverage = (int)( area >> ( PIXEL_BITS * 2 + 1 - 8 ) );
                                                    /* use range 0..256 */
    if ( coverage < 0 )
      coverage = -coverage;

    if ( ras.outline.flags & FT_OUTLINE_EVEN_ODD_FILL )
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

    /* FT_Span.x is a 16-bit short, so limit our coordinates appropriately */
    if ( x >= 32767 )
      x = 32767;

    /* FT_Span.y is an integer, so limit our coordinates appropriately */
    if ( y >= FT_INT_MAX )
      y = FT_INT_MAX;

    if ( coverage )
    {
      FT_Span*  span;
      int       count;


      /* see whether we can add this span to the current list */
      count = ras.num_gray_spans;
      span  = ras.gray_spans + count - 1;
      if ( count > 0                          &&
           ras.span_y == y                    &&
           (int)span->x + span->len == (int)x &&
           span->coverage == coverage         )
      {
        span->len = (unsigned short)( span->len + acount );
        return;
      }

      if ( ras.span_y != y || count >= FT_MAX_GRAY_SPANS )
      {
        if ( ras.render_span && count > 0 )
          ras.render_span( ras.span_y, count, ras.gray_spans,
                           ras.render_span_data );

#ifdef FT_DEBUG_LEVEL_TRACE

        if ( count > 0 )
        {
          int  n;


          FT_TRACE7(( "y = %3d ", ras.span_y ));
          span = ras.gray_spans;
          for ( n = 0; n < count; n++, span++ )
            FT_TRACE7(( "[%d..%d]:%02x ",
                        span->x, span->x + span->len - 1, span->coverage ));
          FT_TRACE7(( "\n" ));
        }

#endif /* FT_DEBUG_LEVEL_TRACE */

        ras.num_gray_spans = 0;
        ras.span_y         = (int)y;

        count = 0;
        span  = ras.gray_spans;
      }
      else
        span++;

      /* add a gray span to the current list */
      span->x        = (short)x;
      span->len      = (unsigned short)acount;
      span->coverage = (unsigned char)coverage;

      ras.num_gray_spans++;
    }
  }


#ifdef FT_DEBUG_LEVEL_TRACE

  /* to be called while in the debugger --                                */
  /* this function causes a compiler warning since it is unused otherwise */
  static void
  gray_dump_cells( RAS_ARG )
  {
    int  yindex;


    for ( yindex = 0; yindex < ras.ycount; yindex++ )
    {
      PCell  cell;


      printf( "%3d:", yindex );

      for ( cell = ras.ycells[yindex]; cell != NULL; cell = cell->next )
        printf( " (%3ld, c:%4ld, a:%6d)", cell->x, cell->cover, cell->area );
      printf( "\n" );
    }
  }

#endif /* FT_DEBUG_LEVEL_TRACE */


  static void
  gray_sweep( RAS_ARG_ const FT_Bitmap*  target )
  {
    int  yindex;

    FT_UNUSED( target );


    if ( ras.num_cells == 0 )
      return;

    ras.num_gray_spans = 0;

    FT_TRACE7(( "gray_sweep: start\n" ));

    for ( yindex = 0; yindex < ras.ycount; yindex++ )
    {
      PCell   cell  = ras.ycells[yindex];
      TCoord  cover = 0;
      TCoord  x     = 0;


      for ( ; cell != NULL; cell = cell->next )
      {
        TPos  area;


        if ( cell->x > x && cover != 0 )
          gray_hline( RAS_VAR_ x, yindex, cover * ( ONE_PIXEL * 2 ),
                      cell->x - x );

        cover += cell->cover;
        area   = cover * ( ONE_PIXEL * 2 ) - cell->area;

        if ( area != 0 && cell->x >= 0 )
          gray_hline( RAS_VAR_ cell->x, yindex, area, 1 );

        x = cell->x + 1;
      }

      if ( cover != 0 )
        gray_hline( RAS_VAR_ x, yindex, cover * ( ONE_PIXEL * 2 ),
                    ras.count_ex - x );
    }

    if ( ras.render_span && ras.num_gray_spans > 0 )
      ras.render_span( ras.span_y, ras.num_gray_spans,
                       ras.gray_spans, ras.render_span_data );

#ifdef FT_DEBUG_LEVEL_TRACE

    if ( ras.num_gray_spans > 0 )
    {
      FT_Span*  span;
      int       n;


      FT_TRACE7(( "y = %3d ", ras.span_y ));
      span = ras.gray_spans;
      for ( n = 0; n < ras.num_gray_spans; n++, span++ )
        FT_TRACE7(( "[%d..%d]:%02x ",
                    span->x, span->x + span->len - 1, span->coverage ));
      FT_TRACE7(( "\n" ));
    }

    FT_TRACE7(( "gray_sweep: end\n" ));

#endif /* FT_DEBUG_LEVEL_TRACE */

  }


#ifdef _STANDALONE_

  /*************************************************************************/
  /*                                                                       */
  /*  The following function should only compile in stand-alone mode,      */
  /*  i.e., when building this component without the rest of FreeType.     */
  /*                                                                       */
  /*************************************************************************/

  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Outline_Decompose                                               */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Walk over an outline's structure to decompose it into individual   */
  /*    segments and Bézier arcs.  This function is also able to emit      */
  /*    `move to' and `close to' operations to indicate the start and end  */
  /*    of new contours in the outline.                                    */
  /*                                                                       */
  /* <Input>                                                               */
  /*    outline        :: A pointer to the source target.                  */
  /*                                                                       */
  /*    func_interface :: A table of `emitters', i.e., function pointers   */
  /*                      called during decomposition to indicate path     */
  /*                      operations.                                      */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    user           :: A typeless pointer which is passed to each       */
  /*                      emitter during the decomposition.  It can be     */
  /*                      used to store the state during the               */
  /*                      decomposition.                                   */
  /*                                                                       */
  /* <Return>                                                              */
  /*    Error code.  0 means success.                                      */
  /*                                                                       */
  static int
  FT_Outline_Decompose( const FT_Outline*        outline,
                        const FT_Outline_Funcs*  func_interface,
                        void*                    user )
  {
#undef SCALED
#define SCALED( x )  ( ( (x) << shift ) - delta )

    FT_Vector   v_last;
    FT_Vector   v_control;
    FT_Vector   v_start;

    FT_Vector*  point;
    FT_Vector*  limit;
    char*       tags;

    int         error;

    int   n;         /* index of contour in outline     */
    int   first;     /* index of first point in contour */
    char  tag;       /* current point's state           */

    int   shift;
    TPos  delta;


    if ( !outline || !func_interface )
      return FT_THROW( Invalid_Argument );

    shift = func_interface->shift;
    delta = func_interface->delta;
    first = 0;

    for ( n = 0; n < outline->n_contours; n++ )
    {
      int  last;  /* index of last point in contour */


      FT_TRACE5(( "FT_Outline_Decompose: Outline %d\n", n ));

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
      tags  = outline->tags   + first;
      tag   = FT_CURVE_TAG( tags[0] );

      /* A contour cannot start with a cubic control point! */
      if ( tag == FT_CURVE_TAG_CUBIC )
        goto Invalid_Outline;

      /* check first point to determine origin */
      if ( tag == FT_CURVE_TAG_CONIC )
      {
        /* first point is conic control.  Yes, this happens. */
        if ( FT_CURVE_TAG( outline->tags[last] ) == FT_CURVE_TAG_ON )
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

      FT_TRACE5(( "  move to (%.2f, %.2f)\n",
                  v_start.x / 64.0, v_start.y / 64.0 ));
      error = func_interface->move_to( &v_start, user );
      if ( error )
        goto Exit;

      while ( point < limit )
      {
        point++;
        tags++;

        tag = FT_CURVE_TAG( tags[0] );
        switch ( tag )
        {
        case FT_CURVE_TAG_ON:  /* emit a single line_to */
          {
            FT_Vector  vec;


            vec.x = SCALED( point->x );
            vec.y = SCALED( point->y );

            FT_TRACE5(( "  line to (%.2f, %.2f)\n",
                        vec.x / 64.0, vec.y / 64.0 ));
            error = func_interface->line_to( &vec, user );
            if ( error )
              goto Exit;
            continue;
          }

        case FT_CURVE_TAG_CONIC:  /* consume conic arcs */
          v_control.x = SCALED( point->x );
          v_control.y = SCALED( point->y );

        Do_Conic:
          if ( point < limit )
          {
            FT_Vector  vec;
            FT_Vector  v_middle;


            point++;
            tags++;
            tag = FT_CURVE_TAG( tags[0] );

            vec.x = SCALED( point->x );
            vec.y = SCALED( point->y );

            if ( tag == FT_CURVE_TAG_ON )
            {
              FT_TRACE5(( "  conic to (%.2f, %.2f)"
                          " with control (%.2f, %.2f)\n",
                          vec.x / 64.0, vec.y / 64.0,
                          v_control.x / 64.0, v_control.y / 64.0 ));
              error = func_interface->conic_to( &v_control, &vec, user );
              if ( error )
                goto Exit;
              continue;
            }

            if ( tag != FT_CURVE_TAG_CONIC )
              goto Invalid_Outline;

            v_middle.x = ( v_control.x + vec.x ) / 2;
            v_middle.y = ( v_control.y + vec.y ) / 2;

            FT_TRACE5(( "  conic to (%.2f, %.2f)"
                        " with control (%.2f, %.2f)\n",
                        v_middle.x / 64.0, v_middle.y / 64.0,
                        v_control.x / 64.0, v_control.y / 64.0 ));
            error = func_interface->conic_to( &v_control, &v_middle, user );
            if ( error )
              goto Exit;

            v_control = vec;
            goto Do_Conic;
          }

          FT_TRACE5(( "  conic to (%.2f, %.2f)"
                      " with control (%.2f, %.2f)\n",
                      v_start.x / 64.0, v_start.y / 64.0,
                      v_control.x / 64.0, v_control.y / 64.0 ));
          error = func_interface->conic_to( &v_control, &v_start, user );
          goto Close;

        default:  /* FT_CURVE_TAG_CUBIC */
          {
            FT_Vector  vec1, vec2;


            if ( point + 1 > limit                             ||
                 FT_CURVE_TAG( tags[1] ) != FT_CURVE_TAG_CUBIC )
              goto Invalid_Outline;

            point += 2;
            tags  += 2;

            vec1.x = SCALED( point[-2].x );
            vec1.y = SCALED( point[-2].y );

            vec2.x = SCALED( point[-1].x );
            vec2.y = SCALED( point[-1].y );

            if ( point <= limit )
            {
              FT_Vector  vec;


              vec.x = SCALED( point->x );
              vec.y = SCALED( point->y );

              FT_TRACE5(( "  cubic to (%.2f, %.2f)"
                          " with controls (%.2f, %.2f) and (%.2f, %.2f)\n",
                          vec.x / 64.0, vec.y / 64.0,
                          vec1.x / 64.0, vec1.y / 64.0,
                          vec2.x / 64.0, vec2.y / 64.0 ));
              error = func_interface->cubic_to( &vec1, &vec2, &vec, user );
              if ( error )
                goto Exit;
              continue;
            }

            FT_TRACE5(( "  cubic to (%.2f, %.2f)"
                        " with controls (%.2f, %.2f) and (%.2f, %.2f)\n",
                        v_start.x / 64.0, v_start.y / 64.0,
                        vec1.x / 64.0, vec1.y / 64.0,
                        vec2.x / 64.0, vec2.y / 64.0 ));
            error = func_interface->cubic_to( &vec1, &vec2, &v_start, user );
            goto Close;
          }
        }
      }

      /* close the contour with a line segment */
      FT_TRACE5(( "  line to (%.2f, %.2f)\n",
                  v_start.x / 64.0, v_start.y / 64.0 ));
      error = func_interface->line_to( &v_start, user );

   Close:
      if ( error )
        goto Exit;

      first = last + 1;
    }

    FT_TRACE5(( "FT_Outline_Decompose: Done\n", n ));
    return 0;

  Exit:
    FT_TRACE5(( "FT_Outline_Decompose: Error %d\n", error ));
    return error;

  Invalid_Outline:
    return FT_THROW( Invalid_Outline );
  }

#endif /* _STANDALONE_ */


  typedef struct  gray_TBand_
  {
    TPos  min, max;

  } gray_TBand;

    FT_DEFINE_OUTLINE_FUNCS(func_interface,
      (FT_Outline_MoveTo_Func) gray_move_to,
      (FT_Outline_LineTo_Func) gray_line_to,
      (FT_Outline_ConicTo_Func)gray_conic_to,
      (FT_Outline_CubicTo_Func)gray_cubic_to,
      0,
      0
    )

  static int
  gray_convert_glyph_inner( RAS_ARG )
  {

    volatile int  error = 0;

#ifdef FT_CONFIG_OPTION_PIC
      FT_Outline_Funcs func_interface;
      Init_Class_func_interface(&func_interface);
#endif

    if ( ft_setjmp( ras.jump_buffer ) == 0 )
    {
      error = FT_Outline_Decompose( &ras.outline, &func_interface, &ras );
      gray_record_cell( RAS_VAR );
    }
    else
      error = FT_THROW( Memory_Overflow );

    return error;
  }


  static int
  gray_convert_glyph( RAS_ARG )
  {
    gray_TBand            bands[40];
    gray_TBand* volatile  band;
    int volatile          n, num_bands;
    TPos volatile         min, max, max_y;
    FT_BBox*              clip;


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
          long   cell_start, cell_end, cell_mod;


          ras.ycells = (PCell*)ras.buffer;
          ras.ycount = band->max - band->min;

          cell_start = sizeof ( PCell ) * ras.ycount;
          cell_mod   = cell_start % sizeof ( TCell );
          if ( cell_mod > 0 )
            cell_start += sizeof ( TCell ) - cell_mod;

          cell_end  = ras.buffer_size;
          cell_end -= cell_end % sizeof ( TCell );

          cells_max = (PCell)( (char*)ras.buffer + cell_end );
          ras.cells = (PCell)( (char*)ras.buffer + cell_start );
          if ( ras.cells >= cells_max )
            goto ReduceBands;

          ras.max_cells = cells_max - ras.cells;
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
          gray_sweep( RAS_VAR_ &ras.target );
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
#ifdef FT_DEBUG_LEVEL_TRACE
          FT_TRACE7(( "gray_convert_glyph: rotten glyph\n" ));
#endif
          return 1;
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

    if ( ras.band_shoot > 8 && ras.band_size > 16 )
      ras.band_size = ras.band_size / 2;

    return 0;
  }


  static int
  gray_raster_render( gray_PRaster             raster,
                      const FT_Raster_Params*  params )
  {
    const FT_Outline*  outline    = (const FT_Outline*)params->source;
    const FT_Bitmap*   target_map = params->target;
    gray_PWorker       worker;


    if ( !raster || !raster->buffer || !raster->buffer_size )
      return FT_THROW( Invalid_Argument );

    if ( !outline )
      return FT_THROW( Invalid_Outline );

    /* return immediately if the outline is empty */
    if ( outline->n_points == 0 || outline->n_contours <= 0 )
      return 0;

    if ( !outline->contours || !outline->points )
      return FT_THROW( Invalid_Outline );

    if ( outline->n_points !=
           outline->contours[outline->n_contours - 1] + 1 )
      return FT_THROW( Invalid_Outline );

    worker = raster->worker;

    /* if direct mode is not set, we must have a target bitmap */
    if ( !( params->flags & FT_RASTER_FLAG_DIRECT ) )
    {
      if ( !target_map )
        return FT_THROW( Invalid_Argument );

      /* nothing to do */
      if ( !target_map->width || !target_map->rows )
        return 0;

      if ( !target_map->buffer )
        return FT_THROW( Invalid_Argument );
    }

    /* this version does not support monochrome rendering */
    if ( !( params->flags & FT_RASTER_FLAG_AA ) )
      return FT_THROW( Invalid_Mode );

    /* compute clipping box */
    if ( !( params->flags & FT_RASTER_FLAG_DIRECT ) )
    {
      /* compute clip box from target pixmap */
      ras.clip_box.xMin = 0;
      ras.clip_box.yMin = 0;
      ras.clip_box.xMax = target_map->width;
      ras.clip_box.yMax = target_map->rows;
    }
    else if ( params->flags & FT_RASTER_FLAG_CLIP )
      ras.clip_box = params->clip_box;
    else
    {
      ras.clip_box.xMin = -32768L;
      ras.clip_box.yMin = -32768L;
      ras.clip_box.xMax =  32767L;
      ras.clip_box.yMax =  32767L;
    }

    gray_init_cells( RAS_VAR_ raster->buffer, raster->buffer_size );

    ras.outline        = *outline;
    ras.num_cells      = 0;
    ras.invalid        = 1;
    ras.band_size      = raster->band_size;
    ras.num_gray_spans = 0;

    if ( params->flags & FT_RASTER_FLAG_DIRECT )
    {
      ras.render_span      = (FT_Raster_Span_Func)params->gray_spans;
      ras.render_span_data = params->user;
    }
    else
    {
      ras.target           = *target_map;
      ras.render_span      = (FT_Raster_Span_Func)gray_render_span;
      ras.render_span_data = &ras;
    }

    return gray_convert_glyph( RAS_VAR );
  }


  /**** RASTER OBJECT CREATION: In stand-alone mode, we simply use *****/
  /****                         a static object.                   *****/

#ifdef _STANDALONE_

  static int
  gray_raster_new( void*       memory,
                   FT_Raster*  araster )
  {
    static gray_TRaster  the_raster;

    FT_UNUSED( memory );


    *araster = (FT_Raster)&the_raster;
    FT_MEM_ZERO( &the_raster, sizeof ( the_raster ) );

    return 0;
  }


  static void
  gray_raster_done( FT_Raster  raster )
  {
    /* nothing */
    FT_UNUSED( raster );
  }

#else /* !_STANDALONE_ */

  static int
  gray_raster_new( FT_Memory   memory,
                   FT_Raster*  araster )
  {
    FT_Error      error;
    gray_PRaster  raster = NULL;


    *araster = 0;
    if ( !FT_ALLOC( raster, sizeof ( gray_TRaster ) ) )
    {
      raster->memory = memory;
      *araster       = (FT_Raster)raster;
    }

    return error;
  }


  static void
  gray_raster_done( FT_Raster  raster )
  {
    FT_Memory  memory = (FT_Memory)((gray_PRaster)raster)->memory;


    FT_FREE( raster );
  }

#endif /* !_STANDALONE_ */


  static void
  gray_raster_reset( FT_Raster  raster,
                     char*      pool_base,
                     long       pool_size )
  {
    gray_PRaster  rast = (gray_PRaster)raster;


    if ( raster )
    {
      if ( pool_base && pool_size >= (long)sizeof ( gray_TWorker ) + 2048 )
      {
        gray_PWorker  worker = (gray_PWorker)pool_base;


        rast->worker      = worker;
        rast->buffer      = pool_base +
                              ( ( sizeof ( gray_TWorker ) +
                                  sizeof ( TCell ) - 1 )  &
                                ~( sizeof ( TCell ) - 1 ) );
        rast->buffer_size = (long)( ( pool_base + pool_size ) -
                                    (char*)rast->buffer ) &
                                      ~( sizeof ( TCell ) - 1 );
        rast->band_size   = (int)( rast->buffer_size /
                                     ( sizeof ( TCell ) * 8 ) );
      }
      else
      {
        rast->buffer      = NULL;
        rast->buffer_size = 0;
        rast->worker      = NULL;
      }
    }
  }


  FT_DEFINE_RASTER_FUNCS(ft_grays_raster,
    FT_GLYPH_FORMAT_OUTLINE,

    (FT_Raster_New_Func)     gray_raster_new,
    (FT_Raster_Reset_Func)   gray_raster_reset,
    (FT_Raster_Set_Mode_Func)0,
    (FT_Raster_Render_Func)  gray_raster_render,
    (FT_Raster_Done_Func)    gray_raster_done
  )


/* END */


/* Local Variables: */
/* coding: utf-8    */
/* End:             */

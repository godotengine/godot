/****************************************************************************
 *
 * ftgrays.c
 *
 *   A new `perfect' anti-aliasing renderer (body).
 *
 * Copyright (C) 2000-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
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
   * This file can be compiled without the rest of the FreeType engine, by
   * defining the STANDALONE_ macro when compiling it.  You also need to
   * put the files `ftgrays.h' and `ftimage.h' into the current
   * compilation directory.  Typically, you could do something like
   *
   * - copy `src/smooth/ftgrays.c' (this file) to your current directory
   *
   * - copy `include/freetype/ftimage.h' and `src/smooth/ftgrays.h' to the
   *   same directory
   *
   * - compile `ftgrays' with the STANDALONE_ macro defined, as in
   *
   *     cc -c -DSTANDALONE_ ftgrays.c
   *
   * The renderer can be initialized with a call to
   * `ft_gray_raster.raster_new'; an anti-aliased bitmap can be generated
   * with a call to `ft_gray_raster.raster_render'.
   *
   * See the comments and documentation in the file `ftimage.h' for more
   * details on how the raster works.
   *
   */

  /**************************************************************************
   *
   * This is a new anti-aliasing scan-converter for FreeType 2.  The
   * algorithm used here is _very_ different from the one in the standard
   * `ftraster' module.  Actually, `ftgrays' computes the _exact_
   * coverage of the outline on each pixel cell by straight segments.
   *
   * It is based on ideas that I initially found in Raph Levien's
   * excellent LibArt graphics library (see https://www.levien.com/libart
   * for more information, though the web pages do not tell anything
   * about the renderer; you'll have to dive into the source code to
   * understand how it works).
   *
   * Note, however, that this is a _very_ different implementation
   * compared to Raph's.  Coverage information is stored in a very
   * different way, and I don't use sorted vector paths.  Also, it doesn't
   * use floating point values.
   *
   * Bézier segments are flattened by splitting them until their deviation
   * from straight line becomes much smaller than a pixel.  Therefore, the
   * pixel coverage by a Bézier curve is calculated approximately.  To
   * estimate the deviation, we use the distance from the control point
   * to the conic chord centre or the cubic chord trisection.  These
   * distances vanish fast after each split.  In the conic case, they vanish
   * predictably and the number of necessary splits can be calculated.
   *
   * This renderer has the following advantages:
   *
   * - It doesn't need an intermediate bitmap.  Instead, one can supply a
   *   callback function that will be called by the renderer to draw gray
   *   spans on any target surface.  You can thus do direct composition on
   *   any kind of bitmap, provided that you give the renderer the right
   *   callback.
   *
   * - A perfect anti-aliaser, i.e., it computes the _exact_ coverage on
   *   each pixel cell by straight segments.
   *
   * - It performs a single pass on the outline (the `standard' FT2
   *   renderer makes two passes).
   *
   * - It can easily be modified to render to _any_ number of gray levels
   *   cheaply.
   *
   * - For small (< 80) pixel sizes, it is faster than the standard
   *   renderer.
   *
   */


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  smooth


#ifdef STANDALONE_


  /* The size in bytes of the render pool used by the scan-line converter  */
  /* to do all of its work.                                                */
#define FT_RENDER_POOL_SIZE  16384L


  /* Auxiliary macros for token concatenation. */
#define FT_ERR_XCAT( x, y )  x ## y
#define FT_ERR_CAT( x, y )   FT_ERR_XCAT( x, y )

#define FT_BEGIN_STMNT  do {
#define FT_END_STMNT    } while ( 0 )

#define FT_MIN( a, b )  ( (a) < (b) ? (a) : (b) )
#define FT_MAX( a, b )  ( (a) > (b) ? (a) : (b) )
#define FT_ABS( a )     ( (a) < 0 ? -(a) : (a) )


  /*
   * Approximate sqrt(x*x+y*y) using the `alpha max plus beta min'
   * algorithm.  We use alpha = 1, beta = 3/8, giving us results with a
   * largest error less than 7% compared to the exact value.
   */
#define FT_HYPOT( x, y )                 \
          ( x = FT_ABS( x ),             \
            y = FT_ABS( y ),             \
            x > y ? x + ( 3 * y >> 3 )   \
                  : y + ( 3 * x >> 3 ) )


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
#define FT_CHAR_BIT   CHAR_BIT
#define FT_UINT_MAX   UINT_MAX
#define FT_INT_MAX    INT_MAX
#define FT_ULONG_MAX  ULONG_MAX

#define ADD_LONG( a, b )                                    \
          (long)( (unsigned long)(a) + (unsigned long)(b) )
#define SUB_LONG( a, b )                                    \
          (long)( (unsigned long)(a) - (unsigned long)(b) )
#define MUL_LONG( a, b )                                    \
          (long)( (unsigned long)(a) * (unsigned long)(b) )
#define NEG_LONG( a )                                       \
          (long)( -(unsigned long)(a) )


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
          ( FT_Throw( FT_ERR_CAT( ErrRaster_, e ),  \
                      __LINE__,                     \
                      __FILE__ )                  | \
            FT_ERR_CAT( ErrRaster_, e )           )

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


#else /* !STANDALONE_ */


#include "ftgrays.h"
#include <freetype/internal/ftobjs.h>
#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftcalc.h>
#include <freetype/ftoutln.h>

#include "ftsmerrs.h"

#define Smooth_Err_Invalid_Mode     Smooth_Err_Cannot_Render_Glyph
#define Smooth_Err_Memory_Overflow  Smooth_Err_Out_Of_Memory
#define ErrRaster_Memory_Overflow   Smooth_Err_Out_Of_Memory


#endif /* !STANDALONE_ */


#ifndef FT_MEM_SET
#define FT_MEM_SET( d, s, c )  ft_memset( d, s, c )
#endif

#ifndef FT_MEM_ZERO
#define FT_MEM_ZERO( dest, count )  FT_MEM_SET( dest, 0, count )
#endif

#ifndef FT_ZERO
#define FT_ZERO( p )  FT_MEM_ZERO( p, sizeof ( *(p) ) )
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

#define RAS_ARG   void
#define RAS_ARG_  /* empty */
#define RAS_VAR   /* empty */
#define RAS_VAR_  /* empty */

#endif /* FT_STATIC_RASTER */


  /* must be at least 6 bits! */
#define PIXEL_BITS  8

#define ONE_PIXEL       ( 1 << PIXEL_BITS )
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
  /* the remainder is always positive.  We use the remainder to keep   */
  /* track of accumulating errors and compensate for them.             */
#define FT_DIV_MOD( type, dividend, divisor, quotient, remainder ) \
  FT_BEGIN_STMNT                                                   \
    (quotient)  = (type)( (dividend) / (divisor) );                \
    (remainder) = (type)( (dividend) % (divisor) );                \
    if ( (remainder) < 0 )                                         \
    {                                                              \
      (quotient)--;                                                \
      (remainder) += (type)(divisor);                              \
    }                                                              \
  FT_END_STMNT

#ifdef  __arm__
  /* Work around a bug specific to GCC which make the compiler fail to */
  /* optimize a division and modulo operation on the same parameters   */
  /* into a single call to `__aeabi_idivmod'.  See                     */
  /*                                                                   */
  /*  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=43721               */
#undef FT_DIV_MOD
#define FT_DIV_MOD( type, dividend, divisor, quotient, remainder ) \
  FT_BEGIN_STMNT                                                   \
    (quotient)  = (type)( (dividend) / (divisor) );                \
    (remainder) = (type)( (dividend) - (quotient) * (divisor) );   \
    if ( (remainder) < 0 )                                         \
    {                                                              \
      (quotient)--;                                                \
      (remainder) += (type)(divisor);                              \
    }                                                              \
  FT_END_STMNT
#endif /* __arm__ */


  /* These macros speed up repetitive divisions by replacing them */
  /* with multiplications and right shifts.                       */
#define FT_UDIVPREP( c, b )                                        \
  long  b ## _r = c ? (long)( FT_ULONG_MAX >> PIXEL_BITS ) / ( b ) \
                    : 0
#define FT_UDIV( a, b )                                                \
  (TCoord)( ( (unsigned long)( a ) * (unsigned long)( b ## _r ) ) >>   \
            ( sizeof( long ) * FT_CHAR_BIT - PIXEL_BITS ) )


  /**************************************************************************
   *
   * TYPE DEFINITIONS
   */

  /* don't change the following types to FT_Int or FT_Pos, since we might */
  /* need to define them to "float" or "double" when experimenting with   */
  /* new algorithms                                                       */

  typedef long  TPos;     /* subpixel coordinate               */
  typedef int   TCoord;   /* integer scanline/pixel coordinate */
  typedef int   TArea;    /* cell areas, coordinate products   */


  typedef struct TCell_*  PCell;

  typedef struct  TCell_
  {
    TCoord  x;     /* same with gray_TWorker.ex    */
    TCoord  cover; /* same with gray_TWorker.cover */
    TArea   area;
    PCell   next;

  } TCell;

  typedef struct TPixmap_
  {
    unsigned char*  origin;  /* pixmap origin at the bottom-left */
    int             pitch;   /* pitch to go down one row */

  } TPixmap;

  /* maximum number of gray cells in the buffer */
#if FT_RENDER_POOL_SIZE > 2048
#define FT_MAX_GRAY_POOL  ( FT_RENDER_POOL_SIZE / sizeof ( TCell ) )
#else
#define FT_MAX_GRAY_POOL  ( 2048 / sizeof ( TCell ) )
#endif

  /* FT_Span buffer size for direct rendering only */
#define FT_MAX_GRAY_SPANS  10


#if defined( _MSC_VER )      /* Visual C++ (and Intel C++) */
  /* We disable the warning `structure was padded due to   */
  /* __declspec(align())' in order to compile cleanly with */
  /* the maximum level of warnings.                        */
#pragma warning( push )
#pragma warning( disable : 4324 )
#endif /* _MSC_VER */

  typedef struct  gray_TWorker_
  {
    ft_jmp_buf  jump_buffer;

    TCoord  ex, ey;
    TCoord  min_ex, max_ex;
    TCoord  min_ey, max_ey;

    TArea   area;
    TCoord  cover;
    int     invalid;

    PCell*      ycells;
    PCell       cells;
    FT_PtrDist  max_cells;
    FT_PtrDist  num_cells;

    TPos    x,  y;

    FT_Outline  outline;
    TPixmap     target;

    FT_Raster_Span_Func  render_span;
    void*                render_span_data;
    FT_Span              spans[FT_MAX_GRAY_SPANS];
    int                  num_spans;

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
    void*         memory;

  } gray_TRaster, *gray_PRaster;


#ifdef FT_DEBUG_LEVEL_TRACE

  /* to be called while in the debugger --                                */
  /* this function causes a compiler warning since it is unused otherwise */
  static void
  gray_dump_cells( RAS_ARG )
  {
    int  y;


    for ( y = ras.min_ey; y < ras.max_ey; y++ )
    {
      PCell  cell = ras.ycells[y - ras.min_ey];


      printf( "%3d:", y );

      for ( ; cell != NULL; cell = cell->next )
        printf( " (%3d, c:%4d, a:%6d)",
                cell->x, cell->cover, cell->area );
      printf( "\n" );
    }
  }

#endif /* FT_DEBUG_LEVEL_TRACE */


  /**************************************************************************
   *
   * Record the current cell in the linked list.
   */
  static void
  gray_record_cell( RAS_ARG )
  {
    PCell  *pcell, cell;
    TCoord  x = ras.ex;


    pcell = &ras.ycells[ras.ey - ras.min_ey];
    while ( ( cell = *pcell ) )
    {
      if ( cell->x > x )
        break;

      if ( cell->x == x )
        goto Found;

      pcell = &cell->next;
    }

    if ( ras.num_cells >= ras.max_cells )
      ft_longjmp( ras.jump_buffer, 1 );

    /* insert new cell */
    cell        = ras.cells + ras.num_cells++;
    cell->x     = x;
    cell->area  = ras.area;
    cell->cover = ras.cover;

    cell->next  = *pcell;
    *pcell      = cell;

    return;

  Found:
    /* update old cell */
    cell->area  += ras.area;
    cell->cover += ras.cover;
  }


  /**************************************************************************
   *
   * Set the current cell to a new position.
   */
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

    /* record the current one if it is valid and substantial */
    if ( !ras.invalid && ( ras.area || ras.cover ) )
      gray_record_cell( RAS_VAR );

    ras.area  = 0;
    ras.cover = 0;
    ras.ex    = FT_MAX( ex, ras.min_ex - 1 );
    ras.ey    = ey;

    ras.invalid = ( ey >= ras.max_ey || ey < ras.min_ey ||
                    ex >= ras.max_ex );
  }


#ifndef FT_LONG64

  /**************************************************************************
   *
   * Render a scanline as one or more cells.
   */
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
    }
    else
    {
      p     = fx1 * dy;
      first = 0;
      incr  = -1;
      dx    = -dx;
    }

    /* the fractional part of y-delta is mod/dx. It is essential to */
    /* keep track of its accumulation for accurate rendering.       */
    /* XXX: y-delta and x-delta below should be related.            */
    FT_DIV_MOD( TCoord, p, dx, delta, mod );

    ras.area  += (TArea)( ( fx1 + first ) * delta );
    ras.cover += delta;
    y1        += delta;
    ex1       += incr;
    gray_set_cell( RAS_VAR_ ex1, ey );

    if ( ex1 != ex2 )
    {
      TCoord  lift, rem;


      p = ONE_PIXEL * dy;
      FT_DIV_MOD( TCoord, p, dx, lift, rem );

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


  /**************************************************************************
   *
   * Render a given line as a series of scanlines.
   */
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
      TArea   area;


      if ( dy > 0)
      {
        first = ONE_PIXEL;
        incr  = 1;
      }
      else
      {
        first = 0;
        incr  = -1;
      }

      delta      = first - fy1;
      ras.area  += (TArea)two_fx * delta;
      ras.cover += delta;
      ey1       += incr;

      gray_set_cell( RAS_VAR_ ex, ey1 );

      delta = first + first - ONE_PIXEL;
      area  = (TArea)two_fx * delta;
      while ( ey1 != ey2 )
      {
        ras.area  += area;
        ras.cover += delta;
        ey1       += incr;

        gray_set_cell( RAS_VAR_ ex, ey1 );
      }

      delta      = fy2 - ONE_PIXEL + first;
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
    FT_DIV_MOD( TCoord, p, dy, delta, mod );

    x = ras.x + delta;
    gray_render_scanline( RAS_VAR_ ey1, ras.x, fy1, x, first );

    ey1 += incr;
    gray_set_cell( RAS_VAR_ TRUNC( x ), ey1 );

    if ( ey1 != ey2 )
    {
      TCoord  lift, rem;


      p    = ONE_PIXEL * dx;
      FT_DIV_MOD( TCoord, p, dy, lift, rem );

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

  /**************************************************************************
   *
   * Render a straight line across multiple cells in any direction.
   */
  static void
  gray_render_line( RAS_ARG_ TPos  to_x,
                             TPos  to_y )
  {
    TPos    dx, dy;
    TCoord  fx1, fy1, fx2, fy2;
    TCoord  ex1, ey1, ex2, ey2;


    ey1 = TRUNC( ras.y );
    ey2 = TRUNC( to_y );

    /* perform vertical clipping */
    if ( ( ey1 >= ras.max_ey && ey2 >= ras.max_ey ) ||
         ( ey1 <  ras.min_ey && ey2 <  ras.min_ey ) )
      goto End;

    ex1 = TRUNC( ras.x );
    ex2 = TRUNC( to_x );

    fx1 = FRACT( ras.x );
    fy1 = FRACT( ras.y );

    dx = to_x - ras.x;
    dy = to_y - ras.y;

    if ( ex1 == ex2 && ey1 == ey2 )       /* inside one cell */
      ;
    else if ( dy == 0 ) /* ex1 != ex2 */  /* any horizontal line */
    {
      gray_set_cell( RAS_VAR_ ex2, ey2 );
      goto End;
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
      TPos  prod = dx * (TPos)fy1 - dy * (TPos)fx1;
      FT_UDIVPREP( ex1 != ex2, dx );
      FT_UDIVPREP( ey1 != ey2, dy );


      /* The fundamental value `prod' determines which side and the  */
      /* exact coordinate where the line exits current cell.  It is  */
      /* also easily updated when moving from one cell to the next.  */
      do
      {
        if      ( prod                                   <= 0 &&
                  prod - dx * ONE_PIXEL                  >  0 ) /* left */
        {
          fx2 = 0;
          fy2 = FT_UDIV( -prod, -dx );
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
          fx2 = FT_UDIV( -prod, dy );
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
          fy2 = FT_UDIV( prod, dx );
          ras.cover += ( fy2 - fy1 );
          ras.area  += ( fy2 - fy1 ) * ( fx1 + fx2 );
          fx1 = 0;
          fy1 = fy2;
          ex1++;
        }
        else /* ( prod                  + dy * ONE_PIXEL <  0 &&
                  prod                                   >  0 )    down */
        {
          fx2 = FT_UDIV( prod, -dy );
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
  gray_split_conic( FT_Vector*  base )
  {
    TPos  a, b;


    base[4].x = base[2].x;
    a = base[0].x + base[1].x;
    b = base[1].x + base[2].x;
    base[3].x = b >> 1;
    base[2].x = ( a + b ) >> 2;
    base[1].x = a >> 1;

    base[4].y = base[2].y;
    a = base[0].y + base[1].y;
    b = base[1].y + base[2].y;
    base[3].y = b >> 1;
    base[2].y = ( a + b ) >> 2;
    base[1].y = a >> 1;
  }


  static void
  gray_render_conic( RAS_ARG_ const FT_Vector*  control,
                              const FT_Vector*  to )
  {
    FT_Vector   bez_stack[16 * 2 + 1];  /* enough to accommodate bisections */
    FT_Vector*  arc = bez_stack;
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

    dx = FT_ABS( arc[2].x + arc[0].x - 2 * arc[1].x );
    dy = FT_ABS( arc[2].y + arc[0].y - 2 * arc[1].y );
    if ( dx < dy )
      dx = dy;

    /* We can calculate the number of necessary bisections because  */
    /* each bisection predictably reduces deviation exactly 4-fold. */
    /* Even 32-bit deviation would vanish after 16 bisections.      */
    draw = 1;
    while ( dx > ONE_PIXEL / 4 )
    {
      dx   >>= 2;
      draw <<= 1;
    }

    /* We use decrement counter to count the total number of segments */
    /* to draw starting from 2^level. Before each draw we split as    */
    /* many times as there are trailing zeros in the counter.         */
    do
    {
      split = draw & ( -draw );  /* isolate the rightmost 1-bit */
      while ( ( split >>= 1 ) )
      {
        gray_split_conic( arc );
        arc += 2;
      }

      gray_render_line( RAS_VAR_ arc[0].x, arc[0].y );
      arc -= 2;

    } while ( --draw );
  }


  static void
  gray_split_cubic( FT_Vector*  base )
  {
    TPos  a, b, c;


    base[6].x = base[3].x;
    a = base[0].x + base[1].x;
    b = base[1].x + base[2].x;
    c = base[2].x + base[3].x;
    base[5].x = c >> 1;
    c += b;
    base[4].x = c >> 2;
    base[1].x = a >> 1;
    a += b;
    base[2].x = a >> 2;
    base[3].x = ( a + c ) >> 3;

    base[6].y = base[3].y;
    a = base[0].y + base[1].y;
    b = base[1].y + base[2].y;
    c = base[2].y + base[3].y;
    base[5].y = c >> 1;
    c += b;
    base[4].y = c >> 2;
    base[1].y = a >> 1;
    a += b;
    base[2].y = a >> 2;
    base[3].y = ( a + c ) >> 3;
  }


  static void
  gray_render_cubic( RAS_ARG_ const FT_Vector*  control1,
                              const FT_Vector*  control2,
                              const FT_Vector*  to )
  {
    FT_Vector   bez_stack[16 * 3 + 1];  /* enough to accommodate bisections */
    FT_Vector*  arc = bez_stack;


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
      /* with each split, control points quickly converge towards  */
      /* chord trisection points and the vanishing distances below */
      /* indicate when the segment is flat enough to draw          */
      if ( FT_ABS( 2 * arc[0].x - 3 * arc[1].x + arc[3].x ) > ONE_PIXEL / 2 ||
           FT_ABS( 2 * arc[0].y - 3 * arc[1].y + arc[3].y ) > ONE_PIXEL / 2 ||
           FT_ABS( arc[0].x - 3 * arc[2].x + 2 * arc[3].x ) > ONE_PIXEL / 2 ||
           FT_ABS( arc[0].y - 3 * arc[2].y + 2 * arc[3].y ) > ONE_PIXEL / 2 )
        goto Split;

      gray_render_line( RAS_VAR_ arc[0].x, arc[0].y );

      if ( arc == bez_stack )
        return;

      arc -= 3;
      continue;

    Split:
      gray_split_cubic( arc );
      arc += 3;
    }
  }


  static int
  gray_move_to( const FT_Vector*  to,
                gray_PWorker      worker )
  {
    TPos  x, y;


    /* start to a new position */
    x = UPSCALE( to->x );
    y = UPSCALE( to->y );

    gray_set_cell( RAS_VAR_ TRUNC( x ), TRUNC( y ) );

    ras.x = x;
    ras.y = y;
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
  gray_hline( RAS_ARG_ TCoord  x,
                       TCoord  y,
                       TArea   coverage,
                       TCoord  acount )
  {
    /* scale the coverage from 0..(ONE_PIXEL*ONE_PIXEL*2) to 0..256  */
    coverage >>= PIXEL_BITS * 2 + 1 - 8;

    /* compute the line's coverage depending on the outline fill rule */
    if ( ras.outline.flags & FT_OUTLINE_EVEN_ODD_FILL )
    {
      coverage &= 511;

      if ( coverage >= 256 )
        coverage = 511 - coverage;
    }
    else  /* default non-zero winding rule */
    {
      if ( coverage < 0 )
        coverage = ~coverage;  /* the same as -coverage - 1 */

      if ( coverage >= 256 )
        coverage = 255;
    }

    if ( ras.num_spans >= 0 )  /* for FT_RASTER_FLAG_DIRECT only */
    {
      FT_Span*  span = ras.spans + ras.num_spans++;


      span->x        = (short)x;
      span->len      = (unsigned short)acount;
      span->coverage = (unsigned char)coverage;

      if ( ras.num_spans == FT_MAX_GRAY_SPANS )
      {
        /* flush the span buffer and reset the count */
        ras.render_span( y, ras.num_spans, ras.spans, ras.render_span_data );
        ras.num_spans = 0;
      }
    }
    else
    {
      unsigned char*  q = ras.target.origin - ras.target.pitch * y + x;
      unsigned char   c = (unsigned char)coverage;


      /* For small-spans it is faster to do it by ourselves than
       * calling `memset'.  This is mainly due to the cost of the
       * function call.
       */
      switch ( acount )
      {
      case 7:
        *q++ = c;
        /* fall through */
      case 6:
        *q++ = c;
        /* fall through */
      case 5:
        *q++ = c;
        /* fall through */
      case 4:
        *q++ = c;
        /* fall through */
      case 3:
        *q++ = c;
        /* fall through */
      case 2:
        *q++ = c;
        /* fall through */
      case 1:
        *q = c;
        /* fall through */
      case 0:
        break;
      default:
        FT_MEM_SET( q, c, acount );
      }
    }
  }


  static void
  gray_sweep( RAS_ARG )
  {
    int  y;


    for ( y = ras.min_ey; y < ras.max_ey; y++ )
    {
      PCell   cell  = ras.ycells[y - ras.min_ey];
      TCoord  x     = ras.min_ex;
      TArea   cover = 0;
      TArea   area;


      for ( ; cell != NULL; cell = cell->next )
      {
        if ( cover != 0 && cell->x > x )
          gray_hline( RAS_VAR_ x, y, cover, cell->x - x );

        cover += (TArea)cell->cover * ( ONE_PIXEL * 2 );
        area   = cover - cell->area;

        if ( area != 0 && cell->x >= ras.min_ex )
          gray_hline( RAS_VAR_ cell->x, y, area, 1 );

        x = cell->x + 1;
      }

      if ( cover != 0 )
        gray_hline( RAS_VAR_ x, y, cover, ras.max_ex - x );

      if ( ras.num_spans > 0 )  /* for FT_RASTER_FLAG_DIRECT only */
      {
        /* flush the span buffer and reset the count */
        ras.render_span( y, ras.num_spans, ras.spans, ras.render_span_data );
        ras.num_spans = 0;
      }
    }
  }


#ifdef STANDALONE_

  /**************************************************************************
   *
   * The following functions should only compile in stand-alone mode,
   * i.e., when building this component without the rest of FreeType.
   *
   */

  /**************************************************************************
   *
   * @Function:
   *   FT_Outline_Decompose
   *
   * @Description:
   *   Walk over an outline's structure to decompose it into individual
   *   segments and Bézier arcs.  This function is also able to emit
   *   `move to' and `close to' operations to indicate the start and end
   *   of new contours in the outline.
   *
   * @Input:
   *   outline ::
   *     A pointer to the source target.
   *
   *   func_interface ::
   *     A table of `emitters', i.e., function pointers
   *     called during decomposition to indicate path
   *     operations.
   *
   * @InOut:
   *   user ::
   *     A typeless pointer which is passed to each
   *     emitter during the decomposition.  It can be
   *     used to store the state during the
   *     decomposition.
   *
   * @Return:
   *   Error code.  0 means success.
   */
  static int
  FT_Outline_Decompose( const FT_Outline*        outline,
                        const FT_Outline_Funcs*  func_interface,
                        void*                    user )
  {
#undef SCALED
#define SCALED( x )  ( (x) * ( 1L << shift ) - delta )

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


    if ( !outline )
      return FT_THROW( Invalid_Outline );

    if ( !func_interface )
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
    FT_TRACE5(( "FT_Outline_Decompose: Error 0x%x\n", error ));
    return error;

  Invalid_Outline:
    return FT_THROW( Invalid_Outline );
  }

#endif /* STANDALONE_ */


  FT_DEFINE_OUTLINE_FUNCS(
    func_interface,

    (FT_Outline_MoveTo_Func) gray_move_to,   /* move_to  */
    (FT_Outline_LineTo_Func) gray_line_to,   /* line_to  */
    (FT_Outline_ConicTo_Func)gray_conic_to,  /* conic_to */
    (FT_Outline_CubicTo_Func)gray_cubic_to,  /* cubic_to */

    0,                                       /* shift    */
    0                                        /* delta    */
  )


  static int
  gray_convert_glyph_inner( RAS_ARG,
                            int  continued )
  {
    int  error;


    if ( ft_setjmp( ras.jump_buffer ) == 0 )
    {
      if ( continued )
        FT_Trace_Disable();
      error = FT_Outline_Decompose( &ras.outline, &func_interface, &ras );
      if ( continued )
        FT_Trace_Enable();

      if ( !ras.invalid )
        gray_record_cell( RAS_VAR );

      FT_TRACE7(( "band [%d..%d]: %ld cell%s\n",
                  ras.min_ey,
                  ras.max_ey,
                  ras.num_cells,
                  ras.num_cells == 1 ? "" : "s" ));
    }
    else
    {
      error = FT_THROW( Memory_Overflow );

      FT_TRACE7(( "band [%d..%d]: to be bisected\n",
                  ras.min_ey, ras.max_ey ));
    }

    return error;
  }


  static int
  gray_convert_glyph( RAS_ARG )
  {
    const TCoord  yMin = ras.min_ey;
    const TCoord  yMax = ras.max_ey;

    TCell    buffer[FT_MAX_GRAY_POOL];
    size_t   height = (size_t)( yMax - yMin );
    size_t   n = FT_MAX_GRAY_POOL / 8;
    TCoord   y;
    TCoord   bands[32];  /* enough to accommodate bisections */
    TCoord*  band;

    int  continued = 0;


    /* set up vertical bands */
    if ( height > n )
    {
      /* two divisions rounded up */
      n       = ( height + n - 1 ) / n;
      height  = ( height + n - 1 ) / n;
    }

    /* memory management */
    n = ( height * sizeof ( PCell ) + sizeof ( TCell ) - 1 ) / sizeof ( TCell );

    ras.cells     = buffer + n;
    ras.max_cells = (FT_PtrDist)( FT_MAX_GRAY_POOL - n );
    ras.ycells    = (PCell*)buffer;

    for ( y = yMin; y < yMax; )
    {
      ras.min_ey = y;
      y         += height;
      ras.max_ey = FT_MIN( y, yMax );

      band    = bands;
      band[1] = ras.min_ey;
      band[0] = ras.max_ey;

      do
      {
        TCoord  width = band[0] - band[1];
        int     error;


        FT_MEM_ZERO( ras.ycells, height * sizeof ( PCell ) );

        ras.num_cells = 0;
        ras.invalid   = 1;
        ras.min_ey    = band[1];
        ras.max_ey    = band[0];

        error     = gray_convert_glyph_inner( RAS_VAR, continued );
        continued = 1;

        if ( !error )
        {
          gray_sweep( RAS_VAR );
          band--;
          continue;
        }
        else if ( error != ErrRaster_Memory_Overflow )
          return 1;

        /* render pool overflow; we will reduce the render band by half */
        width >>= 1;

        /* this should never happen even with tiny rendering pool */
        if ( width == 0 )
        {
          FT_TRACE7(( "gray_convert_glyph: rotten glyph\n" ));
          return 1;
        }

        band++;
        band[1]  = band[0];
        band[0] += width;
      } while ( band >= bands );
    }

    return 0;
  }


  static int
  gray_raster_render( FT_Raster                raster,
                      const FT_Raster_Params*  params )
  {
    const FT_Outline*  outline    = (const FT_Outline*)params->source;
    const FT_Bitmap*   target_map = params->target;

#ifndef FT_STATIC_RASTER
    gray_TWorker  worker[1];
#endif


    if ( !raster )
      return FT_THROW( Invalid_Argument );

    /* this version does not support monochrome rendering */
    if ( !( params->flags & FT_RASTER_FLAG_AA ) )
      return FT_THROW( Invalid_Mode );

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

    ras.outline = *outline;

    if ( params->flags & FT_RASTER_FLAG_DIRECT )
    {
      if ( !params->gray_spans )
        return 0;

      ras.render_span      = (FT_Raster_Span_Func)params->gray_spans;
      ras.render_span_data = params->user;
      ras.num_spans        = 0;

      ras.min_ex = params->clip_box.xMin;
      ras.min_ey = params->clip_box.yMin;
      ras.max_ex = params->clip_box.xMax;
      ras.max_ey = params->clip_box.yMax;
    }
    else
    {
      /* if direct mode is not set, we must have a target bitmap */
      if ( !target_map )
        return FT_THROW( Invalid_Argument );

      /* nothing to do */
      if ( !target_map->width || !target_map->rows )
        return 0;

      if ( !target_map->buffer )
        return FT_THROW( Invalid_Argument );

      if ( target_map->pitch < 0 )
        ras.target.origin = target_map->buffer;
      else
        ras.target.origin = target_map->buffer
              + ( target_map->rows - 1 ) * (unsigned int)target_map->pitch;

      ras.target.pitch = target_map->pitch;

      ras.render_span      = (FT_Raster_Span_Func)NULL;
      ras.render_span_data = NULL;
      ras.num_spans        = -1;  /* invalid */

      ras.min_ex = 0;
      ras.min_ey = 0;
      ras.max_ex = (FT_Pos)target_map->width;
      ras.max_ey = (FT_Pos)target_map->rows;
    }

    /* exit if nothing to do */
    if ( ras.max_ex <= ras.min_ex || ras.max_ey <= ras.min_ey )
      return 0;

    return gray_convert_glyph( RAS_VAR );
  }


  /**** RASTER OBJECT CREATION: In stand-alone mode, we simply use *****/
  /****                         a static object.                   *****/

#ifdef STANDALONE_

  static int
  gray_raster_new( void*       memory,
                   FT_Raster*  araster )
  {
    static gray_TRaster  the_raster;

    FT_UNUSED( memory );


    *araster = (FT_Raster)&the_raster;
    FT_ZERO( &the_raster );

    return 0;
  }


  static void
  gray_raster_done( FT_Raster  raster )
  {
    /* nothing */
    FT_UNUSED( raster );
  }

#else /* !STANDALONE_ */

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

#endif /* !STANDALONE_ */


  static void
  gray_raster_reset( FT_Raster       raster,
                     unsigned char*  pool_base,
                     unsigned long   pool_size )
  {
    FT_UNUSED( raster );
    FT_UNUSED( pool_base );
    FT_UNUSED( pool_size );
  }


  static int
  gray_raster_set_mode( FT_Raster      raster,
                        unsigned long  mode,
                        void*          args )
  {
    FT_UNUSED( raster );
    FT_UNUSED( mode );
    FT_UNUSED( args );


    return 0; /* nothing to do */
  }


  FT_DEFINE_RASTER_FUNCS(
    ft_grays_raster,

    FT_GLYPH_FORMAT_OUTLINE,

    (FT_Raster_New_Func)     gray_raster_new,       /* raster_new      */
    (FT_Raster_Reset_Func)   gray_raster_reset,     /* raster_reset    */
    (FT_Raster_Set_Mode_Func)gray_raster_set_mode,  /* raster_set_mode */
    (FT_Raster_Render_Func)  gray_raster_render,    /* raster_render   */
    (FT_Raster_Done_Func)    gray_raster_done       /* raster_done     */
  )


/* END */


/* Local Variables: */
/* coding: utf-8    */
/* End:             */

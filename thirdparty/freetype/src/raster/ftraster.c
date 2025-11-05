/****************************************************************************
 *
 * ftraster.c
 *
 *   The FreeType glyph rasterizer (body).
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

  /**************************************************************************
   *
   * This file can be compiled without the rest of the FreeType engine, by
   * defining the STANDALONE_ macro when compiling it.  You also need to
   * put the files `ftimage.h' and `ftmisc.h' into the $(incdir)
   * directory.  Typically, you should do something like
   *
   * - copy `src/raster/ftraster.c' (this file) to your current directory
   *
   * - copy `include/freetype/ftimage.h' and `src/raster/ftmisc.h' to your
   *   current directory
   *
   * - compile `ftraster' with the STANDALONE_ macro defined, as in
   *
   *     cc -c -DSTANDALONE_ ftraster.c
   *
   * The renderer can be initialized with a call to
   * `ft_standard_raster.raster_new'; a bitmap can be generated
   * with a call to `ft_standard_raster.raster_render'.
   *
   * See the comments and documentation in the file `ftimage.h' for more
   * details on how the raster works.
   *
   */


  /**************************************************************************
   *
   * This is a rewrite of the FreeType 1.x scan-line converter
   *
   */

#ifdef STANDALONE_

  /* The size in bytes of the render pool used by the scan-line converter  */
  /* to do all of its work.                                                */
#define FT_RENDER_POOL_SIZE  16384L

#define FT_CONFIG_STANDARD_LIBRARY_H  <stdlib.h>

#include <string.h>           /* for memset */

#include "ftmisc.h"
#include "ftimage.h"

#else /* !STANDALONE_ */

#include "ftraster.h"
#include <freetype/internal/ftcalc.h> /* for FT_MulDiv_No_Round */

#endif /* !STANDALONE_ */


  /**************************************************************************
   *
   * A simple technical note on how the raster works
   * -----------------------------------------------
   *
   *   Converting an outline into a bitmap is achieved in several steps:
   *
   *   1 - Decomposing the outline into successive `profiles'.  Each
   *       profile is simply an array of scanline intersections on a given
   *       dimension.  A profile's main attributes are
   *
   *       o its scanline position boundaries, i.e. `Ymin' and `Ymax'
   *
   *       o an array of intersection coordinates for each scanline
   *         between `Ymin' and `Ymax'
   *
   *       o a direction, indicating whether it was built going `up' or
   *         `down', as this is very important for filling rules
   *
   *       o its drop-out mode
   *
   *   2 - Sweeping the target map's scanlines in order to compute segment
   *       `spans' which are then filled.  Additionally, this pass
   *       performs drop-out control.
   *
   *   The outline data is parsed during step 1 only.  The profiles are
   *   built from the bottom of the render pool, used as a stack.  The
   *   following graphics shows the profile list under construction:
   *
   *    __________________________________________________________ _ _
   *   |         |                 |         |                 |
   *   | profile | coordinates for | profile | coordinates for |-->
   *   |    1    |  profile 1      |    2    |  profile 2      |-->
   *   |_________|_________________|_________|_________________|__ _ _
   *
   *   ^                                                       ^
   *   |                                                       |
   * start of render pool                                      top
   *
   *   The top of the profile stack is kept in the `top' variable.
   *
   *   As you can see, a profile record is pushed on top of the render
   *   pool, which is then followed by its coordinates/intersections.  If
   *   a change of direction is detected in the outline, a new profile is
   *   generated until the end of the outline.
   *
   *   Note that, for all generated profiles, the function End_Profile()
   *   is used to record all their bottom-most scanlines as well as the
   *   scanline above their upmost boundary.  These positions are called
   *   `y-turns' because they (sort of) correspond to local extrema.
   *   They are stored in a sorted list built from the top of the render
   *   pool as a downwards stack:
   *
   *     _ _ _______________________________________
   *                           |                    |
   *                        <--| sorted list of     |
   *                        <--|  extrema scanlines |
   *     _ _ __________________|____________________|
   *
   *                           ^                    ^
   *                           |                    |
   *                         maxBuff           sizeBuff = end of pool
   *
   *   This list is later used during the sweep phase in order to
   *   optimize performance (see technical note on the sweep below).
   *
   *   Of course, the raster detects whether the two stacks collide and
   *   handles the situation by bisecting the job and restarting.
   *
   */


  /*************************************************************************/
  /*************************************************************************/
  /**                                                                     **/
  /**  CONFIGURATION MACROS                                               **/
  /**                                                                     **/
  /*************************************************************************/
  /*************************************************************************/


  /*************************************************************************/
  /*************************************************************************/
  /**                                                                     **/
  /**  OTHER MACROS (do not change)                                       **/
  /**                                                                     **/
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  raster


#ifdef STANDALONE_

  /* Auxiliary macros for token concatenation. */
#define FT_ERR_XCAT( x, y )  x ## y
#define FT_ERR_CAT( x, y )   FT_ERR_XCAT( x, y )

  /* This macro is used to indicate that a function parameter is unused. */
  /* Its purpose is simply to reduce compiler warnings.  Note also that  */
  /* simply defining it as `(void)x' doesn't avoid warnings with certain */
  /* ANSI compilers (e.g. LCC).                                          */
#define FT_UNUSED( x )  (x) = (x)

  /* Disable the tracing mechanism for simplicity -- developers can      */
  /* activate it easily by redefining these macros.                      */
#ifndef FT_ERROR
#define FT_ERROR( x )  do { } while ( 0 )     /* nothing */
#endif

#ifndef FT_TRACE
#define FT_TRACE( x )   do { } while ( 0 )    /* nothing */
#define FT_TRACE1( x )  do { } while ( 0 )    /* nothing */
#define FT_TRACE6( x )  do { } while ( 0 )    /* nothing */
#define FT_TRACE7( x )  do { } while ( 0 )    /* nothing */
#endif

#ifndef FT_THROW
#define FT_THROW( e )  FT_ERR_CAT( Raster_Err_, e )
#endif

#define Raster_Err_Ok                       0
#define Raster_Err_Invalid_Outline         -1
#define Raster_Err_Cannot_Render_Glyph     -2
#define Raster_Err_Invalid_Argument        -3
#define Raster_Err_Raster_Overflow         -4
#define Raster_Err_Raster_Uninitialized    -5
#define Raster_Err_Raster_Negative_Height  -6

#define ft_memset  memset

#define FT_DEFINE_RASTER_FUNCS( class_, glyph_format_, raster_new_, \
                                raster_reset_, raster_set_mode_,    \
                                raster_render_, raster_done_ )      \
          const FT_Raster_Funcs class_ =                            \
          {                                                         \
            glyph_format_,                                          \
            raster_new_,                                            \
            raster_reset_,                                          \
            raster_set_mode_,                                       \
            raster_render_,                                         \
            raster_done_                                            \
         };

#else /* !STANDALONE_ */


#include <freetype/internal/ftobjs.h>
#include <freetype/internal/ftdebug.h> /* for FT_TRACE, FT_ERROR, and FT_THROW */

#include "rasterrs.h"


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

  /* FMulDiv means `Fast MulDiv'; it is used in case where `b' is       */
  /* typically a small value and the result of a*b is known to fit into */
  /* 32 bits.                                                           */
#define FMulDiv( a, b, c )  ( (a) * (b) / (c) )

  /* On the other hand, SMulDiv means `Slow MulDiv', and is used typically */
  /* for clipping computations.  It simply uses the FT_MulDiv() function   */
  /* defined in `ftcalc.h'.                                                */
#ifdef FT_INT64
#define SMulDiv( a, b, c )  (Long)( (FT_Int64)(a) * (b) / (c) )
#else
#define SMulDiv  FT_MulDiv_No_Round
#endif

  /* The rasterizer is a very general purpose component; please leave */
  /* the following redefinitions there (you never know your target    */
  /* environment).                                                    */

#ifndef TRUE
#define TRUE   1
#endif

#ifndef FALSE
#define FALSE  0
#endif

#ifndef NULL
#define NULL  (void*)0
#endif

#ifndef SUCCESS
#define SUCCESS  0
#endif

#ifndef FAILURE
#define FAILURE  1
#endif


#define MaxBezier  32   /* The maximum number of stacked Bezier curves. */
                        /* Setting this constant to more than 32 is a   */
                        /* pure waste of space.                         */

#define Pixel_Bits  6   /* fractional bits of *input* coordinates */


  /*************************************************************************/
  /*************************************************************************/
  /**                                                                     **/
  /**  SIMPLE TYPE DECLARATIONS                                           **/
  /**                                                                     **/
  /*************************************************************************/
  /*************************************************************************/

  typedef int             Int;
  typedef unsigned int    UInt;
  typedef short           Short;
  typedef unsigned short  UShort, *PUShort;
  typedef long            Long, *PLong;
  typedef unsigned long   ULong;

  typedef unsigned char   Byte, *PByte;
  typedef char            Bool;

  typedef struct  TPoint_
  {
    Long  x;
    Long  y;

  } TPoint;


  /* values for the `flags' bit field */
#define Flow_Up           0x08U
#define Overshoot_Top     0x10U
#define Overshoot_Bottom  0x20U
#define Dropout           0x40U


  /* States of each line, arc, and profile */
  typedef enum  TStates_
  {
    Unknown_State,
    Ascending_State,
    Descending_State,
    Flat_State

  } TStates;


  typedef struct TProfile_  TProfile;
  typedef TProfile*         PProfile;

  struct  TProfile_
  {
    PProfile    link;        /* link to next profile (various purposes)  */
    PProfile    next;        /* next profile in same contour, used       */
                             /* during drop-out control                  */
    Int         offset;      /* bottom or currently scanned array index  */
    Int         height;      /* profile's height in scanlines            */
    Int         start;       /* profile's starting scanline, also use    */
                             /* as activation counter                    */
    UShort      flags;       /* Bit 0-2: drop-out mode                   */
                             /* Bit 3: profile orientation (up/down)     */
                             /* Bit 4: is top profile?                   */
                             /* Bit 5: is bottom profile?                */
                             /* Bit 6: dropout detected                  */

    FT_F26Dot6  X;           /* current coordinate during sweep          */
    Long        x[1];        /* actually variable array of scanline      */
                             /* intersections with `height` elements     */
  };

  typedef PProfile   TProfileList;
  typedef PProfile*  PProfileList;


#undef RAS_ARG
#undef RAS_ARGS
#undef RAS_VAR
#undef RAS_VARS

#ifdef FT_STATIC_RASTER


#define RAS_ARGS       /* void */
#define RAS_ARG        void

#define RAS_VARS       /* void */
#define RAS_VAR        /* void */

#define FT_UNUSED_RASTER  do { } while ( 0 )


#else /* !FT_STATIC_RASTER */


#define RAS_ARGS       black_PWorker  worker,
#define RAS_ARG        black_PWorker  worker

#define RAS_VARS       worker,
#define RAS_VAR        worker

#define FT_UNUSED_RASTER  FT_UNUSED( worker )


#endif /* !FT_STATIC_RASTER */


  typedef struct black_TWorker_  black_TWorker, *black_PWorker;


  /* prototypes used for sweep function dispatch */
  typedef void
  Function_Sweep_Init( RAS_ARGS Int  min,
                                Int  max );

  typedef void
  Function_Sweep_Span( RAS_ARGS Int         y,
                                FT_F26Dot6  x1,
                                FT_F26Dot6  x2 );

  typedef void
  Function_Sweep_Step( RAS_ARG );


  /* NOTE: These operations are only valid on 2's complement processors */
#undef FLOOR
#undef CEILING
#undef TRUNC
#undef SCALED

#define FLOOR( x )    ( (x) & -ras.precision )
#define CEILING( x )  ( ( (x) + ras.precision - 1 ) & -ras.precision )
#define TRUNC( x )    ( (Long)(x) >> ras.precision_bits )
#define FRAC( x )     ( (x) & ( ras.precision - 1 ) )

  /* scale and shift grid to pixel centers */
#define SCALED( x )   ( (x) * ras.precision_scale - ras.precision_half )

#define IS_BOTTOM_OVERSHOOT( x ) \
          (Bool)( CEILING( x ) - x >= ras.precision_half )
#define IS_TOP_OVERSHOOT( x )    \
          (Bool)( x - FLOOR( x ) >= ras.precision_half )

  /* Smart dropout rounding to find which pixel is closer to span ends. */
  /* To mimic Windows, symmetric cases do not depend on the precision.  */
#define SMART( p, q )  FLOOR( ( (p) + (q) + ras.precision * 63 / 64 ) >> 1 )

#if FT_RENDER_POOL_SIZE > 2048
#define FT_MAX_BLACK_POOL  ( FT_RENDER_POOL_SIZE / sizeof ( Long ) )
#else
#define FT_MAX_BLACK_POOL  ( 2048 / sizeof ( Long ) )
#endif

  /* The most used variables are positioned at the top of the structure. */
  /* Thus, their offset can be coded with less opcodes, resulting in a   */
  /* smaller executable.                                                 */

  struct  black_TWorker_
  {
    Int         precision_bits;     /* precision related variables         */
    Int         precision;
    Int         precision_half;
    Int         precision_scale;
    Int         precision_step;

    PLong       buff;               /* The profiles buffer                 */
    PLong       sizeBuff;           /* Render pool size                    */
    PLong       maxBuff;            /* Profiles buffer size                */
    PLong       top;                /* Current cursor in buffer            */

    FT_Error    error;

    Byte        dropOutControl;     /* current drop_out control method     */

    Long        lastX, lastY;
    Long        minY, maxY;

    UShort      num_Profs;          /* current number of profiles          */
    Int         numTurns;           /* number of Y-turns in outline        */

    PProfile    cProfile;           /* current profile                     */
    PProfile    fProfile;           /* head of linked list of profiles     */
    PProfile    gProfile;           /* contour's first profile in case     */
                                    /* of impact                           */

    TStates     state;              /* rendering state                     */

    FT_Outline  outline;

    Int         bTop;               /* target bitmap max line  index       */
    Int         bRight;             /* target bitmap rightmost index       */
    Int         bPitch;             /* target bitmap pitch                 */
    PByte       bOrigin;            /* target bitmap bottom-left origin    */
    PByte       bLine;              /* target bitmap current line          */

    /* dispatch variables */

    Function_Sweep_Init*  Proc_Sweep_Init;
    Function_Sweep_Span*  Proc_Sweep_Span;
    Function_Sweep_Span*  Proc_Sweep_Drop;
    Function_Sweep_Step*  Proc_Sweep_Step;

  };


  typedef struct  black_TRaster_
  {
    void*          memory;

  } black_TRaster, *black_PRaster;

#ifdef FT_STATIC_RASTER

  static black_TWorker  ras;

#else /* !FT_STATIC_RASTER */

#define ras  (*worker)

#endif /* !FT_STATIC_RASTER */


  /*************************************************************************/
  /*************************************************************************/
  /**                                                                     **/
  /**  PROFILES COMPUTATION                                               **/
  /**                                                                     **/
  /*************************************************************************/
  /*************************************************************************/


  /**************************************************************************
   *
   * @Function:
   *   Set_High_Precision
   *
   * @Description:
   *   Set precision variables according to param flag.
   *
   * @Input:
   *   High ::
   *     Set to True for high precision (typically for ppem < 24),
   *     false otherwise.
   */
  static void
  Set_High_Precision( RAS_ARGS Int  High )
  {
    /*
     * `precision_step' is used in `Bezier_Up' to decide when to split a
     * given y-monotonous Bezier arc that crosses a scanline before
     * approximating it as a straight segment.  The default value of 32 (for
     * low accuracy) corresponds to
     *
     *   32 / 64 == 0.5 pixels,
     *
     * while for the high accuracy case we have
     *
     *   256 / (1 << 12) = 0.0625 pixels.
     *
     */

    if ( High )
    {
      ras.precision_bits   = 12;
      ras.precision_step   = 256;
    }
    else
    {
      ras.precision_bits   = 6;
      ras.precision_step   = 32;
    }

    ras.precision       = 1 << ras.precision_bits;
    ras.precision_half  = ras.precision >> 1;
    ras.precision_scale = ras.precision >> Pixel_Bits;
  }


  /**************************************************************************
   *
   * @Function:
   *   Insert_Y_Turn
   *
   * @Description:
   *   Insert a salient into the sorted list placed on top of the render
   *   pool.
   *
   * @Input:
   *   New y scanline position.
   *
   * @Return:
   *   SUCCESS on success.  FAILURE in case of overflow.
   */
  static Bool
  Insert_Y_Turns( RAS_ARGS Int  y,
                           Int  top )
  {
    Int    n       = ras.numTurns;
    PLong  y_turns = ras.maxBuff;


    /* update top value */
    if ( n == 0 || top > y_turns[n] )
      y_turns[n] = top;

    /* look for first y value that is <= */
    while ( n-- && y < y_turns[n] )
      ;

    /* if it is <, simply insert it, ignore if == */
    if ( n < 0 || y > y_turns[n] )
    {
      ras.maxBuff--;
      if ( ras.maxBuff <= ras.top )
      {
        ras.error = FT_THROW( Raster_Overflow );
        return FAILURE;
      }

      do
      {
        Int  y2 = (Int)y_turns[n];


        y_turns[n] = y;
        y = y2;
      } while ( n-- >= 0 );

      ras.numTurns++;
    }

    return SUCCESS;
  }


  /**************************************************************************
   *
   * @Function:
   *   New_Profile
   *
   * @Description:
   *   Create a new profile in the render pool.
   *
   * @Input:
   *   aState ::
   *     The state/orientation of the new profile.
   *
   * @Return:
   *  SUCCESS on success.  FAILURE in case of overflow or of incoherent
   *  profile.
   */
  static Bool
  New_Profile( RAS_ARGS TStates  aState )
  {
    Long  e;


    if ( !ras.cProfile || ras.cProfile->height )
    {
      ras.cProfile  = (PProfile)ras.top;
      ras.top       = ras.cProfile->x;

      if ( ras.top >= ras.maxBuff )
      {
        FT_TRACE1(( "overflow in New_Profile\n" ));
        ras.error = FT_THROW( Raster_Overflow );
        return FAILURE;
      }

      ras.cProfile->height = 0;
    }

    ras.cProfile->flags = ras.dropOutControl;

    switch ( aState )
    {
    case Ascending_State:
      ras.cProfile->flags |= Flow_Up;
      if ( IS_BOTTOM_OVERSHOOT( ras.lastY ) )
        ras.cProfile->flags |= Overshoot_Bottom;

      e = CEILING( ras.lastY );
      break;

    case Descending_State:
      if ( IS_TOP_OVERSHOOT( ras.lastY ) )
        ras.cProfile->flags |= Overshoot_Top;

      e = FLOOR( ras.lastY );
      break;

    default:
      FT_ERROR(( "New_Profile: invalid profile direction\n" ));
      ras.error = FT_THROW( Invalid_Outline );
      return FAILURE;
    }

    if ( e > ras.maxY )
      e = ras.maxY;
    if ( e < ras.minY )
      e = ras.minY;
    ras.cProfile->start = (Int)TRUNC( e );

    FT_TRACE7(( "  new %s profile = %p, start = %d\n",
                aState == Ascending_State ? "ascending" : "descending",
                (void *)ras.cProfile, ras.cProfile->start ));

    if ( ras.lastY == e )
      *ras.top++ = ras.lastX;

    ras.state = aState;

    return SUCCESS;
  }


  /**************************************************************************
   *
   * @Function:
   *   End_Profile
   *
   * @Description:
   *   Finalize the current profile and record y-turns.
   *
   * @Return:
   *   SUCCESS on success.  FAILURE in case of overflow or incoherency.
   */
  static Bool
  End_Profile( RAS_ARG )
  {
    PProfile  p = ras.cProfile;
    Int       h = (Int)( ras.top - p->x );
    Int       bottom, top;


    if ( h < 0 )
    {
      FT_ERROR(( "End_Profile: negative height encountered\n" ));
      ras.error = FT_THROW( Raster_Negative_Height );
      return FAILURE;
    }

    if ( h > 0 )
    {
      FT_TRACE7(( "  ending profile %p, start = %2d, height = %+3d\n",
                  (void *)p, p->start, p->flags & Flow_Up ? h : -h ));

      p->height = h;

      if ( p->flags & Flow_Up )
      {
        if ( IS_TOP_OVERSHOOT( ras.lastY ) )
          p->flags |= Overshoot_Top;

        bottom    = p->start;
        top       = bottom + h;
        p->offset = 0;
        p->X      = p->x[0];
      }
      else
      {
        if ( IS_BOTTOM_OVERSHOOT( ras.lastY ) )
          p->flags |= Overshoot_Bottom;

        top       = p->start + 1;
        bottom    = top - h;
        p->start  = bottom;
        p->offset = h - 1;
        p->X      = p->x[h - 1];
      }

      if ( Insert_Y_Turns( RAS_VARS bottom, top ) )
        return FAILURE;

      if ( !ras.gProfile )
        ras.gProfile = p;

      /* preliminary values to be finalized */
      p->next = ras.gProfile;
      p->link = (PProfile)ras.top;

      ras.num_Profs++;
    }

    return SUCCESS;
  }


  /**************************************************************************
   *
   * @Function:
   *   Finalize_Profile_Table
   *
   * @Description:
   *   Adjust all links in the profiles list.
   */
  static void
  Finalize_Profile_Table( RAS_ARG )
  {
    UShort    n = ras.num_Profs;
    PProfile  p = ras.fProfile;
    PProfile  q;


    /* there should be at least two profiles, up and down */
    while ( --n )
    {
      q = p->link;

      /* fix the contour loop */
      if ( q->next == p->next )
        p->next = q;

      p = q;
    }

    /* null-terminate */
    p->link = NULL;
  }


  /**************************************************************************
   *
   * @Function:
   *   Split_Conic
   *
   * @Description:
   *   Subdivide one conic Bezier into two joint sub-arcs in the Bezier
   *   stack.
   *
   * @Input:
   *   None (subdivided Bezier is taken from the top of the stack).
   *
   * @Note:
   *   This routine is the `beef' of this component.  It is  _the_ inner
   *   loop that should be optimized to hell to get the best performance.
   */
  static void
  Split_Conic( TPoint*  base )
  {
    Long  a, b;


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

    /* hand optimized.  gcc doesn't seem to be too good at common      */
    /* expression substitution and instruction scheduling ;-)          */
  }


  /**************************************************************************
   *
   * @Function:
   *   Split_Cubic
   *
   * @Description:
   *   Subdivide a third-order Bezier arc into two joint sub-arcs in the
   *   Bezier stack.
   *
   * @Note:
   *   This routine is the `beef' of the component.  It is one of _the_
   *   inner loops that should be optimized like hell to get the best
   *   performance.
   */
  static void
  Split_Cubic( TPoint*  base )
  {
    Long  a, b, c;


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


  /**************************************************************************
   *
   * @Function:
   *   Line_Up
   *
   * @Description:
   *   Compute the x-coordinates of an ascending line segment and store
   *   them in the render pool.
   *
   * @Input:
   *   x1 ::
   *     The x-coordinate of the segment's start point.
   *
   *   y1 ::
   *     The y-coordinate of the segment's start point.
   *
   *   x2 ::
   *     The x-coordinate of the segment's end point.
   *
   *   y2 ::
   *     The y-coordinate of the segment's end point.
   *
   *   miny ::
   *     A lower vertical clipping bound value.
   *
   *   maxy ::
   *     An upper vertical clipping bound value.
   *
   * @Return:
   *   SUCCESS on success, FAILURE on render pool overflow.
   */
  static Bool
  Line_Up( RAS_ARGS Long  x1,
                    Long  y1,
                    Long  x2,
                    Long  y2,
                    Long  miny,
                    Long  maxy )
  {
    Long  e, e2, Dx, Dy;
    Long  Ix, Rx, Ax;
    Int   size;

    PLong  top;


    if ( y2 < miny || y1 > maxy )
      return SUCCESS;

    e2 = y2 > maxy ? maxy : FLOOR( y2 );
    e  = y1 < miny ? miny : CEILING( y1 );

    if ( y1 == e )
      e += ras.precision;

    if ( e2 < e )  /* nothing to do */
      return SUCCESS;

    size = (Int)TRUNC( e2 - e ) + 1;
    top  = ras.top;

    if ( top + size >= ras.maxBuff )
    {
      ras.error = FT_THROW( Raster_Overflow );
      return FAILURE;
    }

    Dx = x2 - x1;
    Dy = y2 - y1;

    if ( Dx == 0 )  /* very easy */
    {
      do
        *top++ = x1;
      while ( --size );
      goto Fin;
    }

    Ix     = SMulDiv( e - y1, Dx, Dy );
    x1    += Ix;
    *top++ = x1;

    if ( --size )
    {
      Ax = Dx * ( e - y1 )    - Dy * Ix;  /* remainder */
      Ix = SMulDiv( ras.precision, Dx, Dy );
      Rx = Dx * ras.precision - Dy * Ix;  /* remainder */
      Dx = 1;

      if ( x2 < x1 )
      {
        Ax = -Ax;
        Rx = -Rx;
        Dx = -Dx;
      }

      do
      {
        x1 += Ix;
        Ax += Rx;
        if ( Ax >= Dy )
        {
          Ax -= Dy;
          x1 += Dx;
        }
        *top++ = x1;
      }
      while ( --size );
    }

  Fin:
    ras.top = top;
    return SUCCESS;
  }


  /**************************************************************************
   *
   * @Function:
   *   Line_Down
   *
   * @Description:
   *   Compute the x-coordinates of an descending line segment and store
   *   them in the render pool.
   *
   * @Input:
   *   x1 ::
   *     The x-coordinate of the segment's start point.
   *
   *   y1 ::
   *     The y-coordinate of the segment's start point.
   *
   *   x2 ::
   *     The x-coordinate of the segment's end point.
   *
   *   y2 ::
   *     The y-coordinate of the segment's end point.
   *
   *   miny ::
   *     A lower vertical clipping bound value.
   *
   *   maxy ::
   *     An upper vertical clipping bound value.
   *
   * @Return:
   *   SUCCESS on success, FAILURE on render pool overflow.
   */
  static Bool
  Line_Down( RAS_ARGS Long  x1,
                      Long  y1,
                      Long  x2,
                      Long  y2,
                      Long  miny,
                      Long  maxy )
  {
    return Line_Up( RAS_VARS x1, -y1, x2, -y2, -maxy, -miny );
  }


  /* A function type describing the functions used to split Bezier arcs */
  typedef void  (*TSplitter)( TPoint*  base );


  /**************************************************************************
   *
   * @Function:
   *   Bezier_Up
   *
   * @Description:
   *   Compute the x-coordinates of an ascending Bezier arc and store
   *   them in the render pool.
   *
   * @Input:
   *   degree ::
   *     The degree of the Bezier arc (either 2 or 3).
   *
   *   splitter ::
   *     The function to split Bezier arcs.
   *
   *   miny ::
   *     A lower vertical clipping bound value.
   *
   *   maxy ::
   *     An upper vertical clipping bound value.
   *
   * @Return:
   *   SUCCESS on success, FAILURE on render pool overflow.
   */
  static Bool
  Bezier_Up( RAS_ARGS Int        degree,
                      TPoint*    arc,
                      TSplitter  splitter,
                      Long       miny,
                      Long       maxy )
  {
    Long  y1, y2, e, e2, dy;
    Long  dx, x2;

    PLong  top;


    y1 = arc[degree].y;
    y2 = arc[0].y;

    if ( y2 < miny || y1 > maxy )
      return SUCCESS;

    e2 = y2 > maxy ? maxy : FLOOR( y2 );
    e  = y1 < miny ? miny : CEILING( y1 );

    if ( y1 == e )
      e += ras.precision;

    if ( e2 < e )  /* nothing to do */
      return SUCCESS;

    top = ras.top;

    if ( ( top + TRUNC( e2 - e ) + 1 ) >= ras.maxBuff )
    {
      ras.error = FT_THROW( Raster_Overflow );
      return FAILURE;
    }

    do
    {
      y2 = arc[0].y;
      x2 = arc[0].x;

      if ( y2 > e )
      {
        dy = y2 - arc[degree].y;
        dx = x2 - arc[degree].x;

        /* split condition should be invariant of direction */
        if (  dy > ras.precision_step ||
              dx > ras.precision_step ||
             -dx > ras.precision_step )
        {
          splitter( arc );
          arc += degree;
        }
        else
        {
          *top++ = x2 - FMulDiv( y2 - e, dx, dy );
          e     += ras.precision;
          arc -= degree;
        }
      }
      else
      {
        if ( y2 == e )
        {
          *top++ = x2;
          e     += ras.precision;
        }
        arc   -= degree;
      }
    }
    while ( e <= e2 );

    ras.top = top;
    return SUCCESS;
  }


  /**************************************************************************
   *
   * @Function:
   *   Bezier_Down
   *
   * @Description:
   *   Compute the x-coordinates of an descending Bezier arc and store
   *   them in the render pool.
   *
   * @Input:
   *   degree ::
   *     The degree of the Bezier arc (either 2 or 3).
   *
   *   splitter ::
   *     The function to split Bezier arcs.
   *
   *   miny ::
   *     A lower vertical clipping bound value.
   *
   *   maxy ::
   *     An upper vertical clipping bound value.
   *
   * @Return:
   *   SUCCESS on success, FAILURE on render pool overflow.
   */
  static Bool
  Bezier_Down( RAS_ARGS Int        degree,
                        TPoint*    arc,
                        TSplitter  splitter,
                        Long       miny,
                        Long       maxy )
  {
    Bool  result;


    arc[0].y = -arc[0].y;
    arc[1].y = -arc[1].y;
    arc[2].y = -arc[2].y;
    if ( degree > 2 )
      arc[3].y = -arc[3].y;

    result = Bezier_Up( RAS_VARS degree, arc, splitter, -maxy, -miny );

    arc[0].y = -arc[0].y;
    return result;
  }


  /**************************************************************************
   *
   * @Function:
   *   Line_To
   *
   * @Description:
   *   Inject a new line segment and adjust the Profiles list.
   *
   * @Input:
   *  x ::
   *    The x-coordinate of the segment's end point (its start point
   *    is stored in `lastX').
   *
   *  y ::
   *    The y-coordinate of the segment's end point (its start point
   *    is stored in `lastY').
   *
   * @Return:
   *  SUCCESS on success, FAILURE on render pool overflow or incorrect
   *  profile.
   */
  static Bool
  Line_To( RAS_ARGS Long  x,
                    Long  y )
  {
    TStates  state;


    if ( y == ras.lastY )
      goto Fin;

    /* First, detect a change of direction */

    state = ras.lastY < y ? Ascending_State : Descending_State;

    if ( ras.state != state )
    {
      /* finalize current profile if any */
      if ( ras.state != Unknown_State &&
           End_Profile( RAS_VAR )     )
        goto Fail;

      /* create a new profile */
      if ( New_Profile( RAS_VARS state ) )
        goto Fail;
    }

    /* Then compute the lines */

    if ( state == Ascending_State )
    {
      if ( Line_Up( RAS_VARS ras.lastX, ras.lastY,
                             x, y, ras.minY, ras.maxY ) )
        goto Fail;
    }
    else
    {
      if ( Line_Down( RAS_VARS ras.lastX, ras.lastY,
                               x, y, ras.minY, ras.maxY ) )
        goto Fail;
    }

  Fin:
    ras.lastX = x;
    ras.lastY = y;
    return SUCCESS;

  Fail:
    return FAILURE;
  }


  /**************************************************************************
   *
   * @Function:
   *   Conic_To
   *
   * @Description:
   *   Inject a new conic arc and adjust the profile list.
   *
   * @Input:
   *  cx ::
   *    The x-coordinate of the arc's new control point.
   *
   *  cy ::
   *    The y-coordinate of the arc's new control point.
   *
   *  x ::
   *    The x-coordinate of the arc's end point (its start point is
   *    stored in `lastX').
   *
   *  y ::
   *    The y-coordinate of the arc's end point (its start point is
   *    stored in `lastY').
   *
   * @Return:
   *  SUCCESS on success, FAILURE on render pool overflow or incorrect
   *  profile.
   */
  static Bool
  Conic_To( RAS_ARGS Long  cx,
                     Long  cy,
                     Long  x,
                     Long  y )
  {
    Long     y1, y2, y3, x3, ymin, ymax;
    TStates  state_bez;
    TPoint   arcs[2 * MaxBezier + 1]; /* The Bezier stack           */
    TPoint*  arc;                     /* current Bezier arc pointer */


    arc      = arcs;
    arc[2].x = ras.lastX;
    arc[2].y = ras.lastY;
    arc[1].x = cx;
    arc[1].y = cy;
    arc[0].x = x;
    arc[0].y = y;

    do
    {
      y1 = arc[2].y;
      y2 = arc[1].y;
      y3 = arc[0].y;
      x3 = arc[0].x;

      /* first, categorize the Bezier arc */

      if ( y1 <= y3 )
      {
        ymin = y1;
        ymax = y3;
      }
      else
      {
        ymin = y3;
        ymax = y1;
      }

      if ( y2 < FLOOR( ymin ) || y2 > CEILING( ymax ) )
      {
        /* this arc has no given direction, split it! */
        Split_Conic( arc );
        arc += 2;
      }
      else if ( y1 == y3 )
      {
        /* this arc is flat, advance position */
        /* and pop it from the Bezier stack   */
        arc -= 2;

        ras.lastX = x3;
        ras.lastY = y3;
      }
      else
      {
        /* the arc is y-monotonous, either ascending or descending */
        /* detect a change of direction                            */
        state_bez = y1 < y3 ? Ascending_State : Descending_State;
        if ( ras.state != state_bez )
        {
          /* finalize current profile if any */
          if ( ras.state != Unknown_State &&
               End_Profile( RAS_VAR )     )
            goto Fail;

          /* create a new profile */
          if ( New_Profile( RAS_VARS state_bez ) )
            goto Fail;
        }

        /* now call the appropriate routine */
        if ( state_bez == Ascending_State )
        {
          if ( Bezier_Up( RAS_VARS 2, arc, Split_Conic,
                                   ras.minY, ras.maxY ) )
            goto Fail;
        }
        else
          if ( Bezier_Down( RAS_VARS 2, arc, Split_Conic,
                                     ras.minY, ras.maxY ) )
            goto Fail;
        arc -= 2;

        ras.lastX = x3;
        ras.lastY = y3;
      }

    } while ( arc >= arcs );

    return SUCCESS;

  Fail:
    return FAILURE;
  }


  /**************************************************************************
   *
   * @Function:
   *   Cubic_To
   *
   * @Description:
   *   Inject a new cubic arc and adjust the profile list.
   *
   * @Input:
   *  cx1 ::
   *    The x-coordinate of the arc's first new control point.
   *
   *  cy1 ::
   *    The y-coordinate of the arc's first new control point.
   *
   *  cx2 ::
   *    The x-coordinate of the arc's second new control point.
   *
   *  cy2 ::
   *    The y-coordinate of the arc's second new control point.
   *
   *  x ::
   *    The x-coordinate of the arc's end point (its start point is
   *    stored in `lastX').
   *
   *  y ::
   *    The y-coordinate of the arc's end point (its start point is
   *    stored in `lastY').
   *
   * @Return:
   *  SUCCESS on success, FAILURE on render pool overflow or incorrect
   *  profile.
   */
  static Bool
  Cubic_To( RAS_ARGS Long  cx1,
                     Long  cy1,
                     Long  cx2,
                     Long  cy2,
                     Long  x,
                     Long  y )
  {
    Long     y1, y2, y3, y4, x4, ymin1, ymax1, ymin2, ymax2;
    TStates  state_bez;
    TPoint   arcs[3 * MaxBezier + 1]; /* The Bezier stack           */
    TPoint*  arc;                     /* current Bezier arc pointer */


    arc      = arcs;
    arc[3].x = ras.lastX;
    arc[3].y = ras.lastY;
    arc[2].x = cx1;
    arc[2].y = cy1;
    arc[1].x = cx2;
    arc[1].y = cy2;
    arc[0].x = x;
    arc[0].y = y;

    do
    {
      y1 = arc[3].y;
      y2 = arc[2].y;
      y3 = arc[1].y;
      y4 = arc[0].y;
      x4 = arc[0].x;

      /* first, categorize the Bezier arc */

      if ( y1 <= y4 )
      {
        ymin1 = y1;
        ymax1 = y4;
      }
      else
      {
        ymin1 = y4;
        ymax1 = y1;
      }

      if ( y2 <= y3 )
      {
        ymin2 = y2;
        ymax2 = y3;
      }
      else
      {
        ymin2 = y3;
        ymax2 = y2;
      }

      if ( ymin2 < FLOOR( ymin1 ) || ymax2 > CEILING( ymax1 ) )
      {
        /* this arc has no given direction, split it! */
        Split_Cubic( arc );
        arc += 3;
      }
      else if ( y1 == y4 )
      {
        /* this arc is flat, advance position */
        /* and pop it from the Bezier stack   */
        arc -= 3;

        ras.lastX = x4;
        ras.lastY = y4;
      }
      else
      {
        state_bez = y1 < y4 ? Ascending_State : Descending_State;

        /* detect a change of direction */
        if ( ras.state != state_bez )
        {
          /* finalize current profile if any */
          if ( ras.state != Unknown_State &&
               End_Profile( RAS_VAR )     )
            goto Fail;

          if ( New_Profile( RAS_VARS state_bez ) )
            goto Fail;
        }

        /* compute intersections */
        if ( state_bez == Ascending_State )
        {
          if ( Bezier_Up( RAS_VARS 3, arc, Split_Cubic,
                                   ras.minY, ras.maxY ) )
            goto Fail;
        }
        else
          if ( Bezier_Down( RAS_VARS 3, arc, Split_Cubic,
                                     ras.minY, ras.maxY ) )
            goto Fail;
        arc -= 3;

        ras.lastX = x4;
        ras.lastY = y4;
      }

    } while ( arc >= arcs );

    return SUCCESS;

  Fail:
    return FAILURE;
  }


#undef  SWAP_
#define SWAP_( x, y )  do                \
                       {                 \
                         Long  swap = x; \
                                         \
                                         \
                         x = y;          \
                         y = swap;       \
                       } while ( 0 )


  /**************************************************************************
   *
   * @Function:
   *   Decompose_Curve
   *
   * @Description:
   *   Scan the outline arrays in order to emit individual segments and
   *   Beziers by calling Line_To() and Bezier_To().  It handles all
   *   weird cases, like when the first point is off the curve, or when
   *   there are simply no `on' points in the contour!
   *
   * @Input:
   *   first ::
   *     The index of the first point in the contour.
   *
   *   last ::
   *     The index of the last point in the contour.
   *
   *   flipped ::
   *     If set, flip the direction of the curve.
   *
   * @Return:
   *   SUCCESS on success, FAILURE on error.
   *
   * @Note:
   *   Unlike FT_Outline_Decompose(), this function handles the scanmode
   *   dropout tags in the individual contours.  Therefore, it cannot be
   *   replaced.
   */
  static Bool
  Decompose_Curve( RAS_ARGS Int  first,
                            Int  last,
                            Int  flipped )
  {
    FT_Vector   v_last;
    FT_Vector   v_control;
    FT_Vector   v_start;

    FT_Vector*  points;
    FT_Vector*  point;
    FT_Vector*  limit;
    FT_Byte*    tags;

    UInt        tag;       /* current point's state           */


    points = ras.outline.points;
    limit  = points + last;

    v_start.x = SCALED( points[first].x );
    v_start.y = SCALED( points[first].y );
    v_last.x  = SCALED( points[last].x );
    v_last.y  = SCALED( points[last].y );

    if ( flipped )
    {
      SWAP_( v_start.x, v_start.y );
      SWAP_( v_last.x, v_last.y );
    }

    v_control = v_start;

    point = points + first;
    tags  = ras.outline.tags + first;

    /* set scan mode if necessary */
    if ( tags[0] & FT_CURVE_TAG_HAS_SCANMODE )
      ras.dropOutControl = (Byte)tags[0] >> 5;

    tag = FT_CURVE_TAG( tags[0] );

    /* A contour cannot start with a cubic control point! */
    if ( tag == FT_CURVE_TAG_CUBIC )
      goto Invalid_Outline;

    /* check first point to determine origin */
    if ( tag == FT_CURVE_TAG_CONIC )
    {
      /* first point is conic control.  Yes, this happens. */
      if ( FT_CURVE_TAG( ras.outline.tags[last] ) == FT_CURVE_TAG_ON )
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

     /* v_last = v_start; */
      }
      point--;
      tags--;
    }

    ras.lastX = v_start.x;
    ras.lastY = v_start.y;

    while ( point < limit )
    {
      point++;
      tags++;

      tag = FT_CURVE_TAG( tags[0] );

      switch ( tag )
      {
      case FT_CURVE_TAG_ON:  /* emit a single line_to */
        {
          Long  x, y;


          x = SCALED( point->x );
          y = SCALED( point->y );
          if ( flipped )
            SWAP_( x, y );

          if ( Line_To( RAS_VARS x, y ) )
            goto Fail;
          continue;
        }

      case FT_CURVE_TAG_CONIC:  /* consume conic arcs */
        v_control.x = SCALED( point[0].x );
        v_control.y = SCALED( point[0].y );

        if ( flipped )
          SWAP_( v_control.x, v_control.y );

      Do_Conic:
        if ( point < limit )
        {
          FT_Vector  v_middle;
          Long       x, y;


          point++;
          tags++;
          tag = FT_CURVE_TAG( tags[0] );

          x = SCALED( point[0].x );
          y = SCALED( point[0].y );

          if ( flipped )
            SWAP_( x, y );

          if ( tag == FT_CURVE_TAG_ON )
          {
            if ( Conic_To( RAS_VARS v_control.x, v_control.y, x, y ) )
              goto Fail;
            continue;
          }

          if ( tag != FT_CURVE_TAG_CONIC )
            goto Invalid_Outline;

          v_middle.x = ( v_control.x + x ) / 2;
          v_middle.y = ( v_control.y + y ) / 2;

          if ( Conic_To( RAS_VARS v_control.x, v_control.y,
                                  v_middle.x,  v_middle.y ) )
            goto Fail;

          v_control.x = x;
          v_control.y = y;

          goto Do_Conic;
        }

        if ( Conic_To( RAS_VARS v_control.x, v_control.y,
                                v_start.x,   v_start.y ) )
          goto Fail;

        goto Close;

      default:  /* FT_CURVE_TAG_CUBIC */
        {
          Long  x1, y1, x2, y2, x3, y3;


          if ( point + 1 > limit                             ||
               FT_CURVE_TAG( tags[1] ) != FT_CURVE_TAG_CUBIC )
            goto Invalid_Outline;

          point += 2;
          tags  += 2;

          x1 = SCALED( point[-2].x );
          y1 = SCALED( point[-2].y );
          x2 = SCALED( point[-1].x );
          y2 = SCALED( point[-1].y );

          if ( flipped )
          {
            SWAP_( x1, y1 );
            SWAP_( x2, y2 );
          }

          if ( point <= limit )
          {
            x3 = SCALED( point[0].x );
            y3 = SCALED( point[0].y );

            if ( flipped )
              SWAP_( x3, y3 );

            if ( Cubic_To( RAS_VARS x1, y1, x2, y2, x3, y3 ) )
              goto Fail;
            continue;
          }

          if ( Cubic_To( RAS_VARS x1, y1, x2, y2, v_start.x, v_start.y ) )
            goto Fail;
          goto Close;
        }
      }
    }

    /* close the contour with a line segment */
    if ( Line_To( RAS_VARS v_start.x, v_start.y ) )
      goto Fail;

  Close:
    return SUCCESS;

  Invalid_Outline:
    ras.error = FT_THROW( Invalid_Outline );

  Fail:
    return FAILURE;
  }


  /**************************************************************************
   *
   * @Function:
   *   Convert_Glyph
   *
   * @Description:
   *   Convert a glyph into a series of segments and arcs and make a
   *   profiles list with them.
   *
   * @Input:
   *   flipped ::
   *     If set, flip the direction of curve.
   *
   * @Return:
   *   SUCCESS on success, FAILURE if any error was encountered during
   *   rendering.
   */
  static Bool
  Convert_Glyph( RAS_ARGS Int  flipped )
  {
    Int  i;
    Int  first, last;


    ras.fProfile = NULL;
    ras.cProfile = NULL;

    ras.top      = ras.buff;
    ras.maxBuff  = ras.sizeBuff - 1;  /* top reserve */

    ras.numTurns  = 0;
    ras.num_Profs = 0;

    last = -1;
    for ( i = 0; i < ras.outline.n_contours; i++ )
    {
      ras.state    = Unknown_State;
      ras.gProfile = NULL;

      first = last + 1;
      last  = ras.outline.contours[i];

      if ( Decompose_Curve( RAS_VARS first, last, flipped ) )
        return FAILURE;

      /* Note that ras.gProfile can stay nil if the contour was */
      /* too small to be drawn or degenerate.                   */
      if ( !ras.gProfile )
        continue;

      /* we must now check whether the extreme arcs join or not */
      if ( FRAC( ras.lastY ) == 0 &&
           ras.lastY >= ras.minY  &&
           ras.lastY <= ras.maxY  )
        if ( ( ras.gProfile->flags & Flow_Up ) ==
               ( ras.cProfile->flags & Flow_Up ) )
          ras.top--;

      if ( End_Profile( RAS_VAR ) )
        return FAILURE;

      if ( !ras.fProfile )
        ras.fProfile = ras.gProfile;
    }

    if ( ras.fProfile )
      Finalize_Profile_Table( RAS_VAR );

    return SUCCESS;
  }


  /*************************************************************************/
  /*************************************************************************/
  /**                                                                     **/
  /**  SCAN-LINE SWEEPS AND DRAWING                                       **/
  /**                                                                     **/
  /*************************************************************************/
  /*************************************************************************/


  /**************************************************************************
   *
   * InsNew
   *
   *   Inserts a new profile in a linked list, sorted by coordinate.
   */
  static void
  InsNew( PProfileList  list,
          PProfile      profile )
  {
    PProfile  *old, current;
    Long       x;


    old     = list;
    current = *old;
    x       = profile->X;

    while ( current && current->X < x )
    {
      old     = &current->link;
      current = *old;
    }

    profile->link = current;
    *old          = profile;
  }


  /**************************************************************************
   *
   * Increment
   *
   *   Advances all profile in the list to the next scanline.  It also
   *   sorts the trace list in the unlikely case of profile crossing.
   *   The profiles are inserted in sorted order.  We might need a single
   *   swap to fix it when profiles (contours) cross.
   *   Bubble sort with immediate restart is good enough and simple.
   */
  static void
  Increment( PProfileList  list,
             Int           flow )
  {
    PProfile  *old, current, next;


    /* First, set the new X coordinates and remove exhausted profiles */
    old = list;
    while ( *old )
    {
      current = *old;
      if ( --current->height )
      {
        current->offset += flow;
        current->X       = current->x[current->offset];
        old = &current->link;
      }
      else
        *old = current->link;  /* remove */
    }

    /* Then make sure the list remains sorted */
    old     = list;
    current = *old;

    if ( !current )
      return;

    while ( current->link )
    {
      next = current->link;

      if ( current->X <= next->X )
      {
        old     = &current->link;
        current = next;
      }
      else
      {
        *old          = next;
        current->link = next->link;
        next->link    = current;

        /* this is likely the only necessary swap -- restart */
        old     = list;
        current = *old;
      }
    }
  }


  /**************************************************************************
   *
   * Vertical Sweep Procedure Set
   *
   * These four routines are used during the vertical black/white sweep
   * phase by the generic Draw_Sweep() function.
   *
   */

  static void
  Vertical_Sweep_Init( RAS_ARGS Int  min,
                                Int  max )
  {
    FT_UNUSED( max );


    ras.bLine = ras.bOrigin - min * ras.bPitch;
  }


  static void
  Vertical_Sweep_Span( RAS_ARGS Int         y,
                                FT_F26Dot6  x1,
                                FT_F26Dot6  x2 )
  {
    Int  e1 = (Int)TRUNC( CEILING( x1 ) );
    Int  e2 = (Int)TRUNC(   FLOOR( x2 ) );

    FT_UNUSED( y );


    FT_TRACE7(( "  y=%d x=[% .*f;% .*f]",
                y,
                ras.precision_bits, (double)x1 / (double)ras.precision,
                ras.precision_bits, (double)x2 / (double)ras.precision ));

    if ( e2 >= 0 && e1 <= ras.bRight )
    {
      PByte  target;

      Int   c1, f1, c2, f2;


      if ( e1 < 0 )
        e1 = 0;
      if ( e2 > ras.bRight )
        e2 = ras.bRight;

      FT_TRACE7(( " -> x=[%d;%d]", e1, e2 ));

      c1 = e1 >> 3;
      c2 = e2 >> 3;

      f1 =  0xFF >> ( e1 & 7 );
      f2 = ~0x7F >> ( e2 & 7 );

      target = ras.bLine + c1;
      c2 -= c1;

      if ( c2 > 0 )
      {
        target[0] |= f1;

        /* memset() is slower than the following code on many platforms. */
        /* This is due to the fact that, in the vast majority of cases,  */
        /* the span length in bytes is relatively small.                 */
        while ( --c2 > 0 )
          *( ++target ) = 0xFF;

        target[1] |= f2;
      }
      else
        *target |= ( f1 & f2 );
    }

    FT_TRACE7(( "\n" ));
  }


  static void
  Vertical_Sweep_Drop( RAS_ARGS Int         y,
                                FT_F26Dot6  x1,
                                FT_F26Dot6  x2 )
  {
    Int  e1 = (Int)TRUNC( x1 );
    Int  e2 = (Int)TRUNC( x2 );
    Int  c1, f1;

    FT_UNUSED( y );


    /* undocumented but confirmed: If the drop-out would result in a  */
    /* pixel outside of the bounding box, use the pixel inside of the */
    /* bounding box instead                                           */
    if ( e1 < 0 || e1 > ras.bRight )
      e1 = e2;

    /* otherwise check that the other pixel isn't set */
    else if ( e2 >=0 && e2 <= ras.bRight )
    {
      c1 = e2 >> 3;
      f1 = 0x80 >> ( e2 & 7 );

      if ( ras.bLine[c1] & f1 )
        return;
    }

    if ( e1 >= 0 && e1 <= ras.bRight )
    {
      c1 = e1 >> 3;
      f1 = 0x80 >> ( e1 & 7 );

      FT_TRACE7(( "  y=%d x=%d%s\n", y, e1,
                  ras.bLine[c1] & f1 ? " redundant" : "" ));

      ras.bLine[c1] |= f1;
    }
  }


  static void
  Vertical_Sweep_Step( RAS_ARG )
  {
    ras.bLine -= ras.bPitch;
  }


  /************************************************************************
   *
   * Horizontal Sweep Procedure Set
   *
   * These four routines are used during the horizontal black/white
   * sweep phase by the generic Draw_Sweep() function.
   *
   */

  static void
  Horizontal_Sweep_Init( RAS_ARGS Int  min,
                                  Int  max )
  {
    /* nothing, really */
    FT_UNUSED_RASTER;
    FT_UNUSED( min );
    FT_UNUSED( max );
  }


  static void
  Horizontal_Sweep_Span( RAS_ARGS Int         y,
                                  FT_F26Dot6  x1,
                                  FT_F26Dot6  x2 )
  {
    Long  e1 = CEILING( x1 );
    Long  e2 =   FLOOR( x2 );


    FT_TRACE7(( "  x=%d y=[% .*f;% .*f]",
                y,
                ras.precision_bits, (double)x1 / (double)ras.precision,
                ras.precision_bits, (double)x2 / (double)ras.precision ));

    /* We should not need this procedure but the vertical sweep   */
    /* mishandles horizontal lines through pixel centers.  So we  */
    /* have to check perfectly aligned span edges here.           */
    /*                                                            */
    /* XXX: Can we handle horizontal lines better and drop this?  */

    if ( x1 == e1 )
    {
      e1 = TRUNC( e1 );

      if ( e1 >= 0 && e1 <= ras.bTop )
      {
        Int    f1;
        PByte  bits;


        bits = ras.bOrigin + ( y >> 3 ) - e1 * ras.bPitch;
        f1   = 0x80 >> ( y & 7 );

        FT_TRACE7(( bits[0] & f1 ? " redundant"
                                 : " -> y=%ld edge", e1 ));

        bits[0] |= f1;
      }
    }

    if ( x2 == e2 )
    {
      e2 = TRUNC( e2 );

      if ( e2 >= 0 && e2 <= ras.bTop )
      {
        Int    f1;
        PByte  bits;


        bits = ras.bOrigin + ( y >> 3 ) - e2 * ras.bPitch;
        f1   = 0x80 >> ( y & 7 );

        FT_TRACE7(( bits[0] & f1 ? " redundant"
                                 : " -> y=%ld edge", e2 ));

        bits[0] |= f1;
      }
    }

    FT_TRACE7(( "\n" ));
  }


  static void
  Horizontal_Sweep_Drop( RAS_ARGS Int         y,
                                  FT_F26Dot6  x1,
                                  FT_F26Dot6  x2 )
  {
    Int    e1 = (Int)TRUNC( x1 );
    Int    e2 = (Int)TRUNC( x2 );
    PByte  bits;
    Int    f1;


    /* undocumented but confirmed: If the drop-out would result in a  */
    /* pixel outside of the bounding box, use the pixel inside of the */
    /* bounding box instead                                           */
    if ( e1 < 0 || e1 > ras.bTop )
      e1 = e2;

    /* otherwise check that the other pixel isn't set */
    else if ( e2 >=0 && e2 <= ras.bTop )
    {
      bits = ras.bOrigin + ( y >> 3 ) - e2 * ras.bPitch;
      f1   = 0x80 >> ( y & 7 );

      if ( *bits & f1 )
        return;
    }

    if ( e1 >= 0 && e1 <= ras.bTop )
    {
      bits  = ras.bOrigin + ( y >> 3 ) - e1 * ras.bPitch;
      f1    = 0x80 >> ( y & 7 );

      FT_TRACE7(( "  x=%d y=%d%s\n", y, e1,
                  *bits & f1 ? " redundant" : "" ));

      *bits |= f1;
    }
  }


  static void
  Horizontal_Sweep_Step( RAS_ARG )
  {
    /* Nothing, really */
    FT_UNUSED_RASTER;
  }


  /**************************************************************************
   *
   * Generic Sweep Drawing routine
   *
   * Note that this routine is executed with the pool containing at least
   * two valid profiles (up and down) and two y-turns (top and bottom).
   *
   */

  static void
  Draw_Sweep( RAS_ARG )
  {
    Int           min_Y, max_Y, dropouts;
    Int           y, y_turn;

    PProfile      *Q, P, P_Left, P_Right;

    TProfileList  waiting    = ras.fProfile;
    TProfileList  draw_left  = NULL;
    TProfileList  draw_right = NULL;


    /* use y_turns to set the drawing range */

    min_Y = (Int)ras.maxBuff[0];
    max_Y = (Int)ras.maxBuff[ras.numTurns] - 1;

    /* now initialize the sweep */

    ras.Proc_Sweep_Init( RAS_VARS min_Y, max_Y );

    /* let's go */

    for ( y = min_Y; y <= max_Y; )
    {
      /* check waiting list for new profile activations */

      Q = &waiting;
      while ( *Q )
      {
        P = *Q;
        if ( P->start == y )
        {
          *Q = P->link;  /* remove */

          /* each active list contains profiles with the same flow */
          /* left and right are arbitrary, correspond to TrueType  */
          if ( P->flags & Flow_Up )
            InsNew( &draw_left,  P );
          else
            InsNew( &draw_right, P );
        }
        else
          Q = &P->link;
      }

      y_turn = (Int)*++ras.maxBuff;

      do
      {
        /* let's trace */

        dropouts = 0;

        P_Left  = draw_left;
        P_Right = draw_right;

        while ( P_Left && P_Right )
        {
          Long  x1 = P_Left ->X;
          Long  x2 = P_Right->X;
          Long  xs;


          /* TrueType should have x2 > x1, but can be opposite */
          /* by mistake or in CFF/Type1, fix it then           */
          if ( x1 > x2 )
          {
            xs = x1;
            x1 = x2;
            x2 = xs;
          }

          if ( CEILING( x1 ) <= FLOOR( x2 ) )
            ras.Proc_Sweep_Span( RAS_VARS y, x1, x2 );

          /* otherwise, bottom ceiling > top floor, it is a drop-out */
          else
          {
            Int  dropOutControl = P_Left->flags & 7;


            /* Drop-out control */

            /*   e2            x2                    x1           e1   */
            /*                                                         */
            /*                 ^                     |                 */
            /*                 |                     |                 */
            /*   +-------------+---------------------+------------+    */
            /*                 |                     |                 */
            /*                 |                     v                 */
            /*                                                         */
            /* pixel         contour              contour       pixel  */
            /* center                                           center */

            /* drop-out mode   scan conversion rules (OpenType specs)  */
            /* ------------------------------------------------------- */
            /*  bit 0          exclude stubs if set                    */
            /*  bit 1          ignore drop-outs if set                 */
            /*  bit 2          smart rounding if set                   */

            if ( dropOutControl & 2 )
              goto Next_Pair;

            /* The specification neither provides an exact definition */
            /* of a `stub' nor gives exact rules to exclude them.     */
            /*                                                        */
            /* Here the constraints we use to recognize a stub.       */
            /*                                                        */
            /*  upper stub:                                           */
            /*                                                        */
            /*   - P_Left and P_Right are in the same contour         */
            /*   - P_Right is the successor of P_Left in that contour */
            /*   - y is the top of P_Left and P_Right                 */
            /*                                                        */
            /*  lower stub:                                           */
            /*                                                        */
            /*   - P_Left and P_Right are in the same contour         */
            /*   - P_Left is the successor of P_Right in that contour */
            /*   - y is the bottom of P_Left                          */
            /*                                                        */
            /* We draw a stub if the following constraints are met.   */
            /*                                                        */
            /*   - for an upper or lower stub, there is top or bottom */
            /*     overshoot, respectively                            */
            /*   - the covered interval is greater or equal to a half */
            /*     pixel                                              */

            if ( dropOutControl & 1 )
            {
              /* upper stub test */
              if ( P_Left->height == 1                &&
                   P_Left->next == P_Right            &&
                   !( P_Left->flags & Overshoot_Top   &&
                      x2 - x1 >= ras.precision_half   ) )
                goto Next_Pair;

              /* lower stub test */
              if ( P_Left->offset == 0                 &&
                   P_Right->next == P_Left             &&
                   !( P_Left->flags & Overshoot_Bottom &&
                      x2 - x1 >= ras.precision_half    ) )
                goto Next_Pair;
            }

            /* select the pixel to set and the other pixel */
            if ( dropOutControl & 4 )
            {
              x2 = SMART( x1, x2 );
              x1 = x1 > x2 ? x2 + ras.precision : x2 - ras.precision;
            }
            else
            {
              x2 = FLOOR  ( x2 );
              x1 = CEILING( x1 );
            }

            P_Left ->X = x2;
            P_Right->X = x1;

            /* mark profile for drop-out processing */
            P_Left->flags |= Dropout;
            dropouts++;
          }

        Next_Pair:
          P_Left  = P_Left->link;
          P_Right = P_Right->link;
        }

        /* handle drop-outs _after_ the span drawing */
        P_Left  = draw_left;
        P_Right = draw_right;

        while ( dropouts )
        {
          if ( P_Left->flags & Dropout )
          {
            ras.Proc_Sweep_Drop( RAS_VARS y, P_Left->X, P_Right->X );

            P_Left->flags &= ~Dropout;
            dropouts--;
          }

          P_Left  = P_Left->link;
          P_Right = P_Right->link;
        }

        ras.Proc_Sweep_Step( RAS_VAR );

        Increment( &draw_left,   1 );
        Increment( &draw_right, -1 );
      }
      while ( ++y < y_turn );
    }
  }


  /**************************************************************************
   *
   * @Function:
   *   Render_Single_Pass
   *
   * @Description:
   *   Perform one sweep with sub-banding.
   *
   * @Input:
   *   flipped ::
   *     If set, flip the direction of the outline.
   *
   * @Return:
   *   Renderer error code.
   */
  static int
  Render_Single_Pass( RAS_ARGS Bool  flipped,
                               Int   y_min,
                               Int   y_max )
  {
    Int  y_mid;
    Int  band_top = 0;
    Int  band_stack[32];  /* enough to bisect 32-bit int bands */


    FT_TRACE6(( "%s pass [%d..%d]\n",
                flipped ? "Horizontal" : "Vertical",
                y_min, y_max ));

    while ( 1 )
    {
      ras.minY = (Long)y_min * ras.precision;
      ras.maxY = (Long)y_max * ras.precision;

      ras.error = Raster_Err_Ok;

      if ( Convert_Glyph( RAS_VARS flipped ) )
      {
        if ( ras.error != Raster_Err_Raster_Overflow )
          return ras.error;

        /* sub-banding */

        if ( y_min == y_max )
          return ras.error;  /* still Raster_Overflow */

        FT_TRACE6(( "band [%d..%d]: to be bisected\n",
                    y_min, y_max ));

        y_mid = ( y_min + y_max ) >> 1;

        band_stack[band_top++] = y_min;
        y_min                  = y_mid + 1;
      }
      else
      {
        FT_TRACE6(( "band [%d..%d]: %hd profiles; %td bytes remaining\n",
                    y_min, y_max, ras.num_Profs,
                    (char*)ras.maxBuff - (char*)ras.top ));

        if ( ras.fProfile )
          Draw_Sweep( RAS_VAR );

        if ( --band_top < 0 )
          break;

        y_max = y_min - 1;
        y_min = band_stack[band_top];
      }
    }

    return Raster_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   Render_Glyph
   *
   * @Description:
   *   Render a glyph in a bitmap.  Sub-banding if needed.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  static FT_Error
  Render_Glyph( RAS_ARG )
  {
    FT_Error  error;
    Long      buffer[FT_MAX_BLACK_POOL];


    ras.buff     = buffer;
    ras.sizeBuff = (&buffer)[1]; /* Points to right after buffer. */

    Set_High_Precision( RAS_VARS ras.outline.flags &
                                 FT_OUTLINE_HIGH_PRECISION );

    ras.dropOutControl = 0;

    if ( ras.outline.flags & FT_OUTLINE_IGNORE_DROPOUTS )
      ras.dropOutControl |= 2;

    if ( ras.outline.flags & FT_OUTLINE_SMART_DROPOUTS )
      ras.dropOutControl |= 4;

    if ( !( ras.outline.flags & FT_OUTLINE_INCLUDE_STUBS ) )
      ras.dropOutControl |= 1;

    FT_TRACE6(( "BW Raster: precision 1/%d, dropout mode %d\n",
                ras.precision, ras.dropOutControl ));

    /* Vertical Sweep */
    ras.Proc_Sweep_Init = Vertical_Sweep_Init;
    ras.Proc_Sweep_Span = Vertical_Sweep_Span;
    ras.Proc_Sweep_Drop = Vertical_Sweep_Drop;
    ras.Proc_Sweep_Step = Vertical_Sweep_Step;

    error = Render_Single_Pass( RAS_VARS 0, 0, ras.bTop );
    if ( error )
      return error;

    /* Horizontal Sweep */
    if ( !( ras.outline.flags & FT_OUTLINE_SINGLE_PASS ) )
    {
      ras.Proc_Sweep_Init = Horizontal_Sweep_Init;
      ras.Proc_Sweep_Span = Horizontal_Sweep_Span;
      ras.Proc_Sweep_Drop = Horizontal_Sweep_Drop;
      ras.Proc_Sweep_Step = Horizontal_Sweep_Step;

      error = Render_Single_Pass( RAS_VARS 1, 0, ras.bRight );
      if ( error )
        return error;
    }

    return Raster_Err_Ok;
  }


  /**** RASTER OBJECT CREATION: In standalone mode, we simply use *****/
  /****                         a static object.                  *****/


#ifdef STANDALONE_


  static int
  ft_black_new( void*       memory,
                FT_Raster  *araster )
  {
     static black_TRaster  the_raster;
     FT_UNUSED( memory );


     *araster = (FT_Raster)&the_raster;
     FT_ZERO( &the_raster );

     return 0;
  }


  static void
  ft_black_done( FT_Raster  raster )
  {
    /* nothing */
    FT_UNUSED( raster );
  }


#else /* !STANDALONE_ */


  static int
  ft_black_new( void*       memory_,    /* FT_Memory     */
                FT_Raster  *araster_ )  /* black_PRaster */
  {
    FT_Memory       memory = (FT_Memory)memory_;
    black_PRaster  *araster = (black_PRaster*)araster_;

    FT_Error       error;
    black_PRaster  raster = NULL;


    if ( !FT_NEW( raster ) )
      raster->memory = memory;

    *araster = raster;

    return error;
  }


  static void
  ft_black_done( FT_Raster  raster_ )   /* black_PRaster */
  {
    black_PRaster  raster = (black_PRaster)raster_;
    FT_Memory      memory = (FT_Memory)raster->memory;


    FT_FREE( raster );
  }


#endif /* !STANDALONE_ */


  static void
  ft_black_reset( FT_Raster  raster,
                  PByte      pool_base,
                  ULong      pool_size )
  {
    FT_UNUSED( raster );
    FT_UNUSED( pool_base );
    FT_UNUSED( pool_size );
  }


  static int
  ft_black_set_mode( FT_Raster  raster,
                     ULong      mode,
                     void*      args )
  {
    FT_UNUSED( raster );
    FT_UNUSED( mode );
    FT_UNUSED( args );

    return 0;
  }


  static int
  ft_black_render( FT_Raster                raster,
                   const FT_Raster_Params*  params )
  {
    const FT_Outline*  outline    = (const FT_Outline*)params->source;
    const FT_Bitmap*   target_map = params->target;

#ifndef FT_STATIC_RASTER
    black_TWorker  worker[1];
#endif


    if ( !raster )
      return FT_THROW( Raster_Uninitialized );

    if ( !outline )
      return FT_THROW( Invalid_Outline );

    /* return immediately if the outline is empty */
    if ( outline->n_points == 0 || outline->n_contours == 0 )
      return Raster_Err_Ok;

    if ( !outline->contours || !outline->points )
      return FT_THROW( Invalid_Outline );

    if ( outline->n_points !=
           outline->contours[outline->n_contours - 1] + 1 )
      return FT_THROW( Invalid_Outline );

    /* this version of the raster does not support direct rendering, sorry */
    if ( params->flags & FT_RASTER_FLAG_DIRECT ||
         params->flags & FT_RASTER_FLAG_AA     )
      return FT_THROW( Cannot_Render_Glyph );

    if ( !target_map )
      return FT_THROW( Invalid_Argument );

    /* nothing to do */
    if ( !target_map->width || !target_map->rows )
      return Raster_Err_Ok;

    if ( !target_map->buffer )
      return FT_THROW( Invalid_Argument );

    ras.outline = *outline;

    ras.bTop    =   (Int)target_map->rows - 1;
    ras.bRight  =   (Int)target_map->width - 1;
    ras.bPitch  =   (Int)target_map->pitch;
    ras.bOrigin = (PByte)target_map->buffer;

    if ( ras.bPitch > 0 )
      ras.bOrigin += ras.bTop * ras.bPitch;

    return Render_Glyph( RAS_VAR );
  }


  FT_DEFINE_RASTER_FUNCS(
    ft_standard_raster,

    FT_GLYPH_FORMAT_OUTLINE,

    ft_black_new,       /* FT_Raster_New_Func      raster_new      */
    ft_black_reset,     /* FT_Raster_Reset_Func    raster_reset    */
    ft_black_set_mode,  /* FT_Raster_Set_Mode_Func raster_set_mode */
    ft_black_render,    /* FT_Raster_Render_Func   raster_render   */
    ft_black_done       /* FT_Raster_Done_Func     raster_done     */
  )


/* END */

/***************************************************************************/
/*                                                                         */
/*  ftdebug.h                                                              */
/*                                                                         */
/*    Debugging and logging component (specification).                     */
/*                                                                         */
/*  Copyright 1996-2016 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/*                                                                         */
/*  IMPORTANT: A description of FreeType's debugging support can be        */
/*             found in `docs/DEBUG.TXT'.  Read it if you need to use or   */
/*             understand this code.                                       */
/*                                                                         */
/***************************************************************************/


#ifndef FTDEBUG_H_
#define FTDEBUG_H_


#include <ft2build.h>
#include FT_CONFIG_CONFIG_H
#include FT_FREETYPE_H


FT_BEGIN_HEADER


  /* force the definition of FT_DEBUG_LEVEL_ERROR if FT_DEBUG_LEVEL_TRACE */
  /* is already defined; this simplifies the following #ifdefs            */
  /*                                                                      */
#ifdef FT_DEBUG_LEVEL_TRACE
#undef  FT_DEBUG_LEVEL_ERROR
#define FT_DEBUG_LEVEL_ERROR
#endif


  /*************************************************************************/
  /*                                                                       */
  /* Define the trace enums as well as the trace levels array when they    */
  /* are needed.                                                           */
  /*                                                                       */
  /*************************************************************************/

#ifdef FT_DEBUG_LEVEL_TRACE

#define FT_TRACE_DEF( x )  trace_ ## x ,

  /* defining the enumeration */
  typedef enum  FT_Trace_
  {
#include FT_INTERNAL_TRACE_H
    trace_count

  } FT_Trace;


  /* defining the array of trace levels, provided by `src/base/ftdebug.c' */
  extern int  ft_trace_levels[trace_count];

#undef FT_TRACE_DEF

#endif /* FT_DEBUG_LEVEL_TRACE */


  /*************************************************************************/
  /*                                                                       */
  /* Define the FT_TRACE macro                                             */
  /*                                                                       */
  /* IMPORTANT!                                                            */
  /*                                                                       */
  /* Each component must define the macro FT_COMPONENT to a valid FT_Trace */
  /* value before using any TRACE macro.                                   */
  /*                                                                       */
  /*************************************************************************/

#ifdef FT_DEBUG_LEVEL_TRACE

#define FT_TRACE( level, varformat )                      \
          do                                              \
          {                                               \
            if ( ft_trace_levels[FT_COMPONENT] >= level ) \
              FT_Message varformat;                       \
          } while ( 0 )

#else /* !FT_DEBUG_LEVEL_TRACE */

#define FT_TRACE( level, varformat )  do { } while ( 0 )      /* nothing */

#endif /* !FT_DEBUG_LEVEL_TRACE */


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Trace_Get_Count                                                 */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Return the number of available trace components.                   */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The number of trace components.  0 if FreeType 2 is not built with */
  /*    FT_DEBUG_LEVEL_TRACE definition.                                   */
  /*                                                                       */
  /* <Note>                                                                */
  /*    This function may be useful if you want to access elements of      */
  /*    the internal `ft_trace_levels' array by an index.                  */
  /*                                                                       */
  FT_BASE( FT_Int )
  FT_Trace_Get_Count( void );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Trace_Get_Name                                                  */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Return the name of a trace component.                              */
  /*                                                                       */
  /* <Input>                                                               */
  /*    The index of the trace component.                                  */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The name of the trace component.  This is a statically allocated   */
  /*    C string, so do not free it after use.  NULL if FreeType 2 is not  */
  /*    built with FT_DEBUG_LEVEL_TRACE definition.                        */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Use @FT_Trace_Get_Count to get the number of available trace       */
  /*    components.                                                        */
  /*                                                                       */
  /*    This function may be useful if you want to control FreeType 2's    */
  /*    debug level in your application.                                   */
  /*                                                                       */
  FT_BASE( const char* )
  FT_Trace_Get_Name( FT_Int  idx );


  /*************************************************************************/
  /*                                                                       */
  /* You need two opening and closing parentheses!                         */
  /*                                                                       */
  /* Example: FT_TRACE0(( "Value is %i", foo ))                            */
  /*                                                                       */
  /* Output of the FT_TRACEX macros is sent to stderr.                     */
  /*                                                                       */
  /*************************************************************************/

#define FT_TRACE0( varformat )  FT_TRACE( 0, varformat )
#define FT_TRACE1( varformat )  FT_TRACE( 1, varformat )
#define FT_TRACE2( varformat )  FT_TRACE( 2, varformat )
#define FT_TRACE3( varformat )  FT_TRACE( 3, varformat )
#define FT_TRACE4( varformat )  FT_TRACE( 4, varformat )
#define FT_TRACE5( varformat )  FT_TRACE( 5, varformat )
#define FT_TRACE6( varformat )  FT_TRACE( 6, varformat )
#define FT_TRACE7( varformat )  FT_TRACE( 7, varformat )


  /*************************************************************************/
  /*                                                                       */
  /* Define the FT_ERROR macro.                                            */
  /*                                                                       */
  /* Output of this macro is sent to stderr.                               */
  /*                                                                       */
  /*************************************************************************/

#ifdef FT_DEBUG_LEVEL_ERROR

#define FT_ERROR( varformat )  FT_Message  varformat

#else  /* !FT_DEBUG_LEVEL_ERROR */

#define FT_ERROR( varformat )  do { } while ( 0 )      /* nothing */

#endif /* !FT_DEBUG_LEVEL_ERROR */


  /*************************************************************************/
  /*                                                                       */
  /* Define the FT_ASSERT and FT_THROW macros.  The call to `FT_Throw'     */
  /* makes it possible to easily set a breakpoint at this function.        */
  /*                                                                       */
  /*************************************************************************/

#ifdef FT_DEBUG_LEVEL_ERROR

#define FT_ASSERT( condition )                                      \
          do                                                        \
          {                                                         \
            if ( !( condition ) )                                   \
              FT_Panic( "assertion failed on line %d of file %s\n", \
                        __LINE__, __FILE__ );                       \
          } while ( 0 )

#define FT_THROW( e )                                   \
          ( FT_Throw( FT_ERR_CAT( FT_ERR_PREFIX, e ),   \
                      __LINE__,                         \
                      __FILE__ )                      | \
            FT_ERR_CAT( FT_ERR_PREFIX, e )            )

#else /* !FT_DEBUG_LEVEL_ERROR */

#define FT_ASSERT( condition )  do { } while ( 0 )

#define FT_THROW( e )  FT_ERR_CAT( FT_ERR_PREFIX, e )

#endif /* !FT_DEBUG_LEVEL_ERROR */


  /*************************************************************************/
  /*                                                                       */
  /* Define `FT_Message' and `FT_Panic' when needed.                       */
  /*                                                                       */
  /*************************************************************************/

#ifdef FT_DEBUG_LEVEL_ERROR

#include "stdio.h"  /* for vfprintf() */

  /* print a message */
  FT_BASE( void )
  FT_Message( const char*  fmt,
              ... );

  /* print a message and exit */
  FT_BASE( void )
  FT_Panic( const char*  fmt,
            ... );

  /* report file name and line number of an error */
  FT_BASE( int )
  FT_Throw( FT_Error     error,
            int          line,
            const char*  file );

#endif /* FT_DEBUG_LEVEL_ERROR */


  FT_BASE( void )
  ft_debug_init( void );

FT_END_HEADER

#endif /* FTDEBUG_H_ */


/* END */

/****************************************************************************
 *
 * ftdebug.h
 *
 *   Debugging and logging component (specification).
 *
 * Copyright (C) 1996-2019 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 *
 * IMPORTANT: A description of FreeType's debugging support can be
 *             found in 'docs/DEBUG.TXT'.  Read it if you need to use or
 *             understand this code.
 *
 */


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


  /**************************************************************************
   *
   * Define the trace enums as well as the trace levels array when they are
   * needed.
   *
   */

#ifdef FT_DEBUG_LEVEL_TRACE

#define FT_TRACE_DEF( x )  trace_ ## x ,

  /* defining the enumeration */
  typedef enum  FT_Trace_
  {
#include FT_INTERNAL_TRACE_H
    trace_count

  } FT_Trace;


  /* a pointer to the array of trace levels, */
  /* provided by `src/base/ftdebug.c'        */
  extern int*  ft_trace_levels;

#undef FT_TRACE_DEF

#endif /* FT_DEBUG_LEVEL_TRACE */


  /**************************************************************************
   *
   * Define the FT_TRACE macro
   *
   * IMPORTANT!
   *
   * Each component must define the macro FT_COMPONENT to a valid FT_Trace
   * value before using any TRACE macro.
   *
   */

#ifdef FT_DEBUG_LEVEL_TRACE

  /* we need two macros here to make cpp expand `FT_COMPONENT' */
#define FT_TRACE_COMP( x )   FT_TRACE_COMP_( x )
#define FT_TRACE_COMP_( x )  trace_ ## x

#define FT_TRACE( level, varformat )                                       \
          do                                                               \
          {                                                                \
            if ( ft_trace_levels[FT_TRACE_COMP( FT_COMPONENT )] >= level ) \
              FT_Message varformat;                                        \
          } while ( 0 )

#else /* !FT_DEBUG_LEVEL_TRACE */

#define FT_TRACE( level, varformat )  do { } while ( 0 )      /* nothing */

#endif /* !FT_DEBUG_LEVEL_TRACE */


  /**************************************************************************
   *
   * @function:
   *   FT_Trace_Get_Count
   *
   * @description:
   *   Return the number of available trace components.
   *
   * @return:
   *   The number of trace components.  0 if FreeType 2 is not built with
   *   FT_DEBUG_LEVEL_TRACE definition.
   *
   * @note:
   *   This function may be useful if you want to access elements of the
   *   internal trace levels array by an index.
   */
  FT_BASE( FT_Int )
  FT_Trace_Get_Count( void );


  /**************************************************************************
   *
   * @function:
   *   FT_Trace_Get_Name
   *
   * @description:
   *   Return the name of a trace component.
   *
   * @input:
   *   The index of the trace component.
   *
   * @return:
   *   The name of the trace component.  This is a statically allocated
   *   C~string, so do not free it after use.  `NULL` if FreeType is not
   *   built with FT_DEBUG_LEVEL_TRACE definition.
   *
   * @note:
   *   Use @FT_Trace_Get_Count to get the number of available trace
   *   components.
   */
  FT_BASE( const char* )
  FT_Trace_Get_Name( FT_Int  idx );


  /**************************************************************************
   *
   * @function:
   *   FT_Trace_Disable
   *
   * @description:
   *   Switch off tracing temporarily.  It can be activated again with
   *   @FT_Trace_Enable.
   */
  FT_BASE( void )
  FT_Trace_Disable( void );


  /**************************************************************************
   *
   * @function:
   *   FT_Trace_Enable
   *
   * @description:
   *   Activate tracing.  Use it after tracing has been switched off with
   *   @FT_Trace_Disable.
   */
  FT_BASE( void )
  FT_Trace_Enable( void );


  /**************************************************************************
   *
   * You need two opening and closing parentheses!
   *
   * Example: FT_TRACE0(( "Value is %i", foo ))
   *
   * Output of the FT_TRACEX macros is sent to stderr.
   *
   */

#define FT_TRACE0( varformat )  FT_TRACE( 0, varformat )
#define FT_TRACE1( varformat )  FT_TRACE( 1, varformat )
#define FT_TRACE2( varformat )  FT_TRACE( 2, varformat )
#define FT_TRACE3( varformat )  FT_TRACE( 3, varformat )
#define FT_TRACE4( varformat )  FT_TRACE( 4, varformat )
#define FT_TRACE5( varformat )  FT_TRACE( 5, varformat )
#define FT_TRACE6( varformat )  FT_TRACE( 6, varformat )
#define FT_TRACE7( varformat )  FT_TRACE( 7, varformat )


  /**************************************************************************
   *
   * Define the FT_ERROR macro.
   *
   * Output of this macro is sent to stderr.
   *
   */

#ifdef FT_DEBUG_LEVEL_ERROR

#define FT_ERROR( varformat )  FT_Message  varformat

#else  /* !FT_DEBUG_LEVEL_ERROR */

#define FT_ERROR( varformat )  do { } while ( 0 )      /* nothing */

#endif /* !FT_DEBUG_LEVEL_ERROR */


  /**************************************************************************
   *
   * Define the FT_ASSERT and FT_THROW macros.  The call to `FT_Throw` makes
   * it possible to easily set a breakpoint at this function.
   *
   */

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


  /**************************************************************************
   *
   * Define `FT_Message` and `FT_Panic` when needed.
   *
   */

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

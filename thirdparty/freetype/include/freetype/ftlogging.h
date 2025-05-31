/****************************************************************************
 *
 * ftlogging.h
 *
 *   Additional debugging APIs.
 *
 * Copyright (C) 2020-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef FTLOGGING_H_
#define FTLOGGING_H_


#include <ft2build.h>
#include FT_CONFIG_CONFIG_H


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @section:
   *   debugging_apis
   *
   * @title:
   *   External Debugging APIs
   *
   * @abstract:
   *   Public APIs to control the `FT_DEBUG_LOGGING` macro.
   *
   * @description:
   *   This section contains the declarations of public functions that
   *   enables fine control of what the `FT_DEBUG_LOGGING` macro outputs.
   *
   */


  /**************************************************************************
   *
   * @function:
   *   FT_Trace_Set_Level
   *
   * @description:
   *   Change the levels of tracing components of FreeType at run time.
   *
   * @input:
   *   tracing_level ::
   *     New tracing value.
   *
   * @example:
   *   The following call makes FreeType trace everything but the 'memory'
   *   component.
   *
   *   ```
   *   FT_Trace_Set_Level( "any:7 memory:0" );
   *   ```
   *
   * @note:
   *   This function does nothing if compilation option `FT_DEBUG_LOGGING`
   *   isn't set.
   *
   * @since:
   *   2.11
   *
   */
  FT_EXPORT( void )
  FT_Trace_Set_Level( const char*  tracing_level );


  /**************************************************************************
   *
   * @function:
   *   FT_Trace_Set_Default_Level
   *
   * @description:
   *   Reset tracing value of FreeType's components to the default value
   *   (i.e., to the value of the `FT2_DEBUG` environment value or to NULL
   *   if `FT2_DEBUG` is not set).
   *
   * @note:
   *   This function does nothing if compilation option `FT_DEBUG_LOGGING`
   *   isn't set.
   *
   * @since:
   *   2.11
   *
   */
  FT_EXPORT( void )
  FT_Trace_Set_Default_Level( void );


  /**************************************************************************
   *
   * @functype:
   *   FT_Custom_Log_Handler
   *
   * @description:
   *   A function typedef that is used to handle the logging of tracing and
   *   debug messages on a file system.
   *
   * @input:
   *   ft_component ::
   *     The name of `FT_COMPONENT` from which the current debug or error
   *     message is produced.
   *
   *   fmt ::
   *     Actual debug or tracing message.
   *
   *   args::
   *     Arguments of debug or tracing messages.
   *
   * @since:
   *   2.11
   *
   */
  typedef void
  (*FT_Custom_Log_Handler)( const char*  ft_component,
                            const char*  fmt,
                            va_list      args );


  /**************************************************************************
   *
   * @function:
   *   FT_Set_Log_Handler
   *
   * @description:
   *   A function to set a custom log handler.
   *
   * @input:
   *   handler ::
   *     New logging function.
   *
   * @note:
   *   This function does nothing if compilation option `FT_DEBUG_LOGGING`
   *   isn't set.
   *
   * @since:
   *   2.11
   *
   */
  FT_EXPORT( void )
  FT_Set_Log_Handler( FT_Custom_Log_Handler  handler );


  /**************************************************************************
   *
   * @function:
   *   FT_Set_Default_Log_Handler
   *
   * @description:
   *   A function to undo the effect of @FT_Set_Log_Handler, resetting the
   *   log handler to FreeType's built-in version.
   *
   * @note:
   *   This function does nothing if compilation option `FT_DEBUG_LOGGING`
   *   isn't set.
   *
   * @since:
   *   2.11
   *
   */
  FT_EXPORT( void )
  FT_Set_Default_Log_Handler( void );

  /* */


FT_END_HEADER

#endif /* FTLOGGING_H_ */


/* END */

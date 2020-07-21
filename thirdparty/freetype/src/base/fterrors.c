/****************************************************************************
 *
 * fterrors.c
 *
 *   FreeType API for error code handling.
 *
 * Copyright (C) 2018-2020 by
 * Armin Hasitzka, David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H
#include FT_ERRORS_H


  /* documentation is in fterrors.h */

  FT_EXPORT_DEF( const char* )
  FT_Error_String( FT_Error  error_code )
  {
    if ( error_code <  0                                ||
         error_code >= FT_ERR_CAT( FT_ERR_PREFIX, Max ) )
      return NULL;

#if defined( FT_CONFIG_OPTION_ERROR_STRINGS ) || \
    defined( FT_DEBUG_LEVEL_ERROR )

#undef FTERRORS_H_
#define FT_ERROR_START_LIST     switch ( FT_ERROR_BASE( error_code ) ) {
#define FT_ERRORDEF( e, v, s )    case v: return s;
#define FT_ERROR_END_LIST       }

#include FT_ERRORS_H

#endif /* defined( FT_CONFIG_OPTION_ERROR_STRINGS ) || ... */

    return NULL;
  }

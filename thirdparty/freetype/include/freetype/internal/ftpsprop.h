/***************************************************************************/
/*                                                                         */
/*  ftpsprop.h                                                             */
/*                                                                         */
/*    Get and set properties of PostScript drivers (specification).        */
/*                                                                         */
/*  Copyright 2017-2018 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef FTPSPROP_H_
#define FTPSPROP_H_


#include <ft2build.h>
#include FT_FREETYPE_H


FT_BEGIN_HEADER


  FT_BASE_CALLBACK( FT_Error )
  ps_property_set( FT_Module    module,         /* PS_Driver */
                   const char*  property_name,
                   const void*  value,
                   FT_Bool      value_is_string );

  FT_BASE_CALLBACK( FT_Error )
  ps_property_get( FT_Module    module,         /* PS_Driver */
                   const char*  property_name,
                   void*        value );


FT_END_HEADER


#endif /* FTPSPROP_H_ */


/* END */

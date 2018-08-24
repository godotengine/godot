/***************************************************************************/
/*                                                                         */
/*  svotval.h                                                              */
/*                                                                         */
/*    The FreeType OpenType validation service (specification).            */
/*                                                                         */
/*  Copyright 2004-2018 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef SVOTVAL_H_
#define SVOTVAL_H_

#include FT_OPENTYPE_VALIDATE_H
#include FT_INTERNAL_VALIDATE_H

FT_BEGIN_HEADER


#define FT_SERVICE_ID_OPENTYPE_VALIDATE  "opentype-validate"


  typedef FT_Error
  (*otv_validate_func)( FT_Face volatile  face,
                        FT_UInt           ot_flags,
                        FT_Bytes         *base,
                        FT_Bytes         *gdef,
                        FT_Bytes         *gpos,
                        FT_Bytes         *gsub,
                        FT_Bytes         *jstf );


  FT_DEFINE_SERVICE( OTvalidate )
  {
    otv_validate_func  validate;
  };

  /* */


FT_END_HEADER


#endif /* SVOTVAL_H_ */


/* END */

/***************************************************************************/
/*                                                                         */
/*  svtteng.h                                                              */
/*                                                                         */
/*    The FreeType TrueType engine query service (specification).          */
/*                                                                         */
/*  Copyright 2006-2018 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef SVTTENG_H_
#define SVTTENG_H_

#include FT_INTERNAL_SERVICE_H
#include FT_MODULE_H


FT_BEGIN_HEADER


  /*
   *  SFNT table loading service.
   */

#define FT_SERVICE_ID_TRUETYPE_ENGINE  "truetype-engine"

  /*
   * Used to implement FT_Get_TrueType_Engine_Type
   */

  FT_DEFINE_SERVICE( TrueTypeEngine )
  {
    FT_TrueTypeEngineType  engine_type;
  };

  /* */


FT_END_HEADER


#endif /* SVTTENG_H_ */


/* END */

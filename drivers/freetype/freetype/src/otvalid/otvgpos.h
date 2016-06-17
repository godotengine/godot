/***************************************************************************/
/*                                                                         */
/*  otvgpos.h                                                              */
/*                                                                         */
/*    OpenType GPOS table validator (specification).                       */
/*                                                                         */
/*  Copyright 2004 by                                                      */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __OTVGPOS_H__
#define __OTVGPOS_H__


FT_BEGIN_HEADER


  FT_LOCAL( void )
  otv_GPOS_subtable_validate( FT_Bytes       table,
                              OTV_Validator  valid );


FT_END_HEADER

#endif /* __OTVGPOS_H__ */


/* END */

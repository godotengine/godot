/***************************************************************************/
/*                                                                         */
/*  psconv.h                                                               */
/*                                                                         */
/*    Some convenience conversions (specification).                        */
/*                                                                         */
/*  Copyright 2006, 2012 by                                                */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __PSCONV_H__
#define __PSCONV_H__


#include <ft2build.h>
#include FT_INTERNAL_POSTSCRIPT_AUX_H

FT_BEGIN_HEADER


  FT_LOCAL( FT_Long )
  PS_Conv_Strtol( FT_Byte**  cursor,
                  FT_Byte*   limit,
                  FT_Long    base );


  FT_LOCAL( FT_Long )
  PS_Conv_ToInt( FT_Byte**  cursor,
                 FT_Byte*   limit );

  FT_LOCAL( FT_Fixed )
  PS_Conv_ToFixed( FT_Byte**  cursor,
                   FT_Byte*   limit,
                   FT_Long    power_ten );

#if 0
  FT_LOCAL( FT_UInt )
  PS_Conv_StringDecode( FT_Byte**  cursor,
                        FT_Byte*   limit,
                        FT_Byte*   buffer,
                        FT_Offset  n );
#endif

  FT_LOCAL( FT_UInt )
  PS_Conv_ASCIIHexDecode( FT_Byte**  cursor,
                          FT_Byte*   limit,
                          FT_Byte*   buffer,
                          FT_Offset  n );

  FT_LOCAL( FT_UInt )
  PS_Conv_EexecDecode( FT_Byte**   cursor,
                       FT_Byte*    limit,
                       FT_Byte*    buffer,
                       FT_Offset   n,
                       FT_UShort*  seed );


FT_END_HEADER

#endif /* __PSCONV_H__ */


/* END */

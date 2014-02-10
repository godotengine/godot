/***************************************************************************/
/*                                                                         */
/*  cf2ft.h                                                                */
/*                                                                         */
/*    FreeType Glue Component to Adobe's Interpreter (specification).      */
/*                                                                         */
/*  Copyright 2013 Adobe Systems Incorporated.                             */
/*                                                                         */
/*  This software, and all works of authorship, whether in source or       */
/*  object code form as indicated by the copyright notice(s) included      */
/*  herein (collectively, the "Work") is made available, and may only be   */
/*  used, modified, and distributed under the FreeType Project License,    */
/*  LICENSE.TXT.  Additionally, subject to the terms and conditions of the */
/*  FreeType Project License, each contributor to the Work hereby grants   */
/*  to any individual or legal entity exercising permissions granted by    */
/*  the FreeType Project License and this section (hereafter, "You" or     */
/*  "Your") a perpetual, worldwide, non-exclusive, no-charge,              */
/*  royalty-free, irrevocable (except as stated in this section) patent    */
/*  license to make, have made, use, offer to sell, sell, import, and      */
/*  otherwise transfer the Work, where such license applies only to those  */
/*  patent claims licensable by such contributor that are necessarily      */
/*  infringed by their contribution(s) alone or by combination of their    */
/*  contribution(s) with the Work to which such contribution(s) was        */
/*  submitted.  If You institute patent litigation against any entity      */
/*  (including a cross-claim or counterclaim in a lawsuit) alleging that   */
/*  the Work or a contribution incorporated within the Work constitutes    */
/*  direct or contributory patent infringement, then any patent licenses   */
/*  granted to You under this License for that Work shall terminate as of  */
/*  the date such litigation is filed.                                     */
/*                                                                         */
/*  By using, modifying, or distributing the Work you indicate that you    */
/*  have read and understood the terms and conditions of the               */
/*  FreeType Project License as well as those provided in this section,    */
/*  and you accept them fully.                                             */
/*                                                                         */
/***************************************************************************/


#ifndef __CF2FT_H__
#define __CF2FT_H__


#include "cf2types.h"


  /* TODO: disable asserts for now */
#define CF2_NDEBUG


#include FT_SYSTEM_H

#include "cf2glue.h"
#include "cffgload.h"    /* for CFF_Decoder */


FT_BEGIN_HEADER


  FT_LOCAL( FT_Error )
  cf2_decoder_parse_charstrings( CFF_Decoder*  decoder,
                                 FT_Byte*      charstring_base,
                                 FT_ULong      charstring_len );

  FT_LOCAL( CFF_SubFont )
  cf2_getSubfont( CFF_Decoder*  decoder );


  FT_LOCAL( CF2_Fixed )
  cf2_getPpemY( CFF_Decoder*  decoder );
  FT_LOCAL( CF2_Fixed )
  cf2_getStdVW( CFF_Decoder*  decoder );
  FT_LOCAL( CF2_Fixed )
  cf2_getStdHW( CFF_Decoder*  decoder );

  FT_LOCAL( void )
  cf2_getBlueMetrics( CFF_Decoder*  decoder,
                      CF2_Fixed*    blueScale,
                      CF2_Fixed*    blueShift,
                      CF2_Fixed*    blueFuzz );
  FT_LOCAL( void )
  cf2_getBlueValues( CFF_Decoder*  decoder,
                     size_t*       count,
                     FT_Pos*      *data );
  FT_LOCAL( void )
  cf2_getOtherBlues( CFF_Decoder*  decoder,
                     size_t*       count,
                     FT_Pos*      *data );
  FT_LOCAL( void )
  cf2_getFamilyBlues( CFF_Decoder*  decoder,
                      size_t*       count,
                      FT_Pos*      *data );
  FT_LOCAL( void )
  cf2_getFamilyOtherBlues( CFF_Decoder*  decoder,
                           size_t*       count,
                           FT_Pos*      *data );

  FT_LOCAL( CF2_Int )
  cf2_getLanguageGroup( CFF_Decoder*  decoder );

  FT_LOCAL( CF2_Int )
  cf2_initGlobalRegionBuffer( CFF_Decoder*  decoder,
                              CF2_UInt      idx,
                              CF2_Buffer    buf );
  FT_LOCAL( FT_Error )
  cf2_getSeacComponent( CFF_Decoder*  decoder,
                        CF2_UInt      code,
                        CF2_Buffer    buf );
  FT_LOCAL( void )
  cf2_freeSeacComponent( CFF_Decoder*  decoder,
                         CF2_Buffer    buf );
  FT_LOCAL( CF2_Int )
  cf2_initLocalRegionBuffer( CFF_Decoder*  decoder,
                             CF2_UInt      idx,
                             CF2_Buffer    buf );

  FT_LOCAL( CF2_Fixed )
  cf2_getDefaultWidthX( CFF_Decoder*  decoder );
  FT_LOCAL( CF2_Fixed )
  cf2_getNominalWidthX( CFF_Decoder*  decoder );


  /*
   * FreeType client outline
   *
   * process output from the charstring interpreter
   */
  typedef struct  CF2_OutlineRec_
  {
    CF2_OutlineCallbacksRec  root;        /* base class must be first */
    CFF_Decoder*             decoder;

  } CF2_OutlineRec, *CF2_Outline;


  FT_LOCAL( void )
  cf2_outline_reset( CF2_Outline  outline );
  FT_LOCAL( void )
  cf2_outline_close( CF2_Outline  outline );


FT_END_HEADER


#endif /* __CF2FT_H__ */


/* END */

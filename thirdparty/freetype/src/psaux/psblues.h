/***************************************************************************/
/*                                                                         */
/*  psblues.h                                                              */
/*                                                                         */
/*    Adobe's code for handling Blue Zones (specification).                */
/*                                                                         */
/*  Copyright 2009-2013 Adobe Systems Incorporated.                        */
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


  /*
   * A `CF2_Blues' object stores the blue zones (horizontal alignment
   * zones) of a font.  These are specified in the CFF private dictionary
   * by `BlueValues', `OtherBlues', `FamilyBlues', and `FamilyOtherBlues'.
   * Each zone is defined by a top and bottom edge in character space.
   * Further, each zone is either a top zone or a bottom zone, as recorded
   * by `bottomZone'.
   *
   * The maximum number of `BlueValues' and `FamilyBlues' is 7 each.
   * However, these are combined to produce a total of 7 zones.
   * Similarly, the maximum number of `OtherBlues' and `FamilyOtherBlues'
   * is 5 and these are combined to produce an additional 5 zones.
   *
   * Blue zones are used to `capture' hints and force them to a common
   * alignment point.  This alignment is recorded in device space in
   * `dsFlatEdge'.  Except for this value, a `CF2_Blues' object could be
   * constructed independently of scaling.  Construction may occur once
   * the matrix is known.  Other features implemented in the Capture
   * method are overshoot suppression, overshoot enforcement, and Blue
   * Boost.
   *
   * Capture is determined by `BlueValues' and `OtherBlues', but the
   * alignment point may be adjusted to the scaled flat edge of
   * `FamilyBlues' or `FamilyOtherBlues'.  No alignment is done to the
   * curved edge of a zone.
   *
   */


#ifndef PSBLUES_H_
#define PSBLUES_H_


#include "psglue.h"


FT_BEGIN_HEADER


  /*
   * `CF2_Hint' is shared by `cf2hints.h' and
   * `cf2blues.h', but `cf2blues.h' depends on
   * `cf2hints.h', so define it here.  Note: The typedef is in
   * `cf2glue.h'.
   *
   */
  enum
  {
    CF2_GhostBottom = 0x1,  /* a single bottom edge           */
    CF2_GhostTop    = 0x2,  /* a single top edge              */
    CF2_PairBottom  = 0x4,  /* the bottom edge of a stem hint */
    CF2_PairTop     = 0x8,  /* the top edge of a stem hint    */
    CF2_Locked      = 0x10, /* this edge has been aligned     */
                            /* by a blue zone                 */
    CF2_Synthetic   = 0x20  /* this edge was synthesized      */
  };


  /*
   * Default value for OS/2 typoAscender/Descender when their difference
   * is not equal to `unitsPerEm'.  The default is based on -250 and 1100
   * in `CF2_Blues', assuming 1000 units per em here.
   *
   */
  enum
  {
    CF2_ICF_Top    = cf2_intToFixed(  880 ),
    CF2_ICF_Bottom = cf2_intToFixed( -120 )
  };


  /*
   * Constant used for hint adjustment and for synthetic em box hint
   * placement.
   */
#define CF2_MIN_COUNTER  cf2_doubleToFixed( 0.5 )


  /* shared typedef is in cf2glue.h */
  struct  CF2_HintRec_
  {
    CF2_UInt  flags;  /* attributes of the edge            */
    size_t    index;  /* index in original stem hint array */
                      /* (if not synthetic)                */
    CF2_Fixed  csCoord;
    CF2_Fixed  dsCoord;
    CF2_Fixed  scale;
  };


  typedef struct  CF2_BlueRec_
  {
    CF2_Fixed  csBottomEdge;
    CF2_Fixed  csTopEdge;
    CF2_Fixed  csFlatEdge; /* may be from either local or Family zones */
    CF2_Fixed  dsFlatEdge; /* top edge of bottom zone or bottom edge   */
                           /* of top zone (rounded)                    */
    FT_Bool  bottomZone;

  } CF2_BlueRec;


  /* max total blue zones is 12 */
  enum
  {
    CF2_MAX_BLUES      = 7,
    CF2_MAX_OTHERBLUES = 5
  };


  typedef struct  CF2_BluesRec_
  {
    CF2_Fixed  scale;
    CF2_UInt   count;
    FT_Bool    suppressOvershoot;
    FT_Bool    doEmBoxHints;

    CF2_Fixed  blueScale;
    CF2_Fixed  blueShift;
    CF2_Fixed  blueFuzz;

    CF2_Fixed  boost;

    CF2_HintRec  emBoxTopEdge;
    CF2_HintRec  emBoxBottomEdge;

    CF2_BlueRec  zone[CF2_MAX_BLUES + CF2_MAX_OTHERBLUES];

  } CF2_BluesRec, *CF2_Blues;


  FT_LOCAL( void )
  cf2_blues_init( CF2_Blues  blues,
                  CF2_Font   font );
  FT_LOCAL( FT_Bool )
  cf2_blues_capture( const CF2_Blues  blues,
                     CF2_Hint         bottomHintEdge,
                     CF2_Hint         topHintEdge );


FT_END_HEADER


#endif /* PSBLUES_H_ */


/* END */

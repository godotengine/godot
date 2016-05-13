/***************************************************************************/
/*                                                                         */
/*  cf2hints.h                                                             */
/*                                                                         */
/*    Adobe's code for handling CFF hints (body).                          */
/*                                                                         */
/*  Copyright 2007-2013 Adobe Systems Incorporated.                        */
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


#ifndef __CF2HINTS_H__
#define __CF2HINTS_H__


FT_BEGIN_HEADER


  enum
  {
    CF2_MAX_HINTS = 96    /* maximum # of hints */
  };


  /*
   * A HintMask object stores a bit mask that specifies which hints in the
   * charstring are active at a given time.  Hints in CFF must be declared
   * at the start, before any drawing operators, with horizontal hints
   * preceding vertical hints.  The HintMask is ordered the same way, with
   * horizontal hints immediately followed by vertical hints.  Clients are
   * responsible for knowing how many of each type are present.
   *
   * The maximum total number of hints is 96, as specified by the CFF
   * specification.
   *
   * A HintMask is built 0 or more times while interpreting a charstring, by
   * the HintMask operator.  There is only one HintMask, but it is built or
   * rebuilt each time there is a hint substitution (HintMask operator) in
   * the charstring.  A default HintMask with all bits set is built if there
   * has been no HintMask operator prior to the first drawing operator.
   *
   */

  typedef struct  CF2_HintMaskRec_
  {
    FT_Error*  error;

    FT_Bool  isValid;
    FT_Bool  isNew;

    size_t  bitCount;
    size_t  byteCount;

    FT_Byte  mask[( CF2_MAX_HINTS + 7 ) / 8];

  } CF2_HintMaskRec, *CF2_HintMask;


  typedef struct  CF2_StemHintRec_
  {
    FT_Bool  used;     /* DS positions are valid         */

    CF2_Fixed  min;    /* original character space value */
    CF2_Fixed  max;

    CF2_Fixed  minDS;  /* DS position after first use    */
    CF2_Fixed  maxDS;

  } CF2_StemHintRec, *CF2_StemHint;


  /*
   * A HintMap object stores a piecewise linear function for mapping
   * y-coordinates from character space to device space, providing
   * appropriate pixel alignment to stem edges.
   *
   * The map is implemented as an array of `CF2_Hint' elements, each
   * representing an edge.  When edges are paired, as from stem hints, the
   * bottom edge must immediately precede the top edge in the array.
   * Element character space AND device space positions must both increase
   * monotonically in the array.  `CF2_Hint' elements are also used as
   * parameters to `cf2_blues_capture'.
   *
   * The `cf2_hintmap_build' method must be called before any drawing
   * operation (beginning with a Move operator) and at each hint
   * substitution (HintMask operator).
   *
   * The `cf2_hintmap_map' method is called to transform y-coordinates at
   * each drawing operation (move, line, curve).
   *
   */

  /* TODO: make this a CF2_ArrStack and add a deep copy method */
  enum
  {
    CF2_MAX_HINT_EDGES = CF2_MAX_HINTS * 2
  };


  typedef struct  CF2_HintMapRec_
  {
    CF2_Font  font;

    /* initial map based on blue zones */
    struct CF2_HintMapRec_*  initialHintMap;

    /* working storage for 2nd pass adjustHints */
    CF2_ArrStack  hintMoves;

    FT_Bool  isValid;
    FT_Bool  hinted;

    CF2_Fixed  scale;
    CF2_UInt   count;

    /* start search from this index */
    CF2_UInt  lastIndex;

    CF2_HintRec  edge[CF2_MAX_HINT_EDGES]; /* 192 */

  } CF2_HintMapRec, *CF2_HintMap;


  FT_LOCAL( FT_Bool )
  cf2_hint_isValid( const CF2_Hint  hint );
  FT_LOCAL( FT_Bool )
  cf2_hint_isTop( const CF2_Hint  hint );
  FT_LOCAL( FT_Bool )
  cf2_hint_isBottom( const CF2_Hint  hint );
  FT_LOCAL( void )
  cf2_hint_lock( CF2_Hint  hint );


  FT_LOCAL( void )
  cf2_hintmap_init( CF2_HintMap   hintmap,
                    CF2_Font      font,
                    CF2_HintMap   initialMap,
                    CF2_ArrStack  hintMoves,
                    CF2_Fixed     scale );
  FT_LOCAL( void )
  cf2_hintmap_build( CF2_HintMap   hintmap,
                     CF2_ArrStack  hStemHintArray,
                     CF2_ArrStack  vStemHintArray,
                     CF2_HintMask  hintMask,
                     CF2_Fixed     hintOrigin,
                     FT_Bool       initialMap );


  /*
   * GlyphPath is a wrapper for drawing operations that scales the
   * coordinates according to the render matrix and HintMap.  It also tracks
   * open paths to control ClosePath and to insert MoveTo for broken fonts.
   *
   */
  typedef struct  CF2_GlyphPathRec_
  {
    /* TODO: gather some of these into a hinting context */

    CF2_Font              font;           /* font instance    */
    CF2_OutlineCallbacks  callbacks;      /* outline consumer */


    CF2_HintMapRec  hintMap;        /* current hint map            */
    CF2_HintMapRec  firstHintMap;   /* saved copy                  */
    CF2_HintMapRec  initialHintMap; /* based on all captured hints */

    CF2_ArrStackRec  hintMoves;  /* list of hint moves for 2nd pass */

    CF2_Fixed  scaleX;         /* matrix a */
    CF2_Fixed  scaleC;         /* matrix c */
    CF2_Fixed  scaleY;         /* matrix d */

    FT_Vector  fractionalTranslation;  /* including deviceXScale */
#if 0
    CF2_Fixed  hShift;    /* character space horizontal shift */
                          /* (for fauxing)                    */
#endif

    FT_Bool  pathIsOpen;     /* true after MoveTo                     */
    FT_Bool  darken;         /* true if stem darkening                */
    FT_Bool  moveIsPending;  /* true between MoveTo and offset MoveTo */

    /* references used to call `cf2_hintmap_build', if necessary */
    CF2_ArrStack         hStemHintArray;
    CF2_ArrStack         vStemHintArray;
    CF2_HintMask         hintMask;     /* ptr to the current mask */
    CF2_Fixed            hintOriginY;  /* copy of current origin  */
    const CF2_BluesRec*  blues;

    CF2_Fixed  xOffset;        /* character space offsets */
    CF2_Fixed  yOffset;

    /* character space miter limit threshold */
    CF2_Fixed  miterLimit;
    /* vertical/horzizontal snap distance in character space */
    CF2_Fixed  snapThreshold;

    FT_Vector  offsetStart0;  /* first and second points of first */
    FT_Vector  offsetStart1;  /* element with offset applied      */

    /* current point, character space, before offset */
    FT_Vector  currentCS;
    /* current point, device space */
    FT_Vector  currentDS;
    FT_Vector  start;         /* start point of subpath */

    /* the following members constitute the `queue' of one element */
    FT_Bool  elemIsQueued;
    CF2_Int  prevElemOp;

    FT_Vector  prevElemP0;
    FT_Vector  prevElemP1;
    FT_Vector  prevElemP2;
    FT_Vector  prevElemP3;

  } CF2_GlyphPathRec, *CF2_GlyphPath;


  FT_LOCAL( void )
  cf2_glyphpath_init( CF2_GlyphPath         glyphpath,
                      CF2_Font              font,
                      CF2_OutlineCallbacks  callbacks,
                      CF2_Fixed             scaleY,
                      /* CF2_Fixed hShift, */
                      CF2_ArrStack          hStemHintArray,
                      CF2_ArrStack          vStemHintArray,
                      CF2_HintMask          hintMask,
                      CF2_Fixed             hintOrigin,
                      const CF2_Blues       blues,
                      const FT_Vector*      fractionalTranslation );
  FT_LOCAL( void )
  cf2_glyphpath_finalize( CF2_GlyphPath  glyphpath );

  FT_LOCAL( void )
  cf2_glyphpath_moveTo( CF2_GlyphPath  glyphpath,
                        CF2_Fixed      x,
                        CF2_Fixed      y );
  FT_LOCAL( void )
  cf2_glyphpath_lineTo( CF2_GlyphPath  glyphpath,
                        CF2_Fixed      x,
                        CF2_Fixed      y );
  FT_LOCAL( void )
  cf2_glyphpath_curveTo( CF2_GlyphPath  glyphpath,
                         CF2_Fixed      x1,
                         CF2_Fixed      y1,
                         CF2_Fixed      x2,
                         CF2_Fixed      y2,
                         CF2_Fixed      x3,
                         CF2_Fixed      y3 );
  FT_LOCAL( void )
  cf2_glyphpath_closeOpenPath( CF2_GlyphPath  glyphpath );


FT_END_HEADER


#endif /* __CF2HINTS_H__ */


/* END */

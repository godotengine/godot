/****************************************************************************
 *
 * pshalgo.h
 *
 *   PostScript hinting algorithm (specification).
 *
 * Copyright (C) 2001-2019 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef PSHALGO_H_
#define PSHALGO_H_


#include "pshrec.h"
#include "pshglob.h"


FT_BEGIN_HEADER


  /* handle to Hint structure */
  typedef struct PSH_HintRec_*  PSH_Hint;


  /* hint bit-flags */
#define PSH_HINT_GHOST   PS_HINT_FLAG_GHOST
#define PSH_HINT_BOTTOM  PS_HINT_FLAG_BOTTOM
#define PSH_HINT_ACTIVE  4U
#define PSH_HINT_FITTED  8U


#define psh_hint_is_active( x )  ( ( (x)->flags & PSH_HINT_ACTIVE ) != 0 )
#define psh_hint_is_ghost( x )   ( ( (x)->flags & PSH_HINT_GHOST  ) != 0 )
#define psh_hint_is_fitted( x )  ( ( (x)->flags & PSH_HINT_FITTED ) != 0 )

#define psh_hint_activate( x )    (x)->flags |=  PSH_HINT_ACTIVE
#define psh_hint_deactivate( x )  (x)->flags &= ~PSH_HINT_ACTIVE
#define psh_hint_set_fitted( x )  (x)->flags |=  PSH_HINT_FITTED


  /* hint structure */
  typedef struct  PSH_HintRec_
  {
    FT_Int    org_pos;
    FT_Int    org_len;
    FT_Pos    cur_pos;
    FT_Pos    cur_len;
    FT_UInt   flags;
    PSH_Hint  parent;
    FT_Int    order;

  } PSH_HintRec;


  /* this is an interpolation zone used for strong points;  */
  /* weak points are interpolated according to their strong */
  /* neighbours                                             */
  typedef struct  PSH_ZoneRec_
  {
    FT_Fixed  scale;
    FT_Fixed  delta;
    FT_Pos    min;
    FT_Pos    max;

  } PSH_ZoneRec, *PSH_Zone;


  typedef struct  PSH_Hint_TableRec_
  {
    FT_UInt        max_hints;
    FT_UInt        num_hints;
    PSH_Hint       hints;
    PSH_Hint*      sort;
    PSH_Hint*      sort_global;
    FT_UInt        num_zones;
    PSH_ZoneRec*   zones;
    PSH_Zone       zone;
    PS_Mask_Table  hint_masks;
    PS_Mask_Table  counter_masks;

  } PSH_Hint_TableRec, *PSH_Hint_Table;


  typedef struct PSH_PointRec_*    PSH_Point;
  typedef struct PSH_ContourRec_*  PSH_Contour;

  enum
  {
    PSH_DIR_NONE  =  4,
    PSH_DIR_UP    = -1,
    PSH_DIR_DOWN  =  1,
    PSH_DIR_LEFT  = -2,
    PSH_DIR_RIGHT =  2
  };

#define PSH_DIR_HORIZONTAL  2
#define PSH_DIR_VERTICAL    1

#define PSH_DIR_COMPARE( d1, d2 )   ( (d1) == (d2) || (d1) == -(d2) )
#define PSH_DIR_IS_HORIZONTAL( d )  PSH_DIR_COMPARE( d, PSH_DIR_HORIZONTAL )
#define PSH_DIR_IS_VERTICAL( d )    PSH_DIR_COMPARE( d, PSH_DIR_VERTICAL )


  /* the following bit-flags are computed once by the glyph */
  /* analyzer, for both dimensions                          */
#define PSH_POINT_OFF     1U      /* point is off the curve */
#define PSH_POINT_SMOOTH  2U      /* point is smooth        */
#define PSH_POINT_INFLEX  4U      /* point is inflection    */


#define psh_point_is_smooth( p )  ( (p)->flags & PSH_POINT_SMOOTH )
#define psh_point_is_off( p )     ( (p)->flags & PSH_POINT_OFF    )
#define psh_point_is_inflex( p )  ( (p)->flags & PSH_POINT_INFLEX )

#define psh_point_set_smooth( p )  (p)->flags |= PSH_POINT_SMOOTH
#define psh_point_set_off( p )     (p)->flags |= PSH_POINT_OFF
#define psh_point_set_inflex( p )  (p)->flags |= PSH_POINT_INFLEX


  /* the following bit-flags are re-computed for each dimension */
#define PSH_POINT_STRONG      16U /* point is strong                           */
#define PSH_POINT_FITTED      32U /* point is already fitted                   */
#define PSH_POINT_EXTREMUM    64U /* point is local extremum                   */
#define PSH_POINT_POSITIVE   128U /* extremum has positive contour flow        */
#define PSH_POINT_NEGATIVE   256U /* extremum has negative contour flow        */
#define PSH_POINT_EDGE_MIN   512U /* point is aligned to left/bottom stem edge */
#define PSH_POINT_EDGE_MAX  1024U /* point is aligned to top/right stem edge   */


#define psh_point_is_strong( p )    ( (p)->flags2 & PSH_POINT_STRONG )
#define psh_point_is_fitted( p )    ( (p)->flags2 & PSH_POINT_FITTED )
#define psh_point_is_extremum( p )  ( (p)->flags2 & PSH_POINT_EXTREMUM )
#define psh_point_is_positive( p )  ( (p)->flags2 & PSH_POINT_POSITIVE )
#define psh_point_is_negative( p )  ( (p)->flags2 & PSH_POINT_NEGATIVE )
#define psh_point_is_edge_min( p )  ( (p)->flags2 & PSH_POINT_EDGE_MIN )
#define psh_point_is_edge_max( p )  ( (p)->flags2 & PSH_POINT_EDGE_MAX )

#define psh_point_set_strong( p )    (p)->flags2 |= PSH_POINT_STRONG
#define psh_point_set_fitted( p )    (p)->flags2 |= PSH_POINT_FITTED
#define psh_point_set_extremum( p )  (p)->flags2 |= PSH_POINT_EXTREMUM
#define psh_point_set_positive( p )  (p)->flags2 |= PSH_POINT_POSITIVE
#define psh_point_set_negative( p )  (p)->flags2 |= PSH_POINT_NEGATIVE
#define psh_point_set_edge_min( p )  (p)->flags2 |= PSH_POINT_EDGE_MIN
#define psh_point_set_edge_max( p )  (p)->flags2 |= PSH_POINT_EDGE_MAX


  typedef struct  PSH_PointRec_
  {
    PSH_Point    prev;
    PSH_Point    next;
    PSH_Contour  contour;
    FT_UInt      flags;
    FT_UInt      flags2;
    FT_Char      dir_in;
    FT_Char      dir_out;
    PSH_Hint     hint;
    FT_Pos       org_u;
    FT_Pos       org_v;
    FT_Pos       cur_u;
#ifdef DEBUG_HINTER
    FT_Pos       org_x;
    FT_Pos       cur_x;
    FT_Pos       org_y;
    FT_Pos       cur_y;
    FT_UInt      flags_x;
    FT_UInt      flags_y;
#endif

  } PSH_PointRec;


  typedef struct  PSH_ContourRec_
  {
    PSH_Point  start;
    FT_UInt    count;

  } PSH_ContourRec;


  typedef struct  PSH_GlyphRec_
  {
    FT_UInt            num_points;
    FT_UInt            num_contours;

    PSH_Point          points;
    PSH_Contour        contours;

    FT_Memory          memory;
    FT_Outline*        outline;
    PSH_Globals        globals;
    PSH_Hint_TableRec  hint_tables[2];

    FT_Bool            vertical;
    FT_Int             major_dir;
    FT_Int             minor_dir;

    FT_Bool            do_horz_hints;
    FT_Bool            do_vert_hints;
    FT_Bool            do_horz_snapping;
    FT_Bool            do_vert_snapping;
    FT_Bool            do_stem_adjust;

  } PSH_GlyphRec, *PSH_Glyph;


#ifdef DEBUG_HINTER
  extern PSH_Hint_Table  ps_debug_hint_table;

  typedef void
  (*PSH_HintFunc)( PSH_Hint  hint,
                   FT_Bool   vertical );

  extern PSH_HintFunc    ps_debug_hint_func;

  extern PSH_Glyph       ps_debug_glyph;
#endif


  extern FT_Error
  ps_hints_apply( PS_Hints        ps_hints,
                  FT_Outline*     outline,
                  PSH_Globals     globals,
                  FT_Render_Mode  hint_mode );


FT_END_HEADER


#endif /* PSHALGO_H_ */


/* END */

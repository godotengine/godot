/****************************************************************************
 *
 * pshglob.h
 *
 *   PostScript hinter global hinting management.
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


#ifndef PSHGLOB_H_
#define PSHGLOB_H_


#include FT_FREETYPE_H
#include FT_INTERNAL_POSTSCRIPT_HINTS_H


FT_BEGIN_HEADER


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                    GLOBAL HINTS INTERNALS                     *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  /**************************************************************************
   *
   * @constant:
   *   PS_GLOBALS_MAX_BLUE_ZONES
   *
   * @description:
   *   The maximum number of blue zones in a font global hints structure.
   *   See @PS_Globals_BluesRec.
   */
#define PS_GLOBALS_MAX_BLUE_ZONES  16


  /**************************************************************************
   *
   * @constant:
   *   PS_GLOBALS_MAX_STD_WIDTHS
   *
   * @description:
   *   The maximum number of standard and snap widths in either the
   *   horizontal or vertical direction.  See @PS_Globals_WidthsRec.
   */
#define PS_GLOBALS_MAX_STD_WIDTHS  16


  /* standard and snap width */
  typedef struct  PSH_WidthRec_
  {
    FT_Int  org;
    FT_Pos  cur;
    FT_Pos  fit;

  } PSH_WidthRec, *PSH_Width;


  /* standard and snap widths table */
  typedef struct  PSH_WidthsRec_
  {
    FT_UInt       count;
    PSH_WidthRec  widths[PS_GLOBALS_MAX_STD_WIDTHS];

  } PSH_WidthsRec, *PSH_Widths;


  typedef struct  PSH_DimensionRec_
  {
    PSH_WidthsRec  stdw;
    FT_Fixed       scale_mult;
    FT_Fixed       scale_delta;

  } PSH_DimensionRec, *PSH_Dimension;


  /* blue zone descriptor */
  typedef struct  PSH_Blue_ZoneRec_
  {
    FT_Int  org_ref;
    FT_Int  org_delta;
    FT_Int  org_top;
    FT_Int  org_bottom;

    FT_Pos  cur_ref;
    FT_Pos  cur_delta;
    FT_Pos  cur_bottom;
    FT_Pos  cur_top;

  } PSH_Blue_ZoneRec, *PSH_Blue_Zone;


  typedef struct  PSH_Blue_TableRec_
  {
    FT_UInt           count;
    PSH_Blue_ZoneRec  zones[PS_GLOBALS_MAX_BLUE_ZONES];

  } PSH_Blue_TableRec, *PSH_Blue_Table;


  /* blue zones table */
  typedef struct  PSH_BluesRec_
  {
    PSH_Blue_TableRec  normal_top;
    PSH_Blue_TableRec  normal_bottom;
    PSH_Blue_TableRec  family_top;
    PSH_Blue_TableRec  family_bottom;

    FT_Fixed           blue_scale;
    FT_Int             blue_shift;
    FT_Int             blue_threshold;
    FT_Int             blue_fuzz;
    FT_Bool            no_overshoots;

  } PSH_BluesRec, *PSH_Blues;


  /* font globals.                                         */
  /* dimension 0 => X coordinates + vertical hints/stems   */
  /* dimension 1 => Y coordinates + horizontal hints/stems */
  typedef struct  PSH_GlobalsRec_
  {
    FT_Memory         memory;
    PSH_DimensionRec  dimension[2];
    PSH_BluesRec      blues;

  } PSH_GlobalsRec;


#define PSH_BLUE_ALIGN_NONE  0
#define PSH_BLUE_ALIGN_TOP   1
#define PSH_BLUE_ALIGN_BOT   2


  typedef struct  PSH_AlignmentRec_
  {
    int     align;
    FT_Pos  align_top;
    FT_Pos  align_bot;

  } PSH_AlignmentRec, *PSH_Alignment;


  FT_LOCAL( void )
  psh_globals_funcs_init( PSH_Globals_FuncsRec*  funcs );


#if 0
  /* snap a stem width to fitter coordinates.  `org_width' is in font */
  /* units.  The result is in device pixels (26.6 format).            */
  FT_LOCAL( FT_Pos )
  psh_dimension_snap_width( PSH_Dimension  dimension,
                            FT_Int         org_width );
#endif

  FT_LOCAL( void )
  psh_globals_set_scale( PSH_Globals  globals,
                         FT_Fixed     x_scale,
                         FT_Fixed     y_scale,
                         FT_Fixed     x_delta,
                         FT_Fixed     y_delta );

  /* snap a stem to one or two blue zones */
  FT_LOCAL( void )
  psh_blues_snap_stem( PSH_Blues      blues,
                       FT_Int         stem_top,
                       FT_Int         stem_bot,
                       PSH_Alignment  alignment );
  /* */

#ifdef DEBUG_HINTER
  extern PSH_Globals  ps_debug_globals;
#endif


FT_END_HEADER


#endif /* PSHGLOB_H_ */


/* END */

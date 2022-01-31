/****************************************************************************
 *
 * cffotypes.h
 *
 *   Basic OpenType/CFF object type definitions (specification).
 *
 * Copyright (C) 2017-2021 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef CFFOTYPES_H_
#define CFFOTYPES_H_

#include <freetype/internal/ftobjs.h>
#include <freetype/internal/cfftypes.h>
#include <freetype/internal/tttypes.h>
#include <freetype/internal/services/svpscmap.h>
#include <freetype/internal/pshints.h>


FT_BEGIN_HEADER


  typedef TT_Face  CFF_Face;


  /**************************************************************************
   *
   * @type:
   *   CFF_Size
   *
   * @description:
   *   A handle to an OpenType size object.
   */
  typedef struct  CFF_SizeRec_
  {
    FT_SizeRec  root;
    FT_ULong    strike_index;    /* 0xFFFFFFFF to indicate invalid */

  } CFF_SizeRec, *CFF_Size;


  /**************************************************************************
   *
   * @type:
   *   CFF_GlyphSlot
   *
   * @description:
   *   A handle to an OpenType glyph slot object.
   */
  typedef struct  CFF_GlyphSlotRec_
  {
    FT_GlyphSlotRec  root;

    FT_Bool  hint;
    FT_Bool  scaled;

    FT_Fixed  x_scale;
    FT_Fixed  y_scale;

  } CFF_GlyphSlotRec, *CFF_GlyphSlot;


  /**************************************************************************
   *
   * @type:
   *   CFF_Internal
   *
   * @description:
   *   The interface to the 'internal' field of `FT_Size`.
   */
  typedef struct  CFF_InternalRec_
  {
    PSH_Globals  topfont;
    PSH_Globals  subfonts[CFF_MAX_CID_FONTS];

  } CFF_InternalRec, *CFF_Internal;


  /**************************************************************************
   *
   * Subglyph transformation record.
   */
  typedef struct  CFF_Transform_
  {
    FT_Fixed    xx, xy;     /* transformation matrix coefficients */
    FT_Fixed    yx, yy;
    FT_F26Dot6  ox, oy;     /* offsets                            */

  } CFF_Transform;


FT_END_HEADER


#endif /* CFFOTYPES_H_ */


/* END */

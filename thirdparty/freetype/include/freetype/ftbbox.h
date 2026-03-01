/****************************************************************************
 *
 * ftbbox.h
 *
 *   FreeType exact bbox computation (specification).
 *
 * Copyright (C) 1996-2025 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


  /**************************************************************************
   *
   * This component has a _single_ role: to compute exact outline bounding
   * boxes.
   *
   * It is separated from the rest of the engine for various technical
   * reasons.  It may well be integrated in 'ftoutln' later.
   *
   */


#ifndef FTBBOX_H_
#define FTBBOX_H_


#include <freetype/freetype.h>

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @section:
   *   outline_processing
   *
   */


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_Get_BBox
   *
   * @description:
   *   Compute the exact bounding box of an outline.  This is slower than
   *   computing the control box.  However, it uses an advanced algorithm
   *   that returns _very_ quickly when the two boxes coincide.  Otherwise,
   *   the outline Bezier arcs are traversed to extract their extrema.
   *
   * @input:
   *   outline ::
   *     A pointer to the source outline.
   *
   * @output:
   *   abbox ::
   *     The outline's exact bounding box.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   If the font is tricky and the glyph has been loaded with
   *   @FT_LOAD_NO_SCALE, the resulting BBox is meaningless.  To get
   *   reasonable values for the BBox it is necessary to load the glyph at a
   *   large ppem value (so that the hinting instructions can properly shift
   *   and scale the subglyphs), then extracting the BBox, which can be
   *   eventually converted back to font units.
   */
  FT_EXPORT( FT_Error )
  FT_Outline_Get_BBox( FT_Outline*  outline,
                       FT_BBox     *abbox );

  /* */


FT_END_HEADER

#endif /* FTBBOX_H_ */


/* END */


/* Local Variables: */
/* coding: utf-8    */
/* End:             */

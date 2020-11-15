/****************************************************************************
 *
 * ftadvanc.h
 *
 *   Quick computation of advance widths (specification only).
 *
 * Copyright (C) 2008-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef FTADVANC_H_
#define FTADVANC_H_


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
   *   quick_advance
   *
   * @title:
   *   Quick retrieval of advance values
   *
   * @abstract:
   *   Retrieve horizontal and vertical advance values without processing
   *   glyph outlines, if possible.
   *
   * @description:
   *   This section contains functions to quickly extract advance values
   *   without handling glyph outlines, if possible.
   *
   * @order:
   *   FT_Get_Advance
   *   FT_Get_Advances
   *
   */


  /**************************************************************************
   *
   * @enum:
   *   FT_ADVANCE_FLAG_FAST_ONLY
   *
   * @description:
   *   A bit-flag to be OR-ed with the `flags` parameter of the
   *   @FT_Get_Advance and @FT_Get_Advances functions.
   *
   *   If set, it indicates that you want these functions to fail if the
   *   corresponding hinting mode or font driver doesn't allow for very quick
   *   advance computation.
   *
   *   Typically, glyphs that are either unscaled, unhinted, bitmapped, or
   *   light-hinted can have their advance width computed very quickly.
   *
   *   Normal and bytecode hinted modes that require loading, scaling, and
   *   hinting of the glyph outline, are extremely slow by comparison.
   */
#define FT_ADVANCE_FLAG_FAST_ONLY  0x20000000L


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Advance
   *
   * @description:
   *   Retrieve the advance value of a given glyph outline in an @FT_Face.
   *
   * @input:
   *   face ::
   *     The source @FT_Face handle.
   *
   *   gindex ::
   *     The glyph index.
   *
   *   load_flags ::
   *     A set of bit flags similar to those used when calling
   *     @FT_Load_Glyph, used to determine what kind of advances you need.
   *
   * @output:
   *   padvance ::
   *     The advance value.  If scaling is performed (based on the value of
   *     `load_flags`), the advance value is in 16.16 format.  Otherwise, it
   *     is in font units.
   *
   *     If @FT_LOAD_VERTICAL_LAYOUT is set, this is the vertical advance
   *     corresponding to a vertical layout.  Otherwise, it is the horizontal
   *     advance in a horizontal layout.
   *
   * @return:
   *   FreeType error code.  0 means success.
   *
   * @note:
   *   This function may fail if you use @FT_ADVANCE_FLAG_FAST_ONLY and if
   *   the corresponding font backend doesn't have a quick way to retrieve
   *   the advances.
   *
   *   A scaled advance is returned in 16.16 format but isn't transformed by
   *   the affine transformation specified by @FT_Set_Transform.
   */
  FT_EXPORT( FT_Error )
  FT_Get_Advance( FT_Face    face,
                  FT_UInt    gindex,
                  FT_Int32   load_flags,
                  FT_Fixed  *padvance );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Advances
   *
   * @description:
   *   Retrieve the advance values of several glyph outlines in an @FT_Face.
   *
   * @input:
   *   face ::
   *     The source @FT_Face handle.
   *
   *   start ::
   *     The first glyph index.
   *
   *   count ::
   *     The number of advance values you want to retrieve.
   *
   *   load_flags ::
   *     A set of bit flags similar to those used when calling
   *     @FT_Load_Glyph.
   *
   * @output:
   *   padvance ::
   *     The advance values.  This array, to be provided by the caller, must
   *     contain at least `count` elements.
   *
   *     If scaling is performed (based on the value of `load_flags`), the
   *     advance values are in 16.16 format.  Otherwise, they are in font
   *     units.
   *
   *     If @FT_LOAD_VERTICAL_LAYOUT is set, these are the vertical advances
   *     corresponding to a vertical layout.  Otherwise, they are the
   *     horizontal advances in a horizontal layout.
   *
   * @return:
   *   FreeType error code.  0 means success.
   *
   * @note:
   *   This function may fail if you use @FT_ADVANCE_FLAG_FAST_ONLY and if
   *   the corresponding font backend doesn't have a quick way to retrieve
   *   the advances.
   *
   *   Scaled advances are returned in 16.16 format but aren't transformed by
   *   the affine transformation specified by @FT_Set_Transform.
   */
  FT_EXPORT( FT_Error )
  FT_Get_Advances( FT_Face    face,
                   FT_UInt    start,
                   FT_UInt    count,
                   FT_Int32   load_flags,
                   FT_Fixed  *padvances );

  /* */


FT_END_HEADER

#endif /* FTADVANC_H_ */


/* END */

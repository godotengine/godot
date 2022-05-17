/****************************************************************************
 *
 * ftbitmap.h
 *
 *   FreeType utility functions for bitmaps (specification).
 *
 * Copyright (C) 2004-2022 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef FTBITMAP_H_
#define FTBITMAP_H_


#include <freetype/freetype.h>
#include <freetype/ftcolor.h>

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @section:
   *   bitmap_handling
   *
   * @title:
   *   Bitmap Handling
   *
   * @abstract:
   *   Handling FT_Bitmap objects.
   *
   * @description:
   *   This section contains functions for handling @FT_Bitmap objects,
   *   automatically adjusting the target's bitmap buffer size as needed.
   *
   *   Note that none of the functions changes the bitmap's 'flow' (as
   *   indicated by the sign of the `pitch` field in @FT_Bitmap).
   *
   *   To set the flow, assign an appropriate positive or negative value to
   *   the `pitch` field of the target @FT_Bitmap object after calling
   *   @FT_Bitmap_Init but before calling any of the other functions
   *   described here.
   */


  /**************************************************************************
   *
   * @function:
   *   FT_Bitmap_Init
   *
   * @description:
   *   Initialize a pointer to an @FT_Bitmap structure.
   *
   * @inout:
   *   abitmap ::
   *     A pointer to the bitmap structure.
   *
   * @note:
   *   A deprecated name for the same function is `FT_Bitmap_New`.
   */
  FT_EXPORT( void )
  FT_Bitmap_Init( FT_Bitmap  *abitmap );


  /* deprecated */
  FT_EXPORT( void )
  FT_Bitmap_New( FT_Bitmap  *abitmap );


  /**************************************************************************
   *
   * @function:
   *   FT_Bitmap_Copy
   *
   * @description:
   *   Copy a bitmap into another one.
   *
   * @input:
   *   library ::
   *     A handle to a library object.
   *
   *   source ::
   *     A handle to the source bitmap.
   *
   * @output:
   *   target ::
   *     A handle to the target bitmap.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   `source->buffer` and `target->buffer` must neither be equal nor
   *   overlap.
   */
  FT_EXPORT( FT_Error )
  FT_Bitmap_Copy( FT_Library        library,
                  const FT_Bitmap  *source,
                  FT_Bitmap        *target );


  /**************************************************************************
   *
   * @function:
   *   FT_Bitmap_Embolden
   *
   * @description:
   *   Embolden a bitmap.  The new bitmap will be about `xStrength` pixels
   *   wider and `yStrength` pixels higher.  The left and bottom borders are
   *   kept unchanged.
   *
   * @input:
   *   library ::
   *     A handle to a library object.
   *
   *   xStrength ::
   *     How strong the glyph is emboldened horizontally.  Expressed in 26.6
   *     pixel format.
   *
   *   yStrength ::
   *     How strong the glyph is emboldened vertically.  Expressed in 26.6
   *     pixel format.
   *
   * @inout:
   *   bitmap ::
   *     A handle to the target bitmap.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   The current implementation restricts `xStrength` to be less than or
   *   equal to~8 if bitmap is of pixel_mode @FT_PIXEL_MODE_MONO.
   *
   *   If you want to embolden the bitmap owned by a @FT_GlyphSlotRec, you
   *   should call @FT_GlyphSlot_Own_Bitmap on the slot first.
   *
   *   Bitmaps in @FT_PIXEL_MODE_GRAY2 and @FT_PIXEL_MODE_GRAY@ format are
   *   converted to @FT_PIXEL_MODE_GRAY format (i.e., 8bpp).
   */
  FT_EXPORT( FT_Error )
  FT_Bitmap_Embolden( FT_Library  library,
                      FT_Bitmap*  bitmap,
                      FT_Pos      xStrength,
                      FT_Pos      yStrength );


  /**************************************************************************
   *
   * @function:
   *   FT_Bitmap_Convert
   *
   * @description:
   *   Convert a bitmap object with depth 1bpp, 2bpp, 4bpp, 8bpp or 32bpp to
   *   a bitmap object with depth 8bpp, making the number of used bytes per
   *   line (a.k.a. the 'pitch') a multiple of `alignment`.
   *
   * @input:
   *   library ::
   *     A handle to a library object.
   *
   *   source ::
   *     The source bitmap.
   *
   *   alignment ::
   *     The pitch of the bitmap is a multiple of this argument.  Common
   *     values are 1, 2, or 4.
   *
   * @output:
   *   target ::
   *     The target bitmap.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   It is possible to call @FT_Bitmap_Convert multiple times without
   *   calling @FT_Bitmap_Done (the memory is simply reallocated).
   *
   *   Use @FT_Bitmap_Done to finally remove the bitmap object.
   *
   *   The `library` argument is taken to have access to FreeType's memory
   *   handling functions.
   *
   *   `source->buffer` and `target->buffer` must neither be equal nor
   *   overlap.
   */
  FT_EXPORT( FT_Error )
  FT_Bitmap_Convert( FT_Library        library,
                     const FT_Bitmap  *source,
                     FT_Bitmap        *target,
                     FT_Int            alignment );


  /**************************************************************************
   *
   * @function:
   *   FT_Bitmap_Blend
   *
   * @description:
   *   Blend a bitmap onto another bitmap, using a given color.
   *
   * @input:
   *   library ::
   *     A handle to a library object.
   *
   *   source ::
   *     The source bitmap, which can have any @FT_Pixel_Mode format.
   *
   *   source_offset ::
   *     The offset vector to the upper left corner of the source bitmap in
   *     26.6 pixel format.  It should represent an integer offset; the
   *     function will set the lowest six bits to zero to enforce that.
   *
   *   color ::
   *     The color used to draw `source` onto `target`.
   *
   * @inout:
   *   target ::
   *     A handle to an `FT_Bitmap` object.  It should be either initialized
   *     as empty with a call to @FT_Bitmap_Init, or it should be of type
   *     @FT_PIXEL_MODE_BGRA.
   *
   *   atarget_offset ::
   *     The offset vector to the upper left corner of the target bitmap in
   *     26.6 pixel format.  It should represent an integer offset; the
   *     function will set the lowest six bits to zero to enforce that.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   This function doesn't perform clipping.
   *
   *   The bitmap in `target` gets allocated or reallocated as needed; the
   *   vector `atarget_offset` is updated accordingly.
   *
   *   In case of allocation or reallocation, the bitmap's pitch is set to
   *   `4 * width`.  Both `source` and `target` must have the same bitmap
   *   flow (as indicated by the sign of the `pitch` field).
   *
   *   `source->buffer` and `target->buffer` must neither be equal nor
   *   overlap.
   *
   * @since:
   *   2.10
   */
  FT_EXPORT( FT_Error )
  FT_Bitmap_Blend( FT_Library         library,
                   const FT_Bitmap*   source,
                   const FT_Vector    source_offset,
                   FT_Bitmap*         target,
                   FT_Vector         *atarget_offset,
                   FT_Color           color );


  /**************************************************************************
   *
   * @function:
   *   FT_GlyphSlot_Own_Bitmap
   *
   * @description:
   *   Make sure that a glyph slot owns `slot->bitmap`.
   *
   * @input:
   *   slot ::
   *     The glyph slot.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   This function is to be used in combination with @FT_Bitmap_Embolden.
   */
  FT_EXPORT( FT_Error )
  FT_GlyphSlot_Own_Bitmap( FT_GlyphSlot  slot );


  /**************************************************************************
   *
   * @function:
   *   FT_Bitmap_Done
   *
   * @description:
   *   Destroy a bitmap object initialized with @FT_Bitmap_Init.
   *
   * @input:
   *   library ::
   *     A handle to a library object.
   *
   *   bitmap ::
   *     The bitmap object to be freed.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   The `library` argument is taken to have access to FreeType's memory
   *   handling functions.
   */
  FT_EXPORT( FT_Error )
  FT_Bitmap_Done( FT_Library  library,
                  FT_Bitmap  *bitmap );


  /* */


FT_END_HEADER

#endif /* FTBITMAP_H_ */


/* END */

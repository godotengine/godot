/****************************************************************************
 *
 * ftoutln.h
 *
 *   Support for the FT_Outline type used to store glyph shapes of
 *   most scalable font formats (specification).
 *
 * Copyright (C) 1996-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef FTOUTLN_H_
#define FTOUTLN_H_


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
   * @title:
   *   Outline Processing
   *
   * @abstract:
   *   Functions to create, transform, and render vectorial glyph images.
   *
   * @description:
   *   This section contains routines used to create and destroy scalable
   *   glyph images known as 'outlines'.  These can also be measured,
   *   transformed, and converted into bitmaps and pixmaps.
   *
   * @order:
   *   FT_Outline
   *   FT_Outline_New
   *   FT_Outline_Done
   *   FT_Outline_Copy
   *   FT_Outline_Translate
   *   FT_Outline_Transform
   *   FT_Outline_Embolden
   *   FT_Outline_EmboldenXY
   *   FT_Outline_Reverse
   *   FT_Outline_Check
   *
   *   FT_Outline_Get_CBox
   *   FT_Outline_Get_BBox
   *
   *   FT_Outline_Get_Bitmap
   *   FT_Outline_Render
   *   FT_Outline_Decompose
   *   FT_Outline_Funcs
   *   FT_Outline_MoveToFunc
   *   FT_Outline_LineToFunc
   *   FT_Outline_ConicToFunc
   *   FT_Outline_CubicToFunc
   *
   *   FT_Orientation
   *   FT_Outline_Get_Orientation
   *
   *   FT_OUTLINE_XXX
   *
   */


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_Decompose
   *
   * @description:
   *   Walk over an outline's structure to decompose it into individual
   *   segments and Bezier arcs.  This function also emits 'move to'
   *   operations to indicate the start of new contours in the outline.
   *
   * @input:
   *   outline ::
   *     A pointer to the source target.
   *
   *   func_interface ::
   *     A table of 'emitters', i.e., function pointers called during
   *     decomposition to indicate path operations.
   *
   * @inout:
   *   user ::
   *     A typeless pointer that is passed to each emitter during the
   *     decomposition.  It can be used to store the state during the
   *     decomposition.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   Degenerate contours, segments, and Bezier arcs may be reported.  In
   *   most cases, it is best to filter these out before using the outline
   *   for stroking or other path modification purposes (which may cause
   *   degenerate segments to become non-degenerate and visible, like when
   *   stroke caps are used or the path is otherwise outset).  Some glyph
   *   outlines may contain deliberate degenerate single points for mark
   *   attachement.
   *
   *   Similarly, the function returns success for an empty outline also
   *   (doing nothing, that is, not calling any emitter); if necessary, you
   *   should filter this out, too.
   */
  FT_EXPORT( FT_Error )
  FT_Outline_Decompose( FT_Outline*              outline,
                        const FT_Outline_Funcs*  func_interface,
                        void*                    user );


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_New
   *
   * @description:
   *   Create a new outline of a given size.
   *
   * @input:
   *   library ::
   *     A handle to the library object from where the outline is allocated.
   *     Note however that the new outline will **not** necessarily be
   *     **freed**, when destroying the library, by @FT_Done_FreeType.
   *
   *   numPoints ::
   *     The maximum number of points within the outline.  Must be smaller
   *     than or equal to 0xFFFF (65535).
   *
   *   numContours ::
   *     The maximum number of contours within the outline.  This value must
   *     be in the range 0 to `numPoints`.
   *
   * @output:
   *   anoutline ::
   *     A handle to the new outline.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   The reason why this function takes a `library` parameter is simply to
   *   use the library's memory allocator.
   */
  FT_EXPORT( FT_Error )
  FT_Outline_New( FT_Library   library,
                  FT_UInt      numPoints,
                  FT_Int       numContours,
                  FT_Outline  *anoutline );


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_Done
   *
   * @description:
   *   Destroy an outline created with @FT_Outline_New.
   *
   * @input:
   *   library ::
   *     A handle of the library object used to allocate the outline.
   *
   *   outline ::
   *     A pointer to the outline object to be discarded.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   If the outline's 'owner' field is not set, only the outline descriptor
   *   will be released.
   */
  FT_EXPORT( FT_Error )
  FT_Outline_Done( FT_Library   library,
                   FT_Outline*  outline );


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_Check
   *
   * @description:
   *   Check the contents of an outline descriptor.
   *
   * @input:
   *   outline ::
   *     A handle to a source outline.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   An empty outline, or an outline with a single point only is also
   *   valid.
   */
  FT_EXPORT( FT_Error )
  FT_Outline_Check( FT_Outline*  outline );


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_Get_CBox
   *
   * @description:
   *   Return an outline's 'control box'.  The control box encloses all the
   *   outline's points, including Bezier control points.  Though it
   *   coincides with the exact bounding box for most glyphs, it can be
   *   slightly larger in some situations (like when rotating an outline that
   *   contains Bezier outside arcs).
   *
   *   Computing the control box is very fast, while getting the bounding box
   *   can take much more time as it needs to walk over all segments and arcs
   *   in the outline.  To get the latter, you can use the 'ftbbox'
   *   component, which is dedicated to this single task.
   *
   * @input:
   *   outline ::
   *     A pointer to the source outline descriptor.
   *
   * @output:
   *   acbox ::
   *     The outline's control box.
   *
   * @note:
   *   See @FT_Glyph_Get_CBox for a discussion of tricky fonts.
   */
  FT_EXPORT( void )
  FT_Outline_Get_CBox( const FT_Outline*  outline,
                       FT_BBox           *acbox );


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_Translate
   *
   * @description:
   *   Apply a simple translation to the points of an outline.
   *
   * @inout:
   *   outline ::
   *     A pointer to the target outline descriptor.
   *
   * @input:
   *   xOffset ::
   *     The horizontal offset.
   *
   *   yOffset ::
   *     The vertical offset.
   */
  FT_EXPORT( void )
  FT_Outline_Translate( const FT_Outline*  outline,
                        FT_Pos             xOffset,
                        FT_Pos             yOffset );


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_Copy
   *
   * @description:
   *   Copy an outline into another one.  Both objects must have the same
   *   sizes (number of points & number of contours) when this function is
   *   called.
   *
   * @input:
   *   source ::
   *     A handle to the source outline.
   *
   * @output:
   *   target ::
   *     A handle to the target outline.
   *
   * @return:
   *   FreeType error code.  0~means success.
   */
  FT_EXPORT( FT_Error )
  FT_Outline_Copy( const FT_Outline*  source,
                   FT_Outline        *target );


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_Transform
   *
   * @description:
   *   Apply a simple 2x2 matrix to all of an outline's points.  Useful for
   *   applying rotations, slanting, flipping, etc.
   *
   * @inout:
   *   outline ::
   *     A pointer to the target outline descriptor.
   *
   * @input:
   *   matrix ::
   *     A pointer to the transformation matrix.
   *
   * @note:
   *   You can use @FT_Outline_Translate if you need to translate the
   *   outline's points.
   */
  FT_EXPORT( void )
  FT_Outline_Transform( const FT_Outline*  outline,
                        const FT_Matrix*   matrix );


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_Embolden
   *
   * @description:
   *   Embolden an outline.  The new outline will be at most 4~times
   *   `strength` pixels wider and higher.  You may think of the left and
   *   bottom borders as unchanged.
   *
   *   Negative `strength` values to reduce the outline thickness are
   *   possible also.
   *
   * @inout:
   *   outline ::
   *     A handle to the target outline.
   *
   * @input:
   *   strength ::
   *     How strong the glyph is emboldened.  Expressed in 26.6 pixel format.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   The used algorithm to increase or decrease the thickness of the glyph
   *   doesn't change the number of points; this means that certain
   *   situations like acute angles or intersections are sometimes handled
   *   incorrectly.
   *
   *   If you need 'better' metrics values you should call
   *   @FT_Outline_Get_CBox or @FT_Outline_Get_BBox.
   *
   *   To get meaningful results, font scaling values must be set with
   *   functions like @FT_Set_Char_Size before calling FT_Render_Glyph.
   *
   * @example:
   *   ```
   *     FT_Load_Glyph( face, index, FT_LOAD_DEFAULT );
   *
   *     if ( face->glyph->format == FT_GLYPH_FORMAT_OUTLINE )
   *       FT_Outline_Embolden( &face->glyph->outline, strength );
   *   ```
   *
   */
  FT_EXPORT( FT_Error )
  FT_Outline_Embolden( FT_Outline*  outline,
                       FT_Pos       strength );


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_EmboldenXY
   *
   * @description:
   *   Embolden an outline.  The new outline will be `xstrength` pixels wider
   *   and `ystrength` pixels higher.  Otherwise, it is similar to
   *   @FT_Outline_Embolden, which uses the same strength in both directions.
   *
   * @since:
   *   2.4.10
   */
  FT_EXPORT( FT_Error )
  FT_Outline_EmboldenXY( FT_Outline*  outline,
                         FT_Pos       xstrength,
                         FT_Pos       ystrength );


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_Reverse
   *
   * @description:
   *   Reverse the drawing direction of an outline.  This is used to ensure
   *   consistent fill conventions for mirrored glyphs.
   *
   * @inout:
   *   outline ::
   *     A pointer to the target outline descriptor.
   *
   * @note:
   *   This function toggles the bit flag @FT_OUTLINE_REVERSE_FILL in the
   *   outline's `flags` field.
   *
   *   It shouldn't be used by a normal client application, unless it knows
   *   what it is doing.
   */
  FT_EXPORT( void )
  FT_Outline_Reverse( FT_Outline*  outline );


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_Get_Bitmap
   *
   * @description:
   *   Render an outline within a bitmap.  The outline's image is simply
   *   OR-ed to the target bitmap.
   *
   * @input:
   *   library ::
   *     A handle to a FreeType library object.
   *
   *   outline ::
   *     A pointer to the source outline descriptor.
   *
   * @inout:
   *   abitmap ::
   *     A pointer to the target bitmap descriptor.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   This function does **not create** the bitmap, it only renders an
   *   outline image within the one you pass to it!  Consequently, the
   *   various fields in `abitmap` should be set accordingly.
   *
   *   It will use the raster corresponding to the default glyph format.
   *
   *   The value of the `num_grays` field in `abitmap` is ignored.  If you
   *   select the gray-level rasterizer, and you want less than 256 gray
   *   levels, you have to use @FT_Outline_Render directly.
   */
  FT_EXPORT( FT_Error )
  FT_Outline_Get_Bitmap( FT_Library        library,
                         FT_Outline*       outline,
                         const FT_Bitmap  *abitmap );


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_Render
   *
   * @description:
   *   Render an outline within a bitmap using the current scan-convert.
   *
   * @input:
   *   library ::
   *     A handle to a FreeType library object.
   *
   *   outline ::
   *     A pointer to the source outline descriptor.
   *
   * @inout:
   *   params ::
   *     A pointer to an @FT_Raster_Params structure used to describe the
   *     rendering operation.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   This advanced function uses @FT_Raster_Params as an argument.
   *   The field `params.source` will be set to `outline` before the scan
   *   converter is called, which means that the value you give to it is
   *   actually ignored.  Either `params.target` must point to preallocated
   *   bitmap, or @FT_RASTER_FLAG_DIRECT must be set in `params.flags`
   *   allowing FreeType rasterizer to be used for direct composition,
   *   translucency, etc.  See @FT_Raster_Params for more details.
   */
  FT_EXPORT( FT_Error )
  FT_Outline_Render( FT_Library         library,
                     FT_Outline*        outline,
                     FT_Raster_Params*  params );


  /**************************************************************************
   *
   * @enum:
   *   FT_Orientation
   *
   * @description:
   *   A list of values used to describe an outline's contour orientation.
   *
   *   The TrueType and PostScript specifications use different conventions
   *   to determine whether outline contours should be filled or unfilled.
   *
   * @values:
   *   FT_ORIENTATION_TRUETYPE ::
   *     According to the TrueType specification, clockwise contours must be
   *     filled, and counter-clockwise ones must be unfilled.
   *
   *   FT_ORIENTATION_POSTSCRIPT ::
   *     According to the PostScript specification, counter-clockwise
   *     contours must be filled, and clockwise ones must be unfilled.
   *
   *   FT_ORIENTATION_FILL_RIGHT ::
   *     This is identical to @FT_ORIENTATION_TRUETYPE, but is used to
   *     remember that in TrueType, everything that is to the right of the
   *     drawing direction of a contour must be filled.
   *
   *   FT_ORIENTATION_FILL_LEFT ::
   *     This is identical to @FT_ORIENTATION_POSTSCRIPT, but is used to
   *     remember that in PostScript, everything that is to the left of the
   *     drawing direction of a contour must be filled.
   *
   *   FT_ORIENTATION_NONE ::
   *     The orientation cannot be determined.  That is, different parts of
   *     the glyph have different orientation.
   *
   */
  typedef enum  FT_Orientation_
  {
    FT_ORIENTATION_TRUETYPE   = 0,
    FT_ORIENTATION_POSTSCRIPT = 1,
    FT_ORIENTATION_FILL_RIGHT = FT_ORIENTATION_TRUETYPE,
    FT_ORIENTATION_FILL_LEFT  = FT_ORIENTATION_POSTSCRIPT,
    FT_ORIENTATION_NONE

  } FT_Orientation;


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_Get_Orientation
   *
   * @description:
   *   This function analyzes a glyph outline and tries to compute its fill
   *   orientation (see @FT_Orientation).  This is done by integrating the
   *   total area covered by the outline. The positive integral corresponds
   *   to the clockwise orientation and @FT_ORIENTATION_POSTSCRIPT is
   *   returned. The negative integral corresponds to the counter-clockwise
   *   orientation and @FT_ORIENTATION_TRUETYPE is returned.
   *
   *   Note that this will return @FT_ORIENTATION_TRUETYPE for empty
   *   outlines.
   *
   * @input:
   *   outline ::
   *     A handle to the source outline.
   *
   * @return:
   *   The orientation.
   *
   */
  FT_EXPORT( FT_Orientation )
  FT_Outline_Get_Orientation( FT_Outline*  outline );


  /* */


FT_END_HEADER

#endif /* FTOUTLN_H_ */


/* END */


/* Local Variables: */
/* coding: utf-8    */
/* End:             */

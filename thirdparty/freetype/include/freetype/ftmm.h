/****************************************************************************
 *
 * ftmm.h
 *
 *   FreeType variation font interface (specification).
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


#ifndef FTMM_H_
#define FTMM_H_

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
   *   multiple_masters
   *
   * @title:
   *   OpenType Font Variations, TrueType GX, and Adobe MM Fonts
   *
   * @abstract:
   *   How to manage variable fonts with multiple design axes.
   *
   * @description:
   *   The following types and functions manage OpenType Font Variations,
   *   Adobe Multiple Master (MM) fonts, and Apple TrueType GX fonts.  These
   *   formats have in common that they allow the selection of specific
   *   design instances by setting design coordinates for one or more axes
   *   like font weight or width.
   *
   *   For historical reasons there are two interfaces.  The first, older one
   *   can be used with Adobe MM fonts only, and the second, newer one is a
   *   unified interface that handles all three font formats.  However, some
   *   differences remain and are documented accordingly; in particular,
   *   Adobe MM fonts don't have named instances (see below).
   *
   *   For Adobe MM fonts, macro @FT_IS_SFNT returns false.  For TrueType GX
   *   and OpenType Font Variations, it returns true.
   *
   *   We use mostly the terminology of the OpenType standard.  Here are some
   *   important technical terms.
   *
   *   * A 'named instance' is a tuple of design coordinates that has a
   *     string ID (i.e., an index into the font's 'name' table) associated
   *     with it.  The font can tell the user that, for example,
   *     [Weight=700,Width=110] is 'Bold'.  Another name for 'named instance'
   *     is 'named style'.
   *
   *       Adobe MM fonts don't have named instances.
   *
   *   * The 'default instance' of a variation font is that instance for
   *     which the nth axis coordinate is equal to the nth default axis
   *     coordinate (i.e., `axis[n].def` as specified in the @FT_MM_Var
   *     structure), with~n covering all axes.  In TrueType GX and OpenType
   *     Font Variations, the default instance is explicitly given.  In Adobe
   *     MM fonts, the `WeightVector` entry as found in the font file is
   *     taken as the default instance.
   *
   *       For TrueType GX and OpenType Font Variations, FreeType synthesizes
   *       a named instance for the default instance if the font does not
   *       contain such an entry.
   *
   *   * 'Design coordinates' are the axis values found in a variation font
   *      file.  Their meaning is specified by the font designer and the
   *      values are rather arbitrary.
   *
   *       For example, the 'weight' axis in design coordinates might vary
   *       between 100 (thin) and 900 (heavy) in font~A, while font~B
   *       contains values between 400 (normal) and 800 (extra bold).
   *
   *   * 'Normalized coordinates' are design coordinates mapped to a standard
   *     range; they are also called 'blend coordinates'.
   *
   *       For TrueType GX and OpenType Font Variations, the range is [-1;1],
   *       with the minimum mapped to value~-1, the default mapped to
   *       value~0, and the maximum mapped to value~1, and all other
   *       coordinates mapped to intervening points.  Please look up the
   *       [OpenType
   *       specification](https://learn.microsoft.com/en-us/typography/opentype/spec/otvaroverview)
   *       on how this mapping works in detail.
   *
   *       For Adobe MM fonts, this standard range is [0;1], with the minimum
   *       mapped to value~0 and the maximum mapped to value~1, and all other
   *       coordinates mapped to intervening points.  Please look up [Adobe
   *       TechNote
   *       #5015](https://adobe-type-tools.github.io/font-tech-notes/pdfs/5015.Type1_Supp.pdf)
   *       on how this mapping works in detail.
   *
   *       Assuming that the two fonts in the previous example are OpenType
   *       Font Variations, both font~A's [100;900] and font~B's [400;800]
   *       coordinate ranges get mapped to [-1;1].
   */


  /**************************************************************************
   *
   * @enum:
   *   T1_MAX_MM_XXX
   *
   * @description:
   *   Adobe MM font limits as defined in their specifications.
   *
   * @values:
   *   T1_MAX_MM_AXIS ::
   *     The maximum number of Adobe MM font axes.
   *
   *   T1_MAX_MM_DESIGNS ::
   *     The maximum number of Adobe MM font designs.
   *
   *   T1_MAX_MM_MAP_POINTS ::
   *     The maximum number of elements in a design map.
   *
   */
#define T1_MAX_MM_AXIS         4
#define T1_MAX_MM_DESIGNS     16
#define T1_MAX_MM_MAP_POINTS  20


  /**************************************************************************
   *
   * @struct:
   *   FT_MM_Axis
   *
   * @description:
   *   A structure to model a given axis in design space for Adobe MM fonts.
   *
   *   This structure can't be used with TrueType GX or OpenType Font
   *   Variations.
   *
   * @fields:
   *   name ::
   *     The axis's name.
   *
   *   minimum ::
   *     The axis's minimum design coordinate.
   *
   *   maximum ::
   *     The axis's maximum design coordinate.
   */
  typedef struct  FT_MM_Axis_
  {
    FT_String*  name;
    FT_Long     minimum;
    FT_Long     maximum;

  } FT_MM_Axis;


  /**************************************************************************
   *
   * @struct:
   *   FT_Multi_Master
   *
   * @description:
   *   A structure to model the axes and space of an Adobe MM font.
   *
   *   This structure can't be used with TrueType GX or OpenType Font
   *   Variations.
   *
   * @fields:
   *   num_axis ::
   *     Number of axes.  Cannot exceed~4.
   *
   *   num_designs ::
   *     Number of designs; should be normally `2^num_axis` even though the
   *     Type~1 specification strangely allows for intermediate designs to be
   *     present.  This number cannot exceed~16.
   *
   *   axis ::
   *     A table of axis descriptors.
   */
  typedef struct  FT_Multi_Master_
  {
    FT_UInt     num_axis;
    FT_UInt     num_designs;
    FT_MM_Axis  axis[T1_MAX_MM_AXIS];

  } FT_Multi_Master;


  /**************************************************************************
   *
   * @struct:
   *   FT_Var_Axis
   *
   * @description:
   *   A structure to model a given axis in design space for Adobe MM fonts,
   *   TrueType GX, and OpenType Font Variations.
   *
   * @fields:
   *   name ::
   *     The axis's name.  Not always meaningful for TrueType GX or OpenType
   *     Font Variations.
   *
   *   minimum ::
   *     The axis's minimum design coordinate.
   *
   *   def ::
   *     The axis's default design coordinate.  FreeType computes meaningful
   *     default values for Adobe MM fonts.
   *
   *   maximum ::
   *     The axis's maximum design coordinate.
   *
   *   tag ::
   *     The axis's tag (the equivalent to 'name' for TrueType GX and
   *     OpenType Font Variations).  FreeType provides default values for
   *     Adobe MM fonts if possible.
   *
   *   strid ::
   *     The axis name entry in the font's 'name' table.  This is another
   *     (and often better) version of the 'name' field for TrueType GX or
   *     OpenType Font Variations.  Not meaningful for Adobe MM fonts.
   *
   * @note:
   *   The fields `minimum`, `def`, and `maximum` are 16.16 fractional values
   *   for TrueType GX and OpenType Font Variations.  For Adobe MM fonts, the
   *   values are whole numbers (i.e., the fractional part is zero).
   */
  typedef struct  FT_Var_Axis_
  {
    FT_String*  name;

    FT_Fixed    minimum;
    FT_Fixed    def;
    FT_Fixed    maximum;

    FT_ULong    tag;
    FT_UInt     strid;

  } FT_Var_Axis;


  /**************************************************************************
   *
   * @struct:
   *   FT_Var_Named_Style
   *
   * @description:
   *   A structure to model a named instance in a TrueType GX or OpenType
   *   Font Variations.
   *
   *   This structure can't be used for Adobe MM fonts.
   *
   * @fields:
   *   coords ::
   *     The design coordinates for this instance.  This is an array with one
   *     entry for each axis.
   *
   *   strid ::
   *     An index into the 'name' table identifying this instance.
   *
   *   psid ::
   *     An index into the 'name' table identifying a PostScript name for
   *     this instance.  Value 0xFFFF indicates a missing entry.
   */
  typedef struct  FT_Var_Named_Style_
  {
    FT_Fixed*  coords;
    FT_UInt    strid;
    FT_UInt    psid;   /* since 2.7.1 */

  } FT_Var_Named_Style;


  /**************************************************************************
   *
   * @struct:
   *   FT_MM_Var
   *
   * @description:
   *   A structure to model the axes and space of Adobe MM fonts, TrueType
   *   GX, or OpenType Font Variations.
   *
   *   Some fields are specific to one format and not to the others.
   *
   * @fields:
   *   num_axis ::
   *     The number of axes.  The maximum value is~4 for Adobe MM fonts; no
   *     limit in TrueType GX or OpenType Font Variations.
   *
   *   num_designs ::
   *     The number of designs; should be normally `2^num_axis` for Adobe MM
   *     fonts.  Not meaningful for TrueType GX or OpenType Font Variations
   *     (where every glyph could have a different number of designs).
   *
   *   num_namedstyles ::
   *     The number of named instances.  For Adobe MM fonts, this value is
   *     always zero.
   *
   *   axis ::
   *     An axis descriptor table.  TrueType GX and OpenType Font Variations
   *     contain slightly more data than Adobe MM fonts.  Memory management
   *     of this pointer is done internally by FreeType.
   *
   *   namedstyle ::
   *     An array of named instances.  Only meaningful for TrueType GX and
   *     OpenType Font Variations.  Memory management of this pointer is done
   *     internally by FreeType.
   */
  typedef struct  FT_MM_Var_
  {
    FT_UInt              num_axis;
    FT_UInt              num_designs;
    FT_UInt              num_namedstyles;
    FT_Var_Axis*         axis;
    FT_Var_Named_Style*  namedstyle;

  } FT_MM_Var;


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Multi_Master
   *
   * @description:
   *   Retrieve a variation descriptor of a given Adobe MM font.
   *
   *   This function can't be used with TrueType GX or OpenType Font
   *   Variations.
   *
   * @input:
   *   face ::
   *     A handle to the source face.
   *
   * @output:
   *   amaster ::
   *     The Adobe MM font's variation descriptor.
   *
   * @return:
   *   FreeType error code.  0~means success.
   */
  FT_EXPORT( FT_Error )
  FT_Get_Multi_Master( FT_Face           face,
                       FT_Multi_Master  *amaster );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_MM_Var
   *
   * @description:
   *   Retrieve a variation descriptor for a given font.
   *
   *   This function works with all supported variation formats.
   *
   * @input:
   *   face ::
   *     A handle to the source face.
   *
   * @output:
   *   amaster ::
   *     The variation descriptor.  Allocates a data structure, which the
   *     user must deallocate with a call to @FT_Done_MM_Var after use.
   *
   * @return:
   *   FreeType error code.  0~means success.
   */
  FT_EXPORT( FT_Error )
  FT_Get_MM_Var( FT_Face      face,
                 FT_MM_Var*  *amaster );


  /**************************************************************************
   *
   * @function:
   *   FT_Done_MM_Var
   *
   * @description:
   *   Free the memory allocated by @FT_Get_MM_Var.
   *
   * @input:
   *   library ::
   *     A handle of the face's parent library object that was used in the
   *     call to @FT_Get_MM_Var to create `amaster`.
   *
   * @return:
   *   FreeType error code.  0~means success.
   */
  FT_EXPORT( FT_Error )
  FT_Done_MM_Var( FT_Library   library,
                  FT_MM_Var   *amaster );


  /**************************************************************************
   *
   * @function:
   *   FT_Set_MM_Design_Coordinates
   *
   * @description:
   *   For Adobe MM fonts, choose an interpolated font design through design
   *   coordinates.
   *
   *   This function can't be used with TrueType GX or OpenType Font
   *   Variations.
   *
   * @inout:
   *   face ::
   *     A handle to the source face.
   *
   * @input:
   *   num_coords ::
   *     The number of available design coordinates.  If it is larger than
   *     the number of axes, ignore the excess values.  If it is smaller than
   *     the number of axes, use default values for the remaining axes.
   *
   *   coords ::
   *     An array of design coordinates.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   [Since 2.8.1] To reset all axes to the default values, call the
   *   function with `num_coords` set to zero and `coords` set to `NULL`.
   *
   *   [Since 2.9] If `num_coords` is larger than zero, this function sets
   *   the @FT_FACE_FLAG_VARIATION bit in @FT_Face's `face_flags` field
   *   (i.e., @FT_IS_VARIATION returns true).  If `num_coords` is zero, this
   *   bit flag gets unset.
   */
  FT_EXPORT( FT_Error )
  FT_Set_MM_Design_Coordinates( FT_Face   face,
                                FT_UInt   num_coords,
                                FT_Long*  coords );


  /**************************************************************************
   *
   * @function:
   *   FT_Set_Var_Design_Coordinates
   *
   * @description:
   *   Choose an interpolated font design through design coordinates.
   *
   *   This function works with all supported variation formats.
   *
   * @inout:
   *   face ::
   *     A handle to the source face.
   *
   * @input:
   *   num_coords ::
   *     The number of available design coordinates.  If it is larger than
   *     the number of axes, ignore the excess values.  If it is smaller than
   *     the number of axes, use default values for the remaining axes.
   *
   *   coords ::
   *     An array of design coordinates.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   The design coordinates are 16.16 fractional values for TrueType GX and
   *   OpenType Font Variations.  For Adobe MM fonts, the values are supposed
   *   to be whole numbers (i.e., the fractional part is zero).
   *
   *   [Since 2.8.1] To reset all axes to the default values, call the
   *   function with `num_coords` set to zero and `coords` set to `NULL`.
   *   [Since 2.9] 'Default values' means the currently selected named
   *   instance (or the base font if no named instance is selected).
   *
   *   [Since 2.9] If `num_coords` is larger than zero, this function sets
   *   the @FT_FACE_FLAG_VARIATION bit in @FT_Face's `face_flags` field
   *   (i.e., @FT_IS_VARIATION returns true).  If `num_coords` is zero, this
   *   bit flag gets unset.
   *
   *   [Since 2.14] This function also sets the @FT_FACE_FLAG_VARIATION bit
   *   in @FT_Face's `face_flags` field (i.e., @FT_IS_VARIATION returns
   *   true) if any of the provided coordinates is different from the face's
   *   default value for the corresponding axis, that is, the set up face is
   *   not at its default position.
   */
  FT_EXPORT( FT_Error )
  FT_Set_Var_Design_Coordinates( FT_Face    face,
                                 FT_UInt    num_coords,
                                 FT_Fixed*  coords );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Var_Design_Coordinates
   *
   * @description:
   *   Get the design coordinates of the currently selected interpolated
   *   font.
   *
   *   This function works with all supported variation formats.
   *
   * @input:
   *   face ::
   *     A handle to the source face.
   *
   *   num_coords ::
   *     The number of design coordinates to retrieve.  If it is larger than
   *     the number of axes, set the excess values to~0.
   *
   * @output:
   *   coords ::
   *     The design coordinates array, which must be allocated by the user.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   The design coordinates are 16.16 fractional values for TrueType GX and
   *   OpenType Font Variations.  For Adobe MM fonts, the values are whole
   *   numbers (i.e., the fractional part is zero).
   *
   * @since:
   *   2.7.1
   */
  FT_EXPORT( FT_Error )
  FT_Get_Var_Design_Coordinates( FT_Face    face,
                                 FT_UInt    num_coords,
                                 FT_Fixed*  coords );


  /**************************************************************************
   *
   * @function:
   *   FT_Set_MM_Blend_Coordinates
   *
   * @description:
   *   Choose an interpolated font design through normalized coordinates.
   *
   *   This function works with all supported variation formats.
   *
   * @inout:
   *   face ::
   *     A handle to the source face.
   *
   * @input:
   *   num_coords ::
   *     The number of available design coordinates.  If it is larger than
   *     the number of axes, ignore the excess values.  If it is smaller than
   *     the number of axes, use default values for the remaining axes.
   *
   *   coords ::
   *     The normalized coordinates array.  Each element is a 16.16
   *     fractional value and must be between 0 and 1.0 for Adobe MM fonts,
   *     and between -1.0 and 1.0 for TrueType GX and OpenType Font
   *     Variations.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   [Since 2.8.1] To reset all axes to the default values, call the
   *   function with `num_coords` set to zero and `coords` set to `NULL`.
   *   [Since 2.9] 'Default values' means the currently selected named
   *   instance (or the base font if no named instance is selected).
   *
   *   [Since 2.9] If `num_coords` is larger than zero, this function sets
   *   the @FT_FACE_FLAG_VARIATION bit in @FT_Face's `face_flags` field
   *   (i.e., @FT_IS_VARIATION returns true).  If `num_coords` is zero, this
   *   bit flag gets unset.
   *
   *   [Since 2.14] This function also sets the @FT_FACE_FLAG_VARIATION bit
   *   in @FT_Face's `face_flags` field (i.e., @FT_IS_VARIATION returns
   *   true) if any of the provided coordinates is different from the face's
   *   default value for the corresponding axis, that is, the set up face is
   *   not at its default position.
   */
  FT_EXPORT( FT_Error )
  FT_Set_MM_Blend_Coordinates( FT_Face    face,
                               FT_UInt    num_coords,
                               FT_Fixed*  coords );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_MM_Blend_Coordinates
   *
   * @description:
   *   Get the normalized coordinates of the currently selected interpolated
   *   font.
   *
   *   This function works with all supported variation formats.
   *
   * @input:
   *   face ::
   *     A handle to the source face.
   *
   *   num_coords ::
   *     The number of normalized coordinates to retrieve.  If it is larger
   *     than the number of axes, set the excess values to~0.5 for Adobe MM
   *     fonts, and to~0 for TrueType GX and OpenType Font Variations.
   *
   * @output:
   *   coords ::
   *     The normalized coordinates array (as 16.16 fractional values), which
   *     must be allocated by the user.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @since:
   *   2.7.1
   */
  FT_EXPORT( FT_Error )
  FT_Get_MM_Blend_Coordinates( FT_Face    face,
                               FT_UInt    num_coords,
                               FT_Fixed*  coords );


  /**************************************************************************
   *
   * @function:
   *   FT_Set_Var_Blend_Coordinates
   *
   * @description:
   *   This is another name of @FT_Set_MM_Blend_Coordinates.
   */
  FT_EXPORT( FT_Error )
  FT_Set_Var_Blend_Coordinates( FT_Face    face,
                                FT_UInt    num_coords,
                                FT_Fixed*  coords );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Var_Blend_Coordinates
   *
   * @description:
   *   This is another name of @FT_Get_MM_Blend_Coordinates.
   *
   * @since:
   *   2.7.1
   */
  FT_EXPORT( FT_Error )
  FT_Get_Var_Blend_Coordinates( FT_Face    face,
                                FT_UInt    num_coords,
                                FT_Fixed*  coords );


  /**************************************************************************
   *
   * @function:
   *   FT_Set_MM_WeightVector
   *
   * @description:
   *   For Adobe MM fonts, choose an interpolated font design by directly
   *   setting the weight vector.
   *
   *   This function can't be used with TrueType GX or OpenType Font
   *   Variations.
   *
   * @inout:
   *   face ::
   *     A handle to the source face.
   *
   * @input:
   *   len ::
   *     The length of the weight vector array.  If it is larger than the
   *     number of designs, the extra values are ignored.  If it is less than
   *     the number of designs, the remaining values are set to zero.
   *
   *   weightvector ::
   *     An array representing the weight vector.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   Adobe MM fonts limit the number of designs, and thus the length of the
   *   weight vector, to 16~elements.
   *
   *   If `len` is larger than zero, this function sets the
   *   @FT_FACE_FLAG_VARIATION bit in @FT_Face's `face_flags` field (i.e.,
   *   @FT_IS_VARIATION returns true).  If `len` is zero, this bit flag is
   *   unset and the weight vector array is reset to the default values.
   *
   *   The Adobe documentation also states that the values in the
   *   `WeightVector` array must total 1.0 +/-~0.001.  In practice this does
   *   not seem to be enforced, so is not enforced here, either.
   *
   * @since:
   *   2.10
   */
  FT_EXPORT( FT_Error )
  FT_Set_MM_WeightVector( FT_Face    face,
                          FT_UInt    len,
                          FT_Fixed*  weightvector );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_MM_WeightVector
   *
   * @description:
   *   For Adobe MM fonts, retrieve the current weight vector of the font.
   *
   *   This function can't be used with TrueType GX or OpenType Font
   *   Variations.
   *
   * @inout:
   *   face ::
   *     A handle to the source face.
   *
   *   len ::
   *     A pointer to the size of the array to be filled.  If the size of the
   *     array is less than the number of designs, `FT_Err_Invalid_Argument`
   *     is returned, and `len` is set to the required size (the number of
   *     designs).  If the size of the array is greater than the number of
   *     designs, the remaining entries are set to~0.  On successful
   *     completion, `len` is set to the number of designs (i.e., the number
   *     of values written to the array).
   *
   * @output:
   *   weightvector ::
   *     An array to be filled; it must be allocated by the user.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   Adobe MM fonts limit the number of designs, and thus the length of the
   *   weight vector, to~16 elements.
   *
   * @since:
   *   2.10
   */
  FT_EXPORT( FT_Error )
  FT_Get_MM_WeightVector( FT_Face    face,
                          FT_UInt*   len,
                          FT_Fixed*  weightvector );


  /**************************************************************************
   *
   * @enum:
   *   FT_VAR_AXIS_FLAG_XXX
   *
   * @description:
   *   A list of bit flags used in the return value of
   *   @FT_Get_Var_Axis_Flags.
   *
   * @values:
   *   FT_VAR_AXIS_FLAG_HIDDEN ::
   *     The variation axis should not be exposed to user interfaces.
   *
   * @since:
   *   2.8.1
   */
#define FT_VAR_AXIS_FLAG_HIDDEN  1


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Var_Axis_Flags
   *
   * @description:
   *   Get the 'flags' field of an OpenType Variation Axis Record.
   *
   *   Not meaningful for Adobe MM fonts (`*flags` is always zero).
   *
   * @input:
   *   master ::
   *     The variation descriptor.
   *
   *   axis_index ::
   *     The index of the requested variation axis.
   *
   * @output:
   *   flags ::
   *     The 'flags' field.  See @FT_VAR_AXIS_FLAG_XXX for possible values.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @since:
   *   2.8.1
   */
  FT_EXPORT( FT_Error )
  FT_Get_Var_Axis_Flags( FT_MM_Var*  master,
                         FT_UInt     axis_index,
                         FT_UInt*    flags );


  /**************************************************************************
   *
   * @function:
   *   FT_Set_Named_Instance
   *
   * @description:
   *   Set or change the current named instance.
   *
   * @input:
   *   face ::
   *     A handle to the source face.
   *
   *   instance_index ::
   *     The index of the requested instance, starting with value~1.  If set
   *     to value~0, FreeType switches to font access without a named
   *     instance.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   The function uses the value of `instance_index` to set bits 16-30 of
   *   the face's `face_index` field.  It also resets any variation applied
   *   to the font, and the @FT_FACE_FLAG_VARIATION bit of the face's
   *   `face_flags` field gets reset to zero (i.e., @FT_IS_VARIATION returns
   *   false).
   *
   *   For Adobe MM fonts, this function resets the current face to the
   *   default instance.
   *
   * @since:
   *   2.9
   */
  FT_EXPORT( FT_Error )
  FT_Set_Named_Instance( FT_Face  face,
                         FT_UInt  instance_index );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Default_Named_Instance
   *
   * @description:
   *   Retrieve the index of the default named instance, to be used with
   *   @FT_Set_Named_Instance.
   *
   *   FreeType synthesizes a named instance for the default instance if the
   *   font does not contain such an entry.
   *
   * @input:
   *   face ::
   *     A handle to the source face.
   *
   * @output:
   *   instance_index ::
   *     The index of the default named instance.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   For Adobe MM fonts, this function always returns zero for
   *   `instance_index`.
   *
   * @since:
   *   2.13.1
   */
  FT_EXPORT( FT_Error )
  FT_Get_Default_Named_Instance( FT_Face   face,
                                 FT_UInt  *instance_index );

  /* */


FT_END_HEADER

#endif /* FTMM_H_ */


/* END */

/***************************************************************************/
/*                                                                         */
/*  ftmm.h                                                                 */
/*                                                                         */
/*    FreeType Multiple Master font interface (specification).             */
/*                                                                         */
/*  Copyright 1996-2017 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef FTMM_H_
#define FTMM_H_


#include <ft2build.h>
#include FT_TYPE1_TABLES_H


FT_BEGIN_HEADER


  /*************************************************************************/
  /*                                                                       */
  /* <Section>                                                             */
  /*    multiple_masters                                                   */
  /*                                                                       */
  /* <Title>                                                               */
  /*    Multiple Masters                                                   */
  /*                                                                       */
  /* <Abstract>                                                            */
  /*    How to manage Multiple Masters fonts.                              */
  /*                                                                       */
  /* <Description>                                                         */
  /*    The following types and functions are used to manage Multiple      */
  /*    Master fonts, i.e., the selection of specific design instances by  */
  /*    setting design axis coordinates.                                   */
  /*                                                                       */
  /*    Besides Adobe MM fonts, the interface supports Apple's TrueType GX */
  /*    and OpenType variation fonts.  Some of the routines only work with */
  /*    Adobe MM fonts, others will work with all three types.  They are   */
  /*    similar enough that a consistent interface makes sense.            */
  /*                                                                       */
  /*************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_MM_Axis                                                         */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A structure to model a given axis in design space for Multiple     */
  /*    Masters fonts.                                                     */
  /*                                                                       */
  /*    This structure can't be used for TrueType GX or OpenType variation */
  /*    fonts.                                                             */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    name    :: The axis's name.                                        */
  /*                                                                       */
  /*    minimum :: The axis's minimum design coordinate.                   */
  /*                                                                       */
  /*    maximum :: The axis's maximum design coordinate.                   */
  /*                                                                       */
  typedef struct  FT_MM_Axis_
  {
    FT_String*  name;
    FT_Long     minimum;
    FT_Long     maximum;

  } FT_MM_Axis;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_Multi_Master                                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A structure to model the axes and space of a Multiple Masters      */
  /*    font.                                                              */
  /*                                                                       */
  /*    This structure can't be used for TrueType GX or OpenType variation */
  /*    fonts.                                                             */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    num_axis    :: Number of axes.  Cannot exceed~4.                   */
  /*                                                                       */
  /*    num_designs :: Number of designs; should be normally 2^num_axis    */
  /*                   even though the Type~1 specification strangely      */
  /*                   allows for intermediate designs to be present.      */
  /*                   This number cannot exceed~16.                       */
  /*                                                                       */
  /*    axis        :: A table of axis descriptors.                        */
  /*                                                                       */
  typedef struct  FT_Multi_Master_
  {
    FT_UInt     num_axis;
    FT_UInt     num_designs;
    FT_MM_Axis  axis[T1_MAX_MM_AXIS];

  } FT_Multi_Master;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_Var_Axis                                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A structure to model a given axis in design space for Multiple     */
  /*    Masters, TrueType GX, and OpenType variation fonts.                */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    name    :: The axis's name.                                        */
  /*               Not always meaningful for TrueType GX or OpenType       */
  /*               variation fonts.                                        */
  /*                                                                       */
  /*    minimum :: The axis's minimum design coordinate.                   */
  /*                                                                       */
  /*    def     :: The axis's default design coordinate.                   */
  /*               FreeType computes meaningful default values for Adobe   */
  /*               MM fonts.                                               */
  /*                                                                       */
  /*    maximum :: The axis's maximum design coordinate.                   */
  /*                                                                       */
  /*    tag     :: The axis's tag (the equivalent to `name' for TrueType   */
  /*               GX and OpenType variation fonts).  FreeType provides    */
  /*               default values for Adobe MM fonts if possible.          */
  /*                                                                       */
  /*    strid   :: The axis name entry in the font's `name' table.  This   */
  /*               is another (and often better) version of the `name'     */
  /*               field for TrueType GX or OpenType variation fonts.  Not */
  /*               meaningful for Adobe MM fonts.                          */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The fields `minimum', `def', and `maximum' are 16.16 fractional    */
  /*    values for TrueType GX and OpenType variation fonts.  For Adobe MM */
  /*    fonts, the values are integers.                                    */
  /*                                                                       */
  typedef struct  FT_Var_Axis_
  {
    FT_String*  name;

    FT_Fixed    minimum;
    FT_Fixed    def;
    FT_Fixed    maximum;

    FT_ULong    tag;
    FT_UInt     strid;

  } FT_Var_Axis;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_Var_Named_Style                                                 */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A structure to model a named instance in a TrueType GX or OpenType */
  /*    variation font.                                                    */
  /*                                                                       */
  /*    This structure can't be used for Adobe MM fonts.                   */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    coords :: The design coordinates for this instance.                */
  /*              This is an array with one entry for each axis.           */
  /*                                                                       */
  /*    strid  :: The entry in `name' table identifying this instance.     */
  /*                                                                       */
  /*    psid   :: The entry in `name' table identifying a PostScript name  */
  /*              for this instance.                                       */
  /*                                                                       */
  typedef struct  FT_Var_Named_Style_
  {
    FT_Fixed*  coords;
    FT_UInt    strid;
    FT_UInt    psid;   /* since 2.7.1 */

  } FT_Var_Named_Style;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_MM_Var                                                          */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A structure to model the axes and space of a Adobe MM, TrueType    */
  /*    GX, or OpenType variation font.                                    */
  /*                                                                       */
  /*    Some fields are specific to one format and not to the others.      */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    num_axis        :: The number of axes.  The maximum value is~4 for */
  /*                       Adobe MM fonts; no limit in TrueType GX or      */
  /*                       OpenType variation fonts.                       */
  /*                                                                       */
  /*    num_designs     :: The number of designs; should be normally       */
  /*                       2^num_axis for Adobe MM fonts.  Not meaningful  */
  /*                       for TrueType GX or OpenType variation fonts     */
  /*                       (where every glyph could have a different       */
  /*                       number of designs).                             */
  /*                                                                       */
  /*    num_namedstyles :: The number of named styles; a `named style' is  */
  /*                       a tuple of design coordinates that has a string */
  /*                       ID (in the `name' table) associated with it.    */
  /*                       The font can tell the user that, for example,   */
  /*                       [Weight=1.5,Width=1.1] is `Bold'.  Another name */
  /*                       for `named style' is `named instance'.          */
  /*                                                                       */
  /*                       For Adobe Multiple Masters fonts, this value is */
  /*                       always zero because the format does not support */
  /*                       named styles.                                   */
  /*                                                                       */
  /*    axis            :: An axis descriptor table.                       */
  /*                       TrueType GX and OpenType variation fonts        */
  /*                       contain slightly more data than Adobe MM fonts. */
  /*                       Memory management of this pointer is done       */
  /*                       internally by FreeType.                         */
  /*                                                                       */
  /*    namedstyle      :: A named style (instance) table.                 */
  /*                       Only meaningful for TrueType GX and OpenType    */
  /*                       variation fonts.  Memory management of this     */
  /*                       pointer is done internally by FreeType.         */
  /*                                                                       */
  typedef struct  FT_MM_Var_
  {
    FT_UInt              num_axis;
    FT_UInt              num_designs;
    FT_UInt              num_namedstyles;
    FT_Var_Axis*         axis;
    FT_Var_Named_Style*  namedstyle;

  } FT_MM_Var;


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Get_Multi_Master                                                */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Retrieve a variation descriptor of a given Adobe MM font.          */
  /*                                                                       */
  /*    This function can't be used with TrueType GX or OpenType variation */
  /*    fonts.                                                             */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face    :: A handle to the source face.                            */
  /*                                                                       */
  /* <Output>                                                              */
  /*    amaster :: The Multiple Masters descriptor.                        */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Get_Multi_Master( FT_Face           face,
                       FT_Multi_Master  *amaster );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Get_MM_Var                                                      */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Retrieve a variation descriptor for a given font.                  */
  /*                                                                       */
  /*    This function works with all supported variation formats.          */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face    :: A handle to the source face.                            */
  /*                                                                       */
  /* <Output>                                                              */
  /*    amaster :: The variation descriptor.                               */
  /*               Allocates a data structure, which the user must         */
  /*               deallocate with `free' after use.                       */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Get_MM_Var( FT_Face      face,
                 FT_MM_Var*  *amaster );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Set_MM_Design_Coordinates                                       */
  /*                                                                       */
  /* <Description>                                                         */
  /*    For Adobe MM fonts, choose an interpolated font design through     */
  /*    design coordinates.                                                */
  /*                                                                       */
  /*    This function can't be used with TrueType GX or OpenType variation */
  /*    fonts.                                                             */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face       :: A handle to the source face.                         */
  /*                                                                       */
  /* <Input>                                                               */
  /*    num_coords :: The number of available design coordinates.  If it   */
  /*                  is larger than the number of axes, ignore the excess */
  /*                  values.  If it is smaller than the number of axes,   */
  /*                  use default values for the remaining axes.           */
  /*                                                                       */
  /*    coords     :: An array of design coordinates.                      */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Set_MM_Design_Coordinates( FT_Face   face,
                                FT_UInt   num_coords,
                                FT_Long*  coords );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Set_Var_Design_Coordinates                                      */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Choose an interpolated font design through design coordinates.     */
  /*                                                                       */
  /*    This function works with all supported variation formats.          */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face       :: A handle to the source face.                         */
  /*                                                                       */
  /* <Input>                                                               */
  /*    num_coords :: The number of available design coordinates.  If it   */
  /*                  is larger than the number of axes, ignore the excess */
  /*                  values.  If it is smaller than the number of axes,   */
  /*                  use default values for the remaining axes.           */
  /*                                                                       */
  /*    coords     :: An array of design coordinates.                      */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Set_Var_Design_Coordinates( FT_Face    face,
                                 FT_UInt    num_coords,
                                 FT_Fixed*  coords );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Get_Var_Design_Coordinates                                      */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Get the design coordinates of the currently selected interpolated  */
  /*    font.                                                              */
  /*                                                                       */
  /*    This function works with all supported variation formats.          */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face       :: A handle to the source face.                         */
  /*                                                                       */
  /*    num_coords :: The number of design coordinates to retrieve.  If it */
  /*                  is larger than the number of axes, set the excess    */
  /*                  values to~0.                                         */
  /*                                                                       */
  /* <Output>                                                              */
  /*    coords     :: The design coordinates array.                        */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Get_Var_Design_Coordinates( FT_Face    face,
                                 FT_UInt    num_coords,
                                 FT_Fixed*  coords );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Set_MM_Blend_Coordinates                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Choose an interpolated font design through normalized blend        */
  /*    coordinates.                                                       */
  /*                                                                       */
  /*    This function works with all supported variation formats.          */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face       :: A handle to the source face.                         */
  /*                                                                       */
  /* <Input>                                                               */
  /*    num_coords :: The number of available design coordinates.  If it   */
  /*                  is larger than the number of axes, ignore the excess */
  /*                  values.  If it is smaller than the number of axes,   */
  /*                  use default values for the remaining axes.           */
  /*                                                                       */
  /*    coords     :: The design coordinates array (each element must be   */
  /*                  between 0 and 1.0 for Adobe MM fonts, and between    */
  /*                  -1.0 and 1.0 for TrueType GX and OpenType variation  */
  /*                  fonts).                                              */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Set_MM_Blend_Coordinates( FT_Face    face,
                               FT_UInt    num_coords,
                               FT_Fixed*  coords );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Get_MM_Blend_Coordinates                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Get the normalized blend coordinates of the currently selected     */
  /*    interpolated font.                                                 */
  /*                                                                       */
  /*    This function works with all supported variation formats.          */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face       :: A handle to the source face.                         */
  /*                                                                       */
  /*    num_coords :: The number of normalized blend coordinates to        */
  /*                  retrieve.  If it is larger than the number of axes,  */
  /*                  set the excess values to~0.5 for Adobe MM fonts, and */
  /*                  to~0 for TrueType GX and OpenType variation fonts.   */
  /*                                                                       */
  /* <Output>                                                              */
  /*    coords     :: The normalized blend coordinates array.              */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Get_MM_Blend_Coordinates( FT_Face    face,
                               FT_UInt    num_coords,
                               FT_Fixed*  coords );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Set_Var_Blend_Coordinates                                       */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This is another name of @FT_Set_MM_Blend_Coordinates.              */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Set_Var_Blend_Coordinates( FT_Face    face,
                                FT_UInt    num_coords,
                                FT_Fixed*  coords );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Get_Var_Blend_Coordinates                                       */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This is another name of @FT_Get_MM_Blend_Coordinates.              */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Get_Var_Blend_Coordinates( FT_Face    face,
                                FT_UInt    num_coords,
                                FT_Fixed*  coords );

  /* */


FT_END_HEADER

#endif /* FTMM_H_ */


/* END */

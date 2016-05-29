/***************************************************************************/
/*                                                                         */
/*  ftmm.h                                                                 */
/*                                                                         */
/*    FreeType Multiple Master font interface (specification).             */
/*                                                                         */
/*  Copyright 1996-2001, 2003, 2004, 2006, 2009 by                         */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __FTMM_H__
#define __FTMM_H__


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
  /*    George Williams has extended this interface to make it work with   */
  /*    both Type~1 Multiple Masters fonts and GX distortable (var)        */
  /*    fonts.  Some of these routines only work with MM fonts, others     */
  /*    will work with both types.  They are similar enough that a         */
  /*    consistent interface makes sense.                                  */
  /*                                                                       */
  /*************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_MM_Axis                                                         */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A simple structure used to model a given axis in design space for  */
  /*    Multiple Masters fonts.                                            */
  /*                                                                       */
  /*    This structure can't be used for GX var fonts.                     */
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
  /*    A structure used to model the axes and space of a Multiple Masters */
  /*    font.                                                              */
  /*                                                                       */
  /*    This structure can't be used for GX var fonts.                     */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    num_axis    :: Number of axes.  Cannot exceed~4.                   */
  /*                                                                       */
  /*    num_designs :: Number of designs; should be normally 2^num_axis    */
  /*                   even though the Type~1 specification strangely      */
  /*                   allows for intermediate designs to be present. This */
  /*                   number cannot exceed~16.                            */
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
  /*    A simple structure used to model a given axis in design space for  */
  /*    Multiple Masters and GX var fonts.                                 */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    name    :: The axis's name.                                        */
  /*               Not always meaningful for GX.                           */
  /*                                                                       */
  /*    minimum :: The axis's minimum design coordinate.                   */
  /*                                                                       */
  /*    def     :: The axis's default design coordinate.                   */
  /*               FreeType computes meaningful default values for MM; it  */
  /*               is then an integer value, not in 16.16 format.          */
  /*                                                                       */
  /*    maximum :: The axis's maximum design coordinate.                   */
  /*                                                                       */
  /*    tag     :: The axis's tag (the GX equivalent to `name').           */
  /*               FreeType provides default values for MM if possible.    */
  /*                                                                       */
  /*    strid   :: The entry in `name' table (another GX version of        */
  /*               `name').                                                */
  /*               Not meaningful for MM.                                  */
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
  /*    A simple structure used to model a named style in a GX var font.   */
  /*                                                                       */
  /*    This structure can't be used for MM fonts.                         */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    coords :: The design coordinates for this style.                   */
  /*              This is an array with one entry for each axis.           */
  /*                                                                       */
  /*    strid  :: The entry in `name' table identifying this style.        */
  /*                                                                       */
  typedef struct  FT_Var_Named_Style_
  {
    FT_Fixed*  coords;
    FT_UInt    strid;

  } FT_Var_Named_Style;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_MM_Var                                                          */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A structure used to model the axes and space of a Multiple Masters */
  /*    or GX var distortable font.                                        */
  /*                                                                       */
  /*    Some fields are specific to one format and not to the other.       */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    num_axis        :: The number of axes.  The maximum value is~4 for */
  /*                       MM; no limit in GX.                             */
  /*                                                                       */
  /*    num_designs     :: The number of designs; should be normally       */
  /*                       2^num_axis for MM fonts.  Not meaningful for GX */
  /*                       (where every glyph could have a different       */
  /*                       number of designs).                             */
  /*                                                                       */
  /*    num_namedstyles :: The number of named styles; only meaningful for */
  /*                       GX which allows certain design coordinates to   */
  /*                       have a string ID (in the `name' table)          */
  /*                       associated with them.  The font can tell the    */
  /*                       user that, for example, Weight=1.5 is `Bold'.   */
  /*                                                                       */
  /*    axis            :: A table of axis descriptors.                    */
  /*                       GX fonts contain slightly more data than MM.    */
  /*                                                                       */
  /*    namedstyles     :: A table of named styles.                        */
  /*                       Only meaningful with GX.                        */
  /*                                                                       */
  typedef struct  FT_MM_Var_
  {
    FT_UInt              num_axis;
    FT_UInt              num_designs;
    FT_UInt              num_namedstyles;
    FT_Var_Axis*         axis;
    FT_Var_Named_Style*  namedstyle;

  } FT_MM_Var;


  /* */


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Get_Multi_Master                                                */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Retrieve the Multiple Master descriptor of a given font.           */
  /*                                                                       */
  /*    This function can't be used with GX fonts.                         */
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
  /*    Retrieve the Multiple Master/GX var descriptor of a given font.    */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face    :: A handle to the source face.                            */
  /*                                                                       */
  /* <Output>                                                              */
  /*    amaster :: The Multiple Masters/GX var descriptor.                 */
  /*               Allocates a data structure, which the user must free    */
  /*               (a single call to FT_FREE will do it).                  */
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
  /*    For Multiple Masters fonts, choose an interpolated font design     */
  /*    through design coordinates.                                        */
  /*                                                                       */
  /*    This function can't be used with GX fonts.                         */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face       :: A handle to the source face.                         */
  /*                                                                       */
  /* <Input>                                                               */
  /*    num_coords :: The number of design coordinates (must be equal to   */
  /*                  the number of axes in the font).                     */
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
  /*    For Multiple Master or GX Var fonts, choose an interpolated font   */
  /*    design through design coordinates.                                 */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face       :: A handle to the source face.                         */
  /*                                                                       */
  /* <Input>                                                               */
  /*    num_coords :: The number of design coordinates (must be equal to   */
  /*                  the number of axes in the font).                     */
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
  /*    FT_Set_MM_Blend_Coordinates                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    For Multiple Masters and GX var fonts, choose an interpolated font */
  /*    design through normalized blend coordinates.                       */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face       :: A handle to the source face.                         */
  /*                                                                       */
  /* <Input>                                                               */
  /*    num_coords :: The number of design coordinates (must be equal to   */
  /*                  the number of axes in the font).                     */
  /*                                                                       */
  /*    coords     :: The design coordinates array (each element must be   */
  /*                  between 0 and 1.0).                                  */
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
  /*    FT_Set_Var_Blend_Coordinates                                       */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This is another name of @FT_Set_MM_Blend_Coordinates.              */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Set_Var_Blend_Coordinates( FT_Face    face,
                                FT_UInt    num_coords,
                                FT_Fixed*  coords );


  /* */


FT_END_HEADER

#endif /* __FTMM_H__ */


/* END */

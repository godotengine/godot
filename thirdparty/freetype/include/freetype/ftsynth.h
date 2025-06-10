/****************************************************************************
 *
 * ftsynth.h
 *
 *   FreeType synthesizing code for emboldening and slanting
 *   (specification).
 *
 * Copyright (C) 2000-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*********                                                       *********/
  /*********        WARNING, THIS IS ALPHA CODE!  THIS API         *********/
  /*********    IS DUE TO CHANGE UNTIL STRICTLY NOTIFIED BY THE    *********/
  /*********            FREETYPE DEVELOPMENT TEAM                  *********/
  /*********                                                       *********/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


  /* Main reason for not lifting the functions in this module to a  */
  /* 'standard' API is that the used parameters for emboldening and */
  /* slanting are not configurable.  Consider the functions as a    */
  /* code resource that should be copied into the application and   */
  /* adapted to the particular needs.                               */


#ifndef FTSYNTH_H_
#define FTSYNTH_H_


#include <freetype/freetype.h>

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER

  /* Embolden a glyph by a 'reasonable' value (which is highly a matter of */
  /* taste).  This function is actually a convenience function, providing  */
  /* a wrapper for @FT_Outline_Embolden and @FT_Bitmap_Embolden.           */
  /*                                                                       */
  /* For emboldened outlines the height, width, and advance metrics are    */
  /* increased by the strength of the emboldening -- this even affects     */
  /* mono-width fonts!                                                     */
  /*                                                                       */
  /* You can also call @FT_Outline_Get_CBox to get precise values.         */
  FT_EXPORT( void )
  FT_GlyphSlot_Embolden( FT_GlyphSlot  slot );

  /* Precisely adjust the glyph weight either horizontally or vertically.  */
  /* The `xdelta` and `ydelta` values are fractions of the face Em size    */
  /* (in fixed-point format).  Considering that a regular face would have  */
  /* stem widths on the order of 0.1 Em, a delta of 0.05 (0x0CCC) should   */
  /* be very noticeable.  To increase or decrease the weight, use positive */
  /* or negative values, respectively.                                     */
  FT_EXPORT( void )
  FT_GlyphSlot_AdjustWeight( FT_GlyphSlot  slot,
                             FT_Fixed      xdelta,
                             FT_Fixed      ydelta );


  /* Slant an outline glyph to the right by about 12 degrees.              */
  FT_EXPORT( void )
  FT_GlyphSlot_Oblique( FT_GlyphSlot  slot );

  /* Slant an outline glyph by a given sine of an angle.  You can apply    */
  /* slant along either x- or y-axis by choosing a corresponding non-zero  */
  /* argument.  If both slants are non-zero, some affine transformation    */
  /* will result.                                                          */
  FT_EXPORT( void )
  FT_GlyphSlot_Slant( FT_GlyphSlot  slot,
                      FT_Fixed      xslant,
                      FT_Fixed      yslant );

  /* */


FT_END_HEADER

#endif /* FTSYNTH_H_ */


/* END */

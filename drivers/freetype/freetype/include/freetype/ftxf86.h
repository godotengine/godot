/***************************************************************************/
/*                                                                         */
/*  ftxf86.h                                                               */
/*                                                                         */
/*    Support functions for X11.                                           */
/*                                                                         */
/*  Copyright 2002, 2003, 2004, 2006, 2007 by                              */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __FTXF86_H__
#define __FTXF86_H__

#include <ft2build.h>
#include FT_FREETYPE_H

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER


  /*************************************************************************/
  /*                                                                       */
  /* <Section>                                                             */
  /*   font_formats                                                        */
  /*                                                                       */
  /* <Title>                                                               */
  /*   Font Formats                                                        */
  /*                                                                       */
  /* <Abstract>                                                            */
  /*   Getting the font format.                                            */
  /*                                                                       */
  /* <Description>                                                         */
  /*   The single function in this section can be used to get the font     */
  /*   format.  Note that this information is not needed normally;         */
  /*   however, there are special cases (like in PDF devices) where it is  */
  /*   important to differentiate, in spite of FreeType's uniform API.     */
  /*                                                                       */
  /*   This function is in the X11/xf86 namespace for historical reasons   */
  /*   and in no way depends on that windowing system.                     */
  /*                                                                       */
  /*************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*   FT_Get_X11_Font_Format                                              */
  /*                                                                       */
  /* <Description>                                                         */
  /*   Return a string describing the format of a given face, using values */
  /*   which can be used as an X11 FONT_PROPERTY.  Possible values are     */
  /*   `TrueType', `Type~1', `BDF', `PCF', `Type~42', `CID~Type~1', `CFF', */
  /*   `PFR', and `Windows~FNT'.                                           */
  /*                                                                       */
  /* <Input>                                                               */
  /*   face ::                                                             */
  /*     Input face handle.                                                */
  /*                                                                       */
  /* <Return>                                                              */
  /*   Font format string.  NULL in case of error.                         */
  /*                                                                       */
  FT_EXPORT( const char* )
  FT_Get_X11_Font_Format( FT_Face  face );

 /* */

FT_END_HEADER

#endif /* __FTXF86_H__ */

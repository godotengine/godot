/***************************************************************************/
/*                                                                         */
/*  cffobjs.h                                                              */
/*                                                                         */
/*    OpenType objects manager (specification).                            */
/*                                                                         */
/*  Copyright 1996-2004, 2006-2008, 2013 by                                */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __CFFOBJS_H__
#define __CFFOBJS_H__


#include <ft2build.h>
#include FT_INTERNAL_OBJECTS_H
#include "cfftypes.h"
#include FT_INTERNAL_TRUETYPE_TYPES_H
#include FT_SERVICE_POSTSCRIPT_CMAPS_H
#include FT_INTERNAL_POSTSCRIPT_HINTS_H


FT_BEGIN_HEADER


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    CFF_Driver                                                         */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A handle to an OpenType driver object.                             */
  /*                                                                       */
  typedef struct CFF_DriverRec_*  CFF_Driver;

  typedef TT_Face  CFF_Face;


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    CFF_Size                                                           */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A handle to an OpenType size object.                               */
  /*                                                                       */
  typedef struct  CFF_SizeRec_
  {
    FT_SizeRec  root;
    FT_ULong    strike_index;    /* 0xFFFFFFFF to indicate invalid */

  } CFF_SizeRec, *CFF_Size;


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    CFF_GlyphSlot                                                      */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A handle to an OpenType glyph slot object.                         */
  /*                                                                       */
  typedef struct  CFF_GlyphSlotRec_
  {
    FT_GlyphSlotRec  root;

    FT_Bool          hint;
    FT_Bool          scaled;

    FT_Fixed         x_scale;
    FT_Fixed         y_scale;

  } CFF_GlyphSlotRec, *CFF_GlyphSlot;


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    CFF_Internal                                                       */
  /*                                                                       */
  /* <Description>                                                         */
  /*    The interface to the `internal' field of `FT_Size'.                */
  /*                                                                       */
  typedef struct  CFF_InternalRec_
  {
    PSH_Globals  topfont;
    PSH_Globals  subfonts[CFF_MAX_CID_FONTS];

  } CFF_InternalRec, *CFF_Internal;


  /*************************************************************************/
  /*                                                                       */
  /* Subglyph transformation record.                                       */
  /*                                                                       */
  typedef struct  CFF_Transform_
  {
    FT_Fixed    xx, xy;     /* transformation matrix coefficients */
    FT_Fixed    yx, yy;
    FT_F26Dot6  ox, oy;     /* offsets        */

  } CFF_Transform;


  /***********************************************************************/
  /*                                                                     */
  /* CFF driver class.                                                   */
  /*                                                                     */
  typedef struct  CFF_DriverRec_
  {
    FT_DriverRec  root;

    FT_UInt  hinting_engine;
    FT_Bool  no_stem_darkening;

  } CFF_DriverRec;


  FT_LOCAL( FT_Error )
  cff_size_init( FT_Size  size );           /* CFF_Size */

  FT_LOCAL( void )
  cff_size_done( FT_Size  size );           /* CFF_Size */

  FT_LOCAL( FT_Error )
  cff_size_request( FT_Size          size,
                    FT_Size_Request  req );

#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS

  FT_LOCAL( FT_Error )
  cff_size_select( FT_Size   size,
                   FT_ULong  strike_index );

#endif

  FT_LOCAL( void )
  cff_slot_done( FT_GlyphSlot  slot );

  FT_LOCAL( FT_Error )
  cff_slot_init( FT_GlyphSlot  slot );


  /*************************************************************************/
  /*                                                                       */
  /* Face functions                                                        */
  /*                                                                       */
  FT_LOCAL( FT_Error )
  cff_face_init( FT_Stream      stream,
                 FT_Face        face,           /* CFF_Face */
                 FT_Int         face_index,
                 FT_Int         num_params,
                 FT_Parameter*  params );

  FT_LOCAL( void )
  cff_face_done( FT_Face  face );               /* CFF_Face */


  /*************************************************************************/
  /*                                                                       */
  /* Driver functions                                                      */
  /*                                                                       */
  FT_LOCAL( FT_Error )
  cff_driver_init( FT_Module  module );         /* CFF_Driver */

  FT_LOCAL( void )
  cff_driver_done( FT_Module  module );         /* CFF_Driver */


FT_END_HEADER

#endif /* __CFFOBJS_H__ */


/* END */

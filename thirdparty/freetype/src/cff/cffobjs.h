/****************************************************************************
 *
 * cffobjs.h
 *
 *   OpenType objects manager (specification).
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


#ifndef CFFOBJS_H_
#define CFFOBJS_H_




FT_BEGIN_HEADER


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


  /**************************************************************************
   *
   * Face functions
   */
  FT_LOCAL( FT_Error )
  cff_face_init( FT_Stream      stream,
                 FT_Face        face,           /* CFF_Face */
                 FT_Int         face_index,
                 FT_Int         num_params,
                 FT_Parameter*  params );

  FT_LOCAL( void )
  cff_face_done( FT_Face  face );               /* CFF_Face */


  /**************************************************************************
   *
   * Driver functions
   */
  FT_LOCAL( FT_Error )
  cff_driver_init( FT_Module  module );         /* PS_Driver */

  FT_LOCAL( void )
  cff_driver_done( FT_Module  module );         /* PS_Driver */


FT_END_HEADER

#endif /* CFFOBJS_H_ */


/* END */

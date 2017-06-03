/***************************************************************************/
/*                                                                         */
/*  sfobjs.h                                                               */
/*                                                                         */
/*    SFNT object management (specification).                              */
/*                                                                         */
/*  Copyright 1996-2016 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef SFOBJS_H_
#define SFOBJS_H_


#include <ft2build.h>
#include FT_INTERNAL_SFNT_H
#include FT_INTERNAL_OBJECTS_H


FT_BEGIN_HEADER


  FT_LOCAL( FT_Error )
  sfnt_init_face( FT_Stream      stream,
                  TT_Face        face,
                  FT_Int         face_instance_index,
                  FT_Int         num_params,
                  FT_Parameter*  params );

  FT_LOCAL( FT_Error )
  sfnt_load_face( FT_Stream      stream,
                  TT_Face        face,
                  FT_Int         face_instance_index,
                  FT_Int         num_params,
                  FT_Parameter*  params );

  FT_LOCAL( void )
  sfnt_done_face( TT_Face  face );

  FT_LOCAL( FT_Error )
  tt_face_get_name( TT_Face      face,
                    FT_UShort    nameid,
                    FT_String**  name );


FT_END_HEADER

#endif /* SFDRIVER_H_ */


/* END */

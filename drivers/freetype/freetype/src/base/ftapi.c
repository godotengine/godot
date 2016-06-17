/***************************************************************************/
/*                                                                         */
/*  ftapi.c                                                                */
/*                                                                         */
/*    The FreeType compatibility functions (body).                         */
/*                                                                         */
/*  Copyright 2002 by                                                      */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include <ft2build.h>
#include FT_LIST_H
#include FT_OUTLINE_H
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_STREAM_H
#include FT_TRUETYPE_TABLES_H
#include FT_OUTLINE_H


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****                                                                 ****/
  /****                 C O M P A T I B I L I T Y                       ****/
  /****                                                                 ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/

  /* backwards compatibility API */

  FT_BASE_DEF( void )
  FT_New_Memory_Stream( FT_Library  library,
                        FT_Byte*    base,
                        FT_ULong    size,
                        FT_Stream   stream )
  {
    FT_UNUSED( library );

    FT_Stream_OpenMemory( stream, base, size );
  }


  FT_BASE_DEF( FT_Error )
  FT_Seek_Stream( FT_Stream  stream,
                  FT_ULong   pos )
  {
    return FT_Stream_Seek( stream, pos );
  }


  FT_BASE_DEF( FT_Error )
  FT_Skip_Stream( FT_Stream  stream,
                  FT_Long    distance )
  {
    return FT_Stream_Skip( stream, distance );
  }


  FT_BASE_DEF( FT_Error )
  FT_Read_Stream( FT_Stream  stream,
                  FT_Byte*   buffer,
                  FT_ULong   count )
  {
    return FT_Stream_Read( stream, buffer, count );
  }


  FT_BASE_DEF( FT_Error )
  FT_Read_Stream_At( FT_Stream  stream,
                     FT_ULong   pos,
                     FT_Byte*   buffer,
                     FT_ULong   count )
  {
    return FT_Stream_ReadAt( stream, pos, buffer, count );
  }


  FT_BASE_DEF( FT_Error )
  FT_Extract_Frame( FT_Stream  stream,
                    FT_ULong   count,
                    FT_Byte**  pbytes )
  {
    return FT_Stream_ExtractFrame( stream, count, pbytes );
  }


  FT_BASE_DEF( void )
  FT_Release_Frame( FT_Stream  stream,
                    FT_Byte**  pbytes )
  {
    FT_Stream_ReleaseFrame( stream, pbytes );
  }

  FT_BASE_DEF( FT_Error )
  FT_Access_Frame( FT_Stream  stream,
                   FT_ULong   count )
  {
    return FT_Stream_EnterFrame( stream, count );
  }


  FT_BASE_DEF( void )
  FT_Forget_Frame( FT_Stream  stream )
  {
    FT_Stream_ExitFrame( stream );
  }


/* END */

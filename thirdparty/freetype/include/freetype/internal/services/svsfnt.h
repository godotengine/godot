/***************************************************************************/
/*                                                                         */
/*  svsfnt.h                                                               */
/*                                                                         */
/*    The FreeType SFNT table loading service (specification).             */
/*                                                                         */
/*  Copyright 2003-2018 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef SVSFNT_H_
#define SVSFNT_H_

#include FT_INTERNAL_SERVICE_H
#include FT_TRUETYPE_TABLES_H


FT_BEGIN_HEADER


  /*
   *  SFNT table loading service.
   */

#define FT_SERVICE_ID_SFNT_TABLE  "sfnt-table"


  /*
   * Used to implement FT_Load_Sfnt_Table().
   */
  typedef FT_Error
  (*FT_SFNT_TableLoadFunc)( FT_Face    face,
                            FT_ULong   tag,
                            FT_Long    offset,
                            FT_Byte*   buffer,
                            FT_ULong*  length );

  /*
   * Used to implement FT_Get_Sfnt_Table().
   */
  typedef void*
  (*FT_SFNT_TableGetFunc)( FT_Face      face,
                           FT_Sfnt_Tag  tag );


  /*
   * Used to implement FT_Sfnt_Table_Info().
   */
  typedef FT_Error
  (*FT_SFNT_TableInfoFunc)( FT_Face    face,
                            FT_UInt    idx,
                            FT_ULong  *tag,
                            FT_ULong  *offset,
                            FT_ULong  *length );


  FT_DEFINE_SERVICE( SFNT_Table )
  {
    FT_SFNT_TableLoadFunc  load_table;
    FT_SFNT_TableGetFunc   get_table;
    FT_SFNT_TableInfoFunc  table_info;
  };


#ifndef FT_CONFIG_OPTION_PIC

#define FT_DEFINE_SERVICE_SFNT_TABLEREC( class_, load_, get_, info_ )  \
  static const FT_Service_SFNT_TableRec  class_ =                      \
  {                                                                    \
    load_, get_, info_                                                 \
  };

#else /* FT_CONFIG_OPTION_PIC */

#define FT_DEFINE_SERVICE_SFNT_TABLEREC( class_, load_, get_, info_ ) \
  void                                                                \
  FT_Init_Class_ ## class_( FT_Service_SFNT_TableRec*  clazz )        \
  {                                                                   \
    clazz->load_table = load_;                                        \
    clazz->get_table  = get_;                                         \
    clazz->table_info = info_;                                        \
  }

#endif /* FT_CONFIG_OPTION_PIC */

  /* */


FT_END_HEADER


#endif /* SVSFNT_H_ */


/* END */

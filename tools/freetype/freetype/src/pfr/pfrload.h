/***************************************************************************/
/*                                                                         */
/*  pfrload.h                                                              */
/*                                                                         */
/*    FreeType PFR loader (specification).                                 */
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


#ifndef __PFRLOAD_H__
#define __PFRLOAD_H__

#include "pfrobjs.h"
#include FT_INTERNAL_STREAM_H


FT_BEGIN_HEADER

#ifdef PFR_CONFIG_NO_CHECKS
#define PFR_CHECK( x )  do { } while ( 0 )
#else
#define PFR_CHECK( x )  do                       \
                        {                        \
                          if ( p + (x) > limit ) \
                            goto Too_Short;      \
                        } while ( 0 )
#endif

#define PFR_NEXT_BYTE( p )    FT_NEXT_BYTE( p )
#define PFR_NEXT_INT8( p )    FT_NEXT_CHAR( p )
#define PFR_NEXT_SHORT( p )   FT_NEXT_SHORT( p )
#define PFR_NEXT_USHORT( p )  FT_NEXT_USHORT( p )
#define PFR_NEXT_LONG( p )    FT_NEXT_OFF3( p )
#define PFR_NEXT_ULONG( p )   FT_NEXT_UOFF3( p )


 /* handling extra items */

  typedef FT_Error
  (*PFR_ExtraItem_ParseFunc)( FT_Byte*    p,
                              FT_Byte*    limit,
                              FT_Pointer  data );

  typedef struct  PFR_ExtraItemRec_
  {
    FT_UInt                  type;
    PFR_ExtraItem_ParseFunc  parser;

  } PFR_ExtraItemRec;

  typedef const struct PFR_ExtraItemRec_*  PFR_ExtraItem;


  FT_LOCAL( FT_Error )
  pfr_extra_items_skip( FT_Byte*  *pp,
                        FT_Byte*  limit );

  FT_LOCAL( FT_Error )
  pfr_extra_items_parse( FT_Byte*      *pp,
                         FT_Byte*       limit,
                         PFR_ExtraItem  item_list,
                         FT_Pointer     item_data );


  /* load a PFR header */
  FT_LOCAL( FT_Error )
  pfr_header_load( PFR_Header  header,
                   FT_Stream   stream );

  /* check a PFR header */
  FT_LOCAL( FT_Bool )
  pfr_header_check( PFR_Header  header );


  /* return number of logical fonts in this file */
  FT_LOCAL( FT_Error )
  pfr_log_font_count( FT_Stream   stream,
                      FT_UInt32   log_section_offset,
                      FT_UInt    *acount );

  /* load a pfr logical font entry */
  FT_LOCAL( FT_Error )
  pfr_log_font_load( PFR_LogFont  log_font,
                     FT_Stream    stream,
                     FT_UInt      face_index,
                     FT_UInt32    section_offset,
                     FT_Bool      size_increment );


  /* load a physical font entry */
  FT_LOCAL( FT_Error )
  pfr_phy_font_load( PFR_PhyFont  phy_font,
                     FT_Stream    stream,
                     FT_UInt32    offset,
                     FT_UInt32    size );

  /* finalize a physical font */
  FT_LOCAL( void )
  pfr_phy_font_done( PFR_PhyFont  phy_font,
                     FT_Memory    memory );

  /* */

FT_END_HEADER

#endif /* __PFRLOAD_H__ */


/* END */

/***************************************************************************/
/*                                                                         */
/*  sfdriver.c                                                             */
/*                                                                         */
/*    High-level SFNT driver interface (body).                             */
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


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_SFNT_H
#include FT_INTERNAL_OBJECTS_H

#include "sfdriver.h"
#include "ttload.h"
#include "sfobjs.h"
#include "sfntpic.h"

#include "sferrors.h"

#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS
#include "ttsbit.h"
#endif

#ifdef TT_CONFIG_OPTION_POSTSCRIPT_NAMES
#include "ttpost.h"
#endif

#ifdef TT_CONFIG_OPTION_BDF
#include "ttbdf.h"
#include FT_SERVICE_BDF_H
#endif

#include "ttcmap.h"
#include "ttkern.h"
#include "ttmtx.h"

#include FT_SERVICE_GLYPH_DICT_H
#include FT_SERVICE_POSTSCRIPT_NAME_H
#include FT_SERVICE_SFNT_H
#include FT_SERVICE_TT_CMAP_H


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_sfdriver


  /*
   *  SFNT TABLE SERVICE
   *
   */

  static void*
  get_sfnt_table( TT_Face      face,
                  FT_Sfnt_Tag  tag )
  {
    void*  table;


    switch ( tag )
    {
    case FT_SFNT_HEAD:
      table = &face->header;
      break;

    case FT_SFNT_HHEA:
      table = &face->horizontal;
      break;

    case FT_SFNT_VHEA:
      table = face->vertical_info ? &face->vertical : NULL;
      break;

    case FT_SFNT_OS2:
      table = face->os2.version == 0xFFFFU ? NULL : &face->os2;
      break;

    case FT_SFNT_POST:
      table = &face->postscript;
      break;

    case FT_SFNT_MAXP:
      table = &face->max_profile;
      break;

    case FT_SFNT_PCLT:
      table = face->pclt.Version ? &face->pclt : NULL;
      break;

    default:
      table = NULL;
    }

    return table;
  }


  static FT_Error
  sfnt_table_info( TT_Face    face,
                   FT_UInt    idx,
                   FT_ULong  *tag,
                   FT_ULong  *offset,
                   FT_ULong  *length )
  {
    if ( !offset || !length )
      return FT_THROW( Invalid_Argument );

    if ( !tag )
      *length = face->num_tables;
    else
    {
      if ( idx >= face->num_tables )
        return FT_THROW( Table_Missing );

      *tag    = face->dir_tables[idx].Tag;
      *offset = face->dir_tables[idx].Offset;
      *length = face->dir_tables[idx].Length;
    }

    return FT_Err_Ok;
  }


  FT_DEFINE_SERVICE_SFNT_TABLEREC(
    sfnt_service_sfnt_table,
    (FT_SFNT_TableLoadFunc)tt_face_load_any,     /* load_table */
    (FT_SFNT_TableGetFunc) get_sfnt_table,       /* get_table  */
    (FT_SFNT_TableInfoFunc)sfnt_table_info )     /* table_info */


#ifdef TT_CONFIG_OPTION_POSTSCRIPT_NAMES

  /*
   *  GLYPH DICT SERVICE
   *
   */

  static FT_Error
  sfnt_get_glyph_name( TT_Face     face,
                       FT_UInt     glyph_index,
                       FT_Pointer  buffer,
                       FT_UInt     buffer_max )
  {
    FT_String*  gname;
    FT_Error    error;


    error = tt_face_get_ps_name( face, glyph_index, &gname );
    if ( !error )
      FT_STRCPYN( buffer, gname, buffer_max );

    return error;
  }


  static FT_UInt
  sfnt_get_name_index( TT_Face     face,
                       FT_String*  glyph_name )
  {
    FT_Face  root = &face->root;

    FT_UInt  i, max_gid = FT_UINT_MAX;


    if ( root->num_glyphs < 0 )
      return 0;
    else if ( (FT_ULong)root->num_glyphs < FT_UINT_MAX )
      max_gid = (FT_UInt)root->num_glyphs;
    else
      FT_TRACE0(( "Ignore glyph names for invalid GID 0x%08x - 0x%08x\n",
                  FT_UINT_MAX, root->num_glyphs ));

    for ( i = 0; i < max_gid; i++ )
    {
      FT_String*  gname;
      FT_Error    error = tt_face_get_ps_name( face, i, &gname );


      if ( error )
        continue;

      if ( !ft_strcmp( glyph_name, gname ) )
        return i;
    }

    return 0;
  }


  FT_DEFINE_SERVICE_GLYPHDICTREC(
    sfnt_service_glyph_dict,
    (FT_GlyphDict_GetNameFunc)  sfnt_get_glyph_name,    /* get_name   */
    (FT_GlyphDict_NameIndexFunc)sfnt_get_name_index )   /* name_index */


#endif /* TT_CONFIG_OPTION_POSTSCRIPT_NAMES */


  /*
   *  POSTSCRIPT NAME SERVICE
   *
   */

  static const char*
  sfnt_get_ps_name( TT_Face  face )
  {
    FT_Int       n, found_win, found_apple;
    const char*  result = NULL;


    /* shouldn't happen, but just in case to avoid memory leaks */
    if ( face->postscript_name )
      return face->postscript_name;

    /* scan the name table to see whether we have a Postscript name here, */
    /* either in Macintosh or Windows platform encodings                  */
    found_win   = -1;
    found_apple = -1;

    for ( n = 0; n < face->num_names; n++ )
    {
      TT_NameEntryRec*  name = face->name_table.names + n;


      if ( name->nameID == 6 && name->stringLength > 0 )
      {
        if ( name->platformID == 3     &&
             name->encodingID == 1     &&
             name->languageID == 0x409 )
          found_win = n;

        if ( name->platformID == 1 &&
             name->encodingID == 0 &&
             name->languageID == 0 )
          found_apple = n;
      }
    }

    if ( found_win != -1 )
    {
      FT_Memory         memory = face->root.memory;
      TT_NameEntryRec*  name   = face->name_table.names + found_win;
      FT_UInt           len    = name->stringLength / 2;
      FT_Error          error  = FT_Err_Ok;

      FT_UNUSED( error );


      if ( !FT_ALLOC( result, name->stringLength + 1 ) )
      {
        FT_Stream   stream = face->name_table.stream;
        FT_String*  r      = (FT_String*)result;
        FT_Char*    p;


        if ( FT_STREAM_SEEK( name->stringOffset ) ||
             FT_FRAME_ENTER( name->stringLength ) )
        {
          FT_FREE( result );
          name->stringLength = 0;
          name->stringOffset = 0;
          FT_FREE( name->string );

          goto Exit;
        }

        p = (FT_Char*)stream->cursor;

        for ( ; len > 0; len--, p += 2 )
        {
          if ( p[0] == 0 && p[1] >= 32 )
            *r++ = p[1];
        }
        *r = '\0';

        FT_FRAME_EXIT();
      }
      goto Exit;
    }

    if ( found_apple != -1 )
    {
      FT_Memory         memory = face->root.memory;
      TT_NameEntryRec*  name   = face->name_table.names + found_apple;
      FT_UInt           len    = name->stringLength;
      FT_Error          error  = FT_Err_Ok;

      FT_UNUSED( error );


      if ( !FT_ALLOC( result, len + 1 ) )
      {
        FT_Stream  stream = face->name_table.stream;


        if ( FT_STREAM_SEEK( name->stringOffset ) ||
             FT_STREAM_READ( result, len )        )
        {
          name->stringOffset = 0;
          name->stringLength = 0;
          FT_FREE( name->string );
          FT_FREE( result );
          goto Exit;
        }
        ((char*)result)[len] = '\0';
      }
    }

  Exit:
    face->postscript_name = result;
    return result;
  }


  FT_DEFINE_SERVICE_PSFONTNAMEREC(
    sfnt_service_ps_name,
    (FT_PsName_GetFunc)sfnt_get_ps_name )     /* get_ps_font_name */


  /*
   *  TT CMAP INFO
   */
  FT_DEFINE_SERVICE_TTCMAPSREC(
    tt_service_get_cmap_info,
    (TT_CMap_Info_GetFunc)tt_get_cmap_info )  /* get_cmap_info */


#ifdef TT_CONFIG_OPTION_BDF

  static FT_Error
  sfnt_get_charset_id( TT_Face       face,
                       const char*  *acharset_encoding,
                       const char*  *acharset_registry )
  {
    BDF_PropertyRec  encoding, registry;
    FT_Error         error;


    /* XXX: I don't know whether this is correct, since
     *      tt_face_find_bdf_prop only returns something correct if we have
     *      previously selected a size that is listed in the BDF table.
     *      Should we change the BDF table format to include single offsets
     *      for `CHARSET_REGISTRY' and `CHARSET_ENCODING'?
     */
    error = tt_face_find_bdf_prop( face, "CHARSET_REGISTRY", &registry );
    if ( !error )
    {
      error = tt_face_find_bdf_prop( face, "CHARSET_ENCODING", &encoding );
      if ( !error )
      {
        if ( registry.type == BDF_PROPERTY_TYPE_ATOM &&
             encoding.type == BDF_PROPERTY_TYPE_ATOM )
        {
          *acharset_encoding = encoding.u.atom;
          *acharset_registry = registry.u.atom;
        }
        else
          error = FT_THROW( Invalid_Argument );
      }
    }

    return error;
  }


  FT_DEFINE_SERVICE_BDFRec(
    sfnt_service_bdf,
    (FT_BDF_GetCharsetIdFunc)sfnt_get_charset_id,     /* get_charset_id */
    (FT_BDF_GetPropertyFunc) tt_face_find_bdf_prop )  /* get_property   */


#endif /* TT_CONFIG_OPTION_BDF */


  /*
   *  SERVICE LIST
   */

#if defined TT_CONFIG_OPTION_POSTSCRIPT_NAMES && defined TT_CONFIG_OPTION_BDF
  FT_DEFINE_SERVICEDESCREC5(
    sfnt_services,
    FT_SERVICE_ID_SFNT_TABLE,           &SFNT_SERVICE_SFNT_TABLE_GET,
    FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &SFNT_SERVICE_PS_NAME_GET,
    FT_SERVICE_ID_GLYPH_DICT,           &SFNT_SERVICE_GLYPH_DICT_GET,
    FT_SERVICE_ID_BDF,                  &SFNT_SERVICE_BDF_GET,
    FT_SERVICE_ID_TT_CMAP,              &TT_SERVICE_CMAP_INFO_GET )
#elif defined TT_CONFIG_OPTION_POSTSCRIPT_NAMES
  FT_DEFINE_SERVICEDESCREC4(
    sfnt_services,
    FT_SERVICE_ID_SFNT_TABLE,           &SFNT_SERVICE_SFNT_TABLE_GET,
    FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &SFNT_SERVICE_PS_NAME_GET,
    FT_SERVICE_ID_GLYPH_DICT,           &SFNT_SERVICE_GLYPH_DICT_GET,
    FT_SERVICE_ID_TT_CMAP,              &TT_SERVICE_CMAP_INFO_GET )
#elif defined TT_CONFIG_OPTION_BDF
  FT_DEFINE_SERVICEDESCREC4(
    sfnt_services,
    FT_SERVICE_ID_SFNT_TABLE,           &SFNT_SERVICE_SFNT_TABLE_GET,
    FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &SFNT_SERVICE_PS_NAME_GET,
    FT_SERVICE_ID_BDF,                  &SFNT_SERVICE_BDF_GET,
    FT_SERVICE_ID_TT_CMAP,              &TT_SERVICE_CMAP_INFO_GET )
#else
  FT_DEFINE_SERVICEDESCREC3(
    sfnt_services,
    FT_SERVICE_ID_SFNT_TABLE,           &SFNT_SERVICE_SFNT_TABLE_GET,
    FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &SFNT_SERVICE_PS_NAME_GET,
    FT_SERVICE_ID_TT_CMAP,              &TT_SERVICE_CMAP_INFO_GET )
#endif


  FT_CALLBACK_DEF( FT_Module_Interface )
  sfnt_get_interface( FT_Module    module,
                      const char*  module_interface )
  {
    /* SFNT_SERVICES_GET dereferences `library' in PIC mode */
#ifdef FT_CONFIG_OPTION_PIC
    FT_Library  library;


    if ( !module )
      return NULL;
    library = module->library;
    if ( !library )
      return NULL;
#else
    FT_UNUSED( module );
#endif

    return ft_service_list_lookup( SFNT_SERVICES_GET, module_interface );
  }


#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS
#define PUT_EMBEDDED_BITMAPS( a )  a
#else
#define PUT_EMBEDDED_BITMAPS( a )  NULL
#endif

#ifdef TT_CONFIG_OPTION_POSTSCRIPT_NAMES
#define PUT_PS_NAMES( a )  a
#else
#define PUT_PS_NAMES( a )  NULL
#endif

  FT_DEFINE_SFNT_INTERFACE(
    sfnt_interface,
    tt_face_goto_table,

    sfnt_init_face,
    sfnt_load_face,
    sfnt_done_face,
    sfnt_get_interface,

    tt_face_load_any,

    tt_face_load_head,
    tt_face_load_hhea,
    tt_face_load_cmap,
    tt_face_load_maxp,
    tt_face_load_os2,
    tt_face_load_post,

    tt_face_load_name,
    tt_face_free_name,

    tt_face_load_kern,
    tt_face_load_gasp,
    tt_face_load_pclt,

    /* see `ttload.h' */
    PUT_EMBEDDED_BITMAPS( tt_face_load_bhed ),

    PUT_EMBEDDED_BITMAPS( tt_face_load_sbit_image ),

    /* see `ttpost.h' */
    PUT_PS_NAMES( tt_face_get_ps_name   ),
    PUT_PS_NAMES( tt_face_free_ps_names ),

    /* since version 2.1.8 */
    tt_face_get_kerning,

    /* since version 2.2 */
    tt_face_load_font_dir,
    tt_face_load_hmtx,

    /* see `ttsbit.h' and `sfnt.h' */
    PUT_EMBEDDED_BITMAPS( tt_face_load_sbit ),
    PUT_EMBEDDED_BITMAPS( tt_face_free_sbit ),

    PUT_EMBEDDED_BITMAPS( tt_face_set_sbit_strike     ),
    PUT_EMBEDDED_BITMAPS( tt_face_load_strike_metrics ),

    tt_face_get_metrics,

    tt_face_get_name
  )


  FT_DEFINE_MODULE(
    sfnt_module_class,

    0,  /* not a font driver or renderer */
    sizeof ( FT_ModuleRec ),

    "sfnt",     /* driver name                            */
    0x10000L,   /* driver version 1.0                     */
    0x20000L,   /* driver requires FreeType 2.0 or higher */

    (const void*)&SFNT_INTERFACE_GET,  /* module specific interface */

    (FT_Module_Constructor)0,
    (FT_Module_Destructor) 0,
    (FT_Module_Requester)  sfnt_get_interface )


/* END */

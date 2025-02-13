/****************************************************************************
 *
 * cffcmap.c
 *
 *   CFF character mapping table (cmap) support (body).
 *
 * Copyright (C) 2002-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <freetype/internal/ftdebug.h>
#include "cffcmap.h"
#include "cffload.h"

#include "cfferrs.h"


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****           CFF STANDARD (AND EXPERT) ENCODING CMAPS            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_CALLBACK_DEF( FT_Error )
  cff_cmap_encoding_init( FT_CMap     cmap,
                          FT_Pointer  pointer )
  {
    CFF_CMapStd   cffcmap  = (CFF_CMapStd)cmap;
    TT_Face       face     = (TT_Face)FT_CMAP_FACE( cmap );
    CFF_Font      cff      = (CFF_Font)face->extra.data;
    CFF_Encoding  encoding = &cff->encoding;

    FT_UNUSED( pointer );


    cffcmap->gids = encoding->codes;

    return 0;
  }


  FT_CALLBACK_DEF( void )
  cff_cmap_encoding_done( FT_CMap  cmap )
  {
    CFF_CMapStd  cffcmap = (CFF_CMapStd)cmap;


    cffcmap->gids = NULL;
  }


  FT_CALLBACK_DEF( FT_UInt )
  cff_cmap_encoding_char_index( FT_CMap    cmap,
                                FT_UInt32  char_code )
  {
    CFF_CMapStd  cffcmap = (CFF_CMapStd)cmap;
    FT_UInt      result  = 0;


    if ( char_code < 256 )
      result = cffcmap->gids[char_code];

    return result;
  }


  FT_CALLBACK_DEF( FT_UInt )
  cff_cmap_encoding_char_next( FT_CMap     cmap,
                               FT_UInt32  *pchar_code )
  {
    CFF_CMapStd  cffcmap   = (CFF_CMapStd)cmap;
    FT_UInt      result    = 0;
    FT_UInt32    char_code = *pchar_code;


    while ( char_code < 255 )
    {
      result = cffcmap->gids[++char_code];
      if ( result )
      {
        *pchar_code = char_code;
        break;
      }
    }

    return result;
  }


  FT_DEFINE_CMAP_CLASS(
    cff_cmap_encoding_class_rec,

    sizeof ( CFF_CMapStdRec ),

    (FT_CMap_InitFunc)     cff_cmap_encoding_init,        /* init       */
    (FT_CMap_DoneFunc)     cff_cmap_encoding_done,        /* done       */
    (FT_CMap_CharIndexFunc)cff_cmap_encoding_char_index,  /* char_index */
    (FT_CMap_CharNextFunc) cff_cmap_encoding_char_next,   /* char_next  */

    (FT_CMap_CharVarIndexFunc)    NULL,  /* char_var_index   */
    (FT_CMap_CharVarIsDefaultFunc)NULL,  /* char_var_default */
    (FT_CMap_VariantListFunc)     NULL,  /* variant_list     */
    (FT_CMap_CharVariantListFunc) NULL,  /* charvariant_list */
    (FT_CMap_VariantCharListFunc) NULL   /* variantchar_list */
  )


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****              CFF SYNTHETIC UNICODE ENCODING CMAP              *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_CALLBACK_DEF( const char* )
  cff_sid_to_glyph_name( void*    face_,  /* TT_Face */
                         FT_UInt  idx )
  {
    TT_Face      face    = (TT_Face)face_;
    CFF_Font     cff     = (CFF_Font)face->extra.data;
    CFF_Charset  charset = &cff->charset;
    FT_UInt      sid     = charset->sids[idx];


    return cff_index_get_sid_string( cff, sid );
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_cmap_unicode_init( FT_CMap      cmap,     /* PS_Unicodes */
                         FT_Pointer   pointer )
  {
    PS_Unicodes         unicodes = (PS_Unicodes)cmap;
    TT_Face             face     = (TT_Face)FT_CMAP_FACE( cmap );
    FT_Memory           memory   = FT_FACE_MEMORY( face );
    CFF_Font            cff      = (CFF_Font)face->extra.data;
    CFF_Charset         charset  = &cff->charset;
    FT_Service_PsCMaps  psnames  = (FT_Service_PsCMaps)cff->psnames;

    FT_UNUSED( pointer );


    /* can't build Unicode map for CID-keyed font */
    /* because we don't know glyph names.         */
    if ( !charset->sids )
      return FT_THROW( No_Unicode_Glyph_Name );

    if ( !psnames->unicodes_init )
      return FT_THROW( Unimplemented_Feature );

    return psnames->unicodes_init( memory,
                                   unicodes,
                                   cff->num_glyphs,
                                   &cff_sid_to_glyph_name,
                                   (PS_FreeGlyphNameFunc)NULL,
                                   (FT_Pointer)face );
  }


  FT_CALLBACK_DEF( void )
  cff_cmap_unicode_done( FT_CMap  cmap )    /* PS_Unicodes */
  {
    PS_Unicodes  unicodes = (PS_Unicodes)cmap;
    FT_Face      face     = FT_CMAP_FACE( cmap );
    FT_Memory    memory   = FT_FACE_MEMORY( face );


    FT_FREE( unicodes->maps );
    unicodes->num_maps = 0;
  }


  FT_CALLBACK_DEF( FT_UInt )
  cff_cmap_unicode_char_index( FT_CMap    cmap,       /* PS_Unicodes */
                               FT_UInt32  char_code )
  {
    PS_Unicodes         unicodes = (PS_Unicodes)cmap;
    TT_Face             face     = (TT_Face)FT_CMAP_FACE( cmap );
    CFF_Font            cff      = (CFF_Font)face->extra.data;
    FT_Service_PsCMaps  psnames  = (FT_Service_PsCMaps)cff->psnames;


    return psnames->unicodes_char_index( unicodes, char_code );
  }


  FT_CALLBACK_DEF( FT_UInt )
  cff_cmap_unicode_char_next( FT_CMap     cmap,        /* PS_Unicodes */
                              FT_UInt32  *pchar_code )
  {
    PS_Unicodes         unicodes = (PS_Unicodes)cmap;
    TT_Face             face     = (TT_Face)FT_CMAP_FACE( cmap );
    CFF_Font            cff      = (CFF_Font)face->extra.data;
    FT_Service_PsCMaps  psnames  = (FT_Service_PsCMaps)cff->psnames;


    return psnames->unicodes_char_next( unicodes, pchar_code );
  }


  FT_DEFINE_CMAP_CLASS(
    cff_cmap_unicode_class_rec,

    sizeof ( PS_UnicodesRec ),

    (FT_CMap_InitFunc)     cff_cmap_unicode_init,        /* init       */
    (FT_CMap_DoneFunc)     cff_cmap_unicode_done,        /* done       */
    (FT_CMap_CharIndexFunc)cff_cmap_unicode_char_index,  /* char_index */
    (FT_CMap_CharNextFunc) cff_cmap_unicode_char_next,   /* char_next  */

    (FT_CMap_CharVarIndexFunc)    NULL,  /* char_var_index   */
    (FT_CMap_CharVarIsDefaultFunc)NULL,  /* char_var_default */
    (FT_CMap_VariantListFunc)     NULL,  /* variant_list     */
    (FT_CMap_CharVariantListFunc) NULL,  /* charvariant_list */
    (FT_CMap_VariantCharListFunc) NULL   /* variantchar_list */
  )


/* END */

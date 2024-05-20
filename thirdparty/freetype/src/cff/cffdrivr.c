/****************************************************************************
 *
 * cffdrivr.c
 *
 *   OpenType font driver implementation (body).
 *
 * Copyright (C) 1996-2023 by
 * David Turner, Robert Wilhelm, Werner Lemberg, and Dominik Röttsches.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <freetype/freetype.h>
#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/sfnt.h>
#include <freetype/internal/psaux.h>
#include <freetype/internal/ftpsprop.h>
#include <freetype/internal/services/svcid.h>
#include <freetype/internal/services/svpsinfo.h>
#include <freetype/internal/services/svpostnm.h>
#include <freetype/internal/services/svttcmap.h>
#include <freetype/internal/services/svcfftl.h>

#include "cffdrivr.h"
#include "cffgload.h"
#include "cffload.h"
#include "cffcmap.h"
#include "cffparse.h"
#include "cffobjs.h"

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include <freetype/internal/services/svmm.h>
#include <freetype/internal/services/svmetric.h>
#endif

#include "cfferrs.h"

#include <freetype/internal/services/svfntfmt.h>
#include <freetype/internal/services/svgldict.h>
#include <freetype/internal/services/svprop.h>
#include <freetype/ftdriver.h>


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  cffdriver


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****                                                                 ****/
  /****                          F A C E S                              ****/
  /****                                                                 ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


  /**************************************************************************
   *
   * @Function:
   *   cff_get_kerning
   *
   * @Description:
   *   A driver method used to return the kerning vector between two
   *   glyphs of the same face.
   *
   * @Input:
   *   face ::
   *     A handle to the source face object.
   *
   *   left_glyph ::
   *     The index of the left glyph in the kern pair.
   *
   *   right_glyph ::
   *     The index of the right glyph in the kern pair.
   *
   * @Output:
   *   kerning ::
   *     The kerning vector.  This is in font units for
   *     scalable formats, and in pixels for fixed-sizes
   *     formats.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   *
   * @Note:
   *   Only horizontal layouts (left-to-right & right-to-left) are
   *   supported by this function.  Other layouts, or more sophisticated
   *   kernings, are out of scope of this method (the basic driver
   *   interface is meant to be simple).
   *
   *   They can be implemented by format-specific interfaces.
   */
  FT_CALLBACK_DEF( FT_Error )
  cff_get_kerning( FT_Face     face,          /* CFF_Face */
                   FT_UInt     left_glyph,
                   FT_UInt     right_glyph,
                   FT_Vector*  kerning )
  {
    CFF_Face      cffface = (CFF_Face)face;
    SFNT_Service  sfnt    = (SFNT_Service)cffface->sfnt;


    kerning->x = 0;
    kerning->y = 0;

    if ( sfnt )
      kerning->x = sfnt->get_kerning( cffface, left_glyph, right_glyph );

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   cff_glyph_load
   *
   * @Description:
   *   A driver method used to load a glyph within a given glyph slot.
   *
   * @Input:
   *   slot ::
   *     A handle to the target slot object where the glyph
   *     will be loaded.
   *
   *   size ::
   *     A handle to the source face size at which the glyph
   *     must be scaled, loaded, etc.
   *
   *   glyph_index ::
   *     The index of the glyph in the font file.
   *
   *   load_flags ::
   *     A flag indicating what to load for this glyph.  The
   *     FT_LOAD_??? constants can be used to control the
   *     glyph loading process (e.g., whether the outline
   *     should be scaled, whether to load bitmaps or not,
   *     whether to hint the outline, etc).
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_CALLBACK_DEF( FT_Error )
  cff_glyph_load( FT_GlyphSlot  slot,        /* CFF_GlyphSlot */
                  FT_Size       size,        /* CFF_Size      */
                  FT_UInt       glyph_index,
                  FT_Int32      load_flags )
  {
    FT_Error       error;
    CFF_GlyphSlot  cffslot = (CFF_GlyphSlot)slot;
    CFF_Size       cffsize = (CFF_Size)size;


    if ( !cffslot )
      return FT_THROW( Invalid_Slot_Handle );

    FT_TRACE1(( "cff_glyph_load: glyph index %d\n", glyph_index ));

    /* check whether we want a scaled outline or bitmap */
    if ( !cffsize )
      load_flags |= FT_LOAD_NO_SCALE | FT_LOAD_NO_HINTING;

    /* reset the size object if necessary */
    if ( load_flags & FT_LOAD_NO_SCALE )
      size = NULL;

    if ( size )
    {
      /* these two objects must have the same parent */
      if ( size->face != slot->face )
        return FT_THROW( Invalid_Face_Handle );
    }

    /* now load the glyph outline if necessary */
    error = cff_slot_load( cffslot, cffsize, glyph_index, load_flags );

    /* force drop-out mode to 2 - irrelevant now */
    /* slot->outline.dropout_mode = 2; */

    return error;
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_get_advances( FT_Face    face,
                    FT_UInt    start,
                    FT_UInt    count,
                    FT_Int32   flags,
                    FT_Fixed*  advances )
  {
    FT_UInt       nn;
    FT_Error      error = FT_Err_Ok;
    FT_GlyphSlot  slot  = face->glyph;


    if ( FT_IS_SFNT( face ) )
    {
      /* OpenType 1.7 mandates that the data from `hmtx' table be used; */
      /* it is no longer necessary that those values are identical to   */
      /* the values in the `CFF' table                                  */

      CFF_Face  cffface = (CFF_Face)face;
      FT_Short  dummy;


      if ( flags & FT_LOAD_VERTICAL_LAYOUT )
      {
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
        /* no fast retrieval for blended MM fonts without VVAR table */
        if ( ( FT_IS_NAMED_INSTANCE( face ) || FT_IS_VARIATION( face ) ) &&
             !( cffface->variation_support & TT_FACE_FLAG_VAR_VADVANCE ) )
          return FT_THROW( Unimplemented_Feature );
#endif

        /* check whether we have data from the `vmtx' table at all; */
        /* otherwise we extract the info from the CFF glyphstrings  */
        /* (instead of synthesizing a global value using the `OS/2' */
        /* table)                                                   */
        if ( !cffface->vertical_info )
          goto Missing_Table;

        for ( nn = 0; nn < count; nn++ )
        {
          FT_UShort  ah;


          ( (SFNT_Service)cffface->sfnt )->get_metrics( cffface,
                                                        1,
                                                        start + nn,
                                                        &dummy,
                                                        &ah );

          FT_TRACE5(( "  idx %d: advance height %d font unit%s\n",
                      start + nn,
                      ah,
                      ah == 1 ? "" : "s" ));
          advances[nn] = ah;
        }
      }
      else
      {
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
        /* no fast retrieval for blended MM fonts without HVAR table */
        if ( ( FT_IS_NAMED_INSTANCE( face ) || FT_IS_VARIATION( face ) ) &&
             !( cffface->variation_support & TT_FACE_FLAG_VAR_HADVANCE ) )
          return FT_THROW( Unimplemented_Feature );
#endif

        /* check whether we have data from the `hmtx' table at all */
        if ( !cffface->horizontal.number_Of_HMetrics )
          goto Missing_Table;

        for ( nn = 0; nn < count; nn++ )
        {
          FT_UShort  aw;


          ( (SFNT_Service)cffface->sfnt )->get_metrics( cffface,
                                                        0,
                                                        start + nn,
                                                        &dummy,
                                                        &aw );

          FT_TRACE5(( "  idx %d: advance width %d font unit%s\n",
                      start + nn,
                      aw,
                      aw == 1 ? "" : "s" ));
          advances[nn] = aw;
        }
      }

      return error;
    }

  Missing_Table:
    flags |= (FT_UInt32)FT_LOAD_ADVANCE_ONLY;

    for ( nn = 0; nn < count; nn++ )
    {
      error = cff_glyph_load( slot, face->size, start + nn, flags );
      if ( error )
        break;

      advances[nn] = ( flags & FT_LOAD_VERTICAL_LAYOUT )
                     ? slot->linearVertAdvance
                     : slot->linearHoriAdvance;
    }

    return error;
  }


  /*
   * GLYPH DICT SERVICE
   *
   */

  FT_CALLBACK_DEF( FT_Error )
  cff_get_glyph_name( FT_Face     face,        /* CFF_Face */
                      FT_UInt     glyph_index,
                      FT_Pointer  buffer,
                      FT_UInt     buffer_max )
  {
    CFF_Face    cffface = (CFF_Face)face;
    CFF_Font    font    = (CFF_Font)cffface->extra.data;
    FT_String*  gname;
    FT_UShort   sid;
    FT_Error    error;


    /* CFF2 table does not have glyph names; */
    /* we need to use `post' table method    */
    if ( font->version_major == 2 )
    {
      FT_Library            library     = FT_FACE_LIBRARY( face );
      FT_Module             sfnt_module = FT_Get_Module( library, "sfnt" );
      FT_Service_GlyphDict  service     =
        (FT_Service_GlyphDict)ft_module_get_service(
                                 sfnt_module,
                                 FT_SERVICE_ID_GLYPH_DICT,
                                 0 );


      if ( service && service->get_name )
        return service->get_name( face, glyph_index, buffer, buffer_max );
      else
      {
        FT_ERROR(( "cff_get_glyph_name:"
                   " cannot get glyph name from a CFF2 font\n" ));
        FT_ERROR(( "                   "
                   " without the `psnames' module\n" ));
        error = FT_THROW( Missing_Module );
        goto Exit;
      }
    }

    if ( !font->psnames )
    {
      FT_ERROR(( "cff_get_glyph_name:"
                 " cannot get glyph name from CFF & CEF fonts\n" ));
      FT_ERROR(( "                   "
                 " without the `psnames' module\n" ));
      error = FT_THROW( Missing_Module );
      goto Exit;
    }

    /* first, locate the sid in the charset table */
    sid = font->charset.sids[glyph_index];

    /* now, look up the name itself */
    gname = cff_index_get_sid_string( font, sid );

    if ( gname )
      FT_STRCPYN( buffer, gname, buffer_max );

    error = FT_Err_Ok;

  Exit:
    return error;
  }


  FT_CALLBACK_DEF( FT_UInt )
  cff_get_name_index( FT_Face           face,        /* CFF_Face */
                      const FT_String*  glyph_name )
  {
    CFF_Face            cffface = (CFF_Face)face;
    CFF_Font            cff     = (CFF_Font)cffface->extra.data;
    CFF_Charset         charset = &cff->charset;
    FT_Service_PsCMaps  psnames;
    FT_String*          name;
    FT_UShort           sid;
    FT_UInt             i;


    /* CFF2 table does not have glyph names; */
    /* we need to use `post' table method    */
    if ( cff->version_major == 2 )
    {
      FT_Library            library     = FT_FACE_LIBRARY( face );
      FT_Module             sfnt_module = FT_Get_Module( library, "sfnt" );
      FT_Service_GlyphDict  service     =
        (FT_Service_GlyphDict)ft_module_get_service(
                                 sfnt_module,
                                 FT_SERVICE_ID_GLYPH_DICT,
                                 0 );


      if ( service && service->name_index )
        return service->name_index( face, glyph_name );
      else
      {
        FT_ERROR(( "cff_get_name_index:"
                   " cannot get glyph index from a CFF2 font\n" ));
        FT_ERROR(( "                   "
                   " without the `psnames' module\n" ));
        return 0;
      }
    }

    FT_FACE_FIND_GLOBAL_SERVICE( face, psnames, POSTSCRIPT_CMAPS );
    if ( !psnames )
      return 0;

    for ( i = 0; i < cff->num_glyphs; i++ )
    {
      sid = charset->sids[i];

      if ( sid > 390 )
        name = cff_index_get_string( cff, sid - 391 );
      else
        name = (FT_String *)psnames->adobe_std_strings( sid );

      if ( !name )
        continue;

      if ( !ft_strcmp( glyph_name, name ) )
        return i;
    }

    return 0;
  }


  FT_DEFINE_SERVICE_GLYPHDICTREC(
    cff_service_glyph_dict,

    cff_get_glyph_name,  /* FT_GlyphDict_GetNameFunc   get_name   */
    cff_get_name_index   /* FT_GlyphDict_NameIndexFunc name_index */
  )


  /*
   * POSTSCRIPT INFO SERVICE
   *
   */

  FT_CALLBACK_DEF( FT_Int )
  cff_ps_has_glyph_names( FT_Face  face )
  {
    return ( face->face_flags & FT_FACE_FLAG_GLYPH_NAMES ) > 0;
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_ps_get_font_info( FT_Face          face,        /* CFF_Face */
                        PS_FontInfoRec*  afont_info )
  {
    CFF_Face  cffface = (CFF_Face)face;
    CFF_Font  cff     = (CFF_Font)cffface->extra.data;
    FT_Error  error   = FT_Err_Ok;


    if ( cffface->is_cff2 )
    {
      error = FT_THROW( Invalid_Argument );
      goto Fail;
    }

    if ( cff && !cff->font_info )
    {
      CFF_FontRecDict  dict      = &cff->top_font.font_dict;
      FT_Memory        memory    = FT_FACE_MEMORY( face );
      PS_FontInfoRec*  font_info = NULL;


      if ( FT_QNEW( font_info ) )
        goto Fail;

      font_info->version     = cff_index_get_sid_string( cff,
                                                         dict->version );
      font_info->notice      = cff_index_get_sid_string( cff,
                                                         dict->notice );
      font_info->full_name   = cff_index_get_sid_string( cff,
                                                         dict->full_name );
      font_info->family_name = cff_index_get_sid_string( cff,
                                                         dict->family_name );
      font_info->weight      = cff_index_get_sid_string( cff,
                                                         dict->weight );
      font_info->italic_angle        = dict->italic_angle;
      font_info->is_fixed_pitch      = dict->is_fixed_pitch;
      font_info->underline_position  = (FT_Short)dict->underline_position;
      font_info->underline_thickness = (FT_UShort)dict->underline_thickness;

      cff->font_info = font_info;
    }

    if ( cff )
      *afont_info = *cff->font_info;

  Fail:
    return error;
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_ps_get_font_extra( FT_Face           face,         /* CFF_Face */
                         PS_FontExtraRec*  afont_extra )
  {
    CFF_Face  cffface = (CFF_Face)face;
    CFF_Font  cff     = (CFF_Font)cffface->extra.data;
    FT_Error  error   = FT_Err_Ok;


    if ( cff && !cff->font_extra )
    {
      CFF_FontRecDict   dict       = &cff->top_font.font_dict;
      FT_Memory         memory     = FT_FACE_MEMORY( face );
      PS_FontExtraRec*  font_extra = NULL;
      FT_String*        embedded_postscript;


      if ( FT_QNEW( font_extra ) )
        goto Fail;

      font_extra->fs_type = 0U;

      embedded_postscript = cff_index_get_sid_string(
                              cff,
                              dict->embedded_postscript );
      if ( embedded_postscript )
      {
        FT_String*  start_fstype;
        FT_String*  start_def;


        /* Identify the XYZ integer in `/FSType XYZ def' substring. */
        if ( ( start_fstype = ft_strstr( embedded_postscript,
                                         "/FSType" ) ) != NULL    &&
             ( start_def = ft_strstr( start_fstype +
                                        sizeof ( "/FSType" ) - 1,
                                      "def" ) ) != NULL           )
        {
          FT_String*  s;


          for ( s = start_fstype + sizeof ( "/FSType" ) - 1;
                s != start_def;
                s++ )
          {
            if ( *s >= '0' && *s <= '9' )
            {
              if ( font_extra->fs_type >= ( FT_USHORT_MAX - 9 ) / 10 )
              {
                /* Overflow - ignore the FSType value.  */
                font_extra->fs_type = 0U;
                break;
              }

              font_extra->fs_type *= 10;
              font_extra->fs_type += (FT_UShort)( *s - '0' );
            }
            else if ( *s != ' ' && *s != '\n' && *s != '\r' )
            {
              /* Non-whitespace character between `/FSType' and next `def' */
              /* - ignore the FSType value.                                */
              font_extra->fs_type = 0U;
              break;
            }
          }
        }
      }

      cff->font_extra = font_extra;
    }

    if ( cff )
      *afont_extra = *cff->font_extra;

  Fail:
    return error;
  }


  FT_DEFINE_SERVICE_PSINFOREC(
    cff_service_ps_info,

    cff_ps_get_font_info,    /* PS_GetFontInfoFunc    ps_get_font_info    */
    cff_ps_get_font_extra,   /* PS_GetFontExtraFunc   ps_get_font_extra   */
    cff_ps_has_glyph_names,  /* PS_HasGlyphNamesFunc  ps_has_glyph_names  */
    /* unsupported with CFF fonts */
    NULL,                    /* PS_GetFontPrivateFunc ps_get_font_private */
    /* not implemented            */
    NULL                     /* PS_GetFontValueFunc   ps_get_font_value   */
  )


  /*
   * POSTSCRIPT NAME SERVICE
   *
   */

  FT_CALLBACK_DEF( const char* )
  cff_get_ps_name( FT_Face  face )    /* CFF_Face */
  {
    CFF_Face      cffface = (CFF_Face)face;
    CFF_Font      cff     = (CFF_Font)cffface->extra.data;
    SFNT_Service  sfnt    = (SFNT_Service)cffface->sfnt;


    /* following the OpenType specification 1.7, we return the name stored */
    /* in the `name' table for a CFF wrapped into an SFNT container        */

    if ( FT_IS_SFNT( face ) && sfnt )
    {
      FT_Library             library     = FT_FACE_LIBRARY( face );
      FT_Module              sfnt_module = FT_Get_Module( library, "sfnt" );
      FT_Service_PsFontName  service     =
        (FT_Service_PsFontName)ft_module_get_service(
                                 sfnt_module,
                                 FT_SERVICE_ID_POSTSCRIPT_FONT_NAME,
                                 0 );


      if ( service && service->get_ps_font_name )
        return service->get_ps_font_name( face );
    }

    return cff ? (const char*)cff->font_name : NULL;
  }


  FT_DEFINE_SERVICE_PSFONTNAMEREC(
    cff_service_ps_name,

    cff_get_ps_name  /* FT_PsName_GetFunc get_ps_font_name */
  )


  /*
   * TT CMAP INFO
   *
   * If the charmap is a synthetic Unicode encoding cmap or
   * a Type 1 standard (or expert) encoding cmap, hide TT CMAP INFO
   * service defined in SFNT module.
   *
   * Otherwise call the service function in the sfnt module.
   *
   */
  FT_CALLBACK_DEF( FT_Error )
  cff_get_cmap_info( FT_CharMap    charmap,
                     TT_CMapInfo  *cmap_info )
  {
    FT_CMap   cmap  = FT_CMAP( charmap );
    FT_Error  error = FT_Err_Ok;

    FT_Face     face    = FT_CMAP_FACE( cmap );
    FT_Library  library = FT_FACE_LIBRARY( face );


    if ( cmap->clazz != &cff_cmap_encoding_class_rec &&
         cmap->clazz != &cff_cmap_unicode_class_rec  )
    {
      FT_Module           sfnt    = FT_Get_Module( library, "sfnt" );
      FT_Service_TTCMaps  service =
        (FT_Service_TTCMaps)ft_module_get_service( sfnt,
                                                   FT_SERVICE_ID_TT_CMAP,
                                                   0 );


      if ( service && service->get_cmap_info )
        error = service->get_cmap_info( charmap, cmap_info );
    }
    else
      error = FT_THROW( Invalid_CharMap_Format );

    return error;
  }


  FT_DEFINE_SERVICE_TTCMAPSREC(
    cff_service_get_cmap_info,

    cff_get_cmap_info  /* TT_CMap_Info_GetFunc get_cmap_info */
  )


  /*
   * CID INFO SERVICE
   *
   */
  FT_CALLBACK_DEF( FT_Error )
  cff_get_ros( FT_Face       face,        /* FT_Face */
               const char*  *registry,
               const char*  *ordering,
               FT_Int       *supplement )
  {
    FT_Error  error   = FT_Err_Ok;
    CFF_Face  cffface = (CFF_Face)face;
    CFF_Font  cff     = (CFF_Font)cffface->extra.data;


    if ( cff )
    {
      CFF_FontRecDict  dict = &cff->top_font.font_dict;


      if ( dict->cid_registry == 0xFFFFU )
      {
        error = FT_THROW( Invalid_Argument );
        goto Fail;
      }

      if ( registry )
      {
        if ( !cff->registry )
          cff->registry = cff_index_get_sid_string( cff,
                                                    dict->cid_registry );
        *registry = cff->registry;
      }

      if ( ordering )
      {
        if ( !cff->ordering )
          cff->ordering = cff_index_get_sid_string( cff,
                                                    dict->cid_ordering );
        *ordering = cff->ordering;
      }

      /*
       * XXX: According to Adobe TechNote #5176, the supplement in CFF
       *      can be a real number. We truncate it to fit public API
       *      since freetype-2.3.6.
       */
      if ( supplement )
      {
        if ( dict->cid_supplement < FT_INT_MIN ||
             dict->cid_supplement > FT_INT_MAX )
          FT_TRACE1(( "cff_get_ros: too large supplement %ld is truncated\n",
                      dict->cid_supplement ));
        *supplement = (FT_Int)dict->cid_supplement;
      }
    }

  Fail:
    return error;
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_get_is_cid( FT_Face   face,    /* CFF_Face */
                  FT_Bool  *is_cid )
  {
    FT_Error  error   = FT_Err_Ok;
    CFF_Face  cffface = (CFF_Face)face;
    CFF_Font  cff     = (CFF_Font)cffface->extra.data;


    *is_cid = 0;

    if ( cff )
    {
      CFF_FontRecDict  dict = &cff->top_font.font_dict;


      if ( dict->cid_registry != 0xFFFFU )
        *is_cid = 1;
    }

    return error;
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_get_cid_from_glyph_index( FT_Face   face,        /* CFF_Face */
                                FT_UInt   glyph_index,
                                FT_UInt  *cid )
  {
    FT_Error  error   = FT_Err_Ok;
    CFF_Face  cffface = (CFF_Face)face;
    CFF_Font  cff     = (CFF_Font)cffface->extra.data;


    if ( cff )
    {
      FT_UInt          c;
      CFF_FontRecDict  dict = &cff->top_font.font_dict;


      if ( dict->cid_registry == 0xFFFFU )
      {
        error = FT_THROW( Invalid_Argument );
        goto Fail;
      }

      if ( glyph_index >= cff->num_glyphs )
      {
        error = FT_THROW( Invalid_Argument );
        goto Fail;
      }

      c = cff->charset.sids[glyph_index];

      if ( cid )
        *cid = c;
    }

  Fail:
    return error;
  }


  FT_DEFINE_SERVICE_CIDREC(
    cff_service_cid_info,

    cff_get_ros,
      /* FT_CID_GetRegistryOrderingSupplementFunc get_ros                  */
    cff_get_is_cid,
      /* FT_CID_GetIsInternallyCIDKeyedFunc       get_is_cid               */
    cff_get_cid_from_glyph_index
      /* FT_CID_GetCIDFromGlyphIndexFunc          get_cid_from_glyph_index */
  )


  /*
   * PROPERTY SERVICE
   *
   */

  FT_DEFINE_SERVICE_PROPERTIESREC(
    cff_service_properties,

    ps_property_set,  /* FT_Properties_SetFunc set_property */
    ps_property_get   /* FT_Properties_GetFunc get_property */
  )

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

  /*
   * MULTIPLE MASTER SERVICE
   *
   */

  FT_CALLBACK_DEF( FT_Error )
  cff_set_mm_blend( FT_Face    face,        /* CFF_Face */
                    FT_UInt    num_coords,
                    FT_Fixed*  coords )
  {
    CFF_Face                 cffface = (CFF_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)cffface->mm;


    return mm->set_mm_blend( face, num_coords, coords );
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_get_mm_blend( FT_Face    face,       /* CFF_Face */
                    FT_UInt    num_coords,
                    FT_Fixed*  coords )
  {
    CFF_Face                 cffface = (CFF_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)cffface->mm;


    return mm->get_mm_blend( face, num_coords, coords );
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_set_mm_weightvector( FT_Face    face,          /* CFF_Face */
                           FT_UInt    len,
                           FT_Fixed*  weightvector )
  {
    CFF_Face                 cffface = (CFF_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)cffface->mm;


    return mm->set_mm_weightvector( face, len, weightvector );
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_get_mm_weightvector( FT_Face    face,          /* CFF_Face */
                           FT_UInt*   len,
                           FT_Fixed*  weightvector )
  {
    CFF_Face                 cffface = (CFF_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)cffface->mm;


    return mm->get_mm_weightvector( face, len, weightvector );
  }


  FT_CALLBACK_DEF( void )
  cff_construct_ps_name( FT_Face  face )  /* CFF_Face */
  {
    CFF_Face                 cffface = (CFF_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)cffface->mm;


    mm->construct_ps_name( face );
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_get_mm_var( FT_Face      face,    /* CFF_Face */
                  FT_MM_Var*  *master )
  {
    CFF_Face                 cffface = (CFF_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)cffface->mm;


    return mm->get_mm_var( face, master );
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_set_var_design( FT_Face    face,       /* CFF_Face */
                      FT_UInt    num_coords,
                      FT_Fixed*  coords )
  {
    CFF_Face                 cffface = (CFF_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)cffface->mm;


    return mm->set_var_design( face, num_coords, coords );
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_get_var_design( FT_Face    face,       /* CFF_Face */
                      FT_UInt    num_coords,
                      FT_Fixed*  coords )
  {
    CFF_Face                 cffface = (CFF_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)cffface->mm;


    return mm->get_var_design( face, num_coords, coords );
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_set_named_instance( FT_Face   face,            /* CFF_Face */
                          FT_UInt   instance_index )
  {
    CFF_Face                 cffface = (CFF_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)cffface->mm;


    return mm->set_named_instance( face, instance_index );
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_get_default_named_instance( FT_Face   face,            /* CFF_Face */
                                  FT_UInt  *instance_index )
  {
    CFF_Face                 cffface = (CFF_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)cffface->mm;


    return mm->get_default_named_instance( face, instance_index );
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_load_item_variation_store( FT_Face          face,       /* CFF_Face */
                                 FT_ULong         offset,
                                 GX_ItemVarStore  itemStore )
  {
    CFF_Face                 cffface = (CFF_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)cffface->mm;


    return mm->load_item_var_store( face, offset, itemStore );
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_load_delta_set_index_mapping( FT_Face            face,   /* CFF_Face */
                                    FT_ULong           offset,
                                    GX_DeltaSetIdxMap  map,
                                    GX_ItemVarStore    itemStore,
                                    FT_ULong           table_len )
  {
    CFF_Face                 cffface = (CFF_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)cffface->mm;


    return mm->load_delta_set_idx_map( face, offset, map,
                                       itemStore, table_len );
  }


  FT_CALLBACK_DEF( FT_Int )
  cff_get_item_delta( FT_Face          face,        /* CFF_Face */
                      GX_ItemVarStore  itemStore,
                      FT_UInt          outerIndex,
                      FT_UInt          innerIndex )
  {
    CFF_Face                 cffface = (CFF_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)cffface->mm;


    return mm->get_item_delta( face, itemStore, outerIndex, innerIndex );
  }


  FT_CALLBACK_DEF( void )
  cff_done_item_variation_store( FT_Face          face,       /* CFF_Face */
                                 GX_ItemVarStore  itemStore )
  {
    CFF_Face                 cffface = (CFF_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)cffface->mm;


    mm->done_item_var_store( face, itemStore );
  }


  FT_CALLBACK_DEF( void )
  cff_done_delta_set_index_map( FT_Face            face,       /* CFF_Face */
                                GX_DeltaSetIdxMap  deltaSetIdxMap )
  {
    CFF_Face                 cffface = (CFF_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)cffface->mm;


    mm->done_delta_set_idx_map( face, deltaSetIdxMap );
  }



  FT_DEFINE_SERVICE_MULTIMASTERSREC(
    cff_service_multi_masters,

    NULL,                /* FT_Get_MM_Func         get_mm                     */
    NULL,                /* FT_Set_MM_Design_Func  set_mm_design              */
    cff_set_mm_blend,    /* FT_Set_MM_Blend_Func   set_mm_blend               */
    cff_get_mm_blend,    /* FT_Get_MM_Blend_Func   get_mm_blend               */
    cff_get_mm_var,      /* FT_Get_MM_Var_Func     get_mm_var                 */
    cff_set_var_design,  /* FT_Set_Var_Design_Func set_var_design             */
    cff_get_var_design,  /* FT_Get_Var_Design_Func get_var_design             */
    cff_set_named_instance,
             /* FT_Set_Named_Instance_Func         set_named_instance         */
    cff_get_default_named_instance,
             /* FT_Get_Default_Named_Instance_Func get_default_named_instance */
    cff_set_mm_weightvector,
             /* FT_Set_MM_WeightVector_Func        set_mm_weightvector        */
    cff_get_mm_weightvector,
             /* FT_Get_MM_WeightVector_Func        get_mm_weightvector        */
    cff_construct_ps_name,
             /* FT_Construct_PS_Name_Func          construct_ps_name          */
    cff_load_delta_set_index_mapping,
             /* FT_Var_Load_Delta_Set_Idx_Map_Func load_delta_set_idx_map     */
    cff_load_item_variation_store,
             /* FT_Var_Load_Item_Var_Store_Func    load_item_variation_store  */
    cff_get_item_delta,
             /* FT_Var_Get_Item_Delta_Func         get_item_delta             */
    cff_done_item_variation_store,
             /* FT_Var_Done_Item_Var_Store_Func    done_item_variation_store  */
    cff_done_delta_set_index_map,
             /* FT_Var_Done_Delta_Set_Idx_Map_Func done_delta_set_index_map   */
    cff_get_var_blend,   /* FT_Get_Var_Blend_Func  get_var_blend              */
    cff_done_blend       /* FT_Done_Blend_Func     done_blend                 */
  )


  /*
   * METRICS VARIATIONS SERVICE
   *
   */

  FT_CALLBACK_DEF( FT_Error )
  cff_hadvance_adjust( FT_Face   face,    /* CFF_Face */
                       FT_UInt   gindex,
                       FT_Int   *avalue )
  {
    CFF_Face  cffface = (CFF_Face)face;
    FT_Service_MetricsVariations
              var     = (FT_Service_MetricsVariations)cffface->tt_var;


    return var->hadvance_adjust( face, gindex, avalue );
  }


  FT_CALLBACK_DEF( void )
  cff_metrics_adjust( FT_Face  face )    /* CFF_Face */
  {
    CFF_Face  cffface = (CFF_Face)face;
    FT_Service_MetricsVariations
              var     = (FT_Service_MetricsVariations)cffface->tt_var;


    var->metrics_adjust( face );
  }


  FT_DEFINE_SERVICE_METRICSVARIATIONSREC(
    cff_service_metrics_variations,

    cff_hadvance_adjust,  /* FT_HAdvance_Adjust_Func hadvance_adjust */
    NULL,                 /* FT_LSB_Adjust_Func      lsb_adjust      */
    NULL,                 /* FT_RSB_Adjust_Func      rsb_adjust      */

    NULL,                 /* FT_VAdvance_Adjust_Func vadvance_adjust */
    NULL,                 /* FT_TSB_Adjust_Func      tsb_adjust      */
    NULL,                 /* FT_BSB_Adjust_Func      bsb_adjust      */
    NULL,                 /* FT_VOrg_Adjust_Func     vorg_adjust     */

    cff_metrics_adjust,   /* FT_Metrics_Adjust_Func  metrics_adjust  */
    NULL                  /* FT_Size_Reset_Func      size_reset      */
  )
#endif


  /*
   * CFFLOAD SERVICE
   *
   */

  FT_DEFINE_SERVICE_CFFLOADREC(
    cff_service_cff_load,

    cff_get_standard_encoding,  /* FT_Get_Standard_Encoding_Func get_standard_encoding */
    cff_load_private_dict,      /* FT_Load_Private_Dict_Func     load_private_dict     */
    cff_fd_select_get,          /* FT_FD_Select_Get_Func         fd_select_get         */
    cff_blend_check_vector,     /* FT_Blend_Check_Vector_Func    blend_check_vector    */
    cff_blend_build_vector      /* FT_Blend_Build_Vector_Func    blend_build_vector    */
  )


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****                                                                 ****/
  /****                D R I V E R  I N T E R F A C E                   ****/
  /****                                                                 ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/

#if defined TT_CONFIG_OPTION_GX_VAR_SUPPORT
  FT_DEFINE_SERVICEDESCREC10(
    cff_services,

    FT_SERVICE_ID_FONT_FORMAT,          FT_FONT_FORMAT_CFF,
    FT_SERVICE_ID_MULTI_MASTERS,        &cff_service_multi_masters,
    FT_SERVICE_ID_METRICS_VARIATIONS,   &cff_service_metrics_variations,
    FT_SERVICE_ID_POSTSCRIPT_INFO,      &cff_service_ps_info,
    FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &cff_service_ps_name,
    FT_SERVICE_ID_GLYPH_DICT,           &cff_service_glyph_dict,
    FT_SERVICE_ID_TT_CMAP,              &cff_service_get_cmap_info,
    FT_SERVICE_ID_CID,                  &cff_service_cid_info,
    FT_SERVICE_ID_PROPERTIES,           &cff_service_properties,
    FT_SERVICE_ID_CFF_LOAD,             &cff_service_cff_load
  )
#else
  FT_DEFINE_SERVICEDESCREC8(
    cff_services,

    FT_SERVICE_ID_FONT_FORMAT,          FT_FONT_FORMAT_CFF,
    FT_SERVICE_ID_POSTSCRIPT_INFO,      &cff_service_ps_info,
    FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &cff_service_ps_name,
    FT_SERVICE_ID_GLYPH_DICT,           &cff_service_glyph_dict,
    FT_SERVICE_ID_TT_CMAP,              &cff_service_get_cmap_info,
    FT_SERVICE_ID_CID,                  &cff_service_cid_info,
    FT_SERVICE_ID_PROPERTIES,           &cff_service_properties,
    FT_SERVICE_ID_CFF_LOAD,             &cff_service_cff_load
  )
#endif


  FT_CALLBACK_DEF( FT_Module_Interface )
  cff_get_interface( FT_Module    driver,       /* CFF_Driver */
                     const char*  module_interface )
  {
    FT_Library           library;
    FT_Module            sfnt;
    FT_Module_Interface  result;


    result = ft_service_list_lookup( cff_services, module_interface );
    if ( result )
      return result;

    /* `driver' is not yet evaluated */
    if ( !driver )
      return NULL;
    library = driver->library;
    if ( !library )
      return NULL;

    /* we pass our request to the `sfnt' module */
    sfnt = FT_Get_Module( library, "sfnt" );

    return sfnt ? sfnt->clazz->get_interface( sfnt, module_interface ) : 0;
  }


  /* The FT_DriverInterface structure is defined in ftdriver.h. */

#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS
#define CFF_SIZE_SELECT cff_size_select
#else
#define CFF_SIZE_SELECT 0
#endif

  FT_DEFINE_DRIVER(
    cff_driver_class,

      FT_MODULE_FONT_DRIVER          |
      FT_MODULE_DRIVER_SCALABLE      |
      FT_MODULE_DRIVER_HAS_HINTER    |
      FT_MODULE_DRIVER_HINTS_LIGHTLY,

      sizeof ( PS_DriverRec ),
      "cff",
      0x10000L,
      0x20000L,

      NULL,   /* module-specific interface */

      cff_driver_init,          /* FT_Module_Constructor  module_init   */
      cff_driver_done,          /* FT_Module_Destructor   module_done   */
      cff_get_interface,        /* FT_Module_Requester    get_interface */

    sizeof ( TT_FaceRec ),
    sizeof ( CFF_SizeRec ),
    sizeof ( CFF_GlyphSlotRec ),

    cff_face_init,              /* FT_Face_InitFunc  init_face */
    cff_face_done,              /* FT_Face_DoneFunc  done_face */
    cff_size_init,              /* FT_Size_InitFunc  init_size */
    cff_size_done,              /* FT_Size_DoneFunc  done_size */
    cff_slot_init,              /* FT_Slot_InitFunc  init_slot */
    cff_slot_done,              /* FT_Slot_DoneFunc  done_slot */

    cff_glyph_load,             /* FT_Slot_LoadFunc  load_glyph */

    cff_get_kerning,            /* FT_Face_GetKerningFunc   get_kerning  */
    NULL,                       /* FT_Face_AttachFunc       attach_file  */
    cff_get_advances,           /* FT_Face_GetAdvancesFunc  get_advances */

    cff_size_request,           /* FT_Size_RequestFunc  request_size */
    CFF_SIZE_SELECT             /* FT_Size_SelectFunc   select_size  */
  )


/* END */

/****************************************************************************
 *
 * cffdrivr.c
 *
 *   OpenType font driver implementation (body).
 *
 * Copyright (C) 1996-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
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
  cff_get_kerning( FT_Face     ttface,          /* TT_Face */
                   FT_UInt     left_glyph,
                   FT_UInt     right_glyph,
                   FT_Vector*  kerning )
  {
    TT_Face       face = (TT_Face)ttface;
    SFNT_Service  sfnt = (SFNT_Service)face->sfnt;


    kerning->x = 0;
    kerning->y = 0;

    if ( sfnt )
      kerning->x = sfnt->get_kerning( face, left_glyph, right_glyph );

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
  cff_glyph_load( FT_GlyphSlot  cffslot,      /* CFF_GlyphSlot */
                  FT_Size       cffsize,      /* CFF_Size      */
                  FT_UInt       glyph_index,
                  FT_Int32      load_flags )
  {
    FT_Error       error;
    CFF_GlyphSlot  slot = (CFF_GlyphSlot)cffslot;
    CFF_Size       size = (CFF_Size)cffsize;


    if ( !slot )
      return FT_THROW( Invalid_Slot_Handle );

    FT_TRACE1(( "cff_glyph_load: glyph index %d\n", glyph_index ));

    /* check whether we want a scaled outline or bitmap */
    if ( !size )
      load_flags |= FT_LOAD_NO_SCALE | FT_LOAD_NO_HINTING;

    /* reset the size object if necessary */
    if ( load_flags & FT_LOAD_NO_SCALE )
      size = NULL;

    if ( size )
    {
      /* these two objects must have the same parent */
      if ( cffsize->face != cffslot->face )
        return FT_THROW( Invalid_Face_Handle );
    }

    /* now load the glyph outline if necessary */
    error = cff_slot_load( slot, size, glyph_index, load_flags );

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

      TT_Face   ttface = (TT_Face)face;
      FT_Short  dummy;


      if ( flags & FT_LOAD_VERTICAL_LAYOUT )
      {
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
        /* no fast retrieval for blended MM fonts without VVAR table */
        if ( ( FT_IS_NAMED_INSTANCE( face ) || FT_IS_VARIATION( face ) ) &&
             !( ttface->variation_support & TT_FACE_FLAG_VAR_VADVANCE )  )
          return FT_THROW( Unimplemented_Feature );
#endif

        /* check whether we have data from the `vmtx' table at all; */
        /* otherwise we extract the info from the CFF glyphstrings  */
        /* (instead of synthesizing a global value using the `OS/2' */
        /* table)                                                   */
        if ( !ttface->vertical_info )
          goto Missing_Table;

        for ( nn = 0; nn < count; nn++ )
        {
          FT_UShort  ah;


          ( (SFNT_Service)ttface->sfnt )->get_metrics( ttface,
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
             !( ttface->variation_support & TT_FACE_FLAG_VAR_HADVANCE )  )
          return FT_THROW( Unimplemented_Feature );
#endif

        /* check whether we have data from the `hmtx' table at all */
        if ( !ttface->horizontal.number_Of_HMetrics )
          goto Missing_Table;

        for ( nn = 0; nn < count; nn++ )
        {
          FT_UShort  aw;


          ( (SFNT_Service)ttface->sfnt )->get_metrics( ttface,
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

  static FT_Error
  cff_get_glyph_name( CFF_Face    face,
                      FT_UInt     glyph_index,
                      FT_Pointer  buffer,
                      FT_UInt     buffer_max )
  {
    CFF_Font    font   = (CFF_Font)face->extra.data;
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
        return service->get_name( FT_FACE( face ),
                                  glyph_index,
                                  buffer,
                                  buffer_max );
      else
      {
        FT_ERROR(( "cff_get_glyph_name:"
                   " cannot get glyph name from a CFF2 font\n"
                   "                   "
                   " without the `psnames' module\n" ));
        error = FT_THROW( Missing_Module );
        goto Exit;
      }
    }

    if ( !font->psnames )
    {
      FT_ERROR(( "cff_get_glyph_name:"
                 " cannot get glyph name from CFF & CEF fonts\n"
                 "                   "
                 " without the `psnames' module\n" ));
      error = FT_THROW( Missing_Module );
      goto Exit;
    }

    /* first, locate the sid in the charset table */
    sid = font->charset.sids[glyph_index];

    /* now, lookup the name itself */
    gname = cff_index_get_sid_string( font, sid );

    if ( gname )
      FT_STRCPYN( buffer, gname, buffer_max );

    error = FT_Err_Ok;

  Exit:
    return error;
  }


  static FT_UInt
  cff_get_name_index( CFF_Face          face,
                      const FT_String*  glyph_name )
  {
    CFF_Font            cff;
    CFF_Charset         charset;
    FT_Service_PsCMaps  psnames;
    FT_String*          name;
    FT_UShort           sid;
    FT_UInt             i;


    cff     = (CFF_FontRec *)face->extra.data;
    charset = &cff->charset;

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
        return service->name_index( FT_FACE( face ), glyph_name );
      else
      {
        FT_ERROR(( "cff_get_name_index:"
                   " cannot get glyph index from a CFF2 font\n"
                   "                   "
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

    (FT_GlyphDict_GetNameFunc)  cff_get_glyph_name,      /* get_name   */
    (FT_GlyphDict_NameIndexFunc)cff_get_name_index       /* name_index */
  )


  /*
   * POSTSCRIPT INFO SERVICE
   *
   */

  static FT_Int
  cff_ps_has_glyph_names( FT_Face  face )
  {
    return ( face->face_flags & FT_FACE_FLAG_GLYPH_NAMES ) > 0;
  }


  static FT_Error
  cff_ps_get_font_info( CFF_Face         face,
                        PS_FontInfoRec*  afont_info )
  {
    CFF_Font  cff   = (CFF_Font)face->extra.data;
    FT_Error  error = FT_Err_Ok;


    if ( cff && !cff->font_info )
    {
      CFF_FontRecDict  dict      = &cff->top_font.font_dict;
      PS_FontInfoRec  *font_info = NULL;
      FT_Memory        memory    = face->root.memory;


      if ( FT_ALLOC( font_info, sizeof ( *font_info ) ) )
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


  static FT_Error
  cff_ps_get_font_extra( CFF_Face          face,
                         PS_FontExtraRec*  afont_extra )
  {
    CFF_Font  cff   = (CFF_Font)face->extra.data;
    FT_Error  error = FT_Err_Ok;


    if ( cff && cff->font_extra == NULL )
    {
      CFF_FontRecDict   dict       = &cff->top_font.font_dict;
      PS_FontExtraRec*  font_extra = NULL;
      FT_Memory         memory     = face->root.memory;
      FT_String*        embedded_postscript;


      if ( FT_ALLOC( font_extra, sizeof ( *font_extra ) ) )
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

    (PS_GetFontInfoFunc)   cff_ps_get_font_info,    /* ps_get_font_info    */
    (PS_GetFontExtraFunc)  cff_ps_get_font_extra,   /* ps_get_font_extra   */
    (PS_HasGlyphNamesFunc) cff_ps_has_glyph_names,  /* ps_has_glyph_names  */
    /* unsupported with CFF fonts */
    (PS_GetFontPrivateFunc)NULL,                    /* ps_get_font_private */
    /* not implemented            */
    (PS_GetFontValueFunc)  NULL                     /* ps_get_font_value   */
  )


  /*
   * POSTSCRIPT NAME SERVICE
   *
   */

  static const char*
  cff_get_ps_name( CFF_Face  face )
  {
    CFF_Font      cff  = (CFF_Font)face->extra.data;
    SFNT_Service  sfnt = (SFNT_Service)face->sfnt;


    /* following the OpenType specification 1.7, we return the name stored */
    /* in the `name' table for a CFF wrapped into an SFNT container        */

    if ( FT_IS_SFNT( FT_FACE( face ) ) && sfnt )
    {
      FT_Library             library     = FT_FACE_LIBRARY( face );
      FT_Module              sfnt_module = FT_Get_Module( library, "sfnt" );
      FT_Service_PsFontName  service     =
        (FT_Service_PsFontName)ft_module_get_service(
                                 sfnt_module,
                                 FT_SERVICE_ID_POSTSCRIPT_FONT_NAME,
                                 0 );


      if ( service && service->get_ps_font_name )
        return service->get_ps_font_name( FT_FACE( face ) );
    }

    return (const char*)cff->font_name;
  }


  FT_DEFINE_SERVICE_PSFONTNAMEREC(
    cff_service_ps_name,

    (FT_PsName_GetFunc)cff_get_ps_name      /* get_ps_font_name */
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
  static FT_Error
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

    (TT_CMap_Info_GetFunc)cff_get_cmap_info    /* get_cmap_info */
  )


  /*
   * CID INFO SERVICE
   *
   */
  static FT_Error
  cff_get_ros( CFF_Face      face,
               const char*  *registry,
               const char*  *ordering,
               FT_Int       *supplement )
  {
    FT_Error  error = FT_Err_Ok;
    CFF_Font  cff   = (CFF_Font)face->extra.data;


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


  static FT_Error
  cff_get_is_cid( CFF_Face  face,
                  FT_Bool  *is_cid )
  {
    FT_Error  error = FT_Err_Ok;
    CFF_Font  cff   = (CFF_Font)face->extra.data;


    *is_cid = 0;

    if ( cff )
    {
      CFF_FontRecDict  dict = &cff->top_font.font_dict;


      if ( dict->cid_registry != 0xFFFFU )
        *is_cid = 1;
    }

    return error;
  }


  static FT_Error
  cff_get_cid_from_glyph_index( CFF_Face  face,
                                FT_UInt   glyph_index,
                                FT_UInt  *cid )
  {
    FT_Error  error = FT_Err_Ok;
    CFF_Font  cff;


    cff = (CFF_Font)face->extra.data;

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

    (FT_CID_GetRegistryOrderingSupplementFunc)
      cff_get_ros,                             /* get_ros                  */
    (FT_CID_GetIsInternallyCIDKeyedFunc)
      cff_get_is_cid,                          /* get_is_cid               */
    (FT_CID_GetCIDFromGlyphIndexFunc)
      cff_get_cid_from_glyph_index             /* get_cid_from_glyph_index */
  )


  /*
   * PROPERTY SERVICE
   *
   */

  FT_DEFINE_SERVICE_PROPERTIESREC(
    cff_service_properties,

    (FT_Properties_SetFunc)ps_property_set,      /* set_property */
    (FT_Properties_GetFunc)ps_property_get )     /* get_property */


#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

  /*
   * MULTIPLE MASTER SERVICE
   *
   */

  static FT_Error
  cff_set_mm_blend( CFF_Face   face,
                    FT_UInt    num_coords,
                    FT_Fixed*  coords )
  {
    FT_Service_MultiMasters  mm = (FT_Service_MultiMasters)face->mm;


    return mm->set_mm_blend( FT_FACE( face ), num_coords, coords );
  }


  static FT_Error
  cff_get_mm_blend( CFF_Face   face,
                    FT_UInt    num_coords,
                    FT_Fixed*  coords )
  {
    FT_Service_MultiMasters  mm = (FT_Service_MultiMasters)face->mm;


    return mm->get_mm_blend( FT_FACE( face ), num_coords, coords );
  }


  static FT_Error
  cff_set_mm_weightvector( CFF_Face   face,
                           FT_UInt    len,
                           FT_Fixed*  weightvector )
  {
    FT_Service_MultiMasters  mm = (FT_Service_MultiMasters)face->mm;


    return mm->set_mm_weightvector( FT_FACE( face ), len, weightvector );
  }


  static FT_Error
  cff_get_mm_weightvector( CFF_Face   face,
                           FT_UInt*   len,
                           FT_Fixed*  weightvector )
  {
    FT_Service_MultiMasters  mm = (FT_Service_MultiMasters)face->mm;


    return mm->get_mm_weightvector( FT_FACE( face ), len, weightvector );
  }


  static FT_Error
  cff_get_mm_var( CFF_Face     face,
                  FT_MM_Var*  *master )
  {
    FT_Service_MultiMasters  mm = (FT_Service_MultiMasters)face->mm;


    return mm->get_mm_var( FT_FACE( face ), master );
  }


  static FT_Error
  cff_set_var_design( CFF_Face   face,
                      FT_UInt    num_coords,
                      FT_Fixed*  coords )
  {
    FT_Service_MultiMasters  mm = (FT_Service_MultiMasters)face->mm;


    return mm->set_var_design( FT_FACE( face ), num_coords, coords );
  }


  static FT_Error
  cff_get_var_design( CFF_Face   face,
                      FT_UInt    num_coords,
                      FT_Fixed*  coords )
  {
    FT_Service_MultiMasters  mm = (FT_Service_MultiMasters)face->mm;


    return mm->get_var_design( FT_FACE( face ), num_coords, coords );
  }


  static FT_Error
  cff_set_instance( CFF_Face  face,
                    FT_UInt   instance_index )
  {
    FT_Service_MultiMasters  mm = (FT_Service_MultiMasters)face->mm;


    return mm->set_instance( FT_FACE( face ), instance_index );
  }


  FT_DEFINE_SERVICE_MULTIMASTERSREC(
    cff_service_multi_masters,

    (FT_Get_MM_Func)             NULL,                    /* get_mm              */
    (FT_Set_MM_Design_Func)      NULL,                    /* set_mm_design       */
    (FT_Set_MM_Blend_Func)       cff_set_mm_blend,        /* set_mm_blend        */
    (FT_Get_MM_Blend_Func)       cff_get_mm_blend,        /* get_mm_blend        */
    (FT_Get_MM_Var_Func)         cff_get_mm_var,          /* get_mm_var          */
    (FT_Set_Var_Design_Func)     cff_set_var_design,      /* set_var_design      */
    (FT_Get_Var_Design_Func)     cff_get_var_design,      /* get_var_design      */
    (FT_Set_Instance_Func)       cff_set_instance,        /* set_instance        */
    (FT_Set_MM_WeightVector_Func)cff_set_mm_weightvector, /* set_mm_weightvector */
    (FT_Get_MM_WeightVector_Func)cff_get_mm_weightvector, /* get_mm_weightvector */

    (FT_Get_Var_Blend_Func)      cff_get_var_blend,       /* get_var_blend       */
    (FT_Done_Blend_Func)         cff_done_blend           /* done_blend          */
  )


  /*
   * METRICS VARIATIONS SERVICE
   *
   */

  static FT_Error
  cff_hadvance_adjust( CFF_Face  face,
                       FT_UInt   gindex,
                       FT_Int   *avalue )
  {
    FT_Service_MetricsVariations  var = (FT_Service_MetricsVariations)face->var;


    return var->hadvance_adjust( FT_FACE( face ), gindex, avalue );
  }


  static void
  cff_metrics_adjust( CFF_Face  face )
  {
    FT_Service_MetricsVariations  var = (FT_Service_MetricsVariations)face->var;


    var->metrics_adjust( FT_FACE( face ) );
  }


  FT_DEFINE_SERVICE_METRICSVARIATIONSREC(
    cff_service_metrics_variations,

    (FT_HAdvance_Adjust_Func)cff_hadvance_adjust,    /* hadvance_adjust */
    (FT_LSB_Adjust_Func)     NULL,                   /* lsb_adjust      */
    (FT_RSB_Adjust_Func)     NULL,                   /* rsb_adjust      */

    (FT_VAdvance_Adjust_Func)NULL,                   /* vadvance_adjust */
    (FT_TSB_Adjust_Func)     NULL,                   /* tsb_adjust      */
    (FT_BSB_Adjust_Func)     NULL,                   /* bsb_adjust      */
    (FT_VOrg_Adjust_Func)    NULL,                   /* vorg_adjust     */

    (FT_Metrics_Adjust_Func) cff_metrics_adjust      /* metrics_adjust  */
  )
#endif


  /*
   * CFFLOAD SERVICE
   *
   */

  FT_DEFINE_SERVICE_CFFLOADREC(
    cff_service_cff_load,

    (FT_Get_Standard_Encoding_Func)cff_get_standard_encoding,
    (FT_Load_Private_Dict_Func)    cff_load_private_dict,
    (FT_FD_Select_Get_Func)        cff_fd_select_get,
    (FT_Blend_Check_Vector_Func)   cff_blend_check_vector,
    (FT_Blend_Build_Vector_Func)   cff_blend_build_vector
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

#if !defined FT_CONFIG_OPTION_NO_GLYPH_NAMES && \
     defined TT_CONFIG_OPTION_GX_VAR_SUPPORT
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
#elif !defined FT_CONFIG_OPTION_NO_GLYPH_NAMES
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
#elif defined TT_CONFIG_OPTION_GX_VAR_SUPPORT
  FT_DEFINE_SERVICEDESCREC9(
    cff_services,

    FT_SERVICE_ID_FONT_FORMAT,          FT_FONT_FORMAT_CFF,
    FT_SERVICE_ID_MULTI_MASTERS,        &cff_service_multi_masters,
    FT_SERVICE_ID_METRICS_VARIATIONS,   &cff_service_metrics_var,
    FT_SERVICE_ID_POSTSCRIPT_INFO,      &cff_service_ps_info,
    FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &cff_service_ps_name,
    FT_SERVICE_ID_TT_CMAP,              &cff_service_get_cmap_info,
    FT_SERVICE_ID_CID,                  &cff_service_cid_info,
    FT_SERVICE_ID_PROPERTIES,           &cff_service_properties,
    FT_SERVICE_ID_CFF_LOAD,             &cff_service_cff_load
  )
#else
  FT_DEFINE_SERVICEDESCREC7(
    cff_services,

    FT_SERVICE_ID_FONT_FORMAT,          FT_FONT_FORMAT_CFF,
    FT_SERVICE_ID_POSTSCRIPT_INFO,      &cff_service_ps_info,
    FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &cff_service_ps_name,
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

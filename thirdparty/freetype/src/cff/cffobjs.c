/****************************************************************************
 *
 * cffobjs.c
 *
 *   OpenType objects manager (body).
 *
 * Copyright (C) 1996-2021 by
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
#include <freetype/internal/ftcalc.h>
#include <freetype/internal/ftstream.h>
#include <freetype/fterrors.h>
#include <freetype/ttnameid.h>
#include <freetype/tttags.h>
#include <freetype/internal/sfnt.h>
#include <freetype/ftdriver.h>

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include <freetype/ftmm.h>
#include <freetype/internal/services/svmm.h>
#include <freetype/internal/services/svmetric.h>
#endif

#include <freetype/internal/cffotypes.h>
#include "cffobjs.h"
#include "cffload.h"
#include "cffcmap.h"

#include "cfferrs.h"

#include <freetype/internal/psaux.h>
#include <freetype/internal/services/svcfftl.h>


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  cffobjs


  /**************************************************************************
   *
   *                           SIZE FUNCTIONS
   *
   */


  static PSH_Globals_Funcs
  cff_size_get_globals_funcs( CFF_Size  size )
  {
    CFF_Face          face     = (CFF_Face)size->root.face;
    CFF_Font          font     = (CFF_Font)face->extra.data;
    PSHinter_Service  pshinter = font->pshinter;
    FT_Module         module;


    module = FT_Get_Module( size->root.face->driver->root.library,
                            "pshinter" );
    return ( module && pshinter && pshinter->get_globals_funcs )
           ? pshinter->get_globals_funcs( module )
           : 0;
  }


  FT_LOCAL_DEF( void )
  cff_size_done( FT_Size  cffsize )        /* CFF_Size */
  {
    FT_Memory     memory   = cffsize->face->memory;
    CFF_Size      size     = (CFF_Size)cffsize;
    CFF_Face      face     = (CFF_Face)size->root.face;
    CFF_Font      font     = (CFF_Font)face->extra.data;
    CFF_Internal  internal = (CFF_Internal)cffsize->internal->module_data;


    if ( internal )
    {
      PSH_Globals_Funcs  funcs;


      funcs = cff_size_get_globals_funcs( size );
      if ( funcs )
      {
        FT_UInt  i;


        funcs->destroy( internal->topfont );

        for ( i = font->num_subfonts; i > 0; i-- )
          funcs->destroy( internal->subfonts[i - 1] );
      }

      FT_FREE( internal );
    }
  }


  /* CFF and Type 1 private dictionaries have slightly different      */
  /* structures; we need to synthesize a Type 1 dictionary on the fly */

  static void
  cff_make_private_dict( CFF_SubFont  subfont,
                         PS_Private   priv )
  {
    CFF_Private  cpriv = &subfont->private_dict;
    FT_UInt      n, count;


    FT_ZERO( priv );

    count = priv->num_blue_values = cpriv->num_blue_values;
    for ( n = 0; n < count; n++ )
      priv->blue_values[n] = (FT_Short)cpriv->blue_values[n];

    count = priv->num_other_blues = cpriv->num_other_blues;
    for ( n = 0; n < count; n++ )
      priv->other_blues[n] = (FT_Short)cpriv->other_blues[n];

    count = priv->num_family_blues = cpriv->num_family_blues;
    for ( n = 0; n < count; n++ )
      priv->family_blues[n] = (FT_Short)cpriv->family_blues[n];

    count = priv->num_family_other_blues = cpriv->num_family_other_blues;
    for ( n = 0; n < count; n++ )
      priv->family_other_blues[n] = (FT_Short)cpriv->family_other_blues[n];

    priv->blue_scale = cpriv->blue_scale;
    priv->blue_shift = (FT_Int)cpriv->blue_shift;
    priv->blue_fuzz  = (FT_Int)cpriv->blue_fuzz;

    priv->standard_width[0]  = (FT_UShort)cpriv->standard_width;
    priv->standard_height[0] = (FT_UShort)cpriv->standard_height;

    count = priv->num_snap_widths = cpriv->num_snap_widths;
    for ( n = 0; n < count; n++ )
      priv->snap_widths[n] = (FT_Short)cpriv->snap_widths[n];

    count = priv->num_snap_heights = cpriv->num_snap_heights;
    for ( n = 0; n < count; n++ )
      priv->snap_heights[n] = (FT_Short)cpriv->snap_heights[n];

    priv->force_bold     = cpriv->force_bold;
    priv->language_group = cpriv->language_group;
    priv->lenIV          = cpriv->lenIV;
  }


  FT_LOCAL_DEF( FT_Error )
  cff_size_init( FT_Size  cffsize )         /* CFF_Size */
  {
    CFF_Size           size  = (CFF_Size)cffsize;
    FT_Error           error = FT_Err_Ok;
    PSH_Globals_Funcs  funcs = cff_size_get_globals_funcs( size );

    FT_Memory     memory   = cffsize->face->memory;
    CFF_Internal  internal = NULL;
    CFF_Face      face     = (CFF_Face)cffsize->face;
    CFF_Font      font     = (CFF_Font)face->extra.data;

    PS_PrivateRec priv;

    FT_UInt       i;

    if ( !funcs )
      goto Exit;

    if ( FT_NEW( internal ) )
      goto Exit;

    cff_make_private_dict( &font->top_font, &priv );
    error = funcs->create( cffsize->face->memory, &priv,
                             &internal->topfont );
    if ( error )
      goto Exit;

    for ( i = font->num_subfonts; i > 0; i-- )
    {
      CFF_SubFont  sub = font->subfonts[i - 1];


      cff_make_private_dict( sub, &priv );
      error = funcs->create( cffsize->face->memory, &priv,
                               &internal->subfonts[i - 1] );
      if ( error )
        goto Exit;
    }

    cffsize->internal->module_data = internal;

    size->strike_index = 0xFFFFFFFFUL;

  Exit:
    if ( error )
    {
      if ( internal )
      {
        for ( i = font->num_subfonts; i > 0; i-- )
          FT_FREE( internal->subfonts[i - 1] );
        FT_FREE( internal->topfont );
      }

      FT_FREE( internal );
    }

    return error;
  }


#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS

  FT_LOCAL_DEF( FT_Error )
  cff_size_select( FT_Size   size,
                   FT_ULong  strike_index )
  {
    CFF_Size           cffsize = (CFF_Size)size;
    PSH_Globals_Funcs  funcs;


    cffsize->strike_index = strike_index;

    FT_Select_Metrics( size->face, strike_index );

    funcs = cff_size_get_globals_funcs( cffsize );

    if ( funcs )
    {
      CFF_Face      face     = (CFF_Face)size->face;
      CFF_Font      font     = (CFF_Font)face->extra.data;
      CFF_Internal  internal = (CFF_Internal)size->internal->module_data;

      FT_Long  top_upm  = (FT_Long)font->top_font.font_dict.units_per_em;
      FT_UInt  i;


      funcs->set_scale( internal->topfont,
                        size->metrics.x_scale, size->metrics.y_scale,
                        0, 0 );

      for ( i = font->num_subfonts; i > 0; i-- )
      {
        CFF_SubFont  sub     = font->subfonts[i - 1];
        FT_Long      sub_upm = (FT_Long)sub->font_dict.units_per_em;
        FT_Pos       x_scale, y_scale;


        if ( top_upm != sub_upm )
        {
          x_scale = FT_MulDiv( size->metrics.x_scale, top_upm, sub_upm );
          y_scale = FT_MulDiv( size->metrics.y_scale, top_upm, sub_upm );
        }
        else
        {
          x_scale = size->metrics.x_scale;
          y_scale = size->metrics.y_scale;
        }

        funcs->set_scale( internal->subfonts[i - 1],
                          x_scale, y_scale, 0, 0 );
      }
    }

    return FT_Err_Ok;
  }

#endif /* TT_CONFIG_OPTION_EMBEDDED_BITMAPS */


  FT_LOCAL_DEF( FT_Error )
  cff_size_request( FT_Size          size,
                    FT_Size_Request  req )
  {
    FT_Error  error;

    CFF_Size           cffsize = (CFF_Size)size;
    PSH_Globals_Funcs  funcs;


#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS

    if ( FT_HAS_FIXED_SIZES( size->face ) )
    {
      CFF_Face      cffface = (CFF_Face)size->face;
      SFNT_Service  sfnt    = (SFNT_Service)cffface->sfnt;
      FT_ULong      strike_index;


      if ( sfnt->set_sbit_strike( cffface, req, &strike_index ) )
        cffsize->strike_index = 0xFFFFFFFFUL;
      else
        return cff_size_select( size, strike_index );
    }

#endif /* TT_CONFIG_OPTION_EMBEDDED_BITMAPS */

    error = FT_Request_Metrics( size->face, req );
    if ( error )
      goto Exit;

    funcs = cff_size_get_globals_funcs( cffsize );

    if ( funcs )
    {
      CFF_Face      cffface  = (CFF_Face)size->face;
      CFF_Font      font     = (CFF_Font)cffface->extra.data;
      CFF_Internal  internal = (CFF_Internal)size->internal->module_data;

      FT_Long  top_upm  = (FT_Long)font->top_font.font_dict.units_per_em;
      FT_UInt  i;


      funcs->set_scale( internal->topfont,
                        size->metrics.x_scale, size->metrics.y_scale,
                        0, 0 );

      for ( i = font->num_subfonts; i > 0; i-- )
      {
        CFF_SubFont  sub     = font->subfonts[i - 1];
        FT_Long      sub_upm = (FT_Long)sub->font_dict.units_per_em;
        FT_Pos       x_scale, y_scale;


        if ( top_upm != sub_upm )
        {
          x_scale = FT_MulDiv( size->metrics.x_scale, top_upm, sub_upm );
          y_scale = FT_MulDiv( size->metrics.y_scale, top_upm, sub_upm );
        }
        else
        {
          x_scale = size->metrics.x_scale;
          y_scale = size->metrics.y_scale;
        }

        funcs->set_scale( internal->subfonts[i - 1],
                          x_scale, y_scale, 0, 0 );
      }
    }

  Exit:
    return error;
  }


  /**************************************************************************
   *
   *                           SLOT  FUNCTIONS
   *
   */

  FT_LOCAL_DEF( void )
  cff_slot_done( FT_GlyphSlot  slot )
  {
    if ( slot->internal )
      slot->internal->glyph_hints = NULL;
  }


  FT_LOCAL_DEF( FT_Error )
  cff_slot_init( FT_GlyphSlot  slot )
  {
    CFF_Face          face     = (CFF_Face)slot->face;
    CFF_Font          font     = (CFF_Font)face->extra.data;
    PSHinter_Service  pshinter = font->pshinter;


    if ( pshinter )
    {
      FT_Module  module;


      module = FT_Get_Module( slot->face->driver->root.library,
                              "pshinter" );
      if ( module )
      {
        T2_Hints_Funcs  funcs;


        funcs = pshinter->get_t2_funcs( module );
        slot->internal->glyph_hints = (void*)funcs;
      }
    }

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   *                          FACE  FUNCTIONS
   *
   */

  static FT_String*
  cff_strcpy( FT_Memory         memory,
              const FT_String*  source )
  {
    FT_Error    error;
    FT_String*  result;


    (void)FT_STRDUP( result, source );

    FT_UNUSED( error );

    return result;
  }


  /* Strip all subset prefixes of the form `ABCDEF+'.  Usually, there */
  /* is only one, but font names like `APCOOG+JFABTD+FuturaBQ-Bold'   */
  /* have been seen in the wild.                                      */

  static void
  remove_subset_prefix( FT_String*  name )
  {
    FT_Int32  idx             = 0;
    FT_Int32  length          = (FT_Int32)ft_strlen( name ) + 1;
    FT_Bool   continue_search = 1;


    while ( continue_search )
    {
      if ( length >= 7 && name[6] == '+' )
      {
        for ( idx = 0; idx < 6; idx++ )
        {
          /* ASCII uppercase letters */
          if ( !( 'A' <= name[idx] && name[idx] <= 'Z' ) )
            continue_search = 0;
        }

        if ( continue_search )
        {
          for ( idx = 7; idx < length; idx++ )
            name[idx - 7] = name[idx];
          length -= 7;
        }
      }
      else
        continue_search = 0;
    }
  }


  /* Remove the style part from the family name (if present). */

  static void
  remove_style( FT_String*        family_name,
                const FT_String*  style_name )
  {
    FT_Int32  family_name_length, style_name_length;


    family_name_length = (FT_Int32)ft_strlen( family_name );
    style_name_length  = (FT_Int32)ft_strlen( style_name );

    if ( family_name_length > style_name_length )
    {
      FT_Int  idx;


      for ( idx = 1; idx <= style_name_length; idx++ )
      {
        if ( family_name[family_name_length - idx] !=
             style_name[style_name_length - idx] )
          break;
      }

      if ( idx > style_name_length )
      {
        /* family_name ends with style_name; remove it */
        idx = family_name_length - style_name_length - 1;

        /* also remove special characters     */
        /* between real family name and style */
        while ( idx > 0                     &&
                ( family_name[idx] == '-' ||
                  family_name[idx] == ' ' ||
                  family_name[idx] == '_' ||
                  family_name[idx] == '+' ) )
          idx--;

        if ( idx > 0 )
          family_name[idx + 1] = '\0';
      }
    }
  }


  FT_LOCAL_DEF( FT_Error )
  cff_face_init( FT_Stream      stream,
                 FT_Face        cffface,        /* CFF_Face */
                 FT_Int         face_index,
                 FT_Int         num_params,
                 FT_Parameter*  params )
  {
    CFF_Face            face        = (CFF_Face)cffface;
    FT_Error            error;
    SFNT_Service        sfnt;
    FT_Service_PsCMaps  psnames;
    PSHinter_Service    pshinter;
    PSAux_Service       psaux;
    FT_Service_CFFLoad  cffload;
    FT_Bool             pure_cff    = 1;
    FT_Bool             cff2        = 0;
    FT_Bool             sfnt_format = 0;
    FT_Library          library     = cffface->driver->root.library;


    sfnt = (SFNT_Service)FT_Get_Module_Interface( library,
                                                  "sfnt" );
    if ( !sfnt )
    {
      FT_ERROR(( "cff_face_init: cannot access `sfnt' module\n" ));
      error = FT_THROW( Missing_Module );
      goto Exit;
    }

    FT_FACE_FIND_GLOBAL_SERVICE( face, psnames, POSTSCRIPT_CMAPS );

    pshinter = (PSHinter_Service)FT_Get_Module_Interface( library,
                                                          "pshinter" );

    psaux = (PSAux_Service)FT_Get_Module_Interface( library,
                                                    "psaux" );
    if ( !psaux )
    {
      FT_ERROR(( "cff_face_init: cannot access `psaux' module\n" ));
      error = FT_THROW( Missing_Module );
      goto Exit;
    }
    face->psaux = psaux;

    FT_FACE_FIND_GLOBAL_SERVICE( face, cffload, CFF_LOAD );

    FT_TRACE2(( "CFF driver\n" ));

    /* create input stream from resource */
    if ( FT_STREAM_SEEK( 0 ) )
      goto Exit;

    /* check whether we have a valid OpenType file */
    FT_TRACE2(( "  " ));
    error = sfnt->init_face( stream, face, face_index, num_params, params );
    if ( !error )
    {
      if ( face->format_tag != TTAG_OTTO )  /* `OTTO'; OpenType/CFF font */
      {
        FT_TRACE2(( "  not an OpenType/CFF font\n" ));
        error = FT_THROW( Unknown_File_Format );
        goto Exit;
      }

      /* if we are performing a simple font format check, exit immediately */
      if ( face_index < 0 )
        return FT_Err_Ok;

      sfnt_format = 1;

      /* now, the font can be either an OpenType/CFF font, or an SVG CEF */
      /* font; in the latter case it doesn't have a `head' table         */
      error = face->goto_table( face, TTAG_head, stream, 0 );
      if ( !error )
      {
        pure_cff = 0;

        /* load font directory */
        error = sfnt->load_face( stream, face, face_index,
                                 num_params, params );
        if ( error )
          goto Exit;
      }
      else
      {
        /* load the `cmap' table explicitly */
        error = sfnt->load_cmap( face, stream );
        if ( error )
          goto Exit;
      }

      /* now load the CFF part of the file; */
      /* give priority to CFF2              */
      error = face->goto_table( face, TTAG_CFF2, stream, 0 );
      if ( !error )
      {
        cff2          = 1;
        face->is_cff2 = cff2;
      }

      if ( FT_ERR_EQ( error, Table_Missing ) )
        error = face->goto_table( face, TTAG_CFF, stream, 0 );

      if ( error )
        goto Exit;
    }
    else
    {
      /* rewind to start of file; we are going to load a pure-CFF font */
      if ( FT_STREAM_SEEK( 0 ) )
        goto Exit;
      error = FT_Err_Ok;
    }

    /* now load and parse the CFF table in the file */
    {
      CFF_Font         cff = NULL;
      CFF_FontRecDict  dict;
      FT_Memory        memory = cffface->memory;
      FT_Int32         flags;
      FT_UInt          i;


      if ( FT_NEW( cff ) )
        goto Exit;

      face->extra.data = cff;
      error = cff_font_load( library,
                             stream,
                             face_index,
                             cff,
                             face,
                             pure_cff,
                             cff2 );
      if ( error )
        goto Exit;

      /* if we are performing a simple font format check, exit immediately */
      /* (this is here for pure CFF)                                       */
      if ( face_index < 0 )
      {
        cffface->num_faces = (FT_Long)cff->num_faces;
        return FT_Err_Ok;
      }

      cff->pshinter = pshinter;
      cff->psnames  = psnames;
      cff->cffload  = cffload;

      cffface->face_index = face_index & 0xFFFF;

      /* Complement the root flags with some interesting information. */
      /* Note that this is only necessary for pure CFF and CEF fonts; */
      /* SFNT based fonts use the `name' table instead.               */

      cffface->num_glyphs = (FT_Long)cff->num_glyphs;

      dict = &cff->top_font.font_dict;

      /* we need the `psnames' module for CFF and CEF formats */
      /* which aren't CID-keyed                               */
      if ( dict->cid_registry == 0xFFFFU && !psnames )
      {
        FT_ERROR(( "cff_face_init:"
                   " cannot open CFF & CEF fonts\n" ));
        FT_ERROR(( "              "
                   " without the `psnames' module\n" ));
        error = FT_THROW( Missing_Module );
        goto Exit;
      }

#ifdef FT_DEBUG_LEVEL_TRACE
      {
        FT_UInt     idx;
        FT_String*  s;


        FT_TRACE4(( "SIDs\n" ));

        /* dump string index, including default strings for convenience */
        for ( idx = 0; idx <= 390; idx++ )
        {
          s = cff_index_get_sid_string( cff, idx );
          if ( s )
            FT_TRACE4(( "  %5d %s\n", idx, s ));
        }

        /* In Multiple Master CFFs, two SIDs hold the Normalize Design  */
        /* Vector (NDV) and Convert Design Vector (CDV) charstrings,    */
        /* which may contain null bytes in the middle of the data, too. */
        /* We thus access `cff->strings' directly.                      */
        for ( idx = 1; idx < cff->num_strings; idx++ )
        {
          FT_Byte*    s1    = cff->strings[idx - 1];
          FT_Byte*    s2    = cff->strings[idx];
          FT_PtrDist  s1len = s2 - s1 - 1; /* without the final null byte */
          FT_PtrDist  l;


          FT_TRACE4(( "  %5d ", idx + 390 ));
          for ( l = 0; l < s1len; l++ )
            FT_TRACE4(( "%c", s1[l] ));
          FT_TRACE4(( "\n" ));
        }

        /* print last element */
        if ( cff->num_strings )
        {
          FT_Byte*    s1    = cff->strings[cff->num_strings - 1];
          FT_Byte*    s2    = cff->string_pool + cff->string_pool_size;
          FT_PtrDist  s1len = s2 - s1 - 1;
          FT_PtrDist  l;


          FT_TRACE4(( "  %5d ", cff->num_strings + 390 ));
          for ( l = 0; l < s1len; l++ )
            FT_TRACE4(( "%c", s1[l] ));
          FT_TRACE4(( "\n" ));
        }
      }
#endif /* FT_DEBUG_LEVEL_TRACE */

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      {
        FT_Service_MultiMasters       mm  = (FT_Service_MultiMasters)face->mm;
        FT_Service_MetricsVariations  var = (FT_Service_MetricsVariations)face->var;

        FT_UInt  instance_index = (FT_UInt)face_index >> 16;


        if ( FT_HAS_MULTIPLE_MASTERS( cffface ) &&
             mm                                 &&
             instance_index > 0                 )
        {
          error = mm->set_instance( cffface, instance_index );
          if ( error )
            goto Exit;

          if ( var )
            var->metrics_adjust( cffface );
        }
      }
#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */

      if ( !dict->has_font_matrix )
        dict->units_per_em = pure_cff ? 1000 : face->root.units_per_EM;

      /* Normalize the font matrix so that `matrix->yy' is 1; if  */
      /* it is zero, we use `matrix->yx' instead.  The scaling is */
      /* done with `units_per_em' then (at this point, it already */
      /* contains the scaling factor, but without normalization   */
      /* of the matrix).                                          */
      /*                                                          */
      /* Note that the offsets must be expressed in integer font  */
      /* units.                                                   */

      {
        FT_Matrix*  matrix = &dict->font_matrix;
        FT_Vector*  offset = &dict->font_offset;
        FT_ULong*   upm    = &dict->units_per_em;
        FT_Fixed    temp;


        temp = matrix->yy ? FT_ABS( matrix->yy )
                          : FT_ABS( matrix->yx );

        if ( temp != 0x10000L )
        {
          *upm = (FT_ULong)FT_DivFix( (FT_Long)*upm, temp );

          matrix->xx = FT_DivFix( matrix->xx, temp );
          matrix->yx = FT_DivFix( matrix->yx, temp );
          matrix->xy = FT_DivFix( matrix->xy, temp );
          matrix->yy = FT_DivFix( matrix->yy, temp );
          offset->x  = FT_DivFix( offset->x,  temp );
          offset->y  = FT_DivFix( offset->y,  temp );
        }

        offset->x >>= 16;
        offset->y >>= 16;
      }

      for ( i = cff->num_subfonts; i > 0; i-- )
      {
        CFF_FontRecDict  sub = &cff->subfonts[i - 1]->font_dict;
        CFF_FontRecDict  top = &cff->top_font.font_dict;

        FT_Matrix*  matrix;
        FT_Vector*  offset;
        FT_ULong*   upm;
        FT_Fixed    temp;


        if ( sub->has_font_matrix )
        {
          FT_Long  scaling;


          /* if we have a top-level matrix, */
          /* concatenate the subfont matrix */

          if ( top->has_font_matrix )
          {
            if ( top->units_per_em > 1 && sub->units_per_em > 1 )
              scaling = (FT_Long)FT_MIN( top->units_per_em,
                                         sub->units_per_em );
            else
              scaling = 1;

            FT_Matrix_Multiply_Scaled( &top->font_matrix,
                                       &sub->font_matrix,
                                       scaling );
            FT_Vector_Transform_Scaled( &sub->font_offset,
                                        &top->font_matrix,
                                        scaling );

            sub->units_per_em = (FT_ULong)
                                  FT_MulDiv( (FT_Long)sub->units_per_em,
                                             (FT_Long)top->units_per_em,
                                             scaling );
          }
        }
        else
        {
          sub->font_matrix = top->font_matrix;
          sub->font_offset = top->font_offset;

          sub->units_per_em = top->units_per_em;
        }

        matrix = &sub->font_matrix;
        offset = &sub->font_offset;
        upm    = &sub->units_per_em;

        temp = matrix->yy ? FT_ABS( matrix->yy )
                          : FT_ABS( matrix->yx );


        if ( temp != 0x10000L )
        {
          *upm = (FT_ULong)FT_DivFix( (FT_Long)*upm, temp );

          matrix->xx = FT_DivFix( matrix->xx, temp );
          matrix->yx = FT_DivFix( matrix->yx, temp );
          matrix->xy = FT_DivFix( matrix->xy, temp );
          matrix->yy = FT_DivFix( matrix->yy, temp );
          offset->x  = FT_DivFix( offset->x,  temp );
          offset->y  = FT_DivFix( offset->y,  temp );
        }

        offset->x >>= 16;
        offset->y >>= 16;
      }

      if ( pure_cff )
      {
        char*  style_name = NULL;


        /* set up num_faces */
        cffface->num_faces = (FT_Long)cff->num_faces;

        /* compute number of glyphs */
        if ( dict->cid_registry != 0xFFFFU )
          cffface->num_glyphs = (FT_Long)( cff->charset.max_cid + 1 );
        else
          cffface->num_glyphs = (FT_Long)cff->charstrings_index.count;

        /* set global bbox, as well as EM size */
        cffface->bbox.xMin =   dict->font_bbox.xMin            >> 16;
        cffface->bbox.yMin =   dict->font_bbox.yMin            >> 16;
        /* no `U' suffix here to 0xFFFF! */
        cffface->bbox.xMax = ( dict->font_bbox.xMax + 0xFFFF ) >> 16;
        cffface->bbox.yMax = ( dict->font_bbox.yMax + 0xFFFF ) >> 16;

        cffface->units_per_EM = (FT_UShort)( dict->units_per_em );

        cffface->ascender  = (FT_Short)( cffface->bbox.yMax );
        cffface->descender = (FT_Short)( cffface->bbox.yMin );

        cffface->height = (FT_Short)( ( cffface->units_per_EM * 12 ) / 10 );
        if ( cffface->height < cffface->ascender - cffface->descender )
          cffface->height = (FT_Short)( cffface->ascender -
                                        cffface->descender );

        cffface->underline_position  =
          (FT_Short)( dict->underline_position >> 16 );
        cffface->underline_thickness =
          (FT_Short)( dict->underline_thickness >> 16 );

        /* retrieve font family & style name */
        if ( dict->family_name )
        {
          char*  family_name;


          family_name = cff_index_get_sid_string( cff, dict->family_name );
          if ( family_name )
            cffface->family_name = cff_strcpy( memory, family_name );
        }

        if ( !cffface->family_name )
        {
          cffface->family_name = cff_index_get_name(
                                   cff,
                                   (FT_UInt)( face_index & 0xFFFF ) );
          if ( cffface->family_name )
            remove_subset_prefix( cffface->family_name );
        }

        if ( cffface->family_name )
        {
          char*  full   = cff_index_get_sid_string( cff,
                                                    dict->full_name );
          char*  fullp  = full;
          char*  family = cffface->family_name;


          /* We try to extract the style name from the full name.   */
          /* We need to ignore spaces and dashes during the search. */
          if ( full && family )
          {
            while ( *fullp )
            {
              /* skip common characters at the start of both strings */
              if ( *fullp == *family )
              {
                family++;
                fullp++;
                continue;
              }

              /* ignore spaces and dashes in full name during comparison */
              if ( *fullp == ' ' || *fullp == '-' )
              {
                fullp++;
                continue;
              }

              /* ignore spaces and dashes in family name during comparison */
              if ( *family == ' ' || *family == '-' )
              {
                family++;
                continue;
              }

              if ( !*family && *fullp )
              {
                /* The full name begins with the same characters as the  */
                /* family name, with spaces and dashes removed.  In this */
                /* case, the remaining string in `fullp' will be used as */
                /* the style name.                                       */
                style_name = cff_strcpy( memory, fullp );

                /* remove the style part from the family name (if present) */
                if ( style_name )
                  remove_style( cffface->family_name, style_name );
              }
              break;
            }
          }
        }
        else
        {
          char  *cid_font_name =
                   cff_index_get_sid_string( cff,
                                             dict->cid_font_name );


          /* do we have a `/FontName' for a CID-keyed font? */
          if ( cid_font_name )
            cffface->family_name = cff_strcpy( memory, cid_font_name );
        }

        if ( style_name )
          cffface->style_name = style_name;
        else
          /* assume "Regular" style if we don't know better */
          cffface->style_name = cff_strcpy( memory, "Regular" );

        /********************************************************************
         *
         * Compute face flags.
         */
        flags = FT_FACE_FLAG_SCALABLE   | /* scalable outlines */
                FT_FACE_FLAG_HORIZONTAL | /* horizontal data   */
                FT_FACE_FLAG_HINTER;      /* has native hinter */

        if ( sfnt_format )
          flags |= FT_FACE_FLAG_SFNT;

        /* fixed width font? */
        if ( dict->is_fixed_pitch )
          flags |= FT_FACE_FLAG_FIXED_WIDTH;

  /* XXX: WE DO NOT SUPPORT KERNING METRICS IN THE GPOS TABLE FOR NOW */
#if 0
        /* kerning available? */
        if ( face->kern_pairs )
          flags |= FT_FACE_FLAG_KERNING;
#endif

        cffface->face_flags |= flags;

        /********************************************************************
         *
         * Compute style flags.
         */
        flags = 0;

        if ( dict->italic_angle )
          flags |= FT_STYLE_FLAG_ITALIC;

        {
          char  *weight = cff_index_get_sid_string( cff,
                                                    dict->weight );


          if ( weight )
            if ( !ft_strcmp( weight, "Bold"  ) ||
                 !ft_strcmp( weight, "Black" ) )
              flags |= FT_STYLE_FLAG_BOLD;
        }

        /* double check */
        if ( !(flags & FT_STYLE_FLAG_BOLD) && cffface->style_name )
          if ( !ft_strncmp( cffface->style_name, "Bold", 4 )  ||
               !ft_strncmp( cffface->style_name, "Black", 5 ) )
            flags |= FT_STYLE_FLAG_BOLD;

        cffface->style_flags = flags;
      }

#ifndef FT_CONFIG_OPTION_NO_GLYPH_NAMES
      /* CID-keyed CFF or CFF2 fonts don't have glyph names -- the SFNT */
      /* loader has unset this flag because of the 3.0 `post' table.    */
      if ( dict->cid_registry == 0xFFFFU && !cff2 )
        cffface->face_flags |= FT_FACE_FLAG_GLYPH_NAMES;
#endif

      if ( dict->cid_registry != 0xFFFFU && pure_cff )
        cffface->face_flags |= FT_FACE_FLAG_CID_KEYED;

      /********************************************************************
       *
       * Compute char maps.
       */

      /* Try to synthesize a Unicode charmap if there is none available */
      /* already.  If an OpenType font contains a Unicode "cmap", we    */
      /* will use it, whatever be in the CFF part of the file.          */
      {
        FT_CharMapRec  cmaprec;
        FT_CharMap     cmap;
        FT_Int         nn;
        CFF_Encoding   encoding = &cff->encoding;


        for ( nn = 0; nn < cffface->num_charmaps; nn++ )
        {
          cmap = cffface->charmaps[nn];

          /* Windows Unicode? */
          if ( cmap->platform_id == TT_PLATFORM_MICROSOFT &&
               cmap->encoding_id == TT_MS_ID_UNICODE_CS   )
            goto Skip_Unicode;

          /* Apple Unicode platform id? */
          if ( cmap->platform_id == TT_PLATFORM_APPLE_UNICODE )
            goto Skip_Unicode; /* Apple Unicode */
        }

        /* since CID-keyed fonts don't contain glyph names, we can't */
        /* construct a cmap                                          */
        if ( pure_cff && cff->top_font.font_dict.cid_registry != 0xFFFFU )
          goto Exit;

        /* we didn't find a Unicode charmap -- synthesize one */
        cmaprec.face        = cffface;
        cmaprec.platform_id = TT_PLATFORM_MICROSOFT;
        cmaprec.encoding_id = TT_MS_ID_UNICODE_CS;
        cmaprec.encoding    = FT_ENCODING_UNICODE;

        nn = cffface->num_charmaps;

        error = FT_CMap_New( &cff_cmap_unicode_class_rec, NULL,
                             &cmaprec, NULL );
        if ( error                                      &&
             FT_ERR_NEQ( error, No_Unicode_Glyph_Name ) &&
             FT_ERR_NEQ( error, Unimplemented_Feature ) )
          goto Exit;
        error = FT_Err_Ok;

        /* if no Unicode charmap was previously selected, select this one */
        if ( !cffface->charmap && nn != cffface->num_charmaps )
          cffface->charmap = cffface->charmaps[nn];

      Skip_Unicode:
        if ( encoding->count > 0 )
        {
          FT_CMap_Class  clazz;


          cmaprec.face        = cffface;
          cmaprec.platform_id = TT_PLATFORM_ADOBE;  /* Adobe platform id */

          if ( encoding->offset == 0 )
          {
            cmaprec.encoding_id = TT_ADOBE_ID_STANDARD;
            cmaprec.encoding    = FT_ENCODING_ADOBE_STANDARD;
            clazz               = &cff_cmap_encoding_class_rec;
          }
          else if ( encoding->offset == 1 )
          {
            cmaprec.encoding_id = TT_ADOBE_ID_EXPERT;
            cmaprec.encoding    = FT_ENCODING_ADOBE_EXPERT;
            clazz               = &cff_cmap_encoding_class_rec;
          }
          else
          {
            cmaprec.encoding_id = TT_ADOBE_ID_CUSTOM;
            cmaprec.encoding    = FT_ENCODING_ADOBE_CUSTOM;
            clazz               = &cff_cmap_encoding_class_rec;
          }

          error = FT_CMap_New( clazz, NULL, &cmaprec, NULL );
        }
      }
    }

  Exit:
    return error;
  }


  FT_LOCAL_DEF( void )
  cff_face_done( FT_Face  cffface )         /* CFF_Face */
  {
    CFF_Face      face = (CFF_Face)cffface;
    FT_Memory     memory;
    SFNT_Service  sfnt;


    if ( !face )
      return;

    memory = cffface->memory;
    sfnt   = (SFNT_Service)face->sfnt;

    if ( sfnt )
      sfnt->done_face( face );

    {
      CFF_Font  cff = (CFF_Font)face->extra.data;


      if ( cff )
      {
        cff_font_done( cff );
        FT_FREE( face->extra.data );
      }
    }

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    cff_done_blend( face );
    face->blend = NULL;
#endif
  }


  FT_LOCAL_DEF( FT_Error )
  cff_driver_init( FT_Module  module )        /* CFF_Driver */
  {
    PS_Driver  driver = (PS_Driver)module;

    FT_UInt32  seed;


    /* set default property values, cf. `ftcffdrv.h' */
    driver->hinting_engine = FT_HINTING_ADOBE;

    driver->no_stem_darkening = TRUE;

    driver->darken_params[0] = CFF_CONFIG_OPTION_DARKENING_PARAMETER_X1;
    driver->darken_params[1] = CFF_CONFIG_OPTION_DARKENING_PARAMETER_Y1;
    driver->darken_params[2] = CFF_CONFIG_OPTION_DARKENING_PARAMETER_X2;
    driver->darken_params[3] = CFF_CONFIG_OPTION_DARKENING_PARAMETER_Y2;
    driver->darken_params[4] = CFF_CONFIG_OPTION_DARKENING_PARAMETER_X3;
    driver->darken_params[5] = CFF_CONFIG_OPTION_DARKENING_PARAMETER_Y3;
    driver->darken_params[6] = CFF_CONFIG_OPTION_DARKENING_PARAMETER_X4;
    driver->darken_params[7] = CFF_CONFIG_OPTION_DARKENING_PARAMETER_Y4;

    /* compute random seed from some memory addresses */
    seed = (FT_UInt32)( (FT_Offset)(char*)&seed          ^
                        (FT_Offset)(char*)&module        ^
                        (FT_Offset)(char*)module->memory );
    seed = seed ^ ( seed >> 10 ) ^ ( seed >> 20 );

    driver->random_seed = (FT_Int32)seed;
    if ( driver->random_seed < 0 )
      driver->random_seed = -driver->random_seed;
    else if ( driver->random_seed == 0 )
      driver->random_seed = 123456789;

    return FT_Err_Ok;
  }


  FT_LOCAL_DEF( void )
  cff_driver_done( FT_Module  module )        /* CFF_Driver */
  {
    FT_UNUSED( module );
  }


/* END */

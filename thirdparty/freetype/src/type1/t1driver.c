/****************************************************************************
 *
 * t1driver.c
 *
 *   Type 1 driver interface (body).
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


#include "t1driver.h"
#include "t1gload.h"
#include "t1load.h"

#include "t1errors.h"

#ifndef T1_CONFIG_OPTION_NO_AFM
#include "t1afm.h"
#endif

#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/fthash.h>
#include <freetype/internal/ftpsprop.h>
#include <freetype/ftdriver.h>

#include <freetype/internal/services/svmm.h>
#include <freetype/internal/services/svgldict.h>
#include <freetype/internal/services/svfntfmt.h>
#include <freetype/internal/services/svpostnm.h>
#include <freetype/internal/services/svpscmap.h>
#include <freetype/internal/services/svpsinfo.h>
#include <freetype/internal/services/svprop.h>
#include <freetype/internal/services/svkern.h>


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  t1driver

  /*
   * GLYPH DICT SERVICE
   *
   */

  FT_CALLBACK_DEF( FT_Error )
  t1_get_glyph_name( FT_Face     face,        /* T1_Face */
                     FT_UInt     glyph_index,
                     FT_Pointer  buffer,
                     FT_UInt     buffer_max )
  {
    T1_Face  t1face = (T1_Face)face;


    FT_STRCPYN( buffer, t1face->type1.glyph_names[glyph_index], buffer_max );

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( FT_UInt )
  t1_get_name_index( FT_Face           face,        /* T1_Face */
                     const FT_String*  glyph_name )
  {
    T1_Face  t1face = (T1_Face)face;
    FT_Int   i;


    for ( i = 0; i < t1face->type1.num_glyphs; i++ )
    {
      FT_String*  gname = t1face->type1.glyph_names[i];


      if ( !ft_strcmp( glyph_name, gname ) )
        return (FT_UInt)i;
    }

    return 0;
  }


  static const FT_Service_GlyphDictRec  t1_service_glyph_dict =
  {
    t1_get_glyph_name,  /* FT_GlyphDict_GetNameFunc   get_name   */
    t1_get_name_index   /* FT_GlyphDict_NameIndexFunc name_index */
  };


  /*
   * POSTSCRIPT NAME SERVICE
   *
   */

  static const char*
  t1_get_ps_name( FT_Face  face )    /* T1_Face */
  {
    T1_Face  t1face = (T1_Face)face;


    return (const char*) t1face->type1.font_name;
  }


  static const FT_Service_PsFontNameRec  t1_service_ps_name =
  {
    (FT_PsName_GetFunc)t1_get_ps_name     /* get_ps_font_name */
  };


  /*
   * MULTIPLE MASTERS SERVICE
   *
   */

#ifndef T1_CONFIG_OPTION_NO_MM_SUPPORT
  static const FT_Service_MultiMastersRec  t1_service_multi_masters =
  {
    T1_Get_Multi_Master,    /* FT_Get_MM_Func             get_mm             */
    T1_Set_MM_Design,       /* FT_Set_MM_Design_Func      set_mm_design      */
    T1_Set_MM_Blend,        /* FT_Set_MM_Blend_Func       set_mm_blend       */
    T1_Get_MM_Blend,        /* FT_Get_MM_Blend_Func       get_mm_blend       */
    T1_Get_MM_Var,          /* FT_Get_MM_Var_Func         get_mm_var         */
    T1_Set_Var_Design,      /* FT_Set_Var_Design_Func     set_var_design     */
    T1_Get_Var_Design,      /* FT_Get_Var_Design_Func     get_var_design     */
    T1_Reset_MM_Blend,      /* FT_Set_Named_Instance_Func set_named_instance */
    NULL,   /* FT_Get_Default_Named_Instance_Func get_default_named_instance */
    T1_Set_MM_WeightVector,
            /* FT_Set_MM_WeightVector_Func        set_mm_weightvector        */
    T1_Get_MM_WeightVector,
            /* FT_Get_MM_WeightVector_Func        get_mm_weightvector        */

    NULL,   /* FT_Construct_PS_Name_Func          construct_ps_name          */
    NULL,   /* FT_Var_Load_Delta_Set_Idx_Map_Func load_delta_set_idx_map     */
    NULL,   /* FT_Var_Load_Item_Var_Store_Func    load_item_variation_store  */
    NULL,   /* FT_Var_Get_Item_Delta_Func         get_item_delta             */
    NULL,   /* FT_Var_Done_Item_Var_Store_Func    done_item_variation_store  */
    NULL,   /* FT_Var_Done_Delta_Set_Idx_Map_Func done_delta_set_index_map   */
    NULL,           /* FT_Get_Var_Blend_Func      get_var_blend              */
    T1_Done_Blend   /* FT_Done_Blend_Func         done_blend                 */
  };
#endif


  /*
   * POSTSCRIPT INFO SERVICE
   *
   */

  static FT_Error
  t1_ps_get_font_info( FT_Face          face,
                       PS_FontInfoRec*  afont_info )
  {
    *afont_info = ((T1_Face)face)->type1.font_info;

    return FT_Err_Ok;
  }


  static FT_Error
  t1_ps_get_font_extra( FT_Face           face,
                        PS_FontExtraRec*  afont_extra )
  {
    *afont_extra = ((T1_Face)face)->type1.font_extra;

    return FT_Err_Ok;
  }


  static FT_Int
  t1_ps_has_glyph_names( FT_Face  face )
  {
    FT_UNUSED( face );

    return 1;
  }


  static FT_Error
  t1_ps_get_font_private( FT_Face         face,
                          PS_PrivateRec*  afont_private )
  {
    *afont_private = ((T1_Face)face)->type1.private_dict;

    return FT_Err_Ok;
  }


  static FT_Long
  t1_ps_get_font_value( FT_Face       face,
                        PS_Dict_Keys  key,
                        FT_UInt       idx,
                        void         *value,
                        FT_Long       value_len_ )
  {
    FT_ULong  retval    = 0; /* always >= 1 if valid */
    FT_ULong  value_len = value_len_ < 0 ? 0 : (FT_ULong)value_len_;

    T1_Face  t1face = (T1_Face)face;
    T1_Font  type1  = &t1face->type1;


    switch ( key )
    {
    case PS_DICT_FONT_TYPE:
      retval = sizeof ( type1->font_type );
      if ( value && value_len >= retval )
        *((FT_Byte *)value) = type1->font_type;
      break;

    case PS_DICT_FONT_MATRIX:
      if ( idx < sizeof ( type1->font_matrix ) /
                   sizeof ( type1->font_matrix.xx ) )
      {
        FT_Fixed  val = 0;


        retval = sizeof ( val );
        if ( value && value_len >= retval )
        {
          switch ( idx )
          {
          case 0:
            val = type1->font_matrix.xx;
            break;
          case 1:
            val = type1->font_matrix.xy;
            break;
          case 2:
            val = type1->font_matrix.yx;
            break;
          case 3:
            val = type1->font_matrix.yy;
            break;
          }
          *((FT_Fixed *)value) = val;
        }
      }
      break;

    case PS_DICT_FONT_BBOX:
      if ( idx < sizeof ( type1->font_bbox ) /
                   sizeof ( type1->font_bbox.xMin ) )
      {
        FT_Fixed  val = 0;


        retval = sizeof ( val );
        if ( value && value_len >= retval )
        {
          switch ( idx )
          {
          case 0:
            val = type1->font_bbox.xMin;
            break;
          case 1:
            val = type1->font_bbox.yMin;
            break;
          case 2:
            val = type1->font_bbox.xMax;
            break;
          case 3:
            val = type1->font_bbox.yMax;
            break;
          }
          *((FT_Fixed *)value) = val;
        }
      }
      break;

    case PS_DICT_PAINT_TYPE:
      retval = sizeof ( type1->paint_type );
      if ( value && value_len >= retval )
        *((FT_Byte *)value) = type1->paint_type;
      break;

    case PS_DICT_FONT_NAME:
      if ( type1->font_name )
      {
        retval = ft_strlen( type1->font_name ) + 1;
        if ( value && value_len >= retval )
          ft_memcpy( value, (void *)( type1->font_name ), retval );
      }
      break;

    case PS_DICT_UNIQUE_ID:
      retval = sizeof ( type1->private_dict.unique_id );
      if ( value && value_len >= retval )
        *((FT_Int *)value) = type1->private_dict.unique_id;
      break;

    case PS_DICT_NUM_CHAR_STRINGS:
      retval = sizeof ( type1->num_glyphs );
      if ( value && value_len >= retval )
        *((FT_Int *)value) = type1->num_glyphs;
      break;

    case PS_DICT_CHAR_STRING_KEY:
      if ( idx < (FT_UInt)type1->num_glyphs )
      {
        retval = ft_strlen( type1->glyph_names[idx] ) + 1;
        if ( value && value_len >= retval )
          ft_memcpy( value, (void *)( type1->glyph_names[idx] ), retval );
      }
      break;

    case PS_DICT_CHAR_STRING:
      if ( idx < (FT_UInt)type1->num_glyphs )
      {
        retval = type1->charstrings_len[idx] + 1;
        if ( value && value_len >= retval )
        {
          ft_memcpy( value, (void *)( type1->charstrings[idx] ),
                     retval - 1 );
          ((FT_Char *)value)[retval - 1] = (FT_Char)'\0';
        }
      }
      break;

    case PS_DICT_ENCODING_TYPE:
      retval = sizeof ( type1->encoding_type );
      if ( value && value_len >= retval )
        *((T1_EncodingType *)value) = type1->encoding_type;
      break;

    case PS_DICT_ENCODING_ENTRY:
      if ( type1->encoding_type == T1_ENCODING_TYPE_ARRAY &&
           idx < (FT_UInt)type1->encoding.num_chars       )
      {
        retval = ft_strlen( type1->encoding.char_name[idx] ) + 1;
        if ( value && value_len >= retval )
          ft_memcpy( value, (void *)( type1->encoding.char_name[idx] ),
                     retval );
      }
      break;

    case PS_DICT_NUM_SUBRS:
      retval = sizeof ( type1->num_subrs );
      if ( value && value_len >= retval )
        *((FT_Int *)value) = type1->num_subrs;
      break;

    case PS_DICT_SUBR:
      {
        FT_Bool  ok = 0;


        if ( type1->subrs_hash )
        {
          /* convert subr index to array index */
          size_t*  val = ft_hash_num_lookup( (FT_Int)idx,
                                             type1->subrs_hash );


          if ( val )
          {
            idx = *val;
            ok  = 1;
          }
        }
        else
        {
          if ( idx < (FT_UInt)type1->num_subrs )
            ok = 1;
        }

        if ( ok && type1->subrs )
        {
          retval = type1->subrs_len[idx] + 1;
          if ( value && value_len >= retval )
          {
            ft_memcpy( value, (void *)( type1->subrs[idx] ), retval - 1 );
            ((FT_Char *)value)[retval - 1] = (FT_Char)'\0';
          }
        }
      }
      break;

    case PS_DICT_STD_HW:
      retval = sizeof ( type1->private_dict.standard_width[0] );
      if ( value && value_len >= retval )
        *((FT_UShort *)value) = type1->private_dict.standard_width[0];
      break;

    case PS_DICT_STD_VW:
      retval = sizeof ( type1->private_dict.standard_height[0] );
      if ( value && value_len >= retval )
        *((FT_UShort *)value) = type1->private_dict.standard_height[0];
      break;

    case PS_DICT_NUM_BLUE_VALUES:
      retval = sizeof ( type1->private_dict.num_blue_values );
      if ( value && value_len >= retval )
        *((FT_Byte *)value) = type1->private_dict.num_blue_values;
      break;

    case PS_DICT_BLUE_VALUE:
      if ( idx < type1->private_dict.num_blue_values )
      {
        retval = sizeof ( type1->private_dict.blue_values[idx] );
        if ( value && value_len >= retval )
          *((FT_Short *)value) = type1->private_dict.blue_values[idx];
      }
      break;

    case PS_DICT_BLUE_SCALE:
      retval = sizeof ( type1->private_dict.blue_scale );
      if ( value && value_len >= retval )
        *((FT_Fixed *)value) = type1->private_dict.blue_scale;
      break;

    case PS_DICT_BLUE_FUZZ:
      retval = sizeof ( type1->private_dict.blue_fuzz );
      if ( value && value_len >= retval )
        *((FT_Int *)value) = type1->private_dict.blue_fuzz;
      break;

    case PS_DICT_BLUE_SHIFT:
      retval = sizeof ( type1->private_dict.blue_shift );
      if ( value && value_len >= retval )
        *((FT_Int *)value) = type1->private_dict.blue_shift;
      break;

    case PS_DICT_NUM_OTHER_BLUES:
      retval = sizeof ( type1->private_dict.num_other_blues );
      if ( value && value_len >= retval )
        *((FT_Byte *)value) = type1->private_dict.num_other_blues;
      break;

    case PS_DICT_OTHER_BLUE:
      if ( idx < type1->private_dict.num_other_blues )
      {
        retval = sizeof ( type1->private_dict.other_blues[idx] );
        if ( value && value_len >= retval )
          *((FT_Short *)value) = type1->private_dict.other_blues[idx];
      }
      break;

    case PS_DICT_NUM_FAMILY_BLUES:
      retval = sizeof ( type1->private_dict.num_family_blues );
      if ( value && value_len >= retval )
        *((FT_Byte *)value) = type1->private_dict.num_family_blues;
      break;

    case PS_DICT_FAMILY_BLUE:
      if ( idx < type1->private_dict.num_family_blues )
      {
        retval = sizeof ( type1->private_dict.family_blues[idx] );
        if ( value && value_len >= retval )
          *((FT_Short *)value) = type1->private_dict.family_blues[idx];
      }
      break;

    case PS_DICT_NUM_FAMILY_OTHER_BLUES:
      retval = sizeof ( type1->private_dict.num_family_other_blues );
      if ( value && value_len >= retval )
        *((FT_Byte *)value) = type1->private_dict.num_family_other_blues;
      break;

    case PS_DICT_FAMILY_OTHER_BLUE:
      if ( idx < type1->private_dict.num_family_other_blues )
      {
        retval = sizeof ( type1->private_dict.family_other_blues[idx] );
        if ( value && value_len >= retval )
          *((FT_Short *)value) = type1->private_dict.family_other_blues[idx];
      }
      break;

    case PS_DICT_NUM_STEM_SNAP_H:
      retval = sizeof ( type1->private_dict.num_snap_widths );
      if ( value && value_len >= retval )
        *((FT_Byte *)value) = type1->private_dict.num_snap_widths;
      break;

    case PS_DICT_STEM_SNAP_H:
      if ( idx < type1->private_dict.num_snap_widths )
      {
        retval = sizeof ( type1->private_dict.snap_widths[idx] );
        if ( value && value_len >= retval )
          *((FT_Short *)value) = type1->private_dict.snap_widths[idx];
      }
      break;

    case PS_DICT_NUM_STEM_SNAP_V:
      retval = sizeof ( type1->private_dict.num_snap_heights );
      if ( value && value_len >= retval )
        *((FT_Byte *)value) = type1->private_dict.num_snap_heights;
      break;

    case PS_DICT_STEM_SNAP_V:
      if ( idx < type1->private_dict.num_snap_heights )
      {
        retval = sizeof ( type1->private_dict.snap_heights[idx] );
        if ( value && value_len >= retval )
          *((FT_Short *)value) = type1->private_dict.snap_heights[idx];
      }
      break;

    case PS_DICT_RND_STEM_UP:
      retval = sizeof ( type1->private_dict.round_stem_up );
      if ( value && value_len >= retval )
        *((FT_Bool *)value) = type1->private_dict.round_stem_up;
      break;

    case PS_DICT_FORCE_BOLD:
      retval = sizeof ( type1->private_dict.force_bold );
      if ( value && value_len >= retval )
        *((FT_Bool *)value) = type1->private_dict.force_bold;
      break;

    case PS_DICT_MIN_FEATURE:
      if ( idx < sizeof ( type1->private_dict.min_feature ) /
                   sizeof ( type1->private_dict.min_feature[0] ) )
      {
        retval = sizeof ( type1->private_dict.min_feature[idx] );
        if ( value && value_len >= retval )
          *((FT_Short *)value) = type1->private_dict.min_feature[idx];
      }
      break;

    case PS_DICT_LEN_IV:
      retval = sizeof ( type1->private_dict.lenIV );
      if ( value && value_len >= retval )
        *((FT_Int *)value) = type1->private_dict.lenIV;
      break;

    case PS_DICT_PASSWORD:
      retval = sizeof ( type1->private_dict.password );
      if ( value && value_len >= retval )
        *((FT_Long *)value) = type1->private_dict.password;
      break;

    case PS_DICT_LANGUAGE_GROUP:
      retval = sizeof ( type1->private_dict.language_group );
      if ( value && value_len >= retval )
        *((FT_Long *)value) = type1->private_dict.language_group;
      break;

    case PS_DICT_IS_FIXED_PITCH:
      retval = sizeof ( type1->font_info.is_fixed_pitch );
      if ( value && value_len >= retval )
        *((FT_Bool *)value) = type1->font_info.is_fixed_pitch;
      break;

    case PS_DICT_UNDERLINE_POSITION:
      retval = sizeof ( type1->font_info.underline_position );
      if ( value && value_len >= retval )
        *((FT_Short *)value) = type1->font_info.underline_position;
      break;

    case PS_DICT_UNDERLINE_THICKNESS:
      retval = sizeof ( type1->font_info.underline_thickness );
      if ( value && value_len >= retval )
        *((FT_UShort *)value) = type1->font_info.underline_thickness;
      break;

    case PS_DICT_FS_TYPE:
      retval = sizeof ( type1->font_extra.fs_type );
      if ( value && value_len >= retval )
        *((FT_UShort *)value) = type1->font_extra.fs_type;
      break;

    case PS_DICT_VERSION:
      if ( type1->font_info.version )
      {
        retval = ft_strlen( type1->font_info.version ) + 1;
        if ( value && value_len >= retval )
          ft_memcpy( value, (void *)( type1->font_info.version ), retval );
      }
      break;

    case PS_DICT_NOTICE:
      if ( type1->font_info.notice )
      {
        retval = ft_strlen( type1->font_info.notice ) + 1;
        if ( value && value_len >= retval )
          ft_memcpy( value, (void *)( type1->font_info.notice ), retval );
      }
      break;

    case PS_DICT_FULL_NAME:
      if ( type1->font_info.full_name )
      {
        retval = ft_strlen( type1->font_info.full_name ) + 1;
        if ( value && value_len >= retval )
          ft_memcpy( value, (void *)( type1->font_info.full_name ), retval );
      }
      break;

    case PS_DICT_FAMILY_NAME:
      if ( type1->font_info.family_name )
      {
        retval = ft_strlen( type1->font_info.family_name ) + 1;
        if ( value && value_len >= retval )
          ft_memcpy( value, (void *)( type1->font_info.family_name ),
                     retval );
      }
      break;

    case PS_DICT_WEIGHT:
      if ( type1->font_info.weight )
      {
        retval = ft_strlen( type1->font_info.weight ) + 1;
        if ( value && value_len >= retval )
          ft_memcpy( value, (void *)( type1->font_info.weight ), retval );
      }
      break;

    case PS_DICT_ITALIC_ANGLE:
      retval = sizeof ( type1->font_info.italic_angle );
      if ( value && value_len >= retval )
        *((FT_Long *)value) = type1->font_info.italic_angle;
      break;
    }

    return retval == 0 ? -1 : (FT_Long)retval;
  }


  static const FT_Service_PsInfoRec  t1_service_ps_info =
  {
    t1_ps_get_font_info,     /* PS_GetFontInfoFunc    ps_get_font_info    */
    t1_ps_get_font_extra,    /* PS_GetFontExtraFunc   ps_get_font_extra   */
    t1_ps_has_glyph_names,   /* PS_HasGlyphNamesFunc  ps_has_glyph_names  */
    t1_ps_get_font_private,  /* PS_GetFontPrivateFunc ps_get_font_private */
    t1_ps_get_font_value,    /* PS_GetFontValueFunc   ps_get_font_value   */
  };


#ifndef T1_CONFIG_OPTION_NO_AFM
  static const FT_Service_KerningRec  t1_service_kerning =
  {
    T1_Get_Track_Kerning,       /* get_track */
  };
#endif


  /*
   * PROPERTY SERVICE
   *
   */

  FT_DEFINE_SERVICE_PROPERTIESREC(
    t1_service_properties,

    ps_property_set,  /* FT_Properties_SetFunc set_property */
    ps_property_get   /* FT_Properties_GetFunc get_property */
  )

  /*
   * SERVICE LIST
   *
   */

  static const FT_ServiceDescRec  t1_services[] =
  {
    { FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &t1_service_ps_name },
    { FT_SERVICE_ID_GLYPH_DICT,           &t1_service_glyph_dict },
    { FT_SERVICE_ID_FONT_FORMAT,          FT_FONT_FORMAT_TYPE_1 },
    { FT_SERVICE_ID_POSTSCRIPT_INFO,      &t1_service_ps_info },
    { FT_SERVICE_ID_PROPERTIES,           &t1_service_properties },

#ifndef T1_CONFIG_OPTION_NO_AFM
    { FT_SERVICE_ID_KERNING,              &t1_service_kerning },
#endif

#ifndef T1_CONFIG_OPTION_NO_MM_SUPPORT
    { FT_SERVICE_ID_MULTI_MASTERS,        &t1_service_multi_masters },
#endif
    { NULL, NULL }
  };


  FT_CALLBACK_DEF( FT_Module_Interface )
  Get_Interface( FT_Module         module,
                 const FT_String*  t1_interface )
  {
    FT_UNUSED( module );

    return ft_service_list_lookup( t1_services, t1_interface );
  }


#ifndef T1_CONFIG_OPTION_NO_AFM

  /**************************************************************************
   *
   * @Function:
   *   Get_Kerning
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
   *   kernings are out of scope of this method (the basic driver
   *   interface is meant to be simple).
   *
   *   They can be implemented by format-specific interfaces.
   */
  static FT_Error
  Get_Kerning( FT_Face     t1face,        /* T1_Face */
               FT_UInt     left_glyph,
               FT_UInt     right_glyph,
               FT_Vector*  kerning )
  {
    T1_Face  face = (T1_Face)t1face;


    kerning->x = 0;
    kerning->y = 0;

    if ( face->afm_data )
      T1_Get_Kerning( (AFM_FontInfo)face->afm_data,
                      left_glyph,
                      right_glyph,
                      kerning );

    return FT_Err_Ok;
  }


#endif /* T1_CONFIG_OPTION_NO_AFM */


  FT_CALLBACK_TABLE_DEF
  const FT_Driver_ClassRec  t1_driver_class =
  {
    {
      FT_MODULE_FONT_DRIVER       |
      FT_MODULE_DRIVER_SCALABLE   |
      FT_MODULE_DRIVER_HAS_HINTER,

      sizeof ( PS_DriverRec ),

      "type1",
      0x10000L,
      0x20000L,

      NULL,    /* module-specific interface */

      T1_Driver_Init,           /* FT_Module_Constructor  module_init   */
      T1_Driver_Done,           /* FT_Module_Destructor   module_done   */
      Get_Interface,            /* FT_Module_Requester    get_interface */
    },

    sizeof ( T1_FaceRec ),
    sizeof ( T1_SizeRec ),
    sizeof ( T1_GlyphSlotRec ),

    T1_Face_Init,               /* FT_Face_InitFunc  init_face */
    T1_Face_Done,               /* FT_Face_DoneFunc  done_face */
    T1_Size_Init,               /* FT_Size_InitFunc  init_size */
    T1_Size_Done,               /* FT_Size_DoneFunc  done_size */
    T1_GlyphSlot_Init,          /* FT_Slot_InitFunc  init_slot */
    T1_GlyphSlot_Done,          /* FT_Slot_DoneFunc  done_slot */

    T1_Load_Glyph,              /* FT_Slot_LoadFunc  load_glyph */

#ifdef T1_CONFIG_OPTION_NO_AFM
    NULL,                       /* FT_Face_GetKerningFunc   get_kerning  */
    NULL,                       /* FT_Face_AttachFunc       attach_file  */
#else
    Get_Kerning,                /* FT_Face_GetKerningFunc   get_kerning  */
    T1_Read_Metrics,            /* FT_Face_AttachFunc       attach_file  */
#endif
    T1_Get_Advances,            /* FT_Face_GetAdvancesFunc  get_advances */

    T1_Size_Request,            /* FT_Size_RequestFunc  request_size */
    NULL                        /* FT_Size_SelectFunc   select_size  */
  };


/* END */

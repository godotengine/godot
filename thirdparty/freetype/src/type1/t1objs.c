/****************************************************************************
 *
 * t1objs.c
 *
 *   Type 1 objects manager (body).
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


#include <freetype/internal/ftcalc.h>
#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/ttnameid.h>
#include <freetype/ftdriver.h>

#include "t1gload.h"
#include "t1load.h"

#include "t1errors.h"

#ifndef T1_CONFIG_OPTION_NO_AFM
#include "t1afm.h"
#endif

#include <freetype/internal/services/svpscmap.h>
#include <freetype/internal/psaux.h>


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  t1objs


  /**************************************************************************
   *
   *                           SIZE FUNCTIONS
   *
   */


  static PSH_Globals_Funcs
  T1_Size_Get_Globals_Funcs( T1_Size  size )
  {
    T1_Face           face     = (T1_Face)size->root.face;
    PSHinter_Service  pshinter = (PSHinter_Service)face->pshinter;
    FT_Module         module;


    module = FT_Get_Module( size->root.face->driver->root.library,
                            "pshinter" );
    return ( module && pshinter && pshinter->get_globals_funcs )
           ? pshinter->get_globals_funcs( module )
           : 0;
  }


  FT_LOCAL_DEF( void )
  T1_Size_Done( FT_Size  t1size )          /* T1_Size */
  {
    T1_Size  size = (T1_Size)t1size;


    if ( t1size->internal->module_data )
    {
      PSH_Globals_Funcs  funcs;


      funcs = T1_Size_Get_Globals_Funcs( size );
      if ( funcs )
        funcs->destroy( (PSH_Globals)t1size->internal->module_data );

      t1size->internal->module_data = NULL;
    }
  }


  FT_LOCAL_DEF( FT_Error )
  T1_Size_Init( FT_Size  t1size )      /* T1_Size */
  {
    T1_Size            size  = (T1_Size)t1size;
    FT_Error           error = FT_Err_Ok;
    PSH_Globals_Funcs  funcs = T1_Size_Get_Globals_Funcs( size );


    if ( funcs )
    {
      PSH_Globals  globals;
      T1_Face      face = (T1_Face)size->root.face;


      error = funcs->create( size->root.face->memory,
                             &face->type1.private_dict, &globals );
      if ( !error )
        t1size->internal->module_data = globals;
    }

    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  T1_Size_Request( FT_Size          t1size,     /* T1_Size */
                   FT_Size_Request  req )
  {
    T1_Size            size  = (T1_Size)t1size;
    PSH_Globals_Funcs  funcs = T1_Size_Get_Globals_Funcs( size );


    FT_Request_Metrics( size->root.face, req );

    if ( funcs )
      funcs->set_scale( (PSH_Globals)t1size->internal->module_data,
                        size->root.metrics.x_scale,
                        size->root.metrics.y_scale,
                        0, 0 );

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   *                           SLOT  FUNCTIONS
   *
   */

  FT_LOCAL_DEF( void )
  T1_GlyphSlot_Done( FT_GlyphSlot  slot )
  {
    slot->internal->glyph_hints = NULL;
  }


  FT_LOCAL_DEF( FT_Error )
  T1_GlyphSlot_Init( FT_GlyphSlot  slot )
  {
    T1_Face           face;
    PSHinter_Service  pshinter;


    face     = (T1_Face)slot->face;
    pshinter = (PSHinter_Service)face->pshinter;

    if ( pshinter )
    {
      FT_Module  module;


      module = FT_Get_Module( slot->face->driver->root.library,
                              "pshinter" );
      if ( module )
      {
        T1_Hints_Funcs  funcs;


        funcs = pshinter->get_t1_funcs( module );
        slot->internal->glyph_hints = (void*)funcs;
      }
    }

    return 0;
  }


  /**************************************************************************
   *
   *                           FACE  FUNCTIONS
   *
   */


  /**************************************************************************
   *
   * @Function:
   *   T1_Face_Done
   *
   * @Description:
   *   The face object destructor.
   *
   * @Input:
   *   face ::
   *     A typeless pointer to the face object to destroy.
   */
  FT_LOCAL_DEF( void )
  T1_Face_Done( FT_Face  t1face )         /* T1_Face */
  {
    T1_Face    face = (T1_Face)t1face;
    FT_Memory  memory;
    T1_Font    type1;


    if ( !face )
      return;

    memory = face->root.memory;
    type1  = &face->type1;

#ifndef T1_CONFIG_OPTION_NO_MM_SUPPORT
    /* release multiple masters information */
    FT_ASSERT( ( face->len_buildchar == 0 ) == ( face->buildchar == NULL ) );

    if ( face->buildchar )
    {
      FT_FREE( face->buildchar );

      face->buildchar     = NULL;
      face->len_buildchar = 0;
    }

    T1_Done_Blend( face );
    face->blend = NULL;
#endif

    /* release font info strings */
    {
      PS_FontInfo  info = &type1->font_info;


      FT_FREE( info->version );
      FT_FREE( info->notice );
      FT_FREE( info->full_name );
      FT_FREE( info->family_name );
      FT_FREE( info->weight );
    }

    /* release top dictionary */
    FT_FREE( type1->charstrings_len );
    FT_FREE( type1->charstrings );
    FT_FREE( type1->glyph_names );

    FT_FREE( type1->subrs );
    FT_FREE( type1->subrs_len );

    ft_hash_num_free( type1->subrs_hash, memory );
    FT_FREE( type1->subrs_hash );

    FT_FREE( type1->subrs_block );
    FT_FREE( type1->charstrings_block );
    FT_FREE( type1->glyph_names_block );

    FT_FREE( type1->encoding.char_index );
    FT_FREE( type1->encoding.char_name );
    FT_FREE( type1->font_name );

#ifndef T1_CONFIG_OPTION_NO_AFM
    /* release afm data if present */
    if ( face->afm_data )
      T1_Done_Metrics( memory, (AFM_FontInfo)face->afm_data );
#endif

    /* release unicode map, if any */
#if 0
    FT_FREE( face->unicode_map_rec.maps );
    face->unicode_map_rec.num_maps = 0;
    face->unicode_map              = NULL;
#endif

    face->root.family_name = NULL;
    face->root.style_name  = NULL;
  }


  /**************************************************************************
   *
   * @Function:
   *   T1_Face_Init
   *
   * @Description:
   *   The face object constructor.
   *
   * @Input:
   *   stream ::
   *     input stream where to load font data.
   *
   *   face_index ::
   *     The index of the font face in the resource.
   *
   *   num_params ::
   *     Number of additional generic parameters.  Ignored.
   *
   *   params ::
   *     Additional generic parameters.  Ignored.
   *
   * @InOut:
   *   face ::
   *     The face record to build.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  T1_Face_Init( FT_Stream      stream,
                FT_Face        t1face,          /* T1_Face */
                FT_Int         face_index,
                FT_Int         num_params,
                FT_Parameter*  params )
  {
    T1_Face             face = (T1_Face)t1face;
    FT_Error            error;
    FT_Service_PsCMaps  psnames;
    PSAux_Service       psaux;
    T1_Font             type1 = &face->type1;
    PS_FontInfo         info = &type1->font_info;

    FT_UNUSED( num_params );
    FT_UNUSED( params );
    FT_UNUSED( stream );


    face->root.num_faces = 1;

    FT_FACE_FIND_GLOBAL_SERVICE( face, psnames, POSTSCRIPT_CMAPS );
    face->psnames = psnames;

    face->psaux = FT_Get_Module_Interface( FT_FACE_LIBRARY( face ),
                                           "psaux" );
    psaux = (PSAux_Service)face->psaux;
    if ( !psaux )
    {
      FT_ERROR(( "T1_Face_Init: cannot access `psaux' module\n" ));
      error = FT_THROW( Missing_Module );
      goto Exit;
    }

    face->pshinter = FT_Get_Module_Interface( FT_FACE_LIBRARY( face ),
                                              "pshinter" );

    FT_TRACE2(( "Type 1 driver\n" ));

    /* open the tokenizer; this will also check the font format */
    error = T1_Open_Face( face );
    if ( error )
      goto Exit;

    FT_TRACE2(( "T1_Face_Init: %p (index %d)\n",
                (void *)face,
                face_index ));

    /* if we just wanted to check the format, leave successfully now */
    if ( face_index < 0 )
      goto Exit;

    /* check the face index */
    if ( ( face_index & 0xFFFF ) > 0 )
    {
      FT_ERROR(( "T1_Face_Init: invalid face index\n" ));
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* now load the font program into the face object */

    /* initialize the face object fields */

    /* set up root face fields */
    {
      FT_Face  root = (FT_Face)&face->root;


      root->num_glyphs = type1->num_glyphs;
      root->face_index = 0;

      root->face_flags |= FT_FACE_FLAG_SCALABLE    |
                          FT_FACE_FLAG_HORIZONTAL  |
                          FT_FACE_FLAG_GLYPH_NAMES |
                          FT_FACE_FLAG_HINTER;

      if ( info->is_fixed_pitch )
        root->face_flags |= FT_FACE_FLAG_FIXED_WIDTH;

      if ( face->blend )
        root->face_flags |= FT_FACE_FLAG_MULTIPLE_MASTERS;

      /* The following code to extract the family and the style is very   */
      /* simplistic and might get some things wrong.  For a full-featured */
      /* algorithm you might have a look at the whitepaper given at       */
      /*                                                                  */
      /*   https://blogs.msdn.com/text/archive/2007/04/23/wpf-font-selection-model.aspx */

      /* get style name -- be careful, some broken fonts only */
      /* have a `/FontName' dictionary entry!                 */
      root->family_name = info->family_name;
      root->style_name  = NULL;

      if ( root->family_name )
      {
        char*  full   = info->full_name;
        char*  family = root->family_name;


        if ( full )
        {
          FT_Bool  the_same = TRUE;


          while ( *full )
          {
            if ( *full == *family )
            {
              family++;
              full++;
            }
            else
            {
              if ( *full == ' ' || *full == '-' )
                full++;
              else if ( *family == ' ' || *family == '-' )
                family++;
              else
              {
                the_same = FALSE;

                if ( !*family )
                  root->style_name = full;
                break;
              }
            }
          }

          if ( the_same )
            root->style_name = (char *)"Regular";
        }
      }
      else
      {
        /* do we have a `/FontName'? */
        if ( type1->font_name )
          root->family_name = type1->font_name;
      }

      if ( !root->style_name )
      {
        if ( info->weight )
          root->style_name = info->weight;
        else
          /* assume `Regular' style because we don't know better */
          root->style_name = (char *)"Regular";
      }

      /* compute style flags */
      root->style_flags = 0;
      if ( info->italic_angle )
        root->style_flags |= FT_STYLE_FLAG_ITALIC;
      if ( info->weight )
      {
        if ( !ft_strcmp( info->weight, "Bold"  ) ||
             !ft_strcmp( info->weight, "Black" ) )
          root->style_flags |= FT_STYLE_FLAG_BOLD;
      }

      /* no embedded bitmap support */
      root->num_fixed_sizes = 0;
      root->available_sizes = NULL;

      root->bbox.xMin =   type1->font_bbox.xMin            >> 16;
      root->bbox.yMin =   type1->font_bbox.yMin            >> 16;
      /* no `U' suffix here to 0xFFFF! */
      root->bbox.xMax = ( type1->font_bbox.xMax + 0xFFFF ) >> 16;
      root->bbox.yMax = ( type1->font_bbox.yMax + 0xFFFF ) >> 16;

      /* Set units_per_EM if we didn't set it in t1_parse_font_matrix. */
      if ( !root->units_per_EM )
        root->units_per_EM = 1000;

      root->ascender  = (FT_Short)( root->bbox.yMax );
      root->descender = (FT_Short)( root->bbox.yMin );

      root->height = (FT_Short)( ( root->units_per_EM * 12 ) / 10 );
      if ( root->height < root->ascender - root->descender )
        root->height = (FT_Short)( root->ascender - root->descender );

      /* now compute the maximum advance width */
      root->max_advance_width =
        (FT_Short)( root->bbox.xMax );
      {
        FT_Pos  max_advance;


        error = T1_Compute_Max_Advance( face, &max_advance );

        /* in case of error, keep the standard width */
        if ( !error )
          root->max_advance_width = (FT_Short)FIXED_TO_INT( max_advance );
        else
          error = FT_Err_Ok;   /* clear error */
      }

      root->max_advance_height = root->height;

      root->underline_position  = (FT_Short)info->underline_position;
      root->underline_thickness = (FT_Short)info->underline_thickness;
    }

    {
      FT_Face  root = &face->root;


      if ( psnames )
      {
        FT_CharMapRec    charmap;
        T1_CMap_Classes  cmap_classes = psaux->t1_cmap_classes;
        FT_CMap_Class    clazz;


        charmap.face = root;

        /* first of all, try to synthesize a Unicode charmap */
        charmap.platform_id = TT_PLATFORM_MICROSOFT;
        charmap.encoding_id = TT_MS_ID_UNICODE_CS;
        charmap.encoding    = FT_ENCODING_UNICODE;

        error = FT_CMap_New( cmap_classes->unicode, NULL, &charmap, NULL );
        if ( error                                      &&
             FT_ERR_NEQ( error, No_Unicode_Glyph_Name ) &&
             FT_ERR_NEQ( error, Unimplemented_Feature ) )
          goto Exit;
        error = FT_Err_Ok;

        /* now, generate an Adobe Standard encoding when appropriate */
        charmap.platform_id = TT_PLATFORM_ADOBE;
        clazz               = NULL;

        switch ( type1->encoding_type )
        {
        case T1_ENCODING_TYPE_STANDARD:
          charmap.encoding    = FT_ENCODING_ADOBE_STANDARD;
          charmap.encoding_id = TT_ADOBE_ID_STANDARD;
          clazz               = cmap_classes->standard;
          break;

        case T1_ENCODING_TYPE_EXPERT:
          charmap.encoding    = FT_ENCODING_ADOBE_EXPERT;
          charmap.encoding_id = TT_ADOBE_ID_EXPERT;
          clazz               = cmap_classes->expert;
          break;

        case T1_ENCODING_TYPE_ARRAY:
          charmap.encoding    = FT_ENCODING_ADOBE_CUSTOM;
          charmap.encoding_id = TT_ADOBE_ID_CUSTOM;
          clazz               = cmap_classes->custom;
          break;

        case T1_ENCODING_TYPE_ISOLATIN1:
          charmap.encoding    = FT_ENCODING_ADOBE_LATIN_1;
          charmap.encoding_id = TT_ADOBE_ID_LATIN_1;
          clazz               = cmap_classes->unicode;
          break;

        default:
          ;
        }

        if ( clazz )
          error = FT_CMap_New( clazz, NULL, &charmap, NULL );
      }
    }

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   T1_Driver_Init
   *
   * @Description:
   *   Initializes a given Type 1 driver object.
   *
   * @Input:
   *   driver ::
   *     A handle to the target driver object.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  T1_Driver_Init( FT_Module  module )
  {
    PS_Driver  driver = (PS_Driver)module;

    FT_UInt32  seed;


    /* set default property values, cf. `ftt1drv.h' */
#ifdef T1_CONFIG_OPTION_OLD_ENGINE
    driver->hinting_engine = FT_HINTING_FREETYPE;
#else
    driver->hinting_engine = FT_HINTING_ADOBE;
#endif

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


  /**************************************************************************
   *
   * @Function:
   *   T1_Driver_Done
   *
   * @Description:
   *   Finalizes a given Type 1 driver.
   *
   * @Input:
   *   driver ::
   *     A handle to the target Type 1 driver.
   */
  FT_LOCAL_DEF( void )
  T1_Driver_Done( FT_Module  driver )
  {
    FT_UNUSED( driver );
  }


/* END */

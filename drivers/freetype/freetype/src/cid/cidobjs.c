/***************************************************************************/
/*                                                                         */
/*  cidobjs.c                                                              */
/*                                                                         */
/*    CID objects manager (body).                                          */
/*                                                                         */
/*  Copyright 1996-2006, 2008, 2010-2011, 2013 by                          */
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
#include FT_INTERNAL_STREAM_H

#include "cidgload.h"
#include "cidload.h"

#include FT_SERVICE_POSTSCRIPT_CMAPS_H
#include FT_INTERNAL_POSTSCRIPT_AUX_H
#include FT_INTERNAL_POSTSCRIPT_HINTS_H

#include "ciderrs.h"


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_cidobjs


  /*************************************************************************/
  /*                                                                       */
  /*                            SLOT  FUNCTIONS                            */
  /*                                                                       */
  /*************************************************************************/

  FT_LOCAL_DEF( void )
  cid_slot_done( FT_GlyphSlot  slot )
  {
    slot->internal->glyph_hints = 0;
  }


  FT_LOCAL_DEF( FT_Error )
  cid_slot_init( FT_GlyphSlot  slot )
  {
    CID_Face          face;
    PSHinter_Service  pshinter;


    face     = (CID_Face)slot->face;
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


  /*************************************************************************/
  /*                                                                       */
  /*                           SIZE  FUNCTIONS                             */
  /*                                                                       */
  /*************************************************************************/


  static PSH_Globals_Funcs
  cid_size_get_globals_funcs( CID_Size  size )
  {
    CID_Face          face     = (CID_Face)size->root.face;
    PSHinter_Service  pshinter = (PSHinter_Service)face->pshinter;
    FT_Module         module;


    module = FT_Get_Module( size->root.face->driver->root.library,
                            "pshinter" );
    return ( module && pshinter && pshinter->get_globals_funcs )
           ? pshinter->get_globals_funcs( module )
           : 0;
  }


  FT_LOCAL_DEF( void )
  cid_size_done( FT_Size  cidsize )         /* CID_Size */
  {
    CID_Size  size = (CID_Size)cidsize;


    if ( cidsize->internal )
    {
      PSH_Globals_Funcs  funcs;


      funcs = cid_size_get_globals_funcs( size );
      if ( funcs )
        funcs->destroy( (PSH_Globals)cidsize->internal );

      cidsize->internal = 0;
    }
  }


  FT_LOCAL_DEF( FT_Error )
  cid_size_init( FT_Size  cidsize )     /* CID_Size */
  {
    CID_Size           size  = (CID_Size)cidsize;
    FT_Error           error = FT_Err_Ok;
    PSH_Globals_Funcs  funcs = cid_size_get_globals_funcs( size );


    if ( funcs )
    {
      PSH_Globals   globals;
      CID_Face      face = (CID_Face)cidsize->face;
      CID_FaceDict  dict = face->cid.font_dicts + face->root.face_index;
      PS_Private    priv = &dict->private_dict;


      error = funcs->create( cidsize->face->memory, priv, &globals );
      if ( !error )
        cidsize->internal = (FT_Size_Internal)(void*)globals;
    }

    return error;
  }


  FT_LOCAL( FT_Error )
  cid_size_request( FT_Size          size,
                    FT_Size_Request  req )
  {
    PSH_Globals_Funcs  funcs;


    FT_Request_Metrics( size->face, req );

    funcs = cid_size_get_globals_funcs( (CID_Size)size );

    if ( funcs )
      funcs->set_scale( (PSH_Globals)size->internal,
                        size->metrics.x_scale,
                        size->metrics.y_scale,
                        0, 0 );

    return FT_Err_Ok;
  }


  /*************************************************************************/
  /*                                                                       */
  /*                           FACE  FUNCTIONS                             */
  /*                                                                       */
  /*************************************************************************/

  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    cid_face_done                                                      */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Finalizes a given face object.                                     */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face :: A pointer to the face object to destroy.                   */
  /*                                                                       */
  FT_LOCAL_DEF( void )
  cid_face_done( FT_Face  cidface )         /* CID_Face */
  {
    CID_Face      face = (CID_Face)cidface;
    FT_Memory     memory;
    CID_FaceInfo  cid;
    PS_FontInfo   info;


    if ( !face )
      return;

    cid    = &face->cid;
    info   = &cid->font_info;
    memory = cidface->memory;

    /* release subrs */
    if ( face->subrs )
    {
      FT_Int  n;


      for ( n = 0; n < cid->num_dicts; n++ )
      {
        CID_Subrs  subr = face->subrs + n;


        if ( subr->code )
        {
          FT_FREE( subr->code[0] );
          FT_FREE( subr->code );
        }
      }

      FT_FREE( face->subrs );
    }

    /* release FontInfo strings */
    FT_FREE( info->version );
    FT_FREE( info->notice );
    FT_FREE( info->full_name );
    FT_FREE( info->family_name );
    FT_FREE( info->weight );

    /* release font dictionaries */
    FT_FREE( cid->font_dicts );
    cid->num_dicts = 0;

    /* release other strings */
    FT_FREE( cid->cid_font_name );
    FT_FREE( cid->registry );
    FT_FREE( cid->ordering );

    cidface->family_name = 0;
    cidface->style_name  = 0;

    FT_FREE( face->binary_data );
    FT_FREE( face->cid_stream );
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    cid_face_init                                                      */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Initializes a given CID face object.                               */
  /*                                                                       */
  /* <Input>                                                               */
  /*    stream     :: The source font stream.                              */
  /*                                                                       */
  /*    face_index :: The index of the font face in the resource.          */
  /*                                                                       */
  /*    num_params :: Number of additional generic parameters.  Ignored.   */
  /*                                                                       */
  /*    params     :: Additional generic parameters.  Ignored.             */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face       :: The newly built face object.                         */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0 means success.                             */
  /*                                                                       */
  FT_LOCAL_DEF( FT_Error )
  cid_face_init( FT_Stream      stream,
                 FT_Face        cidface,        /* CID_Face */
                 FT_Int         face_index,
                 FT_Int         num_params,
                 FT_Parameter*  params )
  {
    CID_Face          face = (CID_Face)cidface;
    FT_Error          error;
    PSAux_Service     psaux;
    PSHinter_Service  pshinter;

    FT_UNUSED( num_params );
    FT_UNUSED( params );
    FT_UNUSED( stream );


    cidface->num_faces = 1;

    psaux = (PSAux_Service)face->psaux;
    if ( !psaux )
    {
      psaux = (PSAux_Service)FT_Get_Module_Interface(
                FT_FACE_LIBRARY( face ), "psaux" );

      if ( !psaux )
      {
        FT_ERROR(( "cid_face_init: cannot access `psaux' module\n" ));
        error = FT_THROW( Missing_Module );
        goto Exit;
      }

      face->psaux = psaux;
    }

    pshinter = (PSHinter_Service)face->pshinter;
    if ( !pshinter )
    {
      pshinter = (PSHinter_Service)FT_Get_Module_Interface(
                   FT_FACE_LIBRARY( face ), "pshinter" );

      face->pshinter = pshinter;
    }

    FT_TRACE2(( "CID driver\n" ));

    /* open the tokenizer; this will also check the font format */
    if ( FT_STREAM_SEEK( 0 ) )
      goto Exit;

    error = cid_face_open( face, face_index );
    if ( error )
      goto Exit;

    /* if we just wanted to check the format, leave successfully now */
    if ( face_index < 0 )
      goto Exit;

    /* check the face index */
    /* XXX: handle CID fonts with more than a single face */
    if ( face_index != 0 )
    {
      FT_ERROR(( "cid_face_init: invalid face index\n" ));
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* now load the font program into the face object */

    /* initialize the face object fields */

    /* set up root face fields */
    {
      CID_FaceInfo  cid  = &face->cid;
      PS_FontInfo   info = &cid->font_info;


      cidface->num_glyphs   = cid->cid_count;
      cidface->num_charmaps = 0;

      cidface->face_index = face_index;
      cidface->face_flags = FT_FACE_FLAG_SCALABLE   | /* scalable outlines */
                            FT_FACE_FLAG_HORIZONTAL | /* horizontal data   */
                            FT_FACE_FLAG_HINTER;      /* has native hinter */

      if ( info->is_fixed_pitch )
        cidface->face_flags |= FT_FACE_FLAG_FIXED_WIDTH;

      /* XXX: TODO: add kerning with .afm support */

      /* get style name -- be careful, some broken fonts only */
      /* have a /FontName dictionary entry!                   */
      cidface->family_name = info->family_name;
      /* assume "Regular" style if we don't know better */
      cidface->style_name = (char *)"Regular";
      if ( cidface->family_name )
      {
        char*  full   = info->full_name;
        char*  family = cidface->family_name;


        if ( full )
        {
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
                if ( !*family )
                  cidface->style_name = full;
                break;
              }
            }
          }
        }
      }
      else
      {
        /* do we have a `/FontName'? */
        if ( cid->cid_font_name )
          cidface->family_name = cid->cid_font_name;
      }

      /* compute style flags */
      cidface->style_flags = 0;
      if ( info->italic_angle )
        cidface->style_flags |= FT_STYLE_FLAG_ITALIC;
      if ( info->weight )
      {
        if ( !ft_strcmp( info->weight, "Bold"  ) ||
             !ft_strcmp( info->weight, "Black" ) )
          cidface->style_flags |= FT_STYLE_FLAG_BOLD;
      }

      /* no embedded bitmap support */
      cidface->num_fixed_sizes = 0;
      cidface->available_sizes = 0;

      cidface->bbox.xMin =   cid->font_bbox.xMin            >> 16;
      cidface->bbox.yMin =   cid->font_bbox.yMin            >> 16;
      /* no `U' suffix here to 0xFFFF! */
      cidface->bbox.xMax = ( cid->font_bbox.xMax + 0xFFFF ) >> 16;
      cidface->bbox.yMax = ( cid->font_bbox.yMax + 0xFFFF ) >> 16;

      if ( !cidface->units_per_EM )
        cidface->units_per_EM = 1000;

      cidface->ascender  = (FT_Short)( cidface->bbox.yMax );
      cidface->descender = (FT_Short)( cidface->bbox.yMin );

      cidface->height = (FT_Short)( ( cidface->units_per_EM * 12 ) / 10 );
      if ( cidface->height < cidface->ascender - cidface->descender )
        cidface->height = (FT_Short)( cidface->ascender - cidface->descender );

      cidface->underline_position  = (FT_Short)info->underline_position;
      cidface->underline_thickness = (FT_Short)info->underline_thickness;
    }

  Exit:
    return error;
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    cid_driver_init                                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Initializes a given CID driver object.                             */
  /*                                                                       */
  /* <Input>                                                               */
  /*    driver :: A handle to the target driver object.                    */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0 means success.                             */
  /*                                                                       */
  FT_LOCAL_DEF( FT_Error )
  cid_driver_init( FT_Module  driver )
  {
    FT_UNUSED( driver );

    return FT_Err_Ok;
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    cid_driver_done                                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Finalizes a given CID driver.                                      */
  /*                                                                       */
  /* <Input>                                                               */
  /*    driver :: A handle to the target CID driver.                       */
  /*                                                                       */
  FT_LOCAL_DEF( void )
  cid_driver_done( FT_Module  driver )
  {
    FT_UNUSED( driver );
  }


/* END */

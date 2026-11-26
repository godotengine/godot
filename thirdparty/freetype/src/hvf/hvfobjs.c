/****************************************************************************
 *
 * hvfobjs.c
 *
 *   HVF objects manager (body).
 *
 * Copyright (C) 2025-2026 by
 * Apple Inc.
 * written by Deborah Goldsmith <goldsmit@apple.com>
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/sfnt.h>
#include <freetype/tttags.h>
#include <freetype/ftmm.h>

#include "hvfobjs.h"
#include "hvferror.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  hvfobjs


#ifdef FT_CONFIG_OPTION_HVF

  /**************************************************************************
   *
   *                             FACE  MANAGEMENT
   *
   */

  static FT_Error
  hvf_load_tables( HVF_Face  face )
  {
    FT_Error   error;
    FT_Stream  stream;
    TT_Face    tt_face = (TT_Face)face; /* Cast to access TT_Face members. */


    /* Use the stream from TT_Face root. */
    stream = FT_FACE_STREAM( face );

    /* Load required 'hvgl' table using memory mapping. */
    error = tt_face->goto_table( tt_face, TTAG_hvgl,
                                 stream, &face->hvgl_size );
    if ( error )
    {
      FT_TRACE2(( "hvf_load_tables: missing required 'hvgl' table\n" ));
      return error;
    }

    /* Memory-map the 'hvgl' table data instead of copying it. */
    if ( FT_FRAME_EXTRACT( face->hvgl_size, face->hvgl_data ) )
    {
      face->hvgl_size = 0;
      FT_TRACE2(( "hvf_load_tables: failed to extract 'hvgl' table\n" ));
      return FT_THROW( Invalid_Table );
    }

    FT_TRACE2(( "hvf_load_tables: memory-mapped 'hvgl' table (%lu bytes)\n",
                face->hvgl_size ));

    /* Load optional 'hvpm' table using memory mapping. */
    error = tt_face->goto_table( tt_face, TTAG_hvpm,
                                 stream, &face->hvpm_size );
    if ( error )
    {
      face->hvpm_data = NULL;  /* Ensure NULL for pointer check. */
      face->hvpm_size = 0;
      FT_TRACE2(( "hvf_load_tables: no 'hvpm' table found (optional)\n" ));
    }
    else
    {
      /* Memory-map the 'hvpm' table data instead of copying it. */
      if ( FT_FRAME_EXTRACT( face->hvpm_size, face->hvpm_data ) )
      {
        face->hvpm_data = NULL;
        face->hvpm_size = 0;
        FT_TRACE2(( "hvf_load_tables: failed to extract 'hvpm' table\n" ));
      }
#ifdef FT_DEBUG_LEVEL_TRACE
      else
        FT_TRACE2(( "hvf_load_tables:"
                    " memory-mapped 'hvpm' table (%lu bytes)\n",
                    face->hvpm_size ));
#endif
    }

    return FT_Err_Ok;
  }


  static FT_Error
  hvf_init_renderer( HVF_Face  face )
  {
    FT_Error   error  = FT_Err_Ok;
    FT_Memory  memory = FT_FACE_MEMORY( face );

    size_t  storage_size;


    if ( !face->hvgl_data )
      return FT_THROW( Invalid_Table );

    /* Create HVF renderer - only once per face. */
    if ( !face->renderer )
    {
      if ( HVF_IS_AVAILABLE )
      {
        int  hvf_result;


        /* Get required storage size. */
        storage_size = HVF_part_renderer_storage_size();
        if ( storage_size == 0 )
        {
          FT_TRACE1(( "hvf_init_renderer:"
                      " HVF_part_renderer_storage_size failed\n" ));
          return FT_THROW( Invalid_Table );
        }

        /* Allocate storage for renderer. */
        if ( FT_ALLOC( face->renderer, storage_size ) )
          return error;

        /* Initialize HVF renderer with 'hvgl' and optional 'hvpm' */
        /* table data.                                             */
        hvf_result =
          HVF_open_part_renderer( face->hvgl_data,
                                  face->hvgl_size,
                                  face->hvpm_data, /* NULL if missing */
                                  face->hvpm_size, /* 0 if missing    */
                                  face->renderer,
                                  storage_size );
        if ( hvf_result != 0 )
        {
          FT_TRACE1(( "hvf_init_renderer:"
                      " HVF_open_part_renderer failed (error %d)\n",
                      hvf_result ));
          FT_FREE( face->renderer );
          return FT_THROW( Invalid_Table );
        }

        face->cache_count = 0;
        FT_TRACE2(( "hvf_init_renderer:"
                    " HVF renderer initialized (%s hvpm)\n",
                    face->hvpm_data ? "with" : "without" ));
      }
      else
      {
        FT_TRACE3(( "hvf_init_renderer: HVF not available at runtime\n" ));
        return FT_THROW( Invalid_Table );
      }
    }

    return error;
  }


  static FT_Error
  hvf_init_axis_coordinates( HVF_Face  face )
  {
    FT_Error    error  = FT_Err_Ok;
    FT_Memory   memory = FT_FACE_MEMORY( face );
    FT_MM_Var*  mm_var = NULL;


    /* Initialize axis data. */
    face->num_axes    = 0;
    face->axis_coords = NULL;

    /* Check whether this is a variable font and get axis information. */
    error = FT_Get_MM_Var( (FT_Face)face, &mm_var );
    if ( error )
    {
      /* Not a variable font or error getting variation info - */
      /* that's fine.                                          */
      FT_TRACE3(( "hvf_init_axis_coordinates:"
                  " not a variable font or no MM info (error %x)\n",
                  error ));
      return FT_Err_Ok;
    }

    if ( mm_var && mm_var->num_axis > 0 )
    {
      face->num_axes = mm_var->num_axis;
      
      /* Allocate storage for HVF axis coordinates. */
      if ( FT_QNEW_ARRAY( *(HVFAxisValue**)&face->axis_coords,
                           face->num_axes ) )
      {
        face->num_axes = 0;
        goto Cleanup;
      }

      /* Initialize coordinates using the shared refresh function. */
      error = hvf_refresh_axis_coordinates( face );
      if ( error )
      {
        FT_FREE( face->axis_coords );
        face->num_axes = 0;
        goto Cleanup;
      }
    }

  Cleanup:
    if ( mm_var )
      FT_Done_MM_Var( FT_FACE_LIBRARY( face ), mm_var );

    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  hvf_refresh_axis_coordinates( HVF_Face  face )
  {
    FT_Error   error  = FT_Err_Ok;
    FT_Memory  memory = FT_FACE_MEMORY( face );

    FT_Fixed*  ft_coords = NULL;
    FT_UInt    i;


    /* Only refresh if we have cached coordinates. */
    if ( !face->axis_coords || face->num_axes == 0 )
      return FT_Err_Ok;

    /* Allocate temporary storage for current FreeType coordinates. */
    if ( FT_QNEW_ARRAY( ft_coords, face->num_axes ) )
      return error;

    /* Get current variation coordinates. */
    error = FT_Get_Var_Blend_Coordinates( (FT_Face)face,
                                          face->num_axes, ft_coords );
    if ( error )
    {
      /* If we can't get coordinates, set to defaults. */
      FT_ARRAY_ZERO( (HVFAxisValue*)face->axis_coords, face->num_axes );
      FT_TRACE3(( "hvf_refresh_axis_coordinates:"
                  " could not get blend coordinates, using defaults\n" ));
      error = FT_Err_Ok;  /* Not fatal. */
    }
    else
    {
      /* Convert FreeType coordinates to HVF coordinates. */
      for ( i = 0; i < face->num_axes; i++ )
      {
        ((HVFAxisValue*)face->axis_coords)[i] =
          FT_COORD_TO_HVF_AXIS( ft_coords[i] );
        FT_TRACE5(( "hvf_refresh_axis_coordinates:"
                    " axis %u: FT coord %ld -> HVF coord %f\n",
                    i, ft_coords[i],
                    ((HVFAxisValue*)face->axis_coords)[i] ));
      }
      FT_TRACE3(( "hvf_refresh_axis_coordinates:"
                  " refreshed %u axis coordinates\n", face->num_axes ));
      
      /* Clear HVF renderer cache since axis coordinates changed. */
      /* Cached parts were rendered with old axis values and are  */
      /* now invalid.                                             */
      if ( face->renderer )
      {
        if ( HVF_IS_AVAILABLE )
        {
          HVF_clear_part_cache( (HVFPartRenderer*)face->renderer );
          face->cache_count = 0;
          FT_TRACE4(( "hvf_refresh_axis_coordinates:"
                      " cleared HVF cache due to axis change\n" ));
        }
#ifdef FT_DEBUG_LEVEL_TRACE
        else
          FT_TRACE3(( "hvf_refresh_axis_coordinates:"
                      " HVF not available at runtime\n" ));
#endif
      }
    }

    /* Free temporary FreeType coordinate storage. */
    FT_FREE( ft_coords );

    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   hvf_face_init
   *
   * @Description:
   *   Initialize a given HVF face object.
   *
   * @Input:
   *   stream ::
   *     The source font stream.
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
   *     The newly built face object.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  hvf_face_init( FT_Stream      stream,
                 FT_Face        face,
                 FT_Int         typeface_index,
                 FT_Int         num_params,
                 FT_Parameter*  parameters )
  {
    FT_Error  error;

    FT_Library    library = FT_FACE_LIBRARY( face );
    SFNT_Service  sfnt;

    TT_Face  tt_face = (TT_Face)face;  /* Cast to access TT_Face members. */


    FT_TRACE2(( "HVF driver\n" ));

    /* Check HVF availability at runtime (Apple platforms only). */
    if ( HVF_IS_AVAILABLE )
      ; /* `HVF_IS_AVAILABLE` must not be negated for strange reasons. */
    else
    {
      FT_TRACE2(( "hvf_face_init: HVF not available at runtime\n" ));
      return FT_THROW( Unknown_File_Format );
    }

    /* Check for SFNT wrapper. */
    sfnt = (SFNT_Service)FT_Get_Module_Interface( library, "sfnt" );
    if ( !sfnt )
    {
      FT_ERROR(( "hvf_face_init: cannot access 'sfnt' module\n" ));
      error = FT_THROW( Missing_Module );
      goto Exit;
    }

    /* Create input stream from resource. */
    if ( FT_STREAM_SEEK( 0 ) )
      goto Exit;

    FT_TRACE2(( "  " ));

    /* Check that we have a valid HVF font. */
    error = sfnt->init_face( stream, tt_face, typeface_index,
                             num_params, parameters );
    if ( error )
    {
      FT_TRACE1(( "hvf_face_init: sfnt initialization failed (%d)\n",
                  error ));
      goto Exit;
    }

    /* Verify this is actually an HVF font by checking for 'hvgl' table. */
    error = tt_face->goto_table( tt_face, TTAG_hvgl, stream, NULL );
    if ( error )
    {
      FT_TRACE1(( "hvf_face_init:"
                  " not an HVF font (missing 'hvgl' table)\n" ));
      error = FT_THROW( Unknown_File_Format );
      goto Exit;
    }

    /* Mark face as scalable. */
    face->face_flags |= FT_FACE_FLAG_SCALABLE;

    /* If we are performing a simple font format check, exit immediately. */
    if ( typeface_index < 0 )
      return FT_Err_Ok;

    /* Load font directory. */
    error = sfnt->load_face( stream, tt_face, typeface_index,
                             num_params, parameters );
    if ( error )
      goto Exit;

    /* Load HVF-specific tables. */
    error = hvf_load_tables( (HVF_Face)face );
    if ( error )
      goto Exit;

    /* Initialize HVF renderer. */
    error = hvf_init_renderer( (HVF_Face)face );
    if ( error )
      goto Exit;

    /* Get variation axis information once during face initialization. */
    error = hvf_init_axis_coordinates( (HVF_Face)face );
    if ( error )
      goto Exit;

    FT_TRACE2(( "hvf_face_init: HVF face initialized successfully\n" ));

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   hvf_face_done
   *
   * @Description:
   *   Finalize a given face object.
   *
   * @Input:
   *   face ::
   *     A pointer to the face object to destroy.
   */
  FT_LOCAL_DEF( void )
  hvf_face_done( FT_Face  face )
  {
    FT_Memory  memory;
    FT_Stream  stream;

    TT_Face   tt_face;
    HVF_Face  hvf_face;

    SFNT_Service  sfnt;


    if ( !face )
      return;

    memory = FT_FACE_MEMORY( face );
    stream = FT_FACE_STREAM( face );

    hvf_face = (HVF_Face)face;
    tt_face  = (TT_Face)face;

    /* Use inherited SFNT service pointer */
    sfnt = (SFNT_Service)tt_face->sfnt;

    /* Clean up HVF renderer. */
    if ( hvf_face->renderer )
    {
      if ( HVF_IS_AVAILABLE )
        HVF_close_part_renderer( (HVFPartRenderer*)hvf_face->renderer );
#ifdef FT_DEBUG_LEVEL_TRACE
      else
        FT_TRACE3(( "hvf_face_done: HVF not available at runtime\n" ));
#endif

      FT_FREE( hvf_face->renderer );
    }

    /* Release axis coordinates storage. */
    if ( hvf_face->axis_coords )
      FT_FREE( hvf_face->axis_coords );

    /* Release memory-mapped table data. */
    if ( hvf_face->hvgl_data )
      FT_FRAME_RELEASE( hvf_face->hvgl_data );
    if ( hvf_face->hvpm_data )
      FT_FRAME_RELEASE( hvf_face->hvpm_data );

    /* Clean up SFNT (this will clean up the TT_Face base class). */
    if ( sfnt )
      sfnt->done_face( tt_face );
  }

#else /* !FT_CONFIG_OPTION_HVF */

  FT_LOCAL_DEF( FT_Error )
  hvf_face_init( FT_Stream      stream,
                 FT_Face        face,
                 FT_Int         typeface_index,
                 FT_Int         num_params,
                 FT_Parameter*  parameters )
  {
    FT_UNUSED( stream );
    FT_UNUSED( face );
    FT_UNUSED( typeface_index );
    FT_UNUSED( num_params );
    FT_UNUSED( parameters );

    return FT_THROW( Unknown_File_Format );
  }

#endif /* !FT_CONFIG_OPTION_HVF */


/* END */

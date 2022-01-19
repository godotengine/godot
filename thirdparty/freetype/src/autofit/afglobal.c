/****************************************************************************
 *
 * afglobal.c
 *
 *   Auto-fitter routines to compute global hinting values (body).
 *
 * Copyright (C) 2003-2021 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include "afglobal.h"
#include "afranges.h"
#include "afshaper.h"
#include "afws-decl.h"
#include <freetype/internal/ftdebug.h>


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  afglobal


#include "aferrors.h"


#undef  SCRIPT
#define SCRIPT( s, S, d, h, H, ss )         \
          AF_DEFINE_SCRIPT_CLASS(           \
            af_ ## s ## _script_class,      \
            AF_SCRIPT_ ## S,                \
            af_ ## s ## _uniranges,         \
            af_ ## s ## _nonbase_uniranges, \
            AF_ ## H,                       \
            ss )

#include "afscript.h"


#undef  STYLE
#define STYLE( s, S, d, ws, sc, ss, c )  \
          AF_DEFINE_STYLE_CLASS(         \
            af_ ## s ## _style_class,    \
            AF_STYLE_ ## S,              \
            ws,                          \
            sc,                          \
            ss,                          \
            c )

#include "afstyles.h"


#undef  WRITING_SYSTEM
#define WRITING_SYSTEM( ws, WS )               \
          &af_ ## ws ## _writing_system_class,

  FT_LOCAL_ARRAY_DEF( AF_WritingSystemClass )
  af_writing_system_classes[] =
  {

#include "afws-iter.h"

    NULL  /* do not remove */
  };


#undef  SCRIPT
#define SCRIPT( s, S, d, h, H, ss )   \
          &af_ ## s ## _script_class,

  FT_LOCAL_ARRAY_DEF( AF_ScriptClass )
  af_script_classes[] =
  {

#include "afscript.h"

    NULL  /* do not remove */
  };


#undef  STYLE
#define STYLE( s, S, d, ws, sc, ss, c ) \
          &af_ ## s ## _style_class,

  FT_LOCAL_ARRAY_DEF( AF_StyleClass )
  af_style_classes[] =
  {

#include "afstyles.h"

    NULL  /* do not remove */
  };


#ifdef FT_DEBUG_LEVEL_TRACE

#undef  STYLE
#define STYLE( s, S, d, ws, sc, ss, c )  #s,

  FT_LOCAL_ARRAY_DEF( char* )
  af_style_names[] =
  {

#include "afstyles.h"

  };

#endif /* FT_DEBUG_LEVEL_TRACE */


  /* Compute the style index of each glyph within a given face. */

  static FT_Error
  af_face_globals_compute_style_coverage( AF_FaceGlobals  globals )
  {
    FT_Error    error;
    FT_Face     face        = globals->face;
    FT_CharMap  old_charmap = face->charmap;
    FT_UShort*  gstyles     = globals->glyph_styles;
    FT_UInt     ss;
    FT_UInt     i;
    FT_UInt     dflt        = ~0U; /* a non-valid value */


    /* the value AF_STYLE_UNASSIGNED means `uncovered glyph' */
    for ( i = 0; i < (FT_UInt)globals->glyph_count; i++ )
      gstyles[i] = AF_STYLE_UNASSIGNED;

    error = FT_Select_Charmap( face, FT_ENCODING_UNICODE );
    if ( error )
    {
      /*
       * Ignore this error; we simply use the fallback style.
       * XXX: Shouldn't we rather disable hinting?
       */
      error = FT_Err_Ok;
      goto Exit;
    }

    /* scan each style in a Unicode charmap */
    for ( ss = 0; af_style_classes[ss]; ss++ )
    {
      AF_StyleClass       style_class =
                            af_style_classes[ss];
      AF_ScriptClass      script_class =
                            af_script_classes[style_class->script];
      AF_Script_UniRange  range;


      if ( !script_class->script_uni_ranges )
        continue;

      /*
       * Scan all Unicode points in the range and set the corresponding
       * glyph style index.
       */
      if ( style_class->coverage == AF_COVERAGE_DEFAULT )
      {
        if ( (FT_UInt)style_class->script ==
             globals->module->default_script )
          dflt = ss;

        for ( range = script_class->script_uni_ranges;
              range->first != 0;
              range++ )
        {
          FT_ULong  charcode = range->first;
          FT_UInt   gindex;


          gindex = FT_Get_Char_Index( face, charcode );

          if ( gindex != 0                                                &&
               gindex < (FT_ULong)globals->glyph_count                    &&
               ( gstyles[gindex] & AF_STYLE_MASK ) == AF_STYLE_UNASSIGNED )
            gstyles[gindex] = (FT_UShort)ss;

          for (;;)
          {
            charcode = FT_Get_Next_Char( face, charcode, &gindex );

            if ( gindex == 0 || charcode > range->last )
              break;

            if ( gindex < (FT_ULong)globals->glyph_count                    &&
                 ( gstyles[gindex] & AF_STYLE_MASK ) == AF_STYLE_UNASSIGNED )
              gstyles[gindex] = (FT_UShort)ss;
          }
        }

        /* do the same for the script's non-base characters */
        for ( range = script_class->script_uni_nonbase_ranges;
              range->first != 0;
              range++ )
        {
          FT_ULong  charcode = range->first;
          FT_UInt   gindex;


          gindex = FT_Get_Char_Index( face, charcode );

          if ( gindex != 0                                          &&
               gindex < (FT_ULong)globals->glyph_count              &&
               ( gstyles[gindex] & AF_STYLE_MASK ) == (FT_UShort)ss )
            gstyles[gindex] |= AF_NONBASE;

          for (;;)
          {
            charcode = FT_Get_Next_Char( face, charcode, &gindex );

            if ( gindex == 0 || charcode > range->last )
              break;

            if ( gindex < (FT_ULong)globals->glyph_count              &&
                 ( gstyles[gindex] & AF_STYLE_MASK ) == (FT_UShort)ss )
              gstyles[gindex] |= AF_NONBASE;
          }
        }
      }
      else
      {
        /* get glyphs not directly addressable by cmap */
        af_shaper_get_coverage( globals, style_class, gstyles, 0 );
      }
    }

    /* handle the remaining default OpenType features ... */
    for ( ss = 0; af_style_classes[ss]; ss++ )
    {
      AF_StyleClass  style_class = af_style_classes[ss];


      if ( style_class->coverage == AF_COVERAGE_DEFAULT )
        af_shaper_get_coverage( globals, style_class, gstyles, 0 );
    }

    /* ... and finally the default OpenType features of the default script */
    af_shaper_get_coverage( globals, af_style_classes[dflt], gstyles, 1 );

    /* mark ASCII digits */
    for ( i = 0x30; i <= 0x39; i++ )
    {
      FT_UInt  gindex = FT_Get_Char_Index( face, i );


      if ( gindex != 0 && gindex < (FT_ULong)globals->glyph_count )
        gstyles[gindex] |= AF_DIGIT;
    }

  Exit:
    /*
     * By default, all uncovered glyphs are set to the fallback style.
     * XXX: Shouldn't we disable hinting or do something similar?
     */
    if ( globals->module->fallback_style != AF_STYLE_UNASSIGNED )
    {
      FT_Long  nn;


      for ( nn = 0; nn < globals->glyph_count; nn++ )
      {
        if ( ( gstyles[nn] & AF_STYLE_MASK ) == AF_STYLE_UNASSIGNED )
        {
          gstyles[nn] &= ~AF_STYLE_MASK;
          gstyles[nn] |= globals->module->fallback_style;
        }
      }
    }

#ifdef FT_DEBUG_LEVEL_TRACE

    FT_TRACE4(( "\n" ));
    FT_TRACE4(( "style coverage\n" ));
    FT_TRACE4(( "==============\n" ));
    FT_TRACE4(( "\n" ));

    for ( ss = 0; af_style_classes[ss]; ss++ )
    {
      AF_StyleClass  style_class = af_style_classes[ss];
      FT_UInt        count       = 0;
      FT_Long        idx;


      FT_TRACE4(( "%s:\n", af_style_names[style_class->style] ));

      for ( idx = 0; idx < globals->glyph_count; idx++ )
      {
        if ( ( gstyles[idx] & AF_STYLE_MASK ) == style_class->style )
        {
          if ( !( count % 10 ) )
            FT_TRACE4(( " " ));

          FT_TRACE4(( " %ld", idx ));
          count++;

          if ( !( count % 10 ) )
            FT_TRACE4(( "\n" ));
        }
      }

      if ( !count )
        FT_TRACE4(( "  (none)\n" ));
      if ( count % 10 )
        FT_TRACE4(( "\n" ));
    }

#endif /* FT_DEBUG_LEVEL_TRACE */

    FT_Set_Charmap( face, old_charmap );
    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  af_face_globals_new( FT_Face          face,
                       AF_FaceGlobals  *aglobals,
                       AF_Module        module )
  {
    FT_Error        error;
    FT_Memory       memory;
    AF_FaceGlobals  globals = NULL;


    memory = face->memory;

    /* we allocate an AF_FaceGlobals structure together */
    /* with the glyph_styles array                      */
    if ( FT_ALLOC( globals,
                   sizeof ( *globals ) +
                     (FT_ULong)face->num_glyphs * sizeof ( FT_UShort ) ) )
      goto Exit;

    globals->face                      = face;
    globals->glyph_count               = face->num_glyphs;
    /* right after the globals structure come the glyph styles */
    globals->glyph_styles              = (FT_UShort*)( globals + 1 );
    globals->module                    = module;
    globals->stem_darkening_for_ppem   = 0;
    globals->darken_x                  = 0;
    globals->darken_y                  = 0;
    globals->standard_vertical_width   = 0;
    globals->standard_horizontal_width = 0;
    globals->scale_down_factor         = 0;

#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ
    globals->hb_font = hb_ft_font_create( face, NULL );
    globals->hb_buf  = hb_buffer_create();
#endif

    error = af_face_globals_compute_style_coverage( globals );
    if ( error )
    {
      af_face_globals_free( globals );
      globals = NULL;
    }
    else
      globals->increase_x_height = AF_PROP_INCREASE_X_HEIGHT_MAX;

  Exit:
    *aglobals = globals;
    return error;
  }


  FT_LOCAL_DEF( void )
  af_face_globals_free( AF_FaceGlobals  globals )
  {
    if ( globals )
    {
      FT_Memory  memory = globals->face->memory;
      FT_UInt    nn;


      for ( nn = 0; nn < AF_STYLE_MAX; nn++ )
      {
        if ( globals->metrics[nn] )
        {
          AF_StyleClass          style_class =
            af_style_classes[nn];
          AF_WritingSystemClass  writing_system_class =
            af_writing_system_classes[style_class->writing_system];


          if ( writing_system_class->style_metrics_done )
            writing_system_class->style_metrics_done( globals->metrics[nn] );

          FT_FREE( globals->metrics[nn] );
        }
      }

#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ
      hb_font_destroy( globals->hb_font );
      hb_buffer_destroy( globals->hb_buf );
#endif

      /* no need to free `globals->glyph_styles'; */
      /* it is part of the `globals' array        */
      FT_FREE( globals );
    }
  }


  FT_LOCAL_DEF( FT_Error )
  af_face_globals_get_metrics( AF_FaceGlobals    globals,
                               FT_UInt           gindex,
                               FT_UInt           options,
                               AF_StyleMetrics  *ametrics )
  {
    AF_StyleMetrics  metrics = NULL;

    AF_Style               style = (AF_Style)options;
    AF_WritingSystemClass  writing_system_class;
    AF_StyleClass          style_class;

    FT_Error  error = FT_Err_Ok;


    if ( gindex >= (FT_ULong)globals->glyph_count )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* if we have a forced style (via `options'), use it, */
    /* otherwise look into `glyph_styles' array           */
    if ( style == AF_STYLE_NONE_DFLT || style + 1 >= AF_STYLE_MAX )
      style = (AF_Style)( globals->glyph_styles[gindex] &
                          AF_STYLE_UNASSIGNED           );

  Again:
    style_class          = af_style_classes[style];
    writing_system_class = af_writing_system_classes
                             [style_class->writing_system];

    metrics = globals->metrics[style];
    if ( !metrics )
    {
      /* create the global metrics object if necessary */
      FT_Memory  memory = globals->face->memory;


      if ( FT_ALLOC( metrics, writing_system_class->style_metrics_size ) )
        goto Exit;

      metrics->style_class = style_class;
      metrics->globals     = globals;

      if ( writing_system_class->style_metrics_init )
      {
        error = writing_system_class->style_metrics_init( metrics,
                                                          globals->face );
        if ( error )
        {
          if ( writing_system_class->style_metrics_done )
            writing_system_class->style_metrics_done( metrics );

          FT_FREE( metrics );

          /* internal error code -1 indicates   */
          /* that no blue zones have been found */
          if ( error == -1 )
          {
            style = (AF_Style)( globals->glyph_styles[gindex] &
                                AF_STYLE_UNASSIGNED           );
            /* IMPORTANT: Clear the error code, see
             * https://gitlab.freedesktop.org/freetype/freetype/-/issues/1063
             */
            error = FT_Err_Ok;
            goto Again;
          }

          goto Exit;
        }
      }

      globals->metrics[style] = metrics;
    }

  Exit:
    *ametrics = metrics;

    return error;
  }


  FT_LOCAL_DEF( FT_Bool )
  af_face_globals_is_digit( AF_FaceGlobals  globals,
                            FT_UInt         gindex )
  {
    if ( gindex < (FT_ULong)globals->glyph_count )
      return FT_BOOL( globals->glyph_styles[gindex] & AF_DIGIT );

    return FT_BOOL( 0 );
  }


/* END */

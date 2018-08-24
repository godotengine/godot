/***************************************************************************/
/*                                                                         */
/*  afshaper.c                                                             */
/*                                                                         */
/*    HarfBuzz interface for accessing OpenType features (body).           */
/*                                                                         */
/*  Copyright 2013-2018 by                                                 */
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
#include FT_FREETYPE_H
#include FT_ADVANCES_H
#include "afglobal.h"
#include "aftypes.h"
#include "afshaper.h"

#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_afshaper


  /*
   * We use `sets' (in the HarfBuzz sense, which comes quite near to the
   * usual mathematical meaning) to manage both lookups and glyph indices.
   *
   * 1. For each coverage, collect lookup IDs in a set.  Note that an
   *    auto-hinter `coverage' is represented by one `feature', and a
   *    feature consists of an arbitrary number of (font specific) `lookup's
   *    that actually do the mapping job.  Please check the OpenType
   *    specification for more details on features and lookups.
   *
   * 2. Create glyph ID sets from the corresponding lookup sets.
   *
   * 3. The glyph set corresponding to AF_COVERAGE_DEFAULT is computed
   *    with all lookups specific to the OpenType script activated.  It
   *    relies on the order of AF_DEFINE_STYLE_CLASS entries so that
   *    special coverages (like `oldstyle figures') don't get overwritten.
   *
   */


  /* load coverage tags */
#undef  COVERAGE
#define COVERAGE( name, NAME, description,             \
                  tag1, tag2, tag3, tag4 )             \
          static const hb_tag_t  name ## _coverage[] = \
          {                                            \
            HB_TAG( tag1, tag2, tag3, tag4 ),          \
            HB_TAG_NONE                                \
          };


#include "afcover.h"


  /* define mapping between coverage tags and AF_Coverage */
#undef  COVERAGE
#define COVERAGE( name, NAME, description, \
                  tag1, tag2, tag3, tag4 ) \
          name ## _coverage,


  static const hb_tag_t*  coverages[] =
  {
#include "afcover.h"

    NULL /* AF_COVERAGE_DEFAULT */
  };


  /* load HarfBuzz script tags */
#undef  SCRIPT
#define SCRIPT( s, S, d, h, H, ss )  h,


  static const hb_script_t  scripts[] =
  {
#include "afscript.h"
  };


  FT_Error
  af_shaper_get_coverage( AF_FaceGlobals  globals,
                          AF_StyleClass   style_class,
                          FT_UShort*      gstyles,
                          FT_Bool         default_script )
  {
    hb_face_t*  face;

    hb_set_t*  gsub_lookups = NULL; /* GSUB lookups for a given script */
    hb_set_t*  gsub_glyphs  = NULL; /* glyphs covered by GSUB lookups  */
    hb_set_t*  gpos_lookups = NULL; /* GPOS lookups for a given script */
    hb_set_t*  gpos_glyphs  = NULL; /* glyphs covered by GPOS lookups  */

    hb_script_t      script;
    const hb_tag_t*  coverage_tags;
    hb_tag_t         script_tags[] = { HB_TAG_NONE,
                                       HB_TAG_NONE,
                                       HB_TAG_NONE,
                                       HB_TAG_NONE };

    hb_codepoint_t  idx;
#ifdef FT_DEBUG_LEVEL_TRACE
    int             count;
#endif


    if ( !globals || !style_class || !gstyles )
      return FT_THROW( Invalid_Argument );

    face = hb_font_get_face( globals->hb_font );

    coverage_tags = coverages[style_class->coverage];
    script        = scripts[style_class->script];

    /* Convert a HarfBuzz script tag into the corresponding OpenType */
    /* tag or tags -- some Indic scripts like Devanagari have an old */
    /* and a new set of features.                                    */
    hb_ot_tags_from_script( script,
                            &script_tags[0],
                            &script_tags[1] );

    /* `hb_ot_tags_from_script' usually returns HB_OT_TAG_DEFAULT_SCRIPT */
    /* as the second tag.  We change that to HB_TAG_NONE except for the  */
    /* default script.                                                   */
    if ( default_script )
    {
      if ( script_tags[0] == HB_TAG_NONE )
        script_tags[0] = HB_OT_TAG_DEFAULT_SCRIPT;
      else
      {
        if ( script_tags[1] == HB_TAG_NONE )
          script_tags[1] = HB_OT_TAG_DEFAULT_SCRIPT;
        else if ( script_tags[1] != HB_OT_TAG_DEFAULT_SCRIPT )
          script_tags[2] = HB_OT_TAG_DEFAULT_SCRIPT;
      }
    }
    else
    {
      /* we use non-standard tags like `khms' for special purposes;       */
      /* HarfBuzz maps them to `DFLT', which we don't want to handle here */
      if ( script_tags[0] == HB_OT_TAG_DEFAULT_SCRIPT )
        goto Exit;

      if ( script_tags[1] == HB_OT_TAG_DEFAULT_SCRIPT )
        script_tags[1] = HB_TAG_NONE;
    }

    gsub_lookups = hb_set_create();
    hb_ot_layout_collect_lookups( face,
                                  HB_OT_TAG_GSUB,
                                  script_tags,
                                  NULL,
                                  coverage_tags,
                                  gsub_lookups );

    if ( hb_set_is_empty( gsub_lookups ) )
      goto Exit; /* nothing to do */

    FT_TRACE4(( "GSUB lookups (style `%s'):\n"
                " ",
                af_style_names[style_class->style] ));

#ifdef FT_DEBUG_LEVEL_TRACE
    count = 0;
#endif

    gsub_glyphs = hb_set_create();
    for ( idx = HB_SET_VALUE_INVALID; hb_set_next( gsub_lookups, &idx ); )
    {
#ifdef FT_DEBUG_LEVEL_TRACE
      FT_TRACE4(( " %d", idx ));
      count++;
#endif

      /* get output coverage of GSUB feature */
      hb_ot_layout_lookup_collect_glyphs( face,
                                          HB_OT_TAG_GSUB,
                                          idx,
                                          NULL,
                                          NULL,
                                          NULL,
                                          gsub_glyphs );
    }

#ifdef FT_DEBUG_LEVEL_TRACE
    if ( !count )
      FT_TRACE4(( " (none)" ));
    FT_TRACE4(( "\n\n" ));
#endif

    FT_TRACE4(( "GPOS lookups (style `%s'):\n"
                " ",
                af_style_names[style_class->style] ));

    gpos_lookups = hb_set_create();
    hb_ot_layout_collect_lookups( face,
                                  HB_OT_TAG_GPOS,
                                  script_tags,
                                  NULL,
                                  coverage_tags,
                                  gpos_lookups );

#ifdef FT_DEBUG_LEVEL_TRACE
    count = 0;
#endif

    gpos_glyphs = hb_set_create();
    for ( idx = HB_SET_VALUE_INVALID; hb_set_next( gpos_lookups, &idx ); )
    {
#ifdef FT_DEBUG_LEVEL_TRACE
      FT_TRACE4(( " %d", idx ));
      count++;
#endif

      /* get input coverage of GPOS feature */
      hb_ot_layout_lookup_collect_glyphs( face,
                                          HB_OT_TAG_GPOS,
                                          idx,
                                          NULL,
                                          gpos_glyphs,
                                          NULL,
                                          NULL );
    }

#ifdef FT_DEBUG_LEVEL_TRACE
    if ( !count )
      FT_TRACE4(( " (none)" ));
    FT_TRACE4(( "\n\n" ));
#endif

    /*
     * We now check whether we can construct blue zones, using glyphs
     * covered by the feature only.  In case there is not a single zone
     * (this is, not a single character is covered), we skip this coverage.
     *
     */
    if ( style_class->coverage != AF_COVERAGE_DEFAULT )
    {
      AF_Blue_Stringset         bss = style_class->blue_stringset;
      const AF_Blue_StringRec*  bs  = &af_blue_stringsets[bss];

      FT_Bool  found = 0;


      for ( ; bs->string != AF_BLUE_STRING_MAX; bs++ )
      {
        const char*  p = &af_blue_strings[bs->string];


        while ( *p )
        {
          hb_codepoint_t  ch;


          GET_UTF8_CHAR( ch, p );

          for ( idx = HB_SET_VALUE_INVALID; hb_set_next( gsub_lookups,
                                                         &idx ); )
          {
            hb_codepoint_t  gidx = FT_Get_Char_Index( globals->face, ch );


            if ( hb_ot_layout_lookup_would_substitute( face, idx,
                                                       &gidx, 1, 1 ) )
            {
              found = 1;
              break;
            }
          }
        }
      }

      if ( !found )
      {
        FT_TRACE4(( "  no blue characters found; style skipped\n" ));
        goto Exit;
      }
    }

    /*
     * Various OpenType features might use the same glyphs at different
     * vertical positions; for example, superscript and subscript glyphs
     * could be the same.  However, the auto-hinter is completely
     * agnostic of OpenType features after the feature analysis has been
     * completed: The engine then simply receives a glyph index and returns a
     * hinted and usually rendered glyph.
     *
     * Consider the superscript feature of font `pala.ttf': Some of the
     * glyphs are `real', this is, they have a zero vertical offset, but
     * most of them are small caps glyphs shifted up to the superscript
     * position (this is, the `sups' feature is present in both the GSUB and
     * GPOS tables).  The code for blue zones computation actually uses a
     * feature's y offset so that the `real' glyphs get correct hints.  But
     * later on it is impossible to decide whether a glyph index belongs to,
     * say, the small caps or superscript feature.
     *
     * For this reason, we don't assign a style to a glyph if the current
     * feature covers the glyph in both the GSUB and the GPOS tables.  This
     * is quite a broad condition, assuming that
     *
     *   (a) glyphs that get used in multiple features are present in a
     *       feature without vertical shift,
     *
     * and
     *
     *   (b) a feature's GPOS data really moves the glyph vertically.
     *
     * Not fulfilling condition (a) makes a font larger; it would also
     * reduce the number of glyphs that could be addressed directly without
     * using OpenType features, so this assumption is rather strong.
     *
     * Condition (b) is much weaker, and there might be glyphs which get
     * missed.  However, the OpenType features we are going to handle are
     * primarily located in GSUB, and HarfBuzz doesn't provide an API to
     * directly get the necessary information from the GPOS table.  A
     * possible solution might be to directly parse the GPOS table to find
     * out whether a glyph gets shifted vertically, but this is something I
     * would like to avoid if not really necessary.
     *
     * Note that we don't follow this logic for the default coverage.
     * Complex scripts like Devanagari have mandatory GPOS features to
     * position many glyph elements, using mark-to-base or mark-to-ligature
     * tables; the number of glyphs missed due to condition (b) would be far
     * too large.
     *
     */
    if ( style_class->coverage != AF_COVERAGE_DEFAULT )
      hb_set_subtract( gsub_glyphs, gpos_glyphs );

#ifdef FT_DEBUG_LEVEL_TRACE
    FT_TRACE4(( "  glyphs without GPOS data (`*' means already assigned)" ));
    count = 0;
#endif

    for ( idx = HB_SET_VALUE_INVALID; hb_set_next( gsub_glyphs, &idx ); )
    {
#ifdef FT_DEBUG_LEVEL_TRACE
      if ( !( count % 10 ) )
        FT_TRACE4(( "\n"
                    "   " ));

      FT_TRACE4(( " %d", idx ));
      count++;
#endif

      /* glyph indices returned by `hb_ot_layout_lookup_collect_glyphs' */
      /* can be arbitrary: some fonts use fake indices for processing   */
      /* internal to GSUB or GPOS, which is fully valid                 */
      if ( idx >= (hb_codepoint_t)globals->glyph_count )
        continue;

      if ( gstyles[idx] == AF_STYLE_UNASSIGNED )
        gstyles[idx] = (FT_UShort)style_class->style;
#ifdef FT_DEBUG_LEVEL_TRACE
      else
        FT_TRACE4(( "*" ));
#endif
    }

#ifdef FT_DEBUG_LEVEL_TRACE
    if ( !count )
      FT_TRACE4(( "\n"
                  "    (none)" ));
    FT_TRACE4(( "\n\n" ));
#endif

  Exit:
    hb_set_destroy( gsub_lookups );
    hb_set_destroy( gsub_glyphs  );
    hb_set_destroy( gpos_lookups );
    hb_set_destroy( gpos_glyphs  );

    return FT_Err_Ok;
  }


  /* construct HarfBuzz features */
#undef  COVERAGE
#define COVERAGE( name, NAME, description,                \
                  tag1, tag2, tag3, tag4 )                \
          static const hb_feature_t  name ## _feature[] = \
          {                                               \
            {                                             \
              HB_TAG( tag1, tag2, tag3, tag4 ),           \
              1, 0, (unsigned int)-1                      \
            }                                             \
          };


#include "afcover.h"


  /* define mapping between HarfBuzz features and AF_Coverage */
#undef  COVERAGE
#define COVERAGE( name, NAME, description, \
                  tag1, tag2, tag3, tag4 ) \
          name ## _feature,


  static const hb_feature_t*  features[] =
  {
#include "afcover.h"

    NULL /* AF_COVERAGE_DEFAULT */
  };


  void*
  af_shaper_buf_create( FT_Face  face )
  {
    FT_UNUSED( face );

    return (void*)hb_buffer_create();
  }


  void
  af_shaper_buf_destroy( FT_Face  face,
                         void*    buf )
  {
    FT_UNUSED( face );

    hb_buffer_destroy( (hb_buffer_t*)buf );
  }


  const char*
  af_shaper_get_cluster( const char*      p,
                         AF_StyleMetrics  metrics,
                         void*            buf_,
                         unsigned int*    count )
  {
    AF_StyleClass        style_class;
    const hb_feature_t*  feature;
    FT_Int               upem;
    const char*          q;
    int                  len;

    hb_buffer_t*    buf = (hb_buffer_t*)buf_;
    hb_font_t*      font;
    hb_codepoint_t  dummy;


    upem        = (FT_Int)metrics->globals->face->units_per_EM;
    style_class = metrics->style_class;
    feature     = features[style_class->coverage];

    font = metrics->globals->hb_font;

    /* we shape at a size of units per EM; this means font units */
    hb_font_set_scale( font, upem, upem );

    while ( *p == ' ' )
      p++;

    /* count bytes up to next space (or end of buffer) */
    q = p;
    while ( !( *q == ' ' || *q == '\0' ) )
      GET_UTF8_CHAR( dummy, q );
    len = (int)( q - p );

    /* feed character(s) to the HarfBuzz buffer */
    hb_buffer_clear_contents( buf );
    hb_buffer_add_utf8( buf, p, len, 0, len );

    /* we let HarfBuzz guess the script and writing direction */
    hb_buffer_guess_segment_properties( buf );

    /* shape buffer, which means conversion from character codes to */
    /* glyph indices, possibly applying a feature                   */
    hb_shape( font, buf, feature, feature ? 1 : 0 );

    if ( feature )
    {
      hb_buffer_t*  hb_buf = metrics->globals->hb_buf;

      unsigned int      gcount;
      hb_glyph_info_t*  ginfo;

      unsigned int      hb_gcount;
      hb_glyph_info_t*  hb_ginfo;


      /* we have to check whether applying a feature does actually change */
      /* glyph indices; otherwise the affected glyph or glyphs aren't     */
      /* available at all in the feature                                  */

      hb_buffer_clear_contents( hb_buf );
      hb_buffer_add_utf8( hb_buf, p, len, 0, len );
      hb_buffer_guess_segment_properties( hb_buf );
      hb_shape( font, hb_buf, NULL, 0 );

      ginfo    = hb_buffer_get_glyph_infos( buf, &gcount );
      hb_ginfo = hb_buffer_get_glyph_infos( hb_buf, &hb_gcount );

      if ( gcount == hb_gcount )
      {
        unsigned int  i;


        for (i = 0; i < gcount; i++ )
          if ( ginfo[i].codepoint != hb_ginfo[i].codepoint )
            break;

        if ( i == gcount )
        {
          /* both buffers have identical glyph indices */
          hb_buffer_clear_contents( buf );
        }
      }
    }

    *count = hb_buffer_get_length( buf );

#ifdef FT_DEBUG_LEVEL_TRACE
    if ( feature && *count > 1 )
      FT_TRACE1(( "af_shaper_get_cluster:"
                  " input character mapped to multiple glyphs\n" ));
#endif

    return q;
  }


  FT_ULong
  af_shaper_get_elem( AF_StyleMetrics  metrics,
                      void*            buf_,
                      unsigned int     idx,
                      FT_Long*         advance,
                      FT_Long*         y_offset )
  {
    hb_buffer_t*          buf = (hb_buffer_t*)buf_;
    hb_glyph_info_t*      ginfo;
    hb_glyph_position_t*  gpos;
    unsigned int          gcount;

    FT_UNUSED( metrics );


    ginfo = hb_buffer_get_glyph_infos( buf, &gcount );
    gpos  = hb_buffer_get_glyph_positions( buf, &gcount );

    if ( idx >= gcount )
      return 0;

    if ( advance )
      *advance = gpos[idx].x_advance;
    if ( y_offset )
      *y_offset = gpos[idx].y_offset;

    return ginfo[idx].codepoint;
  }


#else /* !FT_CONFIG_OPTION_USE_HARFBUZZ */


  FT_Error
  af_shaper_get_coverage( AF_FaceGlobals  globals,
                          AF_StyleClass   style_class,
                          FT_UShort*      gstyles,
                          FT_Bool         default_script )
  {
    FT_UNUSED( globals );
    FT_UNUSED( style_class );
    FT_UNUSED( gstyles );
    FT_UNUSED( default_script );

    return FT_Err_Ok;
  }


  void*
  af_shaper_buf_create( FT_Face  face )
  {
    FT_Error   error;
    FT_Memory  memory = face->memory;
    FT_ULong*  buf;


    FT_MEM_ALLOC( buf, sizeof ( FT_ULong ) );

    return (void*)buf;
  }


  void
  af_shaper_buf_destroy( FT_Face  face,
                         void*    buf )
  {
    FT_Memory  memory = face->memory;


    FT_FREE( buf );
  }


  const char*
  af_shaper_get_cluster( const char*      p,
                         AF_StyleMetrics  metrics,
                         void*            buf_,
                         unsigned int*    count )
  {
    FT_Face    face      = metrics->globals->face;
    FT_ULong   ch, dummy = 0;
    FT_ULong*  buf       = (FT_ULong*)buf_;


    while ( *p == ' ' )
      p++;

    GET_UTF8_CHAR( ch, p );

    /* since we don't have an engine to handle clusters, */
    /* we scan the characters but return zero            */
    while ( !( *p == ' ' || *p == '\0' ) )
      GET_UTF8_CHAR( dummy, p );

    if ( dummy )
    {
      *buf   = 0;
      *count = 0;
    }
    else
    {
      *buf   = FT_Get_Char_Index( face, ch );
      *count = 1;
    }

    return p;
  }


  FT_ULong
  af_shaper_get_elem( AF_StyleMetrics  metrics,
                      void*            buf_,
                      unsigned int     idx,
                      FT_Long*         advance,
                      FT_Long*         y_offset )
  {
    FT_Face   face        = metrics->globals->face;
    FT_ULong  glyph_index = *(FT_ULong*)buf_;

    FT_UNUSED( idx );


    if ( advance )
      FT_Get_Advance( face,
                      glyph_index,
                      FT_LOAD_NO_SCALE         |
                      FT_LOAD_NO_HINTING       |
                      FT_LOAD_IGNORE_TRANSFORM,
                      advance );

    if ( y_offset )
      *y_offset = 0;

    return glyph_index;
  }


#endif /* !FT_CONFIG_OPTION_USE_HARFBUZZ */


/* END */

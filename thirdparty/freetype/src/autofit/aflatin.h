/****************************************************************************
 *
 * aflatin.h
 *
 *   Auto-fitter hinting routines for latin writing system
 *   (specification).
 *
 * Copyright (C) 2003-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef AFLATIN_H_
#define AFLATIN_H_

#include "afhints.h"


FT_BEGIN_HEADER

  /* the `latin' writing system */

  AF_DECLARE_WRITING_SYSTEM_CLASS( af_latin_writing_system_class )


  /* constants are given with units_per_em == 2048 in mind */
#define AF_LATIN_CONSTANT( metrics, c )                                      \
  ( ( (c) * (FT_Long)( (AF_LatinMetrics)(metrics) )->units_per_em ) / 2048 )


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****            L A T I N   G L O B A L   M E T R I C S            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  /*
   * The following declarations could be embedded in the file `aflatin.c';
   * they have been made semi-public to allow alternate writing system
   * hinters to re-use some of them.
   */


#define AF_LATIN_IS_TOP_BLUE( b ) \
          ( (b)->properties & AF_BLUE_PROPERTY_LATIN_TOP )
#define AF_LATIN_IS_SUB_TOP_BLUE( b ) \
          ( (b)->properties & AF_BLUE_PROPERTY_LATIN_SUB_TOP )
#define AF_LATIN_IS_NEUTRAL_BLUE( b ) \
          ( (b)->properties & AF_BLUE_PROPERTY_LATIN_NEUTRAL )
#define AF_LATIN_IS_X_HEIGHT_BLUE( b ) \
          ( (b)->properties & AF_BLUE_PROPERTY_LATIN_X_HEIGHT )
#define AF_LATIN_IS_LONG_BLUE( b ) \
          ( (b)->properties & AF_BLUE_PROPERTY_LATIN_LONG )

#define AF_LATIN_MAX_WIDTHS  16


#define AF_LATIN_BLUE_ACTIVE      ( 1U << 0 ) /* zone height is <= 3/4px   */
#define AF_LATIN_BLUE_TOP         ( 1U << 1 ) /* we have a top blue zone   */
#define AF_LATIN_BLUE_SUB_TOP     ( 1U << 2 ) /* we have a subscript top   */
                                              /* blue zone                 */
#define AF_LATIN_BLUE_NEUTRAL     ( 1U << 3 ) /* we have neutral blue zone */
#define AF_LATIN_BLUE_ADJUSTMENT  ( 1U << 4 ) /* used for scale adjustment */
                                              /* optimization              */


  typedef struct  AF_LatinBlueRec_
  {
    AF_WidthRec  ref;
    AF_WidthRec  shoot;
    FT_Pos       ascender;
    FT_Pos       descender;
    FT_UInt      flags;

  } AF_LatinBlueRec, *AF_LatinBlue;


  typedef struct  AF_LatinAxisRec_
  {
    FT_Fixed         scale;
    FT_Pos           delta;

    FT_UInt          width_count;                 /* number of used widths */
    AF_WidthRec      widths[AF_LATIN_MAX_WIDTHS]; /* widths array          */
    FT_Pos           edge_distance_threshold;   /* used for creating edges */
    FT_Pos           standard_width;         /* the default stem thickness */
    FT_Bool          extra_light;         /* is standard width very light? */

    /* ignored for horizontal metrics */
    FT_UInt          blue_count;
    AF_LatinBlueRec  blues[AF_BLUE_STRINGSET_MAX_LEN];

    FT_Fixed         org_scale;
    FT_Pos           org_delta;

  } AF_LatinAxisRec, *AF_LatinAxis;


  typedef struct  AF_LatinMetricsRec_
  {
    AF_StyleMetricsRec  root;
    FT_UInt             units_per_em;
    AF_LatinAxisRec     axis[AF_DIMENSION_MAX];

  } AF_LatinMetricsRec, *AF_LatinMetrics;


  FT_LOCAL( FT_Error )
  af_latin_metrics_init( AF_StyleMetrics  metrics,
                         FT_Face          face );

  FT_LOCAL( void )
  af_latin_metrics_scale( AF_StyleMetrics  metrics,
                          AF_Scaler        scaler );

  FT_LOCAL( void )
  af_latin_metrics_init_widths( AF_LatinMetrics  metrics,
                                FT_Face          face );

  FT_LOCAL( void )
  af_latin_metrics_check_digits( AF_LatinMetrics  metrics,
                                 FT_Face          face );


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****           L A T I N   G L Y P H   A N A L Y S I S             *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

#define AF_LATIN_HINTS_HORZ_SNAP    ( 1U << 0 ) /* stem width snapping  */
#define AF_LATIN_HINTS_VERT_SNAP    ( 1U << 1 ) /* stem height snapping */
#define AF_LATIN_HINTS_STEM_ADJUST  ( 1U << 2 ) /* stem width/height    */
                                                /* adjustment           */
#define AF_LATIN_HINTS_MONO         ( 1U << 3 ) /* monochrome rendering */


#define AF_LATIN_HINTS_DO_HORZ_SNAP( h )             \
  AF_HINTS_TEST_OTHER( h, AF_LATIN_HINTS_HORZ_SNAP )

#define AF_LATIN_HINTS_DO_VERT_SNAP( h )             \
  AF_HINTS_TEST_OTHER( h, AF_LATIN_HINTS_VERT_SNAP )

#define AF_LATIN_HINTS_DO_STEM_ADJUST( h )             \
  AF_HINTS_TEST_OTHER( h, AF_LATIN_HINTS_STEM_ADJUST )

#define AF_LATIN_HINTS_DO_MONO( h )             \
  AF_HINTS_TEST_OTHER( h, AF_LATIN_HINTS_MONO )


  /*
   * The next functions shouldn't normally be exported.  However, other
   * writing systems might like to use these functions as-is.
   */
  FT_LOCAL( FT_Error )
  af_latin_hints_compute_segments( AF_GlyphHints  hints,
                                   AF_Dimension   dim );

  FT_LOCAL( void )
  af_latin_hints_link_segments( AF_GlyphHints  hints,
                                FT_UInt        width_count,
                                AF_WidthRec*   widths,
                                AF_Dimension   dim );

  FT_LOCAL( FT_Error )
  af_latin_hints_compute_edges( AF_GlyphHints  hints,
                                AF_Dimension   dim );

  FT_LOCAL( FT_Error )
  af_latin_hints_detect_features( AF_GlyphHints  hints,
                                  FT_UInt        width_count,
                                  AF_WidthRec*   widths,
                                  AF_Dimension   dim );

/* */

FT_END_HEADER

#endif /* AFLATIN_H_ */


/* END */

/****************************************************************************
 *
 * ftsdf.h
 *
 *   Signed Distance Field support (specification).
 *
 * Copyright (C) 2020-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * Written by Anuj Verma.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef FTSDF_H_
#define FTSDF_H_

#include <ft2build.h>
#include FT_CONFIG_CONFIG_H
#include <freetype/ftimage.h>

/* common properties and function */
#include "ftsdfcommon.h"

FT_BEGIN_HEADER

  /**************************************************************************
   *
   * @struct:
   *   SDF_Raster_Params
   *
   * @description:
   *   This struct must be passed to the raster render function
   *   @FT_Raster_RenderFunc instead of @FT_Raster_Params because the
   *   rasterizer requires some additional information to render properly.
   *
   * @fields:
   *   root ::
   *     The native raster parameters structure.
   *
   *   spread ::
   *     This is an essential parameter/property required by the renderer.
   *     `spread` defines the maximum unsigned value that is present in the
   *     final SDF output.  For the default value check file
   *     `ftsdfcommon.h`.
   *
   *   flip_sign ::
   *     By default positive values indicate positions inside of contours,
   *     i.e., filled by a contour.  If this property is true then that
   *     output will be the opposite of the default, i.e., negative values
   *     indicate positions inside of contours.
   *
   *   flip_y ::
   *     Setting this parameter to true maked the output image flipped
   *     along the y-axis.
   *
   *   overlaps ::
   *     Set this to true to generate SDF for glyphs having overlapping
   *     contours.  The overlapping support is limited to glyphs that do not
   *     have self-intersecting contours.  Also, removing overlaps require a
   *     considerable amount of extra memory; additionally, it will not work
   *     if generating SDF from bitmap.
   *
   * @note:
   *   All properties are valid for both the 'sdf' and 'bsdf' renderers; the
   *   exception is `overlaps`, which gets ignored by the 'bsdf' renderer.
   *
   */
  typedef struct  SDF_Raster_Params_
  {
    FT_Raster_Params  root;
    FT_UInt           spread;
    FT_Bool           flip_sign;
    FT_Bool           flip_y;
    FT_Bool           overlaps;

  } SDF_Raster_Params;


  /* rasterizer to convert outline to SDF */
  FT_EXPORT_VAR( const FT_Raster_Funcs )  ft_sdf_raster;

  /* rasterizer to convert bitmap to SDF */
  FT_EXPORT_VAR( const FT_Raster_Funcs )  ft_bitmap_sdf_raster;

FT_END_HEADER

#endif /* FTSDF_H_ */


/* END */

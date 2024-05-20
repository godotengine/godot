/****************************************************************************
 *
 * ftbsdf.c
 *
 *   Signed Distance Field support for bitmap fonts (body only).
 *
 * Copyright (C) 2020-2023 by
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


#include <freetype/internal/ftobjs.h>
#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftmemory.h>
#include <freetype/fttrigon.h>

#include "ftsdf.h"
#include "ftsdferrs.h"
#include "ftsdfcommon.h"


  /**************************************************************************
   *
   * A brief technical overview of how the BSDF rasterizer works
   * -----------------------------------------------------------
   *
   * [Notes]:
   *   * SDF stands for Signed Distance Field everywhere.
   *
   *   * BSDF stands for Bitmap to Signed Distance Field rasterizer.
   *
   *   * This renderer converts rasterized bitmaps to SDF.  There is another
   *     renderer called 'sdf', which generates SDF directly from outlines;
   *     see file `ftsdf.c` for more.
   *
   *   * The idea of generating SDF from bitmaps is taken from two research
   *     papers, where one is dependent on the other:
   *
   *     - Per-Erik Danielsson: Euclidean Distance Mapping
   *       http://webstaff.itn.liu.se/~stegu/JFA/Danielsson.pdf
   *
   *       From this paper we use the eight-point sequential Euclidean
   *       distance mapping (8SED).  This is the heart of the process used
   *       in this rasterizer.
   *
   *     - Stefan Gustavson, Robin Strand: Anti-aliased Euclidean distance transform.
   *       http://weber.itn.liu.se/~stegu/aadist/edtaa_preprint.pdf
   *
   *       The original 8SED algorithm discards the pixels' alpha values,
   *       which can contain information about the actual outline of the
   *       glyph.  This paper takes advantage of those alpha values and
   *       approximates outline pretty accurately.
   *
   *   * This rasterizer also works for monochrome bitmaps.  However, the
   *     result is not as accurate since we don't have any way to
   *     approximate outlines from binary bitmaps.
   *
   * ========================================================================
   *
   * Generating SDF from bitmap is done in several steps.
   *
   * (1) The only information we have is the bitmap itself.  It can
   *     be monochrome or anti-aliased.  If it is anti-aliased, pixel values
   *     are nothing but coverage values.  These coverage values can be used
   *     to extract information about the outline of the image.  For
   *     example, if the pixel's alpha value is 0.5, then we can safely
   *     assume that the outline passes through the center of the pixel.
   *
   * (2) Find edge pixels in the bitmap (see `bsdf_is_edge` for more).  For
   *     all edge pixels we use the Anti-aliased Euclidean distance
   *     transform algorithm and compute approximate edge distances (see
   *     `compute_edge_distance` and/or the second paper for more).
   *
   * (3) Now that we have computed approximate distances for edge pixels we
   *     use the 8SED algorithm to basically sweep the entire bitmap and
   *     compute distances for the rest of the pixels.  (Since the algorithm
   *     is pretty convoluted it is only explained briefly in a comment to
   *     function `edt8`.  To see the actual algorithm refer to the first
   *     paper.)
   *
   * (4) Finally, compute the sign for each pixel.  This is done in function
   *     `finalize_sdf`.  The basic idea is that if a pixel's original
   *     alpha/coverage value is greater than 0.5 then it is 'inside' (and
   *     'outside' otherwise).
   *
   * Pseudo Code:
   *
   * ```
   * b  = source bitmap;
   * t  = target bitmap;
   * dm = list of distances; // dimension equal to b
   *
   * foreach grid_point (x, y) in b:
   * {
   *   if (is_edge(x, y)):
   *     dm = approximate_edge_distance(b, x, y);
   *
   *   // do the 8SED on the distances
   *   edt8(dm);
   *
   *   // determine the signs
   *   determine_signs(dm):
   *
   *   // copy SDF data to the target bitmap
   *   copy(dm to t);
   * }
   *
   */


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  bsdf


  /**************************************************************************
   *
   * useful macros
   *
   */

#define ONE  65536 /* 1 in 16.16 */


  /**************************************************************************
   *
   * structs
   *
   */


  /**************************************************************************
   *
   * @Struct:
   *   BSDF_TRaster
   *
   * @Description:
   *   This struct is used in place of @FT_Raster and is stored within the
   *   internal FreeType renderer struct.  While rasterizing this is passed
   *   to the @FT_Raster_RenderFunc function, which then can be used however
   *   we want.
   *
   * @Fields:
   *   memory ::
   *     Used internally to allocate intermediate memory while raterizing.
   *
   */
  typedef struct  BSDF_TRaster_
  {
    FT_Memory  memory;

  } BSDF_TRaster, *BSDF_PRaster;


  /**************************************************************************
   *
   * @Struct:
   *   ED
   *
   * @Description:
   *   Euclidean distance.  It gets used for Euclidean distance transforms;
   *   it can also be interpreted as an edge distance.
   *
   * @Fields:
   *   dist ::
   *     Vector length of the `prox` parameter.  Can be squared or absolute
   *     depending on the `USE_SQUARED_DISTANCES` macro defined in file
   *     `ftsdfcommon.h`.
   *
   *   prox ::
   *     Vector to the nearest edge.  Can also be interpreted as shortest
   *     distance of a point.
   *
   *   alpha ::
   *     Alpha value of the original bitmap from which we generate SDF.
   *     Needed for computing the gradient and determining the proper sign
   *     of a pixel.
   *
   */
  typedef struct  ED_
  {
    FT_16D16      dist;
    FT_16D16_Vec  prox;
    FT_Byte       alpha;

  } ED;


  /**************************************************************************
   *
   * @Struct:
   *   BSDF_Worker
   *
   * @Description:
   *   A convenience struct that is passed to functions while generating
   *   SDF; most of those functions require the same parameters.
   *
   * @Fields:
   *   distance_map ::
   *     A one-dimensional array that gets interpreted as two-dimensional
   *     one.  It contains the Euclidean distances of all points of the
   *     bitmap.
   *
   *   width ::
   *     Width of the above `distance_map`.
   *
   *   rows ::
   *     Number of rows in the above `distance_map`.
   *
   *   params ::
   *     Internal parameters and properties required by the rasterizer.  See
   *     file `ftsdf.h` for more.
   *
   */
  typedef struct  BSDF_Worker_
  {
    ED*  distance_map;

    FT_Int  width;
    FT_Int  rows;

    SDF_Raster_Params  params;

  } BSDF_Worker;


  /**************************************************************************
   *
   * initializer
   *
   */

  static const ED  zero_ed = { 0, { 0, 0 }, 0 };


  /**************************************************************************
   *
   * rasterizer functions
   *
   */

  /**************************************************************************
   *
   * @Function:
   *   bsdf_is_edge
   *
   * @Description:
   *   Check whether a pixel is an edge pixel, i.e., whether it is
   *   surrounded by a completely black pixel (zero alpha), and the current
   *   pixel is not a completely black pixel.
   *
   * @Input:
   *   dm ::
   *     Array of distances.  The parameter must point to the current
   *     pixel, i.e., the pixel that is to be checked for being an edge.
   *
   *   x ::
   *     The x position of the current pixel.
   *
   *   y ::
   *     The y position of the current pixel.
   *
   *   w ::
   *     Width of the bitmap.
   *
   *   r ::
   *     Number of rows in the bitmap.
   *
   * @Return:
   *   1~if the current pixel is an edge pixel, 0~otherwise.
   *
   */

#ifdef CHECK_NEIGHBOR
#undef CHECK_NEIGHBOR
#endif

#define CHECK_NEIGHBOR( x_offset, y_offset )              \
          do                                              \
          {                                               \
            if ( x + x_offset >= 0 && x + x_offset < w && \
                 y + y_offset >= 0 && y + y_offset < r )  \
            {                                             \
              num_neighbors++;                            \
                                                          \
              to_check = dm + y_offset * w + x_offset;    \
              if ( to_check->alpha == 0 )                 \
              {                                           \
                is_edge = 1;                              \
                goto Done;                                \
              }                                           \
            }                                             \
          } while ( 0 )

  static FT_Bool
  bsdf_is_edge( ED*     dm,   /* distance map              */
                FT_Int  x,    /* x index of point to check */
                FT_Int  y,    /* y index of point to check */
                FT_Int  w,    /* width                     */
                FT_Int  r )   /* rows                      */
  {
    FT_Bool  is_edge       = 0;
    ED*      to_check      = NULL;
    FT_Int   num_neighbors = 0;


    if ( dm->alpha == 0 )
      goto Done;

    if ( dm->alpha > 0 && dm->alpha < 255 )
    {
      is_edge = 1;
      goto Done;
    }

    /* up */
    CHECK_NEIGHBOR(  0, -1 );

    /* down */
    CHECK_NEIGHBOR(  0,  1 );

    /* left */
    CHECK_NEIGHBOR( -1,  0 );

    /* right */
    CHECK_NEIGHBOR(  1,  0 );

    /* up left */
    CHECK_NEIGHBOR( -1, -1 );

    /* up right */
    CHECK_NEIGHBOR(  1, -1 );

    /* down left */
    CHECK_NEIGHBOR( -1,  1 );

    /* down right */
    CHECK_NEIGHBOR(  1,  1 );

    if ( num_neighbors != 8 )
      is_edge = 1;

  Done:
    return is_edge;
  }

#undef CHECK_NEIGHBOR


  /**************************************************************************
   *
   * @Function:
   *   compute_edge_distance
   *
   * @Description:
   *   Approximate the outline and compute the distance from `current`
   *   to the approximated outline.
   *
   * @Input:
   *   current ::
   *     Array of Euclidean distances.  `current` must point to the position
   *     for which the distance is to be caculated.  We treat this array as
   *     a two-dimensional array mapped to a one-dimensional array.
   *
   *   x ::
   *     The x coordinate of the `current` parameter in the array.
   *
   *   y ::
   *     The y coordinate of the `current` parameter in the array.
   *
   *   w ::
   *     The width of the distances array.
   *
   *   r ::
   *     Number of rows in the distances array.
   *
   * @Return:
   *   A vector pointing to the approximate edge distance.
   *
   * @Note:
   *   This is a computationally expensive function.  Try to reduce the
   *   number of calls to this function.  Moreover, this must only be used
   *   for edge pixel positions.
   *
   */
  static FT_16D16_Vec
  compute_edge_distance( ED*     current,
                         FT_Int  x,
                         FT_Int  y,
                         FT_Int  w,
                         FT_Int  r )
  {
    /*
     * This function, based on the paper presented by Stefan Gustavson and
     * Robin Strand, gets used to approximate edge distances from
     * anti-aliased bitmaps.
     *
     * The algorithm is as follows.
     *
     * (1) In anti-aliased images, the pixel's alpha value is the coverage
     *     of the pixel by the outline.  For example, if the alpha value is
     *     0.5f we can assume that the outline passes through the center of
     *     the pixel.
     *
     * (2) For this reason we can use that alpha value to approximate the real
     *     distance of the pixel to edge pretty accurately.  A simple
     *     approximation is `(0.5f - alpha)`, assuming that the outline is
     *     parallel to the x or y~axis.  However, in this algorithm we use a
     *     different approximation which is quite accurate even for
     *     non-axis-aligned edges.
     *
     * (3) The only remaining piece of information that we cannot
     *     approximate directly from the alpha is the direction of the edge.
     *     This is where we use Sobel's operator to compute the gradient of
     *     the pixel.  The gradient give us a pretty good approximation of
     *     the edge direction.  We use a 3x3 kernel filter to compute the
     *     gradient.
     *
     * (4) After the above two steps we have both the direction and the
     *     distance to the edge which is used to generate the Signed
     *     Distance Field.
     *
     * References:
     *
     * - Anti-Aliased Euclidean Distance Transform:
     *     http://weber.itn.liu.se/~stegu/aadist/edtaa_preprint.pdf
     * - Sobel Operator:
     *     https://en.wikipedia.org/wiki/Sobel_operator
     */

    FT_16D16_Vec  g = { 0, 0 };
    FT_16D16      dist, current_alpha;
    FT_16D16      a1, temp;
    FT_16D16      gx, gy;
    FT_16D16      alphas[9];


    /* Since our spread cannot be 0, this condition */
    /* can never be true.                           */
    if ( x <= 0 || x >= w - 1 ||
         y <= 0 || y >= r - 1 )
      return g;

    /* initialize the alphas */
    alphas[0] = 256 * (FT_16D16)current[-w - 1].alpha;
    alphas[1] = 256 * (FT_16D16)current[-w    ].alpha;
    alphas[2] = 256 * (FT_16D16)current[-w + 1].alpha;
    alphas[3] = 256 * (FT_16D16)current[    -1].alpha;
    alphas[4] = 256 * (FT_16D16)current[     0].alpha;
    alphas[5] = 256 * (FT_16D16)current[     1].alpha;
    alphas[6] = 256 * (FT_16D16)current[ w - 1].alpha;
    alphas[7] = 256 * (FT_16D16)current[ w    ].alpha;
    alphas[8] = 256 * (FT_16D16)current[ w + 1].alpha;

    current_alpha = alphas[4];

    /* Compute the gradient using the Sobel operator. */
    /* In this case we use the following 3x3 filters: */
    /*                                                */
    /* For x: |   -1     0   -1    |                  */
    /*        | -root(2) 0 root(2) |                  */
    /*        |    -1    0    1    |                  */
    /*                                                */
    /* For y: |   -1 -root(2) -1   |                  */
    /*        |    0    0      0   |                  */
    /*        |    1  root(2)  1   |                  */
    /*                                                */
    /* [Note]: 92681 is root(2) in 16.16 format.      */
    g.x = -alphas[0] -
           FT_MulFix( alphas[3], 92681 ) -
           alphas[6] +
           alphas[2] +
           FT_MulFix( alphas[5], 92681 ) +
           alphas[8];

    g.y = -alphas[0] -
           FT_MulFix( alphas[1], 92681 ) -
           alphas[2] +
           alphas[6] +
           FT_MulFix( alphas[7], 92681 ) +
           alphas[8];

    FT_Vector_NormLen( &g );

    /* The gradient gives us the direction of the    */
    /* edge for the current pixel.  Once we have the */
    /* approximate direction of the edge, we can     */
    /* approximate the edge distance much better.    */

    if ( g.x == 0 || g.y == 0 )
      dist = ONE / 2 - alphas[4];
    else
    {
      gx = g.x;
      gy = g.y;

      gx = FT_ABS( gx );
      gy = FT_ABS( gy );

      if ( gx < gy )
      {
        temp = gx;
        gx   = gy;
        gy   = temp;
      }

      a1 = FT_DivFix( gy, gx ) / 2;

      if ( current_alpha < a1 )
        dist = ( gx + gy ) / 2 -
               square_root( 2 * FT_MulFix( gx,
                                           FT_MulFix( gy,
                                                      current_alpha ) ) );

      else if ( current_alpha < ( ONE - a1 ) )
        dist = FT_MulFix( ONE / 2 - current_alpha, gx );

      else
        dist = -( gx + gy ) / 2 +
               square_root( 2 * FT_MulFix( gx,
                                           FT_MulFix( gy,
                                                      ONE - current_alpha ) ) );
    }

    g.x = FT_MulFix( g.x, dist );
    g.y = FT_MulFix( g.y, dist );

    return g;
  }


  /**************************************************************************
   *
   * @Function:
   *   bsdf_approximate_edge
   *
   * @Description:
   *   Loops over all the pixels and call `compute_edge_distance` only for
   *   edge pixels.  This maked the process a lot faster since
   *   `compute_edge_distance` uses functions such as `FT_Vector_NormLen',
   *   which are quite slow.
   *
   * @InOut:
   *   worker ::
   *     Contains the distance map as well as all the relevant parameters
   *     required by the function.
   *
   * @Return:
   *   FreeType error, 0 means success.
   *
   * @Note:
   *   The function directly manipulates `worker->distance_map`.
   *
   */
  static FT_Error
  bsdf_approximate_edge( BSDF_Worker*  worker )
  {
    FT_Error  error = FT_Err_Ok;
    FT_Int    i, j;
    FT_Int    index;
    ED*       ed;


    if ( !worker || !worker->distance_map )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    ed = worker->distance_map;

    for ( j = 0; j < worker->rows; j++ )
    {
      for ( i = 0; i < worker->width; i++ )
      {
        index = j * worker->width + i;

        if ( bsdf_is_edge( worker->distance_map + index,
                           i, j,
                           worker->width,
                           worker->rows ) )
        {
          /* approximate the edge distance for edge pixels */
          ed[index].prox = compute_edge_distance( ed + index,
                                                  i, j,
                                                  worker->width,
                                                  worker->rows );
          ed[index].dist = VECTOR_LENGTH_16D16( ed[index].prox );
        }
        else
        {
          /* for non-edge pixels assign far away distances */
          ed[index].dist   = 400 * ONE;
          ed[index].prox.x = 200 * ONE;
          ed[index].prox.y = 200 * ONE;
        }
      }
    }

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   bsdf_init_distance_map
   *
   * @Description:
   *   Initialize the distance map according to the '8-point sequential
   *   Euclidean distance mapping' (8SED) algorithm.  Basically it copies
   *   the `source` bitmap alpha values to the `distance_map->alpha`
   *   parameter of `worker`.
   *
   * @Input:
   *   source ::
   *     Source bitmap to copy the data from.
   *
   * @Output:
   *   worker ::
   *     Target distance map to copy the data to.
   *
   * @Return:
   *   FreeType error, 0 means success.
   *
   */
  static FT_Error
  bsdf_init_distance_map( const FT_Bitmap*  source,
                          BSDF_Worker*      worker )
  {
    FT_Error  error = FT_Err_Ok;

    FT_Int    x_diff, y_diff;
    FT_Int    t_i, t_j, s_i, s_j;
    FT_Byte*  s;
    ED*       t;


    /* again check the parameters (probably unnecessary) */
    if ( !source || !worker )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* Because of the way we convert a bitmap to SDF, */
    /* i.e., aligning the source to the center of the */
    /* target, the target's width and rows must be    */
    /* checked before copying.                        */
    if ( worker->width < (FT_Int)source->width ||
         worker->rows  < (FT_Int)source->rows  )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* check pixel mode */
    if ( source->pixel_mode == FT_PIXEL_MODE_NONE )
    {
      FT_ERROR(( "bsdf_copy_source_to_target:"
                 " Invalid pixel mode of source bitmap" ));
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

#ifdef FT_DEBUG_LEVEL_TRACE
    if ( source->pixel_mode == FT_PIXEL_MODE_MONO )
    {
      FT_TRACE0(( "bsdf_copy_source_to_target:"
                  " The `bsdf' renderer can convert monochrome\n" ));
      FT_TRACE0(( "                           "
                  " bitmaps to SDF but the results are not perfect\n" ));
      FT_TRACE0(( "                           "
                  " because there is no way to approximate actual\n" ));
      FT_TRACE0(( "                           "
                  " outlines from monochrome bitmaps.  Consider\n" ));
      FT_TRACE0(( "                           "
                  " using an anti-aliased bitmap instead.\n" ));
    }
#endif

    /* Calculate the width and row differences */
    /* between target and source.              */
    x_diff = worker->width - (int)source->width;
    y_diff = worker->rows - (int)source->rows;

    x_diff /= 2;
    y_diff /= 2;

    t = (ED*)worker->distance_map;
    s = source->buffer;

    /* For now we only support pixel mode `FT_PIXEL_MODE_MONO`  */
    /* and `FT_PIXEL_MODE_GRAY`.  More will be added later.     */
    /*                                                          */
    /* [NOTE]: We can also use @FT_Bitmap_Convert to convert    */
    /*         bitmap to 8bpp.  To avoid extra allocation and   */
    /*         since the target bitmap can be 16bpp we manually */
    /*         convert the source bitmap to the desired bpp.    */

    switch ( source->pixel_mode )
    {
    case FT_PIXEL_MODE_MONO:
      {
        FT_Int  t_width = worker->width;
        FT_Int  t_rows  = worker->rows;
        FT_Int  s_width = (int)source->width;
        FT_Int  s_rows  = (int)source->rows;


        for ( t_j = 0; t_j < t_rows; t_j++ )
        {
          for ( t_i = 0; t_i < t_width; t_i++ )
          {
            FT_Int   t_index = t_j * t_width + t_i;
            FT_Int   s_index;
            FT_Int   div, mod;
            FT_Byte  pixel, byte;


            t[t_index] = zero_ed;

            s_i = t_i - x_diff;
            s_j = t_j - y_diff;

            /* Assign 0 to padding similar to */
            /* the source bitmap.             */
            if ( s_i < 0 || s_i >= s_width ||
                 s_j < 0 || s_j >= s_rows  )
              continue;

            if ( worker->params.flip_y )
              s_index = ( s_rows - s_j - 1 ) * source->pitch;
            else
              s_index = s_j * source->pitch;

            div = s_index + s_i / 8;
            mod = 7 - s_i % 8;

            pixel = s[div];
            byte  = (FT_Byte)( 1 << mod );

            t[t_index].alpha = pixel & byte ? 255 : 0;
          }
        }
      }
      break;

    case FT_PIXEL_MODE_GRAY:
      {
        FT_Int  t_width = worker->width;
        FT_Int  t_rows  = worker->rows;
        FT_Int  s_width = (int)source->width;
        FT_Int  s_rows  = (int)source->rows;


        /* loop over all pixels and assign pixel values from source */
        for ( t_j = 0; t_j < t_rows; t_j++ )
        {
          for ( t_i = 0; t_i < t_width; t_i++ )
          {
            FT_Int  t_index = t_j * t_width + t_i;
            FT_Int  s_index;


            t[t_index] = zero_ed;

            s_i = t_i - x_diff;
            s_j = t_j - y_diff;

            /* Assign 0 to padding similar to */
            /* the source bitmap.             */
            if ( s_i < 0 || s_i >= s_width ||
                 s_j < 0 || s_j >= s_rows  )
              continue;

            if ( worker->params.flip_y )
              s_index = ( s_rows - s_j - 1 ) * s_width + s_i;
            else
              s_index = s_j * s_width + s_i;

            /* simply copy the alpha values */
            t[t_index].alpha = s[s_index];
          }
        }
      }
      break;

    default:
      FT_ERROR(( "bsdf_copy_source_to_target:"
                 " unsopported pixel mode of source bitmap\n" ));

      error = FT_THROW( Unimplemented_Feature );
      break;
    }

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   compare_neighbor
   *
   * @Description:
   *   Compare neighbor pixel (which is defined by the offset) and update
   *   `current` distance if the new distance is shorter than the original.
   *
   * @Input:
   *   x_offset ::
   *     X offset of the neighbor to be checked.  The offset is relative to
   *     the `current`.
   *
   *   y_offset ::
   *     Y offset of the neighbor to be checked.  The offset is relative to
   *     the `current`.
   *
   *   width ::
   *     Width of the `current` array.
   *
   * @InOut:
   *   current ::
   *     Pointer into array of distances.  This parameter must point to the
   *     position whose neighbor is to be checked.  The array is treated as
   *     a two-dimensional array.
   *
   */
  static void
  compare_neighbor( ED*     current,
                    FT_Int  x_offset,
                    FT_Int  y_offset,
                    FT_Int  width )
  {
    ED*           to_check;
    FT_16D16      dist;
    FT_16D16_Vec  dist_vec;


    to_check = current + ( y_offset * width ) + x_offset;

    /*
     * While checking for the nearest point we first approximate the
     * distance of `current` by adding the deviation (which is sqrt(2) at
     * most).  Only if the new value is less than the current value we
     * calculate the actual distances using `FT_Vector_Length`.  This last
     * step can be omitted by using squared distances.
     */

    /*
     * Approximate the distance.  We subtract 1 to avoid precision errors,
     * which could happen because the two directions can be opposite.
     */
    dist = to_check->dist - ONE;

    if ( dist < current->dist )
    {
      dist_vec = to_check->prox;

      dist_vec.x += x_offset * ONE;
      dist_vec.y += y_offset * ONE;
      dist = VECTOR_LENGTH_16D16( dist_vec );

      if ( dist < current->dist )
      {
        current->dist = dist;
        current->prox = dist_vec;
      }
    }
  }


  /**************************************************************************
   *
   * @Function:
   *   first_pass
   *
   * @Description:
   *   First pass of the 8SED algorithm.  Loop over the bitmap from top to
   *   bottom and scan each row left to right, updating the distances in
   *   `worker->distance_map`.
   *
   * @InOut:
   *   worker::
   *     Contains all the relevant parameters.
   *
   */
  static void
  first_pass( BSDF_Worker*  worker )
  {
    FT_Int  i, j; /* iterators    */
    FT_Int  w, r; /* width, rows  */
    ED*     dm;   /* distance map */


    dm = worker->distance_map;
    w  = worker->width;
    r  = worker->rows;

    /* Start scanning from top to bottom and sweep each    */
    /* row back and forth comparing the distances of the   */
    /* neighborhood.  Leave the first row as it has no top */
    /* neighbor; it will be covered in the second scan of  */
    /* the image (from bottom to top).                     */
    for ( j = 1; j < r; j++ )
    {
      FT_Int  index;
      ED*     current;


      /* Forward pass of rows (left -> right).  Leave the first  */
      /* column, which gets covered in the backward pass.        */
      for ( i = 1; i < w - 1; i++ )
      {
        index   = j * w + i;
        current = dm + index;

        /* left-up */
        compare_neighbor( current, -1, -1, w );
        /* up */
        compare_neighbor( current,  0, -1, w );
        /* up-right */
        compare_neighbor( current,  1, -1, w );
        /* left */
        compare_neighbor( current, -1,  0, w );
      }

      /* Backward pass of rows (right -> left).  Leave the last */
      /* column, which was already covered in the forward pass. */
      for ( i = w - 2; i >= 0; i-- )
      {
        index   = j * w + i;
        current = dm + index;

        /* right */
        compare_neighbor( current, 1, 0, w );
      }
    }
  }


  /**************************************************************************
   *
   * @Function:
   *   second_pass
   *
   * @Description:
   *   Second pass of the 8SED algorithm.  Loop over the bitmap from bottom
   *   to top and scan each row left to right, updating the distances in
   *   `worker->distance_map`.
   *
   * @InOut:
   *   worker::
   *     Contains all the relevant parameters.
   *
   */
  static void
  second_pass( BSDF_Worker*  worker )
  {
    FT_Int  i, j; /* iterators    */
    FT_Int  w, r; /* width, rows  */
    ED*     dm;   /* distance map */


    dm = worker->distance_map;
    w  = worker->width;
    r  = worker->rows;

    /* Start scanning from bottom to top and sweep each    */
    /* row back and forth comparing the distances of the   */
    /* neighborhood.  Leave the last row as it has no down */
    /* neighbor; it is already covered in the first scan   */
    /* of the image (from top to bottom).                  */
    for ( j = r - 2; j >= 0; j-- )
    {
      FT_Int  index;
      ED*     current;


      /* Forward pass of rows (left -> right).  Leave the first */
      /* column, which gets covered in the backward pass.       */
      for ( i = 1; i < w - 1; i++ )
      {
        index   = j * w + i;
        current = dm + index;

        /* left-up */
        compare_neighbor( current, -1, 1, w );
        /* up */
        compare_neighbor( current,  0, 1, w );
        /* up-right */
        compare_neighbor( current,  1, 1, w );
        /* left */
        compare_neighbor( current, -1, 0, w );
      }

      /* Backward pass of rows (right -> left).  Leave the last */
      /* column, which was already covered in the forward pass. */
      for ( i = w - 2; i >= 0; i-- )
      {
        index   = j * w + i;
        current = dm + index;

        /* right */
        compare_neighbor( current, 1, 0, w );
      }
    }
  }


  /**************************************************************************
   *
   * @Function:
   *   edt8
   *
   * @Description:
   *   Compute the distance map of the a bitmap.  Execute both first and
   *   second pass of the 8SED algorithm.
   *
   * @InOut:
   *   worker::
   *     Contains all the relevant parameters.
   *
   * @Return:
   *   FreeType error, 0 means success.
   *
   */
  static FT_Error
  edt8( BSDF_Worker*  worker )
  {
    FT_Error  error = FT_Err_Ok;


    if ( !worker || !worker->distance_map )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* first scan of the image */
    first_pass( worker );

    /* second scan of the image */
    second_pass( worker );

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   finalize_sdf
   *
   * @Description:
   *   Copy the SDF data from `worker->distance_map` to the `target` bitmap.
   *   Also transform the data to output format, (which is 6.10 fixed-point
   *   format at the moment).
   *
   * @Input:
   *   worker ::
   *     Contains source distance map and other SDF data.
   *
   * @Output:
   *   target ::
   *     Target bitmap to which the SDF data is copied to.
   *
   * @Return:
   *   FreeType error, 0 means success.
   *
   */
  static FT_Error
  finalize_sdf( BSDF_Worker*      worker,
                const FT_Bitmap*  target )
  {
    FT_Error  error = FT_Err_Ok;

    FT_Int  w, r;
    FT_Int  i, j;

    FT_SDFFormat*  t_buffer;
    FT_16D16       sp_sq, spread;


    if ( !worker || !target )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    w        = (int)target->width;
    r        = (int)target->rows;
    t_buffer = (FT_SDFFormat*)target->buffer;

    if ( w != worker->width ||
         r != worker->rows  )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    spread = (FT_16D16)FT_INT_16D16( worker->params.spread );

#if USE_SQUARED_DISTANCES
    sp_sq = (FT_16D16)FT_INT_16D16( worker->params.spread *
                                    worker->params.spread );
#else
    sp_sq = (FT_16D16)FT_INT_16D16( worker->params.spread );
#endif

    for ( j = 0; j < r; j++ )
    {
      for ( i = 0; i < w; i++ )
      {
        FT_Int        index;
        FT_16D16      dist;
        FT_SDFFormat  final_dist;
        FT_Char       sign;


        index = j * w + i;
        dist  = worker->distance_map[index].dist;

        if ( dist < 0 || dist > sp_sq )
          dist = sp_sq;

#if USE_SQUARED_DISTANCES
        dist = square_root( dist );
#endif

        /* We assume that if the pixel is inside a contour */
        /* its coverage value must be > 127.               */
        sign = worker->distance_map[index].alpha < 127 ? -1 : 1;

        /* flip the sign according to the property */
        if ( worker->params.flip_sign )
          sign = -sign;

        /* concatenate from 16.16 to appropriate format */
        final_dist = map_fixed_to_sdf( dist * sign, spread );

        t_buffer[index] = final_dist;
      }
    }

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * interface functions
   *
   */

  /* called when adding a new module through @FT_Add_Module */
  static FT_Error
  bsdf_raster_new( void*       memory_,    /* FT_Memory     */
                   FT_Raster*  araster_ )  /* BSDF_PRaster* */
  {
    FT_Memory      memory  = (FT_Memory)memory_;
    BSDF_PRaster*  araster = (BSDF_PRaster*)araster_;

    FT_Error      error;
    BSDF_PRaster  raster = NULL;


    if ( !FT_NEW( raster ) )
      raster->memory = memory;

    *araster = raster;

    return error;
  }


  /* unused */
  static void
  bsdf_raster_reset( FT_Raster       raster,
                     unsigned char*  pool_base,
                     unsigned long   pool_size )
  {
    FT_UNUSED( raster );
    FT_UNUSED( pool_base );
    FT_UNUSED( pool_size );
  }


  /* unused */
  static FT_Error
  bsdf_raster_set_mode( FT_Raster      raster,
                        unsigned long  mode,
                        void*          args )
  {
    FT_UNUSED( raster );
    FT_UNUSED( mode );
    FT_UNUSED( args );

    return FT_Err_Ok;
  }


  /* called while rendering through @FT_Render_Glyph */
  static FT_Error
  bsdf_raster_render( FT_Raster                raster,
                      const FT_Raster_Params*  params )
  {
    FT_Error   error  = FT_Err_Ok;
    FT_Memory  memory = NULL;

    const FT_Bitmap*  source = NULL;
    const FT_Bitmap*  target = NULL;

    BSDF_TRaster*  bsdf_raster = (BSDF_TRaster*)raster;
    BSDF_Worker    worker;

    const SDF_Raster_Params*  sdf_params = (const SDF_Raster_Params*)params;


    worker.distance_map = NULL;

    /* check for valid parameters */
    if ( !raster || !params )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* check whether the flag is set */
    if ( sdf_params->root.flags != FT_RASTER_FLAG_SDF )
    {
      error = FT_THROW( Raster_Corrupted );
      goto Exit;
    }

    source = (const FT_Bitmap*)sdf_params->root.source;
    target = (const FT_Bitmap*)sdf_params->root.target;

    /* check source and target bitmap */
    if ( !source || !target )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    memory = bsdf_raster->memory;
    if ( !memory )
    {
      FT_TRACE0(( "bsdf_raster_render: Raster not set up properly,\n" ));
      FT_TRACE0(( "                    unable to find memory handle.\n" ));

      error = FT_THROW( Invalid_Handle );
      goto Exit;
    }

    /* check whether spread is set properly */
    if ( sdf_params->spread > MAX_SPREAD ||
         sdf_params->spread < MIN_SPREAD )
    {
      FT_TRACE0(( "bsdf_raster_render:"
                  " The `spread' field of `SDF_Raster_Params'\n" ));
      FT_TRACE0(( "                   "
                  " is invalid; the value of this field must be\n" ));
      FT_TRACE0(( "                   "
                  " within [%d, %d].\n",
                  MIN_SPREAD, MAX_SPREAD ));
      FT_TRACE0(( "                   "
                  " Also, you must pass `SDF_Raster_Params'\n" ));
      FT_TRACE0(( "                   "
                  " instead of the default `FT_Raster_Params'\n" ));
      FT_TRACE0(( "                   "
                  " while calling this function and set the fields\n" ));
      FT_TRACE0(( "                   "
                  " accordingly.\n" ));

      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* set up the worker */

    /* allocate the distance map */
    if ( FT_QALLOC_MULT( worker.distance_map, target->rows,
                         target->width * sizeof ( *worker.distance_map ) ) )
      goto Exit;

    worker.width  = (int)target->width;
    worker.rows   = (int)target->rows;
    worker.params = *sdf_params;

    FT_CALL( bsdf_init_distance_map( source, &worker ) );
    FT_CALL( bsdf_approximate_edge( &worker ) );
    FT_CALL( edt8( &worker ) );
    FT_CALL( finalize_sdf( &worker, target ) );

    FT_TRACE0(( "bsdf_raster_render: Total memory used = %ld\n",
                worker.width * worker.rows *
                  (long)sizeof ( *worker.distance_map ) ));

  Exit:
    if ( worker.distance_map )
      FT_FREE( worker.distance_map );

    return error;
  }


  /* called while deleting `FT_Library` only if the module is added */
  static void
  bsdf_raster_done( FT_Raster  raster )
  {
    FT_Memory  memory = (FT_Memory)((BSDF_TRaster*)raster)->memory;


    FT_FREE( raster );
  }


  FT_DEFINE_RASTER_FUNCS(
    ft_bitmap_sdf_raster,

    FT_GLYPH_FORMAT_BITMAP,

    (FT_Raster_New_Func)     bsdf_raster_new,       /* raster_new      */
    (FT_Raster_Reset_Func)   bsdf_raster_reset,     /* raster_reset    */
    (FT_Raster_Set_Mode_Func)bsdf_raster_set_mode,  /* raster_set_mode */
    (FT_Raster_Render_Func)  bsdf_raster_render,    /* raster_render   */
    (FT_Raster_Done_Func)    bsdf_raster_done       /* raster_done     */
  )


/* END */

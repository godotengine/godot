/****************************************************************************
 *
 * ftsdfcommon.c
 *
 *   Auxiliary data for Signed Distance Field support (body).
 *
 * Copyright (C) 2020-2025 by
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


#include "ftsdf.h"
#include "ftsdfcommon.h"


  /**************************************************************************
   *
   * format and sign manipulating functions
   *
   */

  /*
   * Convert 16.16 fixed-point values to the desired output format.
   * In this case we reduce 16.16 fixed-point values to normalized
   * 8-bit values.
   *
   * The `max_value` in the parameter is the maximum value in the
   * distance field map and is equal to the spread.  We normalize
   * the distances using this value instead of computing the maximum
   * value for the entire bitmap.
   *
   * You can use this function to map the 16.16 signed values to any
   * format required.  Do note that the output buffer is 8-bit, so only
   * use an 8-bit format for `FT_SDFFormat`, or increase the buffer size in
   * `ftsdfrend.c`.
   */
  FT_LOCAL_DEF( FT_SDFFormat )
  map_fixed_to_sdf( FT_16D16  dist,
                    FT_16D16  max_value )
  {
    FT_SDFFormat  out;
    FT_16D16      udist;


    /* normalize the distance values */
    dist = FT_DivFix( dist, max_value );

    udist = dist < 0 ? -dist : dist;

    /* Reduce the distance values to 8 bits.                   */
    /*                                                         */
    /* Since +1/-1 in 16.16 takes the 16th bit, we right-shift */
    /* the number by 9 to make it fit into the 7-bit range.    */
    /*                                                         */
    /* One bit is reserved for the sign.                       */
    udist >>= 9;

    /* Since `char` can only store a maximum positive value    */
    /* of 127 we need to make sure it does not wrap around and */
    /* give a negative value.                                  */
    if ( dist > 0 && udist > 127 )
      udist = 127;
    if ( dist < 0 && udist > 128 )
      udist = 128;

    /* Output the data; negative values are from [0, 127] and positive    */
    /* from [128, 255].  One important thing is that negative values      */
    /* are inverted here, that means [0, 128] maps to [-128, 0] linearly. */
    /* More on that in `freetype.h` near the documentation of             */
    /* `FT_RENDER_MODE_SDF`.                                              */
    out = dist < 0 ? 128 - (FT_SDFFormat)udist
                   : (FT_SDFFormat)udist + 128;

    return out;
  }


  /*
   * Invert the signed distance packed into the corresponding format.
   * So if the values are negative they will become positive in the
   * chosen format.
   *
   * [Note]: This function should only be used after converting the
   *         16.16 signed distance values to `FT_SDFFormat`.  If that
   *         conversion has not been done, then simply invert the sign
   *         and use the above function to pack the values.
   */
  FT_LOCAL_DEF( FT_SDFFormat )
  invert_sign( FT_SDFFormat  dist )
  {
    return 255 - dist;
  }


/* END */

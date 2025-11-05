/****************************************************************************
 *
 * afadjust.h
 *
 *   Auto-fitter routines to adjust components based on charcode (header).
 *
 * Copyright (C) 2023-2025 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * Written by Craig White <gerzytet@gmail.com>.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef AFADJUST_H_
#define AFADJUST_H_

#include <freetype/fttypes.h>

#include "afglobal.h"
#include "aftypes.h"


FT_BEGIN_HEADER

  /*
   * Adjustment type flags.
   *
   * They also specify topological constraints that the auto-hinter relies
   * on.  For example, using `AF_ADJUST_UP` implies that we have two
   * enclosing contours, one for the base glyph and one for the diacritic
   * above, and no other contour inbetween or above.  With 'enclosing' it is
   * meant that such a contour can contain more inner contours.
   *
   */

  /* Find the topmost contour and push it up until its lowest point is */
  /* one pixel above the highest point not enclosed by that contour.   */
#define AF_ADJUST_UP  0x01

  /* Find the bottommost contour and push it down until its highest point */
  /* is one pixel below the lowest point not enclosed by that contour.    */
#define AF_ADJUST_DOWN  0x02

  /* Find the contour below the topmost contour and push it up, together */
  /* with the topmost contour, until its lowest point is one pixel above */
  /* the highest point not enclosed by that contour.  This flag is       */
  /* mutually exclusive with `AF_ADJUST_UP`.                             */
#define AF_ADJUST_UP2  0x04

  /* Find the contour above the bottommost contour and push it down,  */
  /* together with the bottommost contour, until its highest point is */
  /* one pixel below the lowest point not enclosed by that contour.   */
  /* This flag is mutually exclusive with `AF_ADJUST_DOWN`.           */
#define AF_ADJUST_DOWN2  0x08

  /* The topmost contour is a tilde.  Enlarge it vertically so that it    */
  /* stays legible at small sizes, not degenerating to a horizontal line. */
#define AF_ADJUST_TILDE_TOP  0x10

  /* The bottommost contour is a tilde.  Enlarge it vertically so that it */
  /* stays legible at small sizes, not degenerating to a horizontal line. */
#define AF_ADJUST_TILDE_BOTTOM  0x20

  /* The contour below the topmost contour is a tilde.  Enlarge it        */
  /* vertically so that it stays legible at small sizes, not degenerating */
  /* to a horizontal line.  To be used with `AF_ADJUST_UP2` only.         */
#define AF_ADJUST_TILDE_TOP2  0x40

  /* The contour above the bottommost contour is a tilde.  Enlarge it     */
  /* vertically so that it stays legible at small sizes, not degenerating */
  /* to a horizontal line.  To be used with `AF_ADJUST_DOWN2` only.       */
#define AF_ADJUST_TILDE_BOTTOM2  0x80

  /* Make the auto-hinter ignore any diacritic (either a separate contour */
  /* or part of the base character outline) that is attached to the top   */
  /* of an uppercase base character.                                      */
#define AF_IGNORE_CAPITAL_TOP  0x100

  /* Make the auto-hinter ignore any diacritic (either a separate contour */
  /* or part of the base character outline) that is attached to the       */
  /* bottom of an uppercase base character.                               */
#define AF_IGNORE_CAPITAL_BOTTOM  0x200

  /* Make the auto-hinter ignore any diacritic (either a separate contour */
  /* or part of the base character outline) that is attached to the top   */
  /* of a lowercase base character.                                       */
#define AF_IGNORE_SMALL_TOP  0x400

  /* Make the auto-hinter ignore any diacritic (either a separate contour */
  /* or part of the base character outline) that is attached to the       */
  /* bottom of a lowercase base character.                                */
#define AF_IGNORE_SMALL_BOTTOM  0x800

  /* By default, the AF_ADJUST_XXX flags are applied only if diacritics */
  /* have a 'small' height (based on some heuristic checks).  If this   */
  /* flag is set, no such check is performed.                           */
#define AF_ADJUST_NO_HEIGHT_CHECK  0x1000

  /* No adjustment, i.e., no flag is set. */
#define AF_ADJUST_NONE  0x00


  FT_LOCAL( FT_UInt32 )
  af_adjustment_database_lookup( FT_UInt32  codepoint );

  /* Allocate and populate the reverse character map, */
  /* using the character map within the face.         */
  FT_LOCAL( FT_Error )
  af_reverse_character_map_new( FT_Hash         *map,
                                AF_StyleMetrics  metrics );

  /* Free the reverse character map. */
  FT_LOCAL( FT_Error )
  af_reverse_character_map_done( FT_Hash    map,
                                 FT_Memory  memory );


FT_END_HEADER

#endif /* AFADJUST_H_ */


/* END */

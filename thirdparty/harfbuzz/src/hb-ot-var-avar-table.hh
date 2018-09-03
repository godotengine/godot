/*
 * Copyright © 2017  Google, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_OT_VAR_AVAR_TABLE_HH
#define HB_OT_VAR_AVAR_TABLE_HH

#include "hb-open-type-private.hh"

namespace OT {


struct AxisValueMap
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  public:
  F2DOT14	fromCoord;	/* A normalized coordinate value obtained using
				 * default normalization. */
  F2DOT14	toCoord;	/* The modified, normalized coordinate value. */

  public:
  DEFINE_SIZE_STATIC (4);
};

struct SegmentMaps : ArrayOf<AxisValueMap>
{
  inline int map (int value) const
  {
    /* The following special-cases are not part of OpenType, which requires
     * that at least -1, 0, and +1 must be mapped. But we include these as
     * part of a better error recovery scheme. */

    if (len < 2)
    {
      if (!len)
	return value;
      else /* len == 1*/
	return value - array[0].fromCoord + array[0].toCoord;
    }

    if (value <= array[0].fromCoord)
      return value - array[0].fromCoord + array[0].toCoord;

    unsigned int i;
    unsigned int count = len;
    for (i = 1; i < count && value > array[i].fromCoord; i++)
      ;

    if (value >= array[i].fromCoord)
      return value - array[i].fromCoord + array[i].toCoord;

    if (unlikely (array[i-1].fromCoord == array[i].fromCoord))
      return array[i-1].toCoord;

    int denom = array[i].fromCoord - array[i-1].fromCoord;
    return array[i-1].toCoord +
	   ((array[i].toCoord - array[i-1].toCoord) *
	    (value - array[i-1].fromCoord) + denom/2) / denom;
  }

  DEFINE_SIZE_ARRAY (2, array);
};

/*
 * avar — Axis Variations Table
 */

#define HB_OT_TAG_avar HB_TAG('a','v','a','r')

struct avar
{
  static const hb_tag_t tableTag	= HB_OT_TAG_avar;

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!(version.sanitize (c) &&
		    version.major == 1 &&
		    c->check_struct (this))))
      return_trace (false);

    const SegmentMaps *map = &axisSegmentMapsZ;
    unsigned int count = axisCount;
    for (unsigned int i = 0; i < count; i++)
    {
      if (unlikely (!map->sanitize (c)))
        return_trace (false);
      map = &StructAfter<SegmentMaps> (*map);
    }

    return_trace (true);
  }

  inline void map_coords (int *coords, unsigned int coords_length) const
  {
    unsigned int count = MIN<unsigned int> (coords_length, axisCount);

    const SegmentMaps *map = &axisSegmentMapsZ;
    for (unsigned int i = 0; i < count; i++)
    {
      coords[i] = map->map (coords[i]);
      map = &StructAfter<SegmentMaps> (*map);
    }
  }

  protected:
  FixedVersion<>version;	/* Version of the avar table
				 * initially set to 0x00010000u */
  HBUINT16	reserved;	/* This field is permanently reserved. Set to 0. */
  HBUINT16	axisCount;	/* The number of variation axes in the font. This
				 * must be the same number as axisCount in the
				 * 'fvar' table. */
  SegmentMaps	axisSegmentMapsZ;

  public:
  DEFINE_SIZE_MIN (8);
};

} /* namespace OT */


#endif /* HB_OT_VAR_AVAR_TABLE_HH */

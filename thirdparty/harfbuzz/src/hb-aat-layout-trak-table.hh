/*
 * Copyright © 2018  Ebrahim Byagowi
 * Copyright © 2018  Google, Inc.
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

#ifndef HB_AAT_LAYOUT_TRAK_TABLE_HH
#define HB_AAT_LAYOUT_TRAK_TABLE_HH

#include "hb-aat-layout-common.hh"
#include "hb-ot-layout.hh"
#include "hb-open-type.hh"

/*
 * trak -- Tracking
 * https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6trak.html
 */
#define HB_AAT_TAG_trak HB_TAG('t','r','a','k')


namespace AAT {


struct TrackTableEntry
{
  friend struct TrackData;

  float get_track_value () const { return track.to_float (); }

  int get_value (const void *base, unsigned int index,
		 unsigned int table_size) const
  { return (base+valuesZ).as_array (table_size)[index]; }

  public:
  bool sanitize (hb_sanitize_context_t *c, const void *base,
		 unsigned int table_size) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  (valuesZ.sanitize (c, base, table_size))));
  }

  protected:
  HBFixed	track;		/* Track value for this record. */
  NameID	trackNameID;	/* The 'name' table index for this track.
				 * (a short word or phrase like "loose"
				 * or "very tight") */
  NNOffsetTo<UnsizedArrayOf<FWORD>>
		valuesZ;	/* Offset from start of tracking table to
				 * per-size tracking values for this track. */

  public:
  DEFINE_SIZE_STATIC (8);
};

struct TrackData
{
  float interpolate_at (unsigned int idx,
			float target_size,
			const TrackTableEntry &trackTableEntry,
			const void *base) const
  {
    unsigned int sizes = nSizes;
    hb_array_t<const HBFixed> size_table ((base+sizeTable).arrayZ, sizes);

    float s0 = size_table[idx].to_float ();
    float s1 = size_table[idx + 1].to_float ();
    float t = unlikely (s0 == s1) ? 0.f : (target_size - s0) / (s1 - s0);
    return t * trackTableEntry.get_value (base, idx + 1, sizes) +
	   (1.f - t) * trackTableEntry.get_value (base, idx, sizes);
  }

  int get_tracking (const void *base, float ptem) const
  {
    /*
     * Choose track.
     */
    const TrackTableEntry *trackTableEntry = nullptr;
    unsigned int count = nTracks;
    for (unsigned int i = 0; i < count; i++)
    {
      /* Note: Seems like the track entries are sorted by values.  But the
       * spec doesn't explicitly say that.  It just mentions it in the example. */

      /* For now we only seek for track entries with zero tracking value */

      if (trackTable[i].get_track_value () == 0.f)
      {
	trackTableEntry = &trackTable[i];
	break;
      }
    }
    if (!trackTableEntry) return 0.;

    /*
     * Choose size.
     */
    unsigned int sizes = nSizes;
    if (!sizes) return 0.;
    if (sizes == 1) return trackTableEntry->get_value (base, 0, sizes);

    hb_array_t<const HBFixed> size_table ((base+sizeTable).arrayZ, sizes);
    unsigned int size_index;
    for (size_index = 0; size_index < sizes - 1; size_index++)
      if (size_table[size_index].to_float () >= ptem)
	break;

    return roundf (interpolate_at (size_index ? size_index - 1 : 0, ptem,
				   *trackTableEntry, base));
  }

  bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  sizeTable.sanitize (c, base, nSizes) &&
			  trackTable.sanitize (c, nTracks, base, nSizes)));
  }

  protected:
  HBUINT16	nTracks;	/* Number of separate tracks included in this table. */
  HBUINT16	nSizes;		/* Number of point sizes included in this table. */
  LNNOffsetTo<UnsizedArrayOf<HBFixed>>
		sizeTable;	/* Offset from start of the tracking table to
				 * Array[nSizes] of size values.. */
  UnsizedArrayOf<TrackTableEntry>
		trackTable;	/* Array[nTracks] of TrackTableEntry records. */

  public:
  DEFINE_SIZE_ARRAY (8, trackTable);
};

struct trak
{
  static constexpr hb_tag_t tableTag = HB_AAT_TAG_trak;

  bool has_data () const { return version.to_int (); }

  bool apply (hb_aat_apply_context_t *c) const
  {
    TRACE_APPLY (this);

    hb_mask_t trak_mask = c->plan->trak_mask;

    const float ptem = c->font->ptem;
    if (unlikely (ptem <= 0.f))
      return_trace (false);

    hb_buffer_t *buffer = c->buffer;
    if (HB_DIRECTION_IS_HORIZONTAL (buffer->props.direction))
    {
      const TrackData &trackData = this+horizData;
      int tracking = trackData.get_tracking (this, ptem);
      hb_position_t offset_to_add = c->font->em_scalef_x (tracking / 2);
      hb_position_t advance_to_add = c->font->em_scalef_x (tracking);
      foreach_grapheme (buffer, start, end)
      {
	if (!(buffer->info[start].mask & trak_mask)) continue;
	buffer->pos[start].x_advance += advance_to_add;
	buffer->pos[start].x_offset += offset_to_add;
      }
    }
    else
    {
      const TrackData &trackData = this+vertData;
      int tracking = trackData.get_tracking (this, ptem);
      hb_position_t offset_to_add = c->font->em_scalef_y (tracking / 2);
      hb_position_t advance_to_add = c->font->em_scalef_y (tracking);
      foreach_grapheme (buffer, start, end)
      {
	if (!(buffer->info[start].mask & trak_mask)) continue;
	buffer->pos[start].y_advance += advance_to_add;
	buffer->pos[start].y_offset += offset_to_add;
      }
    }

    return_trace (true);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);

    return_trace (likely (c->check_struct (this) &&
			  version.major == 1 &&
			  horizData.sanitize (c, this, this) &&
			  vertData.sanitize (c, this, this)));
  }

  protected:
  FixedVersion<>version;	/* Version of the tracking table
				 * (0x00010000u for version 1.0). */
  HBUINT16	format;		/* Format of the tracking table (set to 0). */
  OffsetTo<TrackData>
		horizData;	/* Offset from start of tracking table to TrackData
				 * for horizontal text (or 0 if none). */
  OffsetTo<TrackData>
		vertData;	/* Offset from start of tracking table to TrackData
				 * for vertical text (or 0 if none). */
  HBUINT16	reserved;	/* Reserved. Set to 0. */

  public:
  DEFINE_SIZE_STATIC (12);
};

} /* namespace AAT */


#endif /* HB_AAT_LAYOUT_TRAK_TABLE_HH */

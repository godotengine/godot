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

#include "hb-aat-layout-common-private.hh"
#include "hb-ot-layout-private.hh"
#include "hb-open-type-private.hh"

/*
 * trak -- Tracking
 * https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6trak.html
 */
#define HB_AAT_TAG_trak HB_TAG('t','r','a','k')


namespace AAT {


struct TrackTableEntry
{
  friend struct TrackData;

  inline bool sanitize (hb_sanitize_context_t *c, const void *base,
			unsigned int size) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  (valuesZ.sanitize (c, base, size))));
  }

  private:
  inline float get_track_value () const
  {
    return track.to_float ();
  }

  inline int get_value (const void *base, unsigned int index) const
  {
    return (base+valuesZ)[index];
  }

  protected:
  Fixed		track;		/* Track value for this record. */
  NameID	trackNameID;	/* The 'name' table index for this track */
  OffsetTo<UnsizedArrayOf<FWORD> >
		valuesZ;	/* Offset from start of tracking table to
				 * per-size tracking values for this track. */

  public:
  DEFINE_SIZE_STATIC (8);
};

struct TrackData
{
  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  sizeTable.sanitize (c, base, nSizes) &&
		  trackTable.sanitize (c, nTracks, base, nSizes));
  }

  inline float get_tracking (const void *base, float ptem) const
  {
    /* CoreText points are CSS pixels (96 per inch),
     * NOT typographic points (72 per inch).
     *
     * https://developer.apple.com/library/content/documentation/GraphicsAnimation/Conceptual/HighResolutionOSX/Explained/Explained.html
     */
    float csspx = ptem * 96.f / 72.f;
    Fixed fixed_size;
    fixed_size.set_float (csspx);

    /* XXX Clean this up. Make it work with nSizes==1 and 0. */

    unsigned int sizes = nSizes;

    const TrackTableEntry *trackTableEntry = nullptr;
    for (unsigned int i = 0; i < sizes; ++i)
      // For now we only seek for track entries with zero tracking value
      if (trackTable[i].get_track_value () == 0.f)
        trackTableEntry = &trackTable[0];

    // We couldn't match any, exit
    if (!trackTableEntry) return 0.;

    /* TODO bfind() */
    unsigned int size_index;
    UnsizedArrayOf<Fixed> size_table = base+sizeTable;
    for (size_index = 0; size_index < sizes; ++size_index)
      if (size_table[size_index] >= fixed_size)
        break;

    // TODO(ebraminio): We don't attempt to extrapolate to larger or
    // smaller values for now but we should do, per spec
    if (size_index == sizes)
      return trackTableEntry->get_value (base, sizes - 1);
    if (size_index == 0 || size_table[size_index] == fixed_size)
      return trackTableEntry->get_value (base, size_index);

    float s0 = size_table[size_index - 1].to_float ();
    float s1 = size_table[size_index].to_float ();
    float t = (csspx - s0) / (s1 - s0);
    return (float) t * trackTableEntry->get_value (base, size_index) +
	   ((float) 1.0 - t) * trackTableEntry->get_value (base, size_index - 1);
  }

  protected:
  HBUINT16	nTracks;	/* Number of separate tracks included in this table. */
  HBUINT16	nSizes;		/* Number of point sizes included in this table. */
  LOffsetTo<UnsizedArrayOf<Fixed> >
		sizeTable;	/* Offset to array[nSizes] of size values. */
  UnsizedArrayOf<TrackTableEntry>
		trackTable;	/* Array[nTracks] of TrackTableEntry records. */

  public:
  DEFINE_SIZE_ARRAY (8, trackTable);
};

struct trak
{
  static const hb_tag_t tableTag = HB_AAT_TAG_trak;

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);

    return_trace (unlikely (c->check_struct (this) &&
			    horizData.sanitize (c, this, this) &&
			    vertData.sanitize (c, this, this)));
  }

  inline bool apply (hb_aat_apply_context_t *c) const
  {
    TRACE_APPLY (this);

    const float ptem = c->font->ptem;
    if (unlikely (ptem <= 0.f))
      return_trace (false);

    hb_buffer_t *buffer = c->buffer;
    if (HB_DIRECTION_IS_HORIZONTAL (buffer->props.direction))
    {
      const TrackData &trackData = this+horizData;
      float tracking = trackData.get_tracking (this, ptem);
      hb_position_t advance_to_add = c->font->em_scalef_x (tracking / 2);
      foreach_grapheme (buffer, start, end)
      {
	buffer->pos[start].x_offset += advance_to_add;
	buffer->pos[start].x_advance += advance_to_add;
	buffer->pos[end].x_advance += advance_to_add;
      }
    }
    else
    {
      const TrackData &trackData = this+vertData;
      float tracking = trackData.get_tracking (this, ptem);
      hb_position_t advance_to_add = c->font->em_scalef_y (tracking / 2);
      foreach_grapheme (buffer, start, end)
      {
	buffer->pos[start].y_offset += advance_to_add;
	buffer->pos[start].y_advance += advance_to_add;
	buffer->pos[end].y_advance += advance_to_add;
      }
    }

    return_trace (true);
  }

  protected:
  FixedVersion<>	version;	/* Version of the tracking table--currently
					 * 0x00010000u for version 1.0. */
  HBUINT16		format; 	/* Format of the tracking table */
  OffsetTo<TrackData>	horizData;	/* TrackData for horizontal text */
  OffsetTo<TrackData>	vertData;	/* TrackData for vertical text */
  HBUINT16		reserved;	/* Reserved. Set to 0. */

  public:
  DEFINE_SIZE_MIN (12);
};

} /* namespace AAT */


#endif /* HB_AAT_LAYOUT_TRAK_TABLE_HH */

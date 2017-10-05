/*
 * Copyright © 2015  Google, Inc.
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

#ifndef HB_OT_GLYF_TABLE_HH
#define HB_OT_GLYF_TABLE_HH

#include "hb-open-type-private.hh"
#include "hb-ot-head-table.hh"
#include "hb-subset-glyf.hh"
#include "hb-subset-plan.hh"
#include "hb-subset-private.hh"

namespace OT {


/*
 * loca -- Index to Location
 * https://docs.microsoft.com/en-us/typography/opentype/spec/loca
 */
#define HB_OT_TAG_loca HB_TAG('l','o','c','a')


struct loca
{
  friend struct glyf;

  static const hb_tag_t tableTag = HB_OT_TAG_loca;

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (true);
  }

  protected:
  HBUINT8		dataZ[VAR];		/* Location data. */
  DEFINE_SIZE_ARRAY (0, dataZ);
};


/*
 * glyf -- TrueType Glyph Data
 * https://docs.microsoft.com/en-us/typography/opentype/spec/glyf
 */
#define HB_OT_TAG_glyf HB_TAG('g','l','y','f')


struct glyf
{
  static const hb_tag_t tableTag = HB_OT_TAG_glyf;

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    /* We don't check for anything specific here.  The users of the
     * struct do all the hard work... */
    return_trace (true);
  }

  inline bool subset (hb_subset_plan_t *plan) const
  {
    hb_blob_t *glyf_prime = nullptr;
    hb_blob_t *loca_prime = nullptr;

    bool success = true;
    bool use_short_loca = false;
    if (hb_subset_glyf_and_loca (plan, &use_short_loca, &glyf_prime, &loca_prime)) {
      success = success && plan->add_table (HB_OT_TAG_glyf, glyf_prime);
      success = success && plan->add_table (HB_OT_TAG_loca, loca_prime);
      success = success && _add_head_and_set_loca_version (plan, use_short_loca);
    } else {
      success = false;
    }
    hb_blob_destroy (loca_prime);
    hb_blob_destroy (glyf_prime);

    return success;
  }

  static bool
  _add_head_and_set_loca_version (hb_subset_plan_t *plan, bool use_short_loca)
  {
    hb_blob_t *head_blob = hb_sanitize_context_t().reference_table<head> (plan->source);
    hb_blob_t *head_prime_blob = hb_blob_copy_writable_or_fail (head_blob);
    hb_blob_destroy (head_blob);

    if (unlikely (!head_prime_blob))
      return false;

    head *head_prime = (head *) hb_blob_get_data_writable (head_prime_blob, nullptr);
    head_prime->indexToLocFormat.set (use_short_loca ? 0 : 1);
    bool success = plan->add_table (HB_OT_TAG_head, head_prime_blob);

    hb_blob_destroy (head_prime_blob);
    return success;
  }

  struct GlyphHeader
  {
    HBINT16		numberOfContours;	/* If the number of contours is
                                                 * greater than or equal to zero,
                                                 * this is a simple glyph; if negative,
                                                 * this is a composite glyph. */
    FWORD		xMin;			/* Minimum x for coordinate data. */
    FWORD		yMin;			/* Minimum y for coordinate data. */
    FWORD		xMax;			/* Maximum x for coordinate data. */
    FWORD		yMax;			/* Maximum y for coordinate data. */

    DEFINE_SIZE_STATIC (10);
  };

  struct CompositeGlyphHeader
  {
    enum composite_glyph_flag_t {
      ARG_1_AND_2_ARE_WORDS =      0x0001,
      ARGS_ARE_XY_VALUES =         0x0002,
      ROUND_XY_TO_GRID =           0x0004,
      WE_HAVE_A_SCALE =            0x0008,
      MORE_COMPONENTS =            0x0020,
      WE_HAVE_AN_X_AND_Y_SCALE =   0x0040,
      WE_HAVE_A_TWO_BY_TWO =       0x0080,
      WE_HAVE_INSTRUCTIONS =       0x0100,
      USE_MY_METRICS =             0x0200,
      OVERLAP_COMPOUND =           0x0400,
      SCALED_COMPONENT_OFFSET =    0x0800,
      UNSCALED_COMPONENT_OFFSET =  0x1000
    };

    HBUINT16 flags;
    HBUINT16 glyphIndex;

    inline unsigned int get_size (void) const
    {
      unsigned int size = min_size;
      if (flags & ARG_1_AND_2_ARE_WORDS) {
        // arg1 and 2 are int16
        size += 4;
      } else {
        // arg1 and 2 are int8
        size += 2;
      }
      if (flags & WE_HAVE_A_SCALE) {
        // One x 16 bit (scale)
        size += 2;
      } else if (flags & WE_HAVE_AN_X_AND_Y_SCALE) {
        // Two x 16 bit (xscale, yscale)
        size += 4;
      } else if (flags & WE_HAVE_A_TWO_BY_TWO) {
        // Four x 16 bit (xscale, scale01, scale10, yscale)
        size += 8;
      }
      return size;
    }

    struct Iterator
    {
      const char *glyph_start;
      const char *glyph_end;
      const CompositeGlyphHeader *current;

      inline bool move_to_next ()
      {
	if (current->flags & CompositeGlyphHeader::MORE_COMPONENTS)
	{
	  const CompositeGlyphHeader *possible =
	    &StructAfter<CompositeGlyphHeader, CompositeGlyphHeader> (*current);
	  if (!in_range (possible))
	    return false;
	  current = possible;
	  return true;
	}
	return false;
      }

      inline bool in_range (const CompositeGlyphHeader *composite) const
      {
	return (const char *) composite >= glyph_start
	  && ((const char *) composite + CompositeGlyphHeader::min_size) <= glyph_end
	  && ((const char *) composite + composite->get_size()) <= glyph_end;
      }
    };

    static inline bool get_iterator (const char * glyph_data,
				     unsigned int length,
				     CompositeGlyphHeader::Iterator *iterator /* OUT */)
    {
      if (length < GlyphHeader::static_size)
	return false; /* Empty glyph; zero extents. */

      const GlyphHeader &glyph_header = StructAtOffset<GlyphHeader> (glyph_data, 0);
      if (glyph_header.numberOfContours < 0)
      {
        const CompositeGlyphHeader *possible =
	  &StructAfter<CompositeGlyphHeader, GlyphHeader> (glyph_header);

	iterator->glyph_start = glyph_data;
	iterator->glyph_end = (const char *) glyph_data + length;
	if (!iterator->in_range (possible))
          return false;
        iterator->current = possible;
        return true;
      }

      return false;
    }

    DEFINE_SIZE_MIN (4);
  };

  struct accelerator_t
  {
    inline void init (hb_face_t *face)
    {
      memset (this, 0, sizeof (accelerator_t));

      hb_blob_t *head_blob = hb_sanitize_context_t().reference_table<head> (face);
      const head *head_table = head_blob->as<head> ();
      if (head_table->indexToLocFormat > 1 || head_table->glyphDataFormat != 0)
      {
	/* Unknown format.  Leave num_glyphs=0, that takes care of disabling us. */
	hb_blob_destroy (head_blob);
	return;
      }
      short_offset = 0 == head_table->indexToLocFormat;
      hb_blob_destroy (head_blob);

      loca_blob = hb_sanitize_context_t().reference_table<loca> (face);
      loca_table = loca_blob->as<loca> ();
      glyf_blob = hb_sanitize_context_t().reference_table<glyf> (face);
      glyf_table = glyf_blob->as<glyf> ();

      num_glyphs = MAX (1u, hb_blob_get_length (loca_blob) / (short_offset ? 2 : 4)) - 1;
      glyf_len = hb_blob_get_length (glyf_blob);
    }

    inline void fini (void)
    {
      hb_blob_destroy (loca_blob);
      hb_blob_destroy (glyf_blob);
    }

    /*
     * Returns true if the referenced glyph is a valid glyph and a composite glyph.
     * If true is returned a pointer to the composite glyph will be written into
     * composite.
     */
    inline bool get_composite (hb_codepoint_t glyph,
			       CompositeGlyphHeader::Iterator *composite /* OUT */) const
    {
      if (unlikely (!num_glyphs))
	return false;

      unsigned int start_offset, end_offset;
      if (!get_offsets (glyph, &start_offset, &end_offset))
        return false; /* glyph not found */

      return CompositeGlyphHeader::get_iterator ((const char*) this->glyf_table + start_offset,
						 end_offset - start_offset,
						 composite);
    }

    enum simple_glyph_flag_t {
      FLAG_X_SHORT = 0x02,
      FLAG_Y_SHORT = 0x04,
      FLAG_REPEAT = 0x08,
      FLAG_X_SAME = 0x10,
      FLAG_Y_SAME = 0x20
    };

    /* based on FontTools _g_l_y_f.py::trim */
    inline bool remove_padding(unsigned int start_offset,
                               unsigned int *end_offset) const
    {
      if (*end_offset - start_offset < GlyphHeader::static_size)
        return true;

      const char *glyph = ((const char *) glyf_table) + start_offset;
      const char * const glyph_end = glyph + (*end_offset - start_offset);
      const GlyphHeader &glyph_header = StructAtOffset<GlyphHeader> (glyph, 0);
      int16_t num_contours = (int16_t) glyph_header.numberOfContours;

      if (num_contours < 0)
        /* Trimming for composites not implemented.
         * If removing hints it falls out of that. */
        return true;
      else if (num_contours > 0)
      {
        /* simple glyph w/contours, possibly trimmable */
        glyph += GlyphHeader::static_size + 2 * num_contours;

        if (unlikely (glyph + 2 >= glyph_end)) return false;
        uint16_t nCoordinates = (uint16_t) StructAtOffset<HBUINT16>(glyph - 2, 0) + 1;
        uint16_t nInstructions = (uint16_t) StructAtOffset<HBUINT16>(glyph, 0);

        glyph += 2 + nInstructions;
        if (unlikely (glyph + 2 >= glyph_end)) return false;

        unsigned int coordBytes = 0;
        unsigned int coordsWithFlags = 0;
        while (glyph < glyph_end)
        {
          uint8_t flag = (uint8_t) *glyph;
          glyph++;

          unsigned int repeat = 1;
          if (flag & FLAG_REPEAT)
          {
            if (glyph >= glyph_end)
            {
              DEBUG_MSG(SUBSET, nullptr, "Bad flag");
              return false;
            }
            repeat = ((uint8_t) *glyph) + 1;
            glyph++;
          }

          unsigned int xBytes, yBytes;
          xBytes = yBytes = 0;
          if (flag & FLAG_X_SHORT)
            xBytes = 1;
          else if ((flag & FLAG_X_SAME) == 0)
            xBytes = 2;

          if (flag & FLAG_Y_SHORT)
            yBytes = 1;
          else if ((flag & FLAG_Y_SAME) == 0)
            yBytes = 2;

          coordBytes += (xBytes + yBytes) * repeat;
          coordsWithFlags += repeat;
          if (coordsWithFlags >= nCoordinates)
            break;
        }

        if (coordsWithFlags != nCoordinates)
        {
          DEBUG_MSG(SUBSET, nullptr, "Expect %d coords to have flags, got flags for %d", nCoordinates, coordsWithFlags);
          return false;
        }
        glyph += coordBytes;

        if (glyph < glyph_end)
          *end_offset -= glyph_end - glyph;
      }
      return true;
    }

    inline bool get_offsets (hb_codepoint_t  glyph,
                             unsigned int   *start_offset /* OUT */,
                             unsigned int   *end_offset   /* OUT */) const
    {
      if (unlikely (glyph >= num_glyphs))
	return false;

      if (short_offset)
      {
        const HBUINT16 *offsets = (const HBUINT16 *) loca_table->dataZ;
	*start_offset = 2 * offsets[glyph];
	*end_offset   = 2 * offsets[glyph + 1];
      }
      else
      {
        const HBUINT32 *offsets = (const HBUINT32 *) loca_table->dataZ;

	*start_offset = offsets[glyph];
	*end_offset   = offsets[glyph + 1];
      }

      if (*start_offset > *end_offset || *end_offset > glyf_len)
	return false;

      return true;
    }

    inline bool get_instruction_offsets(unsigned int start_offset,
                                        unsigned int end_offset,
                                        unsigned int *instruction_start /* OUT */,
                                        unsigned int *instruction_end /* OUT */) const
    {
      if (end_offset - start_offset < GlyphHeader::static_size)
      {
        *instruction_start = 0;
        *instruction_end = 0;
        return true; /* Empty glyph; no instructions. */
      }
      const GlyphHeader &glyph_header = StructAtOffset<GlyphHeader> (glyf_table, start_offset);
      int16_t num_contours = (int16_t) glyph_header.numberOfContours;
      if (num_contours < 0)
      {
        CompositeGlyphHeader::Iterator composite_it;
        if (unlikely (!CompositeGlyphHeader::get_iterator (
            (const char*) this->glyf_table + start_offset,
             end_offset - start_offset, &composite_it))) return false;
        const CompositeGlyphHeader *last;
        do {
          last = composite_it.current;
        } while (composite_it.move_to_next());

        if ( (uint16_t) last->flags & CompositeGlyphHeader::WE_HAVE_INSTRUCTIONS)
          *instruction_start = ((char *) last - (char *) glyf_table->dataZ) + last->get_size();
        else
          *instruction_start = end_offset;
        *instruction_end = end_offset;
        if (unlikely (*instruction_start > *instruction_end))
        {
          DEBUG_MSG(SUBSET, nullptr, "Invalid instruction offset, %d is outside [%d, %d]", *instruction_start, start_offset, end_offset);
          return false;
        }
      }
      else
      {
        unsigned int instruction_length_offset = start_offset + GlyphHeader::static_size + 2 * num_contours;
	if (unlikely (instruction_length_offset + 2 > end_offset))
	{
	  DEBUG_MSG(SUBSET, nullptr, "Glyph size is too short, missing field instructionLength.");
	  return false;
	}

        const HBUINT16 &instruction_length = StructAtOffset<HBUINT16> (glyf_table, instruction_length_offset);
	unsigned int start = instruction_length_offset + 2;
	unsigned int end = start + (uint16_t) instruction_length;
	if (unlikely (end > end_offset)) // Out of bounds of the current glyph
	{
	  DEBUG_MSG(SUBSET, nullptr, "The instructions array overruns the glyph's boundaries.");
	  return false;
	}

	*instruction_start = start;
        *instruction_end = end;
      }
      return true;
    }

    inline bool get_extents (hb_codepoint_t glyph,
			     hb_glyph_extents_t *extents) const
    {
      unsigned int start_offset, end_offset;
      if (!get_offsets (glyph, &start_offset, &end_offset))
        return false;

      if (end_offset - start_offset < GlyphHeader::static_size)
	return true; /* Empty glyph; zero extents. */

      const GlyphHeader &glyph_header = StructAtOffset<GlyphHeader> (glyf_table, start_offset);

      extents->x_bearing = MIN (glyph_header.xMin, glyph_header.xMax);
      extents->y_bearing = MAX (glyph_header.yMin, glyph_header.yMax);
      extents->width     = MAX (glyph_header.xMin, glyph_header.xMax) - extents->x_bearing;
      extents->height    = MIN (glyph_header.yMin, glyph_header.yMax) - extents->y_bearing;

      return true;
    }

    private:
    bool short_offset;
    unsigned int num_glyphs;
    const loca *loca_table;
    const glyf *glyf_table;
    hb_blob_t *loca_blob;
    hb_blob_t *glyf_blob;
    unsigned int glyf_len;
  };

  protected:
  HBUINT8		dataZ[VAR];		/* Glyphs data. */

  DEFINE_SIZE_ARRAY (0, dataZ);
};

} /* namespace OT */


#endif /* HB_OT_GLYF_TABLE_HH */

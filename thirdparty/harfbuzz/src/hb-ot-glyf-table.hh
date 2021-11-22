/*
 * Copyright © 2015  Google, Inc.
 * Copyright © 2019  Adobe Inc.
 * Copyright © 2019  Ebrahim Byagowi
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
 * Google Author(s): Behdad Esfahbod, Garret Rieger, Roderick Sheeter
 * Adobe Author(s): Michiharu Ariza
 */

#ifndef HB_OT_GLYF_TABLE_HH
#define HB_OT_GLYF_TABLE_HH

#include "hb-open-type.hh"
#include "hb-ot-head-table.hh"
#include "hb-ot-hmtx-table.hh"
#include "hb-ot-var-gvar-table.hh"
#include "hb-draw.hh"

namespace OT {


/*
 * loca -- Index to Location
 * https://docs.microsoft.com/en-us/typography/opentype/spec/loca
 */
#define HB_OT_TAG_loca HB_TAG('l','o','c','a')

#ifndef HB_MAX_COMPOSITE_OPERATIONS
#define HB_MAX_COMPOSITE_OPERATIONS 100000
#endif


struct loca
{
  friend struct glyf;

  static constexpr hb_tag_t tableTag = HB_OT_TAG_loca;

  bool sanitize (hb_sanitize_context_t *c HB_UNUSED) const
  {
    TRACE_SANITIZE (this);
    return_trace (true);
  }

  protected:
  UnsizedArrayOf<HBUINT8>
		dataZ;	/* Location data. */
  public:
  DEFINE_SIZE_MIN (0);	/* In reality, this is UNBOUNDED() type; but since we always
			 * check the size externally, allow Null() object of it by
			 * defining it _MIN instead. */
};


/*
 * glyf -- TrueType Glyph Data
 * https://docs.microsoft.com/en-us/typography/opentype/spec/glyf
 */
#define HB_OT_TAG_glyf HB_TAG('g','l','y','f')


struct glyf
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_glyf;

  bool sanitize (hb_sanitize_context_t *c HB_UNUSED) const
  {
    TRACE_SANITIZE (this);
    /* Runtime checks as eager sanitizing each glyph is costy */
    return_trace (true);
  }

  template<typename Iterator,
	   hb_requires (hb_is_source_of (Iterator, unsigned int))>
  static bool
  _add_loca_and_head (hb_subset_plan_t * plan, Iterator padded_offsets)
  {
    unsigned max_offset =
    + padded_offsets
    | hb_reduce (hb_add, 0)
    ;
    unsigned num_offsets = padded_offsets.len () + 1;
    bool use_short_loca = max_offset < 0x1FFFF;
    unsigned entry_size = use_short_loca ? 2 : 4;
    char *loca_prime_data = (char *) hb_calloc (entry_size, num_offsets);

    if (unlikely (!loca_prime_data)) return false;

    DEBUG_MSG (SUBSET, nullptr, "loca entry_size %d num_offsets %d "
				"max_offset %d size %d",
	       entry_size, num_offsets, max_offset, entry_size * num_offsets);

    if (use_short_loca)
      _write_loca (padded_offsets, 1, hb_array ((HBUINT16 *) loca_prime_data, num_offsets));
    else
      _write_loca (padded_offsets, 0, hb_array ((HBUINT32 *) loca_prime_data, num_offsets));

    hb_blob_t *loca_blob = hb_blob_create (loca_prime_data,
					   entry_size * num_offsets,
					   HB_MEMORY_MODE_WRITABLE,
					   loca_prime_data,
					   hb_free);

    bool result = plan->add_table (HB_OT_TAG_loca, loca_blob)
	       && _add_head_and_set_loca_version (plan, use_short_loca);

    hb_blob_destroy (loca_blob);
    return result;
  }

  template<typename IteratorIn, typename IteratorOut,
	   hb_requires (hb_is_source_of (IteratorIn, unsigned int)),
	   hb_requires (hb_is_sink_of (IteratorOut, unsigned))>
  static void
  _write_loca (IteratorIn it, unsigned right_shift, IteratorOut dest)
  {
    unsigned int offset = 0;
    dest << 0;
    + it
    | hb_map ([=, &offset] (unsigned int padded_size)
	      {
		offset += padded_size;
		DEBUG_MSG (SUBSET, nullptr, "loca entry offset %d", offset);
		return offset >> right_shift;
	      })
    | hb_sink (dest)
    ;
  }

  /* requires source of SubsetGlyph complains the identifier isn't declared */
  template <typename Iterator>
  bool serialize (hb_serialize_context_t *c,
		  Iterator it,
		  const hb_subset_plan_t *plan)
  {
    TRACE_SERIALIZE (this);
    unsigned init_len = c->length ();
    for (const auto &_ : it) _.serialize (c, plan);

    /* As a special case when all glyph in the font are empty, add a zero byte
     * to the table, so that OTS doesn’t reject it, and to make the table work
     * on Windows as well.
     * See https://github.com/khaledhosny/ots/issues/52 */
    if (init_len == c->length ())
    {
      HBUINT8 empty_byte;
      empty_byte = 0;
      c->copy (empty_byte);
    }
    return_trace (true);
  }

  /* Byte region(s) per glyph to output
     unpadded, hints removed if so requested
     If we fail to process a glyph we produce an empty (0-length) glyph */
  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);

    glyf *glyf_prime = c->serializer->start_embed <glyf> ();
    if (unlikely (!c->serializer->check_success (glyf_prime))) return_trace (false);

    hb_vector_t<SubsetGlyph> glyphs;
    _populate_subset_glyphs (c->plan, &glyphs);

    glyf_prime->serialize (c->serializer, hb_iter (glyphs), c->plan);

    auto padded_offsets =
    + hb_iter (glyphs)
    | hb_map (&SubsetGlyph::padded_size)
    ;

    if (unlikely (c->serializer->in_error ())) return_trace (false);
    return_trace (c->serializer->check_success (_add_loca_and_head (c->plan,
								    padded_offsets)));
  }

  template <typename SubsetGlyph>
  void
  _populate_subset_glyphs (const hb_subset_plan_t   *plan,
			   hb_vector_t<SubsetGlyph> *glyphs /* OUT */) const
  {
    OT::glyf::accelerator_t glyf;
    glyf.init (plan->source);

    + hb_range (plan->num_output_glyphs ())
    | hb_map ([&] (hb_codepoint_t new_gid)
	      {
		SubsetGlyph subset_glyph = {0};
		subset_glyph.new_gid = new_gid;

		/* should never fail: all old gids should be mapped */
		if (!plan->old_gid_for_new_gid (new_gid, &subset_glyph.old_gid))
		  return subset_glyph;

		if (new_gid == 0 &&
                    !(plan->flags & HB_SUBSET_FLAGS_NOTDEF_OUTLINE))
		  subset_glyph.source_glyph = Glyph ();
		else
		  subset_glyph.source_glyph = glyf.glyph_for_gid (subset_glyph.old_gid, true);
		if (plan->flags & HB_SUBSET_FLAGS_NO_HINTING)
                  subset_glyph.drop_hints_bytes ();
		else
                  subset_glyph.dest_start = subset_glyph.source_glyph.get_bytes ();
		return subset_glyph;
	      })
    | hb_sink (glyphs)
    ;

    glyf.fini ();
  }

  static bool
  _add_head_and_set_loca_version (hb_subset_plan_t *plan, bool use_short_loca)
  {
    hb_blob_t *head_blob = hb_sanitize_context_t ().reference_table<head> (plan->source);
    hb_blob_t *head_prime_blob = hb_blob_copy_writable_or_fail (head_blob);
    hb_blob_destroy (head_blob);

    if (unlikely (!head_prime_blob))
      return false;

    head *head_prime = (head *) hb_blob_get_data_writable (head_prime_blob, nullptr);
    head_prime->indexToLocFormat = use_short_loca ? 0 : 1;
    bool success = plan->add_table (HB_OT_TAG_head, head_prime_blob);

    hb_blob_destroy (head_prime_blob);
    return success;
  }

  struct CompositeGlyphChain
  {
    protected:
    enum composite_glyph_flag_t
    {
      ARG_1_AND_2_ARE_WORDS	= 0x0001,
      ARGS_ARE_XY_VALUES	= 0x0002,
      ROUND_XY_TO_GRID		= 0x0004,
      WE_HAVE_A_SCALE		= 0x0008,
      MORE_COMPONENTS		= 0x0020,
      WE_HAVE_AN_X_AND_Y_SCALE	= 0x0040,
      WE_HAVE_A_TWO_BY_TWO	= 0x0080,
      WE_HAVE_INSTRUCTIONS	= 0x0100,
      USE_MY_METRICS		= 0x0200,
      OVERLAP_COMPOUND		= 0x0400,
      SCALED_COMPONENT_OFFSET	= 0x0800,
      UNSCALED_COMPONENT_OFFSET = 0x1000
    };

    public:
    unsigned int get_size () const
    {
      unsigned int size = min_size;
      /* arg1 and 2 are int16 */
      if (flags & ARG_1_AND_2_ARE_WORDS) size += 4;
      /* arg1 and 2 are int8 */
      else size += 2;

      /* One x 16 bit (scale) */
      if (flags & WE_HAVE_A_SCALE) size += 2;
      /* Two x 16 bit (xscale, yscale) */
      else if (flags & WE_HAVE_AN_X_AND_Y_SCALE) size += 4;
      /* Four x 16 bit (xscale, scale01, scale10, yscale) */
      else if (flags & WE_HAVE_A_TWO_BY_TWO) size += 8;

      return size;
    }

    void set_glyph_index (hb_codepoint_t new_gid) { glyphIndex = new_gid; }
    hb_codepoint_t get_glyph_index ()       const { return glyphIndex; }

    void drop_instructions_flag ()  { flags = (uint16_t) flags & ~WE_HAVE_INSTRUCTIONS; }
    void set_overlaps_flag ()
    {
      flags = (uint16_t) flags | OVERLAP_COMPOUND;
    }

    bool has_instructions ()  const { return   flags & WE_HAVE_INSTRUCTIONS; }

    bool has_more ()          const { return   flags & MORE_COMPONENTS; }
    bool is_use_my_metrics () const { return   flags & USE_MY_METRICS; }
    bool is_anchored ()       const { return !(flags & ARGS_ARE_XY_VALUES); }
    void get_anchor_points (unsigned int &point1, unsigned int &point2) const
    {
      const HBUINT8 *p = &StructAfter<const HBUINT8> (glyphIndex);
      if (flags & ARG_1_AND_2_ARE_WORDS)
      {
	point1 = ((const HBUINT16 *) p)[0];
	point2 = ((const HBUINT16 *) p)[1];
      }
      else
      {
	point1 = p[0];
	point2 = p[1];
      }
    }

    void transform_points (contour_point_vector_t &points) const
    {
      float matrix[4];
      contour_point_t trans;
      if (get_transformation (matrix, trans))
      {
	if (scaled_offsets ())
	{
	  points.translate (trans);
	  points.transform (matrix);
	}
	else
	{
	  points.transform (matrix);
	  points.translate (trans);
	}
      }
    }

    protected:
    bool scaled_offsets () const
    { return (flags & (SCALED_COMPONENT_OFFSET | UNSCALED_COMPONENT_OFFSET)) == SCALED_COMPONENT_OFFSET; }

    bool get_transformation (float (&matrix)[4], contour_point_t &trans) const
    {
      matrix[0] = matrix[3] = 1.f;
      matrix[1] = matrix[2] = 0.f;

      int tx, ty;
      const HBINT8 *p = &StructAfter<const HBINT8> (glyphIndex);
      if (flags & ARG_1_AND_2_ARE_WORDS)
      {
	tx = *(const HBINT16 *) p;
	p += HBINT16::static_size;
	ty = *(const HBINT16 *) p;
	p += HBINT16::static_size;
      }
      else
      {
	tx = *p++;
	ty = *p++;
      }
      if (is_anchored ()) tx = ty = 0;

      trans.init ((float) tx, (float) ty);

      {
	const F2DOT14 *points = (const F2DOT14 *) p;
	if (flags & WE_HAVE_A_SCALE)
	{
	  matrix[0] = matrix[3] = points[0].to_float ();
	  return true;
	}
	else if (flags & WE_HAVE_AN_X_AND_Y_SCALE)
	{
	  matrix[0] = points[0].to_float ();
	  matrix[3] = points[1].to_float ();
	  return true;
	}
	else if (flags & WE_HAVE_A_TWO_BY_TWO)
	{
	  matrix[0] = points[0].to_float ();
	  matrix[1] = points[1].to_float ();
	  matrix[2] = points[2].to_float ();
	  matrix[3] = points[3].to_float ();
	  return true;
	}
      }
      return tx || ty;
    }

    protected:
    HBUINT16	flags;
    HBGlyphID16	glyphIndex;
    public:
    DEFINE_SIZE_MIN (4);
  };

  struct composite_iter_t : hb_iter_with_fallback_t<composite_iter_t, const CompositeGlyphChain &>
  {
    typedef const CompositeGlyphChain *__item_t__;
    composite_iter_t (hb_bytes_t glyph_, __item_t__ current_) :
        glyph (glyph_), current (nullptr), current_size (0)
    {
      set_next (current_);
    }

    composite_iter_t () : glyph (hb_bytes_t ()), current (nullptr), current_size (0) {}

    const CompositeGlyphChain &__item__ () const { return *current; }
    bool __more__ () const { return current; }
    void __next__ ()
    {
      if (!current->has_more ()) { current = nullptr; return; }

      set_next (&StructAtOffset<CompositeGlyphChain> (current, current_size));
    }
    bool operator != (const composite_iter_t& o) const
    { return glyph != o.glyph || current != o.current; }


    void set_next (const CompositeGlyphChain *composite)
    {
      if (!glyph.check_range (composite, CompositeGlyphChain::min_size))
      {
        current = nullptr;
        current_size = 0;
        return;
      }
      unsigned size = composite->get_size ();
      if (!glyph.check_range (composite, size))
      {
        current = nullptr;
        current_size = 0;
        return;
      }

      current = composite;
      current_size = size;
    }

    private:
    hb_bytes_t glyph;
    __item_t__ current;
    unsigned current_size;
  };

  enum phantom_point_index_t
  {
    PHANTOM_LEFT   = 0,
    PHANTOM_RIGHT  = 1,
    PHANTOM_TOP    = 2,
    PHANTOM_BOTTOM = 3,
    PHANTOM_COUNT  = 4
  };

  struct accelerator_t;

  struct Glyph
  {
    enum simple_glyph_flag_t
    {
      FLAG_ON_CURVE       = 0x01,
      FLAG_X_SHORT        = 0x02,
      FLAG_Y_SHORT        = 0x04,
      FLAG_REPEAT         = 0x08,
      FLAG_X_SAME         = 0x10,
      FLAG_Y_SAME         = 0x20,
      FLAG_OVERLAP_SIMPLE = 0x40,
      FLAG_RESERVED2      = 0x80
    };

    private:
    struct GlyphHeader
    {
      bool has_data () const { return numberOfContours; }

      bool get_extents (hb_font_t *font, const accelerator_t &glyf_accelerator,
		        hb_codepoint_t gid, hb_glyph_extents_t *extents) const
      {
	/* Undocumented rasterizer behavior: shift glyph to the left by (lsb - xMin), i.e., xMin = lsb */
	/* extents->x_bearing = hb_min (glyph_header.xMin, glyph_header.xMax); */
	extents->x_bearing = font->em_scale_x (glyf_accelerator.hmtx->get_side_bearing (gid));
	extents->y_bearing = font->em_scale_y (hb_max (yMin, yMax));
	extents->width     = font->em_scale_x (hb_max (xMin, xMax) - hb_min (xMin, xMax));
	extents->height    = font->em_scale_y (hb_min (yMin, yMax) - hb_max (yMin, yMax));

	return true;
      }

      HBINT16	numberOfContours;
			/* If the number of contours is
			 * greater than or equal to zero,
			 * this is a simple glyph; if negative,
			 * this is a composite glyph. */
      FWORD	xMin;	/* Minimum x for coordinate data. */
      FWORD	yMin;	/* Minimum y for coordinate data. */
      FWORD	xMax;	/* Maximum x for coordinate data. */
      FWORD	yMax;	/* Maximum y for coordinate data. */
      public:
      DEFINE_SIZE_STATIC (10);
    };

    struct SimpleGlyph
    {
      const GlyphHeader &header;
      hb_bytes_t bytes;
      SimpleGlyph (const GlyphHeader &header_, hb_bytes_t bytes_) :
	header (header_), bytes (bytes_) {}

      unsigned int instruction_len_offset () const
      { return GlyphHeader::static_size + 2 * header.numberOfContours; }

      unsigned int length (unsigned int instruction_len) const
      { return instruction_len_offset () + 2 + instruction_len; }

      unsigned int instructions_length () const
      {
	unsigned int instruction_length_offset = instruction_len_offset ();
	if (unlikely (instruction_length_offset + 2 > bytes.length)) return 0;

	const HBUINT16 &instructionLength = StructAtOffset<HBUINT16> (&bytes, instruction_length_offset);
	/* Out of bounds of the current glyph */
	if (unlikely (length (instructionLength) > bytes.length)) return 0;
	return instructionLength;
      }

      const Glyph trim_padding () const
      {
	/* based on FontTools _g_l_y_f.py::trim */
	const uint8_t *glyph = (uint8_t*) bytes.arrayZ;
	const uint8_t *glyph_end = glyph + bytes.length;
	/* simple glyph w/contours, possibly trimmable */
	glyph += instruction_len_offset ();

	if (unlikely (glyph + 2 >= glyph_end)) return Glyph ();
	unsigned int num_coordinates = StructAtOffset<HBUINT16> (glyph - 2, 0) + 1;
	unsigned int num_instructions = StructAtOffset<HBUINT16> (glyph, 0);

	glyph += 2 + num_instructions;

	unsigned int coord_bytes = 0;
	unsigned int coords_with_flags = 0;
	while (glyph < glyph_end)
	{
	  uint8_t flag = *glyph;
	  glyph++;

	  unsigned int repeat = 1;
	  if (flag & FLAG_REPEAT)
	  {
	    if (unlikely (glyph >= glyph_end)) return Glyph ();
	    repeat = *glyph + 1;
	    glyph++;
	  }

	  unsigned int xBytes, yBytes;
	  xBytes = yBytes = 0;
	  if (flag & FLAG_X_SHORT) xBytes = 1;
	  else if ((flag & FLAG_X_SAME) == 0) xBytes = 2;

	  if (flag & FLAG_Y_SHORT) yBytes = 1;
	  else if ((flag & FLAG_Y_SAME) == 0) yBytes = 2;

	  coord_bytes += (xBytes + yBytes) * repeat;
	  coords_with_flags += repeat;
	  if (coords_with_flags >= num_coordinates) break;
	}

	if (unlikely (coords_with_flags != num_coordinates)) return Glyph ();
	return Glyph (bytes.sub_array (0, bytes.length + coord_bytes - (glyph_end - glyph)));
      }

      /* zero instruction length */
      void drop_hints ()
      {
	GlyphHeader &glyph_header = const_cast<GlyphHeader &> (header);
	(HBUINT16 &) StructAtOffset<HBUINT16> (&glyph_header, instruction_len_offset ()) = 0;
      }

      void drop_hints_bytes (hb_bytes_t &dest_start, hb_bytes_t &dest_end) const
      {
	unsigned int instructions_len = instructions_length ();
	unsigned int glyph_length = length (instructions_len);
	dest_start = bytes.sub_array (0, glyph_length - instructions_len);
	dest_end = bytes.sub_array (glyph_length, bytes.length - glyph_length);
      }

      void set_overlaps_flag ()
      {
        if (unlikely (!header.numberOfContours)) return;

        unsigned flags_offset = length (instructions_length ());
        if (unlikely (length (flags_offset + 1) > bytes.length)) return;

	HBUINT8 &first_flag = (HBUINT8 &) StructAtOffset<HBUINT16> (&bytes, flags_offset);
        first_flag = (uint8_t) first_flag | FLAG_OVERLAP_SIMPLE;
      }

      static bool read_points (const HBUINT8 *&p /* IN/OUT */,
			       contour_point_vector_t &points_ /* IN/OUT */,
			       const hb_bytes_t &bytes,
			       void (* setter) (contour_point_t &_, float v),
			       const simple_glyph_flag_t short_flag,
			       const simple_glyph_flag_t same_flag)
      {
	float v = 0;
	for (unsigned i = 0; i < points_.length; i++)
	{
	  uint8_t flag = points_[i].flag;
	  if (flag & short_flag)
	  {
	    if (unlikely (!bytes.check_range (p))) return false;
	    if (flag & same_flag)
	      v += *p++;
	    else
	      v -= *p++;
	  }
	  else
	  {
	    if (!(flag & same_flag))
	    {
	      if (unlikely (!bytes.check_range ((const HBUINT16 *) p))) return false;
	      v += *(const HBINT16 *) p;
	      p += HBINT16::static_size;
	    }
	  }
	  setter (points_[i], v);
	}
	return true;
      }

      bool get_contour_points (contour_point_vector_t &points_ /* OUT */,
			       bool phantom_only = false) const
      {
	const HBUINT16 *endPtsOfContours = &StructAfter<HBUINT16> (header);
	int num_contours = header.numberOfContours;
	if (unlikely (!bytes.check_range (&endPtsOfContours[num_contours + 1]))) return false;
	unsigned int num_points = endPtsOfContours[num_contours - 1] + 1;

	points_.resize (num_points);
	for (unsigned int i = 0; i < points_.length; i++) points_[i].init ();
	if (phantom_only) return true;

	for (int i = 0; i < num_contours; i++)
	  points_[endPtsOfContours[i]].is_end_point = true;

	/* Skip instructions */
	const HBUINT8 *p = &StructAtOffset<HBUINT8> (&endPtsOfContours[num_contours + 1],
						     endPtsOfContours[num_contours]);

	/* Read flags */
	for (unsigned int i = 0; i < num_points; i++)
	{
	  if (unlikely (!bytes.check_range (p))) return false;
	  uint8_t flag = *p++;
	  points_[i].flag = flag;
	  if (flag & FLAG_REPEAT)
	  {
	    if (unlikely (!bytes.check_range (p))) return false;
	    unsigned int repeat_count = *p++;
	    while ((repeat_count-- > 0) && (++i < num_points))
	      points_[i].flag = flag;
	  }
	}

	/* Read x & y coordinates */
	return read_points (p, points_, bytes, [] (contour_point_t &p, float v) { p.x = v; },
			    FLAG_X_SHORT, FLAG_X_SAME)
	    && read_points (p, points_, bytes, [] (contour_point_t &p, float v) { p.y = v; },
			    FLAG_Y_SHORT, FLAG_Y_SAME);
      }
    };

    struct CompositeGlyph
    {
      const GlyphHeader &header;
      hb_bytes_t bytes;
      CompositeGlyph (const GlyphHeader &header_, hb_bytes_t bytes_) :
	header (header_), bytes (bytes_) {}

      composite_iter_t get_iterator () const
      { return composite_iter_t (bytes, &StructAfter<CompositeGlyphChain, GlyphHeader> (header)); }

      unsigned int instructions_length (hb_bytes_t bytes) const
      {
	unsigned int start = bytes.length;
	unsigned int end = bytes.length;
	const CompositeGlyphChain *last = nullptr;
	for (auto &item : get_iterator ())
	  last = &item;
	if (unlikely (!last)) return 0;

	if (last->has_instructions ())
	  start = (char *) last - &bytes + last->get_size ();
	if (unlikely (start > end)) return 0;
	return end - start;
      }

      /* Trimming for composites not implemented.
       * If removing hints it falls out of that. */
      const Glyph trim_padding () const { return Glyph (bytes); }

      void drop_hints ()
      {
	for (const auto &_ : get_iterator ())
	  const_cast<CompositeGlyphChain &> (_).drop_instructions_flag ();
      }

      /* Chop instructions off the end */
      void drop_hints_bytes (hb_bytes_t &dest_start) const
      { dest_start = bytes.sub_array (0, bytes.length - instructions_length (bytes)); }

      void set_overlaps_flag ()
      {
        const_cast<CompositeGlyphChain &> (StructAfter<CompositeGlyphChain, GlyphHeader> (header))
                .set_overlaps_flag ();
      }
    };

    enum glyph_type_t { EMPTY, SIMPLE, COMPOSITE };

    public:
    composite_iter_t get_composite_iterator () const
    {
      if (type != COMPOSITE) return composite_iter_t ();
      return CompositeGlyph (*header, bytes).get_iterator ();
    }

    const Glyph trim_padding () const
    {
      switch (type) {
      case COMPOSITE: return CompositeGlyph (*header, bytes).trim_padding ();
      case SIMPLE:    return SimpleGlyph (*header, bytes).trim_padding ();
      default:        return bytes;
      }
    }

    void drop_hints ()
    {
      switch (type) {
      case COMPOSITE: CompositeGlyph (*header, bytes).drop_hints (); return;
      case SIMPLE:    SimpleGlyph (*header, bytes).drop_hints (); return;
      default:        return;
      }
    }

    void set_overlaps_flag ()
    {
      switch (type) {
      case COMPOSITE: CompositeGlyph (*header, bytes).set_overlaps_flag (); return;
      case SIMPLE:    SimpleGlyph (*header, bytes).set_overlaps_flag (); return;
      default:        return;
      }
    }

    void drop_hints_bytes (hb_bytes_t &dest_start, hb_bytes_t &dest_end) const
    {
      switch (type) {
      case COMPOSITE: CompositeGlyph (*header, bytes).drop_hints_bytes (dest_start); return;
      case SIMPLE:    SimpleGlyph (*header, bytes).drop_hints_bytes (dest_start, dest_end); return;
      default:        return;
      }
    }

    /* Note: Recursively calls itself.
     * all_points includes phantom points
     */
    bool get_points (hb_font_t *font, const accelerator_t &glyf_accelerator,
		     contour_point_vector_t &all_points /* OUT */,
		     bool phantom_only = false,
		     unsigned int depth = 0) const
    {
      if (unlikely (depth > HB_MAX_NESTING_LEVEL)) return false;
      contour_point_vector_t points;

      switch (type) {
      case COMPOSITE:
      {
	/* pseudo component points for each component in composite glyph */
	unsigned num_points = hb_len (CompositeGlyph (*header, bytes).get_iterator ());
	if (unlikely (!points.resize (num_points))) return false;
	for (unsigned i = 0; i < points.length; i++)
	  points[i].init ();
	break;
      }
      case SIMPLE:
	if (unlikely (!SimpleGlyph (*header, bytes).get_contour_points (points, phantom_only)))
	  return false;
	break;
      }

      /* Init phantom points */
      if (unlikely (!points.resize (points.length + PHANTOM_COUNT))) return false;
      hb_array_t<contour_point_t> phantoms = points.sub_array (points.length - PHANTOM_COUNT, PHANTOM_COUNT);
      {
	for (unsigned i = 0; i < PHANTOM_COUNT; ++i) phantoms[i].init ();
	int h_delta = (int) header->xMin - glyf_accelerator.hmtx->get_side_bearing (gid);
	int v_orig  = (int) header->yMax + glyf_accelerator.vmtx->get_side_bearing (gid);
	unsigned h_adv = glyf_accelerator.hmtx->get_advance (gid);
	unsigned v_adv = glyf_accelerator.vmtx->get_advance (gid);
	phantoms[PHANTOM_LEFT].x = h_delta;
	phantoms[PHANTOM_RIGHT].x = h_adv + h_delta;
	phantoms[PHANTOM_TOP].y = v_orig;
	phantoms[PHANTOM_BOTTOM].y = v_orig - (int) v_adv;
      }

#ifndef HB_NO_VAR
      if (unlikely (!glyf_accelerator.gvar->apply_deltas_to_points (gid, font, points.as_array ())))
	return false;
#endif

      switch (type) {
      case SIMPLE:
	all_points.extend (points.as_array ());
	break;
      case COMPOSITE:
      {
	unsigned int comp_index = 0;
	for (auto &item : get_composite_iterator ())
	{
	  contour_point_vector_t comp_points;
	  if (unlikely (!glyf_accelerator.glyph_for_gid (item.get_glyph_index ())
					 .get_points (font, glyf_accelerator, comp_points,
			 			      phantom_only, depth + 1)
			|| comp_points.length < PHANTOM_COUNT))
	    return false;

	  /* Copy phantom points from component if USE_MY_METRICS flag set */
	  if (item.is_use_my_metrics ())
	    for (unsigned int i = 0; i < PHANTOM_COUNT; i++)
	      phantoms[i] = comp_points[comp_points.length - PHANTOM_COUNT + i];

	  /* Apply component transformation & translation */
	  item.transform_points (comp_points);

	  /* Apply translation from gvar */
	  comp_points.translate (points[comp_index]);

	  if (item.is_anchored ())
	  {
	    unsigned int p1, p2;
	    item.get_anchor_points (p1, p2);
	    if (likely (p1 < all_points.length && p2 < comp_points.length))
	    {
	      contour_point_t delta;
	      delta.init (all_points[p1].x - comp_points[p2].x,
			  all_points[p1].y - comp_points[p2].y);

	      comp_points.translate (delta);
	    }
	  }

	  all_points.extend (comp_points.sub_array (0, comp_points.length - PHANTOM_COUNT));

	  comp_index++;
	}

	all_points.extend (phantoms);
      } break;
      default:
	all_points.extend (phantoms);
      }

      if (depth == 0) /* Apply at top level */
      {
	/* Undocumented rasterizer behavior:
	 * Shift points horizontally by the updated left side bearing
	 */
	contour_point_t delta;
	delta.init (-phantoms[PHANTOM_LEFT].x, 0.f);
	if (delta.x) all_points.translate (delta);
      }

      return true;
    }

    bool get_extents (hb_font_t *font, const accelerator_t &glyf_accelerator,
		      hb_glyph_extents_t *extents) const
    {
      if (type == EMPTY) return true; /* Empty glyph; zero extents. */
      return header->get_extents (font, glyf_accelerator, gid, extents);
    }

    hb_bytes_t get_bytes () const { return bytes; }

    Glyph (hb_bytes_t bytes_ = hb_bytes_t (),
	   hb_codepoint_t gid_ = (hb_codepoint_t) -1) : bytes (bytes_), gid (gid_),
							header (bytes.as<GlyphHeader> ())
    {
      int num_contours = header->numberOfContours;
      if (unlikely (num_contours == 0)) type = EMPTY;
      else if (num_contours > 0) type = SIMPLE;
      else type = COMPOSITE; /* negative numbers */
    }

    protected:
    hb_bytes_t bytes;
    hb_codepoint_t gid;
    const GlyphHeader *header;
    unsigned type;
  };

  struct accelerator_t
  {
    void init (hb_face_t *face_)
    {
      short_offset = false;
      num_glyphs = 0;
      loca_table = nullptr;
      glyf_table = nullptr;
#ifndef HB_NO_VAR
      gvar = nullptr;
#endif
      hmtx = nullptr;
      vmtx = nullptr;
      face = face_;
      const OT::head &head = *face->table.head;
      if (head.indexToLocFormat > 1 || head.glyphDataFormat > 0)
	/* Unknown format.  Leave num_glyphs=0, that takes care of disabling us. */
	return;
      short_offset = 0 == head.indexToLocFormat;

      loca_table = hb_sanitize_context_t ().reference_table<loca> (face);
      glyf_table = hb_sanitize_context_t ().reference_table<glyf> (face);
#ifndef HB_NO_VAR
      gvar = face->table.gvar;
#endif
      hmtx = face->table.hmtx;
      vmtx = face->table.vmtx;

      num_glyphs = hb_max (1u, loca_table.get_length () / (short_offset ? 2 : 4)) - 1;
      num_glyphs = hb_min (num_glyphs, face->get_num_glyphs ());
    }

    void fini ()
    {
      loca_table.destroy ();
      glyf_table.destroy ();
    }

    protected:
    template<typename T>
    bool get_points (hb_font_t *font, hb_codepoint_t gid, T consumer) const
    {
      if (gid >= num_glyphs) return false;

      /* Making this allocfree is not that easy
	 https://github.com/harfbuzz/harfbuzz/issues/2095
	 mostly because of gvar handling in VF fonts,
	 perhaps a separate path for non-VF fonts can be considered */
      contour_point_vector_t all_points;

      bool phantom_only = !consumer.is_consuming_contour_points ();
      if (unlikely (!glyph_for_gid (gid).get_points (font, *this, all_points, phantom_only)))
	return false;

      if (consumer.is_consuming_contour_points ())
      {
	for (unsigned point_index = 0; point_index + 4 < all_points.length; ++point_index)
	  consumer.consume_point (all_points[point_index]);
	consumer.points_end ();
      }

      /* Where to write phantoms, nullptr if not requested */
      contour_point_t *phantoms = consumer.get_phantoms_sink ();
      if (phantoms)
	for (unsigned i = 0; i < PHANTOM_COUNT; ++i)
	  phantoms[i] = all_points[all_points.length - PHANTOM_COUNT + i];

      return true;
    }

#ifndef HB_NO_VAR
    struct points_aggregator_t
    {
      hb_font_t *font;
      hb_glyph_extents_t *extents;
      contour_point_t *phantoms;

      struct contour_bounds_t
      {
	contour_bounds_t () { min_x = min_y = FLT_MAX; max_x = max_y = -FLT_MAX; }

	void add (const contour_point_t &p)
	{
	  min_x = hb_min (min_x, p.x);
	  min_y = hb_min (min_y, p.y);
	  max_x = hb_max (max_x, p.x);
	  max_y = hb_max (max_y, p.y);
	}

	bool empty () const { return (min_x >= max_x) || (min_y >= max_y); }

	void get_extents (hb_font_t *font, hb_glyph_extents_t *extents)
	{
	  if (unlikely (empty ()))
	  {
	    extents->width = 0;
	    extents->x_bearing = 0;
	    extents->height = 0;
	    extents->y_bearing = 0;
	    return;
	  }
	  extents->x_bearing = font->em_scalef_x (min_x);
	  extents->width = font->em_scalef_x (max_x) - extents->x_bearing;
	  extents->y_bearing = font->em_scalef_y (max_y);
	  extents->height = font->em_scalef_y (min_y) - extents->y_bearing;
	}

	protected:
	float min_x, min_y, max_x, max_y;
      } bounds;

      points_aggregator_t (hb_font_t *font_, hb_glyph_extents_t *extents_, contour_point_t *phantoms_)
      {
	font = font_;
	extents = extents_;
	phantoms = phantoms_;
	if (extents) bounds = contour_bounds_t ();
      }

      void consume_point (const contour_point_t &point) { bounds.add (point); }
      void points_end () { bounds.get_extents (font, extents); }

      bool is_consuming_contour_points () { return extents; }
      contour_point_t *get_phantoms_sink () { return phantoms; }
    };

    public:
    unsigned
    get_advance_var (hb_font_t *font, hb_codepoint_t gid, bool is_vertical) const
    {
      if (unlikely (gid >= num_glyphs)) return 0;

      bool success = false;

      contour_point_t phantoms[PHANTOM_COUNT];
      if (likely (font->num_coords == gvar->get_axis_count ()))
	success = get_points (font, gid, points_aggregator_t (font, nullptr, phantoms));

      if (unlikely (!success))
	return is_vertical ? vmtx->get_advance (gid) : hmtx->get_advance (gid);

      float result = is_vertical
		   ? phantoms[PHANTOM_TOP].y - phantoms[PHANTOM_BOTTOM].y
		   : phantoms[PHANTOM_RIGHT].x - phantoms[PHANTOM_LEFT].x;
      return hb_clamp (roundf (result), 0.f, (float) UINT_MAX / 2);
    }

    int get_side_bearing_var (hb_font_t *font, hb_codepoint_t gid, bool is_vertical) const
    {
      if (unlikely (gid >= num_glyphs)) return 0;

      hb_glyph_extents_t extents;

      contour_point_t phantoms[PHANTOM_COUNT];
      if (unlikely (!get_points (font, gid, points_aggregator_t (font, &extents, phantoms))))
	return is_vertical ? vmtx->get_side_bearing (gid) : hmtx->get_side_bearing (gid);

      return is_vertical
	   ? ceilf (phantoms[PHANTOM_TOP].y) - extents.y_bearing
	   : floorf (phantoms[PHANTOM_LEFT].x);
    }
#endif

    public:
    bool get_extents (hb_font_t *font, hb_codepoint_t gid, hb_glyph_extents_t *extents) const
    {
      if (unlikely (gid >= num_glyphs)) return false;

#ifndef HB_NO_VAR
      if (font->num_coords && font->num_coords == gvar->get_axis_count ())
	return get_points (font, gid, points_aggregator_t (font, extents, nullptr));
#endif
      return glyph_for_gid (gid).get_extents (font, *this, extents);
    }

    const Glyph
    glyph_for_gid (hb_codepoint_t gid, bool needs_padding_removal = false) const
    {
      if (unlikely (gid >= num_glyphs)) return Glyph ();

      unsigned int start_offset, end_offset;

      if (short_offset)
      {
	const HBUINT16 *offsets = (const HBUINT16 *) loca_table->dataZ.arrayZ;
	start_offset = 2 * offsets[gid];
	end_offset   = 2 * offsets[gid + 1];
      }
      else
      {
	const HBUINT32 *offsets = (const HBUINT32 *) loca_table->dataZ.arrayZ;
	start_offset = offsets[gid];
	end_offset   = offsets[gid + 1];
      }

      if (unlikely (start_offset > end_offset || end_offset > glyf_table.get_length ()))
	return Glyph ();

      Glyph glyph (hb_bytes_t ((const char *) this->glyf_table + start_offset,
			       end_offset - start_offset), gid);
      return needs_padding_removal ? glyph.trim_padding () : glyph;
    }

    unsigned
    add_gid_and_children (hb_codepoint_t gid,
			  hb_set_t *gids_to_retain,
			  unsigned depth = 0,
			  unsigned operation_count = 0) const
    {
      if (unlikely (depth++ > HB_MAX_NESTING_LEVEL)) return operation_count;
      if (unlikely (operation_count++ > HB_MAX_COMPOSITE_OPERATIONS)) return operation_count;
      /* Check if is already visited */
      if (gids_to_retain->has (gid)) return operation_count;

      gids_to_retain->add (gid);

      auto it = glyph_for_gid (gid).get_composite_iterator ();
      while (it)
      {
        auto item = *(it++);
        operation_count =
            add_gid_and_children (item.get_glyph_index (), gids_to_retain, depth, operation_count);
      }

      return operation_count;
    }

#ifdef HB_EXPERIMENTAL_API
    struct path_builder_t
    {
      hb_font_t *font;
      draw_helper_t *draw_helper;

      struct optional_point_t
      {
	optional_point_t () { has_data = false; }
	optional_point_t (float x_, float y_) { x = x_; y = y_; has_data = true; }

	bool has_data;
	float x;
	float y;

	optional_point_t lerp (optional_point_t p, float t)
	{ return optional_point_t (x + t * (p.x - x), y + t * (p.y - y)); }
      } first_oncurve, first_offcurve, last_offcurve;

      path_builder_t (hb_font_t *font_, draw_helper_t &draw_helper_)
      {
	font = font_;
	draw_helper = &draw_helper_;
	first_oncurve = first_offcurve = last_offcurve = optional_point_t ();
      }

      /* based on https://github.com/RazrFalcon/ttf-parser/blob/4f32821/src/glyf.rs#L287
	 See also:
	 * https://developer.apple.com/fonts/TrueType-Reference-Manual/RM01/Chap1.html
	 * https://stackoverflow.com/a/20772557 */
      void consume_point (const contour_point_t &point)
      {
	/* Skip empty contours */
	if (unlikely (point.is_end_point && !first_oncurve.has_data && !first_offcurve.has_data))
	  return;

	bool is_on_curve = point.flag & Glyph::FLAG_ON_CURVE;
	optional_point_t p (point.x, point.y);
	if (!first_oncurve.has_data)
	{
	  if (is_on_curve)
	  {
	    first_oncurve = p;
	    draw_helper->move_to (font->em_scalef_x (p.x), font->em_scalef_y (p.y));
	  }
	  else
	  {
	    if (first_offcurve.has_data)
	    {
	      optional_point_t mid = first_offcurve.lerp (p, .5f);
	      first_oncurve = mid;
	      last_offcurve = p;
	      draw_helper->move_to (font->em_scalef_x (mid.x), font->em_scalef_y (mid.y));
	    }
	    else
	      first_offcurve = p;
	  }
	}
	else
	{
	  if (last_offcurve.has_data)
	  {
	    if (is_on_curve)
	    {
	      draw_helper->quadratic_to (font->em_scalef_x (last_offcurve.x), font->em_scalef_y (last_offcurve.y),
					 font->em_scalef_x (p.x), font->em_scalef_y (p.y));
	      last_offcurve = optional_point_t ();
	    }
	    else
	    {
	      optional_point_t mid = last_offcurve.lerp (p, .5f);
	      draw_helper->quadratic_to (font->em_scalef_x (last_offcurve.x), font->em_scalef_y (last_offcurve.y),
					 font->em_scalef_x (mid.x), font->em_scalef_y (mid.y));
	      last_offcurve = p;
	    }
	  }
	  else
	  {
	    if (is_on_curve)
	      draw_helper->line_to (font->em_scalef_x (p.x), font->em_scalef_y (p.y));
	    else
	      last_offcurve = p;
	  }
	}

	if (point.is_end_point)
	{
	  if (first_offcurve.has_data && last_offcurve.has_data)
	  {
	    optional_point_t mid = last_offcurve.lerp (first_offcurve, .5f);
	    draw_helper->quadratic_to (font->em_scalef_x (last_offcurve.x), font->em_scalef_y (last_offcurve.y),
				       font->em_scalef_x (mid.x), font->em_scalef_y (mid.y));
	    last_offcurve = optional_point_t ();
	    /* now check the rest */
	  }

	  if (first_offcurve.has_data && first_oncurve.has_data)
	    draw_helper->quadratic_to (font->em_scalef_x (first_offcurve.x), font->em_scalef_y (first_offcurve.y),
				       font->em_scalef_x (first_oncurve.x), font->em_scalef_y (first_oncurve.y));
	  else if (last_offcurve.has_data && first_oncurve.has_data)
	    draw_helper->quadratic_to (font->em_scalef_x (last_offcurve.x), font->em_scalef_y (last_offcurve.y),
				       font->em_scalef_x (first_oncurve.x), font->em_scalef_y (first_oncurve.y));
	  else if (first_oncurve.has_data)
	    draw_helper->line_to (font->em_scalef_x (first_oncurve.x), font->em_scalef_y (first_oncurve.y));

	  /* Getting ready for the next contour */
	  first_oncurve = first_offcurve = last_offcurve = optional_point_t ();
	  draw_helper->end_path ();
	}
      }
      void points_end () {}

      bool is_consuming_contour_points () { return true; }
      contour_point_t *get_phantoms_sink () { return nullptr; }
    };

    bool
    get_path (hb_font_t *font, hb_codepoint_t gid, draw_helper_t &draw_helper) const
    { return get_points (font, gid, path_builder_t (font, draw_helper)); }
#endif

#ifndef HB_NO_VAR
    const gvar_accelerator_t *gvar;
#endif
    const hmtx_accelerator_t *hmtx;
    const vmtx_accelerator_t *vmtx;

    private:
    bool short_offset;
    unsigned int num_glyphs;
    hb_blob_ptr_t<loca> loca_table;
    hb_blob_ptr_t<glyf> glyf_table;
    hb_face_t *face;
  };

  struct SubsetGlyph
  {
    hb_codepoint_t new_gid;
    hb_codepoint_t old_gid;
    Glyph source_glyph;
    hb_bytes_t dest_start;  /* region of source_glyph to copy first */
    hb_bytes_t dest_end;    /* region of source_glyph to copy second */

    bool serialize (hb_serialize_context_t *c,
		    const hb_subset_plan_t *plan) const
    {
      TRACE_SERIALIZE (this);

      hb_bytes_t dest_glyph = dest_start.copy (c);
      dest_glyph = hb_bytes_t (&dest_glyph, dest_glyph.length + dest_end.copy (c).length);
      unsigned int pad_length = padding ();
      DEBUG_MSG (SUBSET, nullptr, "serialize %d byte glyph, width %d pad %d", dest_glyph.length, dest_glyph.length + pad_length, pad_length);

      HBUINT8 pad;
      pad = 0;
      while (pad_length > 0)
      {
	c->embed (pad);
	pad_length--;
      }

      if (unlikely (!dest_glyph.length)) return_trace (true);

      /* update components gids */
      for (auto &_ : Glyph (dest_glyph).get_composite_iterator ())
      {
	hb_codepoint_t new_gid;
	if (plan->new_gid_for_old_gid (_.get_glyph_index (), &new_gid))
	  const_cast<CompositeGlyphChain &> (_).set_glyph_index (new_gid);
      }

      if (plan->flags & HB_SUBSET_FLAGS_NO_HINTING)
        Glyph (dest_glyph).drop_hints ();

      if (plan->flags & HB_SUBSET_FLAGS_SET_OVERLAPS_FLAG)
        Glyph (dest_glyph).set_overlaps_flag ();

      return_trace (true);
    }

    void drop_hints_bytes ()
    { source_glyph.drop_hints_bytes (dest_start, dest_end); }

    unsigned int      length () const { return dest_start.length + dest_end.length; }
    /* pad to 2 to ensure 2-byte loca will be ok */
    unsigned int     padding () const { return length () % 2; }
    unsigned int padded_size () const { return length () + padding (); }
  };

  protected:
  UnsizedArrayOf<HBUINT8>
		dataZ;	/* Glyphs data. */
  public:
  DEFINE_SIZE_MIN (0);	/* In reality, this is UNBOUNDED() type; but since we always
			 * check the size externally, allow Null() object of it by
			 * defining it _MIN instead. */
};

struct glyf_accelerator_t : glyf::accelerator_t {};

} /* namespace OT */


#endif /* HB_OT_GLYF_TABLE_HH */

#ifndef OT_GLYF_COMPOSITEGLYPH_HH
#define OT_GLYF_COMPOSITEGLYPH_HH


#include "../../hb-open-type.hh"


namespace OT {
namespace glyf_impl {


struct CompositeGlyphRecord
{
  protected:
  enum composite_glyph_flag_t
  {
    ARG_1_AND_2_ARE_WORDS	= 0x0001,
    ARGS_ARE_XY_VALUES		= 0x0002,
    ROUND_XY_TO_GRID		= 0x0004,
    WE_HAVE_A_SCALE		= 0x0008,
    MORE_COMPONENTS		= 0x0020,
    WE_HAVE_AN_X_AND_Y_SCALE	= 0x0040,
    WE_HAVE_A_TWO_BY_TWO	= 0x0080,
    WE_HAVE_INSTRUCTIONS	= 0x0100,
    USE_MY_METRICS		= 0x0200,
    OVERLAP_COMPOUND		= 0x0400,
    SCALED_COMPONENT_OFFSET	= 0x0800,
    UNSCALED_COMPONENT_OFFSET	= 0x1000,
#ifndef HB_NO_BEYOND_64K
    GID_IS_24BIT		= 0x2000
#endif
  };

  public:
  unsigned int get_size () const
  {
    unsigned int size = min_size;
    /* glyphIndex is 24bit instead of 16bit */
#ifndef HB_NO_BEYOND_64K
    if (flags & GID_IS_24BIT) size += HBGlyphID24::static_size - HBGlyphID16::static_size;
#endif
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
    const auto *p = &StructAfter<const HBUINT8> (flags);
#ifndef HB_NO_BEYOND_64K
    if (flags & GID_IS_24BIT)
      p += HBGlyphID24::static_size;
    else
#endif
      p += HBGlyphID16::static_size;
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

    const auto *p = &StructAfter<const HBINT8> (flags);
#ifndef HB_NO_BEYOND_64K
    if (flags & GID_IS_24BIT)
      p += HBGlyphID24::static_size;
    else
#endif
      p += HBGlyphID16::static_size;
    int tx, ty;
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

  public:
  hb_codepoint_t get_gid () const
  {
#ifndef HB_NO_BEYOND_64K
    if (flags & GID_IS_24BIT)
      return StructAfter<const HBGlyphID24> (flags);
    else
#endif
      return StructAfter<const HBGlyphID16> (flags);
  }
  void set_gid (hb_codepoint_t gid)
  {
#ifndef HB_NO_BEYOND_64K
    if (flags & GID_IS_24BIT)
      StructAfter<HBGlyphID24> (flags) = gid;
    else
#endif
      /* TODO assert? */
      StructAfter<HBGlyphID16> (flags) = gid;
  }

  protected:
  HBUINT16	flags;
  HBUINT24	pad;
  public:
  DEFINE_SIZE_MIN (4);
};

struct composite_iter_t : hb_iter_with_fallback_t<composite_iter_t, const CompositeGlyphRecord &>
{
  typedef const CompositeGlyphRecord *__item_t__;
  composite_iter_t (hb_bytes_t glyph_, __item_t__ current_) :
      glyph (glyph_), current (nullptr), current_size (0)
  {
    set_current (current_);
  }

  composite_iter_t () : glyph (hb_bytes_t ()), current (nullptr), current_size (0) {}

  item_t __item__ () const { return *current; }
  bool __more__ () const { return current; }
  void __next__ ()
  {
    if (!current->has_more ()) { current = nullptr; return; }

    set_current (&StructAtOffset<CompositeGlyphRecord> (current, current_size));
  }
  composite_iter_t __end__ () const { return composite_iter_t (); }
  bool operator != (const composite_iter_t& o) const
  { return current != o.current; }


  void set_current (__item_t__ current_)
  {
    if (!glyph.check_range (current_, CompositeGlyphRecord::min_size))
    {
      current = nullptr;
      current_size = 0;
      return;
    }
    unsigned size = current_->get_size ();
    if (!glyph.check_range (current_, size))
    {
      current = nullptr;
      current_size = 0;
      return;
    }

    current = current_;
    current_size = size;
  }

  private:
  hb_bytes_t glyph;
  __item_t__ current;
  unsigned current_size;
};

struct CompositeGlyph
{
  const GlyphHeader &header;
  hb_bytes_t bytes;
  CompositeGlyph (const GlyphHeader &header_, hb_bytes_t bytes_) :
    header (header_), bytes (bytes_) {}

  composite_iter_t iter () const
  { return composite_iter_t (bytes, &StructAfter<CompositeGlyphRecord, GlyphHeader> (header)); }

  unsigned int instructions_length (hb_bytes_t bytes) const
  {
    unsigned int start = bytes.length;
    unsigned int end = bytes.length;
    const CompositeGlyphRecord *last = nullptr;
    for (auto &item : iter ())
      last = &item;
    if (unlikely (!last)) return 0;

    if (last->has_instructions ())
      start = (char *) last - &bytes + last->get_size ();
    if (unlikely (start > end)) return 0;
    return end - start;
  }

  /* Trimming for composites not implemented.
   * If removing hints it falls out of that. */
  const hb_bytes_t trim_padding () const { return bytes; }

  void drop_hints ()
  {
    for (const auto &_ : iter ())
      const_cast<CompositeGlyphRecord &> (_).drop_instructions_flag ();
  }

  /* Chop instructions off the end */
  void drop_hints_bytes (hb_bytes_t &dest_start) const
  { dest_start = bytes.sub_array (0, bytes.length - instructions_length (bytes)); }

  void set_overlaps_flag ()
  {
    CompositeGlyphRecord& glyph_chain = const_cast<CompositeGlyphRecord &> (
	StructAfter<CompositeGlyphRecord, GlyphHeader> (header));
    if (!bytes.check_range(&glyph_chain, CompositeGlyphRecord::min_size))
      return;
    glyph_chain.set_overlaps_flag ();
  }
};


} /* namespace glyf_impl */
} /* namespace OT */


#endif /* OT_GLYF_COMPOSITEGLYPH_HH */

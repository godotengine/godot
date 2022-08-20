#ifndef OT_GLYF_SUBSETGLYPH_HH
#define OT_GLYF_SUBSETGLYPH_HH


#include "../../hb-open-type.hh"


namespace OT {
namespace glyf_impl {


struct SubsetGlyph
{
  hb_codepoint_t new_gid;
  hb_codepoint_t old_gid;
  Glyph source_glyph;
  hb_bytes_t dest_start;  /* region of source_glyph to copy first */
  hb_bytes_t dest_end;    /* region of source_glyph to copy second */

  bool serialize (hb_serialize_context_t *c,
		  bool use_short_loca,
		  const hb_subset_plan_t *plan) const
  {
    TRACE_SERIALIZE (this);

    hb_bytes_t dest_glyph = dest_start.copy (c);
    dest_glyph = hb_bytes_t (&dest_glyph, dest_glyph.length + dest_end.copy (c).length);
    unsigned int pad_length = use_short_loca ? padding () : 0;
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
      if (plan->new_gid_for_old_gid (_.get_gid(), &new_gid))
	const_cast<CompositeGlyphRecord &> (_).set_gid (new_gid);
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


} /* namespace glyf_impl */
} /* namespace OT */


#endif /* OT_GLYF_SUBSETGLYPH_HH */

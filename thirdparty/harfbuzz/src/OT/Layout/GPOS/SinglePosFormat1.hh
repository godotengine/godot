#ifndef OT_LAYOUT_GPOS_SINGLEPOSFORMAT1_HH
#define OT_LAYOUT_GPOS_SINGLEPOSFORMAT1_HH

#include "Common.hh"
#include "ValueFormat.hh"

namespace OT {
namespace Layout {
namespace GPOS_impl {

struct SinglePosFormat1
{
  protected:
  HBUINT16      format;                 /* Format identifier--format = 1 */
  Offset16To<Coverage>
                coverage;               /* Offset to Coverage table--from
                                         * beginning of subtable */
  ValueFormat   valueFormat;            /* Defines the types of data in the
                                         * ValueRecord */
  ValueRecord   values;                 /* Defines positioning
                                         * value(s)--applied to all glyphs in
                                         * the Coverage table */
  public:
  DEFINE_SIZE_ARRAY (6, values);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
                  coverage.sanitize (c, this) &&
                  valueFormat.sanitize_value (c, this, values));
  }

  bool intersects (const hb_set_t *glyphs) const
  { return (this+coverage).intersects (glyphs); }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}
  void collect_variation_indices (hb_collect_variation_indices_context_t *c) const
  {
    if (!valueFormat.has_device ()) return;

    auto it =
    + hb_iter (this+coverage)
    | hb_filter (c->glyph_set)
    ;

    if (!it) return;
    valueFormat.collect_variation_indices (c, this, values.as_array (valueFormat.get_len ()));
  }

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  { if (unlikely (!(this+coverage).collect_coverage (c->input))) return; }

  const Coverage &get_coverage () const { return this+coverage; }

  ValueFormat get_value_format () const { return valueFormat; }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;
    unsigned int index = (this+coverage).get_coverage  (buffer->cur().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    valueFormat.apply_value (c, this, values, buffer->cur_pos());

    buffer->idx++;
    return_trace (true);
  }

  template<typename Iterator,
      typename SrcLookup,
      hb_requires (hb_is_iterator (Iterator))>
  void serialize (hb_serialize_context_t *c,
                  const SrcLookup *src,
                  Iterator it,
                  ValueFormat newFormat,
                  const hb_map_t *layout_variation_idx_map)
  {
    if (unlikely (!c->extend_min (this))) return;
    if (unlikely (!c->check_assign (valueFormat,
                                    newFormat,
                                    HB_SERIALIZE_ERROR_INT_OVERFLOW))) return;

    for (const hb_array_t<const Value>& _ : + it | hb_map (hb_second))
    {
      src->get_value_format ().copy_values (c, newFormat, src,  &_, layout_variation_idx_map);
      // Only serialize the first entry in the iterator, the rest are assumed to
      // be the same.
      break;
    }

    auto glyphs =
    + it
    | hb_map_retains_sorting (hb_first)
    ;

    coverage.serialize_serialize (c, glyphs);
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    auto it =
    + hb_iter (this+coverage)
    | hb_filter (glyphset)
    | hb_map_retains_sorting (glyph_map)
    | hb_zip (hb_repeat (values.as_array (valueFormat.get_len ())))
    ;

    bool ret = bool (it);
    SinglePos_serialize (c->serializer, this, it, c->plan->layout_variation_idx_map);
    return_trace (ret);
  }
};

}
}
}

#endif /* OT_LAYOUT_GPOS_SINGLEPOSFORMAT1_HH */

#ifndef OT_LAYOUT_GSUB_LIGATURESET_HH
#define OT_LAYOUT_GSUB_LIGATURESET_HH

#include "Common.hh"
#include "Ligature.hh"

namespace OT {
namespace Layout {
namespace GSUB_impl {

template <typename Types>
struct LigatureSet
{
  protected:
  Array16OfOffset16To<Ligature<Types>>
                ligature;               /* Array LigatureSet tables
                                         * ordered by preference */
  public:
  DEFINE_SIZE_ARRAY (2, ligature);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (ligature.sanitize (c, this));
  }

  bool intersects (const hb_set_t *glyphs) const
  {
    return
    + hb_iter (ligature)
    | hb_map (hb_add (this))
    | hb_map ([glyphs] (const Ligature<Types> &_) { return _.intersects (glyphs); })
    | hb_any
    ;
  }

  void closure (hb_closure_context_t *c) const
  {
    + hb_iter (ligature)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const Ligature<Types> &_) { _.closure (c); })
    ;
  }

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    + hb_iter (ligature)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const Ligature<Types> &_) { _.collect_glyphs (c); })
    ;
  }

  bool would_apply (hb_would_apply_context_t *c) const
  {
    return
    + hb_iter (ligature)
    | hb_map (hb_add (this))
    | hb_map ([c] (const Ligature<Types> &_) { return _.would_apply (c); })
    | hb_any
    ;
  }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    unsigned int num_ligs = ligature.len;
    for (unsigned int i = 0; i < num_ligs; i++)
    {
      const auto &lig = this+ligature[i];
      if (lig.apply (c)) return_trace (true);
    }

    return_trace (false);
  }

  bool serialize (hb_serialize_context_t *c,
                  hb_array_t<const HBGlyphID16> ligatures,
                  hb_array_t<const unsigned int> component_count_list,
                  hb_array_t<const HBGlyphID16> &component_list /* Starting from second for each ligature */)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);
    if (unlikely (!ligature.serialize (c, ligatures.length))) return_trace (false);
    for (unsigned int i = 0; i < ligatures.length; i++)
    {
      unsigned int component_count = (unsigned) hb_max ((int) component_count_list[i] - 1, 0);
      if (unlikely (!ligature[i].serialize_serialize (c,
                                                      ligatures[i],
                                                      component_list.sub_array (0, component_count))))
        return_trace (false);
      component_list += component_count;
    }
    return_trace (true);
  }

  bool subset (hb_subset_context_t *c, unsigned coverage_idx) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);

    + hb_iter (ligature)
    | hb_filter (subset_offset_array (c, out->ligature, this, coverage_idx))
    | hb_drain
    ;

    if (bool (out->ligature))
      // Ensure Coverage table is always packed after this.
      c->serializer->add_virtual_link (coverage_idx);

    return_trace (bool (out->ligature));
  }
};

}
}
}

#endif  /* OT_LAYOUT_GSUB_LIGATURESET_HH */

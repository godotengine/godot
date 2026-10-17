#ifndef OT_LAYOUT_GPOS_LIGATUREARRAY_HH
#define OT_LAYOUT_GPOS_LIGATUREARRAY_HH

namespace OT {
namespace Layout {
namespace GPOS_impl {


typedef AnchorMatrix LigatureAttach;    /* component-major--
                                         * in order of writing direction--,
                                         * mark-minor--
                                         * ordered by class--zero-based. */

/* Array of LigatureAttach tables ordered by LigatureCoverage Index */
struct LigatureArray : List16OfOffset16To<LigatureAttach>
{
  template <typename Iterator,
            hb_requires (hb_is_iterator (Iterator))>
  bool subset (hb_subset_context_t *c,
               Iterator             coverage,
               unsigned             class_count,
               const hb_map_t      *klass_mapping,
               hb_sorted_vector_t<hb_codepoint_t> &new_coverage /* OUT */) const
  {
    TRACE_SUBSET (this);
    const hb_map_t &glyph_map = c->plan->glyph_map_gsub;

    auto *out = c->serializer->start_embed (this);
    if (unlikely (!c->serializer->extend_min (out)))  return_trace (false);

    bool ret = false;
    for (const auto _ : + hb_zip (coverage, *this)
                        | hb_filter (glyph_map, hb_first))
    {
      const LigatureAttach& src = (this + _.second);
      bool non_empty = + hb_range (src.rows * class_count)
                       | hb_filter ([=] (unsigned index) { return klass_mapping->has (index % class_count); })
                       | hb_map ([&] (const unsigned index) { return !src.offset_is_null (index / class_count, index % class_count, class_count); })
                       | hb_any;

      if (!non_empty) continue;

      auto *matrix = out->serialize_append (c->serializer);
      if (unlikely (!matrix)) return_trace (false);

      auto indexes =
          + hb_range (src.rows * class_count)
          | hb_filter ([=] (unsigned index) { return klass_mapping->has (index % class_count); })
          ;
      ret |= matrix->serialize_subset (c,
				       _.second,
				       this,
				       src.rows,
				       indexes);

      hb_codepoint_t new_gid = glyph_map.get (_.first);
      new_coverage.push (new_gid);
    }
    return_trace (ret);
  }
};


}
}
}

#endif /* OT_LAYOUT_GPOS_LIGATUREARRAY_HH */

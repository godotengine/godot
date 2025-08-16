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
               const hb_map_t      *klass_mapping) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();

    auto *out = c->serializer->start_embed (this);
    if (unlikely (!c->serializer->extend_min (out)))  return_trace (false);

    bool ret = false;
    for (const auto _ : + hb_zip (coverage, *this)
                  | hb_filter (glyphset, hb_first))
    {
      auto *matrix = out->serialize_append (c->serializer);
      if (unlikely (!matrix)) return_trace (false);

      const LigatureAttach& src = (this + _.second);
      auto indexes =
          + hb_range (src.rows * class_count)
          | hb_filter ([=] (unsigned index) { return klass_mapping->has (index % class_count); })
          ;
      ret |= matrix->serialize_subset (c,
				       _.second,
				       this,
				       src.rows,
				       indexes);
    }
    return_trace (ret);
  }
};


}
}
}

#endif /* OT_LAYOUT_GPOS_LIGATUREARRAY_HH */

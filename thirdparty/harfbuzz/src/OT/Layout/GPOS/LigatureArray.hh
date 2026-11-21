#ifndef OT_LAYOUT_GPOS_LIGATUREARRAY_HH
#define OT_LAYOUT_GPOS_LIGATUREARRAY_HH

namespace OT {
namespace Layout {
namespace GPOS_impl {

HB_INTERNAL inline
void collect_ligature_variation_indices (hb_collect_variation_indices_context_t *c,
                                        const Coverage &coverage,
                                        const void *base,
                                        glyph_data_t offsets,
                                        unsigned count,
                                        unsigned class_count,
                                        const hb_map_t &klass_mapping)
{
  for (const auto _ : hb_zip (coverage, hb_range (count)))
  {
    if (!c->glyph_set->has (_.first)) continue;
    hb_barrier ();
    unsigned offset = offsets[_.second];
    if (!offset) continue;

    const void *matrix = &StructAtOffset<const HBUINT16> (base, offset);
    /* LigatureAttach always has a 16-bit component count; its anchor
     * offsets have the same width as the LigatureArray offsets. */
    unsigned rows = *static_cast<const HBUINT16 *> (matrix);
    glyph_data_t anchors = offsets;
    anchors.data = (const char *) matrix + HBUINT16::static_size;

    hb_sorted_vector_t<unsigned> indexes;
    collect_ligature_anchor_indexes (rows, class_count, klass_mapping, indexes);
    collect_anchor_variation_indices (c, matrix, anchors, indexes.iter ());
  }
}


/* Array of LigatureAttach tables ordered by LigatureCoverage Index */
template <typename Types>
struct LigatureArray :
  Types::template ArrayOf<typename Types::template OffsetTo<AnchorMatrix<Types, HBUINT16>>>
{
  typedef AnchorMatrix<Types, HBUINT16> LigatureAttach;
  typedef typename Types::template OffsetTo<LigatureAttach> LigatureAttachOffset;
  typedef typename Types::template ArrayOf<LigatureAttachOffset> Super;

  const LigatureAttach& operator [] (int i_) const
  {
    unsigned int i = (unsigned int) i_;
    if (unlikely (i >= this->len)) return Null (LigatureAttach);
    hb_barrier ();
    return this+this->arrayZ[i];
  }

  template <typename ...Ts>
  bool sanitize (hb_sanitize_context_t *c, Ts&&... ds) const
  {
    TRACE_SANITIZE (this);
    return_trace (Super::sanitize (c, this, std::forward<Ts> (ds)...));
  }

  void collect_variation_indices (hb_collect_variation_indices_context_t *c,
                                  const Coverage &coverage,
                                  unsigned class_count,
                                  const hb_map_t &klass_mapping) const
  {
    collect_ligature_variation_indices (c, coverage, this, this->arrayZ,
                                        this->len, class_count, klass_mapping);
  }

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

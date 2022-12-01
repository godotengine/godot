#ifndef OT_LAYOUT_GPOS_MARKRECORD_HH
#define OT_LAYOUT_GPOS_MARKRECORD_HH

namespace OT {
namespace Layout {
namespace GPOS_impl {

struct MarkRecord
{
  friend struct MarkArray;

  public:
  HBUINT16      klass;                  /* Class defined for this mark */
  Offset16To<Anchor>
                markAnchor;             /* Offset to Anchor table--from
                                         * beginning of MarkArray table */
  public:
  DEFINE_SIZE_STATIC (4);

  unsigned get_class () const { return (unsigned) klass; }
  bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && markAnchor.sanitize (c, base));
  }

  MarkRecord *subset (hb_subset_context_t    *c,
                      const void             *src_base,
                      const hb_map_t         *klass_mapping) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->embed (this);
    if (unlikely (!out)) return_trace (nullptr);

    out->klass = klass_mapping->get (klass);
    out->markAnchor.serialize_subset (c, markAnchor, src_base);
    return_trace (out);
  }

  void collect_variation_indices (hb_collect_variation_indices_context_t *c,
                                  const void *src_base) const
  {
    (src_base+markAnchor).collect_variation_indices (c);
  }
};


}
}
}

#endif /* OT_LAYOUT_GPOS_MARKRECORD_HH */

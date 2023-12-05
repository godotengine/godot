#ifndef OT_LAYOUT_GPOS_ANCHORMATRIX_HH
#define OT_LAYOUT_GPOS_ANCHORMATRIX_HH

namespace OT {
namespace Layout {
namespace GPOS_impl {

struct AnchorMatrix
{
  HBUINT16      rows;                   /* Number of rows */
  UnsizedArrayOf<Offset16To<Anchor>>
                matrixZ;                /* Matrix of offsets to Anchor tables--
                                         * from beginning of AnchorMatrix table */
  public:
  DEFINE_SIZE_ARRAY (2, matrixZ);

  bool sanitize (hb_sanitize_context_t *c, unsigned int cols) const
  {
    TRACE_SANITIZE (this);
    if (!c->check_struct (this)) return_trace (false);
    if (unlikely (hb_unsigned_mul_overflows (rows, cols))) return_trace (false);
    unsigned int count = rows * cols;
    if (!c->check_array (matrixZ.arrayZ, count)) return_trace (false);

    if (c->lazy_some_gpos)
      return_trace (true);

    for (unsigned int i = 0; i < count; i++)
      if (!matrixZ[i].sanitize (c, this)) return_trace (false);
    return_trace (true);
  }

  const Anchor& get_anchor (hb_ot_apply_context_t *c,
			    unsigned int row, unsigned int col,
			    unsigned int cols, bool *found) const
  {
    *found = false;
    if (unlikely (row >= rows || col >= cols)) return Null (Anchor);
    auto &offset = matrixZ[row * cols + col];
    if (unlikely (!offset.sanitize (&c->sanitizer, this))) return Null (Anchor);
    *found = !offset.is_null ();
    return this+offset;
  }

  template <typename Iterator,
            hb_requires (hb_is_iterator (Iterator))>
  void collect_variation_indices (hb_collect_variation_indices_context_t *c,
                                  Iterator index_iter) const
  {
    for (unsigned i : index_iter)
      (this+matrixZ[i]).collect_variation_indices (c);
  }

  template <typename Iterator,
      hb_requires (hb_is_iterator (Iterator))>
  bool subset (hb_subset_context_t *c,
               unsigned             num_rows,
               Iterator             index_iter) const
  {
    TRACE_SUBSET (this);

    auto *out = c->serializer->start_embed (this);

    if (!index_iter) return_trace (false);
    if (unlikely (!c->serializer->extend_min (out)))  return_trace (false);

    out->rows = num_rows;
    bool ret = false;
    for (const unsigned i : index_iter)
    {
      auto *offset = c->serializer->embed (matrixZ[i]);
      if (!offset) return_trace (false);
      ret |= offset->serialize_subset (c, matrixZ[i], this);
    }

    return_trace (ret);
  }
};


}
}
}

#endif /* OT_LAYOUT_GPOS_ANCHORMATRIX_HH */

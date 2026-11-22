#ifndef OT_LAYOUT_GPOS_MARKARRAY_HH
#define OT_LAYOUT_GPOS_MARKARRAY_HH

#include "AnchorMatrix.hh"
#include "MarkRecord.hh"

namespace OT {
namespace Layout {
namespace GPOS_impl {

static inline hb_position_t
resolve_cross_offset (const hb_glyph_position_t *pos,
		      unsigned int len,
		      unsigned int glyph_pos,
		      hb_direction_t direction)
{
  bool horizontal = HB_DIRECTION_IS_HORIZONTAL (direction);
  hb_position_t offset = horizontal ? pos[glyph_pos].y_offset : pos[glyph_pos].x_offset;
  while (pos[glyph_pos].attach_type() & ATTACH_TYPE_CURSIVE)
  {
    int chain = pos[glyph_pos].attach_chain();
    if (!chain)
      break;
    unsigned int parent = (int) glyph_pos + chain;
    if (unlikely (parent >= len))
      break;
    glyph_pos = parent;
    offset = hb_saturate_add (offset,
			      horizontal ? pos[glyph_pos].y_offset : pos[glyph_pos].x_offset);
  }
  return offset;
}

static inline bool
apply_mark_attachment (hb_ot_apply_context_t *c,
                       const Anchor &mark_anchor,
                       const Anchor &glyph_anchor,
                       unsigned glyph_pos)
{
  hb_buffer_t *buffer = c->buffer;
  float mark_x, mark_y, base_x, base_y;

  buffer->unsafe_to_break (glyph_pos, buffer->idx + 1);
  mark_anchor.get_anchor (c, buffer->cur().codepoint, &mark_x, &mark_y);
  glyph_anchor.get_anchor (c, buffer->info[glyph_pos].codepoint, &base_x, &base_y);

  if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
  {
    c->buffer->message (c->font,
			  "attaching mark glyph at %u to glyph at %u",
			  c->buffer->idx, glyph_pos);
  }

  hb_position_t base_offset = resolve_cross_offset (buffer->pos, buffer->len, glyph_pos,
						      buffer->props.direction);

  hb_glyph_position_t &mark = buffer->cur_pos();
  mark.attach_chain() = (int) glyph_pos - (int) buffer->idx;
  if (mark.attach_chain() != (int) glyph_pos - (int) buffer->idx)
  {
    mark.attach_chain() = 0;
    goto overflow;
  }
  mark.attach_type() = ATTACH_TYPE_MARK;
  mark.x_offset = roundf (base_x - mark_x);
  mark.y_offset = roundf (base_y - mark_y);
  if (HB_DIRECTION_IS_HORIZONTAL (buffer->props.direction))
    mark.y_offset = hb_saturate_add (mark.y_offset, base_offset);
  else
    mark.x_offset = hb_saturate_add (mark.x_offset, base_offset);
  buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_GPOS_ATTACHMENT;

  if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
  {
    c->buffer->message (c->font,
			  "attached mark glyph at %u to glyph at %u",
			  c->buffer->idx, glyph_pos);
  }

overflow:
  buffer->idx++;
  return true;
}

HB_INTERNAL inline
void collect_mark_variation_indices (hb_collect_variation_indices_context_t *c,
                                    const Coverage &coverage,
                                    const void *base,
                                    glyph_data_t anchors,
                                    unsigned count,
                                    unsigned record_size)
{
  for (const auto _ : hb_zip (coverage, hb_range (count)))
  {
    if (!c->glyph_set->has (_.first)) continue;
    glyph_data_t anchor = anchors;
    anchor.data = (const char *) anchors.data + _.second * record_size;
    unsigned offset = anchor[0];
    if (offset)
      StructAtOffset<const Anchor> (base, offset).collect_variation_indices (c);
  }
}

template <typename Types>
struct MarkArray : Types::template ArrayOf<MarkRecord<Types>>
                                                /* Array of MarkRecords--in Coverage order */
{
  typedef typename Types::template ArrayOf<MarkRecord<Types>> Super;

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (Super::sanitize (c, this));
  }

  void collect_variation_indices (hb_collect_variation_indices_context_t *c,
                                  const Coverage &coverage) const
  {
    if (!this->len) return;
    collect_mark_variation_indices (c, coverage, this,
                                   &this->arrayZ[0].markAnchor,
                                   this->len, MarkRecord<Types>::static_size);
  }

  template <typename AnchorMatrixType>
  bool apply (hb_ot_apply_context_t *c,
              unsigned int mark_index, unsigned int glyph_index,
              const AnchorMatrixType &anchors, unsigned int class_count,
              unsigned int glyph_pos) const
  {
    TRACE_APPLY (this);
    const MarkRecord<Types> &record = Super::operator[](mark_index);
    unsigned int mark_class = record.klass;

    const Anchor& mark_anchor = this + record.markAnchor;
    bool found;
    const Anchor& glyph_anchor = anchors.get_anchor (c, glyph_index, mark_class, class_count, &found);
    /* If this subtable doesn't have an anchor for this base and this class,
     * return false such that the subsequent subtables have a chance at it. */
    if (unlikely (!found)) return_trace (false);

    return_trace (apply_mark_attachment (c, mark_anchor, glyph_anchor, glyph_pos));
  }

  template <typename Iterator,
      hb_requires (hb_is_iterator (Iterator))>
  bool subset (hb_subset_context_t *c,
               Iterator             coverage,
               const hb_map_t      *klass_mapping) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();

    auto* out = c->serializer->start_embed (this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);

    auto mark_iter =
    + hb_zip (coverage, this->iter ())
    | hb_filter (glyphset, hb_first)
    | hb_map (hb_second)
    ;

    bool ret = false;
    unsigned new_length = 0;
    for (const auto& mark_record : mark_iter) {
      ret |= mark_record.subset (c, this, klass_mapping);
      new_length++;
    }

    if (unlikely (!c->serializer->check_assign (out->len, new_length,
                                                HB_SERIALIZE_ERROR_ARRAY_OVERFLOW)))
      return_trace (false);

    return_trace (ret);
  }
};

HB_INTERNAL inline
void Markclass_closure_and_remap_indexes (const Coverage &mark_coverage,
                                    const void *mark_records,
                                    unsigned count,
                                    unsigned record_size,
                                    const hb_set_t &glyphset,
                                    hb_map_t *klass_mapping)
{
  hb_set_t orig_classes;
  for (const auto _ : hb_zip (mark_coverage, hb_range (count)))
    if (glyphset.has (_.first))
    {
      const auto *klass = reinterpret_cast<const HBUINT16 *>
        ((const char *) mark_records + _.second * record_size);
      orig_classes.add (*klass);
    }

  unsigned idx = 0;
  for (auto klass : orig_classes.iter ())
  {
    if (klass_mapping->has (klass)) continue;
    klass_mapping->set (klass, idx);
    idx++;
  }
}

template <typename Types>
HB_INTERNAL inline
void Markclass_closure_and_remap_indexes (const Coverage     &mark_coverage,
                                          const MarkArray<Types> &mark_array,
                                          const hb_set_t     &glyphset,
                                          hb_map_t*           klass_mapping /* INOUT */)
{
  Markclass_closure_and_remap_indexes (mark_coverage,
                                       mark_array.arrayZ, mark_array.len,
                                       MarkRecord<Types>::static_size,
                                       glyphset, klass_mapping);
}

}
}
}

#endif /* OT_LAYOUT_GPOS_MARKARRAY_HH */

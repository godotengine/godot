#ifndef OT_LAYOUT_GPOS_PAIRPOSFORMAT2_HH
#define OT_LAYOUT_GPOS_PAIRPOSFORMAT2_HH

#include "ValueFormat.hh"

namespace OT {
namespace Layout {
namespace GPOS_impl {

HB_INTERNAL inline
void collect_pair_variation_indices (hb_collect_variation_indices_context_t *c,
                                     const ValueBase *base,
                                     const Coverage &coverage,
                                     const ClassDef &class_def1,
                                     const ClassDef &class_def2,
                                     const ValueFormat &valueFormat1,
                                     const ValueFormat &valueFormat2,
                                     unsigned class1_count,
                                     unsigned class2_count,
                                     const ValueRecord &values)
{
  if (!coverage.intersects (c->glyph_set) ||
      !class_def2.intersects (c->glyph_set)) return;
  if ((!valueFormat1.has_device ()) && (!valueFormat2.has_device ())) return;

  hb_set_t klass1_glyphs, klass2_glyphs;
  if (!class_def1.collect_coverage (&klass1_glyphs)) return;
  if (!class_def2.collect_coverage (&klass2_glyphs)) return;

  hb_set_t class1_set, class2_set;
  for (const unsigned cp : + c->glyph_set->iter () | hb_filter (coverage))
  {
    if (!klass1_glyphs.has (cp)) class1_set.add (0);
    else
    {
      unsigned klass1 = class_def1.get (cp);
      class1_set.add (klass1);
    }
  }

  class2_set.add (0);
  for (const unsigned cp : + c->glyph_set->iter () | hb_filter (klass2_glyphs))
  {
    unsigned klass2 = class_def2.get (cp);
    class2_set.add (klass2);
  }

  if (class1_set.is_empty ()
      || class2_set.is_empty ()
      || (class2_set.get_population() == 1 && class2_set.has(0)))
    return;

  unsigned len1 = valueFormat1.get_len ();
  unsigned len2 = valueFormat2.get_len ();
  const hb_array_t<const Value> values_array = values.as_array ((unsigned)class1_count * (unsigned) class2_count * (len1 + len2));
  for (const unsigned class1_idx : class1_set.iter ())
  {
    for (const unsigned class2_idx : class2_set.iter ())
    {
      unsigned start_offset = (class1_idx * (unsigned) class2_count + class2_idx) * (len1 + len2);
      if (valueFormat1.has_device ())
        valueFormat1.collect_variation_indices (c, base, values_array.sub_array (start_offset, len1));

      if (valueFormat2.has_device ())
        valueFormat2.collect_variation_indices (c, base, values_array.sub_array (start_offset+len1, len2));
    }
  }
}

/* Keep class-matrix traversal independent of the source table width. */
struct pair_pos_class_values_t
{
  const ValueBase *base;
  const ValueRecord &values;
  const ValueFormat &valueFormat1;
  const ValueFormat &valueFormat2;
  unsigned class1_count;
  unsigned class2_count;

  hb_pair_t<unsigned, unsigned> compute_effective_value_formats (const hb_map_t &klass1_map,
                                                               const hb_map_t &klass2_map,
                                                               bool strip_hints, bool strip_empty,
                                                               const hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> *varidx_delta_map = nullptr) const
  {
    unsigned len1 = valueFormat1.get_len ();
    unsigned len2 = valueFormat2.get_len ();
    unsigned record_size = len1 + len2;

    unsigned format1 = 0;
    unsigned format2 = 0;

    for (unsigned class1_idx : + hb_range (class1_count) | hb_filter (klass1_map))
    {
      for (unsigned class2_idx : + hb_range (class2_count) | hb_filter (klass2_map))
      {
        unsigned idx = (class1_idx * class2_count + class2_idx) * record_size;
        format1 = format1 | valueFormat1.get_effective_format (&values[idx], strip_hints, strip_empty, base, varidx_delta_map);
        format2 = format2 | valueFormat2.get_effective_format (&values[idx + len1], strip_hints, strip_empty, base, varidx_delta_map);
      }

      if (!varidx_delta_map &&
          format1 == valueFormat1 && format2 == valueFormat2)
        break;
    }

    return hb_pair (format1, format2);
  }

  bool serialize (hb_subset_context_t *c,
                  const hb_map_t &klass1_map,
                  const hb_map_t &klass2_map,
                  ValueFormat newFormat1,
                  ValueFormat newFormat2) const
  {
    unsigned len1 = valueFormat1.get_len ();
    unsigned len2 = valueFormat2.get_len ();
    unsigned total_len = len1 + len2;
    hb_vector_t<unsigned> class2_idxs (+ hb_range (class2_count) | hb_filter (klass2_map));
    if (unlikely (!c->serializer->propagate_error (class2_idxs)))
      return false;
    for (unsigned class1_idx : + hb_range (class1_count) | hb_filter (klass1_map))
    {
      for (unsigned class2_idx : class2_idxs)
      {
        unsigned idx = (class1_idx * class2_count + class2_idx) * total_len;
        valueFormat1.copy_values (c->serializer, newFormat1, base, &values[idx], &c->plan->layout_variation_idx_delta_map);
        valueFormat2.copy_values (c->serializer, newFormat2, base, &values[idx + len1], &c->plan->layout_variation_idx_delta_map);
      }
    }

    return true;
  }
};

template <typename Types>
struct PairPosFormat2_4 : ValueBase
{
  protected:
  HBUINT16      format;                 /* Format identifier--format = 2 */
  typename Types::template LOffsetTo<Coverage>
                coverage;               /* Offset to Coverage table--from
                                         * beginning of subtable */
  ValueFormat   valueFormat1;           /* ValueRecord definition--for the
                                         * first glyph of the pair--may be zero
                                         * (0) */
  ValueFormat   valueFormat2;           /* ValueRecord definition--for the
                                         * second glyph of the pair--may be
                                         * zero (0) */
  typename Types::template LOffsetTo<ClassDef>
                classDef1;              /* Offset to ClassDef table--from
                                         * beginning of PairPos subtable--for
                                         * the first glyph of the pair */
  typename Types::template LOffsetTo<ClassDef>
                classDef2;              /* Offset to ClassDef table--from
                                         * beginning of PairPos subtable--for
                                         * the second glyph of the pair */
  HBUINT16      class1Count;            /* Number of classes in ClassDef1
                                         * table--includes Class0 */
  HBUINT16      class2Count;            /* Number of classes in ClassDef2
                                         * table--includes Class0 */
  ValueRecord   values;                 /* Matrix of value pairs:
                                         * class1-major, class2-minor,
                                         * Each entry has value1 and value2 */
  public:
  DEFINE_SIZE_ARRAY (10 + 3 * Types::LOffset::static_size, values);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!(c->check_struct (this)
       && coverage.sanitize (c, this)
       && classDef1.sanitize (c, this)
       && classDef2.sanitize (c, this))) return_trace (false);

    unsigned int len1 = valueFormat1.get_len ();
    unsigned int len2 = valueFormat2.get_len ();
    unsigned int stride = HBUINT16::static_size * (len1 + len2);
    unsigned int count = (unsigned int) class1Count * (unsigned int) class2Count;
    return_trace (c->check_range ((const void *) values,
                                  count,
                                  stride) &&
		  (c->lazy_some_gpos ||
		   (valueFormat1.sanitize_values_stride_unsafe (c, this, &values[0], count, stride) &&
		    valueFormat2.sanitize_values_stride_unsafe (c, this, &values[len1], count, stride))));
  }

  bool intersects (const hb_set_t *glyphs) const
  {
    return (this+coverage).intersects (glyphs) &&
           (this+classDef2).intersects (glyphs);
  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}
  void collect_variation_indices (hb_collect_variation_indices_context_t *c) const
  {
    collect_pair_variation_indices (c, this, this+coverage,
                                    this+classDef1, this+classDef2,
                                    valueFormat1, valueFormat2,
                                    class1Count, class2Count, values);
  }

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+coverage).collect_coverage (c->input))) return;
    if (unlikely (!(this+classDef2).collect_coverage (c->input))) return;
  }

  const Coverage &get_coverage () const { return this+coverage; }

  struct external_cache_t
  {
    hb_ot_layout_mapping_cache_t coverage;
    hb_ot_layout_mapping_cache_t first;
    hb_ot_layout_mapping_cache_t second;
  };
  void *external_cache_create () const
  {
    external_cache_t *cache = (external_cache_t *) hb_malloc (sizeof (external_cache_t));
    if (likely (cache))
    {
      cache->coverage.clear ();
      cache->first.clear ();
      cache->second.clear ();
    }
    return cache;
  }

  bool apply (hb_ot_apply_context_t *c, void *external_cache) const
  {
    TRACE_APPLY (this);

    hb_buffer_t *buffer = c->buffer;

#ifndef HB_NO_OT_LAYOUT_LOOKUP_CACHE
    external_cache_t *cache = (external_cache_t *) external_cache;
    unsigned int index = (this+coverage).get_coverage  (buffer->cur().codepoint, cache ? &cache->coverage : nullptr);
#else
    unsigned int index = (this+coverage).get_coverage  (buffer->cur().codepoint);
#endif
    if (index == NOT_COVERED) return_trace (false);

    auto &skippy_iter = c->iter_input;
    skippy_iter.reset_fast (buffer->idx);
    unsigned unsafe_to;
    if (unlikely (!skippy_iter.next (&unsafe_to)))
    {
      buffer->unsafe_to_concat (buffer->idx, unsafe_to);
      return_trace (false);
    }

#ifndef HB_NO_OT_LAYOUT_LOOKUP_CACHE
    unsigned int klass1 = (this+classDef1).get_class (buffer->cur().codepoint, cache ? &cache->first : nullptr);
    unsigned int klass2 = (this+classDef2).get_class (buffer->info[skippy_iter.idx].codepoint, cache ? &cache->second : nullptr);
#else
    unsigned int klass1 = (this+classDef1).get_class (buffer->cur().codepoint);
    unsigned int klass2 = (this+classDef2).get_class (buffer->info[skippy_iter.idx].codepoint);
#endif
    if (unlikely (klass1 >= class1Count || klass2 >= class2Count))
    {
      buffer->unsafe_to_concat (buffer->idx, skippy_iter.idx + 1);
      return_trace (false);
    }

    unsigned int len1 = valueFormat1.get_len ();
    unsigned int len2 = valueFormat2.get_len ();
    unsigned int record_len = len1 + len2;

    const Value *v = &values[record_len * (klass1 * class2Count + klass2)];

    bool applied_first = false, applied_second = false;


    /* Isolate simple kerning and apply it half to each side.
     * Results in better cursor positioning / underline drawing.
     *
     * Disabled, because causes issues... :-(
     * https://github.com/harfbuzz/harfbuzz/issues/3408
     * https://github.com/harfbuzz/harfbuzz/pull/3235#issuecomment-1029814978
     */
#ifndef HB_SPLIT_KERN
    if (false)
#endif
    {
      if (!len2)
      {
        const hb_direction_t dir = buffer->props.direction;
        const bool horizontal = HB_DIRECTION_IS_HORIZONTAL (dir);
        const bool backward = HB_DIRECTION_IS_BACKWARD (dir);
        unsigned mask = horizontal ? ValueFormat::xAdvance : ValueFormat::yAdvance;
        if (backward)
          mask |= mask >> 2; /* Add eg. xPlacement in RTL. */
        /* Add Devices. */
        mask |= mask << 4;

        if (valueFormat1 & ~mask)
          goto bail;

        /* Is simple kern. Apply value on an empty position slot,
         * then split it between sides. */

        hb_glyph_position_t pos{};
        if (valueFormat1.apply_value (c, this, v, pos))
        {
          hb_position_t *src  = &pos.x_advance;
          hb_position_t *dst1 = &buffer->cur_pos().x_advance;
          hb_position_t *dst2 = &buffer->pos[skippy_iter.idx].x_advance;
          unsigned i = horizontal ? 0 : 1;

          hb_position_t kern  = src[i];
          hb_position_t kern1 = kern >> 1;
          hb_position_t kern2 = kern - kern1;

          if (!backward)
          {
            dst1[i] = hb_saturate_add (dst1[i], kern1);
            dst2[i] = hb_saturate_add (dst2[i], kern2);
            dst2[i + 2] = hb_saturate_add (dst2[i + 2], kern2);
          }
          else
          {
            dst1[i] = hb_saturate_add (dst1[i], kern1);
            dst1[i + 2] = hb_clamp_to<hb_position_t> ((int64_t) dst1[i + 2] + src[i + 2] - kern2);
            dst2[i] = hb_saturate_add (dst2[i], kern2);
          }

          applied_first = applied_second = kern != 0;
          goto success;
        }
        goto boring;
      }
    }
    bail:

    if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
    {
      c->buffer->message (c->font,
			  "try kerning glyphs at %u,%u",
			  c->buffer->idx, skippy_iter.idx);
    }

    applied_first = len1 && valueFormat1.apply_value (c, this, v, buffer->cur_pos());
    applied_second = len2 && valueFormat2.apply_value (c, this, v + len1, buffer->pos[skippy_iter.idx]);

    if (applied_first || applied_second)
      if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
      {
	c->buffer->message (c->font,
			    "kerned glyphs at %u,%u",
			    c->buffer->idx, skippy_iter.idx);
      }

    if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
    {
      c->buffer->message (c->font,
			  "tried kerning glyphs at %u,%u",
			  c->buffer->idx, skippy_iter.idx);
    }

    success:
    if (applied_first || applied_second)
      buffer->unsafe_to_break (buffer->idx, skippy_iter.idx + 1);
    else
    boring:
      buffer->unsafe_to_concat (buffer->idx, skippy_iter.idx + 1);

    if (len2)
    {
      skippy_iter.idx++;
      // https://github.com/harfbuzz/harfbuzz/issues/3824
      // https://github.com/harfbuzz/harfbuzz/issues/3888#issuecomment-1326781116
      buffer->unsafe_to_break (buffer->idx, skippy_iter.idx + 1);
    }

    buffer->idx = skippy_iter.idx;

    return_trace (true);
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);
    out->format = format;

    hb_map_t klass1_map;
    out->classDef1.serialize_subset (c, classDef1, this, &klass1_map, true, true, &(this + coverage));
    out->class1Count = klass1_map.get_population ();

    hb_map_t klass2_map;
    out->classDef2.serialize_subset (c, classDef2, this, &klass2_map, true, false);
    out->class2Count = klass2_map.get_population ();

    pair_pos_class_values_t class_values {this, values, valueFormat1, valueFormat2,
                                         class1Count, class2Count};

    hb_pair_t<unsigned, unsigned> newFormats = hb_pair (valueFormat1, valueFormat2);

    if (c->plan->normalized_coords)
    {
      /* in case of full instancing, all var device flags will be dropped so no
       * need to strip hints here */
      newFormats = class_values.compute_effective_value_formats (klass1_map, klass2_map, false, false, &c->plan->layout_variation_idx_delta_map);
    }
    /* do not strip hints for VF */
    else if (c->plan->flags & HB_SUBSET_FLAGS_NO_HINTING)
    {
      hb_blob_t* blob = hb_face_reference_table (c->plan->source, HB_TAG ('f','v','a','r'));
      bool has_fvar = (blob != hb_blob_get_empty ());
      hb_blob_destroy (blob);

      bool strip = !has_fvar;
      /* special case: strip hints when a VF has no GDEF varstore after
       * subsetting*/
      if (has_fvar && !c->plan->has_gdef_varstore)
        strip = true;
      newFormats = class_values.compute_effective_value_formats (klass1_map, klass2_map, strip, true);
    }

    out->valueFormat1 = newFormats.first;
    out->valueFormat2 = newFormats.second;

    if (!class_values.serialize (c, klass1_map, klass2_map,
                                 out->valueFormat1, out->valueFormat2))
      return_trace (false);

    bool ret = out->coverage.serialize_subset(c, coverage, this);
    return_trace (out->class1Count && out->class2Count && ret);
  }
};

}
}
}

#endif  // OT_LAYOUT_GPOS_PAIRPOSFORMAT2_HH

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
  public:
  Array16OfOffset16To<Ligature<Types>>
                ligature;               /* Array LigatureSet tables
                                         * ordered by preference */
  
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

  bool intersects_lig_glyph (const hb_set_t *glyphs) const
  {
    return
    + hb_iter (ligature)
    | hb_map (hb_add (this))
    | hb_map ([glyphs] (const Ligature<Types> &_) { 
      return _.intersects_lig_glyph (glyphs) && _.intersects (glyphs);
    })
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

  template <typename set_t>
  void collect_seconds (set_t &s) const
  {
    + hb_iter (ligature)
    | hb_map (hb_add (this))
    | hb_apply ([&s] (const Ligature<Types> &_) { _.collect_second (s); })
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

  bool apply (hb_ot_apply_context_t *c, const hb_set_digest_t *seconds = nullptr) const
  {
    TRACE_APPLY (this);

    unsigned int num_ligs = ligature.len;

#ifndef HB_NO_OT_RULESETS_FAST_PATH
    if (HB_OPTIMIZE_SIZE_VAL || num_ligs <= 1)
#endif
    {
    slow:
      for (unsigned int i = 0; i < num_ligs; i++)
      {
	const auto &lig = this+ligature.arrayZ[i];
	if (lig.apply (c)) return_trace (true);
      }
      return_trace (false);
    }

    /* This version is optimized for speed by matching the second component
     * of the ligature here, instead of calling into the ligation code.
     *
     * This is replicated in ChainRuleSet and RuleSet. */

    auto &skippy_iter = c->iter_input;
    skippy_iter.reset (c->buffer->idx);
    skippy_iter.set_match_func (match_always, nullptr);
    skippy_iter.set_glyph_data ((HBUINT16 *) nullptr);
    unsigned unsafe_to;
    hb_codepoint_t second = (unsigned) -1;
    bool matched = skippy_iter.next (&unsafe_to);
    if (likely (matched))
    {
      second = c->buffer->info[skippy_iter.idx].codepoint;
      unsafe_to = skippy_iter.idx + 1;

      if (skippy_iter.may_skip (c->buffer->info[skippy_iter.idx]))
      {
	/* Can't use the fast path if eg. the next char is a default-ignorable
	 * or other skippable. */
        goto slow;
      }
    }
    else
      goto slow;

    if (seconds && !seconds->may_have (second))
      return_trace (false);
    bool unsafe_to_concat = false;
    for (unsigned int i = 0; i < num_ligs; i++)
    {
      const auto &lig = this+ligature.arrayZ[i];
      if (unlikely (lig.component.lenP1 <= 1) ||
	  lig.component.arrayZ[0] == second)
      {
	if (lig.apply (c))
	{
	  if (unsafe_to_concat)
	    c->buffer->unsafe_to_concat (c->buffer->idx, unsafe_to);
	  return_trace (true);
	}
      }
      else if (likely (lig.component.lenP1 > 1))
        unsafe_to_concat = true;
    }
    if (likely (unsafe_to_concat))
      c->buffer->unsafe_to_concat (c->buffer->idx, unsafe_to);

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

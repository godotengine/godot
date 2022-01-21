/*
 * Copyright © 2007,2008,2009,2010  Red Hat, Inc.
 * Copyright © 2010,2012,2013  Google, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Red Hat Author(s): Behdad Esfahbod
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_OT_LAYOUT_GSUB_TABLE_HH
#define HB_OT_LAYOUT_GSUB_TABLE_HH

#include "hb-ot-layout-gsubgpos.hh"


namespace OT {

typedef hb_pair_t<hb_codepoint_t, hb_codepoint_t> hb_codepoint_pair_t;

template<typename Iterator>
static void SingleSubst_serialize (hb_serialize_context_t *c,
				   Iterator it);


struct SingleSubstFormat1
{
  bool intersects (const hb_set_t *glyphs) const
  { return (this+coverage).intersects (glyphs); }

  bool may_have_non_1to1 () const
  { return false; }

  void closure (hb_closure_context_t *c) const
  {
    unsigned d = deltaGlyphID;

    + hb_iter (this+coverage)
    | hb_filter (c->parent_active_glyphs ())
    | hb_map ([d] (hb_codepoint_t g) { return (g + d) & 0xFFFFu; })
    | hb_sink (c->output)
    ;

  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+coverage).collect_coverage (c->input))) return;
    unsigned d = deltaGlyphID;
    + hb_iter (this+coverage)
    | hb_map ([d] (hb_codepoint_t g) { return (g + d) & 0xFFFFu; })
    | hb_sink (c->output)
    ;
  }

  const Coverage &get_coverage () const { return this+coverage; }

  bool would_apply (hb_would_apply_context_t *c) const
  { return c->len == 1 && (this+coverage).get_coverage (c->glyphs[0]) != NOT_COVERED; }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_codepoint_t glyph_id = c->buffer->cur().codepoint;
    unsigned int index = (this+coverage).get_coverage (glyph_id);
    if (likely (index == NOT_COVERED)) return_trace (false);

    /* According to the Adobe Annotated OpenType Suite, result is always
     * limited to 16bit. */
    glyph_id = (glyph_id + deltaGlyphID) & 0xFFFFu;
    c->replace_glyph (glyph_id);

    return_trace (true);
  }

  template<typename Iterator,
	   hb_requires (hb_is_sorted_source_of (Iterator, hb_codepoint_t))>
  bool serialize (hb_serialize_context_t *c,
		  Iterator glyphs,
		  unsigned delta)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);
    if (unlikely (!coverage.serialize_serialize (c, glyphs))) return_trace (false);
    c->check_assign (deltaGlyphID, delta, HB_SERIALIZE_ERROR_INT_OVERFLOW);
    return_trace (true);
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    hb_codepoint_t delta = deltaGlyphID;

    auto it =
    + hb_iter (this+coverage)
    | hb_filter (glyphset)
    | hb_map_retains_sorting ([&] (hb_codepoint_t g) {
				return hb_codepoint_pair_t (g,
							    (g + delta) & 0xFFFF); })
    | hb_filter (glyphset, hb_second)
    | hb_map_retains_sorting ([&] (hb_codepoint_pair_t p) -> hb_codepoint_pair_t
			      { return hb_pair (glyph_map[p.first], glyph_map[p.second]); })
    ;

    bool ret = bool (it);
    SingleSubst_serialize (c->serializer, it);
    return_trace (ret);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && deltaGlyphID.sanitize (c));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  Offset16To<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of Substitution table */
  HBUINT16	deltaGlyphID;		/* Add to original GlyphID to get
					 * substitute GlyphID, modulo 0x10000 */
  public:
  DEFINE_SIZE_STATIC (6);
};

struct SingleSubstFormat2
{
  bool intersects (const hb_set_t *glyphs) const
  { return (this+coverage).intersects (glyphs); }

  bool may_have_non_1to1 () const
  { return false; }

  void closure (hb_closure_context_t *c) const
  {
    + hb_zip (this+coverage, substitute)
    | hb_filter (c->parent_active_glyphs (), hb_first)
    | hb_map (hb_second)
    | hb_sink (c->output)
    ;

  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+coverage).collect_coverage (c->input))) return;
    + hb_zip (this+coverage, substitute)
    | hb_map (hb_second)
    | hb_sink (c->output)
    ;
  }

  const Coverage &get_coverage () const { return this+coverage; }

  bool would_apply (hb_would_apply_context_t *c) const
  { return c->len == 1 && (this+coverage).get_coverage (c->glyphs[0]) != NOT_COVERED; }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    unsigned int index = (this+coverage).get_coverage (c->buffer->cur().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    if (unlikely (index >= substitute.len)) return_trace (false);

    c->replace_glyph (substitute[index]);

    return_trace (true);
  }

  template<typename Iterator,
	   hb_requires (hb_is_sorted_source_of (Iterator,
						hb_codepoint_pair_t))>
  bool serialize (hb_serialize_context_t *c,
		  Iterator it)
  {
    TRACE_SERIALIZE (this);
    auto substitutes =
      + it
      | hb_map (hb_second)
      ;
    auto glyphs =
      + it
      | hb_map_retains_sorting (hb_first)
      ;
    if (unlikely (!c->extend_min (this))) return_trace (false);
    if (unlikely (!substitute.serialize (c, substitutes))) return_trace (false);
    if (unlikely (!coverage.serialize_serialize (c, glyphs))) return_trace (false);
    return_trace (true);
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    auto it =
    + hb_zip (this+coverage, substitute)
    | hb_filter (glyphset, hb_first)
    | hb_filter (glyphset, hb_second)
    | hb_map_retains_sorting ([&] (hb_pair_t<hb_codepoint_t, const HBGlyphID16 &> p) -> hb_codepoint_pair_t
			      { return hb_pair (glyph_map[p.first], glyph_map[p.second]); })
    ;

    bool ret = bool (it);
    SingleSubst_serialize (c->serializer, it);
    return_trace (ret);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && substitute.sanitize (c));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 2 */
  Offset16To<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of Substitution table */
  Array16Of<HBGlyphID16>
		substitute;		/* Array of substitute
					 * GlyphIDs--ordered by Coverage Index */
  public:
  DEFINE_SIZE_ARRAY (6, substitute);
};

struct SingleSubst
{

  template<typename Iterator,
	   hb_requires (hb_is_sorted_source_of (Iterator,
						const hb_codepoint_pair_t))>
  bool serialize (hb_serialize_context_t *c,
		  Iterator glyphs)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (u.format))) return_trace (false);
    unsigned format = 2;
    unsigned delta = 0;
    if (glyphs)
    {
      format = 1;
      auto get_delta = [=] (hb_codepoint_pair_t _)
		       { return (unsigned) (_.second - _.first) & 0xFFFF; };
      delta = get_delta (*glyphs);
      if (!hb_all (++(+glyphs), delta, get_delta)) format = 2;
    }
    u.format = format;
    switch (u.format) {
    case 1: return_trace (u.format1.serialize (c,
					       + glyphs
					       | hb_map_retains_sorting (hb_first),
					       delta));
    case 2: return_trace (u.format2.serialize (c, glyphs));
    default:return_trace (false);
    }
  }

  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, Ts&&... ds) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1, std::forward<Ts> (ds)...));
    case 2: return_trace (c->dispatch (u.format2, std::forward<Ts> (ds)...));
    default:return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT16		format;		/* Format identifier */
  SingleSubstFormat1	format1;
  SingleSubstFormat2	format2;
  } u;
};

template<typename Iterator>
static void
SingleSubst_serialize (hb_serialize_context_t *c,
		       Iterator it)
{ c->start_embed<SingleSubst> ()->serialize (c, it); }

struct Sequence
{
  bool intersects (const hb_set_t *glyphs) const
  { return hb_all (substitute, glyphs); }

  void closure (hb_closure_context_t *c) const
  { c->output->add_array (substitute.arrayZ, substitute.len); }

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  { c->output->add_array (substitute.arrayZ, substitute.len); }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    unsigned int count = substitute.len;

    /* Special-case to make it in-place and not consider this
     * as a "multiplied" substitution. */
    if (unlikely (count == 1))
    {
      c->replace_glyph (substitute.arrayZ[0]);
      return_trace (true);
    }
    /* Spec disallows this, but Uniscribe allows it.
     * https://github.com/harfbuzz/harfbuzz/issues/253 */
    else if (unlikely (count == 0))
    {
      c->buffer->delete_glyph ();
      return_trace (true);
    }

    unsigned int klass = _hb_glyph_info_is_ligature (&c->buffer->cur()) ?
			 HB_OT_LAYOUT_GLYPH_PROPS_BASE_GLYPH : 0;
    unsigned lig_id = _hb_glyph_info_get_lig_id (&c->buffer->cur());

    for (unsigned int i = 0; i < count; i++)
    {
      /* If is attached to a ligature, don't disturb that.
       * https://github.com/harfbuzz/harfbuzz/issues/3069 */
      if (!lig_id)
	_hb_glyph_info_set_lig_props_for_component (&c->buffer->cur(), i);
      c->output_glyph_for_component (substitute.arrayZ[i], klass);
    }
    c->buffer->skip_glyph ();

    return_trace (true);
  }

  template <typename Iterator,
	    hb_requires (hb_is_source_of (Iterator, hb_codepoint_t))>
  bool serialize (hb_serialize_context_t *c,
		  Iterator subst)
  {
    TRACE_SERIALIZE (this);
    return_trace (substitute.serialize (c, subst));
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    if (!intersects (&glyphset)) return_trace (false);

    auto it =
    + hb_iter (substitute)
    | hb_map (glyph_map)
    ;

    auto *out = c->serializer->start_embed (*this);
    return_trace (out->serialize (c->serializer, it));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (substitute.sanitize (c));
  }

  protected:
  Array16Of<HBGlyphID16>
		substitute;		/* String of GlyphIDs to substitute */
  public:
  DEFINE_SIZE_ARRAY (2, substitute);
};

struct MultipleSubstFormat1
{
  bool intersects (const hb_set_t *glyphs) const
  { return (this+coverage).intersects (glyphs); }

  bool may_have_non_1to1 () const
  { return true; }

  void closure (hb_closure_context_t *c) const
  {
    + hb_zip (this+coverage, sequence)
    | hb_filter (c->parent_active_glyphs (), hb_first)
    | hb_map (hb_second)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const Sequence &_) { _.closure (c); })
    ;
  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+coverage).collect_coverage (c->input))) return;
    + hb_zip (this+coverage, sequence)
    | hb_map (hb_second)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const Sequence &_) { _.collect_glyphs (c); })
    ;
  }

  const Coverage &get_coverage () const { return this+coverage; }

  bool would_apply (hb_would_apply_context_t *c) const
  { return c->len == 1 && (this+coverage).get_coverage (c->glyphs[0]) != NOT_COVERED; }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);

    unsigned int index = (this+coverage).get_coverage (c->buffer->cur().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    return_trace ((this+sequence[index]).apply (c));
  }

  bool serialize (hb_serialize_context_t *c,
		  hb_sorted_array_t<const HBGlyphID16> glyphs,
		  hb_array_t<const unsigned int> substitute_len_list,
		  hb_array_t<const HBGlyphID16> substitute_glyphs_list)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);
    if (unlikely (!sequence.serialize (c, glyphs.length))) return_trace (false);
    for (unsigned int i = 0; i < glyphs.length; i++)
    {
      unsigned int substitute_len = substitute_len_list[i];
      if (unlikely (!sequence[i]
                        .serialize_serialize (c, substitute_glyphs_list.sub_array (0, substitute_len))))
	return_trace (false);
      substitute_glyphs_list += substitute_len;
    }
    return_trace (coverage.serialize_serialize (c, glyphs));
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);
    out->format = format;

    hb_sorted_vector_t<hb_codepoint_t> new_coverage;
    + hb_zip (this+coverage, sequence)
    | hb_filter (glyphset, hb_first)
    | hb_filter (subset_offset_array (c, out->sequence, this), hb_second)
    | hb_map (hb_first)
    | hb_map (glyph_map)
    | hb_sink (new_coverage)
    ;
    out->coverage.serialize_serialize (c->serializer, new_coverage.iter ());
    return_trace (bool (new_coverage));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && sequence.sanitize (c, this));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  Offset16To<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of Substitution table */
  Array16OfOffset16To<Sequence>
		sequence;		/* Array of Sequence tables
					 * ordered by Coverage Index */
  public:
  DEFINE_SIZE_ARRAY (6, sequence);
};

struct MultipleSubst
{
  bool serialize (hb_serialize_context_t *c,
		  hb_sorted_array_t<const HBGlyphID16> glyphs,
		  hb_array_t<const unsigned int> substitute_len_list,
		  hb_array_t<const HBGlyphID16> substitute_glyphs_list)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (u.format))) return_trace (false);
    unsigned int format = 1;
    u.format = format;
    switch (u.format) {
    case 1: return_trace (u.format1.serialize (c, glyphs, substitute_len_list, substitute_glyphs_list));
    default:return_trace (false);
    }
  }

  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, Ts&&... ds) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1, std::forward<Ts> (ds)...));
    default:return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT16		format;		/* Format identifier */
  MultipleSubstFormat1	format1;
  } u;
};

struct AlternateSet
{
  bool intersects (const hb_set_t *glyphs) const
  { return hb_any (alternates, glyphs); }

  void closure (hb_closure_context_t *c) const
  { c->output->add_array (alternates.arrayZ, alternates.len); }

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  { c->output->add_array (alternates.arrayZ, alternates.len); }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    unsigned int count = alternates.len;

    if (unlikely (!count)) return_trace (false);

    hb_mask_t glyph_mask = c->buffer->cur().mask;
    hb_mask_t lookup_mask = c->lookup_mask;

    /* Note: This breaks badly if two features enabled this lookup together. */
    unsigned int shift = hb_ctz (lookup_mask);
    unsigned int alt_index = ((lookup_mask & glyph_mask) >> shift);

    /* If alt_index is MAX_VALUE, randomize feature if it is the rand feature. */
    if (alt_index == HB_OT_MAP_MAX_VALUE && c->random)
    {
      /* Maybe we can do better than unsafe-to-break all; but since we are
       * changing random state, it would be hard to track that.  Good 'nough. */
      c->buffer->unsafe_to_break (0, c->buffer->len);
      alt_index = c->random_number () % count + 1;
    }

    if (unlikely (alt_index > count || alt_index == 0)) return_trace (false);

    c->replace_glyph (alternates[alt_index - 1]);

    return_trace (true);
  }

  unsigned
  get_alternates (unsigned        start_offset,
		  unsigned       *alternate_count  /* IN/OUT.  May be NULL. */,
		  hb_codepoint_t *alternate_glyphs /* OUT.     May be NULL. */) const
  {
    if (alternates.len && alternate_count)
    {
      + alternates.sub_array (start_offset, alternate_count)
      | hb_sink (hb_array (alternate_glyphs, *alternate_count))
      ;
    }
    return alternates.len;
  }

  template <typename Iterator,
	    hb_requires (hb_is_source_of (Iterator, hb_codepoint_t))>
  bool serialize (hb_serialize_context_t *c,
		  Iterator alts)
  {
    TRACE_SERIALIZE (this);
    return_trace (alternates.serialize (c, alts));
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    auto it =
      + hb_iter (alternates)
      | hb_filter (glyphset)
      | hb_map (glyph_map)
      ;

    auto *out = c->serializer->start_embed (*this);
    return_trace (out->serialize (c->serializer, it) &&
		  out->alternates);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (alternates.sanitize (c));
  }

  protected:
  Array16Of<HBGlyphID16>
		alternates;		/* Array of alternate GlyphIDs--in
					 * arbitrary order */
  public:
  DEFINE_SIZE_ARRAY (2, alternates);
};

struct AlternateSubstFormat1
{
  bool intersects (const hb_set_t *glyphs) const
  { return (this+coverage).intersects (glyphs); }

  bool may_have_non_1to1 () const
  { return false; }

  void closure (hb_closure_context_t *c) const
  {
    + hb_zip (this+coverage, alternateSet)
    | hb_filter (c->parent_active_glyphs (), hb_first)
    | hb_map (hb_second)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const AlternateSet &_) { _.closure (c); })
    ;

  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+coverage).collect_coverage (c->input))) return;
    + hb_zip (this+coverage, alternateSet)
    | hb_map (hb_second)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const AlternateSet &_) { _.collect_glyphs (c); })
    ;
  }

  const Coverage &get_coverage () const { return this+coverage; }

  bool would_apply (hb_would_apply_context_t *c) const
  { return c->len == 1 && (this+coverage).get_coverage (c->glyphs[0]) != NOT_COVERED; }

  unsigned
  get_glyph_alternates (hb_codepoint_t  gid,
			unsigned        start_offset,
			unsigned       *alternate_count  /* IN/OUT.  May be NULL. */,
			hb_codepoint_t *alternate_glyphs /* OUT.     May be NULL. */) const
  { return (this+alternateSet[(this+coverage).get_coverage (gid)])
	   .get_alternates (start_offset, alternate_count, alternate_glyphs); }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);

    unsigned int index = (this+coverage).get_coverage (c->buffer->cur().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    return_trace ((this+alternateSet[index]).apply (c));
  }

  bool serialize (hb_serialize_context_t *c,
		  hb_sorted_array_t<const HBGlyphID16> glyphs,
		  hb_array_t<const unsigned int> alternate_len_list,
		  hb_array_t<const HBGlyphID16> alternate_glyphs_list)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);
    if (unlikely (!alternateSet.serialize (c, glyphs.length))) return_trace (false);
    for (unsigned int i = 0; i < glyphs.length; i++)
    {
      unsigned int alternate_len = alternate_len_list[i];
      if (unlikely (!alternateSet[i]
                        .serialize_serialize (c, alternate_glyphs_list.sub_array (0, alternate_len))))
	return_trace (false);
      alternate_glyphs_list += alternate_len;
    }
    return_trace (coverage.serialize_serialize (c, glyphs));
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);
    out->format = format;

    hb_sorted_vector_t<hb_codepoint_t> new_coverage;
    + hb_zip (this+coverage, alternateSet)
    | hb_filter (glyphset, hb_first)
    | hb_filter (subset_offset_array (c, out->alternateSet, this), hb_second)
    | hb_map (hb_first)
    | hb_map (glyph_map)
    | hb_sink (new_coverage)
    ;
    out->coverage.serialize_serialize (c->serializer, new_coverage.iter ());
    return_trace (bool (new_coverage));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && alternateSet.sanitize (c, this));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  Offset16To<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of Substitution table */
  Array16OfOffset16To<AlternateSet>
		alternateSet;		/* Array of AlternateSet tables
					 * ordered by Coverage Index */
  public:
  DEFINE_SIZE_ARRAY (6, alternateSet);
};

struct AlternateSubst
{
  bool serialize (hb_serialize_context_t *c,
		  hb_sorted_array_t<const HBGlyphID16> glyphs,
		  hb_array_t<const unsigned int> alternate_len_list,
		  hb_array_t<const HBGlyphID16> alternate_glyphs_list)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (u.format))) return_trace (false);
    unsigned int format = 1;
    u.format = format;
    switch (u.format) {
    case 1: return_trace (u.format1.serialize (c, glyphs, alternate_len_list, alternate_glyphs_list));
    default:return_trace (false);
    }
  }

  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, Ts&&... ds) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1, std::forward<Ts> (ds)...));
    default:return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT16		format;		/* Format identifier */
  AlternateSubstFormat1	format1;
  } u;
};


struct Ligature
{
  bool intersects (const hb_set_t *glyphs) const
  { return hb_all (component, glyphs); }

  void closure (hb_closure_context_t *c) const
  {
    if (!intersects (c->glyphs)) return;
    c->output->add (ligGlyph);
  }

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    c->input->add_array (component.arrayZ, component.get_length ());
    c->output->add (ligGlyph);
  }

  bool would_apply (hb_would_apply_context_t *c) const
  {
    if (c->len != component.lenP1)
      return false;

    for (unsigned int i = 1; i < c->len; i++)
      if (likely (c->glyphs[i] != component[i]))
	return false;

    return true;
  }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    unsigned int count = component.lenP1;

    if (unlikely (!count)) return_trace (false);

    /* Special-case to make it in-place and not consider this
     * as a "ligated" substitution. */
    if (unlikely (count == 1))
    {
      c->replace_glyph (ligGlyph);
      return_trace (true);
    }

    unsigned int total_component_count = 0;

    unsigned int match_length = 0;
    unsigned int match_positions[HB_MAX_CONTEXT_LENGTH];

    if (likely (!match_input (c, count,
			      &component[1],
			      match_glyph,
			      nullptr,
			      &match_length,
			      match_positions,
			      &total_component_count)))
      return_trace (false);

    ligate_input (c,
		  count,
		  match_positions,
		  match_length,
		  ligGlyph,
		  total_component_count);

    return_trace (true);
  }

  template <typename Iterator,
	    hb_requires (hb_is_source_of (Iterator, hb_codepoint_t))>
  bool serialize (hb_serialize_context_t *c,
		  hb_codepoint_t ligature,
		  Iterator components /* Starting from second */)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);
    ligGlyph = ligature;
    if (unlikely (!component.serialize (c, components))) return_trace (false);
    return_trace (true);
  }

  bool subset (hb_subset_context_t *c, unsigned coverage_idx) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    if (!intersects (&glyphset) || !glyphset.has (ligGlyph)) return_trace (false);
    // Ensure Coverage table is always packed after this.
    c->serializer->add_virtual_link (coverage_idx);

    auto it =
      + hb_iter (component)
      | hb_map (glyph_map)
      ;

    auto *out = c->serializer->start_embed (*this);
    return_trace (out->serialize (c->serializer,
				  glyph_map[ligGlyph],
				  it));
  }

  public:
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (ligGlyph.sanitize (c) && component.sanitize (c));
  }

  protected:
  HBGlyphID16	ligGlyph;		/* GlyphID of ligature to substitute */
  HeadlessArrayOf<HBGlyphID16>
		component;		/* Array of component GlyphIDs--start
					 * with the second  component--ordered
					 * in writing direction */
  public:
  DEFINE_SIZE_ARRAY (4, component);
};

struct LigatureSet
{
  bool intersects (const hb_set_t *glyphs) const
  {
    return
    + hb_iter (ligature)
    | hb_map (hb_add (this))
    | hb_map ([glyphs] (const Ligature &_) { return _.intersects (glyphs); })
    | hb_any
    ;
  }

  void closure (hb_closure_context_t *c) const
  {
    + hb_iter (ligature)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const Ligature &_) { _.closure (c); })
    ;
  }

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    + hb_iter (ligature)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const Ligature &_) { _.collect_glyphs (c); })
    ;
  }

  bool would_apply (hb_would_apply_context_t *c) const
  {
    return
    + hb_iter (ligature)
    | hb_map (hb_add (this))
    | hb_map ([c] (const Ligature &_) { return _.would_apply (c); })
    | hb_any
    ;
  }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    unsigned int num_ligs = ligature.len;
    for (unsigned int i = 0; i < num_ligs; i++)
    {
      const Ligature &lig = this+ligature[i];
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

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (ligature.sanitize (c, this));
  }

  protected:
  Array16OfOffset16To<Ligature>
		ligature;		/* Array LigatureSet tables
					 * ordered by preference */
  public:
  DEFINE_SIZE_ARRAY (2, ligature);
};

struct LigatureSubstFormat1
{
  bool intersects (const hb_set_t *glyphs) const
  {
    return
    + hb_zip (this+coverage, ligatureSet)
    | hb_filter (*glyphs, hb_first)
    | hb_map (hb_second)
    | hb_map ([this, glyphs] (const Offset16To<LigatureSet> &_)
	      { return (this+_).intersects (glyphs); })
    | hb_any
    ;
  }

  bool may_have_non_1to1 () const
  { return true; }

  void closure (hb_closure_context_t *c) const
  {
    + hb_zip (this+coverage, ligatureSet)
    | hb_filter (c->parent_active_glyphs (), hb_first)
    | hb_map (hb_second)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const LigatureSet &_) { _.closure (c); })
    ;

  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+coverage).collect_coverage (c->input))) return;

    + hb_zip (this+coverage, ligatureSet)
    | hb_map (hb_second)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const LigatureSet &_) { _.collect_glyphs (c); })
    ;
  }

  const Coverage &get_coverage () const { return this+coverage; }

  bool would_apply (hb_would_apply_context_t *c) const
  {
    unsigned int index = (this+coverage).get_coverage (c->glyphs[0]);
    if (likely (index == NOT_COVERED)) return false;

    const LigatureSet &lig_set = this+ligatureSet[index];
    return lig_set.would_apply (c);
  }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);

    unsigned int index = (this+coverage).get_coverage (c->buffer->cur ().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    const LigatureSet &lig_set = this+ligatureSet[index];
    return_trace (lig_set.apply (c));
  }

  bool serialize (hb_serialize_context_t *c,
		  hb_sorted_array_t<const HBGlyphID16> first_glyphs,
		  hb_array_t<const unsigned int> ligature_per_first_glyph_count_list,
		  hb_array_t<const HBGlyphID16> ligatures_list,
		  hb_array_t<const unsigned int> component_count_list,
		  hb_array_t<const HBGlyphID16> component_list /* Starting from second for each ligature */)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);
    if (unlikely (!ligatureSet.serialize (c, first_glyphs.length))) return_trace (false);
    for (unsigned int i = 0; i < first_glyphs.length; i++)
    {
      unsigned int ligature_count = ligature_per_first_glyph_count_list[i];
      if (unlikely (!ligatureSet[i]
                        .serialize_serialize (c,
                                              ligatures_list.sub_array (0, ligature_count),
                                              component_count_list.sub_array (0, ligature_count),
                                              component_list))) return_trace (false);
      ligatures_list += ligature_count;
      component_count_list += ligature_count;
    }
    return_trace (coverage.serialize_serialize (c, first_glyphs));
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);
    out->format = format;

    // Due to a bug in some older versions of windows 7 the Coverage table must be
    // packed after the LigatureSet and Ligature tables, so serialize Coverage first
    // which places it last in the packed order.
    hb_set_t new_coverage;
    + hb_zip (this+coverage, hb_iter (ligatureSet) | hb_map (hb_add (this)))
    | hb_filter (glyphset, hb_first)
    | hb_filter ([&] (const LigatureSet& _) {
      return _.intersects (&glyphset);
    }, hb_second)
    | hb_map (hb_first)
    | hb_sink (new_coverage);

    if (!c->serializer->push<Coverage> ()
        ->serialize (c->serializer,
                     + new_coverage.iter () | hb_map_retains_sorting (glyph_map)))
    {
      c->serializer->pop_discard ();
      return_trace (false);
    }

    unsigned coverage_idx = c->serializer->pop_pack ();
     c->serializer->add_link (out->coverage, coverage_idx);

    + hb_zip (this+coverage, ligatureSet)
    | hb_filter (new_coverage, hb_first)
    | hb_map (hb_second)
    // to ensure that the repacker always orders the coverage table after the LigatureSet
    // and LigatureSubtable's they will be linked to the Coverage table via a virtual link
    // the coverage table object idx is passed down to facilitate this.
    | hb_apply (subset_offset_array (c, out->ligatureSet, this, coverage_idx))
    ;

    return_trace (bool (new_coverage));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && ligatureSet.sanitize (c, this));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  Offset16To<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of Substitution table */
  Array16OfOffset16To<LigatureSet>
		ligatureSet;		/* Array LigatureSet tables
					 * ordered by Coverage Index */
  public:
  DEFINE_SIZE_ARRAY (6, ligatureSet);
};

struct LigatureSubst
{
  bool serialize (hb_serialize_context_t *c,
		  hb_sorted_array_t<const HBGlyphID16> first_glyphs,
		  hb_array_t<const unsigned int> ligature_per_first_glyph_count_list,
		  hb_array_t<const HBGlyphID16> ligatures_list,
		  hb_array_t<const unsigned int> component_count_list,
		  hb_array_t<const HBGlyphID16> component_list /* Starting from second for each ligature */)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (u.format))) return_trace (false);
    unsigned int format = 1;
    u.format = format;
    switch (u.format) {
    case 1: return_trace (u.format1.serialize (c,
					       first_glyphs,
					       ligature_per_first_glyph_count_list,
					       ligatures_list,
					       component_count_list,
					       component_list));
    default:return_trace (false);
    }
  }

  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, Ts&&... ds) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1, std::forward<Ts> (ds)...));
    default:return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT16		format;		/* Format identifier */
  LigatureSubstFormat1	format1;
  } u;
};


struct ContextSubst : Context {};

struct ChainContextSubst : ChainContext {};

struct ExtensionSubst : Extension<ExtensionSubst>
{
  typedef struct SubstLookupSubTable SubTable;
  bool is_reverse () const;
};


struct ReverseChainSingleSubstFormat1
{
  bool intersects (const hb_set_t *glyphs) const
  {
    if (!(this+coverage).intersects (glyphs))
      return false;

    const Array16OfOffset16To<Coverage> &lookahead = StructAfter<Array16OfOffset16To<Coverage>> (backtrack);

    unsigned int count;

    count = backtrack.len;
    for (unsigned int i = 0; i < count; i++)
      if (!(this+backtrack[i]).intersects (glyphs))
	return false;

    count = lookahead.len;
    for (unsigned int i = 0; i < count; i++)
      if (!(this+lookahead[i]).intersects (glyphs))
	return false;

    return true;
  }

  bool may_have_non_1to1 () const
  { return false; }

  void closure (hb_closure_context_t *c) const
  {
    if (!intersects (c->glyphs)) return;

    const Array16OfOffset16To<Coverage> &lookahead = StructAfter<Array16OfOffset16To<Coverage>> (backtrack);
    const Array16Of<HBGlyphID16> &substitute = StructAfter<Array16Of<HBGlyphID16>> (lookahead);

    + hb_zip (this+coverage, substitute)
    | hb_filter (c->parent_active_glyphs (), hb_first)
    | hb_map (hb_second)
    | hb_sink (c->output)
    ;
  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+coverage).collect_coverage (c->input))) return;

    unsigned int count;

    count = backtrack.len;
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!(this+backtrack[i]).collect_coverage (c->before))) return;

    const Array16OfOffset16To<Coverage> &lookahead = StructAfter<Array16OfOffset16To<Coverage>> (backtrack);
    count = lookahead.len;
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!(this+lookahead[i]).collect_coverage (c->after))) return;

    const Array16Of<HBGlyphID16> &substitute = StructAfter<Array16Of<HBGlyphID16>> (lookahead);
    count = substitute.len;
    c->output->add_array (substitute.arrayZ, substitute.len);
  }

  const Coverage &get_coverage () const { return this+coverage; }

  bool would_apply (hb_would_apply_context_t *c) const
  { return c->len == 1 && (this+coverage).get_coverage (c->glyphs[0]) != NOT_COVERED; }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    if (unlikely (c->nesting_level_left != HB_MAX_NESTING_LEVEL))
      return_trace (false); /* No chaining to this type */

    unsigned int index = (this+coverage).get_coverage (c->buffer->cur ().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    const Array16OfOffset16To<Coverage> &lookahead = StructAfter<Array16OfOffset16To<Coverage>> (backtrack);
    const Array16Of<HBGlyphID16> &substitute = StructAfter<Array16Of<HBGlyphID16>> (lookahead);

    if (unlikely (index >= substitute.len)) return_trace (false);

    unsigned int start_index = 0, end_index = 0;
    if (match_backtrack (c,
			 backtrack.len, (HBUINT16 *) backtrack.arrayZ,
			 match_coverage, this,
			 &start_index) &&
	match_lookahead (c,
			 lookahead.len, (HBUINT16 *) lookahead.arrayZ,
			 match_coverage, this,
			 1, &end_index))
    {
      c->buffer->unsafe_to_break_from_outbuffer (start_index, end_index);
      c->replace_glyph_inplace (substitute[index]);
      /* Note: We DON'T decrease buffer->idx.  The main loop does it
       * for us.  This is useful for preventing surprises if someone
       * calls us through a Context lookup. */
      return_trace (true);
    }

    return_trace (false);
  }

  template<typename Iterator,
           hb_requires (hb_is_iterator (Iterator))>
  bool serialize_coverage_offset_array (hb_subset_context_t *c, Iterator it) const
  {
    TRACE_SERIALIZE (this);
    auto *out = c->serializer->start_embed<Array16OfOffset16To<Coverage>> ();

    if (unlikely (!c->serializer->allocate_size<HBUINT16> (HBUINT16::static_size)))
      return_trace (false);

    for (auto& offset : it) {
      auto *o = out->serialize_append (c->serializer);
      if (unlikely (!o) || !o->serialize_subset (c, offset, this))
        return_trace (false);
    }

    return_trace (true);
  }

  template<typename Iterator, typename BacktrackIterator, typename LookaheadIterator,
           hb_requires (hb_is_sorted_source_of (Iterator, hb_codepoint_pair_t)),
           hb_requires (hb_is_iterator (BacktrackIterator)),
           hb_requires (hb_is_iterator (LookaheadIterator))>
  bool serialize (hb_subset_context_t *c,
                  Iterator coverage_subst_iter,
                  BacktrackIterator backtrack_iter,
                  LookaheadIterator lookahead_iter) const
  {
    TRACE_SERIALIZE (this);

    auto *out = c->serializer->start_embed (this);
    if (unlikely (!c->serializer->check_success (out))) return_trace (false);
    if (unlikely (!c->serializer->embed (this->format))) return_trace (false);
    if (unlikely (!c->serializer->embed (this->coverage))) return_trace (false);

    if (!serialize_coverage_offset_array (c, backtrack_iter)) return_trace (false);
    if (!serialize_coverage_offset_array (c, lookahead_iter)) return_trace (false);

    auto *substitute_out = c->serializer->start_embed<Array16Of<HBGlyphID16>> ();
    auto substitutes =
    + coverage_subst_iter
    | hb_map (hb_second)
    ;

    auto glyphs =
    + coverage_subst_iter
    | hb_map_retains_sorting (hb_first)
    ;
    if (unlikely (! c->serializer->check_success (substitute_out->serialize (c->serializer, substitutes))))
        return_trace (false);

    if (unlikely (!out->coverage.serialize_serialize (c->serializer, glyphs)))
      return_trace (false);
    return_trace (true);
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    const Array16OfOffset16To<Coverage> &lookahead = StructAfter<Array16OfOffset16To<Coverage>> (backtrack);
    const Array16Of<HBGlyphID16> &substitute = StructAfter<Array16Of<HBGlyphID16>> (lookahead);

    auto it =
    + hb_zip (this+coverage, substitute)
    | hb_filter (glyphset, hb_first)
    | hb_filter (glyphset, hb_second)
    | hb_map_retains_sorting ([&] (hb_pair_t<hb_codepoint_t, const HBGlyphID16 &> p) -> hb_codepoint_pair_t
                              { return hb_pair (glyph_map[p.first], glyph_map[p.second]); })
    ;

    return_trace (bool (it) && serialize (c, it, backtrack.iter (), lookahead.iter ()));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!(coverage.sanitize (c, this) && backtrack.sanitize (c, this)))
      return_trace (false);
    const Array16OfOffset16To<Coverage> &lookahead = StructAfter<Array16OfOffset16To<Coverage>> (backtrack);
    if (!lookahead.sanitize (c, this))
      return_trace (false);
    const Array16Of<HBGlyphID16> &substitute = StructAfter<Array16Of<HBGlyphID16>> (lookahead);
    return_trace (substitute.sanitize (c));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  Offset16To<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of table */
  Array16OfOffset16To<Coverage>
		backtrack;		/* Array of coverage tables
					 * in backtracking sequence, in glyph
					 * sequence order */
  Array16OfOffset16To<Coverage>
		lookaheadX;		/* Array of coverage tables
					 * in lookahead sequence, in glyph
					 * sequence order */
  Array16Of<HBGlyphID16>
		substituteX;		/* Array of substitute
					 * GlyphIDs--ordered by Coverage Index */
  public:
  DEFINE_SIZE_MIN (10);
};

struct ReverseChainSingleSubst
{
  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, Ts&&... ds) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1, std::forward<Ts> (ds)...));
    default:return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT16				format;		/* Format identifier */
  ReverseChainSingleSubstFormat1	format1;
  } u;
};



/*
 * SubstLookup
 */

struct SubstLookupSubTable
{
  friend struct Lookup;
  friend struct SubstLookup;

  enum Type {
    Single		= 1,
    Multiple		= 2,
    Alternate		= 3,
    Ligature		= 4,
    Context		= 5,
    ChainContext	= 6,
    Extension		= 7,
    ReverseChainSingle	= 8
  };

  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, unsigned int lookup_type, Ts&&... ds) const
  {
    TRACE_DISPATCH (this, lookup_type);
    switch (lookup_type) {
    case Single:		return_trace (u.single.dispatch (c, std::forward<Ts> (ds)...));
    case Multiple:		return_trace (u.multiple.dispatch (c, std::forward<Ts> (ds)...));
    case Alternate:		return_trace (u.alternate.dispatch (c, std::forward<Ts> (ds)...));
    case Ligature:		return_trace (u.ligature.dispatch (c, std::forward<Ts> (ds)...));
    case Context:		return_trace (u.context.dispatch (c, std::forward<Ts> (ds)...));
    case ChainContext:		return_trace (u.chainContext.dispatch (c, std::forward<Ts> (ds)...));
    case Extension:		return_trace (u.extension.dispatch (c, std::forward<Ts> (ds)...));
    case ReverseChainSingle:	return_trace (u.reverseChainContextSingle.dispatch (c, std::forward<Ts> (ds)...));
    default:			return_trace (c->default_return_value ());
    }
  }

  bool intersects (const hb_set_t *glyphs, unsigned int lookup_type) const
  {
    hb_intersects_context_t c (glyphs);
    return dispatch (&c, lookup_type);
  }

  protected:
  union {
  SingleSubst			single;
  MultipleSubst			multiple;
  AlternateSubst		alternate;
  LigatureSubst			ligature;
  ContextSubst			context;
  ChainContextSubst		chainContext;
  ExtensionSubst		extension;
  ReverseChainSingleSubst	reverseChainContextSingle;
  } u;
  public:
  DEFINE_SIZE_MIN (0);
};


struct SubstLookup : Lookup
{
  typedef SubstLookupSubTable SubTable;

  const SubTable& get_subtable (unsigned int i) const
  { return Lookup::get_subtable<SubTable> (i); }

  static inline bool lookup_type_is_reverse (unsigned int lookup_type)
  { return lookup_type == SubTable::ReverseChainSingle; }

  bool is_reverse () const
  {
    unsigned int type = get_type ();
    if (unlikely (type == SubTable::Extension))
      return reinterpret_cast<const ExtensionSubst &> (get_subtable (0)).is_reverse ();
    return lookup_type_is_reverse (type);
  }

  bool may_have_non_1to1 () const
  {
    hb_have_non_1to1_context_t c;
    return dispatch (&c);
  }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    return_trace (dispatch (c));
  }

  bool intersects (const hb_set_t *glyphs) const
  {
    hb_intersects_context_t c (glyphs);
    return dispatch (&c);
  }

  hb_closure_context_t::return_t closure (hb_closure_context_t *c, unsigned int this_index) const
  {
    if (!c->should_visit_lookup (this_index))
      return hb_closure_context_t::default_return_value ();

    c->set_recurse_func (dispatch_closure_recurse_func);

    hb_closure_context_t::return_t ret = dispatch (c);

    c->flush ();

    return ret;
  }

  hb_closure_lookups_context_t::return_t closure_lookups (hb_closure_lookups_context_t *c, unsigned this_index) const
  {
    if (c->is_lookup_visited (this_index))
      return hb_closure_lookups_context_t::default_return_value ();

    c->set_lookup_visited (this_index);
    if (!intersects (c->glyphs))
    {
      c->set_lookup_inactive (this_index);
      return hb_closure_lookups_context_t::default_return_value ();
    }

    c->set_recurse_func (dispatch_closure_lookups_recurse_func);

    hb_closure_lookups_context_t::return_t ret = dispatch (c);
    return ret;
  }

  hb_collect_glyphs_context_t::return_t collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    c->set_recurse_func (dispatch_recurse_func<hb_collect_glyphs_context_t>);
    return dispatch (c);
  }

  template <typename set_t>
  void collect_coverage (set_t *glyphs) const
  {
    hb_collect_coverage_context_t<set_t> c (glyphs);
    dispatch (&c);
  }

  bool would_apply (hb_would_apply_context_t *c,
		    const hb_ot_layout_lookup_accelerator_t *accel) const
  {
    if (unlikely (!c->len)) return false;
    if (!accel->may_have (c->glyphs[0])) return false;
      return dispatch (c);
  }

  static inline bool apply_recurse_func (hb_ot_apply_context_t *c, unsigned int lookup_index);

  bool serialize_single (hb_serialize_context_t *c,
			 uint32_t lookup_props,
			 hb_sorted_array_t<const HBGlyphID16> glyphs,
			 hb_array_t<const HBGlyphID16> substitutes)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!Lookup::serialize (c, SubTable::Single, lookup_props, 1))) return_trace (false);
    if (c->push<SubTable> ()->u.single.serialize (c, hb_zip (glyphs, substitutes)))
    {
      c->add_link (get_subtables<SubTable> ()[0], c->pop_pack ());
      return_trace (true);
    }
    c->pop_discard ();
    return_trace (false);
  }

  bool serialize_multiple (hb_serialize_context_t *c,
			   uint32_t lookup_props,
			   hb_sorted_array_t<const HBGlyphID16> glyphs,
			   hb_array_t<const unsigned int> substitute_len_list,
			   hb_array_t<const HBGlyphID16> substitute_glyphs_list)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!Lookup::serialize (c, SubTable::Multiple, lookup_props, 1))) return_trace (false);
    if (c->push<SubTable> ()->u.multiple.
        serialize (c,
                   glyphs,
                   substitute_len_list,
                   substitute_glyphs_list))
    {
      c->add_link (get_subtables<SubTable> ()[0], c->pop_pack ());
      return_trace (true);
    }
    c->pop_discard ();
    return_trace (false);
  }

  bool serialize_alternate (hb_serialize_context_t *c,
			    uint32_t lookup_props,
			    hb_sorted_array_t<const HBGlyphID16> glyphs,
			    hb_array_t<const unsigned int> alternate_len_list,
			    hb_array_t<const HBGlyphID16> alternate_glyphs_list)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!Lookup::serialize (c, SubTable::Alternate, lookup_props, 1))) return_trace (false);

    if (c->push<SubTable> ()->u.alternate.
        serialize (c,
                   glyphs,
                   alternate_len_list,
                   alternate_glyphs_list))
    {
      c->add_link (get_subtables<SubTable> ()[0], c->pop_pack ());
      return_trace (true);
    }
    c->pop_discard ();
    return_trace (false);
  }

  bool serialize_ligature (hb_serialize_context_t *c,
			   uint32_t lookup_props,
			   hb_sorted_array_t<const HBGlyphID16> first_glyphs,
			   hb_array_t<const unsigned int> ligature_per_first_glyph_count_list,
			   hb_array_t<const HBGlyphID16> ligatures_list,
			   hb_array_t<const unsigned int> component_count_list,
			   hb_array_t<const HBGlyphID16> component_list /* Starting from second for each ligature */)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!Lookup::serialize (c, SubTable::Ligature, lookup_props, 1))) return_trace (false);
    if (c->push<SubTable> ()->u.ligature.
        serialize (c,
                   first_glyphs,
                   ligature_per_first_glyph_count_list,
                   ligatures_list,
                   component_count_list,
                   component_list))
    {
      c->add_link (get_subtables<SubTable> ()[0], c->pop_pack ());
      return_trace (true);
    }
    c->pop_discard ();
    return_trace (false);
  }

  template <typename context_t>
  static inline typename context_t::return_t dispatch_recurse_func (context_t *c, unsigned int lookup_index);

  static inline typename hb_closure_context_t::return_t closure_glyphs_recurse_func (hb_closure_context_t *c, unsigned lookup_index, hb_set_t *covered_seq_indices, unsigned seq_index, unsigned end_index);

  static inline hb_closure_context_t::return_t dispatch_closure_recurse_func (hb_closure_context_t *c, unsigned lookup_index, hb_set_t *covered_seq_indices, unsigned seq_index, unsigned end_index)
  {
    if (!c->should_visit_lookup (lookup_index))
      return hb_empty_t ();

    hb_closure_context_t::return_t ret = closure_glyphs_recurse_func (c, lookup_index, covered_seq_indices, seq_index, end_index);

    /* While in theory we should flush here, it will cause timeouts because a recursive
     * lookup can keep growing the glyph set.  Skip, and outer loop will retry up to
     * HB_CLOSURE_MAX_STAGES time, which should be enough for every realistic font. */
    //c->flush ();

    return ret;
  }

  HB_INTERNAL static hb_closure_lookups_context_t::return_t dispatch_closure_lookups_recurse_func (hb_closure_lookups_context_t *c, unsigned lookup_index);

  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, Ts&&... ds) const
  { return Lookup::dispatch<SubTable> (c, std::forward<Ts> (ds)...); }

  bool subset (hb_subset_context_t *c) const
  { return Lookup::subset<SubTable> (c); }

  bool sanitize (hb_sanitize_context_t *c) const
  { return Lookup::sanitize<SubTable> (c); }
};

/*
 * GSUB -- Glyph Substitution
 * https://docs.microsoft.com/en-us/typography/opentype/spec/gsub
 */

struct GSUB : GSUBGPOS
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_GSUB;

  const SubstLookup& get_lookup (unsigned int i) const
  { return static_cast<const SubstLookup &> (GSUBGPOS::get_lookup (i)); }

  bool subset (hb_subset_context_t *c) const
  {
    hb_subset_layout_context_t l (c, tableTag, c->plan->gsub_lookups, c->plan->gsub_langsys, c->plan->gsub_features);
    return GSUBGPOS::subset<SubstLookup> (&l);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  { return GSUBGPOS::sanitize<SubstLookup> (c); }

  HB_INTERNAL bool is_blocklisted (hb_blob_t *blob,
				   hb_face_t *face) const;

  void closure_lookups (hb_face_t      *face,
			const hb_set_t *glyphs,
			hb_set_t       *lookup_indexes /* IN/OUT */) const
  { GSUBGPOS::closure_lookups<SubstLookup> (face, glyphs, lookup_indexes); }

  typedef GSUBGPOS::accelerator_t<GSUB> accelerator_t;
};


struct GSUB_accelerator_t : GSUB::accelerator_t {};


/* Out-of-class implementation for methods recursing */

#ifndef HB_NO_OT_LAYOUT
/*static*/ inline bool ExtensionSubst::is_reverse () const
{
  return SubstLookup::lookup_type_is_reverse (get_type ());
}
template <typename context_t>
/*static*/ typename context_t::return_t SubstLookup::dispatch_recurse_func (context_t *c, unsigned int lookup_index)
{
  const SubstLookup &l = c->face->table.GSUB.get_relaxed ()->table->get_lookup (lookup_index);
  return l.dispatch (c);
}

/*static*/ typename hb_closure_context_t::return_t SubstLookup::closure_glyphs_recurse_func (hb_closure_context_t *c, unsigned lookup_index, hb_set_t *covered_seq_indices, unsigned seq_index, unsigned end_index)
{
  const SubstLookup &l = c->face->table.GSUB.get_relaxed ()->table->get_lookup (lookup_index);
  if (l.may_have_non_1to1 ())
      hb_set_add_range (covered_seq_indices, seq_index, end_index);
  return l.dispatch (c);
}

/*static*/ inline hb_closure_lookups_context_t::return_t SubstLookup::dispatch_closure_lookups_recurse_func (hb_closure_lookups_context_t *c, unsigned this_index)
{
  const SubstLookup &l = c->face->table.GSUB.get_relaxed ()->table->get_lookup (this_index);
  return l.closure_lookups (c, this_index);
}

/*static*/ bool SubstLookup::apply_recurse_func (hb_ot_apply_context_t *c, unsigned int lookup_index)
{
  const SubstLookup &l = c->face->table.GSUB.get_relaxed ()->table->get_lookup (lookup_index);
  unsigned int saved_lookup_props = c->lookup_props;
  unsigned int saved_lookup_index = c->lookup_index;
  c->set_lookup_index (lookup_index);
  c->set_lookup_props (l.get_props ());
  bool ret = l.dispatch (c);
  c->set_lookup_index (saved_lookup_index);
  c->set_lookup_props (saved_lookup_props);
  return ret;
}
#endif


} /* namespace OT */


#endif /* HB_OT_LAYOUT_GSUB_TABLE_HH */

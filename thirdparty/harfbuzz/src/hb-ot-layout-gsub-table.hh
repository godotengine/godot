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

#include "hb-ot-layout-gsubgpos-private.hh"


namespace OT {


struct SingleSubstFormat1
{
  inline void closure (hb_closure_context_t *c) const
  {
    TRACE_CLOSURE (this);
    Coverage::Iter iter;
    for (iter.init (this+coverage); iter.more (); iter.next ())
    {
      /* TODO Switch to range-based API to work around malicious fonts.
       * https://github.com/harfbuzz/harfbuzz/issues/363 */
      hb_codepoint_t glyph_id = iter.get_glyph ();
      if (c->glyphs->has (glyph_id))
	c->out->add ((glyph_id + deltaGlyphID) & 0xFFFFu);
    }
  }

  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    if (unlikely (!(this+coverage).add_coverage (c->input))) return;
    Coverage::Iter iter;
    for (iter.init (this+coverage); iter.more (); iter.next ())
    {
      /* TODO Switch to range-based API to work around malicious fonts.
       * https://github.com/harfbuzz/harfbuzz/issues/363 */
      hb_codepoint_t glyph_id = iter.get_glyph ();
      c->output->add ((glyph_id + deltaGlyphID) & 0xFFFFu);
    }
  }

  inline const Coverage &get_coverage (void) const
  {
    return this+coverage;
  }

  inline bool would_apply (hb_would_apply_context_t *c) const
  {
    TRACE_WOULD_APPLY (this);
    return_trace (c->len == 1 && (this+coverage).get_coverage (c->glyphs[0]) != NOT_COVERED);
  }

  inline bool apply (hb_ot_apply_context_t *c) const
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

  inline bool serialize (hb_serialize_context_t *c,
			 Supplier<GlyphID> &glyphs,
			 unsigned int num_glyphs,
			 int delta)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (*this))) return_trace (false);
    if (unlikely (!coverage.serialize (c, this).serialize (c, glyphs, num_glyphs))) return_trace (false);
    deltaGlyphID.set (delta); /* TODO(serilaize) overflow? */
    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && deltaGlyphID.sanitize (c));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  OffsetTo<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of Substitution table */
  HBINT16	deltaGlyphID;		/* Add to original GlyphID to get
					 * substitute GlyphID */
  public:
  DEFINE_SIZE_STATIC (6);
};

struct SingleSubstFormat2
{
  inline void closure (hb_closure_context_t *c) const
  {
    TRACE_CLOSURE (this);
    Coverage::Iter iter;
    unsigned int count = substitute.len;
    for (iter.init (this+coverage); iter.more (); iter.next ())
    {
      if (unlikely (iter.get_coverage () >= count))
        break; /* Work around malicious fonts. https://github.com/harfbuzz/harfbuzz/issues/363 */
      if (c->glyphs->has (iter.get_glyph ()))
	c->out->add (substitute[iter.get_coverage ()]);
    }
  }

  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    if (unlikely (!(this+coverage).add_coverage (c->input))) return;
    Coverage::Iter iter;
    unsigned int count = substitute.len;
    for (iter.init (this+coverage); iter.more (); iter.next ())
    {
      if (unlikely (iter.get_coverage () >= count))
        break; /* Work around malicious fonts. https://github.com/harfbuzz/harfbuzz/issues/363 */
      c->output->add (substitute[iter.get_coverage ()]);
    }
  }

  inline const Coverage &get_coverage (void) const
  {
    return this+coverage;
  }

  inline bool would_apply (hb_would_apply_context_t *c) const
  {
    TRACE_WOULD_APPLY (this);
    return_trace (c->len == 1 && (this+coverage).get_coverage (c->glyphs[0]) != NOT_COVERED);
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_codepoint_t glyph_id = c->buffer->cur().codepoint;
    unsigned int index = (this+coverage).get_coverage (glyph_id);
    if (likely (index == NOT_COVERED)) return_trace (false);

    if (unlikely (index >= substitute.len)) return_trace (false);

    glyph_id = substitute[index];
    c->replace_glyph (glyph_id);

    return_trace (true);
  }

  inline bool serialize (hb_serialize_context_t *c,
			 Supplier<GlyphID> &glyphs,
			 Supplier<GlyphID> &substitutes,
			 unsigned int num_glyphs)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (*this))) return_trace (false);
    if (unlikely (!substitute.serialize (c, substitutes, num_glyphs))) return_trace (false);
    if (unlikely (!coverage.serialize (c, this).serialize (c, glyphs, num_glyphs))) return_trace (false);
    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && substitute.sanitize (c));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 2 */
  OffsetTo<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of Substitution table */
  ArrayOf<GlyphID>
		substitute;		/* Array of substitute
					 * GlyphIDs--ordered by Coverage Index */
  public:
  DEFINE_SIZE_ARRAY (6, substitute);
};

struct SingleSubst
{
  inline bool serialize (hb_serialize_context_t *c,
			 Supplier<GlyphID> &glyphs,
			 Supplier<GlyphID> &substitutes,
			 unsigned int num_glyphs)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (u.format))) return_trace (false);
    unsigned int format = 2;
    int delta = 0;
    if (num_glyphs) {
      format = 1;
      /* TODO(serialize) check for wrap-around */
      delta = substitutes[0] - glyphs[0];
      for (unsigned int i = 1; i < num_glyphs; i++)
	if (delta != substitutes[i] - glyphs[i]) {
	  format = 2;
	  break;
	}
    }
    u.format.set (format);
    switch (u.format) {
    case 1: return_trace (u.format1.serialize (c, glyphs, num_glyphs, delta));
    case 2: return_trace (u.format2.serialize (c, glyphs, substitutes, num_glyphs));
    default:return_trace (false);
    }
  }

  template <typename context_t>
  inline typename context_t::return_t dispatch (context_t *c) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1));
    case 2: return_trace (c->dispatch (u.format2));
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


struct Sequence
{
  inline void closure (hb_closure_context_t *c) const
  {
    TRACE_CLOSURE (this);
    unsigned int count = substitute.len;
    for (unsigned int i = 0; i < count; i++)
      c->out->add (substitute[i]);
  }

  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    c->output->add_array (substitute.arrayZ, substitute.len);
  }

  inline bool apply (hb_ot_apply_context_t *c) const
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

    for (unsigned int i = 0; i < count; i++) {
      _hb_glyph_info_set_lig_props_for_component (&c->buffer->cur(), i);
      c->output_glyph_for_component (substitute.arrayZ[i], klass);
    }
    c->buffer->skip_glyph ();

    return_trace (true);
  }

  inline bool serialize (hb_serialize_context_t *c,
			 Supplier<GlyphID> &glyphs,
			 unsigned int num_glyphs)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (*this))) return_trace (false);
    if (unlikely (!substitute.serialize (c, glyphs, num_glyphs))) return_trace (false);
    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (substitute.sanitize (c));
  }

  protected:
  ArrayOf<GlyphID>
		substitute;		/* String of GlyphIDs to substitute */
  public:
  DEFINE_SIZE_ARRAY (2, substitute);
};

struct MultipleSubstFormat1
{
  inline void closure (hb_closure_context_t *c) const
  {
    TRACE_CLOSURE (this);
    Coverage::Iter iter;
    unsigned int count = sequence.len;
    for (iter.init (this+coverage); iter.more (); iter.next ())
    {
      if (unlikely (iter.get_coverage () >= count))
        break; /* Work around malicious fonts. https://github.com/harfbuzz/harfbuzz/issues/363 */
      if (c->glyphs->has (iter.get_glyph ()))
	(this+sequence[iter.get_coverage ()]).closure (c);
    }
  }

  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    if (unlikely (!(this+coverage).add_coverage (c->input))) return;
    unsigned int count = sequence.len;
    for (unsigned int i = 0; i < count; i++)
	(this+sequence[i]).collect_glyphs (c);
  }

  inline const Coverage &get_coverage (void) const
  {
    return this+coverage;
  }

  inline bool would_apply (hb_would_apply_context_t *c) const
  {
    TRACE_WOULD_APPLY (this);
    return_trace (c->len == 1 && (this+coverage).get_coverage (c->glyphs[0]) != NOT_COVERED);
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);

    unsigned int index = (this+coverage).get_coverage (c->buffer->cur().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    return_trace ((this+sequence[index]).apply (c));
  }

  inline bool serialize (hb_serialize_context_t *c,
			 Supplier<GlyphID> &glyphs,
			 Supplier<unsigned int> &substitute_len_list,
			 unsigned int num_glyphs,
			 Supplier<GlyphID> &substitute_glyphs_list)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (*this))) return_trace (false);
    if (unlikely (!sequence.serialize (c, num_glyphs))) return_trace (false);
    for (unsigned int i = 0; i < num_glyphs; i++)
      if (unlikely (!sequence[i].serialize (c, this).serialize (c,
								substitute_glyphs_list,
								substitute_len_list[i]))) return_trace (false);
    substitute_len_list += num_glyphs;
    if (unlikely (!coverage.serialize (c, this).serialize (c, glyphs, num_glyphs))) return_trace (false);
    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && sequence.sanitize (c, this));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  OffsetTo<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of Substitution table */
  OffsetArrayOf<Sequence>
		sequence;		/* Array of Sequence tables
					 * ordered by Coverage Index */
  public:
  DEFINE_SIZE_ARRAY (6, sequence);
};

struct MultipleSubst
{
  inline bool serialize (hb_serialize_context_t *c,
			 Supplier<GlyphID> &glyphs,
			 Supplier<unsigned int> &substitute_len_list,
			 unsigned int num_glyphs,
			 Supplier<GlyphID> &substitute_glyphs_list)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (u.format))) return_trace (false);
    unsigned int format = 1;
    u.format.set (format);
    switch (u.format) {
    case 1: return_trace (u.format1.serialize (c, glyphs, substitute_len_list, num_glyphs, substitute_glyphs_list));
    default:return_trace (false);
    }
  }

  template <typename context_t>
  inline typename context_t::return_t dispatch (context_t *c) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1));
    default:return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT16		format;		/* Format identifier */
  MultipleSubstFormat1	format1;
  } u;
};


typedef ArrayOf<GlyphID> AlternateSet;	/* Array of alternate GlyphIDs--in
					 * arbitrary order */

struct AlternateSubstFormat1
{
  inline void closure (hb_closure_context_t *c) const
  {
    TRACE_CLOSURE (this);
    Coverage::Iter iter;
    unsigned int count = alternateSet.len;
    for (iter.init (this+coverage); iter.more (); iter.next ())
    {
      if (unlikely (iter.get_coverage () >= count))
        break; /* Work around malicious fonts. https://github.com/harfbuzz/harfbuzz/issues/363 */
      if (c->glyphs->has (iter.get_glyph ())) {
	const AlternateSet &alt_set = this+alternateSet[iter.get_coverage ()];
	unsigned int count = alt_set.len;
	for (unsigned int i = 0; i < count; i++)
	  c->out->add (alt_set[i]);
      }
    }
  }

  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    if (unlikely (!(this+coverage).add_coverage (c->input))) return;
    Coverage::Iter iter;
    unsigned int count = alternateSet.len;
    for (iter.init (this+coverage); iter.more (); iter.next ())
    {
      if (unlikely (iter.get_coverage () >= count))
        break; /* Work around malicious fonts. https://github.com/harfbuzz/harfbuzz/issues/363 */
      const AlternateSet &alt_set = this+alternateSet[iter.get_coverage ()];
      c->output->add_array (alt_set.arrayZ, alt_set.len);
    }
  }

  inline const Coverage &get_coverage (void) const
  {
    return this+coverage;
  }

  inline bool would_apply (hb_would_apply_context_t *c) const
  {
    TRACE_WOULD_APPLY (this);
    return_trace (c->len == 1 && (this+coverage).get_coverage (c->glyphs[0]) != NOT_COVERED);
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_codepoint_t glyph_id = c->buffer->cur().codepoint;

    unsigned int index = (this+coverage).get_coverage (glyph_id);
    if (likely (index == NOT_COVERED)) return_trace (false);

    const AlternateSet &alt_set = this+alternateSet[index];

    if (unlikely (!alt_set.len)) return_trace (false);

    hb_mask_t glyph_mask = c->buffer->cur().mask;
    hb_mask_t lookup_mask = c->lookup_mask;

    /* Note: This breaks badly if two features enabled this lookup together. */
    unsigned int shift = hb_ctz (lookup_mask);
    unsigned int alt_index = ((lookup_mask & glyph_mask) >> shift);

    if (unlikely (alt_index > alt_set.len || alt_index == 0)) return_trace (false);

    glyph_id = alt_set[alt_index - 1];

    c->replace_glyph (glyph_id);

    return_trace (true);
  }

  inline bool serialize (hb_serialize_context_t *c,
			 Supplier<GlyphID> &glyphs,
			 Supplier<unsigned int> &alternate_len_list,
			 unsigned int num_glyphs,
			 Supplier<GlyphID> &alternate_glyphs_list)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (*this))) return_trace (false);
    if (unlikely (!alternateSet.serialize (c, num_glyphs))) return_trace (false);
    for (unsigned int i = 0; i < num_glyphs; i++)
      if (unlikely (!alternateSet[i].serialize (c, this).serialize (c,
								    alternate_glyphs_list,
								    alternate_len_list[i]))) return_trace (false);
    alternate_len_list += num_glyphs;
    if (unlikely (!coverage.serialize (c, this).serialize (c, glyphs, num_glyphs))) return_trace (false);
    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && alternateSet.sanitize (c, this));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  OffsetTo<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of Substitution table */
  OffsetArrayOf<AlternateSet>
		alternateSet;		/* Array of AlternateSet tables
					 * ordered by Coverage Index */
  public:
  DEFINE_SIZE_ARRAY (6, alternateSet);
};

struct AlternateSubst
{
  inline bool serialize (hb_serialize_context_t *c,
			 Supplier<GlyphID> &glyphs,
			 Supplier<unsigned int> &alternate_len_list,
			 unsigned int num_glyphs,
			 Supplier<GlyphID> &alternate_glyphs_list)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (u.format))) return_trace (false);
    unsigned int format = 1;
    u.format.set (format);
    switch (u.format) {
    case 1: return_trace (u.format1.serialize (c, glyphs, alternate_len_list, num_glyphs, alternate_glyphs_list));
    default:return_trace (false);
    }
  }

  template <typename context_t>
  inline typename context_t::return_t dispatch (context_t *c) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1));
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
  inline void closure (hb_closure_context_t *c) const
  {
    TRACE_CLOSURE (this);
    unsigned int count = component.len;
    for (unsigned int i = 1; i < count; i++)
      if (!c->glyphs->has (component[i]))
        return;
    c->out->add (ligGlyph);
  }

  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    c->input->add_array (component.arrayZ, component.len ? component.len - 1 : 0);
    c->output->add (ligGlyph);
  }

  inline bool would_apply (hb_would_apply_context_t *c) const
  {
    TRACE_WOULD_APPLY (this);
    if (c->len != component.len)
      return_trace (false);

    for (unsigned int i = 1; i < c->len; i++)
      if (likely (c->glyphs[i] != component[i]))
	return_trace (false);

    return_trace (true);
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    unsigned int count = component.len;

    if (unlikely (!count)) return_trace (false);

    /* Special-case to make it in-place and not consider this
     * as a "ligated" substitution. */
    if (unlikely (count == 1))
    {
      c->replace_glyph (ligGlyph);
      return_trace (true);
    }

    bool is_mark_ligature = false;
    unsigned int total_component_count = 0;

    unsigned int match_length = 0;
    unsigned int match_positions[HB_MAX_CONTEXT_LENGTH];

    if (likely (!match_input (c, count,
			      &component[1],
			      match_glyph,
			      nullptr,
			      &match_length,
			      match_positions,
			      &is_mark_ligature,
			      &total_component_count)))
      return_trace (false);

    ligate_input (c,
		  count,
		  match_positions,
		  match_length,
		  ligGlyph,
		  is_mark_ligature,
		  total_component_count);

    return_trace (true);
  }

  inline bool serialize (hb_serialize_context_t *c,
			 GlyphID ligature,
			 Supplier<GlyphID> &components, /* Starting from second */
			 unsigned int num_components /* Including first component */)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (*this))) return_trace (false);
    ligGlyph = ligature;
    if (unlikely (!component.serialize (c, components, num_components))) return_trace (false);
    return_trace (true);
  }

  public:
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (ligGlyph.sanitize (c) && component.sanitize (c));
  }

  protected:
  GlyphID	ligGlyph;		/* GlyphID of ligature to substitute */
  HeadlessArrayOf<GlyphID>
		component;		/* Array of component GlyphIDs--start
					 * with the second  component--ordered
					 * in writing direction */
  public:
  DEFINE_SIZE_ARRAY (4, component);
};

struct LigatureSet
{
  inline void closure (hb_closure_context_t *c) const
  {
    TRACE_CLOSURE (this);
    unsigned int num_ligs = ligature.len;
    for (unsigned int i = 0; i < num_ligs; i++)
      (this+ligature[i]).closure (c);
  }

  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    unsigned int num_ligs = ligature.len;
    for (unsigned int i = 0; i < num_ligs; i++)
      (this+ligature[i]).collect_glyphs (c);
  }

  inline bool would_apply (hb_would_apply_context_t *c) const
  {
    TRACE_WOULD_APPLY (this);
    unsigned int num_ligs = ligature.len;
    for (unsigned int i = 0; i < num_ligs; i++)
    {
      const Ligature &lig = this+ligature[i];
      if (lig.would_apply (c))
        return_trace (true);
    }
    return_trace (false);
  }

  inline bool apply (hb_ot_apply_context_t *c) const
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

  inline bool serialize (hb_serialize_context_t *c,
			 Supplier<GlyphID> &ligatures,
			 Supplier<unsigned int> &component_count_list,
			 unsigned int num_ligatures,
			 Supplier<GlyphID> &component_list /* Starting from second for each ligature */)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (*this))) return_trace (false);
    if (unlikely (!ligature.serialize (c, num_ligatures))) return_trace (false);
    for (unsigned int i = 0; i < num_ligatures; i++)
      if (unlikely (!ligature[i].serialize (c, this).serialize (c,
								ligatures[i],
								component_list,
								component_count_list[i]))) return_trace (false);
    ligatures += num_ligatures;
    component_count_list += num_ligatures;
    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (ligature.sanitize (c, this));
  }

  protected:
  OffsetArrayOf<Ligature>
		ligature;		/* Array LigatureSet tables
					 * ordered by preference */
  public:
  DEFINE_SIZE_ARRAY (2, ligature);
};

struct LigatureSubstFormat1
{
  inline void closure (hb_closure_context_t *c) const
  {
    TRACE_CLOSURE (this);
    Coverage::Iter iter;
    unsigned int count = ligatureSet.len;
    for (iter.init (this+coverage); iter.more (); iter.next ())
    {
      if (unlikely (iter.get_coverage () >= count))
        break; /* Work around malicious fonts. https://github.com/harfbuzz/harfbuzz/issues/363 */
      if (c->glyphs->has (iter.get_glyph ()))
	(this+ligatureSet[iter.get_coverage ()]).closure (c);
    }
  }

  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    if (unlikely (!(this+coverage).add_coverage (c->input))) return;
    Coverage::Iter iter;
    unsigned int count = ligatureSet.len;
    for (iter.init (this+coverage); iter.more (); iter.next ())
    {
      if (unlikely (iter.get_coverage () >= count))
        break; /* Work around malicious fonts. https://github.com/harfbuzz/harfbuzz/issues/363 */
      (this+ligatureSet[iter.get_coverage ()]).collect_glyphs (c);
    }
  }

  inline const Coverage &get_coverage (void) const
  {
    return this+coverage;
  }

  inline bool would_apply (hb_would_apply_context_t *c) const
  {
    TRACE_WOULD_APPLY (this);
    unsigned int index = (this+coverage).get_coverage (c->glyphs[0]);
    if (likely (index == NOT_COVERED)) return_trace (false);

    const LigatureSet &lig_set = this+ligatureSet[index];
    return_trace (lig_set.would_apply (c));
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_codepoint_t glyph_id = c->buffer->cur().codepoint;

    unsigned int index = (this+coverage).get_coverage (glyph_id);
    if (likely (index == NOT_COVERED)) return_trace (false);

    const LigatureSet &lig_set = this+ligatureSet[index];
    return_trace (lig_set.apply (c));
  }

  inline bool serialize (hb_serialize_context_t *c,
			 Supplier<GlyphID> &first_glyphs,
			 Supplier<unsigned int> &ligature_per_first_glyph_count_list,
			 unsigned int num_first_glyphs,
			 Supplier<GlyphID> &ligatures_list,
			 Supplier<unsigned int> &component_count_list,
			 Supplier<GlyphID> &component_list /* Starting from second for each ligature */)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (*this))) return_trace (false);
    if (unlikely (!ligatureSet.serialize (c, num_first_glyphs))) return_trace (false);
    for (unsigned int i = 0; i < num_first_glyphs; i++)
      if (unlikely (!ligatureSet[i].serialize (c, this).serialize (c,
								   ligatures_list,
								   component_count_list,
								   ligature_per_first_glyph_count_list[i],
								   component_list))) return_trace (false);
    ligature_per_first_glyph_count_list += num_first_glyphs;
    if (unlikely (!coverage.serialize (c, this).serialize (c, first_glyphs, num_first_glyphs))) return_trace (false);
    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && ligatureSet.sanitize (c, this));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  OffsetTo<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of Substitution table */
  OffsetArrayOf<LigatureSet>
		ligatureSet;		/* Array LigatureSet tables
					 * ordered by Coverage Index */
  public:
  DEFINE_SIZE_ARRAY (6, ligatureSet);
};

struct LigatureSubst
{
  inline bool serialize (hb_serialize_context_t *c,
			 Supplier<GlyphID> &first_glyphs,
			 Supplier<unsigned int> &ligature_per_first_glyph_count_list,
			 unsigned int num_first_glyphs,
			 Supplier<GlyphID> &ligatures_list,
			 Supplier<unsigned int> &component_count_list,
			 Supplier<GlyphID> &component_list /* Starting from second for each ligature */)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (u.format))) return_trace (false);
    unsigned int format = 1;
    u.format.set (format);
    switch (u.format) {
    case 1: return_trace (u.format1.serialize (c,
					       first_glyphs,
					       ligature_per_first_glyph_count_list,
					       num_first_glyphs,
					       ligatures_list,
					       component_count_list,
					       component_list));
    default:return_trace (false);
    }
  }

  template <typename context_t>
  inline typename context_t::return_t dispatch (context_t *c) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1));
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
  typedef struct SubstLookupSubTable LookupSubTable;

  inline bool is_reverse (void) const;
};


struct ReverseChainSingleSubstFormat1
{
  inline void closure (hb_closure_context_t *c) const
  {
    TRACE_CLOSURE (this);
    const OffsetArrayOf<Coverage> &lookahead = StructAfter<OffsetArrayOf<Coverage> > (backtrack);

    unsigned int count;

    count = backtrack.len;
    for (unsigned int i = 0; i < count; i++)
      if (!(this+backtrack[i]).intersects (c->glyphs))
        return;

    count = lookahead.len;
    for (unsigned int i = 0; i < count; i++)
      if (!(this+lookahead[i]).intersects (c->glyphs))
        return;

    const ArrayOf<GlyphID> &substitute = StructAfter<ArrayOf<GlyphID> > (lookahead);
    Coverage::Iter iter;
    count = substitute.len;
    for (iter.init (this+coverage); iter.more (); iter.next ())
    {
      if (unlikely (iter.get_coverage () >= count))
        break; /* Work around malicious fonts. https://github.com/harfbuzz/harfbuzz/issues/363 */
      if (c->glyphs->has (iter.get_glyph ()))
	c->out->add (substitute[iter.get_coverage ()]);
    }
  }

  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    if (unlikely (!(this+coverage).add_coverage (c->input))) return;

    unsigned int count;

    count = backtrack.len;
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!(this+backtrack[i]).add_coverage (c->before))) return;

    const OffsetArrayOf<Coverage> &lookahead = StructAfter<OffsetArrayOf<Coverage> > (backtrack);
    count = lookahead.len;
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!(this+lookahead[i]).add_coverage (c->after))) return;

    const ArrayOf<GlyphID> &substitute = StructAfter<ArrayOf<GlyphID> > (lookahead);
    count = substitute.len;
    c->output->add_array (substitute.arrayZ, substitute.len);
  }

  inline const Coverage &get_coverage (void) const
  {
    return this+coverage;
  }

  inline bool would_apply (hb_would_apply_context_t *c) const
  {
    TRACE_WOULD_APPLY (this);
    return_trace (c->len == 1 && (this+coverage).get_coverage (c->glyphs[0]) != NOT_COVERED);
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    if (unlikely (c->nesting_level_left != HB_MAX_NESTING_LEVEL))
      return_trace (false); /* No chaining to this type */

    unsigned int index = (this+coverage).get_coverage (c->buffer->cur().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    const OffsetArrayOf<Coverage> &lookahead = StructAfter<OffsetArrayOf<Coverage> > (backtrack);
    const ArrayOf<GlyphID> &substitute = StructAfter<ArrayOf<GlyphID> > (lookahead);

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

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!(coverage.sanitize (c, this) && backtrack.sanitize (c, this)))
      return_trace (false);
    const OffsetArrayOf<Coverage> &lookahead = StructAfter<OffsetArrayOf<Coverage> > (backtrack);
    if (!lookahead.sanitize (c, this))
      return_trace (false);
    const ArrayOf<GlyphID> &substitute = StructAfter<ArrayOf<GlyphID> > (lookahead);
    return_trace (substitute.sanitize (c));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  OffsetTo<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of table */
  OffsetArrayOf<Coverage>
		backtrack;		/* Array of coverage tables
					 * in backtracking sequence, in  glyph
					 * sequence order */
  OffsetArrayOf<Coverage>
		lookaheadX;		/* Array of coverage tables
					 * in lookahead sequence, in glyph
					 * sequence order */
  ArrayOf<GlyphID>
		substituteX;		/* Array of substitute
					 * GlyphIDs--ordered by Coverage Index */
  public:
  DEFINE_SIZE_MIN (10);
};

struct ReverseChainSingleSubst
{
  template <typename context_t>
  inline typename context_t::return_t dispatch (context_t *c) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1));
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

  template <typename context_t>
  inline typename context_t::return_t dispatch (context_t *c, unsigned int lookup_type) const
  {
    TRACE_DISPATCH (this, lookup_type);
    if (unlikely (!c->may_dispatch (this, &u.sub_format))) return_trace (c->no_dispatch_return_value ());
    switch (lookup_type) {
    case Single:		return_trace (u.single.dispatch (c));
    case Multiple:		return_trace (u.multiple.dispatch (c));
    case Alternate:		return_trace (u.alternate.dispatch (c));
    case Ligature:		return_trace (u.ligature.dispatch (c));
    case Context:		return_trace (u.context.dispatch (c));
    case ChainContext:		return_trace (u.chainContext.dispatch (c));
    case Extension:		return_trace (u.extension.dispatch (c));
    case ReverseChainSingle:	return_trace (u.reverseChainContextSingle.dispatch (c));
    default:			return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT16			sub_format;
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
  DEFINE_SIZE_UNION (2, sub_format);
};


struct SubstLookup : Lookup
{
  inline const SubstLookupSubTable& get_subtable (unsigned int i) const
  { return Lookup::get_subtable<SubstLookupSubTable> (i); }

  inline static bool lookup_type_is_reverse (unsigned int lookup_type)
  { return lookup_type == SubstLookupSubTable::ReverseChainSingle; }

  inline bool is_reverse (void) const
  {
    unsigned int type = get_type ();
    if (unlikely (type == SubstLookupSubTable::Extension))
      return CastR<ExtensionSubst> (get_subtable(0)).is_reverse ();
    return lookup_type_is_reverse (type);
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    return_trace (dispatch (c));
  }

  inline hb_closure_context_t::return_t closure (hb_closure_context_t *c, unsigned int this_index) const
  {
    TRACE_CLOSURE (this);
    if (!c->should_visit_lookup (this_index))
      return_trace (HB_VOID);

    c->set_recurse_func (dispatch_closure_recurse_func);

    hb_closure_context_t::return_t ret = dispatch (c);

    c->flush ();

    return_trace (ret);
  }

  inline hb_collect_glyphs_context_t::return_t collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    c->set_recurse_func (dispatch_recurse_func<hb_collect_glyphs_context_t>);
    return_trace (dispatch (c));
  }

  template <typename set_t>
  inline void add_coverage (set_t *glyphs) const
  {
    hb_add_coverage_context_t<set_t> c (glyphs);
    dispatch (&c);
  }

  inline bool would_apply (hb_would_apply_context_t *c,
			   const hb_ot_layout_lookup_accelerator_t *accel) const
  {
    TRACE_WOULD_APPLY (this);
    if (unlikely (!c->len))  return_trace (false);
    if (!accel->may_have (c->glyphs[0]))  return_trace (false);
      return_trace (dispatch (c));
  }

  static bool apply_recurse_func (hb_ot_apply_context_t *c, unsigned int lookup_index);

  inline SubstLookupSubTable& serialize_subtable (hb_serialize_context_t *c,
						  unsigned int i)
  { return get_subtables<SubstLookupSubTable> ()[i].serialize (c, this); }

  inline bool serialize_single (hb_serialize_context_t *c,
				uint32_t lookup_props,
			        Supplier<GlyphID> &glyphs,
			        Supplier<GlyphID> &substitutes,
			        unsigned int num_glyphs)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!Lookup::serialize (c, SubstLookupSubTable::Single, lookup_props, 1))) return_trace (false);
    return_trace (serialize_subtable (c, 0).u.single.serialize (c, glyphs, substitutes, num_glyphs));
  }

  inline bool serialize_multiple (hb_serialize_context_t *c,
				  uint32_t lookup_props,
				  Supplier<GlyphID> &glyphs,
				  Supplier<unsigned int> &substitute_len_list,
				  unsigned int num_glyphs,
				  Supplier<GlyphID> &substitute_glyphs_list)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!Lookup::serialize (c, SubstLookupSubTable::Multiple, lookup_props, 1))) return_trace (false);
    return_trace (serialize_subtable (c, 0).u.multiple.serialize (c,
								  glyphs,
								  substitute_len_list,
								  num_glyphs,
								  substitute_glyphs_list));
  }

  inline bool serialize_alternate (hb_serialize_context_t *c,
				   uint32_t lookup_props,
				   Supplier<GlyphID> &glyphs,
				   Supplier<unsigned int> &alternate_len_list,
				   unsigned int num_glyphs,
				   Supplier<GlyphID> &alternate_glyphs_list)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!Lookup::serialize (c, SubstLookupSubTable::Alternate, lookup_props, 1))) return_trace (false);
    return_trace (serialize_subtable (c, 0).u.alternate.serialize (c,
								   glyphs,
								   alternate_len_list,
								   num_glyphs,
								   alternate_glyphs_list));
  }

  inline bool serialize_ligature (hb_serialize_context_t *c,
				  uint32_t lookup_props,
				  Supplier<GlyphID> &first_glyphs,
				  Supplier<unsigned int> &ligature_per_first_glyph_count_list,
				  unsigned int num_first_glyphs,
				  Supplier<GlyphID> &ligatures_list,
				  Supplier<unsigned int> &component_count_list,
				  Supplier<GlyphID> &component_list /* Starting from second for each ligature */)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!Lookup::serialize (c, SubstLookupSubTable::Ligature, lookup_props, 1))) return_trace (false);
    return_trace (serialize_subtable (c, 0).u.ligature.serialize (c,
								  first_glyphs,
								  ligature_per_first_glyph_count_list,
								  num_first_glyphs,
								  ligatures_list,
								  component_count_list,
								  component_list));
  }

  template <typename context_t>
  static inline typename context_t::return_t dispatch_recurse_func (context_t *c, unsigned int lookup_index);

  static inline hb_closure_context_t::return_t dispatch_closure_recurse_func (hb_closure_context_t *c, unsigned int lookup_index)
  {
    if (!c->should_visit_lookup (lookup_index))
      return HB_VOID;

    hb_closure_context_t::return_t ret = dispatch_recurse_func (c, lookup_index);

    c->flush ();

    return ret;
  }

  template <typename context_t>
  inline typename context_t::return_t dispatch (context_t *c) const
  { return Lookup::dispatch<SubstLookupSubTable> (c); }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!Lookup::sanitize (c))) return_trace (false);
    if (unlikely (!dispatch (c))) return_trace (false);

    if (unlikely (get_type () == SubstLookupSubTable::Extension))
    {
      /* The spec says all subtables of an Extension lookup should
       * have the same type, which shall not be the Extension type
       * itself (but we already checked for that).
       * This is specially important if one has a reverse type! */
      unsigned int type = get_subtable (0).u.extension.get_type ();
      unsigned int count = get_subtable_count ();
      for (unsigned int i = 1; i < count; i++)
        if (get_subtable (i).u.extension.get_type () != type)
	  return_trace (false);
    }
    return_trace (true);
  }
};

typedef OffsetListOf<SubstLookup> SubstLookupList;

/*
 * GSUB -- Glyph Substitution
 * https://docs.microsoft.com/en-us/typography/opentype/spec/gsub
 */

struct GSUB : GSUBGPOS
{
  static const hb_tag_t tableTag	= HB_OT_TAG_GSUB;

  inline const SubstLookup& get_lookup (unsigned int i) const
  { return CastR<SubstLookup> (GSUBGPOS::get_lookup (i)); }

  static inline void substitute_start (hb_font_t *font, hb_buffer_t *buffer);

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!GSUBGPOS::sanitize (c))) return_trace (false);
    const OffsetTo<SubstLookupList> &list = CastR<OffsetTo<SubstLookupList> > (lookupList);
    return_trace (list.sanitize (c, this));
  }
};


void
GSUB::substitute_start (hb_font_t *font, hb_buffer_t *buffer)
{
  _hb_buffer_assert_gsubgpos_vars (buffer);

  const GDEF &gdef = *hb_ot_layout_from_face (font->face)->table.GDEF;
  unsigned int count = buffer->len;
  for (unsigned int i = 0; i < count; i++)
  {
    _hb_glyph_info_set_glyph_props (&buffer->info[i], gdef.get_glyph_props (buffer->info[i].codepoint));
    _hb_glyph_info_clear_lig_props (&buffer->info[i]);
    buffer->info[i].syllable() = 0;
  }
}


/* Out-of-class implementation for methods recursing */

/*static*/ inline bool ExtensionSubst::is_reverse (void) const
{
  unsigned int type = get_type ();
  if (unlikely (type == SubstLookupSubTable::Extension))
    return CastR<ExtensionSubst> (get_subtable<LookupSubTable>()).is_reverse ();
  return SubstLookup::lookup_type_is_reverse (type);
}

template <typename context_t>
/*static*/ inline typename context_t::return_t SubstLookup::dispatch_recurse_func (context_t *c, unsigned int lookup_index)
{
  const GSUB &gsub = *(hb_ot_layout_from_face (c->face)->table.GSUB);
  const SubstLookup &l = gsub.get_lookup (lookup_index);
  return l.dispatch (c);
}

/*static*/ inline bool SubstLookup::apply_recurse_func (hb_ot_apply_context_t *c, unsigned int lookup_index)
{
  const GSUB &gsub = *(hb_ot_layout_from_face (c->face)->table.GSUB);
  const SubstLookup &l = gsub.get_lookup (lookup_index);
  unsigned int saved_lookup_props = c->lookup_props;
  unsigned int saved_lookup_index = c->lookup_index;
  c->set_lookup_index (lookup_index);
  c->set_lookup_props (l.get_props ());
  bool ret = l.dispatch (c);
  c->set_lookup_index (saved_lookup_index);
  c->set_lookup_props (saved_lookup_props);
  return ret;
}


} /* namespace OT */


#endif /* HB_OT_LAYOUT_GSUB_TABLE_HH */

/*
 * Copyright © 2011,2012  Google, Inc.
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
 * Google Author(s): Behdad Esfahbod
 */

#include "hb.hh"

#ifndef HB_NO_OT_SHAPE

#include "hb-ot-shape-fallback.hh"
#include "hb-kern.hh"

static unsigned int
recategorize_combining_class (hb_codepoint_t u,
			      unsigned int klass)
{
  if (klass >= 200)
    return klass;

  /* Thai / Lao need some per-character work. */
  if ((u & ~0xFF) == 0x0E00u)
  {
    if (unlikely (klass == 0))
    {
      switch (u)
      {
	case 0x0E31u:
	case 0x0E34u:
	case 0x0E35u:
	case 0x0E36u:
	case 0x0E37u:
	case 0x0E47u:
	case 0x0E4Cu:
	case 0x0E4Du:
	case 0x0E4Eu:
	  klass = HB_UNICODE_COMBINING_CLASS_ABOVE_RIGHT;
	  break;

	case 0x0EB1u:
	case 0x0EB4u:
	case 0x0EB5u:
	case 0x0EB6u:
	case 0x0EB7u:
	case 0x0EBBu:
	case 0x0ECCu:
	case 0x0ECDu:
	  klass = HB_UNICODE_COMBINING_CLASS_ABOVE;
	  break;

	case 0x0EBCu:
	  klass = HB_UNICODE_COMBINING_CLASS_BELOW;
	  break;
      }
    } else {
      /* Thai virama is below-right */
      if (u == 0x0E3Au)
	klass = HB_UNICODE_COMBINING_CLASS_BELOW_RIGHT;
    }
  }

  switch (klass)
  {

    /* Hebrew */

    case HB_MODIFIED_COMBINING_CLASS_CCC10: /* sheva */
    case HB_MODIFIED_COMBINING_CLASS_CCC11: /* hataf segol */
    case HB_MODIFIED_COMBINING_CLASS_CCC12: /* hataf patah */
    case HB_MODIFIED_COMBINING_CLASS_CCC13: /* hataf qamats */
    case HB_MODIFIED_COMBINING_CLASS_CCC14: /* hiriq */
    case HB_MODIFIED_COMBINING_CLASS_CCC15: /* tsere */
    case HB_MODIFIED_COMBINING_CLASS_CCC16: /* segol */
    case HB_MODIFIED_COMBINING_CLASS_CCC17: /* patah */
    case HB_MODIFIED_COMBINING_CLASS_CCC18: /* qamats & qamats qatan */
    case HB_MODIFIED_COMBINING_CLASS_CCC20: /* qubuts */
    case HB_MODIFIED_COMBINING_CLASS_CCC22: /* meteg */
      return HB_UNICODE_COMBINING_CLASS_BELOW;

    case HB_MODIFIED_COMBINING_CLASS_CCC23: /* rafe */
      return HB_UNICODE_COMBINING_CLASS_ATTACHED_ABOVE;

    case HB_MODIFIED_COMBINING_CLASS_CCC24: /* shin dot */
      return HB_UNICODE_COMBINING_CLASS_ABOVE_RIGHT;

    case HB_MODIFIED_COMBINING_CLASS_CCC25: /* sin dot */
    case HB_MODIFIED_COMBINING_CLASS_CCC19: /* holam & holam haser for vav */
      return HB_UNICODE_COMBINING_CLASS_ABOVE_LEFT;

    case HB_MODIFIED_COMBINING_CLASS_CCC26: /* point varika */
      return HB_UNICODE_COMBINING_CLASS_ABOVE;

    case HB_MODIFIED_COMBINING_CLASS_CCC21: /* dagesh */
      break;


    /* Arabic and Syriac */

    case HB_MODIFIED_COMBINING_CLASS_CCC27: /* fathatan */
    case HB_MODIFIED_COMBINING_CLASS_CCC28: /* dammatan */
    case HB_MODIFIED_COMBINING_CLASS_CCC30: /* fatha */
    case HB_MODIFIED_COMBINING_CLASS_CCC31: /* damma */
    case HB_MODIFIED_COMBINING_CLASS_CCC33: /* shadda */
    case HB_MODIFIED_COMBINING_CLASS_CCC34: /* sukun */
    case HB_MODIFIED_COMBINING_CLASS_CCC35: /* superscript alef */
    case HB_MODIFIED_COMBINING_CLASS_CCC36: /* superscript alaph */
      return HB_UNICODE_COMBINING_CLASS_ABOVE;

    case HB_MODIFIED_COMBINING_CLASS_CCC29: /* kasratan */
    case HB_MODIFIED_COMBINING_CLASS_CCC32: /* kasra */
      return HB_UNICODE_COMBINING_CLASS_BELOW;


    /* Thai */

    case HB_MODIFIED_COMBINING_CLASS_CCC103: /* sara u / sara uu */
      return HB_UNICODE_COMBINING_CLASS_BELOW_RIGHT;

    case HB_MODIFIED_COMBINING_CLASS_CCC107: /* mai */
      return HB_UNICODE_COMBINING_CLASS_ABOVE_RIGHT;


    /* Lao */

    case HB_MODIFIED_COMBINING_CLASS_CCC118: /* sign u / sign uu */
      return HB_UNICODE_COMBINING_CLASS_BELOW;

    case HB_MODIFIED_COMBINING_CLASS_CCC122: /* mai */
      return HB_UNICODE_COMBINING_CLASS_ABOVE;


    /* Tibetan */

    case HB_MODIFIED_COMBINING_CLASS_CCC129: /* sign aa */
      return HB_UNICODE_COMBINING_CLASS_BELOW;

    case HB_MODIFIED_COMBINING_CLASS_CCC130: /* sign i*/
      return HB_UNICODE_COMBINING_CLASS_ABOVE;

    case HB_MODIFIED_COMBINING_CLASS_CCC132: /* sign u */
      return HB_UNICODE_COMBINING_CLASS_BELOW;

  }

  return klass;
}

void
_hb_ot_shape_fallback_mark_position_recategorize_marks (const hb_ot_shape_plan_t *plan HB_UNUSED,
							hb_font_t *font HB_UNUSED,
							hb_buffer_t  *buffer)
{
#ifdef HB_NO_OT_SHAPE_FALLBACK
  return;
#endif

  unsigned int count = buffer->len;
  hb_glyph_info_t *info = buffer->info;
  for (unsigned int i = 0; i < count; i++)
    if (_hb_glyph_info_get_general_category (&info[i]) == HB_UNICODE_GENERAL_CATEGORY_NON_SPACING_MARK) {
      unsigned int combining_class = _hb_glyph_info_get_modified_combining_class (&info[i]);
      combining_class = recategorize_combining_class (info[i].codepoint, combining_class);
      _hb_glyph_info_set_modified_combining_class (&info[i], combining_class);
    }
}


static void
zero_mark_advances (hb_buffer_t *buffer,
		    unsigned int start,
		    unsigned int end,
		    bool adjust_offsets_when_zeroing)
{
  hb_glyph_info_t *info = buffer->info;
  for (unsigned int i = start; i < end; i++)
    if (_hb_glyph_info_get_general_category (&info[i]) == HB_UNICODE_GENERAL_CATEGORY_NON_SPACING_MARK)
    {
      if (adjust_offsets_when_zeroing)
      {
	buffer->pos[i].x_offset -= buffer->pos[i].x_advance;
	buffer->pos[i].y_offset -= buffer->pos[i].y_advance;
      }
      buffer->pos[i].x_advance = 0;
      buffer->pos[i].y_advance = 0;
    }
}

static inline void
position_mark (const hb_ot_shape_plan_t *plan HB_UNUSED,
	       hb_font_t *font,
	       hb_buffer_t  *buffer,
	       hb_glyph_extents_t &base_extents,
	       unsigned int i,
	       unsigned int combining_class)
{
  hb_glyph_extents_t mark_extents;
  if (!font->get_glyph_extents (buffer->info[i].codepoint, &mark_extents))
    return;

  hb_position_t y_gap = font->y_scale / 16;

  hb_glyph_position_t &pos = buffer->pos[i];
  pos.x_offset = pos.y_offset = 0;


  /* We don't position LEFT and RIGHT marks. */

  /* X positioning */
  switch (combining_class)
  {
    case HB_UNICODE_COMBINING_CLASS_DOUBLE_BELOW:
    case HB_UNICODE_COMBINING_CLASS_DOUBLE_ABOVE:
      if (buffer->props.direction == HB_DIRECTION_LTR) {
	pos.x_offset += base_extents.x_bearing + base_extents.width - mark_extents.width / 2 - mark_extents.x_bearing;
	break;
      } else if (buffer->props.direction == HB_DIRECTION_RTL) {
	pos.x_offset += base_extents.x_bearing - mark_extents.width / 2 - mark_extents.x_bearing;
	break;
      }
      HB_FALLTHROUGH;

    default:
    case HB_UNICODE_COMBINING_CLASS_ATTACHED_BELOW:
    case HB_UNICODE_COMBINING_CLASS_ATTACHED_ABOVE:
    case HB_UNICODE_COMBINING_CLASS_BELOW:
    case HB_UNICODE_COMBINING_CLASS_ABOVE:
      /* Center align. */
      pos.x_offset += base_extents.x_bearing + (base_extents.width - mark_extents.width) / 2 - mark_extents.x_bearing;
      break;

    case HB_UNICODE_COMBINING_CLASS_ATTACHED_BELOW_LEFT:
    case HB_UNICODE_COMBINING_CLASS_BELOW_LEFT:
    case HB_UNICODE_COMBINING_CLASS_ABOVE_LEFT:
      /* Left align. */
      pos.x_offset += base_extents.x_bearing - mark_extents.x_bearing;
      break;

    case HB_UNICODE_COMBINING_CLASS_ATTACHED_ABOVE_RIGHT:
    case HB_UNICODE_COMBINING_CLASS_BELOW_RIGHT:
    case HB_UNICODE_COMBINING_CLASS_ABOVE_RIGHT:
      /* Right align. */
      pos.x_offset += base_extents.x_bearing + base_extents.width - mark_extents.width - mark_extents.x_bearing;
      break;
  }

  /* Y positioning */
  switch (combining_class)
  {
    case HB_UNICODE_COMBINING_CLASS_DOUBLE_BELOW:
    case HB_UNICODE_COMBINING_CLASS_BELOW_LEFT:
    case HB_UNICODE_COMBINING_CLASS_BELOW:
    case HB_UNICODE_COMBINING_CLASS_BELOW_RIGHT:
      /* Add gap, fall-through. */
      base_extents.height -= y_gap;
      HB_FALLTHROUGH;

    case HB_UNICODE_COMBINING_CLASS_ATTACHED_BELOW_LEFT:
    case HB_UNICODE_COMBINING_CLASS_ATTACHED_BELOW:
      pos.y_offset = base_extents.y_bearing + base_extents.height - mark_extents.y_bearing;
      /* Never shift up "below" marks. */
      if ((y_gap > 0) == (pos.y_offset > 0))
      {
	base_extents.height -= pos.y_offset;
	pos.y_offset = 0;
      }
      base_extents.height += mark_extents.height;
      break;

    case HB_UNICODE_COMBINING_CLASS_DOUBLE_ABOVE:
    case HB_UNICODE_COMBINING_CLASS_ABOVE_LEFT:
    case HB_UNICODE_COMBINING_CLASS_ABOVE:
    case HB_UNICODE_COMBINING_CLASS_ABOVE_RIGHT:
      /* Add gap, fall-through. */
      base_extents.y_bearing += y_gap;
      base_extents.height -= y_gap;
      HB_FALLTHROUGH;

    case HB_UNICODE_COMBINING_CLASS_ATTACHED_ABOVE:
    case HB_UNICODE_COMBINING_CLASS_ATTACHED_ABOVE_RIGHT:
      pos.y_offset = base_extents.y_bearing - (mark_extents.y_bearing + mark_extents.height);
      /* Don't shift down "above" marks too much. */
      if ((y_gap > 0) != (pos.y_offset > 0))
      {
	int correction = -pos.y_offset / 2;
	base_extents.y_bearing += correction;
	base_extents.height -= correction;
	pos.y_offset += correction;
      }
      base_extents.y_bearing -= mark_extents.height;
      base_extents.height += mark_extents.height;
      break;
  }
}

static inline void
position_around_base (const hb_ot_shape_plan_t *plan,
		      hb_font_t *font,
		      hb_buffer_t  *buffer,
		      unsigned int base,
		      unsigned int end,
		      bool adjust_offsets_when_zeroing)
{
  hb_direction_t horiz_dir = HB_DIRECTION_INVALID;

  buffer->unsafe_to_break (base, end);

  hb_glyph_extents_t base_extents;
  if (!font->get_glyph_extents (buffer->info[base].codepoint,
				&base_extents))
  {
    /* If extents don't work, zero marks and go home. */
    zero_mark_advances (buffer, base + 1, end, adjust_offsets_when_zeroing);
    return;
  }
  base_extents.y_bearing += buffer->pos[base].y_offset;
  /* Use horizontal advance for horizontal positioning.
   * Generally a better idea.  Also works for zero-ink glyphs.  See:
   * https://github.com/harfbuzz/harfbuzz/issues/1532 */
  base_extents.x_bearing = 0;
  base_extents.width = font->get_glyph_h_advance (buffer->info[base].codepoint);

  unsigned int lig_id = _hb_glyph_info_get_lig_id (&buffer->info[base]);
  /* Use integer for num_lig_components such that it doesn't convert to unsigned
   * when we divide or multiply by it. */
  int num_lig_components = _hb_glyph_info_get_lig_num_comps (&buffer->info[base]);

  hb_position_t x_offset = 0, y_offset = 0;
  if (HB_DIRECTION_IS_FORWARD (buffer->props.direction)) {
    x_offset -= buffer->pos[base].x_advance;
    y_offset -= buffer->pos[base].y_advance;
  }

  hb_glyph_extents_t component_extents = base_extents;
  int last_lig_component = -1;
  unsigned int last_combining_class = 255;
  hb_glyph_extents_t cluster_extents = base_extents; /* Initialization is just to shut gcc up. */
  hb_glyph_info_t *info = buffer->info;
  for (unsigned int i = base + 1; i < end; i++)
    if (_hb_glyph_info_get_modified_combining_class (&info[i]))
    {
      if (num_lig_components > 1) {
	unsigned int this_lig_id = _hb_glyph_info_get_lig_id (&info[i]);
	int this_lig_component = _hb_glyph_info_get_lig_comp (&info[i]) - 1;
	/* Conditions for attaching to the last component. */
	if (!lig_id || lig_id != this_lig_id || this_lig_component >= num_lig_components)
	  this_lig_component = num_lig_components - 1;
	if (last_lig_component != this_lig_component)
	{
	  last_lig_component = this_lig_component;
	  last_combining_class = 255;
	  component_extents = base_extents;
	  if (unlikely (horiz_dir == HB_DIRECTION_INVALID)) {
	    if (HB_DIRECTION_IS_HORIZONTAL (plan->props.direction))
	      horiz_dir = plan->props.direction;
	    else
	      horiz_dir = hb_script_get_horizontal_direction (plan->props.script);
	  }
	  if (horiz_dir == HB_DIRECTION_LTR)
	    component_extents.x_bearing += (this_lig_component * component_extents.width) / num_lig_components;
	  else
	    component_extents.x_bearing += ((num_lig_components - 1 - this_lig_component) * component_extents.width) / num_lig_components;
	  component_extents.width /= num_lig_components;
	}
      }

      unsigned int this_combining_class = _hb_glyph_info_get_modified_combining_class (&info[i]);
      if (last_combining_class != this_combining_class)
      {
	last_combining_class = this_combining_class;
	cluster_extents = component_extents;
      }

      position_mark (plan, font, buffer, cluster_extents, i, this_combining_class);

      buffer->pos[i].x_advance = 0;
      buffer->pos[i].y_advance = 0;
      buffer->pos[i].x_offset += x_offset;
      buffer->pos[i].y_offset += y_offset;

    } else {
      if (HB_DIRECTION_IS_FORWARD (buffer->props.direction)) {
	x_offset -= buffer->pos[i].x_advance;
	y_offset -= buffer->pos[i].y_advance;
      } else {
	x_offset += buffer->pos[i].x_advance;
	y_offset += buffer->pos[i].y_advance;
      }
    }
}

static inline void
position_cluster (const hb_ot_shape_plan_t *plan,
		  hb_font_t *font,
		  hb_buffer_t  *buffer,
		  unsigned int start,
		  unsigned int end,
		  bool adjust_offsets_when_zeroing)
{
  if (end - start < 2)
    return;

  /* Find the base glyph */
  hb_glyph_info_t *info = buffer->info;
  for (unsigned int i = start; i < end; i++)
    if (!_hb_glyph_info_is_unicode_mark (&info[i]))
    {
      /* Find mark glyphs */
      unsigned int j;
      for (j = i + 1; j < end; j++)
	if (!_hb_glyph_info_is_unicode_mark (&info[j]))
	  break;

      position_around_base (plan, font, buffer, i, j, adjust_offsets_when_zeroing);

      i = j - 1;
    }
}

void
_hb_ot_shape_fallback_mark_position (const hb_ot_shape_plan_t *plan,
				     hb_font_t *font,
				     hb_buffer_t  *buffer,
				     bool adjust_offsets_when_zeroing)
{
#ifdef HB_NO_OT_SHAPE_FALLBACK
  return;
#endif

  if (!buffer->message (font, "start fallback mark"))
    return;

  _hb_buffer_assert_gsubgpos_vars (buffer);

  unsigned int start = 0;
  unsigned int count = buffer->len;
  hb_glyph_info_t *info = buffer->info;
  for (unsigned int i = 1; i < count; i++)
    if (likely (!_hb_glyph_info_is_unicode_mark (&info[i]))) {
      position_cluster (plan, font, buffer, start, i, adjust_offsets_when_zeroing);
      start = i;
    }
  position_cluster (plan, font, buffer, start, count, adjust_offsets_when_zeroing);

  (void) buffer->message (font, "end fallback mark");
}


#ifndef HB_DISABLE_DEPRECATED
struct hb_ot_shape_fallback_kern_driver_t
{
  hb_ot_shape_fallback_kern_driver_t (hb_font_t   *font_,
				      hb_buffer_t *buffer) :
    font (font_), direction (buffer->props.direction) {}

  hb_position_t get_kerning (hb_codepoint_t first, hb_codepoint_t second) const
  {
    hb_position_t kern = 0;
    font->get_glyph_kerning_for_direction (first, second,
					   direction,
					   &kern, &kern);
    return kern;
  }

  hb_font_t *font;
  hb_direction_t direction;
};
#endif

/* Performs font-assisted kerning. */
void
_hb_ot_shape_fallback_kern (const hb_ot_shape_plan_t *plan,
			    hb_font_t *font,
			    hb_buffer_t *buffer)
{
#ifdef HB_NO_OT_SHAPE_FALLBACK
  return;
#endif

#ifndef HB_DISABLE_DEPRECATED
  if (HB_DIRECTION_IS_HORIZONTAL (buffer->props.direction) ?
      !font->has_glyph_h_kerning_func () :
      !font->has_glyph_v_kerning_func ())
    return;

  if (!buffer->message (font, "start fallback kern"))
    return;

  bool reverse = HB_DIRECTION_IS_BACKWARD (buffer->props.direction);

  if (reverse)
    buffer->reverse ();

  hb_ot_shape_fallback_kern_driver_t driver (font, buffer);
  OT::hb_kern_machine_t<hb_ot_shape_fallback_kern_driver_t> machine (driver);
  machine.kern (font, buffer, plan->kern_mask, false);

  if (reverse)
    buffer->reverse ();

  (void) buffer->message (font, "end fallback kern");
#endif
}


/* Adjusts width of various spaces. */
void
_hb_ot_shape_fallback_spaces (const hb_ot_shape_plan_t *plan HB_UNUSED,
			      hb_font_t *font,
			      hb_buffer_t  *buffer)
{
  hb_glyph_info_t *info = buffer->info;
  hb_glyph_position_t *pos = buffer->pos;
  bool horizontal = HB_DIRECTION_IS_HORIZONTAL (buffer->props.direction);
  unsigned int count = buffer->len;
  for (unsigned int i = 0; i < count; i++)
    if (_hb_glyph_info_is_unicode_space (&info[i]) && !_hb_glyph_info_ligated (&info[i]))
    {
      /* If font had no ASCII space and we used the invisible glyph, give it a 1/4 EM default advance. */
      if (buffer->invisible && info[i].codepoint == buffer->invisible)
      {
        if (horizontal)
	  pos[i].x_advance = +font->x_scale / 4;
        else
	  pos[i].y_advance = -font->y_scale / 4;
      }

      hb_unicode_funcs_t::space_t space_type = _hb_glyph_info_get_unicode_space_fallback_type (&info[i]);
      hb_codepoint_t glyph;
      typedef hb_unicode_funcs_t t;
      switch (space_type)
      {
	case t::NOT_SPACE: /* Shouldn't happen. */
	case t::SPACE:
	  break;

	case t::SPACE_EM:
	case t::SPACE_EM_2:
	case t::SPACE_EM_3:
	case t::SPACE_EM_4:
	case t::SPACE_EM_5:
	case t::SPACE_EM_6:
	case t::SPACE_EM_16:
	  if (horizontal)
	    pos[i].x_advance = +(font->x_scale + ((int) space_type)/2) / (int) space_type;
	  else
	    pos[i].y_advance = -(font->y_scale + ((int) space_type)/2) / (int) space_type;
	  break;

	case t::SPACE_4_EM_18:
	  if (horizontal)
	    pos[i].x_advance = (int64_t) +font->x_scale * 4 / 18;
	  else
	    pos[i].y_advance = (int64_t) -font->y_scale * 4 / 18;
	  break;

	case t::SPACE_FIGURE:
	  for (char u = '0'; u <= '9'; u++)
	    if (font->get_nominal_glyph (u, &glyph))
	    {
	      if (horizontal)
		pos[i].x_advance = font->get_glyph_h_advance (glyph);
	      else
		pos[i].y_advance = font->get_glyph_v_advance (glyph);
	      break;
	    }
	  break;

	case t::SPACE_PUNCTUATION:
	  if (font->get_nominal_glyph ('.', &glyph) ||
	      font->get_nominal_glyph (',', &glyph))
	  {
	    if (horizontal)
	      pos[i].x_advance = font->get_glyph_h_advance (glyph);
	    else
	      pos[i].y_advance = font->get_glyph_v_advance (glyph);
	  }
	  break;

	case t::SPACE_NARROW:
	  /* Half-space?
	   * Unicode doc https://unicode.org/charts/PDF/U2000.pdf says ~1/4 or 1/5 of EM.
	   * However, in my testing, many fonts have their regular space being about that
	   * size.  To me, a percentage of the space width makes more sense.  Half is as
	   * good as any. */
	  if (horizontal)
	    pos[i].x_advance /= 2;
	  else
	    pos[i].y_advance /= 2;
	  break;
      }
    }
}


#endif

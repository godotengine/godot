/*
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
 * Author(s): Khaled Hosny
 */

#include "hb.hh"

#if HAVE_KBTS

#include "hb-shaper-impl.hh"

#define KB_TEXT_SHAPE_IMPLEMENTATION
#define KB_TEXT_SHAPE_STATIC
#define KB_TEXT_SHAPE_NO_CRT
#define KBTS_MEMSET hb_memset

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "kb_text_shape.h"
#pragma GCC diagnostic pop


hb_kbts_face_data_t *
_hb_kbts_shaper_face_data_create (hb_face_t *face)
{
  hb_blob_t *blob = hb_face_reference_blob (face);

  unsigned int blob_length;
  const char *blob_data = hb_blob_get_data (blob, &blob_length);
  if (unlikely (!blob_length))
  {
    DEBUG_MSG (KBTS, blob, "Empty blob");
    hb_blob_destroy (blob);
    return nullptr;
  }

  void *data = hb_malloc (blob_length);
  if (likely (data))
    hb_memcpy (data, blob_data, blob_length);

  hb_blob_destroy (blob);
  blob = nullptr;

  if (unlikely (!data))
  {
    DEBUG_MSG (KBTS, face, "Failed to allocate memory for font data");
    return nullptr;
  }

  kbts_font *kb_font = (kbts_font *) hb_calloc (1, sizeof (kbts_font));
  if (unlikely (!kb_font))
  {
    hb_free (data);
    return nullptr;
  }

  size_t memory_size;
  {
    unsigned scratch_size = kbts_ReadFontHeader (kb_font, data, blob_length);
    void *scratch = hb_malloc (scratch_size);
    memory_size = kbts_ReadFontData (kb_font, scratch, scratch_size);
    hb_free (scratch);
  }

  void *memory = hb_malloc (memory_size);
  if (unlikely (!kbts_PostReadFontInitialize (kb_font, memory, memory_size)))
  {
    DEBUG_MSG (KBTS, face, "kbts_PostReadFontInitialize failed");
    hb_free (memory);
    hb_free (data);
    hb_free (kb_font);
    return nullptr;
  }

  return (hb_kbts_face_data_t *) kb_font;
}

void
_hb_kbts_shaper_face_data_destroy (hb_kbts_face_data_t *data)
{
  kbts_font *font = (kbts_font *) data;

  assert (kbts_FontIsValid (font));

  hb_free (font->FileBase);
  hb_free (font->GlyphLookupMatrix);
  hb_free (font);
}

hb_kbts_font_data_t *
_hb_kbts_shaper_font_data_create (hb_font_t *font)
{
  return (hb_kbts_font_data_t *) HB_SHAPER_DATA_SUCCEEDED;
}

void
_hb_kbts_shaper_font_data_destroy (hb_kbts_font_data_t *data)
{
}

hb_bool_t
_hb_kbts_shape (hb_shape_plan_t    *shape_plan,
		hb_font_t          *font,
		hb_buffer_t        *buffer,
		const hb_feature_t *features,
		unsigned int        num_features)
{
  hb_face_t *face = font->face;
  kbts_font *kb_font = (kbts_font *) (const void *) face->data.kbts;

  kbts_direction kb_direction;
  switch (buffer->props.direction)
  {
    case HB_DIRECTION_LTR: kb_direction = KBTS_DIRECTION_LTR; break;
    case HB_DIRECTION_RTL: kb_direction = KBTS_DIRECTION_RTL; break;
    case HB_DIRECTION_TTB:
    case HB_DIRECTION_BTT:
      DEBUG_MSG (KBTS, face, "Vertical direction is not supported");
      return false;
    default:
    case HB_DIRECTION_INVALID:
      DEBUG_MSG (KBTS, face, "Invalid direction");
      return false;
  }

  kbts_script kb_script = KBTS_SCRIPT_DONT_KNOW;
  kbts_language kb_language = KBTS_LANGUAGE_DEFAULT;
  {
    hb_tag_t scripts[HB_OT_MAX_TAGS_PER_SCRIPT];
    hb_tag_t language;
    unsigned int script_count = ARRAY_LENGTH (scripts);
    unsigned int language_count = 1;

    hb_ot_tags_from_script_and_language (buffer->props.script, buffer->props.language,
					 &script_count, scripts,
					 &language_count, &language);

    for (unsigned int i = 0; i < script_count && scripts[i] != HB_TAG_NONE; ++i)
    {
      kb_script = kbts_ScriptTagToScript (hb_uint32_swap (scripts[i]));
      if (kb_script != KBTS_SCRIPT_DONT_KNOW)
        break;
    }

    if (language_count)
      kb_language = (kbts_language) hb_uint32_swap (language);
  }

  hb_vector_t<kbts_glyph> kb_glyphs;
  if (unlikely (!kb_glyphs.resize_full (buffer->len, false, true)))
    return false;

  for (size_t i = 0; i < buffer->len; ++i)
    kb_glyphs.arrayZ[i] = kbts_CodepointToGlyph (kb_font, buffer->info[i].codepoint);

  if (num_features)
  {
    for (unsigned int i = 0; i < num_features; ++i)
    {
      hb_feature_t feature = features[i];
      for (unsigned int j = 0; j < buffer->len; ++j)
      {
	kbts_glyph *kb_glyph = &kb_glyphs.arrayZ[j];
	if (hb_in_range (j, feature.start, feature.end))
	{
	  if (!kb_glyph->Config)
	    kb_glyph->Config = (kbts_glyph_config *) hb_calloc (1, sizeof (kbts_glyph_config));
	  kbts_glyph_config *config = kb_glyph->Config;
	  while (!kbts_GlyphConfigOverrideFeatureFromTag (config, hb_uint32_swap (feature.tag),
	                                                  feature.value > 1, feature.value))
	  {
	    config->FeatureOverrides = (kbts_feature_override *) hb_realloc (config->FeatureOverrides,
									     config->RequiredFeatureOverrideCapacity);
	    config->FeatureOverrideCapacity += 1;
	  }
	}
      }
    }
  }

  kbts_shape_state *kb_shape_state;
  {
    size_t kb_shape_state_size = kbts_SizeOfShapeState (kb_font);
    void *kb_shape_state_buffer = hb_malloc (kb_shape_state_size);
    if (unlikely (!kb_shape_state_buffer))
    {
      DEBUG_MSG (KBTS, face, "Failed to allocate memory for shape state");
      return false;
    }
    kb_shape_state = kbts_PlaceShapeState (kb_shape_state_buffer, kb_shape_state_size);
  }
  kbts_shape_config kb_shape_config = kbts_ShapeConfig (kb_font, kb_script, kb_language);
  uint32_t glyph_count = buffer->len;
  uint32_t glyph_capacity = kb_glyphs.length;
  while (kbts_Shape (kb_shape_state, &kb_shape_config, KBTS_DIRECTION_LTR, kb_direction,
		     kb_glyphs.arrayZ, &glyph_count, glyph_capacity))
  {
    glyph_capacity = kb_shape_state->RequiredGlyphCapacity;
    /* kb increases capacity by a fixed number only. We increase it by 50% to
     * avoid O(n^2) behavior in case of expanding text.
     *
     * https://github.com/JimmyLefevre/kb/issues/32
     */
    glyph_capacity += glyph_capacity / 2;
    if (unlikely (!kb_glyphs.resize_full (glyph_capacity, false, true)))
      return false;
  }

  hb_buffer_set_content_type (buffer, HB_BUFFER_CONTENT_TYPE_GLYPHS);
  hb_buffer_set_length (buffer, glyph_count);

  hb_glyph_info_t *info = buffer->info;
  hb_glyph_position_t *pos = buffer->pos;

  buffer->clear_positions ();
  for (size_t i = 0; i < glyph_count; ++i)
  {
    kbts_glyph kb_glyph = kb_glyphs.arrayZ[i];
    info[i].codepoint = kb_glyph.Id;
    info[i].cluster = 0; // FIXME
    pos[i].x_advance = font->em_scalef_x (kb_glyph.AdvanceX);
    pos[i].y_advance = font->em_scalef_y (kb_glyph.AdvanceY);
    pos[i].x_offset = font->em_scalef_x (kb_glyph.OffsetX);
    pos[i].y_offset = font->em_scalef_y (kb_glyph.OffsetY);

    if (kb_glyph.Config)
      hb_free (kb_glyph.Config->FeatureOverrides);
  }

  hb_free (kb_shape_state);

  buffer->clear_glyph_flags ();
  buffer->unsafe_to_break ();

  return true;
}

#endif
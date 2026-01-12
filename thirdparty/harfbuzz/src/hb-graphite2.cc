/*
 * Copyright © 2011  Martin Hosken
 * Copyright © 2011  SIL International
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

#ifdef HAVE_GRAPHITE2

#include "hb-shaper-impl.hh"

#include "hb-graphite2.h"

#include <graphite2/Segment.h>

#include "hb-ot-layout.h"


/**
 * SECTION:hb-graphite2
 * @title: hb-graphite2
 * @short_description: Graphite2 integration
 * @include: hb-graphite2.h
 *
 * Functions for using HarfBuzz with fonts that include Graphite features.
 * 
 * For Graphite features to work, you must be sure that HarfBuzz was compiled
 * with the `graphite2` shaping engine enabled. Currently, the default is to
 * not enable `graphite2` shaping.
 **/


/*
 * shaper face data
 */

typedef struct hb_graphite2_tablelist_t
{
  struct hb_graphite2_tablelist_t *next;
  hb_blob_t *blob;
  unsigned int tag;
} hb_graphite2_tablelist_t;

struct hb_graphite2_face_data_t
{
  hb_face_t *face;
  gr_face   *grface;
  hb_atomic_t<hb_graphite2_tablelist_t *> tlist;
};

static const void *hb_graphite2_get_table (const void *data, unsigned int tag, size_t *len)
{
  hb_graphite2_face_data_t *face_data = (hb_graphite2_face_data_t *) data;
  hb_graphite2_tablelist_t *tlist = face_data->tlist;

  hb_blob_t *blob = nullptr;

  for (hb_graphite2_tablelist_t *p = tlist; p; p = p->next)
    if (p->tag == tag) {
      blob = p->blob;
      break;
    }

  if (unlikely (!blob))
  {
    blob = face_data->face->reference_table (tag);

    hb_graphite2_tablelist_t *p = (hb_graphite2_tablelist_t *) hb_calloc (1, sizeof (hb_graphite2_tablelist_t));
    if (unlikely (!p)) {
      hb_blob_destroy (blob);
      return nullptr;
    }
    p->blob = blob;
    p->tag = tag;

retry:
    hb_graphite2_tablelist_t *tlist = face_data->tlist;
    p->next = tlist;

    if (unlikely (!face_data->tlist.cmpexch (tlist, p)))
      goto retry;
  }

  unsigned int tlen;
  const char *d = hb_blob_get_data (blob, &tlen);
  *len = tlen;
  return d;
}

hb_graphite2_face_data_t *
_hb_graphite2_shaper_face_data_create (hb_face_t *face)
{
  hb_blob_t *silf_blob = face->reference_table (HB_GRAPHITE2_TAG_SILF);
  /* Umm, we just reference the table to check whether it exists.
   * Maybe add better API for this? */
  if (!hb_blob_get_length (silf_blob))
  {
    hb_blob_destroy (silf_blob);
    return nullptr;
  }
  hb_blob_destroy (silf_blob);

  hb_graphite2_face_data_t *data = (hb_graphite2_face_data_t *) hb_calloc (1, sizeof (hb_graphite2_face_data_t));
  if (unlikely (!data))
    return nullptr;

  data->face = face;
  const gr_face_ops ops = {sizeof(gr_face_ops), &hb_graphite2_get_table, NULL};
  data->grface = gr_make_face_with_ops (data, &ops, gr_face_preloadAll);

  if (unlikely (!data->grface)) {
    hb_free (data);
    return nullptr;
  }

  return data;
}

void
_hb_graphite2_shaper_face_data_destroy (hb_graphite2_face_data_t *data)
{
  hb_graphite2_tablelist_t *tlist = data->tlist;

  while (tlist)
  {
    hb_graphite2_tablelist_t *old = tlist;
    hb_blob_destroy (tlist->blob);
    tlist = tlist->next;
    hb_free (old);
  }

  gr_face_destroy (data->grface);

  hb_free (data);
}

/**
 * hb_graphite2_face_get_gr_face: (skip)
 * @face: @hb_face_t to query
 *
 * Fetches the Graphite2 gr_face corresponding to the specified
 * #hb_face_t face object.
 *
 * Return value: the gr_face found
 *
 * Since: 0.9.10
 */
gr_face *
hb_graphite2_face_get_gr_face (hb_face_t *face)
{
  const hb_graphite2_face_data_t *data = face->data.graphite2;
  return data ? data->grface : nullptr;
}


/*
 * shaper font data
 */

struct hb_graphite2_font_data_t {};

hb_graphite2_font_data_t *
_hb_graphite2_shaper_font_data_create (hb_font_t *font HB_UNUSED)
{
  return (hb_graphite2_font_data_t *) HB_SHAPER_DATA_SUCCEEDED;
}

void
_hb_graphite2_shaper_font_data_destroy (hb_graphite2_font_data_t *data HB_UNUSED)
{
}

#ifndef HB_DISABLE_DEPRECATED
/**
 * hb_graphite2_font_get_gr_font: (skip)
 * @font: An #hb_font_t
 *
 * Always returns `NULL`. Use hb_graphite2_face_get_gr_face() instead.
 *
 * Return value: (nullable): Graphite2 font associated with @font.
 *
 * Since: 0.9.10
 * Deprecated: 1.4.2
 */
gr_font *
hb_graphite2_font_get_gr_font (hb_font_t *font HB_UNUSED)
{
  return nullptr;
}
#endif


/*
 * shaper
 */

struct hb_graphite2_cluster_t {
  unsigned int base_char;
  unsigned int num_chars;
  unsigned int base_glyph;
  unsigned int num_glyphs;
  unsigned int cluster;
  int advance;
};

hb_bool_t
_hb_graphite2_shape (hb_shape_plan_t    *shape_plan HB_UNUSED,
		     hb_font_t          *font,
		     hb_buffer_t        *buffer,
		     const hb_feature_t *features,
		     unsigned int        num_features)
{
  hb_face_t *face = font->face;
  gr_face *grface = face->data.graphite2->grface;

  const char *lang = hb_language_to_string (hb_buffer_get_language (buffer));
  const char *lang_end = lang ? strchr (lang, '-') : nullptr;
  int lang_len = lang_end ? lang_end - lang : -1;
  gr_feature_val *feats = gr_face_featureval_for_lang (grface, lang ? hb_tag_from_string (lang, lang_len) : 0);

  for (unsigned int i = 0; i < num_features; i++)
  {
    const gr_feature_ref *fref = gr_face_find_fref (grface, features[i].tag);
    if (fref)
      gr_fref_set_feature_value (fref, features[i].value, feats);
  }

  hb_direction_t direction = buffer->props.direction;
  hb_direction_t horiz_dir = hb_script_get_horizontal_direction (buffer->props.script);
  /* TODO vertical:
   * The only BTT vertical script is Ogham, but it's not clear to me whether OpenType
   * Ogham fonts are supposed to be implemented BTT or not.  Need to research that
   * first. */
  if ((HB_DIRECTION_IS_HORIZONTAL (direction) &&
       direction != horiz_dir && horiz_dir != HB_DIRECTION_INVALID) ||
      (HB_DIRECTION_IS_VERTICAL   (direction) &&
       direction != HB_DIRECTION_TTB))
  {
    hb_buffer_reverse_clusters (buffer);
    direction = HB_DIRECTION_REVERSE (direction);
  }

  gr_segment *seg = nullptr;
  const gr_slot *is;
  unsigned int ci = 0, ic = 0;
  int curradvx = 0, curradvy = 0;

  unsigned int scratch_size;
  hb_buffer_t::scratch_buffer_t *scratch = buffer->get_scratch_buffer (&scratch_size);

  uint32_t *chars = (uint32_t *) scratch;

  for (unsigned int i = 0; i < buffer->len; ++i)
    chars[i] = buffer->info[i].codepoint;

  seg = gr_make_seg (nullptr, grface,
		     HB_TAG_NONE, // https://github.com/harfbuzz/harfbuzz/issues/3439#issuecomment-1442650148
		     feats,
		     gr_utf32, chars, buffer->len,
		     2 | (direction == HB_DIRECTION_RTL ? 1 : 0));

  if (unlikely (!seg)) {
    if (feats) gr_featureval_destroy (feats);
    return false;
  }

  unsigned int glyph_count = gr_seg_n_slots (seg);
  if (unlikely (!glyph_count)) {
    if (feats) gr_featureval_destroy (feats);
    gr_seg_destroy (seg);
    buffer->len = 0;
    return true;
  }

  (void) buffer->ensure (glyph_count);
  scratch = buffer->get_scratch_buffer (&scratch_size);
  while ((DIV_CEIL (sizeof (hb_graphite2_cluster_t) * buffer->len, sizeof (*scratch)) +
	  DIV_CEIL (sizeof (hb_codepoint_t) * glyph_count, sizeof (*scratch))) > scratch_size)
  {
    if (unlikely (!buffer->ensure (buffer->allocated * 2)))
    {
      if (feats) gr_featureval_destroy (feats);
      gr_seg_destroy (seg);
      return false;
    }
    scratch = buffer->get_scratch_buffer (&scratch_size);
  }

#define ALLOCATE_ARRAY(Type, name, len) \
  Type *name = (Type *) scratch; \
  do { \
    unsigned int _consumed = DIV_CEIL ((len) * sizeof (Type), sizeof (*scratch)); \
    assert (_consumed <= scratch_size); \
    scratch += _consumed; \
    scratch_size -= _consumed; \
  } while (0)

  ALLOCATE_ARRAY (hb_graphite2_cluster_t, clusters, buffer->len);
  ALLOCATE_ARRAY (hb_codepoint_t, gids, glyph_count);

#undef ALLOCATE_ARRAY

  hb_memset (clusters, 0, sizeof (clusters[0]) * buffer->len);

  hb_codepoint_t *pg = gids;
  clusters[0].cluster = buffer->info[0].cluster;
  unsigned int upem = hb_face_get_upem (face);
  float xscale = (float) font->x_scale / upem;
  float yscale = (float) font->y_scale / upem;
  yscale *= yscale / xscale;
  unsigned int curradv = 0;
  if (HB_DIRECTION_IS_BACKWARD (direction))
  {
    curradv = gr_slot_origin_X(gr_seg_first_slot(seg)) * xscale;
    clusters[0].advance = gr_seg_advance_X(seg) * xscale - curradv;
  }
  else
    clusters[0].advance = 0;
  for (is = gr_seg_first_slot (seg), ic = 0; is; is = gr_slot_next_in_segment (is), ic++)
  {
    unsigned int before = gr_slot_before (is);
    unsigned int after = gr_slot_after (is);
    *pg = gr_slot_gid (is);
    pg++;
    while (clusters[ci].base_char > before && ci)
    {
      clusters[ci-1].num_chars += clusters[ci].num_chars;
      clusters[ci-1].num_glyphs += clusters[ci].num_glyphs;
      clusters[ci-1].advance += clusters[ci].advance;
      ci--;
    }

    if (gr_slot_can_insert_before (is) && clusters[ci].num_chars && before >= clusters[ci].base_char + clusters[ci].num_chars)
    {
      hb_graphite2_cluster_t *c = clusters + ci + 1;
      c->base_char = clusters[ci].base_char + clusters[ci].num_chars;
      c->cluster = buffer->info[c->base_char].cluster;
      c->num_chars = before - c->base_char;
      c->base_glyph = ic;
      c->num_glyphs = 0;
      if (HB_DIRECTION_IS_BACKWARD (direction))
      {
	c->advance = curradv - gr_slot_origin_X(is) * xscale;
	curradv -= c->advance;
      }
      else
      {
	auto origin_X = gr_slot_origin_X (is) * xscale;
	c->advance = 0;
	clusters[ci].advance += origin_X - curradv;
	curradv = origin_X;
      }
      ci++;
    }
    clusters[ci].num_glyphs++;

    if (clusters[ci].base_char + clusters[ci].num_chars < after + 1)
	clusters[ci].num_chars = after + 1 - clusters[ci].base_char;
  }

  if (HB_DIRECTION_IS_BACKWARD (direction))
    clusters[ci].advance += curradv;
  else
    clusters[ci].advance += gr_seg_advance_X(seg) * xscale - curradv;
  ci++;

  for (unsigned int i = 0; i < ci; ++i)
  {
    for (unsigned int j = 0; j < clusters[i].num_glyphs; ++j)
    {
      hb_glyph_info_t *info = &buffer->info[clusters[i].base_glyph + j];
      info->codepoint = gids[clusters[i].base_glyph + j];
      info->cluster = clusters[i].cluster;
      info->var1.i32 = clusters[i].advance;     // all glyphs in the cluster get the same advance
    }
  }
  buffer->len = glyph_count;

  /* Positioning. */
  unsigned int currclus = UINT_MAX;
  const hb_glyph_info_t *info = buffer->info;
  hb_glyph_position_t *pPos = hb_buffer_get_glyph_positions (buffer, nullptr);
  if (!HB_DIRECTION_IS_BACKWARD (direction))
  {
    curradvx = 0;
    for (is = gr_seg_first_slot (seg); is; pPos++, ++info, is = gr_slot_next_in_segment (is))
    {
      pPos->x_offset = gr_slot_origin_X (is) * xscale - curradvx;
      pPos->y_offset = gr_slot_origin_Y (is) * yscale - curradvy;
      if (info->cluster != currclus) {
	pPos->x_advance = info->var1.i32;
	curradvx += pPos->x_advance;
	currclus = info->cluster;
      } else
	pPos->x_advance = 0.;

      pPos->y_advance = gr_slot_advance_Y (is, grface, nullptr) * yscale;
      curradvy += pPos->y_advance;
    }
  }
  else
  {
    curradvx = gr_seg_advance_X(seg) * xscale;
    for (is = gr_seg_first_slot (seg); is; pPos++, info++, is = gr_slot_next_in_segment (is))
    {
      if (info->cluster != currclus)
      {
	pPos->x_advance = info->var1.i32;
	curradvx -= pPos->x_advance;
	currclus = info->cluster;
      } else
	pPos->x_advance = 0.;

      pPos->y_advance = gr_slot_advance_Y (is, grface, nullptr) * yscale;
      curradvy -= pPos->y_advance;
      pPos->x_offset = gr_slot_origin_X (is) * xscale - info->var1.i32 - curradvx + pPos->x_advance;
      pPos->y_offset = gr_slot_origin_Y (is) * yscale - curradvy;
    }
    hb_buffer_reverse_clusters (buffer);
  }

  if (feats) gr_featureval_destroy (feats);
  gr_seg_destroy (seg);

  buffer->clear_glyph_flags ();
  buffer->unsafe_to_break ();

  return true;
}


#endif

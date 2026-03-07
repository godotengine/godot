/*
 * Copyright © 2026  Behdad Esfahbod
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
 * Author(s): Behdad Esfahbod
 */

#include "hb.hh"

#include "hb-face.hh"
#include "hb-vector-svg-subset.hh"
#include "OT/Color/svg/svg.hh"
#include "hb-vector-svg-utils.hh"
#include "hb-map.hh"
#include "hb-ot-color.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

static bool
hb_svg_append_with_prefix (hb_vector_t<char> *out,
                           const char *s,
                           unsigned n,
                           const char *prefix,
                           unsigned prefix_len)
{
  unsigned i = 0;
  while (i < n)
  {
    if (i + 4 <= n && !memcmp (s + i, "id=\"", 4))
    {
      if (!hb_svg_append_len (out, s + i, 4)) return false;
      i += 4;
      if (!hb_svg_append_len (out, prefix, prefix_len)) return false;
      while (i < n && s[i] != '"')
      {
        if (!hb_svg_append_c (out, s[i])) return false;
        i++;
      }
      continue;
    }
    if (i + 4 <= n && !memcmp (s + i, "id='", 4))
    {
      if (!hb_svg_append_len (out, s + i, 4)) return false;
      i += 4;
      if (!hb_svg_append_len (out, prefix, prefix_len)) return false;
      while (i < n && s[i] != '\'')
      {
        if (!hb_svg_append_c (out, s[i])) return false;
        i++;
      }
      continue;
    }
    if (i + 7 <= n && !memcmp (s + i, "href=\"#", 7))
    {
      if (!hb_svg_append_len (out, s + i, 7)) return false;
      i += 7;
      if (!hb_svg_append_len (out, prefix, prefix_len)) return false;
      while (i < n && s[i] != '"')
      {
        if (!hb_svg_append_c (out, s[i])) return false;
        i++;
      }
      continue;
    }
    if (i + 7 <= n && !memcmp (s + i, "href='#", 7))
    {
      if (!hb_svg_append_len (out, s + i, 7)) return false;
      i += 7;
      if (!hb_svg_append_len (out, prefix, prefix_len)) return false;
      while (i < n && s[i] != '\'')
      {
        if (!hb_svg_append_c (out, s[i])) return false;
        i++;
      }
      continue;
    }
    if (i + 13 <= n && !memcmp (s + i, "xlink:href=\"#", 13))
    {
      if (!hb_svg_append_len (out, s + i, 13)) return false;
      i += 13;
      if (!hb_svg_append_len (out, prefix, prefix_len)) return false;
      while (i < n && s[i] != '"')
      {
        if (!hb_svg_append_c (out, s[i])) return false;
        i++;
      }
      continue;
    }
    if (i + 13 <= n && !memcmp (s + i, "xlink:href='#", 13))
    {
      if (!hb_svg_append_len (out, s + i, 13)) return false;
      i += 13;
      if (!hb_svg_append_len (out, prefix, prefix_len)) return false;
      while (i < n && s[i] != '\'')
      {
        if (!hb_svg_append_c (out, s[i])) return false;
        i++;
      }
      continue;
    }
    if (i + 5 <= n && !memcmp (s + i, "url(#", 5))
    {
      if (!hb_svg_append_len (out, s + i, 5)) return false;
      i += 5;
      if (!hb_svg_append_len (out, prefix, prefix_len)) return false;
      while (i < n && s[i] != ')')
      {
        if (!hb_svg_append_c (out, s[i])) return false;
        i++;
      }
      continue;
    }
    if (i + 6 <= n && !memcmp (s + i, "url(\"#", 6))
    {
      if (!hb_svg_append_len (out, s + i, 6)) return false;
      i += 6;
      if (!hb_svg_append_len (out, prefix, prefix_len)) return false;
      while (i < n && s[i] != '"')
      {
        if (!hb_svg_append_c (out, s[i])) return false;
        i++;
      }
      continue;
    }
    if (i + 6 <= n && !memcmp (s + i, "url('#", 6))
    {
      if (!hb_svg_append_len (out, s + i, 6)) return false;
      i += 6;
      if (!hb_svg_append_len (out, prefix, prefix_len)) return false;
      while (i < n && s[i] != '\'')
      {
        if (!hb_svg_append_c (out, s[i])) return false;
        i++;
      }
      continue;
    }
    if (!hb_svg_append_c (out, s[i]))
      return false;
    i++;
  }
  return true;
}

static bool
hb_svg_add_unique_id (hb_vector_t<OT::SVG::svg_id_span_t> *v,
                      hb_hashmap_t<OT::SVG::svg_id_span_t, hb_bool_t> *seen_ids,
                      const char *p,
                      unsigned n)
{
  if (!n) return false;
  OT::SVG::svg_id_span_t key = {p, n};
  if (seen_ids->has (key))
    return false;
  if (unlikely (!seen_ids->set (key, true)))
    return false;
  auto *slot = v->push ();
  if (!slot) return false;
  *slot = key;
  return true;
}

static bool
hb_svg_collect_refs (const char *s,
                     unsigned n,
                     hb_vector_t<OT::SVG::svg_id_span_t> *ids,
                     hb_hashmap_t<OT::SVG::svg_id_span_t, hb_bool_t> *seen_ids)
{
  unsigned i = 0;
  while (i < n)
  {
    if (i + 7 <= n && !memcmp (s + i, "href=\"#", 7))
    {
      i += 7;
      unsigned b = i;
      while (i < n && s[i] != '"' && s[i] != '\'' && s[i] != ' ' && s[i] != '>') i++;
      if (i > b && unlikely (!hb_svg_add_unique_id (ids, seen_ids, s + b, i - b))) return false;
      continue;
    }
    if (i + 7 <= n && !memcmp (s + i, "href='#", 7))
    {
      i += 7;
      unsigned b = i;
      while (i < n && s[i] != '\'' && s[i] != '"' && s[i] != ' ' && s[i] != '>') i++;
      if (i > b && unlikely (!hb_svg_add_unique_id (ids, seen_ids, s + b, i - b))) return false;
      continue;
    }
    if (i + 13 <= n && !memcmp (s + i, "xlink:href=\"#", 13))
    {
      i += 13;
      unsigned b = i;
      while (i < n && s[i] != '"' && s[i] != '\'' && s[i] != ' ' && s[i] != '>') i++;
      if (i > b && unlikely (!hb_svg_add_unique_id (ids, seen_ids, s + b, i - b))) return false;
      continue;
    }
    if (i + 13 <= n && !memcmp (s + i, "xlink:href='#", 13))
    {
      i += 13;
      unsigned b = i;
      while (i < n && s[i] != '\'' && s[i] != '"' && s[i] != ' ' && s[i] != '>') i++;
      if (i > b && unlikely (!hb_svg_add_unique_id (ids, seen_ids, s + b, i - b))) return false;
      continue;
    }
    if (i + 5 <= n && !memcmp (s + i, "url(#", 5))
    {
      i += 5;
      unsigned b = i;
      while (i < n && s[i] != ')') i++;
      if (i > b && unlikely (!hb_svg_add_unique_id (ids, seen_ids, s + b, i - b))) return false;
      continue;
    }
    if (i + 6 <= n && !memcmp (s + i, "url(\"#", 6))
    {
      i += 6;
      unsigned b = i;
      while (i < n && s[i] != '"') i++;
      if (i > b && unlikely (!hb_svg_add_unique_id (ids, seen_ids, s + b, i - b))) return false;
      continue;
    }
    if (i + 6 <= n && !memcmp (s + i, "url('#", 6))
    {
      i += 6;
      unsigned b = i;
      while (i < n && s[i] != '\'') i++;
      if (i > b && unlikely (!hb_svg_add_unique_id (ids, seen_ids, s + b, i - b))) return false;
      continue;
    }
    i++;
  }
  return true;
}

bool
hb_svg_subset_glyph_image (hb_face_t *face,
                           hb_blob_t *image,
                           hb_codepoint_t glyph,
                           unsigned *image_counter,
                           hb_vector_t<char> *defs_dst,
                           hb_vector_t<char> *body_dst)
{
  if (glyph == HB_CODEPOINT_INVALID || !image_counter || !defs_dst || !body_dst)
    return false;

  unsigned len;
  const char *svg = hb_blob_get_data (image, &len);
  if (!svg || !len)
    return false;

  unsigned doc_index = 0;
  if (!hb_ot_color_glyph_get_svg_document_index (face, glyph, &doc_index))
    return false;

  hb_codepoint_t start_glyph = HB_CODEPOINT_INVALID;
  hb_codepoint_t end_glyph = HB_CODEPOINT_INVALID;
  if (!hb_ot_color_get_svg_document_glyph_range (face, doc_index, &start_glyph, &end_glyph))
    return false;

  auto *doc_cache = face->table.SVG->get_or_create_doc_cache (image, svg, len,
                                                              doc_index, start_glyph, end_glyph);
  if (!doc_cache)
    return false;
  svg = face->table.SVG->doc_cache_get_svg (doc_cache, &len);

  unsigned glyph_start = 0, glyph_end = 0;
  if (!face->table.SVG->doc_cache_get_glyph_span (doc_cache, glyph, &glyph_start, &glyph_end))
    return false;

  auto *defs_entries = face->table.SVG->doc_cache_get_defs_entries (doc_cache);

  hb_vector_t<OT::SVG::svg_id_span_t> needed_ids;
  needed_ids.alloc (16);
  hb_hashmap_t<OT::SVG::svg_id_span_t, hb_bool_t> needed_ids_set;
  if (!hb_svg_collect_refs (svg + glyph_start, glyph_end - glyph_start,
                            &needed_ids, &needed_ids_set))
    return false;

  hb_vector_t<unsigned> chosen_defs;
  chosen_defs.alloc (16);
  hb_vector_t<uint8_t> chosen_def_marks;
  if (unlikely (!chosen_def_marks.resize ((int) defs_entries->length)))
    return false;
  hb_memset (chosen_def_marks.arrayZ, 0, defs_entries->length);
  for (unsigned qi = 0; qi < needed_ids.length; qi++)
  {
    const auto &need = needed_ids.arrayZ[qi];
    for (unsigned i = 0; i < defs_entries->length; i++)
    {
      const auto &e = defs_entries->arrayZ[i];
      if (e.id.len == need.len && !memcmp (e.id.p, need.p, need.len))
      {
        if (!chosen_def_marks.arrayZ[i])
        {
          chosen_def_marks.arrayZ[i] = 1;
          if (unlikely (!chosen_defs.push (i)))
            return false;
          if (!hb_svg_collect_refs (svg + e.start, e.end - e.start,
                                    &needed_ids, &needed_ids_set))
            return false;
        }
        break;
      }
    }
  }

  char prefix[32];
  int prefix_len = snprintf (prefix, sizeof (prefix), "hbimg%u_", (*image_counter)++);
  if (prefix_len <= 0 || (unsigned) prefix_len >= sizeof (prefix))
    return false;

  body_dst->alloc (body_dst->length + (glyph_end - glyph_start) + (unsigned) prefix_len + 32);

  for (unsigned i = 0; i < chosen_defs.length; i++)
  {
    const auto &e = defs_entries->arrayZ[chosen_defs.arrayZ[i]];
    if (!hb_svg_append_with_prefix (defs_dst, svg + e.start, e.end - e.start, prefix, (unsigned) prefix_len))
      return false;
    if (!hb_svg_append_c (defs_dst, '\n')) return false;
  }

  return hb_svg_append_with_prefix (body_dst,
                                    svg + glyph_start,
                                    glyph_end - glyph_start,
                                    prefix,
                                    (unsigned) prefix_len);
}

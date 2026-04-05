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
#ifndef HB_NO_SVG
#include "OT/Color/svg/svg.hh"
#include "hb-raster-svg-parse.hh"
#endif
#include "hb-vector-svg-utils.hh"
#include "hb-map.hh"
#include "hb-ot-color.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

#ifndef HB_NO_SVG
#include "hb-vector-svg.hh"

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
  if (!n) return true;
  OT::SVG::svg_id_span_t key = {p, n};
  if (seen_ids->has (key))
    return true;
  if (unlikely (!seen_ids->set (key, true)))
    return false;
  auto *slot = v->push ();
  if (unlikely (v->in_error ()))
    return false;
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

static bool
hb_svg_find_root_open_tag (const char *svg,
                           unsigned len,
                           unsigned *root_open_len,
                           bool *missing_viewport)
{
  hb_svg_xml_parser_t parser (svg, len);
  hb_svg_token_type_t tok = parser.next ();
  if (!((tok == SVG_TOKEN_OPEN_TAG || tok == SVG_TOKEN_SELF_CLOSE_TAG) &&
        parser.tag_name.eq ("svg")))
    return false;

  if (missing_viewport)
  {
    *missing_viewport =
      parser.find_attr ("width").is_null () ||
      parser.find_attr ("height").is_null ();
  }

  if (root_open_len)
    *root_open_len = (unsigned) (parser.p - parser.tag_start);
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
  bool ret = false;
  hb_blob_t *normalized_image = nullptr;
  unsigned len = 0;
  const char *svg = nullptr;
  unsigned doc_index = 0;
  hb_codepoint_t start_glyph = HB_CODEPOINT_INVALID;
  hb_codepoint_t end_glyph = HB_CODEPOINT_INVALID;
  const OT::SVG::svg_doc_cache_t *doc_cache = nullptr;
  unsigned glyph_start = 0, glyph_end = 0;
  const hb_vector_t<OT::SVG::svg_defs_entry_t> *defs_entries = nullptr;
  hb_vector_t<OT::SVG::svg_id_span_t> needed_ids;
  hb_hashmap_t<OT::SVG::svg_id_span_t, hb_bool_t> needed_ids_set;
  hb_vector_t<unsigned> chosen_defs;
  hb_vector_t<uint8_t> chosen_def_marks;
  unsigned root_open_len = 0;
  bool glyph_is_root_svg = false;
  bool root_missing_viewport = false;
  char prefix[32];
  int prefix_len = 0;

  if (glyph == HB_CODEPOINT_INVALID || !image_counter || !defs_dst || !body_dst)
    goto done;

  normalized_image = OT::hb_ot_svg_reference_normalized_blob (image, &svg, &len);
  if (!normalized_image || !svg || !len)
    goto done;

  if (!hb_ot_color_glyph_get_svg_document_index (face, glyph, &doc_index))
    goto done;

  if (!hb_ot_color_get_svg_document_glyph_range (face, doc_index, &start_glyph, &end_glyph))
    goto done;

  doc_cache = face->table.SVG->get_or_create_doc_cache (normalized_image, svg, len,
                                                        doc_index, start_glyph, end_glyph);
  if (!doc_cache)
    goto done;
  svg = face->table.SVG->doc_cache_get_svg (doc_cache, &len);

  if (!face->table.SVG->doc_cache_get_glyph_span (doc_cache, glyph, &glyph_start, &glyph_end))
    goto done;

  defs_entries = face->table.SVG->doc_cache_get_defs_entries (doc_cache);
  if (!hb_svg_find_root_open_tag (svg, len, &root_open_len, &root_missing_viewport))
    goto done;
  glyph_is_root_svg = glyph_start == 0;

  needed_ids.alloc (16);
  if (!hb_svg_collect_refs (svg + glyph_start, glyph_end - glyph_start,
                            &needed_ids, &needed_ids_set))
    goto done;

  chosen_defs.alloc (16);
  if (unlikely (!chosen_def_marks.resize ((int) defs_entries->length)))
    goto done;
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
          if (unlikely (!chosen_defs.push_or_fail (i)))
            goto done;
          if (!hb_svg_collect_refs (svg + e.start, e.end - e.start,
                                    &needed_ids, &needed_ids_set))
            goto done;
        }
        break;
      }
    }
  }

  prefix_len = snprintf (prefix, sizeof (prefix), "hbimg%u_", (*image_counter)++);
  if (prefix_len <= 0 || (unsigned) prefix_len >= sizeof (prefix))
    goto done;

  body_dst->alloc (body_dst->length + (glyph_end - glyph_start) + 64);

  for (unsigned i = 0; i < chosen_defs.length; i++)
  {
    const auto &e = defs_entries->arrayZ[chosen_defs.arrayZ[i]];
    if (!hb_svg_append_with_prefix (defs_dst, svg + e.start, e.end - e.start, prefix, (unsigned) prefix_len))
      goto done;
    if (!hb_svg_append_c (defs_dst, '\n'))
      goto done;
  }

  if (glyph_is_root_svg && root_missing_viewport && root_open_len > 1)
  {
    unsigned upem = face->get_upem ();
    if (!hb_svg_append_with_prefix (body_dst,
                                    svg + glyph_start,
                                    root_open_len - 1,
                                    prefix,
                                    (unsigned) prefix_len))
      goto done;
    if (!hb_svg_append_str (body_dst, " width=\""))
      goto done;
    if (!hb_svg_append_unsigned (body_dst, upem))
      goto done;
    if (!hb_svg_append_str (body_dst, "\" height=\""))
      goto done;
    if (!hb_svg_append_unsigned (body_dst, upem))
      goto done;
    if (!hb_svg_append_str (body_dst, "\" overflow=\"visible"))
      goto done;
    if (!hb_svg_append_c (body_dst, '"'))
      goto done;
    if (!hb_svg_append_c (body_dst, '>'))
      goto done;
    ret = hb_svg_append_with_prefix (body_dst,
                                     svg + glyph_start + root_open_len,
                                     glyph_end - glyph_start - root_open_len,
                                     prefix,
                                     (unsigned) prefix_len);
  }
  else if (glyph_is_root_svg && root_open_len > 1)
  {
    if (!hb_svg_append_with_prefix (body_dst,
                                    svg + glyph_start,
                                    root_open_len - 1,
                                    prefix,
                                    (unsigned) prefix_len))
      goto done;
    if (!hb_svg_append_str (body_dst, " overflow=\"visible\""))
      goto done;
    if (!hb_svg_append_c (body_dst, '>'))
      goto done;
    ret = hb_svg_append_with_prefix (body_dst,
                                     svg + glyph_start + root_open_len,
                                     glyph_end - glyph_start - root_open_len,
                                     prefix,
                                     (unsigned) prefix_len);
  }
  else
    ret = hb_svg_append_with_prefix (body_dst,
                                     svg + glyph_start,
                                     glyph_end - glyph_start,
                                     prefix,
                                     (unsigned) prefix_len);

done:
  hb_blob_destroy (normalized_image);
  return ret;
}
#else
bool
hb_svg_subset_glyph_image (hb_face_t *face HB_UNUSED,
                           hb_blob_t *image HB_UNUSED,
                           hb_codepoint_t glyph HB_UNUSED,
                           unsigned *image_counter HB_UNUSED,
                           hb_vector_t<char> *defs_dst HB_UNUSED,
                           hb_vector_t<char> *body_dst HB_UNUSED)
{
  return false;
}
#endif

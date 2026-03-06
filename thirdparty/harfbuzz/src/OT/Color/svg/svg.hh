/*
 * Copyright © 2018  Ebrahim Byagowi
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
 */

#ifndef OT_COLOR_SVG_SVG_HH
#define OT_COLOR_SVG_SVG_HH

#include "../../../hb-open-type.hh"
#include "../../../hb-blob.hh"
#include "../../../hb-map.hh"
#include "../../../hb-paint.hh"
#include <ctype.h>
#include <string.h>

/*
 * SVG -- SVG (Scalable Vector Graphics)
 * https://docs.microsoft.com/en-us/typography/opentype/spec/svg
 */

#define HB_OT_TAG_SVG HB_TAG('S','V','G',' ')


namespace OT {


struct SVGDocumentIndexEntry
{
  int cmp (hb_codepoint_t g) const
  { return g < startGlyphID ? -1 : g > endGlyphID ? 1 : 0; }

  hb_codepoint_t get_start_glyph () const
  { return startGlyphID; }

  hb_codepoint_t get_end_glyph () const
  { return endGlyphID; }

  hb_blob_t *reference_blob (hb_blob_t *svg_blob, unsigned int index_offset) const
  {
    return hb_blob_create_sub_blob (svg_blob,
				    index_offset + (unsigned int) svgDoc,
				    svgDocLength);
  }

  bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  hb_barrier () &&
		  svgDoc.sanitize (c, base, svgDocLength));
  }

  protected:
  HBUINT16	startGlyphID;	/* The first glyph ID in the range described by
				 * this index entry. */
  HBUINT16	endGlyphID;	/* The last glyph ID in the range described by
				 * this index entry. Must be >= startGlyphID. */
  NNOffset32To<UnsizedArrayOf<HBUINT8>>
		svgDoc;		/* Offset from the beginning of the SVG Document Index
				 * to an SVG document. Must be non-zero. */
  HBUINT32	svgDocLength;	/* Length of the SVG document.
				 * Must be non-zero. */
  public:
  DEFINE_SIZE_STATIC (12);
};

struct SVG
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_SVG;

  struct svg_id_span_t
  {
    const char *p;
    unsigned len;

    bool operator == (const svg_id_span_t &o) const
    {
      return len == o.len && !memcmp (p, o.p, len);
    }

    uint32_t hash () const
    {
      uint32_t h = hb_hash (len);
      for (unsigned i = 0; i < len; i++)
        h = h * 33u + (unsigned char) p[i];
      return h;
    }
  };

  struct svg_defs_entry_t
  {
    svg_id_span_t id;
    unsigned start;
    unsigned end;
  };

  struct svg_doc_cache_t
  {
    hb_blob_t *blob = nullptr;
    const char *svg = nullptr;
    unsigned len = 0;
    hb_vector_t<svg_defs_entry_t> defs_entries;
    hb_codepoint_t start_glyph = HB_CODEPOINT_INVALID;
    hb_codepoint_t end_glyph = HB_CODEPOINT_INVALID;
    hb_vector_t<hb_pair_t<uint32_t, uint32_t>> glyph_spans;
    hb_hashmap_t<svg_id_span_t, hb_pair_t<uint32_t, uint32_t>> id_spans;
  };

  bool has_data () const { return svgDocEntries; }

  struct accelerator_t
  {
    accelerator_t (hb_face_t *face);
    ~accelerator_t ();

    hb_blob_t *reference_blob_for_glyph (hb_codepoint_t glyph_id) const
    {
      return table->get_glyph_entry (glyph_id).reference_blob (table.get_blob (),
							       table->svgDocEntries);
    }

    unsigned get_document_count () const
    { return table->get_document_count (); }

    bool get_glyph_document_index (hb_codepoint_t glyph_id, unsigned *index) const
    { return table->get_glyph_document_index (glyph_id, index); }

    bool get_document_glyph_range (unsigned index,
                                   hb_codepoint_t *start_glyph,
                                   hb_codepoint_t *end_glyph) const
    { return table->get_document_glyph_range (index, start_glyph, end_glyph); }

    bool has_data () const { return table->has_data (); }

    const svg_doc_cache_t *
    get_or_create_doc_cache (hb_blob_t *image,
                             const char *svg,
                             unsigned len,
                             unsigned doc_index,
                             hb_codepoint_t start_glyph,
                             hb_codepoint_t end_glyph) const;

    const char *
    doc_cache_get_svg (const svg_doc_cache_t *doc,
                       unsigned *len) const;

    const hb_vector_t<svg_defs_entry_t> *
    doc_cache_get_defs_entries (const svg_doc_cache_t *doc) const;

    bool
    doc_cache_get_glyph_span (const svg_doc_cache_t *doc,
                              hb_codepoint_t glyph,
                              unsigned *start,
                              unsigned *end) const;

    bool
    doc_cache_find_id_span (const svg_doc_cache_t *doc,
                            svg_id_span_t id,
                            unsigned *start,
                            unsigned *end) const;

    bool
    doc_cache_find_id_cstr (const svg_doc_cache_t *doc,
                            const char *id,
                            unsigned *start,
                            unsigned *end) const;

    bool paint_glyph (hb_font_t *font HB_UNUSED, hb_codepoint_t glyph, hb_paint_funcs_t *funcs, void *data) const
    {
      if (!has_data ())
        return false;

      hb_blob_t *blob = reference_blob_for_glyph (glyph);

      if (blob == hb_blob_get_empty ())
        return false;

      bool ret = funcs->image (data,
			       blob,
			       0, 0,
			       HB_PAINT_IMAGE_FORMAT_SVG,
			       0.f,
			       nullptr);

      hb_blob_destroy (blob);

      return ret;
    }

    private:
    svg_doc_cache_t *
    make_doc_cache (hb_blob_t *image,
                    const char *svg,
                    unsigned len,
                    hb_codepoint_t start_glyph,
                    hb_codepoint_t end_glyph) const;

    static void destroy_doc_cache (svg_doc_cache_t *doc);

    hb_blob_ptr_t<SVG> table;
    mutable hb_vector_t<hb_atomic_t<svg_doc_cache_t *>> doc_caches;
    public:
    DEFINE_SIZE_STATIC (sizeof (hb_blob_ptr_t<SVG>) +
                        sizeof (hb_vector_t<hb_atomic_t<svg_doc_cache_t *>>));
  };

  const SVGDocumentIndexEntry &get_glyph_entry (hb_codepoint_t glyph_id) const
  { return (this+svgDocEntries).bsearch (glyph_id); }

  unsigned get_document_count () const
  {
    if (!has_data ())
      return 0;
    return (this + svgDocEntries).len;
  }

  bool get_glyph_document_index (hb_codepoint_t glyph_id, unsigned *index) const
  {
    if (!has_data ())
      return false;
    return (this + svgDocEntries).bfind (glyph_id, index);
  }

  bool get_document_glyph_range (unsigned index,
                                 hb_codepoint_t *start_glyph,
                                 hb_codepoint_t *end_glyph) const
  {
    if (!has_data ())
      return false;

    const auto &entries = this + svgDocEntries;
    if (index >= entries.len)
      return false;

    const auto &entry = entries.arrayZ[index];
    if (start_glyph) *start_glyph = entry.get_start_glyph ();
    if (end_glyph) *end_glyph = entry.get_end_glyph ();
    return true;
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  (this+svgDocEntries).sanitize_shallow (c)));
  }

  protected:
  HBUINT16	version;	/* Table version (starting at 0). */
  Offset32To<SortedArray16Of<SVGDocumentIndexEntry>>
		svgDocEntries;	/* Offset (relative to the start of the SVG table) to the
				 * SVG Documents Index. Must be non-zero. */
				/* Array of SVG Document Index Entries. */
  HBUINT32	reserved;	/* Set to 0. */
  public:
  DEFINE_SIZE_STATIC (10);
};

namespace _hb_svg_cache_impl {

struct glyph_entry_t
{
  hb_codepoint_t glyph;
  uint32_t start;
  uint32_t end;
};

struct id_entry_t
{
  SVG::svg_id_span_t id;
  uint32_t start;
  uint32_t end;
};

struct open_elem_t
{
  unsigned start;
  SVG::svg_id_span_t id;
  bool in_defs_content;
  bool is_defs;
};

static const unsigned MAX_DEPTH = 128;

static inline int
find_substr (const char *s,
             unsigned n,
             unsigned from,
             const char *needle,
             unsigned needle_len)
{
  if (!needle_len || from >= n || needle_len > n)
    return -1;
  for (unsigned i = from; i + needle_len <= n; i++)
    if (s[i] == needle[0] && !memcmp (s + i, needle, needle_len))
      return (int) i;
  return -1;
}

static inline bool
parse_id_in_start_tag (const char *svg,
                       unsigned tag_start,
                       unsigned tag_end,
                       SVG::svg_id_span_t *id)
{
  unsigned p = tag_start;
  while (p + 4 <= tag_end)
  {
    if (!memcmp (svg + p, "id=\"", 4))
    {
      unsigned b = p + 4;
      unsigned e = b;
      while (e < tag_end && svg[e] != '"') e++;
      if (e <= tag_end && e > b)
      {
        *id = {svg + b, e - b};
        return true;
      }
    }
    if (!memcmp (svg + p, "id='", 4))
    {
      unsigned b = p + 4;
      unsigned e = b;
      while (e < tag_end && svg[e] != '\'') e++;
      if (e <= tag_end && e > b)
      {
        *id = {svg + b, e - b};
        return true;
      }
    }
    p++;
  }
  return false;
}

static inline bool
parse_glyph_id_span (const SVG::svg_id_span_t &id,
                     hb_codepoint_t *glyph)
{
  if (id.len <= 5 || memcmp (id.p, "glyph", 5))
    return false;

  hb_codepoint_t gid = 0;
  for (unsigned i = 5; i < id.len; i++)
  {
    unsigned char c = (unsigned char) id.p[i];
    if (c < '0' || c > '9')
      return false;
    hb_codepoint_t digit = (hb_codepoint_t) (c - '0');
    if (unlikely (gid > HB_CODEPOINT_INVALID / 10 ||
                  (gid == HB_CODEPOINT_INVALID / 10 &&
                   digit > HB_CODEPOINT_INVALID % 10)))
      return false;
    gid = (hb_codepoint_t) (gid * 10 + digit);
  }

  *glyph = gid;
  return true;
}

static inline bool
parse_cache_entries_linear (const char *svg,
                            unsigned len,
                            hb_vector_t<SVG::svg_defs_entry_t> *defs_entries,
                            hb_vector_t<glyph_entry_t> *glyph_spans,
                            hb_vector_t<id_entry_t> *id_entries)
{
  open_elem_t stack[MAX_DEPTH];
  unsigned depth = 0;
  defs_entries->alloc (256);
  id_entries->alloc (256);

  unsigned defs_depth = 0;
  unsigned i = 0;
  while (i < len)
  {
    if (svg[i] != '<')
    {
      i++;
      continue;
    }

    if (i + 4 <= len && !memcmp (svg + i, "<!--", 4))
    {
      int cend = find_substr (svg, len, i + 4, "-->", 3);
      if (cend < 0) return false;
      i = (unsigned) cend + 3;
      continue;
    }
    if (i + 9 <= len && !memcmp (svg + i, "<![CDATA[", 9))
    {
      int cend = find_substr (svg, len, i + 9, "]]>", 3);
      if (cend < 0) return false;
      i = (unsigned) cend + 3;
      continue;
    }

    bool closing = (i + 1 < len && svg[i + 1] == '/');
    bool special = (i + 1 < len && (svg[i + 1] == '!' || svg[i + 1] == '?'));

    unsigned gt = i + 1;
    char quote = 0;
    while (gt < len)
    {
      char c = svg[gt];
      if (quote)
      {
        if (c == quote) quote = 0;
      }
      else
      {
        if (c == '"' || c == '\'')
          quote = c;
        else if (c == '>')
          break;
      }
      gt++;
    }
    if (gt >= len)
      return false;

    if (special)
    {
      i = gt + 1;
      continue;
    }

    unsigned p = i + (closing ? 2 : 1);
    while (p < gt && isspace ((unsigned char) svg[p])) p++;
    const char *name = svg + p;
    unsigned name_len = 0;
    while (p + name_len < gt)
    {
      unsigned char c = (unsigned char) name[name_len];
      if (!(isalnum (c) || c == '_' || c == '-' || c == ':'))
        break;
      name_len++;
    }
    bool is_defs = (name_len == 4 && !memcmp (name, "defs", 4));

    if (closing)
    {
      if (!depth)
      {
        i = gt + 1;
        continue;
      }

      open_elem_t e = stack[--depth];
      unsigned end = gt + 1;

      if (e.id.len)
      {
        auto *id_slot = id_entries->push ();
        if (unlikely (!id_slot))
          return false;
        *id_slot = {e.id, (uint32_t) e.start, (uint32_t) end};

        if (e.in_defs_content)
        {
          auto *slot = defs_entries->push ();
          if (unlikely (!slot))
            return false;
          slot->id = e.id;
          slot->start = e.start;
          slot->end = end;
        }

        hb_codepoint_t gid;
        if (parse_glyph_id_span (e.id, &gid))
        {
          auto *span = glyph_spans->push ();
          if (unlikely (!span))
            return false;
          span->glyph = gid;
          span->start = (uint32_t) e.start;
          span->end = (uint32_t) end;
        }
      }

      if (e.is_defs && defs_depth)
        defs_depth--;

      i = end;
      continue;
    }

    SVG::svg_id_span_t id = {nullptr, 0};
    parse_id_in_start_tag (svg, i, gt, &id);

    unsigned r = gt;
    while (r > i && isspace ((unsigned char) svg[r - 1])) r--;
    bool self_closing = (r > i && svg[r - 1] == '/');

    open_elem_t e = {i, id, defs_depth > 0, is_defs};

    if (self_closing)
    {
      unsigned end = gt + 1;
      if (e.id.len)
      {
        auto *id_slot = id_entries->push ();
        if (unlikely (!id_slot))
          return false;
        *id_slot = {e.id, (uint32_t) e.start, (uint32_t) end};

        if (e.in_defs_content)
        {
          auto *slot = defs_entries->push ();
          if (unlikely (!slot))
            return false;
          slot->id = e.id;
          slot->start = e.start;
          slot->end = end;
        }

        hb_codepoint_t gid;
        if (parse_glyph_id_span (e.id, &gid))
        {
          auto *span = glyph_spans->push ();
          if (unlikely (!span))
            return false;
          span->glyph = gid;
          span->start = (uint32_t) e.start;
          span->end = (uint32_t) end;
        }
      }
    }
    else
    {
      if (unlikely (depth >= MAX_DEPTH))
        return false;
      stack[depth++] = e;
      if (is_defs)
        defs_depth++;
    }

    i = gt + 1;
  }

  return true;
}

} /* namespace _hb_svg_cache_impl */

inline
SVG::accelerator_t::accelerator_t (hb_face_t *face)
{
  table = hb_sanitize_context_t ().reference_table<SVG> (face);
  doc_caches.init ();
  unsigned doc_count = table->get_document_count ();
  if (doc_count && unlikely (!doc_caches.resize (doc_count)))
    doc_caches.resize (0);
  for (unsigned i = 0; i < doc_caches.length; i++)
    doc_caches.arrayZ[i].set_relaxed (nullptr);
}

inline
SVG::accelerator_t::~accelerator_t ()
{
  for (unsigned i = 0; i < doc_caches.length; i++)
    destroy_doc_cache (doc_caches.arrayZ[i].get_relaxed ());
  doc_caches.fini ();
  table.destroy ();
}

inline void
SVG::accelerator_t::destroy_doc_cache (svg_doc_cache_t *doc)
{
  if (!doc)
    return;
  doc->glyph_spans.fini ();
  doc->defs_entries.fini ();
  doc->id_spans.fini ();
  hb_blob_destroy (doc->blob);
  hb_free (doc);
}

inline SVG::svg_doc_cache_t *
SVG::accelerator_t::make_doc_cache (hb_blob_t *image,
                                    const char *svg,
                                    unsigned len,
                                    hb_codepoint_t start_glyph,
                                    hb_codepoint_t end_glyph) const
{
  static const uint32_t INVALID_SPAN = 0xFFFFFFFFu;

  auto *doc = (svg_doc_cache_t *) hb_malloc (sizeof (svg_doc_cache_t));
  if (!doc)
    return nullptr;

  doc->blob = nullptr;
  doc->svg = nullptr;
  doc->len = 0;
  doc->defs_entries.init ();
  doc->start_glyph = HB_CODEPOINT_INVALID;
  doc->end_glyph = HB_CODEPOINT_INVALID;
  doc->glyph_spans.init ();
  doc->id_spans.init ();

  doc->blob = hb_blob_reference (image);
  doc->svg = svg;
  doc->len = len;
  doc->start_glyph = start_glyph;
  doc->end_glyph = end_glyph;

  if (unlikely (start_glyph == HB_CODEPOINT_INVALID || end_glyph < start_glyph))
  {
    destroy_doc_cache (doc);
    return nullptr;
  }

  unsigned glyph_count = end_glyph - start_glyph + 1;
  if (!doc->glyph_spans.resize ((int) glyph_count))
  {
    destroy_doc_cache (doc);
    return nullptr;
  }
  for (unsigned i = 0; i < glyph_count; i++)
    doc->glyph_spans.arrayZ[i] = hb_pair_t<uint32_t, uint32_t> (INVALID_SPAN, INVALID_SPAN);

  hb_vector_t<_hb_svg_cache_impl::glyph_entry_t> glyph_spans;
  glyph_spans.init ();
  hb_vector_t<_hb_svg_cache_impl::id_entry_t> id_entries;
  id_entries.init ();
  if (!_hb_svg_cache_impl::parse_cache_entries_linear (svg, len,
                                                       &doc->defs_entries,
                                                       &glyph_spans,
                                                       &id_entries))
  {
    id_entries.fini ();
    glyph_spans.fini ();
    destroy_doc_cache (doc);
    return nullptr;
  }

  for (unsigned i = 0; i < glyph_spans.length; i++)
  {
    const auto &span = glyph_spans.arrayZ[i];
    if (unlikely (span.glyph < start_glyph || span.glyph > end_glyph))
      continue;
    doc->glyph_spans.arrayZ[span.glyph - start_glyph] = hb_pair_t<uint32_t, uint32_t> (span.start, span.end);
  }

  for (unsigned i = 0; i < id_entries.length; i++)
  {
    const auto &e = id_entries.arrayZ[i];
    hb_pair_t<uint32_t, uint32_t> *out = nullptr;
    if (doc->id_spans.has (e.id, &out))
      continue;
    if (unlikely (!doc->id_spans.set (e.id, hb_pair_t<uint32_t, uint32_t> (e.start, e.end))))
    {
      id_entries.fini ();
      glyph_spans.fini ();
      destroy_doc_cache (doc);
      return nullptr;
    }
  }

  id_entries.fini ();
  glyph_spans.fini ();
  return doc;
}

inline const SVG::svg_doc_cache_t *
SVG::accelerator_t::get_or_create_doc_cache (hb_blob_t *image,
                                              const char *svg,
                                              unsigned len,
                                              unsigned doc_index,
                                              hb_codepoint_t start_glyph,
                                              hb_codepoint_t end_glyph) const
{
  if (doc_index >= doc_caches.length)
    return nullptr;

  auto &slot = doc_caches.arrayZ[doc_index];
  auto *doc = slot.get_acquire ();
  if (doc)
    return doc;

  auto *fresh = make_doc_cache (image, svg, len, start_glyph, end_glyph);
  if (!fresh)
    return nullptr;

  auto *expected = (svg_doc_cache_t *) nullptr;
  if (slot.cmpexch (expected, fresh))
    return fresh;

  destroy_doc_cache (fresh);
  return expected;
}

inline const char *
SVG::accelerator_t::doc_cache_get_svg (const svg_doc_cache_t *doc,
                                       unsigned *len) const
{
  if (!doc)
  {
    if (len) *len = 0;
    return nullptr;
  }
  if (len) *len = doc->len;
  return doc->svg;
}

inline const hb_vector_t<SVG::svg_defs_entry_t> *
SVG::accelerator_t::doc_cache_get_defs_entries (const svg_doc_cache_t *doc) const
{
  return doc ? &doc->defs_entries : nullptr;
}

inline bool
SVG::accelerator_t::doc_cache_get_glyph_span (const svg_doc_cache_t *doc,
                                              hb_codepoint_t glyph,
                                              unsigned *start,
                                              unsigned *end) const
{
  static const uint32_t INVALID_SPAN = 0xFFFFFFFFu;
  if (!doc || doc->start_glyph == HB_CODEPOINT_INVALID ||
      glyph < doc->start_glyph || glyph > doc->end_glyph)
    return false;

  const auto &span = doc->glyph_spans.arrayZ[glyph - doc->start_glyph];
  if (span.first == INVALID_SPAN)
    return false;

  if (start) *start = span.first;
  if (end) *end = span.second;
  return true;
}

inline bool
SVG::accelerator_t::doc_cache_find_id_span (const svg_doc_cache_t *doc,
                                            svg_id_span_t id,
                                            unsigned *start,
                                            unsigned *end) const
{
  if (!doc || !id.p || !id.len)
    return false;
  hb_pair_t<uint32_t, uint32_t> *span = nullptr;
  if (!doc->id_spans.has (id, &span))
    return false;
  if (start) *start = span->first;
  if (end) *end = span->second;
  return true;
}

inline bool
SVG::accelerator_t::doc_cache_find_id_cstr (const svg_doc_cache_t *doc,
                                            const char *id,
                                            unsigned *start,
                                            unsigned *end) const
{
  if (!id) return false;
  svg_id_span_t key = {id, (unsigned) strlen (id)};
  return doc_cache_find_id_span (doc, key, start, end);
}

struct SVG_accelerator_t : SVG::accelerator_t {
  SVG_accelerator_t (hb_face_t *face) : SVG::accelerator_t (face) {}
};

} /* namespace OT */


#endif /* OT_COLOR_SVG_SVG_HH */

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

#include "hb-vector-paint.hh"
#include "hb-vector-draw.hh"
#include "hb-paint.hh"

#include <math.h>
#include <stdio.h>

#ifdef HAVE_ZLIB
#include <zlib.h>
#endif

/* PDF paint backend for COLRv0/v1 color font rendering.
 *
 * Supports:
 *   - Solid colors with alpha (via ExtGState)
 *   - Glyph and rectangle clipping
 *   - Affine transforms
 *   - Groups (save/restore)
 *   - Linear gradients (PDF Type 2 axial shading)
 *   - Radial gradients (PDF Type 3 radial shading)
 *   - Sweep gradients (approximated via solid fallback)
 */


/* ---- PDF object collector ---- */

struct hb_pdf_obj_t
{
  hb_vector_buf_t data;
};

/* Collects extra PDF objects (shadings, functions, ExtGState)
 * during painting.  Referenced from the content stream by name
 * (e.g. /SH0, /GS0) and emitted at render time. */
static bool
hb_pdf_build_indexed_smask (hb_vector_buf_t *out,
			    const char *idat_data, unsigned idat_len,
			    unsigned width, unsigned height,
			    const uint8_t *trns, unsigned trns_len);

struct hb_pdf_resources_t
{
  hb_vector_t<hb_pdf_obj_t> objects;   /* extra objects, starting at id 5 */
  hb_vector_buf_t extgstate_dict;    /* /GS0 5 0 R /GS1 6 0 R ... */
  hb_vector_buf_t shading_dict;      /* /SH0 7 0 R ... */
  hb_vector_buf_t xobject_dict;      /* /Im0 8 0 R ... */
  unsigned extgstate_count = 0;
  unsigned shading_count = 0;
  unsigned xobject_count = 0;

  unsigned add_object (hb_vector_buf_t &&obj_data)
  {
    unsigned id = 5 + objects.length; /* objects 1-4 are fixed */
    hb_pdf_obj_t obj;
    obj.data = std::move (obj_data);
    objects.push (std::move (obj));
    return id;
  }

  /* Add ExtGState for fill opacity, return resource name index. */
  unsigned add_extgstate_alpha (float alpha)
  {
    unsigned idx = extgstate_count++;
    hb_vector_buf_t obj;
    obj.append_str ("<< /Type /ExtGState /ca ");
    obj.append_num (alpha, 4);
    obj.append_str (" >>");
    unsigned obj_id = add_object (std::move (obj));

    extgstate_dict.append_str ("/GS");
    extgstate_dict.append_unsigned (idx);
    extgstate_dict.append_c (' ');
    extgstate_dict.append_unsigned (obj_id);
    extgstate_dict.append_str (" 0 R ");
    return idx;
  }

  /* Add ExtGState for blend mode, return resource name index. */
  unsigned add_extgstate_blend (const char *bm)
  {
    unsigned idx = extgstate_count++;
    hb_vector_buf_t obj;
    obj.append_str ("<< /Type /ExtGState /BM /");
    obj.append_str (bm);
    obj.append_str (" >>");
    unsigned obj_id = add_object (std::move (obj));

    extgstate_dict.append_str ("/GS");
    extgstate_dict.append_unsigned (idx);
    extgstate_dict.append_c (' ');
    extgstate_dict.append_unsigned (obj_id);
    extgstate_dict.append_str (" 0 R ");
    return idx;
  }

  /* Add ExtGState with an SMask (soft mask) referencing a Form XObject
   * that paints a DeviceGray shading.  Returns the ExtGState resource index. */
  unsigned add_extgstate_smask (unsigned alpha_shading_id,
				float bbox_x, float bbox_y,
				float bbox_w, float bbox_h,
				unsigned precision)
  {
    /* Form XObject: transparency group painting the alpha shading. */
    hb_vector_buf_t form_stream;
    form_stream.append_str ("/SHa sh\n");

    hb_vector_buf_t form;
    form.append_str ("<< /Type /XObject /Subtype /Form\n");
    form.append_str ("/BBox [");
    form.append_num (bbox_x, precision);
    form.append_c (' ');
    form.append_num (bbox_y, precision);
    form.append_c (' ');
    form.append_num (bbox_x + bbox_w, precision);
    form.append_c (' ');
    form.append_num (bbox_y + bbox_h, precision);
    form.append_str ("]\n/Group << /S /Transparency /CS /DeviceGray >>\n");
    form.append_str ("/Resources << /Shading << /SHa ");
    form.append_unsigned (alpha_shading_id);
    form.append_str (" 0 R >> >>\n");
    form.append_str ("/Length ");
    form.append_unsigned (form_stream.length);
    form.append_str (" >>\nstream\n");
    form.append_len (form_stream.arrayZ, form_stream.length);
    form.append_str ("endstream");
    unsigned form_id = add_object (std::move (form));

    /* ExtGState with luminosity soft mask. */
    unsigned idx = extgstate_count++;
    hb_vector_buf_t gs;
    gs.append_str ("<< /Type /ExtGState\n");
    gs.append_str ("/SMask << /Type /Mask /S /Luminosity /G ");
    gs.append_unsigned (form_id);
    gs.append_str (" 0 R >> >>");
    unsigned gs_id = add_object (std::move (gs));

    extgstate_dict.append_str ("/GS");
    extgstate_dict.append_unsigned (idx);
    extgstate_dict.append_c (' ');
    extgstate_dict.append_unsigned (gs_id);
    extgstate_dict.append_str (" 0 R ");
    return idx;
  }

  /* Add a shading, return resource name index. */
  unsigned add_shading (hb_vector_buf_t &&shading_data)
  {
    unsigned obj_id = add_object (std::move (shading_data));
    return add_shading_by_id (obj_id);
  }

  /* Register an already-allocated object as a shading resource. */
  unsigned add_shading_by_id (unsigned obj_id)
  {
    unsigned idx = shading_count++;
    shading_dict.append_str ("/SH");
    shading_dict.append_unsigned (idx);
    shading_dict.append_c (' ');
    shading_dict.append_unsigned (obj_id);
    shading_dict.append_str (" 0 R ");
    return idx;
  }

  /* Add an XObject image, return resource name index.
   * idat_data/idat_len is the concatenated PNG IDAT payload (zlib data).
   * colors is 1 (gray), 3 (RGB), or 4 (RGBA); for indexed, pass colors=1.
   * plte/plte_len is the PLTE chunk data for indexed images (may be null). */
  unsigned add_xobject_png_image (const char *idat_data, unsigned idat_len,
				  unsigned width, unsigned height,
				  unsigned colors, bool has_alpha,
				  const uint8_t *plte = nullptr,
				  unsigned plte_len = 0,
				  const uint8_t *trns = nullptr,
				  unsigned trns_len = 0)
  {
    unsigned idx = xobject_count++;
    hb_vector_buf_t obj;
    obj.append_str ("<< /Type /XObject /Subtype /Image\n");
    obj.append_str ("/Width ");
    obj.append_unsigned (width);
    obj.append_str (" /Height ");
    obj.append_unsigned (height);
    obj.append_str ("\n/BitsPerComponent 8\n");

    if (plte && plte_len >= 3)
    {
      /* Indexed color: /ColorSpace [/Indexed /DeviceRGB N <hex palette>] */
      unsigned n_entries = plte_len / 3;
      obj.append_str ("/ColorSpace [/Indexed /DeviceRGB ");
      obj.append_unsigned (n_entries - 1);
      obj.append_str (" <");
      for (unsigned i = 0; i < n_entries * 3; i++)
      {
	obj.append_c ("0123456789ABCDEF"[plte[i] >> 4]);
	obj.append_c ("0123456789ABCDEF"[plte[i] & 0xF]);
      }
      obj.append_str (">]\n");
    }
    else
    {
      obj.append_str ("/ColorSpace ");
      unsigned color_channels = has_alpha ? colors - 1 : colors;
      obj.append_str (color_channels == 1 ? "/DeviceGray" : "/DeviceRGB");
      obj.append_c ('\n');
    }

    /* Build SMask for indexed images with tRNS transparency. */
    unsigned smask_id = 0;
    if (plte && trns && trns_len)
    {
      hb_vector_buf_t smask_stream;
      if (hb_pdf_build_indexed_smask (&smask_stream, idat_data, idat_len,
				      width, height, trns, trns_len))
      {
	hb_vector_buf_t smask_obj;
	smask_obj.append_str ("<< /Type /XObject /Subtype /Image\n");
	smask_obj.append_str ("/Width ");
	smask_obj.append_unsigned (width);
	smask_obj.append_str (" /Height ");
	smask_obj.append_unsigned (height);
	smask_obj.append_str ("\n/ColorSpace /DeviceGray /BitsPerComponent 8\n");
	smask_obj.append_str ("/Length ");
	smask_obj.append_unsigned (smask_stream.length);
	smask_obj.append_str (" >>\nstream\n");
	smask_obj.append_len (smask_stream.arrayZ, smask_stream.length);
	smask_obj.append_str ("\nendstream");
	smask_id = add_object (std::move (smask_obj));
      }
    }

    if (smask_id)
    {
      obj.append_str ("/SMask ");
      obj.append_unsigned (smask_id);
      obj.append_str (" 0 R\n");
    }

    obj.append_str ("/Filter /FlateDecode\n");
    obj.append_str ("/DecodeParms << /Predictor 15 /Colors ");
    obj.append_unsigned (colors);
    obj.append_str (" /BitsPerComponent 8 /Columns ");
    obj.append_unsigned (width);
    obj.append_str (" >>\n");
    obj.append_str ("/Length ");
    obj.append_unsigned (idat_len);
    obj.append_str (" >>\nstream\n");
    obj.append_len (idat_data, idat_len);
    obj.append_str ("\nendstream");

    (void) has_alpha;

    unsigned obj_id = add_object (std::move (obj));

    xobject_dict.append_str ("/Im");
    xobject_dict.append_unsigned (idx);
    xobject_dict.append_c (' ');
    xobject_dict.append_unsigned (obj_id);
    xobject_dict.append_str (" 0 R ");
    return idx;
  }
};

/* Store resources pointer in the paint struct's defs buffer
 * (repurposed — defs is unused for PDF). */
static hb_pdf_resources_t *
hb_pdf_get_resources (hb_vector_paint_t *paint)
{
  if (!paint->defs.length)
  {
    /* First call: allocate and store. */
    auto *res = (hb_pdf_resources_t *) hb_calloc (1, sizeof (hb_pdf_resources_t));
    if (unlikely (!res)) return nullptr;
    new (res) hb_pdf_resources_t ();
    if (unlikely (!paint->defs.resize (sizeof (void *))))
    {
      res->~hb_pdf_resources_t ();
      hb_free (res);
      return nullptr;
    }
    memcpy (paint->defs.arrayZ, &res, sizeof (void *));
  }
  hb_pdf_resources_t *res;
  memcpy (&res, paint->defs.arrayZ, sizeof (void *));
  return res;
}

void
hb_vector_paint_pdf_free_resources (hb_vector_paint_t *paint)
{
  if (paint->defs.length >= sizeof (void *))
  {
    hb_pdf_resources_t *res;
    memcpy (&res, paint->defs.arrayZ, sizeof (void *));
    if (res)
    {
      res->~hb_pdf_resources_t ();
      hb_free (res);
    }
  }
  paint->defs.clear ();
}


/* ---- helpers ---- */


/* Emit a glyph outline as PDF path operators into buf. */
static void
hb_pdf_emit_glyph_path (hb_vector_paint_t *paint,
			 hb_font_t *font,
			 hb_codepoint_t glyph,
			 hb_vector_buf_t *buf)
{
  hb_vector_path_sink_t sink = {&paint->path, paint->get_precision (),
			       paint->x_scale_factor, paint->y_scale_factor};
  paint->path.clear ();
  hb_font_draw_glyph (font, glyph,
		       hb_vector_pdf_path_draw_funcs_get (),
		       &sink);
  buf->append_len (paint->path.arrayZ, paint->path.length);
  paint->path.clear ();
}


/* ---- paint callbacks ---- */

static void
hb_pdf_paint_push_transform (hb_paint_funcs_t *,
			     void *paint_data,
			     float xx, float yx,
			     float xy, float yy,
			     float dx, float dy,
			     void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!paint->ensure_initialized ()))
    return;

  auto &body = paint->current_body ();
  unsigned sprec = body.scale_precision ();
  body.append_str ("q\n");
  body.append_num (xx, sprec);
  body.append_c (' ');
  body.append_num (yx, sprec);
  body.append_c (' ');
  body.append_num (xy, sprec);
  body.append_c (' ');
  body.append_num (yy, sprec);
  body.append_c (' ');
  body.append_num (paint->sx (dx));
  body.append_c (' ');
  body.append_num (paint->sy (dy));
  body.append_str (" cm\n");
}

static void
hb_pdf_paint_pop_transform (hb_paint_funcs_t *,
			    void *paint_data,
			    void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!paint->ensure_initialized ()))
    return;
  paint->current_body ().append_str ("Q\n");
}

static void
hb_pdf_paint_push_clip_glyph (hb_paint_funcs_t *,
			      void *paint_data,
			      hb_codepoint_t glyph,
			      hb_font_t *font,
			      void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!paint->ensure_initialized ()))
    return;

  auto &body = paint->current_body ();
  body.append_str ("q\n");
  hb_pdf_emit_glyph_path (paint, font, glyph, &body);
  body.append_str ("W n\n");
}

static void
hb_pdf_paint_push_clip_rectangle (hb_paint_funcs_t *,
				  void *paint_data,
				  float xmin, float ymin,
				  float xmax, float ymax,
				  void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!paint->ensure_initialized ()))
    return;

  auto &body = paint->current_body ();
  body.append_str ("q\n");
  body.append_num (paint->sx (xmin));
  body.append_c (' ');
  body.append_num (paint->sy (ymin));
  body.append_c (' ');
  body.append_num (paint->sx (xmax - xmin));
  body.append_c (' ');
  body.append_num (paint->sy (ymax - ymin));
  body.append_str (" re W n\n");
}

static hb_draw_funcs_t *
hb_pdf_paint_push_clip_path_start (hb_paint_funcs_t *,
				   void *paint_data,
				   void **draw_data,
				   void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!paint->ensure_initialized ()))
  {
    *draw_data = nullptr;
    return nullptr;
  }

  auto &body = paint->current_body ();
  body.append_str ("q\n");
  /* Stream path operators straight into the body; end() seals
   * the path with "W n" to turn it into the clip region.
   * Coordinates arrive in font-scale; the sink divides by
   * scale_factor so they land in output space. */
  paint->clip_path_sink = {&body, paint->get_precision (),
			   paint->x_scale_factor,
			   paint->y_scale_factor};
  *draw_data = &paint->clip_path_sink;
  return hb_vector_pdf_path_draw_funcs_get ();
}

static void
hb_pdf_paint_push_clip_path_end (hb_paint_funcs_t *,
				 void *paint_data,
				 void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!paint->ensure_initialized ()))
    return;
  paint->current_body ().append_str ("W n\n");
}

static void
hb_pdf_paint_pop_clip (hb_paint_funcs_t *,
		       void *paint_data,
		       void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!paint->ensure_initialized ()))
    return;
  paint->current_body ().append_str ("Q\n");
}

/* Paint a solid color, including alpha via ExtGState. */
static void
hb_pdf_paint_solid_color (hb_vector_paint_t *paint, hb_color_t c)
{
  auto &body = paint->current_body ();

  float r = hb_color_get_red (c) / 255.f;
  float g = hb_color_get_green (c) / 255.f;
  float b = hb_color_get_blue (c) / 255.f;
  float a = hb_color_get_alpha (c) / 255.f;

  if (a < 1.f / 255.f)
    return;

  /* Set alpha via ExtGState if needed. */
  if (a < 1.f - 1.f / 512.f)
  {
    auto *res = hb_pdf_get_resources (paint);
    if (res)
    {
      unsigned gs_idx = res->add_extgstate_alpha (a);
      body.append_str ("/GS");
      body.append_unsigned (gs_idx);
      body.append_str (" gs\n");
    }
  }

  /* Set fill color. */
  body.append_num (r, 4);
  body.append_c (' ');
  body.append_num (g, 4);
  body.append_c (' ');
  body.append_num (b, 4);
  body.append_str (" rg\n");

  /* Paint a huge rect (will be clipped). */
  body.append_str ("-32767 -32767 65534 65534 re f\n");
}

static void
hb_pdf_paint_color (hb_paint_funcs_t *,
		    void *paint_data,
		    hb_bool_t,
		    hb_color_t color,
		    void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!paint->ensure_initialized ()))
    return;

  hb_pdf_paint_solid_color (paint, color);
}

/* Build an uncompressed alpha mask from indexed PNG IDAT data + tRNS.
 * Decompresses IDAT, un-filters PNG rows, maps palette indices to alpha. */
static bool
hb_pdf_build_indexed_smask (hb_vector_buf_t *out,
			    const char *idat_data, unsigned idat_len,
			    unsigned width, unsigned height,
			    const uint8_t *trns, unsigned trns_len)
{
#ifndef HAVE_ZLIB
  (void) out; (void) idat_data; (void) idat_len;
  (void) width; (void) height; (void) trns; (void) trns_len;
  return false;
#else
  /* Decompress IDAT (zlib). */
  unsigned raw_len = (width + 1) * height; /* 1 filter byte per row + width bytes */
  uint8_t *raw = (uint8_t *) hb_malloc (raw_len);
  if (!raw) return false;
  HB_SCOPE_GUARD (hb_free (raw));

  z_stream stream = {};
  stream.next_in = (Bytef *) idat_data;
  stream.avail_in = idat_len;
  stream.next_out = (Bytef *) raw;
  stream.avail_out = raw_len;

  if (inflateInit (&stream) != Z_OK)
    return false;
  int status = inflate (&stream, Z_FINISH);
  unsigned long total_out = stream.total_out;
  inflateEnd (&stream);
  if (status != Z_STREAM_END || total_out != raw_len)
    return false;

  /* Un-filter and map to alpha. */
  if (!out->resize (width * height))
    return false;

  uint8_t *unfiltered = (uint8_t *) hb_malloc (width);
  if (!unfiltered)
    return false;
  HB_SCOPE_GUARD (hb_free (unfiltered));

  uint8_t *prev_unfiltered = (uint8_t *) hb_calloc (width, 1);
  if (!prev_unfiltered)
    return false;
  HB_SCOPE_GUARD (hb_free (prev_unfiltered));

  for (unsigned y = 0; y < height; y++)
  {
    uint8_t *row = raw + y * (width + 1);
    uint8_t filter = row[0];
    uint8_t *pixels = row + 1;

    for (unsigned x = 0; x < width; x++)
    {
      uint8_t a = (x > 0) ? unfiltered[x - 1] : 0;
      uint8_t b = prev_unfiltered[x];
      uint8_t c = (x > 0) ? prev_unfiltered[x - 1] : 0;
      uint8_t val = pixels[x];

      switch (filter)
      {
      case 0: unfiltered[x] = val; break; /* None */
      case 1: unfiltered[x] = val + a; break; /* Sub */
      case 2: unfiltered[x] = val + b; break; /* Up */
      case 3: unfiltered[x] = val + (uint8_t) (((unsigned) a + b) / 2); break; /* Average */
      case 4: /* Paeth */
      {
	int p = (int) a + (int) b - (int) c;
	int pa = abs (p - (int) a);
	int pb = abs (p - (int) b);
	int pc = abs (p - (int) c);
	uint8_t pr = (pa <= pb && pa <= pc) ? a : (pb <= pc) ? b : c;
	unfiltered[x] = val + pr;
	break;
      }
      default: unfiltered[x] = val; break;
      }

      /* Map palette index to alpha. */
      uint8_t idx = unfiltered[x];
      out->arrayZ[y * width + x] = (idx < trns_len) ? trns[idx] : 0xFF;
    }

    /* Swap buffers. */
    uint8_t *tmp = prev_unfiltered;
    prev_unfiltered = unfiltered;
    unfiltered = tmp;
  }

  return true;
#endif
}

/* Read a big-endian uint32 from a byte pointer. */
static inline uint32_t
hb_pdf_png_u32 (const uint8_t *p)
{
  return ((uint32_t) p[0] << 24) | ((uint32_t) p[1] << 16) |
	 ((uint32_t) p[2] << 8)  | (uint32_t) p[3];
}

static hb_bool_t
hb_pdf_paint_image (hb_paint_funcs_t *,
		    void *paint_data,
		    hb_blob_t *image,
		    unsigned width,
		    unsigned height,
		    hb_tag_t format,
		    float slant HB_UNUSED,
		    hb_glyph_extents_t *extents,
		    void *)
{
  if (format != HB_TAG ('p','n','g',' '))
    return false;
  if (!extents || !width || !height)
    return false;

  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!paint->ensure_initialized ()))
    return false;
  auto *res = hb_pdf_get_resources (paint);
  if (unlikely (!res))
    return false;

  unsigned len = 0;
  const uint8_t *data = (const uint8_t *) hb_blob_get_data (image, &len);
  if (!data || len < 8)
    return false;

  /* Verify PNG signature. */
  static const uint8_t png_sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  if (hb_memcmp (data, png_sig, 8) != 0)
    return false;

  /* Parse PNG chunks: extract IHDR and concatenate IDAT payloads. */
  unsigned png_w = 0, png_h = 0;
  unsigned colors = 3; /* default RGB */
  uint8_t color_type = 2;
  bool has_alpha = false;
  const uint8_t *plte_data = nullptr;
  unsigned plte_len = 0;
  const uint8_t *trns_data = nullptr;
  unsigned trns_len = 0;
  hb_vector_buf_t idat;

  unsigned pos = 8;
  while (pos + 12 <= len)
  {
    uint32_t chunk_len = hb_pdf_png_u32 (data + pos);
    uint32_t chunk_type = hb_pdf_png_u32 (data + pos + 4);
    if (pos + 12 + chunk_len > len)
      break;
    const uint8_t *chunk_data = data + pos + 8;

    if (chunk_type == 0x49484452u) /* IHDR */
    {
      if (chunk_len < 13) return false;
      png_w = hb_pdf_png_u32 (chunk_data);
      png_h = hb_pdf_png_u32 (chunk_data + 4);
      uint8_t bit_depth = chunk_data[8];
      color_type = chunk_data[9];
      if (bit_depth != 8) return false; /* only 8-bit supported */

      switch (color_type)
      {
      case 0: colors = 1; has_alpha = false; break; /* Grayscale */
      case 2: colors = 3; has_alpha = false; break; /* RGB */
      case 3: colors = 1; has_alpha = false; break; /* Indexed */
      case 4: colors = 2; has_alpha = true;  break; /* Gray+Alpha */
      case 6: colors = 4; has_alpha = true;  break; /* RGBA */
      default: return false;
      }
    }
    else if (chunk_type == 0x504C5445u) /* PLTE */
    {
      plte_data = chunk_data;
      plte_len = chunk_len;
    }
    else if (chunk_type == 0x74524E53u) /* tRNS */
    {
      trns_data = chunk_data;
      trns_len = chunk_len;
    }
    else if (chunk_type == 0x49444154u) /* IDAT */
    {
      idat.append_len ((const char *) chunk_data, chunk_len);
    }
    else if (chunk_type == 0x49454E44u) /* IEND */
      break;

    pos += 12 + chunk_len;
  }

  if (!png_w || !png_h || !idat.length)
    return false;

  unsigned im_idx = res->add_xobject_png_image (idat.arrayZ, idat.length,
						 png_w, png_h,
						 colors, has_alpha,
						 plte_data, plte_len,
						 trns_data, trns_len);

  /* Emit: save state, set CTM to map image (0,0)-(1,1) to extents, paint. */
  auto &body = paint->current_body ();
  body.append_str ("q\n");

  /* Image space: (0,0) at bottom-left, (1,1) at top-right.
   * We need to map to extents (x_bearing, y_bearing+height) .. (x_bearing+width, y_bearing).
   * Since font coords are Y-up like PDF, y_bearing is the top. */
  float ix = (float) extents->x_bearing;
  float iy = (float) extents->y_bearing + (float) extents->height;
  float iw = (float) extents->width;
  float ih = (float) -extents->height; /* negative because image Y goes up but height is negative in extents */

  body.append_num (paint->sx (iw));
  body.append_str (" 0 0 ");
  body.append_num (paint->sy (ih));
  body.append_c (' ');
  body.append_num (paint->sx (ix));
  body.append_c (' ');
  body.append_num (paint->sy (iy));
  body.append_str (" cm\n");

  body.append_str ("/Im");
  body.append_unsigned (im_idx);
  body.append_str (" Do\nQ\n");

  return true;
}

/* ---- Gradient helpers ---- */

static bool
hb_pdf_gradient_needs_alpha (hb_array_t<const hb_color_stop_t> stops)
{
  for (const auto &s : stops)
    if (hb_color_get_alpha (s.color) != 255)
      return true;
  return false;
}

/* Build a PDF Type 2 (exponential interpolation) function for
 * a single alpha stop pair (DeviceGray, scalar output). */
static void
hb_pdf_build_alpha_interpolation_function (hb_vector_buf_t *obj,
					   float a0, float a1)
{
  obj->append_str ("<< /FunctionType 2 /Domain [0 1] /N 1\n");
  obj->append_str ("/C0 [");
  obj->append_num (a0, 4);
  obj->append_str ("]\n/C1 [");
  obj->append_num (a1, 4);
  obj->append_str ("] >>");
}

/* Build a stitching function (Type 3) for the alpha channel of
 * pre-populated paint->color_stops_scratch (already sorted+normalized). */
static unsigned
hb_pdf_build_alpha_gradient_function_from_stops (hb_pdf_resources_t *res,
						 hb_vector_paint_t *paint)
{
  unsigned count = paint->color_stops_scratch.length;

  if (count < 2)
  {
    float a = count ? hb_color_get_alpha (paint->color_stops_scratch.arrayZ[0].color) / 255.f : 1.f;
    hb_vector_buf_t obj;
    hb_pdf_build_alpha_interpolation_function (&obj, a, a);
    return res->add_object (std::move (obj));
  }

  if (count == 2)
  {
    hb_vector_buf_t obj;
    hb_pdf_build_alpha_interpolation_function (&obj,
      hb_color_get_alpha (paint->color_stops_scratch.arrayZ[0].color) / 255.f,
      hb_color_get_alpha (paint->color_stops_scratch.arrayZ[1].color) / 255.f);
    return res->add_object (std::move (obj));
  }

  hb_vector_t<unsigned> sub_func_ids;
  for (unsigned i = 0; i + 1 < count; i++)
  {
    hb_vector_buf_t sub;
    hb_pdf_build_alpha_interpolation_function (&sub,
      hb_color_get_alpha (paint->color_stops_scratch.arrayZ[i].color) / 255.f,
      hb_color_get_alpha (paint->color_stops_scratch.arrayZ[i + 1].color) / 255.f);
    sub_func_ids.push (res->add_object (std::move (sub)));
  }

  hb_vector_buf_t obj;
  obj.append_str ("<< /FunctionType 3 /Domain [0 1]\n");
  obj.append_str ("/Functions [");
  for (unsigned i = 0; i < sub_func_ids.length; i++)
  {
    if (i) obj.append_c (' ');
    obj.append_unsigned (sub_func_ids.arrayZ[i]);
    obj.append_str (" 0 R");
  }
  obj.append_str ("]\n/Bounds [");
  for (unsigned i = 1; i + 1 < count; i++)
  {
    if (i > 1) obj.append_c (' ');
    obj.append_num (paint->color_stops_scratch.arrayZ[i].offset, 4);
  }
  obj.append_str ("]\n/Encode [");
  for (unsigned i = 0; i + 1 < count; i++)
  {
    if (i) obj.append_c (' ');
    obj.append_str ("0 1");
  }
  obj.append_str ("] >>");
  return res->add_object (std::move (obj));
}

/* Build a PDF Type 2 (exponential interpolation) function for
 * a single color stop pair. */
static void
hb_pdf_build_interpolation_function (hb_vector_buf_t *obj,
				     hb_color_t c0, hb_color_t c1)
{
  obj->append_str ("<< /FunctionType 2 /Domain [0 1] /N 1\n");
  obj->append_str ("/C0 [");
  obj->append_num (hb_color_get_red (c0) / 255.f, 4);
  obj->append_c (' ');
  obj->append_num (hb_color_get_green (c0) / 255.f, 4);
  obj->append_c (' ');
  obj->append_num (hb_color_get_blue (c0) / 255.f, 4);
  obj->append_str ("]\n/C1 [");
  obj->append_num (hb_color_get_red (c1) / 255.f, 4);
  obj->append_c (' ');
  obj->append_num (hb_color_get_green (c1) / 255.f, 4);
  obj->append_c (' ');
  obj->append_num (hb_color_get_blue (c1) / 255.f, 4);
  obj->append_str ("] >>");
}

/* Build a stitching function (Type 3) from pre-populated
 * paint->color_stops_scratch (already sorted+normalized). */
static unsigned
hb_pdf_build_gradient_function_from_stops (hb_pdf_resources_t *res,
					   hb_vector_paint_t *paint)
{
  unsigned count = paint->color_stops_scratch.length;

  if (count < 2)
  {
    /* Single stop: constant function. */
    hb_color_t c = count ? paint->color_stops_scratch.arrayZ[0].color
			 : HB_COLOR (0, 0, 0, 255);
    hb_vector_buf_t obj;
    hb_pdf_build_interpolation_function (&obj, c, c);
    return res->add_object (std::move (obj));
  }

  /* Sort by offset. */
  paint->color_stops_scratch.as_array ().qsort (
    [] (const hb_color_stop_t &a, const hb_color_stop_t &b)
    { return a.offset < b.offset; });

  if (count == 2)
  {
    /* Two stops: single interpolation function. */
    hb_vector_buf_t obj;
    hb_pdf_build_interpolation_function (&obj,
					  paint->color_stops_scratch.arrayZ[0].color,
					  paint->color_stops_scratch.arrayZ[1].color);
    return res->add_object (std::move (obj));
  }

  /* Multiple stops: create sub-functions and stitch. */
  hb_vector_t<unsigned> sub_func_ids;
  for (unsigned i = 0; i + 1 < count; i++)
  {
    hb_vector_buf_t sub;
    hb_pdf_build_interpolation_function (&sub,
					  paint->color_stops_scratch.arrayZ[i].color,
					  paint->color_stops_scratch.arrayZ[i + 1].color);
    sub_func_ids.push (res->add_object (std::move (sub)));
  }

  /* Stitching function (Type 3). */
  hb_vector_buf_t obj;
  obj.append_str ("<< /FunctionType 3 /Domain [0 1]\n");

  /* Functions array. */
  obj.append_str ("/Functions [");
  for (unsigned i = 0; i < sub_func_ids.length; i++)
  {
    if (i) obj.append_c (' ');
    obj.append_unsigned (sub_func_ids.arrayZ[i]);
    obj.append_str (" 0 R");
  }
  obj.append_str ("]\n");

  /* Bounds. */
  obj.append_str ("/Bounds [");
  for (unsigned i = 1; i + 1 < count; i++)
  {
    if (i > 1) obj.append_c (' ');
    obj.append_num (paint->color_stops_scratch.arrayZ[i].offset, 4);
  }
  obj.append_str ("]\n");

  /* Encode array. */
  obj.append_str ("/Encode [");
  for (unsigned i = 0; i + 1 < count; i++)
  {
    if (i) obj.append_c (' ');
    obj.append_str ("0 1");
  }
  obj.append_str ("] >>");

  return res->add_object (std::move (obj));
}

static void
hb_pdf_paint_linear_gradient (hb_paint_funcs_t *,
			      void *paint_data,
			      hb_color_line_t *color_line,
			      float x0, float y0,
			      float x1, float y1,
			      float x2 HB_UNUSED, float y2 HB_UNUSED,
			      void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!paint->ensure_initialized ()))
    return;
  auto *res = hb_pdf_get_resources (paint);
  if (unlikely (!res))
    return;

  /* Fetch, normalize stops to [0,1], and adjust coordinates. */
  if (!paint->fetch_color_stops (color_line))
    return;
  hb_vector_t<hb_color_stop_t> &stops = paint->color_stops_scratch;

  float mn, mx;
  hb_paint_normalize_color_line (stops.arrayZ, stops.length, &mn, &mx);
  float gx0 = x0 + mn * (x1 - x0);
  float gy0 = y0 + mn * (y1 - y0);
  float gx1 = x0 + mx * (x1 - x0);
  float gy1 = y0 + mx * (y1 - y0);

  unsigned func_id = hb_pdf_build_gradient_function_from_stops (res, paint);

  hb_paint_extend_t extend = hb_color_line_get_extend (color_line);
  const char *extend_str = (extend == HB_PAINT_EXTEND_PAD)
			    ? "/Extend [true true]\n" : "";

  /* Build Type 2 (axial) shading — color only. */
  hb_vector_buf_t sh;
  sh.append_str ("<< /ShadingType 2 /ColorSpace /DeviceRGB\n");
  sh.append_str ("/Coords [");
  sh.append_num (paint->sx (gx0));
  sh.append_c (' ');
  sh.append_num (paint->sy (gy0));
  sh.append_c (' ');
  sh.append_num (paint->sx (gx1));
  sh.append_c (' ');
  sh.append_num (paint->sy (gy1));
  sh.append_str ("]\n/Function ");
  sh.append_unsigned (func_id);
  sh.append_str (" 0 R\n");
  sh.append_str (extend_str);
  sh.append_str (">>");

  unsigned sh_idx = res->add_shading (std::move (sh));

  auto &body = paint->current_body ();

  bool needs_alpha = hb_pdf_gradient_needs_alpha (stops);
  if (needs_alpha)
  {
    unsigned alpha_func_id = hb_pdf_build_alpha_gradient_function_from_stops (res, paint);

    hb_vector_buf_t ash;
    ash.append_str ("<< /ShadingType 2 /ColorSpace /DeviceGray\n");
    ash.append_str ("/Coords [");
    ash.append_num (paint->sx (gx0));
    ash.append_c (' ');
    ash.append_num (paint->sy (gy0));
    ash.append_c (' ');
    ash.append_num (paint->sx (gx1));
    ash.append_c (' ');
    ash.append_num (paint->sy (gy1));
    ash.append_str ("]\n/Function ");
    ash.append_unsigned (alpha_func_id);
    ash.append_str (" 0 R\n");
    ash.append_str (extend_str);
    ash.append_str (">>");
    unsigned alpha_sh_id = res->add_object (std::move (ash));

    unsigned gs_idx = res->add_extgstate_smask (alpha_sh_id,
						paint->sx (gx0), paint->sy (gy0),
						paint->sx (gx1 - gx0), paint->sy (gy1 - gy0),
						paint->get_precision ());
    body.append_str ("/GS");
    body.append_unsigned (gs_idx);
    body.append_str (" gs\n");
  }

  body.append_str ("/SH");
  body.append_unsigned (sh_idx);
  body.append_str (" sh\n");
}

static void
hb_pdf_paint_radial_gradient (hb_paint_funcs_t *,
			      void *paint_data,
			      hb_color_line_t *color_line,
			      float x0, float y0, float r0,
			      float x1, float y1, float r1,
			      void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!paint->ensure_initialized ()))
    return;
  auto *res = hb_pdf_get_resources (paint);
  if (unlikely (!res))
    return;

  /* Fetch, normalize stops to [0,1], and adjust coordinates. */
  if (!paint->fetch_color_stops (color_line))
    return;
  hb_vector_t<hb_color_stop_t> &stops = paint->color_stops_scratch;

  float mn, mx;
  hb_paint_normalize_color_line (stops.arrayZ, stops.length, &mn, &mx);
  float gx0 = x0 + mn * (x1 - x0);
  float gy0 = y0 + mn * (y1 - y0);
  float gr0 = r0 + mn * (r1 - r0);
  float gx1 = x0 + mx * (x1 - x0);
  float gy1 = y0 + mx * (y1 - y0);
  float gr1 = r0 + mx * (r1 - r0);

  unsigned func_id = hb_pdf_build_gradient_function_from_stops (res, paint);

  hb_paint_extend_t extend = hb_color_line_get_extend (color_line);
  const char *extend_str = (extend == HB_PAINT_EXTEND_PAD)
			    ? "/Extend [true true]\n" : "";

  /* Build Type 3 (radial) shading — color only. */
  hb_vector_buf_t sh;
  sh.append_str ("<< /ShadingType 3 /ColorSpace /DeviceRGB\n");
  sh.append_str ("/Coords [");
  sh.append_num (paint->sx (gx0));
  sh.append_c (' ');
  sh.append_num (paint->sy (gy0));
  sh.append_c (' ');
  sh.append_num (paint->sx (gr0));
  sh.append_c (' ');
  sh.append_num (paint->sx (gx1));
  sh.append_c (' ');
  sh.append_num (paint->sy (gy1));
  sh.append_c (' ');
  sh.append_num (paint->sx (gr1));
  sh.append_str ("]\n/Function ");
  sh.append_unsigned (func_id);
  sh.append_str (" 0 R\n");
  sh.append_str (extend_str);
  sh.append_str (">>");

  unsigned sh_idx = res->add_shading (std::move (sh));

  auto &body = paint->current_body ();

  bool needs_alpha = hb_pdf_gradient_needs_alpha (stops);
  if (needs_alpha)
  {
    unsigned alpha_func_id = hb_pdf_build_alpha_gradient_function_from_stops (res, paint);

    hb_vector_buf_t ash;
    ash.append_str ("<< /ShadingType 3 /ColorSpace /DeviceGray\n");
    ash.append_str ("/Coords [");
    ash.append_num (paint->sx (gx0));
    ash.append_c (' ');
    ash.append_num (paint->sy (gy0));
    ash.append_c (' ');
    ash.append_num (paint->sx (gr0));
    ash.append_c (' ');
    ash.append_num (paint->sx (gx1));
    ash.append_c (' ');
    ash.append_num (paint->sy (gy1));
    ash.append_c (' ');
    ash.append_num (paint->sx (gr1));
    ash.append_str ("]\n/Function ");
    ash.append_unsigned (alpha_func_id);
    ash.append_str (" 0 R\n");
    ash.append_str (extend_str);
    ash.append_str (">>");
    unsigned alpha_sh_id = res->add_object (std::move (ash));

    /* BBox: enclosing square of the outer circle. */
    float cx = (gr1 >= gr0) ? gx1 : gx0;
    float cy = (gr1 >= gr0) ? gy1 : gy0;
    float rr = hb_max (gr0, gr1);
    unsigned gs_idx = res->add_extgstate_smask (alpha_sh_id,
						paint->sx (cx - rr), paint->sy (cy - rr),
						paint->sx (2 * rr), paint->sy (2 * rr),
						paint->get_precision ());
    body.append_str ("/GS");
    body.append_unsigned (gs_idx);
    body.append_str (" gs\n");
  }

  body.append_str ("/SH");
  body.append_unsigned (sh_idx);
  body.append_str (" sh\n");
}


/* Encode a 16-bit big-endian unsigned value into buf. */
static void
hb_pdf_encode_u16 (hb_vector_buf_t *buf, uint16_t v)
{
  char bytes[2] = {(char) (v >> 8), (char) (v & 0xFF)};
  buf->append_len (bytes, 2);
}

/* Encode a coordinate as 16-bit value relative to Decode range. */
static void
hb_pdf_encode_coord (hb_vector_buf_t *buf,
		     float val, float lo, float hi)
{
  float t = (val - lo) / (hi - lo);
  t = hb_clamp (t, 0.f, 1.f);
  hb_pdf_encode_u16 (buf, (uint16_t) (t * 65535.f + 0.5f));
}

/* Encode RGB from hb_color_t as 3 bytes. */
static void
hb_pdf_encode_color_rgb (hb_vector_buf_t *buf, hb_color_t c)
{
  char rgb[3] = {(char) hb_color_get_red (c),
		 (char) hb_color_get_green (c),
		 (char) hb_color_get_blue (c)};
  buf->append_len (rgb, 3);
}

/* Encode alpha from hb_color_t as 1 byte (gray). */
static void
hb_pdf_encode_color_alpha (hb_vector_buf_t *buf, hb_color_t c)
{
  char a = (char) hb_color_get_alpha (c);
  buf->append_len (&a, 1);
}

/* Encode one Coons patch control point. */
static void
hb_pdf_encode_point (hb_vector_buf_t *buf,
		     float x, float y,
		     float xlo, float xhi,
		     float ylo, float yhi)
{
  hb_pdf_encode_coord (buf, x, xlo, xhi);
  hb_pdf_encode_coord (buf, y, ylo, yhi);
}

/* Emit one Coons patch sector into the mesh stream(s).
 * Splits large arcs into sub-patches of max 90°.
 * If alpha_mesh is non-null, emits a parallel DeviceGray
 * patch with the alpha channel. */
static void
hb_pdf_add_sweep_patch (hb_vector_buf_t *mesh,
			hb_vector_buf_t *alpha_mesh,
			float cx, float cy,
			float xlo, float xhi, float ylo, float yhi,
			float a0, hb_color_t c0_in,
			float a1, hb_color_t c1_in)
{
  const float R = 32767.f;
  const float eps = 0.5f;
  const float MAX_SECTOR = (float) M_PI / 2.f;

  int num_splits = (int) ceilf (fabsf (a1 - a0) / MAX_SECTOR);
  if (num_splits < 1) num_splits = 1;

  for (int s = 0; s < num_splits; s++)
  {
    float k0 = (float) s / num_splits;
    float k1 = (float) (s + 1) / num_splits;
    float sa0 = a0 + k0 * (a1 - a0);
    float sa1 = a0 + k1 * (a1 - a0);
    hb_color_t sc0 = hb_color_lerp (c0_in, c1_in, k0);
    hb_color_t sc1 = hb_color_lerp (c0_in, c1_in, k1);

    float da = sa1 - sa0;
    float kappa = (4.f / 3.f) * tanf (da / 4.f);

    float cos0 = cosf (sa0), sin0 = sinf (sa0);
    float cos1 = cosf (sa1), sin1 = sinf (sa1);

    float p0x = cx + eps * cos0, p0y = cy + eps * sin0;
    float p3x = cx + R * cos0,   p3y = cy + R * sin0;
    float p6x = cx + R * cos1,   p6y = cy + R * sin1;
    float p9x = cx + eps * cos1, p9y = cy + eps * sin1;

    /* Edge 1: p0→p3, radial straight line. */
    float e1_1x = p0x + (p3x - p0x) / 3.f;
    float e1_1y = p0y + (p3y - p0y) / 3.f;
    float e1_2x = p0x + 2.f * (p3x - p0x) / 3.f;
    float e1_2y = p0y + 2.f * (p3y - p0y) / 3.f;

    /* Edge 2: p3→p6, outer arc. */
    float e2_1x = p3x + kappa * R * (-sin0);
    float e2_1y = p3y + kappa * R * ( cos0);
    float e2_2x = p6x - kappa * R * (-sin1);
    float e2_2y = p6y - kappa * R * ( cos1);

    /* Edge 3: p6→p9, radial straight line. */
    float e3_1x = p6x + (p9x - p6x) / 3.f;
    float e3_1y = p6y + (p9y - p6y) / 3.f;
    float e3_2x = p6x + 2.f * (p9x - p6x) / 3.f;
    float e3_2y = p6y + 2.f * (p9y - p6y) / 3.f;

    /* Edge 4: p9→p0, inner arc. */
    float e4_1x = p9x + kappa * eps * (-sin1);
    float e4_1y = p9y + kappa * eps * ( cos1);
    float e4_2x = p0x - kappa * eps * (-sin0);
    float e4_2y = p0y - kappa * eps * ( cos0);

    mesh->append_c ('\0'); /* flag = 0, new patch */

    hb_pdf_encode_point (mesh, p0x, p0y, xlo, xhi, ylo, yhi);
    hb_pdf_encode_point (mesh, e1_1x, e1_1y, xlo, xhi, ylo, yhi);
    hb_pdf_encode_point (mesh, e1_2x, e1_2y, xlo, xhi, ylo, yhi);
    hb_pdf_encode_point (mesh, p3x, p3y, xlo, xhi, ylo, yhi);
    hb_pdf_encode_point (mesh, e2_1x, e2_1y, xlo, xhi, ylo, yhi);
    hb_pdf_encode_point (mesh, e2_2x, e2_2y, xlo, xhi, ylo, yhi);
    hb_pdf_encode_point (mesh, p6x, p6y, xlo, xhi, ylo, yhi);
    hb_pdf_encode_point (mesh, e3_1x, e3_1y, xlo, xhi, ylo, yhi);
    hb_pdf_encode_point (mesh, e3_2x, e3_2y, xlo, xhi, ylo, yhi);
    hb_pdf_encode_point (mesh, p9x, p9y, xlo, xhi, ylo, yhi);
    hb_pdf_encode_point (mesh, e4_1x, e4_1y, xlo, xhi, ylo, yhi);
    hb_pdf_encode_point (mesh, e4_2x, e4_2y, xlo, xhi, ylo, yhi);

    hb_pdf_encode_color_rgb (mesh, sc0); /* inner start */
    hb_pdf_encode_color_rgb (mesh, sc0); /* outer start */
    hb_pdf_encode_color_rgb (mesh, sc1); /* outer end */
    hb_pdf_encode_color_rgb (mesh, sc1); /* inner end */

    if (alpha_mesh)
    {
      alpha_mesh->append_c ('\0');

      hb_pdf_encode_point (alpha_mesh, p0x, p0y, xlo, xhi, ylo, yhi);
      hb_pdf_encode_point (alpha_mesh, e1_1x, e1_1y, xlo, xhi, ylo, yhi);
      hb_pdf_encode_point (alpha_mesh, e1_2x, e1_2y, xlo, xhi, ylo, yhi);
      hb_pdf_encode_point (alpha_mesh, p3x, p3y, xlo, xhi, ylo, yhi);
      hb_pdf_encode_point (alpha_mesh, e2_1x, e2_1y, xlo, xhi, ylo, yhi);
      hb_pdf_encode_point (alpha_mesh, e2_2x, e2_2y, xlo, xhi, ylo, yhi);
      hb_pdf_encode_point (alpha_mesh, p6x, p6y, xlo, xhi, ylo, yhi);
      hb_pdf_encode_point (alpha_mesh, e3_1x, e3_1y, xlo, xhi, ylo, yhi);
      hb_pdf_encode_point (alpha_mesh, e3_2x, e3_2y, xlo, xhi, ylo, yhi);
      hb_pdf_encode_point (alpha_mesh, p9x, p9y, xlo, xhi, ylo, yhi);
      hb_pdf_encode_point (alpha_mesh, e4_1x, e4_1y, xlo, xhi, ylo, yhi);
      hb_pdf_encode_point (alpha_mesh, e4_2x, e4_2y, xlo, xhi, ylo, yhi);

      hb_pdf_encode_color_alpha (alpha_mesh, sc0);
      hb_pdf_encode_color_alpha (alpha_mesh, sc0);
      hb_pdf_encode_color_alpha (alpha_mesh, sc1);
      hb_pdf_encode_color_alpha (alpha_mesh, sc1);
    }
  }
}

/* Callback context + trampoline for hb_paint_sweep_gradient_tiles. */
struct hb_pdf_sweep_ctx_t {
  hb_vector_buf_t *mesh;
  hb_vector_buf_t *alpha_mesh;
  float cx, cy, xlo, xhi, ylo, yhi;
};

static void
hb_pdf_sweep_emit_patch (float a0, hb_color_t c0,
			 float a1, hb_color_t c1,
			 void *user_data)
{
  auto *ctx = (hb_pdf_sweep_ctx_t *) user_data;
  hb_pdf_add_sweep_patch (ctx->mesh, ctx->alpha_mesh,
			  ctx->cx, ctx->cy,
			  ctx->xlo, ctx->xhi, ctx->ylo, ctx->yhi,
			  a0, c0, a1, c1);
}

static void
hb_pdf_paint_sweep_gradient (hb_paint_funcs_t *,
			     void *paint_data,
			     hb_color_line_t *color_line,
			     float cx, float cy,
			     float start_angle,
			     float end_angle,
			     void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!paint->ensure_initialized ()))
    return;
  auto *res = hb_pdf_get_resources (paint);
  if (unlikely (!res))
    return;

  /* Get and sort color stops. */
  if (!paint->fetch_color_stops (color_line))
    return;
  hb_vector_t<hb_color_stop_t> &stops = paint->color_stops_scratch;
  stops.as_array ().qsort (
    [] (const hb_color_stop_t &a, const hb_color_stop_t &b)
    { return a.offset < b.offset; });

  hb_paint_extend_t extend = hb_color_line_get_extend (color_line);

  const float R = 32767.f;
  float scx = paint->sx (cx), scy = paint->sy (cy);
  float xlo = scx - R - 1, xhi = scx + R + 1;
  float ylo = scy - R - 1, yhi = scy + R + 1;

  bool needs_alpha = hb_pdf_gradient_needs_alpha (stops);

  hb_vector_buf_t mesh;
  hb_vector_buf_t alpha_mesh;
  mesh.alloc (256);
  if (needs_alpha)
    alpha_mesh.alloc (256);

  hb_pdf_sweep_ctx_t ctx { &mesh, needs_alpha ? &alpha_mesh : nullptr,
			    scx, scy, xlo, xhi, ylo, yhi };
  hb_paint_sweep_gradient_tiles (stops.arrayZ, stops.length, extend,
				 start_angle, end_angle,
				 hb_pdf_sweep_emit_patch, &ctx);

  if (!mesh.length)
    return;

  auto hb_pdf_build_mesh_shading = [&] (hb_vector_buf_t &m,
					 const char *cs,
					 const char *decode_suffix) -> unsigned
  {
    hb_vector_buf_t sh;
    sh.append_str ("<< /ShadingType 6 /ColorSpace /");
    sh.append_str (cs);
    sh.append_str ("\n/BitsPerCoordinate 16 /BitsPerComponent 8 /BitsPerFlag 8\n");
    sh.append_str ("/Decode [");
    sh.append_num (xlo, 2);
    sh.append_c (' ');
    sh.append_num (xhi, 2);
    sh.append_c (' ');
    sh.append_num (ylo, 2);
    sh.append_c (' ');
    sh.append_num (yhi, 2);
    sh.append_str (decode_suffix);
    sh.append_str ("]\n/Length ");
    sh.append_unsigned (m.length);
    sh.append_str (" >>\nstream\n");
    sh.append_len (m.arrayZ, m.length);
    sh.append_str ("\nendstream");
    return res->add_object (std::move (sh));
  };

  unsigned sh_obj_id = hb_pdf_build_mesh_shading (mesh, "DeviceRGB",
						   " 0 1 0 1 0 1");
  unsigned sh_idx = res->add_shading_by_id (sh_obj_id);

  auto &body = paint->current_body ();

  if (needs_alpha && alpha_mesh.length)
  {
    unsigned alpha_sh_id = hb_pdf_build_mesh_shading (alpha_mesh, "DeviceGray",
						       " 0 1");
    unsigned gs_idx = res->add_extgstate_smask (alpha_sh_id,
						xlo, ylo,
						xhi - xlo, yhi - ylo,
						paint->get_precision ());
    body.append_str ("/GS");
    body.append_unsigned (gs_idx);
    body.append_str (" gs\n");
  }

  body.append_str ("/SH");
  body.append_unsigned (sh_idx);
  body.append_str (" sh\n");
}

static const char *
hb_pdf_blend_mode_name (hb_paint_composite_mode_t mode)
{
  switch (mode)
  {
  case HB_PAINT_COMPOSITE_MODE_MULTIPLY:       return "Multiply";
  case HB_PAINT_COMPOSITE_MODE_SCREEN:         return "Screen";
  case HB_PAINT_COMPOSITE_MODE_OVERLAY:        return "Overlay";
  case HB_PAINT_COMPOSITE_MODE_DARKEN:         return "Darken";
  case HB_PAINT_COMPOSITE_MODE_LIGHTEN:        return "Lighten";
  case HB_PAINT_COMPOSITE_MODE_COLOR_DODGE:    return "ColorDodge";
  case HB_PAINT_COMPOSITE_MODE_COLOR_BURN:     return "ColorBurn";
  case HB_PAINT_COMPOSITE_MODE_HARD_LIGHT:     return "HardLight";
  case HB_PAINT_COMPOSITE_MODE_SOFT_LIGHT:     return "SoftLight";
  case HB_PAINT_COMPOSITE_MODE_DIFFERENCE:     return "Difference";
  case HB_PAINT_COMPOSITE_MODE_EXCLUSION:      return "Exclusion";
  case HB_PAINT_COMPOSITE_MODE_HSL_HUE:        return "Hue";
  case HB_PAINT_COMPOSITE_MODE_HSL_SATURATION: return "Saturation";
  case HB_PAINT_COMPOSITE_MODE_HSL_COLOR:      return "Color";
  case HB_PAINT_COMPOSITE_MODE_HSL_LUMINOSITY: return "Luminosity";
  /* Porter-Duff modes have no PDF blend-mode equivalent; approximate
   * the two that have a plausible color-blend analog, and let the
   * rest fall through to Normal (SRC_OVER). */
  case HB_PAINT_COMPOSITE_MODE_PLUS:           return "Screen";
  case HB_PAINT_COMPOSITE_MODE_XOR:            return "Difference";
  case HB_PAINT_COMPOSITE_MODE_CLEAR:
  case HB_PAINT_COMPOSITE_MODE_SRC:
  case HB_PAINT_COMPOSITE_MODE_DEST:
  case HB_PAINT_COMPOSITE_MODE_SRC_OVER:
  case HB_PAINT_COMPOSITE_MODE_DEST_OVER:
  case HB_PAINT_COMPOSITE_MODE_SRC_IN:
  case HB_PAINT_COMPOSITE_MODE_DEST_IN:
  case HB_PAINT_COMPOSITE_MODE_SRC_OUT:
  case HB_PAINT_COMPOSITE_MODE_DEST_OUT:
  case HB_PAINT_COMPOSITE_MODE_SRC_ATOP:
  case HB_PAINT_COMPOSITE_MODE_DEST_ATOP:
  default:                                     return nullptr; /* Normal */
  }
}

static void
hb_pdf_paint_push_group (hb_paint_funcs_t *,
			 void *paint_data,
			 void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!paint->ensure_initialized ()))
    return;
  paint->current_body ().append_str ("q\n");
}

static void
hb_pdf_paint_push_group_for (hb_paint_funcs_t *,
			     void *paint_data,
			     hb_paint_composite_mode_t mode,
			     void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!paint->ensure_initialized ()))
    return;

  auto &body = paint->current_body ();
  body.append_str ("q\n");

  const char *bm = hb_pdf_blend_mode_name (mode);
  if (bm)
  {
    auto *res = hb_pdf_get_resources (paint);
    if (likely (res))
    {
      unsigned gs_idx = res->add_extgstate_blend (bm);
      body.append_str ("/GS");
      body.append_unsigned (gs_idx);
      body.append_str (" gs\n");
    }
  }
}

static void
hb_pdf_paint_pop_group (hb_paint_funcs_t *,
			void *paint_data,
			hb_paint_composite_mode_t mode HB_UNUSED,
			void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!paint->ensure_initialized ()))
    return;
  paint->current_body ().append_str ("Q\n");
}


/* ---- lazy loader for paint funcs ---- */

static inline void free_static_pdf_paint_funcs ();

static struct hb_pdf_paint_funcs_lazy_loader_t
  : hb_paint_funcs_lazy_loader_t<hb_pdf_paint_funcs_lazy_loader_t>
{
  static hb_paint_funcs_t *create ()
  {
    hb_paint_funcs_t *funcs = hb_paint_funcs_create ();
    hb_paint_funcs_set_push_transform_func (funcs, (hb_paint_push_transform_func_t) hb_pdf_paint_push_transform, nullptr, nullptr);
    hb_paint_funcs_set_pop_transform_func (funcs, (hb_paint_pop_transform_func_t) hb_pdf_paint_pop_transform, nullptr, nullptr);
    hb_paint_funcs_set_push_clip_glyph_func (funcs, (hb_paint_push_clip_glyph_func_t) hb_pdf_paint_push_clip_glyph, nullptr, nullptr);
    hb_paint_funcs_set_push_clip_rectangle_func (funcs, (hb_paint_push_clip_rectangle_func_t) hb_pdf_paint_push_clip_rectangle, nullptr, nullptr);
    hb_paint_funcs_set_push_clip_path_start_func (funcs, (hb_paint_push_clip_path_start_func_t) hb_pdf_paint_push_clip_path_start, nullptr, nullptr);
    hb_paint_funcs_set_push_clip_path_end_func (funcs, (hb_paint_push_clip_path_end_func_t) hb_pdf_paint_push_clip_path_end, nullptr, nullptr);
    hb_paint_funcs_set_pop_clip_func (funcs, (hb_paint_pop_clip_func_t) hb_pdf_paint_pop_clip, nullptr, nullptr);
    hb_paint_funcs_set_color_func (funcs, (hb_paint_color_func_t) hb_pdf_paint_color, nullptr, nullptr);
    hb_paint_funcs_set_image_func (funcs, (hb_paint_image_func_t) hb_pdf_paint_image, nullptr, nullptr);
    hb_paint_funcs_set_linear_gradient_func (funcs, (hb_paint_linear_gradient_func_t) hb_pdf_paint_linear_gradient, nullptr, nullptr);
    hb_paint_funcs_set_radial_gradient_func (funcs, (hb_paint_radial_gradient_func_t) hb_pdf_paint_radial_gradient, nullptr, nullptr);
    hb_paint_funcs_set_sweep_gradient_func (funcs, (hb_paint_sweep_gradient_func_t) hb_pdf_paint_sweep_gradient, nullptr, nullptr);
    hb_paint_funcs_set_push_group_func (funcs, (hb_paint_push_group_func_t) hb_pdf_paint_push_group, nullptr, nullptr);
    hb_paint_funcs_set_push_group_for_func (funcs, (hb_paint_push_group_for_func_t) hb_pdf_paint_push_group_for, nullptr, nullptr);
    hb_paint_funcs_set_pop_group_func (funcs, (hb_paint_pop_group_func_t) hb_pdf_paint_pop_group, nullptr, nullptr);
    hb_paint_funcs_make_immutable (funcs);
    hb_atexit (free_static_pdf_paint_funcs);
    return funcs;
  }
} static_pdf_paint_funcs;

static inline void
free_static_pdf_paint_funcs ()
{
  static_pdf_paint_funcs.free_instance ();
}

hb_paint_funcs_t *
hb_vector_paint_pdf_funcs_get ()
{
  return static_pdf_paint_funcs.get_unconst ();
}


/* ---- render ---- */

hb_blob_t *
hb_vector_paint_render_pdf (hb_vector_paint_t *paint)
{
  if (!paint->has_extents)
    return nullptr;
  if (!paint->group_stack.length ||
      !paint->group_stack.arrayZ[0].length)
    return nullptr;

  hb_vector_buf_t &content = paint->group_stack.arrayZ[0];
  hb_pdf_resources_t *res = hb_pdf_get_resources (paint);

  float ex = paint->extents.x;
  float ey = paint->extents.y;
  float ew = paint->extents.width;
  float eh = paint->extents.height;

  unsigned num_extra = res ? res->objects.length : 0;
  unsigned total_objects = 4 + num_extra; /* 1-based: 1..total_objects */

  /* Build PDF. */
  hb_vector_buf_t out;
  hb_buf_recover_recycled (paint->recycled_blob, &out);
  out.alloc (content.length + num_extra * 128 + 1024);

  hb_vector_t<unsigned> offsets;
  if (unlikely (!offsets.resize (total_objects)))
    return nullptr;

  out.append_str ("%PDF-1.4\n%\xC0\xC1\xC2\xC3\n");

  /* Object 1: Catalog */
  offsets.arrayZ[0] = out.length;
  out.append_str ("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

  /* Object 2: Pages */
  offsets.arrayZ[1] = out.length;
  out.append_str ("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

  /* Object 3: Page */
  offsets.arrayZ[2] = out.length;
  out.append_str ("3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [");
  out.append_num (ex);
  out.append_c (' ');
  out.append_num (-(ey + eh));
  out.append_c (' ');
  out.append_num (ex + ew);
  out.append_c (' ');
  out.append_num (-ey);
  out.append_str ("]\n/Contents 4 0 R");

  /* Resources. */
  bool has_resources = res &&
    (res->extgstate_dict.length || res->shading_dict.length || res->xobject_dict.length);
  if (has_resources)
  {
    out.append_str ("\n/Resources <<");
    if (res->extgstate_dict.length)
    {
      out.append_str (" /ExtGState << ");
      out.append_len (res->extgstate_dict.arrayZ, res->extgstate_dict.length);
      out.append_str (">>");
    }
    if (res->shading_dict.length)
    {
      out.append_str (" /Shading << ");
      out.append_len (res->shading_dict.arrayZ, res->shading_dict.length);
      out.append_str (">>");
    }
    if (res->xobject_dict.length)
    {
      out.append_str (" /XObject << ");
      out.append_len (res->xobject_dict.arrayZ, res->xobject_dict.length);
      out.append_str (">>");
    }
    out.append_str (" >>");
  }

  out.append_str (" >>\nendobj\n");

  /* Build content stream: optional background rect + glyph content. */
  hb_vector_buf_t bg_prefix;
  if (hb_color_get_alpha (paint->background))
  {
    float r = hb_color_get_red (paint->background) / 255.f;
    float g = hb_color_get_green (paint->background) / 255.f;
    float b = hb_color_get_blue (paint->background) / 255.f;
    float a = hb_color_get_alpha (paint->background) / 255.f;
    if (a < 1.f - 1.f / 512.f)
    {
      if (res)
      {
	unsigned gs_idx = res->add_extgstate_alpha (a);
	bg_prefix.append_str ("/GS");
	bg_prefix.append_unsigned (gs_idx);
	bg_prefix.append_str (" gs\n");
      }
    }
    bg_prefix.append_num (r, 4);
    bg_prefix.append_c (' ');
    bg_prefix.append_num (g, 4);
    bg_prefix.append_c (' ');
    bg_prefix.append_num (b, 4);
    bg_prefix.append_str (" rg\n");
    bg_prefix.append_num (ex);
    bg_prefix.append_c (' ');
    bg_prefix.append_num (-(ey + eh));
    bg_prefix.append_c (' ');
    bg_prefix.append_num (ew);
    bg_prefix.append_c (' ');
    bg_prefix.append_num (eh);
    bg_prefix.append_str (" re f\n");
  }
  unsigned stream_len = bg_prefix.length + content.length;

  /* Object 4: Content stream */
  offsets.arrayZ[3] = out.length;
  out.append_str ("4 0 obj\n<< /Length ");
  out.append_unsigned (stream_len);
  out.append_str (" >>\nstream\n");
  out.append_len (bg_prefix.arrayZ, bg_prefix.length);
  out.append_len (content.arrayZ, content.length);
  out.append_str ("endstream\nendobj\n");

  /* Extra objects (functions, shadings, ExtGState). */
  for (unsigned i = 0; i < num_extra; i++)
  {
    offsets.arrayZ[4 + i] = out.length;
    out.append_unsigned (5 + i);
    out.append_str (" 0 obj\n");
    auto &obj = res->objects.arrayZ[i];
    out.append_len (obj.data.arrayZ, obj.data.length);
    out.append_str ("\nendobj\n");
  }

  /* Cross-reference table */
  unsigned xref_offset = out.length;
  out.append_str ("xref\n0 ");
  out.append_unsigned (total_objects + 1);
  out.append_str ("\n0000000000 65535 f \n");
  for (unsigned i = 0; i < total_objects; i++)
  {
    char tmp[21];
    snprintf (tmp, sizeof (tmp), "%010u 00000 n \n", offsets.arrayZ[i]);
    out.append_len (tmp, 20);
  }

  /* Trailer */
  out.append_str ("trailer\n<< /Size ");
  out.append_unsigned (total_objects + 1);
  out.append_str (" /Root 1 0 R >>\nstartxref\n");
  out.append_unsigned (xref_offset);
  out.append_str ("\n%%EOF\n");

  hb_blob_t *blob = hb_buf_blob_from (&paint->recycled_blob, &out);

  hb_vector_paint_clear (paint);

  return blob;
}

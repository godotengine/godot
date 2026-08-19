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

#ifndef HB_VECTOR_INTERNAL_HH
#define HB_VECTOR_INTERNAL_HH

#include "hb-vector.h"

struct hb_vector_blob_meta_t
{
  char *data;
  int allocated;
  bool transferred;
  bool in_replace;
};

static hb_user_data_key_t hb_vector_blob_meta_user_data_key;

static inline void
hb_vector_blob_meta_set_buffer (hb_vector_blob_meta_t *meta,
				char *data,
				int allocated)
{
  meta->data = data;
  meta->allocated = allocated;
  meta->transferred = false;
}

static inline void
hb_vector_blob_meta_release_buffer (hb_vector_blob_meta_t *meta)
{
  if (!meta)
    return;
  if (!meta->transferred && meta->data)
    hb_free (meta->data);
  meta->data = nullptr;
  meta->allocated = 0;
  meta->transferred = true;
}

static inline void
hb_vector_blob_meta_destroy (void *data)
{
  auto *meta = (hb_vector_blob_meta_t *) data;
  hb_vector_blob_meta_release_buffer (meta);
  if (meta->in_replace)
  {
    meta->in_replace = false;
    return;
  }
  hb_free (meta);
}

static inline hb_blob_t *
hb_buf_blob_from (hb_blob_t **recycled_blob,
		  hb_vector_buf_t *buf)
{
  unsigned len = 0;
  int allocated = 0;
  char *data = buf->steal (&len, &allocated);
  if (!data)
    return nullptr;

  hb_blob_t *blob = nullptr;
  if (*recycled_blob)
    blob = *recycled_blob;
  bool reused_blob = blob && blob != hb_blob_get_empty ();
  bool new_meta = false;
  auto *meta = reused_blob
             ? (hb_vector_blob_meta_t *) hb_blob_get_user_data (blob, &hb_vector_blob_meta_user_data_key)
             : nullptr;
  if (!meta)
  {
    meta = (hb_vector_blob_meta_t *) hb_malloc (sizeof (hb_vector_blob_meta_t));
    if (!meta)
    {
      hb_free (data);
      return nullptr;
    }
    meta->data = nullptr;
    meta->allocated = 0;
    meta->transferred = true;
    meta->in_replace = false;
    new_meta = true;
  }

  if (reused_blob)
  {
    meta->in_replace = true;
    blob->replace_buffer (data, len, HB_MEMORY_MODE_WRITABLE, meta, hb_vector_blob_meta_destroy);
    hb_vector_blob_meta_set_buffer (meta, data, allocated);
  }
  else
  {
    hb_vector_blob_meta_set_buffer (meta, data, allocated);
    blob = hb_blob_create_or_fail (data, len, HB_MEMORY_MODE_WRITABLE, meta, hb_vector_blob_meta_destroy);
    if (unlikely (!blob))
      return nullptr;
  }

  if (unlikely (blob == hb_blob_get_empty ()))
  {
    if (new_meta)
      hb_free (meta);
    hb_free (data);
    return nullptr;
  }

  if (new_meta &&
      !hb_blob_set_user_data (blob,
                              &hb_vector_blob_meta_user_data_key,
                              meta,
                              nullptr,
                              true))
  {
    if (!reused_blob)
      hb_blob_destroy (blob);
    return nullptr;
  }

  if (*recycled_blob)
    *recycled_blob = nullptr;

  return blob;
}

static inline void
hb_buf_recover_recycled (hb_blob_t *blob,
			 hb_vector_buf_t *buf)
{
  if (!blob)
    return;

  auto *meta = (hb_vector_blob_meta_t *) hb_blob_get_user_data (blob, &hb_vector_blob_meta_user_data_key);
  if (!meta || meta->transferred || !meta->data)
    return;

  buf->recycle_buffer (meta->data, 0, meta->allocated);
  meta->data = nullptr;
  meta->allocated = 0;
  meta->transferred = true;
}

static inline void
hb_vector_transform_point (const hb_transform_t<> &t,
			   float x_scale_factor,
			   float y_scale_factor,
			   float x, float y,
			   float *tx, float *ty)
{
  float xx = x, yy = y;
  t.transform_point (xx, yy);
  /* Setters force scale factors > 0; trust them here. */
  *tx = xx / x_scale_factor;
  *ty = yy / y_scale_factor;
}

static inline hb_bool_t
hb_vector_set_glyph_extents_common (const hb_transform_t<> &transform,
				    float x_scale_factor,
				    float y_scale_factor,
				    const hb_glyph_extents_t *glyph_extents,
				    hb_vector_extents_t *extents,
				    hb_bool_t *has_extents)
{
  float x0 = (float) glyph_extents->x_bearing;
  float y0 = (float) glyph_extents->y_bearing;
  float x1 = x0 + glyph_extents->width;
  float y1 = y0 + glyph_extents->height;

  float px[4] = {x0, x0, x1, x1};
  float py[4] = {y0, y1, y0, y1};

  float tx, ty;
  hb_vector_transform_point (transform, x_scale_factor, y_scale_factor, px[0], py[0], &tx, &ty);
  float tx_min = tx, tx_max = tx;
  float ty_min = ty, ty_max = ty;

  for (unsigned i = 1; i < 4; i++)
  {
    hb_vector_transform_point (transform, x_scale_factor, y_scale_factor, px[i], py[i], &tx, &ty);
    tx_min = hb_min (tx_min, tx);
    tx_max = hb_max (tx_max, tx);
    ty_min = hb_min (ty_min, ty);
    ty_max = hb_max (ty_max, ty);
  }

  if (tx_max <= tx_min || ty_max <= ty_min)
    return false;

  if (*has_extents)
  {
    float x0 = hb_min (extents->x, tx_min);
    float y0 = hb_min (extents->y, ty_min);
    float x1 = hb_max (extents->x + extents->width, tx_max);
    float y1 = hb_max (extents->y + extents->height, ty_max);
    *extents = {x0, y0, x1 - x0, y1 - y0};
  }
  else
  {
    *extents = {tx_min, ty_min, tx_max - tx_min, ty_max - ty_min};
    *has_extents = true;
  }
  return true;
}

static inline void
hb_vector_svg_append_instance_transform (hb_vector_buf_t *out,
					 unsigned precision,
					 float x_scale_factor,
					 float y_scale_factor,
					 float xx, float yx,
					 float xy, float yy,
					 float tx, float ty)
{
  if (xx == 1.f && yx == 0.f && xy == 0.f && yy == 1.f)
  {
    out->append_str ("translate(");
    out->append_num (tx / x_scale_factor, precision);
    out->append_c (',');
    out->append_num (-ty / y_scale_factor, precision);
    out->append_str (") scale(1,-1)");
  }
  else
  {
    unsigned sprec = out->scale_precision ();
    out->append_str ("matrix(");
    out->append_num (xx, sprec);
    out->append_c (',');
    out->append_num (yx, sprec);
    out->append_c (',');
    out->append_num (-xy, sprec);
    out->append_c (',');
    out->append_num (-yy, sprec);
    out->append_c (',');
    out->append_num (tx / x_scale_factor, precision);
    out->append_c (',');
    out->append_num (-ty / y_scale_factor, precision);
    out->append_c (')');
  }
}

#endif /* HB_VECTOR_INTERNAL_HH */

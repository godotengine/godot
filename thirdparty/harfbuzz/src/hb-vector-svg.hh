#ifndef HB_VECTOR_SVG_HH
#define HB_VECTOR_SVG_HH

#include "hb-vector.h"

static inline bool
hb_svg_append_str (hb_vector_t<char> *buf, const char *s)
{
  return hb_svg_append_len (buf, s, (unsigned) strlen (s));
}

static inline bool
hb_svg_append_unsigned (hb_vector_t<char> *buf, unsigned v)
{
  char tmp[10];
  unsigned n = 0;
  do {
    tmp[n++] = (char) ('0' + (v % 10));
    v /= 10;
  } while (v);

  unsigned old_len = buf->length;
  if (unlikely (!buf->resize_dirty ((int) (old_len + n))))
    return false;

  for (unsigned i = 0; i < n; i++)
    buf->arrayZ[old_len + i] = tmp[n - 1 - i];
  return true;
}

static inline bool
hb_svg_append_hex_byte (hb_vector_t<char> *buf, unsigned v)
{
  static const char hex[] = "0123456789ABCDEF";
  char tmp[2] = {hex[(v >> 4) & 15], hex[v & 15]};
  return hb_svg_append_len (buf, tmp, 2);
}

static inline bool
hb_svg_append_base64 (hb_vector_t<char> *buf,
                      const uint8_t *data,
                      unsigned len)
{
  static const char b64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  unsigned out_len = ((len + 2) / 3) * 4;
  unsigned old_len = buf->length;
  if (unlikely (!buf->resize_dirty ((int) (old_len + out_len))))
    return false;

  char *dst = buf->arrayZ + old_len;
  unsigned di = 0;
  unsigned i = 0;
  while (i + 2 < len)
  {
    unsigned v = ((unsigned) data[i] << 16) |
                 ((unsigned) data[i + 1] << 8) |
                 ((unsigned) data[i + 2]);
    dst[di++] = b64[(v >> 18) & 63];
    dst[di++] = b64[(v >> 12) & 63];
    dst[di++] = b64[(v >> 6) & 63];
    dst[di++] = b64[v & 63];
    i += 3;
  }

  if (i < len)
  {
    unsigned v = (unsigned) data[i] << 16;
    if (i + 1 < len)
      v |= (unsigned) data[i + 1] << 8;
    dst[di++] = b64[(v >> 18) & 63];
    dst[di++] = b64[(v >> 12) & 63];
    dst[di++] = (i + 1 < len) ? b64[(v >> 6) & 63] : '=';
    dst[di++] = '=';
  }

  return true;
}

struct hb_svg_blob_meta_t
{
  char *data;
  int allocated;
  bool transferred;
  bool in_replace;
};

static hb_user_data_key_t hb_svg_blob_meta_user_data_key;

static inline void
hb_svg_blob_meta_set_buffer (hb_svg_blob_meta_t *meta,
                             char *data,
                             int allocated)
{
  meta->data = data;
  meta->allocated = allocated;
  meta->transferred = false;
}

static inline void
hb_svg_blob_meta_release_buffer (hb_svg_blob_meta_t *meta)
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
hb_svg_blob_meta_destroy (void *data)
{
  auto *meta = (hb_svg_blob_meta_t *) data;
  hb_svg_blob_meta_release_buffer (meta);
  if (meta->in_replace)
  {
    meta->in_replace = false;
    return;
  }
  hb_free (meta);
}

static inline hb_blob_t *
hb_svg_blob_from_buffer (hb_blob_t **recycled_blob,
                         hb_vector_t<char> *buf)
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
             ? (hb_svg_blob_meta_t *) hb_blob_get_user_data (blob, &hb_svg_blob_meta_user_data_key)
             : nullptr;
  if (!meta)
  {
    meta = (hb_svg_blob_meta_t *) hb_malloc (sizeof (hb_svg_blob_meta_t));
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
    blob->replace_buffer (data, len, HB_MEMORY_MODE_WRITABLE, meta, hb_svg_blob_meta_destroy);
    hb_svg_blob_meta_set_buffer (meta, data, allocated);
  }
  else
  {
    hb_svg_blob_meta_set_buffer (meta, data, allocated);
    blob = hb_blob_create_or_fail (data, len, HB_MEMORY_MODE_WRITABLE, meta, hb_svg_blob_meta_destroy);
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
                              &hb_svg_blob_meta_user_data_key,
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
hb_svg_recover_recycled_buffer (hb_blob_t *blob,
                                hb_vector_t<char> *buf)
{
  if (!blob)
    return;

  auto *meta = (hb_svg_blob_meta_t *) hb_blob_get_user_data (blob, &hb_svg_blob_meta_user_data_key);
  if (!meta || meta->transferred || !meta->data)
    return;

  buf->recycle_buffer (meta->data, 0, meta->allocated);
  meta->data = nullptr;
  meta->allocated = 0;
  meta->transferred = true;
}

static inline void
hb_svg_append_color (hb_vector_t<char> *buf,
                     hb_color_t color,
                     bool with_alpha)
{
  static const char hex[] = "0123456789ABCDEF";
  unsigned r = hb_color_get_red (color);
  unsigned g = hb_color_get_green (color);
  unsigned b = hb_color_get_blue (color);
  unsigned a = hb_color_get_alpha (color);
  hb_svg_append_c (buf, '#');
  if (((r >> 4) == (r & 0xF)) &&
      ((g >> 4) == (g & 0xF)) &&
      ((b >> 4) == (b & 0xF)))
  {
    hb_svg_append_c (buf, hex[r & 0xF]);
    hb_svg_append_c (buf, hex[g & 0xF]);
    hb_svg_append_c (buf, hex[b & 0xF]);
  }
  else
  {
    hb_svg_append_hex_byte (buf, r);
    hb_svg_append_hex_byte (buf, g);
    hb_svg_append_hex_byte (buf, b);
  }
  if (with_alpha && a != 255)
  {
    hb_svg_append_str (buf, "\" fill-opacity=\"");
    hb_svg_append_num (buf, a / 255.f, 4);
  }
}

static inline void
hb_svg_transform_point (const hb_transform_t<> &t,
                        float x_scale_factor,
                        float y_scale_factor,
                        float x, float y,
                        float *tx, float *ty)
{
  float xx = x, yy = y;
  t.transform_point (xx, yy);
  *tx = xx / (x_scale_factor > 0 ? x_scale_factor : 1.f);
  *ty = yy / (y_scale_factor > 0 ? y_scale_factor : 1.f);
}

static inline hb_bool_t
hb_svg_set_glyph_extents_common (const hb_transform_t<> &transform,
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
  hb_svg_transform_point (transform, x_scale_factor, y_scale_factor, px[0], py[0], &tx, &ty);
  float tx_min = tx, tx_max = tx;
  float ty_min = ty, ty_max = ty;

  for (unsigned i = 1; i < 4; i++)
  {
    hb_svg_transform_point (transform, x_scale_factor, y_scale_factor, px[i], py[i], &tx, &ty);
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
hb_svg_append_instance_transform (hb_vector_t<char> *out,
                                  unsigned precision,
                                  float x_scale_factor,
                                  float y_scale_factor,
                                  float xx, float yx,
                                  float xy, float yy,
                                  float tx, float ty)
{
  unsigned sprec = hb_svg_scale_precision (precision);
  if (xx == 1.f && yx == 0.f && xy == 0.f && yy == 1.f)
  {
    float sx = 1.f / x_scale_factor;
    float sy = 1.f / y_scale_factor;
    hb_svg_append_str (out, "translate(");
    hb_svg_append_num (out, tx / x_scale_factor, precision);
    hb_svg_append_c (out, ',');
    hb_svg_append_num (out, -ty / y_scale_factor, precision);
    hb_svg_append_str (out, ") scale(");
    hb_svg_append_num (out, sx, sprec, true);
    hb_svg_append_c (out, ',');
    hb_svg_append_num (out, -sy, sprec, true);
    hb_svg_append_c (out, ')');
  }
  else
  {
    hb_svg_append_str (out, "matrix(");
    hb_svg_append_num (out, xx / x_scale_factor, sprec, true);
    hb_svg_append_c (out, ',');
    hb_svg_append_num (out, yx / y_scale_factor, sprec, true);
    hb_svg_append_c (out, ',');
    hb_svg_append_num (out, -xy / x_scale_factor, sprec, true);
    hb_svg_append_c (out, ',');
    hb_svg_append_num (out, -yy / y_scale_factor, sprec, true);
    hb_svg_append_c (out, ',');
    hb_svg_append_num (out, tx / x_scale_factor, precision);
    hb_svg_append_c (out, ',');
    hb_svg_append_num (out, -ty / y_scale_factor, precision);
    hb_svg_append_c (out, ')');
  }
}

static inline void
hb_svg_append_image_instance_translate (hb_vector_t<char> *out,
                                        unsigned precision,
                                        float x_scale_factor,
                                        float y_scale_factor,
                                        float tx, float ty)
{
  hb_svg_append_str (out, "translate(");
  hb_svg_append_num (out, tx / x_scale_factor, precision);
  hb_svg_append_c (out, ',');
  hb_svg_append_num (out, -ty / y_scale_factor, precision);
  hb_svg_append_c (out, ')');
}

#endif /* HB_VECTOR_SVG_HH */

// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// functions for sample output.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdlib.h>
#include "../dec/vp8i.h"
#include "./webpi.h"
#include "../dsp/dsp.h"
#include "../dsp/yuv.h"
#include "../utils/utils.h"

//------------------------------------------------------------------------------
// Main YUV<->RGB conversion functions

static int EmitYUV(const VP8Io* const io, WebPDecParams* const p) {
  WebPDecBuffer* output = p->output;
  const WebPYUVABuffer* const buf = &output->u.YUVA;
  uint8_t* const y_dst = buf->y + io->mb_y * buf->y_stride;
  uint8_t* const u_dst = buf->u + (io->mb_y >> 1) * buf->u_stride;
  uint8_t* const v_dst = buf->v + (io->mb_y >> 1) * buf->v_stride;
  const int mb_w = io->mb_w;
  const int mb_h = io->mb_h;
  const int uv_w = (mb_w + 1) / 2;
  const int uv_h = (mb_h + 1) / 2;
  int j;
  for (j = 0; j < mb_h; ++j) {
    memcpy(y_dst + j * buf->y_stride, io->y + j * io->y_stride, mb_w);
  }
  for (j = 0; j < uv_h; ++j) {
    memcpy(u_dst + j * buf->u_stride, io->u + j * io->uv_stride, uv_w);
    memcpy(v_dst + j * buf->v_stride, io->v + j * io->uv_stride, uv_w);
  }
  return io->mb_h;
}

// Point-sampling U/V sampler.
static int EmitSampledRGB(const VP8Io* const io, WebPDecParams* const p) {
  WebPDecBuffer* const output = p->output;
  WebPRGBABuffer* const buf = &output->u.RGBA;
  uint8_t* const dst = buf->rgba + io->mb_y * buf->stride;
  WebPSamplerProcessPlane(io->y, io->y_stride,
                          io->u, io->v, io->uv_stride,
                          dst, buf->stride, io->mb_w, io->mb_h,
                          WebPSamplers[output->colorspace]);
  return io->mb_h;
}

//------------------------------------------------------------------------------
// Fancy upsampling

#ifdef FANCY_UPSAMPLING
static int EmitFancyRGB(const VP8Io* const io, WebPDecParams* const p) {
  int num_lines_out = io->mb_h;   // a priori guess
  const WebPRGBABuffer* const buf = &p->output->u.RGBA;
  uint8_t* dst = buf->rgba + io->mb_y * buf->stride;
  WebPUpsampleLinePairFunc upsample = WebPUpsamplers[p->output->colorspace];
  const uint8_t* cur_y = io->y;
  const uint8_t* cur_u = io->u;
  const uint8_t* cur_v = io->v;
  const uint8_t* top_u = p->tmp_u;
  const uint8_t* top_v = p->tmp_v;
  int y = io->mb_y;
  const int y_end = io->mb_y + io->mb_h;
  const int mb_w = io->mb_w;
  const int uv_w = (mb_w + 1) / 2;

  if (y == 0) {
    // First line is special cased. We mirror the u/v samples at boundary.
    upsample(cur_y, NULL, cur_u, cur_v, cur_u, cur_v, dst, NULL, mb_w);
  } else {
    // We can finish the left-over line from previous call.
    upsample(p->tmp_y, cur_y, top_u, top_v, cur_u, cur_v,
             dst - buf->stride, dst, mb_w);
    ++num_lines_out;
  }
  // Loop over each output pairs of row.
  for (; y + 2 < y_end; y += 2) {
    top_u = cur_u;
    top_v = cur_v;
    cur_u += io->uv_stride;
    cur_v += io->uv_stride;
    dst += 2 * buf->stride;
    cur_y += 2 * io->y_stride;
    upsample(cur_y - io->y_stride, cur_y,
             top_u, top_v, cur_u, cur_v,
             dst - buf->stride, dst, mb_w);
  }
  // move to last row
  cur_y += io->y_stride;
  if (io->crop_top + y_end < io->crop_bottom) {
    // Save the unfinished samples for next call (as we're not done yet).
    memcpy(p->tmp_y, cur_y, mb_w * sizeof(*p->tmp_y));
    memcpy(p->tmp_u, cur_u, uv_w * sizeof(*p->tmp_u));
    memcpy(p->tmp_v, cur_v, uv_w * sizeof(*p->tmp_v));
    // The fancy upsampler leaves a row unfinished behind
    // (except for the very last row)
    num_lines_out--;
  } else {
    // Process the very last row of even-sized picture
    if (!(y_end & 1)) {
      upsample(cur_y, NULL, cur_u, cur_v, cur_u, cur_v,
               dst + buf->stride, NULL, mb_w);
    }
  }
  return num_lines_out;
}

#endif    /* FANCY_UPSAMPLING */

//------------------------------------------------------------------------------

static void FillAlphaPlane(uint8_t* dst, int w, int h, int stride) {
  int j;
  for (j = 0; j < h; ++j) {
    memset(dst, 0xff, w * sizeof(*dst));
    dst += stride;
  }
}

static int EmitAlphaYUV(const VP8Io* const io, WebPDecParams* const p,
                        int expected_num_lines_out) {
  const uint8_t* alpha = io->a;
  const WebPYUVABuffer* const buf = &p->output->u.YUVA;
  const int mb_w = io->mb_w;
  const int mb_h = io->mb_h;
  uint8_t* dst = buf->a + io->mb_y * buf->a_stride;
  int j;
  (void)expected_num_lines_out;
  assert(expected_num_lines_out == mb_h);
  if (alpha != NULL) {
    for (j = 0; j < mb_h; ++j) {
      memcpy(dst, alpha, mb_w * sizeof(*dst));
      alpha += io->width;
      dst += buf->a_stride;
    }
  } else if (buf->a != NULL) {
    // the user requested alpha, but there is none, set it to opaque.
    FillAlphaPlane(dst, mb_w, mb_h, buf->a_stride);
  }
  return 0;
}

static int GetAlphaSourceRow(const VP8Io* const io,
                             const uint8_t** alpha, int* const num_rows) {
  int start_y = io->mb_y;
  *num_rows = io->mb_h;

  // Compensate for the 1-line delay of the fancy upscaler.
  // This is similar to EmitFancyRGB().
  if (io->fancy_upsampling) {
    if (start_y == 0) {
      // We don't process the last row yet. It'll be done during the next call.
      --*num_rows;
    } else {
      --start_y;
      // Fortunately, *alpha data is persistent, so we can go back
      // one row and finish alpha blending, now that the fancy upscaler
      // completed the YUV->RGB interpolation.
      *alpha -= io->width;
    }
    if (io->crop_top + io->mb_y + io->mb_h == io->crop_bottom) {
      // If it's the very last call, we process all the remaining rows!
      *num_rows = io->crop_bottom - io->crop_top - start_y;
    }
  }
  return start_y;
}

static int EmitAlphaRGB(const VP8Io* const io, WebPDecParams* const p,
                        int expected_num_lines_out) {
  const uint8_t* alpha = io->a;
  if (alpha != NULL) {
    const int mb_w = io->mb_w;
    const WEBP_CSP_MODE colorspace = p->output->colorspace;
    const int alpha_first =
        (colorspace == MODE_ARGB || colorspace == MODE_Argb);
    const WebPRGBABuffer* const buf = &p->output->u.RGBA;
    int num_rows;
    const int start_y = GetAlphaSourceRow(io, &alpha, &num_rows);
    uint8_t* const base_rgba = buf->rgba + start_y * buf->stride;
    uint8_t* const dst = base_rgba + (alpha_first ? 0 : 3);
    const int has_alpha = WebPDispatchAlpha(alpha, io->width, mb_w,
                                            num_rows, dst, buf->stride);
    (void)expected_num_lines_out;
    assert(expected_num_lines_out == num_rows);
    // has_alpha is true if there's non-trivial alpha to premultiply with.
    if (has_alpha && WebPIsPremultipliedMode(colorspace)) {
      WebPApplyAlphaMultiply(base_rgba, alpha_first,
                             mb_w, num_rows, buf->stride);
    }
  }
  return 0;
}

static int EmitAlphaRGBA4444(const VP8Io* const io, WebPDecParams* const p,
                             int expected_num_lines_out) {
  const uint8_t* alpha = io->a;
  if (alpha != NULL) {
    const int mb_w = io->mb_w;
    const WEBP_CSP_MODE colorspace = p->output->colorspace;
    const WebPRGBABuffer* const buf = &p->output->u.RGBA;
    int num_rows;
    const int start_y = GetAlphaSourceRow(io, &alpha, &num_rows);
    uint8_t* const base_rgba = buf->rgba + start_y * buf->stride;
#ifdef WEBP_SWAP_16BIT_CSP
    uint8_t* alpha_dst = base_rgba;
#else
    uint8_t* alpha_dst = base_rgba + 1;
#endif
    uint32_t alpha_mask = 0x0f;
    int i, j;
    for (j = 0; j < num_rows; ++j) {
      for (i = 0; i < mb_w; ++i) {
        // Fill in the alpha value (converted to 4 bits).
        const uint32_t alpha_value = alpha[i] >> 4;
        alpha_dst[2 * i] = (alpha_dst[2 * i] & 0xf0) | alpha_value;
        alpha_mask &= alpha_value;
      }
      alpha += io->width;
      alpha_dst += buf->stride;
    }
    (void)expected_num_lines_out;
    assert(expected_num_lines_out == num_rows);
    if (alpha_mask != 0x0f && WebPIsPremultipliedMode(colorspace)) {
      WebPApplyAlphaMultiply4444(base_rgba, mb_w, num_rows, buf->stride);
    }
  }
  return 0;
}

//------------------------------------------------------------------------------
// YUV rescaling (no final RGB conversion needed)

static int Rescale(const uint8_t* src, int src_stride,
                   int new_lines, WebPRescaler* const wrk) {
  int num_lines_out = 0;
  while (new_lines > 0) {    // import new contributions of source rows.
    const int lines_in = WebPRescalerImport(wrk, new_lines, src, src_stride);
    src += lines_in * src_stride;
    new_lines -= lines_in;
    num_lines_out += WebPRescalerExport(wrk);    // emit output row(s)
  }
  return num_lines_out;
}

static int EmitRescaledYUV(const VP8Io* const io, WebPDecParams* const p) {
  const int mb_h = io->mb_h;
  const int uv_mb_h = (mb_h + 1) >> 1;
  WebPRescaler* const scaler = &p->scaler_y;
  int num_lines_out = 0;
  if (WebPIsAlphaMode(p->output->colorspace) && io->a != NULL) {
    // Before rescaling, we premultiply the luma directly into the io->y
    // internal buffer. This is OK since these samples are not used for
    // intra-prediction (the top samples are saved in cache_y_/u_/v_).
    // But we need to cast the const away, though.
    WebPMultRows((uint8_t*)io->y, io->y_stride,
                 io->a, io->width, io->mb_w, mb_h, 0);
  }
  num_lines_out = Rescale(io->y, io->y_stride, mb_h, scaler);
  Rescale(io->u, io->uv_stride, uv_mb_h, &p->scaler_u);
  Rescale(io->v, io->uv_stride, uv_mb_h, &p->scaler_v);
  return num_lines_out;
}

static int EmitRescaledAlphaYUV(const VP8Io* const io, WebPDecParams* const p,
                                int expected_num_lines_out) {
  const WebPYUVABuffer* const buf = &p->output->u.YUVA;
  if (io->a != NULL) {
    uint8_t* dst_y = buf->y + p->last_y * buf->y_stride;
    const uint8_t* src_a = buf->a + p->last_y * buf->a_stride;
    const int num_lines_out = Rescale(io->a, io->width, io->mb_h, &p->scaler_a);
    (void)expected_num_lines_out;
    assert(expected_num_lines_out == num_lines_out);
    if (num_lines_out > 0) {   // unmultiply the Y
      WebPMultRows(dst_y, buf->y_stride, src_a, buf->a_stride,
                   p->scaler_a.dst_width, num_lines_out, 1);
    }
  } else if (buf->a != NULL) {
    // the user requested alpha, but there is none, set it to opaque.
    assert(p->last_y + expected_num_lines_out <= io->scaled_height);
    FillAlphaPlane(buf->a + p->last_y * buf->a_stride,
                   io->scaled_width, expected_num_lines_out, buf->a_stride);
  }
  return 0;
}

static int InitYUVRescaler(const VP8Io* const io, WebPDecParams* const p) {
  const int has_alpha = WebPIsAlphaMode(p->output->colorspace);
  const WebPYUVABuffer* const buf = &p->output->u.YUVA;
  const int out_width  = io->scaled_width;
  const int out_height = io->scaled_height;
  const int uv_out_width  = (out_width + 1) >> 1;
  const int uv_out_height = (out_height + 1) >> 1;
  const int uv_in_width  = (io->mb_w + 1) >> 1;
  const int uv_in_height = (io->mb_h + 1) >> 1;
  const size_t work_size = 2 * out_width;   // scratch memory for luma rescaler
  const size_t uv_work_size = 2 * uv_out_width;  // and for each u/v ones
  size_t tmp_size;
  rescaler_t* work;

  tmp_size = (work_size + 2 * uv_work_size) * sizeof(*work);
  if (has_alpha) {
    tmp_size += work_size * sizeof(*work);
  }
  p->memory = WebPSafeMalloc(1ULL, tmp_size);
  if (p->memory == NULL) {
    return 0;   // memory error
  }
  work = (rescaler_t*)p->memory;
  WebPRescalerInit(&p->scaler_y, io->mb_w, io->mb_h,
                   buf->y, out_width, out_height, buf->y_stride, 1,
                   work);
  WebPRescalerInit(&p->scaler_u, uv_in_width, uv_in_height,
                   buf->u, uv_out_width, uv_out_height, buf->u_stride, 1,
                   work + work_size);
  WebPRescalerInit(&p->scaler_v, uv_in_width, uv_in_height,
                   buf->v, uv_out_width, uv_out_height, buf->v_stride, 1,
                   work + work_size + uv_work_size);
  p->emit = EmitRescaledYUV;

  if (has_alpha) {
    WebPRescalerInit(&p->scaler_a, io->mb_w, io->mb_h,
                     buf->a, out_width, out_height, buf->a_stride, 1,
                     work + work_size + 2 * uv_work_size);
    p->emit_alpha = EmitRescaledAlphaYUV;
    WebPInitAlphaProcessing();
  }
  return 1;
}

//------------------------------------------------------------------------------
// RGBA rescaling

static int ExportRGB(WebPDecParams* const p, int y_pos) {
  const WebPYUV444Converter convert =
      WebPYUV444Converters[p->output->colorspace];
  const WebPRGBABuffer* const buf = &p->output->u.RGBA;
  uint8_t* dst = buf->rgba + y_pos * buf->stride;
  int num_lines_out = 0;
  // For RGB rescaling, because of the YUV420, current scan position
  // U/V can be +1/-1 line from the Y one.  Hence the double test.
  while (WebPRescalerHasPendingOutput(&p->scaler_y) &&
         WebPRescalerHasPendingOutput(&p->scaler_u)) {
    assert(y_pos + num_lines_out < p->output->height);
    assert(p->scaler_u.y_accum == p->scaler_v.y_accum);
    WebPRescalerExportRow(&p->scaler_y);
    WebPRescalerExportRow(&p->scaler_u);
    WebPRescalerExportRow(&p->scaler_v);
    convert(p->scaler_y.dst, p->scaler_u.dst, p->scaler_v.dst,
            dst, p->scaler_y.dst_width);
    dst += buf->stride;
    ++num_lines_out;
  }
  return num_lines_out;
}

static int EmitRescaledRGB(const VP8Io* const io, WebPDecParams* const p) {
  const int mb_h = io->mb_h;
  const int uv_mb_h = (mb_h + 1) >> 1;
  int j = 0, uv_j = 0;
  int num_lines_out = 0;
  while (j < mb_h) {
    const int y_lines_in =
        WebPRescalerImport(&p->scaler_y, mb_h - j,
                           io->y + j * io->y_stride, io->y_stride);
    j += y_lines_in;
    if (WebPRescaleNeededLines(&p->scaler_u, uv_mb_h - uv_j)) {
      const int u_lines_in =
          WebPRescalerImport(&p->scaler_u, uv_mb_h - uv_j,
                             io->u + uv_j * io->uv_stride, io->uv_stride);
      const int v_lines_in =
          WebPRescalerImport(&p->scaler_v, uv_mb_h - uv_j,
                             io->v + uv_j * io->uv_stride, io->uv_stride);
      (void)v_lines_in;   // remove a gcc warning
      assert(u_lines_in == v_lines_in);
      uv_j += u_lines_in;
    }
    num_lines_out += ExportRGB(p, p->last_y + num_lines_out);
  }
  return num_lines_out;
}

static int ExportAlpha(WebPDecParams* const p, int y_pos, int max_lines_out) {
  const WebPRGBABuffer* const buf = &p->output->u.RGBA;
  uint8_t* const base_rgba = buf->rgba + y_pos * buf->stride;
  const WEBP_CSP_MODE colorspace = p->output->colorspace;
  const int alpha_first =
      (colorspace == MODE_ARGB || colorspace == MODE_Argb);
  uint8_t* dst = base_rgba + (alpha_first ? 0 : 3);
  int num_lines_out = 0;
  const int is_premult_alpha = WebPIsPremultipliedMode(colorspace);
  uint32_t non_opaque = 0;
  const int width = p->scaler_a.dst_width;

  while (WebPRescalerHasPendingOutput(&p->scaler_a) &&
         num_lines_out < max_lines_out) {
    assert(y_pos + num_lines_out < p->output->height);
    WebPRescalerExportRow(&p->scaler_a);
    non_opaque |= WebPDispatchAlpha(p->scaler_a.dst, 0, width, 1, dst, 0);
    dst += buf->stride;
    ++num_lines_out;
  }
  if (is_premult_alpha && non_opaque) {
    WebPApplyAlphaMultiply(base_rgba, alpha_first,
                           width, num_lines_out, buf->stride);
  }
  return num_lines_out;
}

static int ExportAlphaRGBA4444(WebPDecParams* const p, int y_pos,
                               int max_lines_out) {
  const WebPRGBABuffer* const buf = &p->output->u.RGBA;
  uint8_t* const base_rgba = buf->rgba + y_pos * buf->stride;
#ifdef WEBP_SWAP_16BIT_CSP
  uint8_t* alpha_dst = base_rgba;
#else
  uint8_t* alpha_dst = base_rgba + 1;
#endif
  int num_lines_out = 0;
  const WEBP_CSP_MODE colorspace = p->output->colorspace;
  const int width = p->scaler_a.dst_width;
  const int is_premult_alpha = WebPIsPremultipliedMode(colorspace);
  uint32_t alpha_mask = 0x0f;

  while (WebPRescalerHasPendingOutput(&p->scaler_a) &&
         num_lines_out < max_lines_out) {
    int i;
    assert(y_pos + num_lines_out < p->output->height);
    WebPRescalerExportRow(&p->scaler_a);
    for (i = 0; i < width; ++i) {
      // Fill in the alpha value (converted to 4 bits).
      const uint32_t alpha_value = p->scaler_a.dst[i] >> 4;
      alpha_dst[2 * i] = (alpha_dst[2 * i] & 0xf0) | alpha_value;
      alpha_mask &= alpha_value;
    }
    alpha_dst += buf->stride;
    ++num_lines_out;
  }
  if (is_premult_alpha && alpha_mask != 0x0f) {
    WebPApplyAlphaMultiply4444(base_rgba, width, num_lines_out, buf->stride);
  }
  return num_lines_out;
}

static int EmitRescaledAlphaRGB(const VP8Io* const io, WebPDecParams* const p,
                                int expected_num_out_lines) {
  if (io->a != NULL) {
    WebPRescaler* const scaler = &p->scaler_a;
    int lines_left = expected_num_out_lines;
    const int y_end = p->last_y + lines_left;
    while (lines_left > 0) {
      const int row_offset = scaler->src_y - io->mb_y;
      WebPRescalerImport(scaler, io->mb_h + io->mb_y - scaler->src_y,
                         io->a + row_offset * io->width, io->width);
      lines_left -= p->emit_alpha_row(p, y_end - lines_left, lines_left);
    }
  }
  return 0;
}

static int InitRGBRescaler(const VP8Io* const io, WebPDecParams* const p) {
  const int has_alpha = WebPIsAlphaMode(p->output->colorspace);
  const int out_width  = io->scaled_width;
  const int out_height = io->scaled_height;
  const int uv_in_width  = (io->mb_w + 1) >> 1;
  const int uv_in_height = (io->mb_h + 1) >> 1;
  const size_t work_size = 2 * out_width;   // scratch memory for one rescaler
  rescaler_t* work;  // rescalers work area
  uint8_t* tmp;   // tmp storage for scaled YUV444 samples before RGB conversion
  size_t tmp_size1, tmp_size2, total_size;

  tmp_size1 = 3 * work_size;
  tmp_size2 = 3 * out_width;
  if (has_alpha) {
    tmp_size1 += work_size;
    tmp_size2 += out_width;
  }
  total_size = tmp_size1 * sizeof(*work) + tmp_size2 * sizeof(*tmp);
  p->memory = WebPSafeMalloc(1ULL, total_size);
  if (p->memory == NULL) {
    return 0;   // memory error
  }
  work = (rescaler_t*)p->memory;
  tmp = (uint8_t*)(work + tmp_size1);
  WebPRescalerInit(&p->scaler_y, io->mb_w, io->mb_h,
                   tmp + 0 * out_width, out_width, out_height, 0, 1,
                   work + 0 * work_size);
  WebPRescalerInit(&p->scaler_u, uv_in_width, uv_in_height,
                   tmp + 1 * out_width, out_width, out_height, 0, 1,
                   work + 1 * work_size);
  WebPRescalerInit(&p->scaler_v, uv_in_width, uv_in_height,
                   tmp + 2 * out_width, out_width, out_height, 0, 1,
                   work + 2 * work_size);
  p->emit = EmitRescaledRGB;
  WebPInitYUV444Converters();

  if (has_alpha) {
    WebPRescalerInit(&p->scaler_a, io->mb_w, io->mb_h,
                     tmp + 3 * out_width, out_width, out_height, 0, 1,
                     work + 3 * work_size);
    p->emit_alpha = EmitRescaledAlphaRGB;
    if (p->output->colorspace == MODE_RGBA_4444 ||
        p->output->colorspace == MODE_rgbA_4444) {
      p->emit_alpha_row = ExportAlphaRGBA4444;
    } else {
      p->emit_alpha_row = ExportAlpha;
    }
    WebPInitAlphaProcessing();
  }
  return 1;
}

//------------------------------------------------------------------------------
// Default custom functions

static int CustomSetup(VP8Io* io) {
  WebPDecParams* const p = (WebPDecParams*)io->opaque;
  const WEBP_CSP_MODE colorspace = p->output->colorspace;
  const int is_rgb = WebPIsRGBMode(colorspace);
  const int is_alpha = WebPIsAlphaMode(colorspace);

  p->memory = NULL;
  p->emit = NULL;
  p->emit_alpha = NULL;
  p->emit_alpha_row = NULL;
  if (!WebPIoInitFromOptions(p->options, io, is_alpha ? MODE_YUV : MODE_YUVA)) {
    return 0;
  }
  if (is_alpha && WebPIsPremultipliedMode(colorspace)) {
    WebPInitUpsamplers();
  }
  if (io->use_scaling) {
    const int ok = is_rgb ? InitRGBRescaler(io, p) : InitYUVRescaler(io, p);
    if (!ok) {
      return 0;    // memory error
    }
  } else {
    if (is_rgb) {
      WebPInitSamplers();
      p->emit = EmitSampledRGB;   // default
      if (io->fancy_upsampling) {
#ifdef FANCY_UPSAMPLING
        const int uv_width = (io->mb_w + 1) >> 1;
        p->memory = WebPSafeMalloc(1ULL, (size_t)(io->mb_w + 2 * uv_width));
        if (p->memory == NULL) {
          return 0;   // memory error.
        }
        p->tmp_y = (uint8_t*)p->memory;
        p->tmp_u = p->tmp_y + io->mb_w;
        p->tmp_v = p->tmp_u + uv_width;
        p->emit = EmitFancyRGB;
        WebPInitUpsamplers();
#endif
      }
    } else {
      p->emit = EmitYUV;
    }
    if (is_alpha) {  // need transparency output
      p->emit_alpha =
          (colorspace == MODE_RGBA_4444 || colorspace == MODE_rgbA_4444) ?
              EmitAlphaRGBA4444
          : is_rgb ? EmitAlphaRGB
          : EmitAlphaYUV;
      if (is_rgb) {
        WebPInitAlphaProcessing();
      }
    }
  }

  if (is_rgb) {
    VP8YUVInit();
  }
  return 1;
}

//------------------------------------------------------------------------------

static int CustomPut(const VP8Io* io) {
  WebPDecParams* const p = (WebPDecParams*)io->opaque;
  const int mb_w = io->mb_w;
  const int mb_h = io->mb_h;
  int num_lines_out;
  assert(!(io->mb_y & 1));

  if (mb_w <= 0 || mb_h <= 0) {
    return 0;
  }
  num_lines_out = p->emit(io, p);
  if (p->emit_alpha != NULL) {
    p->emit_alpha(io, p, num_lines_out);
  }
  p->last_y += num_lines_out;
  return 1;
}

//------------------------------------------------------------------------------

static void CustomTeardown(const VP8Io* io) {
  WebPDecParams* const p = (WebPDecParams*)io->opaque;
  WebPSafeFree(p->memory);
  p->memory = NULL;
}

//------------------------------------------------------------------------------
// Main entry point

void WebPInitCustomIo(WebPDecParams* const params, VP8Io* const io) {
  io->put      = CustomPut;
  io->setup    = CustomSetup;
  io->teardown = CustomTeardown;
  io->opaque   = params;
}

//------------------------------------------------------------------------------

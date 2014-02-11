// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
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

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

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
  WebPDecBuffer* output = p->output;
  const WebPRGBABuffer* const buf = &output->u.RGBA;
  uint8_t* dst = buf->rgba + io->mb_y * buf->stride;
  const uint8_t* y_src = io->y;
  const uint8_t* u_src = io->u;
  const uint8_t* v_src = io->v;
  const WebPSampleLinePairFunc sample = WebPSamplers[output->colorspace];
  const int mb_w = io->mb_w;
  const int last = io->mb_h - 1;
  int j;
  for (j = 0; j < last; j += 2) {
    sample(y_src, y_src + io->y_stride, u_src, v_src,
           dst, dst + buf->stride, mb_w);
    y_src += 2 * io->y_stride;
    u_src += io->uv_stride;
    v_src += io->uv_stride;
    dst += 2 * buf->stride;
  }
  if (j == last) {  // Just do the last line twice
    sample(y_src, y_src, u_src, v_src, dst, dst, mb_w);
  }
  return io->mb_h;
}

//------------------------------------------------------------------------------
// YUV444 -> RGB conversion

#if 0   // TODO(skal): this is for future rescaling.
static int EmitRGB(const VP8Io* const io, WebPDecParams* const p) {
  WebPDecBuffer* output = p->output;
  const WebPRGBABuffer* const buf = &output->u.RGBA;
  uint8_t* dst = buf->rgba + io->mb_y * buf->stride;
  const uint8_t* y_src = io->y;
  const uint8_t* u_src = io->u;
  const uint8_t* v_src = io->v;
  const WebPYUV444Converter convert = WebPYUV444Converters[output->colorspace];
  const int mb_w = io->mb_w;
  const int last = io->mb_h;
  int j;
  for (j = 0; j < last; ++j) {
    convert(y_src, u_src, v_src, dst, mb_w);
    y_src += io->y_stride;
    u_src += io->uv_stride;
    v_src += io->uv_stride;
    dst += buf->stride;
  }
  return io->mb_h;
}
#endif

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
    upsample(NULL, cur_y, cur_u, cur_v, cur_u, cur_v, NULL, dst, mb_w);
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

static int EmitAlphaYUV(const VP8Io* const io, WebPDecParams* const p) {
  const uint8_t* alpha = io->a;
  const WebPYUVABuffer* const buf = &p->output->u.YUVA;
  const int mb_w = io->mb_w;
  const int mb_h = io->mb_h;
  uint8_t* dst = buf->a + io->mb_y * buf->a_stride;
  int j;

  if (alpha != NULL) {
    for (j = 0; j < mb_h; ++j) {
      memcpy(dst, alpha, mb_w * sizeof(*dst));
      alpha += io->width;
      dst += buf->a_stride;
    }
  } else if (buf->a != NULL) {
    // the user requested alpha, but there is none, set it to opaque.
    for (j = 0; j < mb_h; ++j) {
      memset(dst, 0xff, mb_w * sizeof(*dst));
      dst += buf->a_stride;
    }
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

static int EmitAlphaRGB(const VP8Io* const io, WebPDecParams* const p) {
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
    uint8_t* dst = base_rgba + (alpha_first ? 0 : 3);
    uint32_t alpha_mask = 0xff;
    int i, j;

    for (j = 0; j < num_rows; ++j) {
      for (i = 0; i < mb_w; ++i) {
        const uint32_t alpha_value = alpha[i];
        dst[4 * i] = alpha_value;
        alpha_mask &= alpha_value;
      }
      alpha += io->width;
      dst += buf->stride;
    }
    // alpha_mask is < 0xff if there's non-trivial alpha to premultiply with.
    if (alpha_mask != 0xff && WebPIsPremultipliedMode(colorspace)) {
      WebPApplyAlphaMultiply(base_rgba, alpha_first,
                             mb_w, num_rows, buf->stride);
    }
  }
  return 0;
}

static int EmitAlphaRGBA4444(const VP8Io* const io, WebPDecParams* const p) {
  const uint8_t* alpha = io->a;
  if (alpha != NULL) {
    const int mb_w = io->mb_w;
    const WEBP_CSP_MODE colorspace = p->output->colorspace;
    const WebPRGBABuffer* const buf = &p->output->u.RGBA;
    int num_rows;
    const int start_y = GetAlphaSourceRow(io, &alpha, &num_rows);
    uint8_t* const base_rgba = buf->rgba + start_y * buf->stride;
    uint8_t* alpha_dst = base_rgba + 1;
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
  const int num_lines_out = Rescale(io->y, io->y_stride, mb_h, &p->scaler_y);
  Rescale(io->u, io->uv_stride, uv_mb_h, &p->scaler_u);
  Rescale(io->v, io->uv_stride, uv_mb_h, &p->scaler_v);
  return num_lines_out;
}

static int EmitRescaledAlphaYUV(const VP8Io* const io, WebPDecParams* const p) {
  if (io->a != NULL) {
    Rescale(io->a, io->width, io->mb_h, &p->scaler_a);
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
  int32_t* work;

  tmp_size = work_size + 2 * uv_work_size;
  if (has_alpha) {
    tmp_size += work_size;
  }
  p->memory = calloc(1, tmp_size * sizeof(*work));
  if (p->memory == NULL) {
    return 0;   // memory error
  }
  work = (int32_t*)p->memory;
  WebPRescalerInit(&p->scaler_y, io->mb_w, io->mb_h,
                   buf->y, out_width, out_height, buf->y_stride, 1,
                   io->mb_w, out_width, io->mb_h, out_height,
                   work);
  WebPRescalerInit(&p->scaler_u, uv_in_width, uv_in_height,
                   buf->u, uv_out_width, uv_out_height, buf->u_stride, 1,
                   uv_in_width, uv_out_width,
                   uv_in_height, uv_out_height,
                   work + work_size);
  WebPRescalerInit(&p->scaler_v, uv_in_width, uv_in_height,
                   buf->v, uv_out_width, uv_out_height, buf->v_stride, 1,
                   uv_in_width, uv_out_width,
                   uv_in_height, uv_out_height,
                   work + work_size + uv_work_size);
  p->emit = EmitRescaledYUV;

  if (has_alpha) {
    WebPRescalerInit(&p->scaler_a, io->mb_w, io->mb_h,
                     buf->a, out_width, out_height, buf->a_stride, 1,
                     io->mb_w, out_width, io->mb_h, out_height,
                     work + work_size + 2 * uv_work_size);
    p->emit_alpha = EmitRescaledAlphaYUV;
  }
  return 1;
}

//------------------------------------------------------------------------------
// RGBA rescaling

static int ExportRGB(WebPDecParams* const p, int y_pos) {
  const WebPYUV444Converter convert =
      WebPYUV444Converters[p->output->colorspace];
  const WebPRGBABuffer* const buf = &p->output->u.RGBA;
  uint8_t* dst = buf->rgba + (p->last_y + y_pos) * buf->stride;
  int num_lines_out = 0;
  // For RGB rescaling, because of the YUV420, current scan position
  // U/V can be +1/-1 line from the Y one.  Hence the double test.
  while (WebPRescalerHasPendingOutput(&p->scaler_y) &&
         WebPRescalerHasPendingOutput(&p->scaler_u)) {
    assert(p->last_y + y_pos + num_lines_out < p->output->height);
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
    const int u_lines_in =
        WebPRescalerImport(&p->scaler_u, uv_mb_h - uv_j,
                           io->u + uv_j * io->uv_stride, io->uv_stride);
    const int v_lines_in =
        WebPRescalerImport(&p->scaler_v, uv_mb_h - uv_j,
                           io->v + uv_j * io->uv_stride, io->uv_stride);
    (void)v_lines_in;   // remove a gcc warning
    assert(u_lines_in == v_lines_in);
    j += y_lines_in;
    uv_j += u_lines_in;
    num_lines_out += ExportRGB(p, num_lines_out);
  }
  return num_lines_out;
}

static int ExportAlpha(WebPDecParams* const p, int y_pos) {
  const WebPRGBABuffer* const buf = &p->output->u.RGBA;
  uint8_t* const base_rgba = buf->rgba + (p->last_y + y_pos) * buf->stride;
  const WEBP_CSP_MODE colorspace = p->output->colorspace;
  const int alpha_first =
      (colorspace == MODE_ARGB || colorspace == MODE_Argb);
  uint8_t* dst = base_rgba + (alpha_first ? 0 : 3);
  int num_lines_out = 0;
  const int is_premult_alpha = WebPIsPremultipliedMode(colorspace);
  uint32_t alpha_mask = 0xff;
  const int width = p->scaler_a.dst_width;

  while (WebPRescalerHasPendingOutput(&p->scaler_a)) {
    int i;
    assert(p->last_y + y_pos + num_lines_out < p->output->height);
    WebPRescalerExportRow(&p->scaler_a);
    for (i = 0; i < width; ++i) {
      const uint32_t alpha_value = p->scaler_a.dst[i];
      dst[4 * i] = alpha_value;
      alpha_mask &= alpha_value;
    }
    dst += buf->stride;
    ++num_lines_out;
  }
  if (is_premult_alpha && alpha_mask != 0xff) {
    WebPApplyAlphaMultiply(base_rgba, alpha_first,
                           width, num_lines_out, buf->stride);
  }
  return num_lines_out;
}

static int ExportAlphaRGBA4444(WebPDecParams* const p, int y_pos) {
  const WebPRGBABuffer* const buf = &p->output->u.RGBA;
  uint8_t* const base_rgba = buf->rgba + (p->last_y + y_pos) * buf->stride;
  uint8_t* alpha_dst = base_rgba + 1;
  int num_lines_out = 0;
  const WEBP_CSP_MODE colorspace = p->output->colorspace;
  const int width = p->scaler_a.dst_width;
  const int is_premult_alpha = WebPIsPremultipliedMode(colorspace);
  uint32_t alpha_mask = 0x0f;

  while (WebPRescalerHasPendingOutput(&p->scaler_a)) {
    int i;
    assert(p->last_y + y_pos + num_lines_out < p->output->height);
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

static int EmitRescaledAlphaRGB(const VP8Io* const io, WebPDecParams* const p) {
  if (io->a != NULL) {
    WebPRescaler* const scaler = &p->scaler_a;
    int j = 0;
    int pos = 0;
    while (j < io->mb_h) {
      j += WebPRescalerImport(scaler, io->mb_h - j,
                              io->a + j * io->width, io->width);
      pos += p->emit_alpha_row(p, pos);
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
  int32_t* work;  // rescalers work area
  uint8_t* tmp;   // tmp storage for scaled YUV444 samples before RGB conversion
  size_t tmp_size1, tmp_size2;

  tmp_size1 = 3 * work_size;
  tmp_size2 = 3 * out_width;
  if (has_alpha) {
    tmp_size1 += work_size;
    tmp_size2 += out_width;
  }
  p->memory = calloc(1, tmp_size1 * sizeof(*work) + tmp_size2 * sizeof(*tmp));
  if (p->memory == NULL) {
    return 0;   // memory error
  }
  work = (int32_t*)p->memory;
  tmp = (uint8_t*)(work + tmp_size1);
  WebPRescalerInit(&p->scaler_y, io->mb_w, io->mb_h,
                   tmp + 0 * out_width, out_width, out_height, 0, 1,
                   io->mb_w, out_width, io->mb_h, out_height,
                   work + 0 * work_size);
  WebPRescalerInit(&p->scaler_u, uv_in_width, uv_in_height,
                   tmp + 1 * out_width, out_width, out_height, 0, 1,
                   io->mb_w, 2 * out_width, io->mb_h, 2 * out_height,
                   work + 1 * work_size);
  WebPRescalerInit(&p->scaler_v, uv_in_width, uv_in_height,
                   tmp + 2 * out_width, out_width, out_height, 0, 1,
                   io->mb_w, 2 * out_width, io->mb_h, 2 * out_height,
                   work + 2 * work_size);
  p->emit = EmitRescaledRGB;

  if (has_alpha) {
    WebPRescalerInit(&p->scaler_a, io->mb_w, io->mb_h,
                     tmp + 3 * out_width, out_width, out_height, 0, 1,
                     io->mb_w, out_width, io->mb_h, out_height,
                     work + 3 * work_size);
    p->emit_alpha = EmitRescaledAlphaRGB;
    if (p->output->colorspace == MODE_RGBA_4444 ||
        p->output->colorspace == MODE_rgbA_4444) {
      p->emit_alpha_row = ExportAlphaRGBA4444;
    } else {
      p->emit_alpha_row = ExportAlpha;
    }
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

  if (io->use_scaling) {
    const int ok = is_rgb ? InitRGBRescaler(io, p) : InitYUVRescaler(io, p);
    if (!ok) {
      return 0;    // memory error
    }
  } else {
    if (is_rgb) {
      p->emit = EmitSampledRGB;   // default
#ifdef FANCY_UPSAMPLING
      if (io->fancy_upsampling) {
        const int uv_width = (io->mb_w + 1) >> 1;
        p->memory = malloc(io->mb_w + 2 * uv_width);
        if (p->memory == NULL) {
          return 0;   // memory error.
        }
        p->tmp_y = (uint8_t*)p->memory;
        p->tmp_u = p->tmp_y + io->mb_w;
        p->tmp_v = p->tmp_u + uv_width;
        p->emit = EmitFancyRGB;
        WebPInitUpsamplers();
      }
#endif
    } else {
      p->emit = EmitYUV;
    }
    if (is_alpha) {  // need transparency output
      if (WebPIsPremultipliedMode(colorspace)) WebPInitPremultiply();
      p->emit_alpha =
          (colorspace == MODE_RGBA_4444 || colorspace == MODE_rgbA_4444) ?
              EmitAlphaRGBA4444
          : is_rgb ? EmitAlphaRGB
          : EmitAlphaYUV;
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
  if (p->emit_alpha) {
    p->emit_alpha(io, p);
  }
  p->last_y += num_lines_out;
  return 1;
}

//------------------------------------------------------------------------------

static void CustomTeardown(const VP8Io* io) {
  WebPDecParams* const p = (WebPDecParams*)io->opaque;
  free(p->memory);
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

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

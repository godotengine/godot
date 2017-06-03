// Copyright 2010 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Main decoding functions for WEBP images.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <stdlib.h>

#include "./vp8i_dec.h"
#include "./vp8li_dec.h"
#include "./webpi_dec.h"
#include "../utils/utils.h"
#include "../webp/mux_types.h"  // ALPHA_FLAG

//------------------------------------------------------------------------------
// RIFF layout is:
//   Offset  tag
//   0...3   "RIFF" 4-byte tag
//   4...7   size of image data (including metadata) starting at offset 8
//   8...11  "WEBP"   our form-type signature
// The RIFF container (12 bytes) is followed by appropriate chunks:
//   12..15  "VP8 ": 4-bytes tags, signaling the use of VP8 video format
//   16..19  size of the raw VP8 image data, starting at offset 20
//   20....  the VP8 bytes
// Or,
//   12..15  "VP8L": 4-bytes tags, signaling the use of VP8L lossless format
//   16..19  size of the raw VP8L image data, starting at offset 20
//   20....  the VP8L bytes
// Or,
//   12..15  "VP8X": 4-bytes tags, describing the extended-VP8 chunk.
//   16..19  size of the VP8X chunk starting at offset 20.
//   20..23  VP8X flags bit-map corresponding to the chunk-types present.
//   24..26  Width of the Canvas Image.
//   27..29  Height of the Canvas Image.
// There can be extra chunks after the "VP8X" chunk (ICCP, ANMF, VP8, VP8L,
// XMP, EXIF  ...)
// All sizes are in little-endian order.
// Note: chunk data size must be padded to multiple of 2 when written.

// Validates the RIFF container (if detected) and skips over it.
// If a RIFF container is detected, returns:
//     VP8_STATUS_BITSTREAM_ERROR for invalid header,
//     VP8_STATUS_NOT_ENOUGH_DATA for truncated data if have_all_data is true,
// and VP8_STATUS_OK otherwise.
// In case there are not enough bytes (partial RIFF container), return 0 for
// *riff_size. Else return the RIFF size extracted from the header.
static VP8StatusCode ParseRIFF(const uint8_t** const data,
                               size_t* const data_size, int have_all_data,
                               size_t* const riff_size) {
  assert(data != NULL);
  assert(data_size != NULL);
  assert(riff_size != NULL);

  *riff_size = 0;  // Default: no RIFF present.
  if (*data_size >= RIFF_HEADER_SIZE && !memcmp(*data, "RIFF", TAG_SIZE)) {
    if (memcmp(*data + 8, "WEBP", TAG_SIZE)) {
      return VP8_STATUS_BITSTREAM_ERROR;  // Wrong image file signature.
    } else {
      const uint32_t size = GetLE32(*data + TAG_SIZE);
      // Check that we have at least one chunk (i.e "WEBP" + "VP8?nnnn").
      if (size < TAG_SIZE + CHUNK_HEADER_SIZE) {
        return VP8_STATUS_BITSTREAM_ERROR;
      }
      if (size > MAX_CHUNK_PAYLOAD) {
        return VP8_STATUS_BITSTREAM_ERROR;
      }
      if (have_all_data && (size > *data_size - CHUNK_HEADER_SIZE)) {
        return VP8_STATUS_NOT_ENOUGH_DATA;  // Truncated bitstream.
      }
      // We have a RIFF container. Skip it.
      *riff_size = size;
      *data += RIFF_HEADER_SIZE;
      *data_size -= RIFF_HEADER_SIZE;
    }
  }
  return VP8_STATUS_OK;
}

// Validates the VP8X header and skips over it.
// Returns VP8_STATUS_BITSTREAM_ERROR for invalid VP8X header,
//         VP8_STATUS_NOT_ENOUGH_DATA in case of insufficient data, and
//         VP8_STATUS_OK otherwise.
// If a VP8X chunk is found, found_vp8x is set to true and *width_ptr,
// *height_ptr and *flags_ptr are set to the corresponding values extracted
// from the VP8X chunk.
static VP8StatusCode ParseVP8X(const uint8_t** const data,
                               size_t* const data_size,
                               int* const found_vp8x,
                               int* const width_ptr, int* const height_ptr,
                               uint32_t* const flags_ptr) {
  const uint32_t vp8x_size = CHUNK_HEADER_SIZE + VP8X_CHUNK_SIZE;
  assert(data != NULL);
  assert(data_size != NULL);
  assert(found_vp8x != NULL);

  *found_vp8x = 0;

  if (*data_size < CHUNK_HEADER_SIZE) {
    return VP8_STATUS_NOT_ENOUGH_DATA;  // Insufficient data.
  }

  if (!memcmp(*data, "VP8X", TAG_SIZE)) {
    int width, height;
    uint32_t flags;
    const uint32_t chunk_size = GetLE32(*data + TAG_SIZE);
    if (chunk_size != VP8X_CHUNK_SIZE) {
      return VP8_STATUS_BITSTREAM_ERROR;  // Wrong chunk size.
    }

    // Verify if enough data is available to validate the VP8X chunk.
    if (*data_size < vp8x_size) {
      return VP8_STATUS_NOT_ENOUGH_DATA;  // Insufficient data.
    }
    flags = GetLE32(*data + 8);
    width = 1 + GetLE24(*data + 12);
    height = 1 + GetLE24(*data + 15);
    if (width * (uint64_t)height >= MAX_IMAGE_AREA) {
      return VP8_STATUS_BITSTREAM_ERROR;  // image is too large
    }

    if (flags_ptr != NULL) *flags_ptr = flags;
    if (width_ptr != NULL) *width_ptr = width;
    if (height_ptr != NULL) *height_ptr = height;
    // Skip over VP8X header bytes.
    *data += vp8x_size;
    *data_size -= vp8x_size;
    *found_vp8x = 1;
  }
  return VP8_STATUS_OK;
}

// Skips to the next VP8/VP8L chunk header in the data given the size of the
// RIFF chunk 'riff_size'.
// Returns VP8_STATUS_BITSTREAM_ERROR if any invalid chunk size is encountered,
//         VP8_STATUS_NOT_ENOUGH_DATA in case of insufficient data, and
//         VP8_STATUS_OK otherwise.
// If an alpha chunk is found, *alpha_data and *alpha_size are set
// appropriately.
static VP8StatusCode ParseOptionalChunks(const uint8_t** const data,
                                         size_t* const data_size,
                                         size_t const riff_size,
                                         const uint8_t** const alpha_data,
                                         size_t* const alpha_size) {
  const uint8_t* buf;
  size_t buf_size;
  uint32_t total_size = TAG_SIZE +           // "WEBP".
                        CHUNK_HEADER_SIZE +  // "VP8Xnnnn".
                        VP8X_CHUNK_SIZE;     // data.
  assert(data != NULL);
  assert(data_size != NULL);
  buf = *data;
  buf_size = *data_size;

  assert(alpha_data != NULL);
  assert(alpha_size != NULL);
  *alpha_data = NULL;
  *alpha_size = 0;

  while (1) {
    uint32_t chunk_size;
    uint32_t disk_chunk_size;   // chunk_size with padding

    *data = buf;
    *data_size = buf_size;

    if (buf_size < CHUNK_HEADER_SIZE) {  // Insufficient data.
      return VP8_STATUS_NOT_ENOUGH_DATA;
    }

    chunk_size = GetLE32(buf + TAG_SIZE);
    if (chunk_size > MAX_CHUNK_PAYLOAD) {
      return VP8_STATUS_BITSTREAM_ERROR;          // Not a valid chunk size.
    }
    // For odd-sized chunk-payload, there's one byte padding at the end.
    disk_chunk_size = (CHUNK_HEADER_SIZE + chunk_size + 1) & ~1;
    total_size += disk_chunk_size;

    // Check that total bytes skipped so far does not exceed riff_size.
    if (riff_size > 0 && (total_size > riff_size)) {
      return VP8_STATUS_BITSTREAM_ERROR;          // Not a valid chunk size.
    }

    // Start of a (possibly incomplete) VP8/VP8L chunk implies that we have
    // parsed all the optional chunks.
    // Note: This check must occur before the check 'buf_size < disk_chunk_size'
    // below to allow incomplete VP8/VP8L chunks.
    if (!memcmp(buf, "VP8 ", TAG_SIZE) ||
        !memcmp(buf, "VP8L", TAG_SIZE)) {
      return VP8_STATUS_OK;
    }

    if (buf_size < disk_chunk_size) {             // Insufficient data.
      return VP8_STATUS_NOT_ENOUGH_DATA;
    }

    if (!memcmp(buf, "ALPH", TAG_SIZE)) {         // A valid ALPH header.
      *alpha_data = buf + CHUNK_HEADER_SIZE;
      *alpha_size = chunk_size;
    }

    // We have a full and valid chunk; skip it.
    buf += disk_chunk_size;
    buf_size -= disk_chunk_size;
  }
}

// Validates the VP8/VP8L Header ("VP8 nnnn" or "VP8L nnnn") and skips over it.
// Returns VP8_STATUS_BITSTREAM_ERROR for invalid (chunk larger than
//         riff_size) VP8/VP8L header,
//         VP8_STATUS_NOT_ENOUGH_DATA in case of insufficient data, and
//         VP8_STATUS_OK otherwise.
// If a VP8/VP8L chunk is found, *chunk_size is set to the total number of bytes
// extracted from the VP8/VP8L chunk header.
// The flag '*is_lossless' is set to 1 in case of VP8L chunk / raw VP8L data.
static VP8StatusCode ParseVP8Header(const uint8_t** const data_ptr,
                                    size_t* const data_size, int have_all_data,
                                    size_t riff_size, size_t* const chunk_size,
                                    int* const is_lossless) {
  const uint8_t* const data = *data_ptr;
  const int is_vp8 = !memcmp(data, "VP8 ", TAG_SIZE);
  const int is_vp8l = !memcmp(data, "VP8L", TAG_SIZE);
  const uint32_t minimal_size =
      TAG_SIZE + CHUNK_HEADER_SIZE;  // "WEBP" + "VP8 nnnn" OR
                                     // "WEBP" + "VP8Lnnnn"
  assert(data != NULL);
  assert(data_size != NULL);
  assert(chunk_size != NULL);
  assert(is_lossless != NULL);

  if (*data_size < CHUNK_HEADER_SIZE) {
    return VP8_STATUS_NOT_ENOUGH_DATA;  // Insufficient data.
  }

  if (is_vp8 || is_vp8l) {
    // Bitstream contains VP8/VP8L header.
    const uint32_t size = GetLE32(data + TAG_SIZE);
    if ((riff_size >= minimal_size) && (size > riff_size - minimal_size)) {
      return VP8_STATUS_BITSTREAM_ERROR;  // Inconsistent size information.
    }
    if (have_all_data && (size > *data_size - CHUNK_HEADER_SIZE)) {
      return VP8_STATUS_NOT_ENOUGH_DATA;  // Truncated bitstream.
    }
    // Skip over CHUNK_HEADER_SIZE bytes from VP8/VP8L Header.
    *chunk_size = size;
    *data_ptr += CHUNK_HEADER_SIZE;
    *data_size -= CHUNK_HEADER_SIZE;
    *is_lossless = is_vp8l;
  } else {
    // Raw VP8/VP8L bitstream (no header).
    *is_lossless = VP8LCheckSignature(data, *data_size);
    *chunk_size = *data_size;
  }

  return VP8_STATUS_OK;
}

//------------------------------------------------------------------------------

// Fetch '*width', '*height', '*has_alpha' and fill out 'headers' based on
// 'data'. All the output parameters may be NULL. If 'headers' is NULL only the
// minimal amount will be read to fetch the remaining parameters.
// If 'headers' is non-NULL this function will attempt to locate both alpha
// data (with or without a VP8X chunk) and the bitstream chunk (VP8/VP8L).
// Note: The following chunk sequences (before the raw VP8/VP8L data) are
// considered valid by this function:
// RIFF + VP8(L)
// RIFF + VP8X + (optional chunks) + VP8(L)
// ALPH + VP8 <-- Not a valid WebP format: only allowed for internal purpose.
// VP8(L)     <-- Not a valid WebP format: only allowed for internal purpose.
static VP8StatusCode ParseHeadersInternal(const uint8_t* data,
                                          size_t data_size,
                                          int* const width,
                                          int* const height,
                                          int* const has_alpha,
                                          int* const has_animation,
                                          int* const format,
                                          WebPHeaderStructure* const headers) {
  int canvas_width = 0;
  int canvas_height = 0;
  int image_width = 0;
  int image_height = 0;
  int found_riff = 0;
  int found_vp8x = 0;
  int animation_present = 0;
  const int have_all_data = (headers != NULL) ? headers->have_all_data : 0;

  VP8StatusCode status;
  WebPHeaderStructure hdrs;

  if (data == NULL || data_size < RIFF_HEADER_SIZE) {
    return VP8_STATUS_NOT_ENOUGH_DATA;
  }
  memset(&hdrs, 0, sizeof(hdrs));
  hdrs.data = data;
  hdrs.data_size = data_size;

  // Skip over RIFF header.
  status = ParseRIFF(&data, &data_size, have_all_data, &hdrs.riff_size);
  if (status != VP8_STATUS_OK) {
    return status;   // Wrong RIFF header / insufficient data.
  }
  found_riff = (hdrs.riff_size > 0);

  // Skip over VP8X.
  {
    uint32_t flags = 0;
    status = ParseVP8X(&data, &data_size, &found_vp8x,
                       &canvas_width, &canvas_height, &flags);
    if (status != VP8_STATUS_OK) {
      return status;  // Wrong VP8X / insufficient data.
    }
    animation_present = !!(flags & ANIMATION_FLAG);
    if (!found_riff && found_vp8x) {
      // Note: This restriction may be removed in the future, if it becomes
      // necessary to send VP8X chunk to the decoder.
      return VP8_STATUS_BITSTREAM_ERROR;
    }
    if (has_alpha != NULL) *has_alpha = !!(flags & ALPHA_FLAG);
    if (has_animation != NULL) *has_animation = animation_present;
    if (format != NULL) *format = 0;   // default = undefined

    image_width = canvas_width;
    image_height = canvas_height;
    if (found_vp8x && animation_present && headers == NULL) {
      status = VP8_STATUS_OK;
      goto ReturnWidthHeight;  // Just return features from VP8X header.
    }
  }

  if (data_size < TAG_SIZE) {
    status = VP8_STATUS_NOT_ENOUGH_DATA;
    goto ReturnWidthHeight;
  }

  // Skip over optional chunks if data started with "RIFF + VP8X" or "ALPH".
  if ((found_riff && found_vp8x) ||
      (!found_riff && !found_vp8x && !memcmp(data, "ALPH", TAG_SIZE))) {
    status = ParseOptionalChunks(&data, &data_size, hdrs.riff_size,
                                 &hdrs.alpha_data, &hdrs.alpha_data_size);
    if (status != VP8_STATUS_OK) {
      goto ReturnWidthHeight;  // Invalid chunk size / insufficient data.
    }
  }

  // Skip over VP8/VP8L header.
  status = ParseVP8Header(&data, &data_size, have_all_data, hdrs.riff_size,
                          &hdrs.compressed_size, &hdrs.is_lossless);
  if (status != VP8_STATUS_OK) {
    goto ReturnWidthHeight;  // Wrong VP8/VP8L chunk-header / insufficient data.
  }
  if (hdrs.compressed_size > MAX_CHUNK_PAYLOAD) {
    return VP8_STATUS_BITSTREAM_ERROR;
  }

  if (format != NULL && !animation_present) {
    *format = hdrs.is_lossless ? 2 : 1;
  }

  if (!hdrs.is_lossless) {
    if (data_size < VP8_FRAME_HEADER_SIZE) {
      status = VP8_STATUS_NOT_ENOUGH_DATA;
      goto ReturnWidthHeight;
    }
    // Validates raw VP8 data.
    if (!VP8GetInfo(data, data_size, (uint32_t)hdrs.compressed_size,
                    &image_width, &image_height)) {
      return VP8_STATUS_BITSTREAM_ERROR;
    }
  } else {
    if (data_size < VP8L_FRAME_HEADER_SIZE) {
      status = VP8_STATUS_NOT_ENOUGH_DATA;
      goto ReturnWidthHeight;
    }
    // Validates raw VP8L data.
    if (!VP8LGetInfo(data, data_size, &image_width, &image_height, has_alpha)) {
      return VP8_STATUS_BITSTREAM_ERROR;
    }
  }
  // Validates image size coherency.
  if (found_vp8x) {
    if (canvas_width != image_width || canvas_height != image_height) {
      return VP8_STATUS_BITSTREAM_ERROR;
    }
  }
  if (headers != NULL) {
    *headers = hdrs;
    headers->offset = data - headers->data;
    assert((uint64_t)(data - headers->data) < MAX_CHUNK_PAYLOAD);
    assert(headers->offset == headers->data_size - data_size);
  }
 ReturnWidthHeight:
  if (status == VP8_STATUS_OK ||
      (status == VP8_STATUS_NOT_ENOUGH_DATA && found_vp8x && headers == NULL)) {
    if (has_alpha != NULL) {
      // If the data did not contain a VP8X/VP8L chunk the only definitive way
      // to set this is by looking for alpha data (from an ALPH chunk).
      *has_alpha |= (hdrs.alpha_data != NULL);
    }
    if (width != NULL) *width = image_width;
    if (height != NULL) *height = image_height;
    return VP8_STATUS_OK;
  } else {
    return status;
  }
}

VP8StatusCode WebPParseHeaders(WebPHeaderStructure* const headers) {
  // status is marked volatile as a workaround for a clang-3.8 (aarch64) bug
  volatile VP8StatusCode status;
  int has_animation = 0;
  assert(headers != NULL);
  // fill out headers, ignore width/height/has_alpha.
  status = ParseHeadersInternal(headers->data, headers->data_size,
                                NULL, NULL, NULL, &has_animation,
                                NULL, headers);
  if (status == VP8_STATUS_OK || status == VP8_STATUS_NOT_ENOUGH_DATA) {
    // TODO(jzern): full support of animation frames will require API additions.
    if (has_animation) {
      status = VP8_STATUS_UNSUPPORTED_FEATURE;
    }
  }
  return status;
}

//------------------------------------------------------------------------------
// WebPDecParams

void WebPResetDecParams(WebPDecParams* const params) {
  if (params != NULL) {
    memset(params, 0, sizeof(*params));
  }
}

//------------------------------------------------------------------------------
// "Into" decoding variants

// Main flow
static VP8StatusCode DecodeInto(const uint8_t* const data, size_t data_size,
                                WebPDecParams* const params) {
  VP8StatusCode status;
  VP8Io io;
  WebPHeaderStructure headers;

  headers.data = data;
  headers.data_size = data_size;
  headers.have_all_data = 1;
  status = WebPParseHeaders(&headers);   // Process Pre-VP8 chunks.
  if (status != VP8_STATUS_OK) {
    return status;
  }

  assert(params != NULL);
  VP8InitIo(&io);
  io.data = headers.data + headers.offset;
  io.data_size = headers.data_size - headers.offset;
  WebPInitCustomIo(params, &io);  // Plug the I/O functions.

  if (!headers.is_lossless) {
    VP8Decoder* const dec = VP8New();
    if (dec == NULL) {
      return VP8_STATUS_OUT_OF_MEMORY;
    }
    dec->alpha_data_ = headers.alpha_data;
    dec->alpha_data_size_ = headers.alpha_data_size;

    // Decode bitstream header, update io->width/io->height.
    if (!VP8GetHeaders(dec, &io)) {
      status = dec->status_;   // An error occurred. Grab error status.
    } else {
      // Allocate/check output buffers.
      status = WebPAllocateDecBuffer(io.width, io.height, params->options,
                                     params->output);
      if (status == VP8_STATUS_OK) {  // Decode
        // This change must be done before calling VP8Decode()
        dec->mt_method_ = VP8GetThreadMethod(params->options, &headers,
                                             io.width, io.height);
        VP8InitDithering(params->options, dec);
        if (!VP8Decode(dec, &io)) {
          status = dec->status_;
        }
      }
    }
    VP8Delete(dec);
  } else {
    VP8LDecoder* const dec = VP8LNew();
    if (dec == NULL) {
      return VP8_STATUS_OUT_OF_MEMORY;
    }
    if (!VP8LDecodeHeader(dec, &io)) {
      status = dec->status_;   // An error occurred. Grab error status.
    } else {
      // Allocate/check output buffers.
      status = WebPAllocateDecBuffer(io.width, io.height, params->options,
                                     params->output);
      if (status == VP8_STATUS_OK) {  // Decode
        if (!VP8LDecodeImage(dec)) {
          status = dec->status_;
        }
      }
    }
    VP8LDelete(dec);
  }

  if (status != VP8_STATUS_OK) {
    WebPFreeDecBuffer(params->output);
  } else {
    if (params->options != NULL && params->options->flip) {
      // This restores the original stride values if options->flip was used
      // during the call to WebPAllocateDecBuffer above.
      status = WebPFlipBuffer(params->output);
    }
  }
  return status;
}

// Helpers
static uint8_t* DecodeIntoRGBABuffer(WEBP_CSP_MODE colorspace,
                                     const uint8_t* const data,
                                     size_t data_size,
                                     uint8_t* const rgba,
                                     int stride, size_t size) {
  WebPDecParams params;
  WebPDecBuffer buf;
  if (rgba == NULL) {
    return NULL;
  }
  WebPInitDecBuffer(&buf);
  WebPResetDecParams(&params);
  params.output = &buf;
  buf.colorspace    = colorspace;
  buf.u.RGBA.rgba   = rgba;
  buf.u.RGBA.stride = stride;
  buf.u.RGBA.size   = size;
  buf.is_external_memory = 1;
  if (DecodeInto(data, data_size, &params) != VP8_STATUS_OK) {
    return NULL;
  }
  return rgba;
}

uint8_t* WebPDecodeRGBInto(const uint8_t* data, size_t data_size,
                           uint8_t* output, size_t size, int stride) {
  return DecodeIntoRGBABuffer(MODE_RGB, data, data_size, output, stride, size);
}

uint8_t* WebPDecodeRGBAInto(const uint8_t* data, size_t data_size,
                            uint8_t* output, size_t size, int stride) {
  return DecodeIntoRGBABuffer(MODE_RGBA, data, data_size, output, stride, size);
}

uint8_t* WebPDecodeARGBInto(const uint8_t* data, size_t data_size,
                            uint8_t* output, size_t size, int stride) {
  return DecodeIntoRGBABuffer(MODE_ARGB, data, data_size, output, stride, size);
}

uint8_t* WebPDecodeBGRInto(const uint8_t* data, size_t data_size,
                           uint8_t* output, size_t size, int stride) {
  return DecodeIntoRGBABuffer(MODE_BGR, data, data_size, output, stride, size);
}

uint8_t* WebPDecodeBGRAInto(const uint8_t* data, size_t data_size,
                            uint8_t* output, size_t size, int stride) {
  return DecodeIntoRGBABuffer(MODE_BGRA, data, data_size, output, stride, size);
}

uint8_t* WebPDecodeYUVInto(const uint8_t* data, size_t data_size,
                           uint8_t* luma, size_t luma_size, int luma_stride,
                           uint8_t* u, size_t u_size, int u_stride,
                           uint8_t* v, size_t v_size, int v_stride) {
  WebPDecParams params;
  WebPDecBuffer output;
  if (luma == NULL) return NULL;
  WebPInitDecBuffer(&output);
  WebPResetDecParams(&params);
  params.output = &output;
  output.colorspace      = MODE_YUV;
  output.u.YUVA.y        = luma;
  output.u.YUVA.y_stride = luma_stride;
  output.u.YUVA.y_size   = luma_size;
  output.u.YUVA.u        = u;
  output.u.YUVA.u_stride = u_stride;
  output.u.YUVA.u_size   = u_size;
  output.u.YUVA.v        = v;
  output.u.YUVA.v_stride = v_stride;
  output.u.YUVA.v_size   = v_size;
  output.is_external_memory = 1;
  if (DecodeInto(data, data_size, &params) != VP8_STATUS_OK) {
    return NULL;
  }
  return luma;
}

//------------------------------------------------------------------------------

static uint8_t* Decode(WEBP_CSP_MODE mode, const uint8_t* const data,
                       size_t data_size, int* const width, int* const height,
                       WebPDecBuffer* const keep_info) {
  WebPDecParams params;
  WebPDecBuffer output;

  WebPInitDecBuffer(&output);
  WebPResetDecParams(&params);
  params.output = &output;
  output.colorspace = mode;

  // Retrieve (and report back) the required dimensions from bitstream.
  if (!WebPGetInfo(data, data_size, &output.width, &output.height)) {
    return NULL;
  }
  if (width != NULL) *width = output.width;
  if (height != NULL) *height = output.height;

  // Decode
  if (DecodeInto(data, data_size, &params) != VP8_STATUS_OK) {
    return NULL;
  }
  if (keep_info != NULL) {    // keep track of the side-info
    WebPCopyDecBuffer(&output, keep_info);
  }
  // return decoded samples (don't clear 'output'!)
  return WebPIsRGBMode(mode) ? output.u.RGBA.rgba : output.u.YUVA.y;
}

uint8_t* WebPDecodeRGB(const uint8_t* data, size_t data_size,
                       int* width, int* height) {
  return Decode(MODE_RGB, data, data_size, width, height, NULL);
}

uint8_t* WebPDecodeRGBA(const uint8_t* data, size_t data_size,
                        int* width, int* height) {
  return Decode(MODE_RGBA, data, data_size, width, height, NULL);
}

uint8_t* WebPDecodeARGB(const uint8_t* data, size_t data_size,
                        int* width, int* height) {
  return Decode(MODE_ARGB, data, data_size, width, height, NULL);
}

uint8_t* WebPDecodeBGR(const uint8_t* data, size_t data_size,
                       int* width, int* height) {
  return Decode(MODE_BGR, data, data_size, width, height, NULL);
}

uint8_t* WebPDecodeBGRA(const uint8_t* data, size_t data_size,
                        int* width, int* height) {
  return Decode(MODE_BGRA, data, data_size, width, height, NULL);
}

uint8_t* WebPDecodeYUV(const uint8_t* data, size_t data_size,
                       int* width, int* height, uint8_t** u, uint8_t** v,
                       int* stride, int* uv_stride) {
  WebPDecBuffer output;   // only to preserve the side-infos
  uint8_t* const out = Decode(MODE_YUV, data, data_size,
                              width, height, &output);

  if (out != NULL) {
    const WebPYUVABuffer* const buf = &output.u.YUVA;
    *u = buf->u;
    *v = buf->v;
    *stride = buf->y_stride;
    *uv_stride = buf->u_stride;
    assert(buf->u_stride == buf->v_stride);
  }
  return out;
}

static void DefaultFeatures(WebPBitstreamFeatures* const features) {
  assert(features != NULL);
  memset(features, 0, sizeof(*features));
}

static VP8StatusCode GetFeatures(const uint8_t* const data, size_t data_size,
                                 WebPBitstreamFeatures* const features) {
  if (features == NULL || data == NULL) {
    return VP8_STATUS_INVALID_PARAM;
  }
  DefaultFeatures(features);

  // Only parse enough of the data to retrieve the features.
  return ParseHeadersInternal(data, data_size,
                              &features->width, &features->height,
                              &features->has_alpha, &features->has_animation,
                              &features->format, NULL);
}

//------------------------------------------------------------------------------
// WebPGetInfo()

int WebPGetInfo(const uint8_t* data, size_t data_size,
                int* width, int* height) {
  WebPBitstreamFeatures features;

  if (GetFeatures(data, data_size, &features) != VP8_STATUS_OK) {
    return 0;
  }

  if (width != NULL) {
    *width  = features.width;
  }
  if (height != NULL) {
    *height = features.height;
  }

  return 1;
}

//------------------------------------------------------------------------------
// Advance decoding API

int WebPInitDecoderConfigInternal(WebPDecoderConfig* config,
                                  int version) {
  if (WEBP_ABI_IS_INCOMPATIBLE(version, WEBP_DECODER_ABI_VERSION)) {
    return 0;   // version mismatch
  }
  if (config == NULL) {
    return 0;
  }
  memset(config, 0, sizeof(*config));
  DefaultFeatures(&config->input);
  WebPInitDecBuffer(&config->output);
  return 1;
}

VP8StatusCode WebPGetFeaturesInternal(const uint8_t* data, size_t data_size,
                                      WebPBitstreamFeatures* features,
                                      int version) {
  if (WEBP_ABI_IS_INCOMPATIBLE(version, WEBP_DECODER_ABI_VERSION)) {
    return VP8_STATUS_INVALID_PARAM;   // version mismatch
  }
  if (features == NULL) {
    return VP8_STATUS_INVALID_PARAM;
  }
  return GetFeatures(data, data_size, features);
}

VP8StatusCode WebPDecode(const uint8_t* data, size_t data_size,
                         WebPDecoderConfig* config) {
  WebPDecParams params;
  VP8StatusCode status;

  if (config == NULL) {
    return VP8_STATUS_INVALID_PARAM;
  }

  status = GetFeatures(data, data_size, &config->input);
  if (status != VP8_STATUS_OK) {
    if (status == VP8_STATUS_NOT_ENOUGH_DATA) {
      return VP8_STATUS_BITSTREAM_ERROR;  // Not-enough-data treated as error.
    }
    return status;
  }

  WebPResetDecParams(&params);
  params.options = &config->options;
  params.output = &config->output;
  if (WebPAvoidSlowMemory(params.output, &config->input)) {
    // decoding to slow memory: use a temporary in-mem buffer to decode into.
    WebPDecBuffer in_mem_buffer;
    WebPInitDecBuffer(&in_mem_buffer);
    in_mem_buffer.colorspace = config->output.colorspace;
    in_mem_buffer.width = config->input.width;
    in_mem_buffer.height = config->input.height;
    params.output = &in_mem_buffer;
    status = DecodeInto(data, data_size, &params);
    if (status == VP8_STATUS_OK) {  // do the slow-copy
      status = WebPCopyDecBufferPixels(&in_mem_buffer, &config->output);
    }
    WebPFreeDecBuffer(&in_mem_buffer);
  } else {
    status = DecodeInto(data, data_size, &params);
  }

  return status;
}

//------------------------------------------------------------------------------
// Cropping and rescaling.

int WebPIoInitFromOptions(const WebPDecoderOptions* const options,
                          VP8Io* const io, WEBP_CSP_MODE src_colorspace) {
  const int W = io->width;
  const int H = io->height;
  int x = 0, y = 0, w = W, h = H;

  // Cropping
  io->use_cropping = (options != NULL) && (options->use_cropping > 0);
  if (io->use_cropping) {
    w = options->crop_width;
    h = options->crop_height;
    x = options->crop_left;
    y = options->crop_top;
    if (!WebPIsRGBMode(src_colorspace)) {   // only snap for YUV420
      x &= ~1;
      y &= ~1;
    }
    if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > W || y + h > H) {
      return 0;  // out of frame boundary error
    }
  }
  io->crop_left   = x;
  io->crop_top    = y;
  io->crop_right  = x + w;
  io->crop_bottom = y + h;
  io->mb_w = w;
  io->mb_h = h;

  // Scaling
  io->use_scaling = (options != NULL) && (options->use_scaling > 0);
  if (io->use_scaling) {
    int scaled_width = options->scaled_width;
    int scaled_height = options->scaled_height;
    if (!WebPRescalerGetScaledDimensions(w, h, &scaled_width, &scaled_height)) {
      return 0;
    }
    io->scaled_width = scaled_width;
    io->scaled_height = scaled_height;
  }

  // Filter
  io->bypass_filtering = (options != NULL) && options->bypass_filtering;

  // Fancy upsampler
#ifdef FANCY_UPSAMPLING
  io->fancy_upsampling = (options == NULL) || (!options->no_fancy_upsampling);
#endif

  if (io->use_scaling) {
    // disable filter (only for large downscaling ratio).
    io->bypass_filtering = (io->scaled_width < W * 3 / 4) &&
                           (io->scaled_height < H * 3 / 4);
    io->fancy_upsampling = 0;
  }
  return 1;
}

//------------------------------------------------------------------------------

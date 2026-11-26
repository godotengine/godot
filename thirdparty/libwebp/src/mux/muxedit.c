// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Set and delete APIs for mux.
//
// Authors: Urvang (urvang@google.com)
//          Vikas (vikasa@google.com)

#include <assert.h>
#include <stddef.h>
#include <string.h>

#include "src/dec/vp8_dec.h"
#include "src/mux/muxi.h"
#include "src/utils/utils.h"
#include "src/webp/format_constants.h"
#include "src/webp/mux.h"
#include "src/webp/mux_types.h"
#include "src/webp/types.h"

//------------------------------------------------------------------------------
// Life of a mux object.

static void MuxInit(WebPMux* const mux) {
  assert(mux != NULL);
  memset(mux, 0, sizeof(*mux));
  mux->canvas_width = 0;     // just to be explicit
  mux->canvas_height = 0;
}

WebPMux* WebPNewInternal(int version) {
  if (WEBP_ABI_IS_INCOMPATIBLE(version, WEBP_MUX_ABI_VERSION)) {
    return NULL;
  } else {
    WebPMux* const mux = (WebPMux*)WebPSafeMalloc(1ULL, sizeof(WebPMux));
    if (mux != NULL) MuxInit(mux);
    return mux;
  }
}

// Delete all images in 'wpi_list'.
static void DeleteAllImages(WebPMuxImage** const wpi_list) {
  while (*wpi_list != NULL) {
    *wpi_list = MuxImageDelete(*wpi_list);
  }
}

static void MuxRelease(WebPMux* const mux) {
  assert(mux != NULL);
  DeleteAllImages(&mux->images);
  ChunkListDelete(&mux->vp8x);
  ChunkListDelete(&mux->iccp);
  ChunkListDelete(&mux->anim);
  ChunkListDelete(&mux->exif);
  ChunkListDelete(&mux->xmp);
  ChunkListDelete(&mux->unknown);
}

void WebPMuxDelete(WebPMux* mux) {
  if (mux != NULL) {
    MuxRelease(mux);
    WebPSafeFree(mux);
  }
}

//------------------------------------------------------------------------------
// Helper method(s).

// Handy MACRO, makes MuxSet() very symmetric to MuxGet().
#define SWITCH_ID_LIST(INDEX, LIST)                                            \
  do {                                                                         \
    if (idx == (INDEX)) {                                                      \
      err = ChunkAssignData(&chunk, data, copy_data, tag);                     \
      if (err == WEBP_MUX_OK) {                                                \
        err = ChunkSetHead(&chunk, (LIST));                                    \
        if (err != WEBP_MUX_OK) ChunkRelease(&chunk);                          \
      }                                                                        \
      return err;                                                              \
    }                                                                          \
  } while (0)

static WebPMuxError MuxSet(WebPMux* const mux, uint32_t tag,
                           const WebPData* const data, int copy_data) {
  WebPChunk chunk;
  WebPMuxError err = WEBP_MUX_NOT_FOUND;
  const CHUNK_INDEX idx = ChunkGetIndexFromTag(tag);
  assert(mux != NULL);
  assert(!IsWPI(kChunks[idx].id));

  ChunkInit(&chunk);
  SWITCH_ID_LIST(IDX_VP8X,    &mux->vp8x);
  SWITCH_ID_LIST(IDX_ICCP,    &mux->iccp);
  SWITCH_ID_LIST(IDX_ANIM,    &mux->anim);
  SWITCH_ID_LIST(IDX_EXIF,    &mux->exif);
  SWITCH_ID_LIST(IDX_XMP,     &mux->xmp);
  SWITCH_ID_LIST(IDX_UNKNOWN, &mux->unknown);
  return err;
}
#undef SWITCH_ID_LIST

// Create data for frame given image data, offsets and duration.
static WebPMuxError CreateFrameData(
    int width, int height, const WebPMuxFrameInfo* const info,
    WebPData* const frame) {
  uint8_t* frame_bytes;
  const size_t frame_size = kChunks[IDX_ANMF].size;

  assert(width > 0 && height > 0 && info->duration >= 0);
  assert(info->dispose_method == (info->dispose_method & 1));
  // Note: assertion on upper bounds is done in PutLE24().

  frame_bytes = (uint8_t*)WebPSafeMalloc(1ULL, frame_size);
  if (frame_bytes == NULL) return WEBP_MUX_MEMORY_ERROR;

  PutLE24(frame_bytes + 0, info->x_offset / 2);
  PutLE24(frame_bytes + 3, info->y_offset / 2);

  PutLE24(frame_bytes + 6, width - 1);
  PutLE24(frame_bytes + 9, height - 1);
  PutLE24(frame_bytes + 12, info->duration);
  frame_bytes[15] =
      (info->blend_method == WEBP_MUX_NO_BLEND ? 2 : 0) |
      (info->dispose_method == WEBP_MUX_DISPOSE_BACKGROUND ? 1 : 0);

  frame->bytes = frame_bytes;
  frame->size = frame_size;
  return WEBP_MUX_OK;
}

// Outputs image data given a bitstream. The bitstream can either be a
// single-image WebP file or raw VP8/VP8L data.
// Also outputs 'is_lossless' to be true if the given bitstream is lossless.
static WebPMuxError GetImageData(const WebPData* const bitstream,
                                 WebPData* const image, WebPData* const alpha,
                                 int* const is_lossless) {
  WebPDataInit(alpha);  // Default: no alpha.
  if (bitstream->size < TAG_SIZE ||
      memcmp(bitstream->bytes, "RIFF", TAG_SIZE)) {
    // It is NOT webp file data. Return input data as is.
    *image = *bitstream;
  } else {
    // It is webp file data. Extract image data from it.
    const WebPMuxImage* wpi;
    WebPMux* const mux = WebPMuxCreate(bitstream, 0);
    if (mux == NULL) return WEBP_MUX_BAD_DATA;
    wpi = mux->images;
    assert(wpi != NULL && wpi->img != NULL);
    *image = wpi->img->data;
    if (wpi->alpha != NULL) {
      *alpha = wpi->alpha->data;
    }
    WebPMuxDelete(mux);
  }
  *is_lossless = VP8LCheckSignature(image->bytes, image->size);
  return WEBP_MUX_OK;
}

static WebPMuxError DeleteChunks(WebPChunk** chunk_list, uint32_t tag) {
  WebPMuxError err = WEBP_MUX_NOT_FOUND;
  assert(chunk_list);
  while (*chunk_list) {
    WebPChunk* const chunk = *chunk_list;
    if (chunk->tag == tag) {
      *chunk_list = ChunkDelete(chunk);
      err = WEBP_MUX_OK;
    } else {
      chunk_list = &chunk->next;
    }
  }
  return err;
}

static WebPMuxError MuxDeleteAllNamedData(WebPMux* const mux, uint32_t tag) {
  const WebPChunkId id = ChunkGetIdFromTag(tag);
  assert(mux != NULL);
  if (IsWPI(id)) return WEBP_MUX_INVALID_ARGUMENT;
  return DeleteChunks(MuxGetChunkListFromId(mux, id), tag);
}

//------------------------------------------------------------------------------
// Set API(s).

WebPMuxError WebPMuxSetChunk(WebPMux* mux, const char fourcc[4],
                             const WebPData* chunk_data, int copy_data) {
  uint32_t tag;
  WebPMuxError err;
  if (mux == NULL || fourcc == NULL || chunk_data == NULL ||
      chunk_data->bytes == NULL || chunk_data->size > MAX_CHUNK_PAYLOAD) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }
  tag = ChunkGetTagFromFourCC(fourcc);

  // Delete existing chunk(s) with the same 'fourcc'.
  err = MuxDeleteAllNamedData(mux, tag);
  if (err != WEBP_MUX_OK && err != WEBP_MUX_NOT_FOUND) return err;

  // Add the given chunk.
  return MuxSet(mux, tag, chunk_data, copy_data);
}

// Creates a chunk from given 'data' and sets it as 1st chunk in 'chunk_list'.
static WebPMuxError AddDataToChunkList(
    const WebPData* const data, int copy_data, uint32_t tag,
    WebPChunk** chunk_list) {
  WebPChunk chunk;
  WebPMuxError err;
  ChunkInit(&chunk);
  err = ChunkAssignData(&chunk, data, copy_data, tag);
  if (err != WEBP_MUX_OK) goto Err;
  err = ChunkSetHead(&chunk, chunk_list);
  if (err != WEBP_MUX_OK) goto Err;
  return WEBP_MUX_OK;
 Err:
  ChunkRelease(&chunk);
  return err;
}

// Extracts image & alpha data from the given bitstream and then sets wpi.alpha
// and wpi.img appropriately.
static WebPMuxError SetAlphaAndImageChunks(
    const WebPData* const bitstream, int copy_data, WebPMuxImage* const wpi) {
  int is_lossless = 0;
  WebPData image, alpha;
  WebPMuxError err = GetImageData(bitstream, &image, &alpha, &is_lossless);
  const int image_tag =
      is_lossless ? kChunks[IDX_VP8L].tag : kChunks[IDX_VP8].tag;
  if (err != WEBP_MUX_OK) return err;
  if (alpha.bytes != NULL) {
    err = AddDataToChunkList(&alpha, copy_data, kChunks[IDX_ALPHA].tag,
                             &wpi->alpha);
    if (err != WEBP_MUX_OK) return err;
  }
  err = AddDataToChunkList(&image, copy_data, image_tag, &wpi->img);
  if (err != WEBP_MUX_OK) return err;
  return MuxImageFinalize(wpi) ? WEBP_MUX_OK : WEBP_MUX_INVALID_ARGUMENT;
}

WebPMuxError WebPMuxSetImage(WebPMux* mux, const WebPData* bitstream,
                             int copy_data) {
  WebPMuxImage wpi;
  WebPMuxError err;

  if (mux == NULL || bitstream == NULL || bitstream->bytes == NULL ||
      bitstream->size > MAX_CHUNK_PAYLOAD) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  if (mux->images != NULL) {
    // Only one 'simple image' can be added in mux. So, remove present images.
    DeleteAllImages(&mux->images);
  }

  MuxImageInit(&wpi);
  err = SetAlphaAndImageChunks(bitstream, copy_data, &wpi);
  if (err != WEBP_MUX_OK) goto Err;

  // Add this WebPMuxImage to mux.
  err = MuxImagePush(&wpi, &mux->images);
  if (err != WEBP_MUX_OK) goto Err;

  // All is well.
  return WEBP_MUX_OK;

 Err:  // Something bad happened.
  MuxImageRelease(&wpi);
  return err;
}

WebPMuxError WebPMuxPushFrame(WebPMux* mux, const WebPMuxFrameInfo* info,
                              int copy_data) {
  WebPMuxImage wpi;
  WebPMuxError err;

  if (mux == NULL || info == NULL) return WEBP_MUX_INVALID_ARGUMENT;

  if (info->id != WEBP_CHUNK_ANMF) return WEBP_MUX_INVALID_ARGUMENT;

  if (info->bitstream.bytes == NULL ||
      info->bitstream.size > MAX_CHUNK_PAYLOAD) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  if (mux->images != NULL) {
    const WebPMuxImage* const image = mux->images;
    const uint32_t image_id = (image->header != NULL) ?
        ChunkGetIdFromTag(image->header->tag) : WEBP_CHUNK_IMAGE;
    if (image_id != info->id) {
      return WEBP_MUX_INVALID_ARGUMENT;  // Conflicting frame types.
    }
  }

  MuxImageInit(&wpi);
  err = SetAlphaAndImageChunks(&info->bitstream, copy_data, &wpi);
  if (err != WEBP_MUX_OK) goto Err;
  assert(wpi.img != NULL);  // As SetAlphaAndImageChunks() was successful.

  {
    WebPData frame;
    const uint32_t tag = kChunks[IDX_ANMF].tag;
    WebPMuxFrameInfo tmp = *info;
    tmp.x_offset &= ~1;  // Snap offsets to even.
    tmp.y_offset &= ~1;
    if (tmp.x_offset < 0 || tmp.x_offset >= MAX_POSITION_OFFSET ||
        tmp.y_offset < 0 || tmp.y_offset >= MAX_POSITION_OFFSET ||
        (tmp.duration < 0 || tmp.duration >= MAX_DURATION) ||
        tmp.dispose_method != (tmp.dispose_method & 1)) {
      err = WEBP_MUX_INVALID_ARGUMENT;
      goto Err;
    }
    err = CreateFrameData(wpi.width, wpi.height, &tmp, &frame);
    if (err != WEBP_MUX_OK) goto Err;
    // Add frame chunk (with copy_data = 1).
    err = AddDataToChunkList(&frame, 1, tag, &wpi.header);
    WebPDataClear(&frame);  // frame owned by wpi.header now.
    if (err != WEBP_MUX_OK) goto Err;
  }

  // Add this WebPMuxImage to mux.
  err = MuxImagePush(&wpi, &mux->images);
  if (err != WEBP_MUX_OK) goto Err;

  // All is well.
  return WEBP_MUX_OK;

 Err:  // Something bad happened.
  MuxImageRelease(&wpi);
  return err;
}

WebPMuxError WebPMuxSetAnimationParams(WebPMux* mux,
                                       const WebPMuxAnimParams* params) {
  WebPMuxError err;
  uint8_t data[ANIM_CHUNK_SIZE];
  const WebPData anim = { data, ANIM_CHUNK_SIZE };

  if (mux == NULL || params == NULL) return WEBP_MUX_INVALID_ARGUMENT;
  if (params->loop_count < 0 || params->loop_count >= MAX_LOOP_COUNT) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  // Delete any existing ANIM chunk(s).
  err = MuxDeleteAllNamedData(mux, kChunks[IDX_ANIM].tag);
  if (err != WEBP_MUX_OK && err != WEBP_MUX_NOT_FOUND) return err;

  // Set the animation parameters.
  PutLE32(data, params->bgcolor);
  PutLE16(data + 4, params->loop_count);
  return MuxSet(mux, kChunks[IDX_ANIM].tag, &anim, 1);
}

WebPMuxError WebPMuxSetCanvasSize(WebPMux* mux,
                                  int width, int height) {
  WebPMuxError err;
  if (mux == NULL) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }
  if (width < 0 || height < 0 ||
      width > MAX_CANVAS_SIZE || height > MAX_CANVAS_SIZE) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }
  if (width * (uint64_t)height >= MAX_IMAGE_AREA) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }
  if ((width * height) == 0 && (width | height) != 0) {
    // one of width / height is zero, but not both -> invalid!
    return WEBP_MUX_INVALID_ARGUMENT;
  }
  // If we already assembled a VP8X chunk, invalidate it.
  err = MuxDeleteAllNamedData(mux, kChunks[IDX_VP8X].tag);
  if (err != WEBP_MUX_OK && err != WEBP_MUX_NOT_FOUND) return err;

  mux->canvas_width = width;
  mux->canvas_height = height;
  return WEBP_MUX_OK;
}

//------------------------------------------------------------------------------
// Delete API(s).

WebPMuxError WebPMuxDeleteChunk(WebPMux* mux, const char fourcc[4]) {
  if (mux == NULL || fourcc == NULL) return WEBP_MUX_INVALID_ARGUMENT;
  return MuxDeleteAllNamedData(mux, ChunkGetTagFromFourCC(fourcc));
}

WebPMuxError WebPMuxDeleteFrame(WebPMux* mux, uint32_t nth) {
  if (mux == NULL) return WEBP_MUX_INVALID_ARGUMENT;
  return MuxImageDeleteNth(&mux->images, nth);
}

//------------------------------------------------------------------------------
// Assembly of the WebP RIFF file.

static WebPMuxError GetFrameInfo(
    const WebPChunk* const frame_chunk,
    int* const x_offset, int* const y_offset, int* const duration) {
  const WebPData* const data = &frame_chunk->data;
  const size_t expected_data_size = ANMF_CHUNK_SIZE;
  assert(frame_chunk->tag == kChunks[IDX_ANMF].tag);
  assert(frame_chunk != NULL);
  if (data->size != expected_data_size) return WEBP_MUX_INVALID_ARGUMENT;

  *x_offset = 2 * GetLE24(data->bytes + 0);
  *y_offset = 2 * GetLE24(data->bytes + 3);
  *duration = GetLE24(data->bytes + 12);
  return WEBP_MUX_OK;
}

static WebPMuxError GetImageInfo(const WebPMuxImage* const wpi,
                                 int* const x_offset, int* const y_offset,
                                 int* const duration,
                                 int* const width, int* const height) {
  const WebPChunk* const frame_chunk = wpi->header;
  WebPMuxError err;
  assert(wpi != NULL);
  assert(frame_chunk != NULL);

  // Get offsets and duration from ANMF chunk.
  err = GetFrameInfo(frame_chunk, x_offset, y_offset, duration);
  if (err != WEBP_MUX_OK) return err;

  // Get width and height from VP8/VP8L chunk.
  if (width != NULL) *width = wpi->width;
  if (height != NULL) *height = wpi->height;
  return WEBP_MUX_OK;
}

// Returns the tightest dimension for the canvas considering the image list.
static WebPMuxError GetAdjustedCanvasSize(const WebPMux* const mux,
                                          int* const width, int* const height) {
  WebPMuxImage* wpi = NULL;
  assert(mux != NULL);
  assert(width != NULL && height != NULL);

  wpi = mux->images;
  assert(wpi != NULL);
  assert(wpi->img != NULL);

  if (wpi->next != NULL) {
    int max_x = 0, max_y = 0;
    // if we have a chain of wpi's, header is necessarily set
    assert(wpi->header != NULL);
    // Aggregate the bounding box for animation frames.
    for (; wpi != NULL; wpi = wpi->next) {
      int x_offset = 0, y_offset = 0, duration = 0, w = 0, h = 0;
      const WebPMuxError err = GetImageInfo(wpi, &x_offset, &y_offset,
                                            &duration, &w, &h);
      const int max_x_pos = x_offset + w;
      const int max_y_pos = y_offset + h;
      if (err != WEBP_MUX_OK) return err;
      assert(x_offset < MAX_POSITION_OFFSET);
      assert(y_offset < MAX_POSITION_OFFSET);

      if (max_x_pos > max_x) max_x = max_x_pos;
      if (max_y_pos > max_y) max_y = max_y_pos;
    }
    *width = max_x;
    *height = max_y;
  } else {
    // For a single image, canvas dimensions are same as image dimensions.
    *width = wpi->width;
    *height = wpi->height;
  }
  return WEBP_MUX_OK;
}

// VP8X format:
// Total Size : 10,
// Flags  : 4 bytes,
// Width  : 3 bytes,
// Height : 3 bytes.
static WebPMuxError CreateVP8XChunk(WebPMux* const mux) {
  WebPMuxError err = WEBP_MUX_OK;
  uint32_t flags = 0;
  int width = 0;
  int height = 0;
  uint8_t data[VP8X_CHUNK_SIZE];
  const WebPData vp8x = { data, VP8X_CHUNK_SIZE };
  const WebPMuxImage* images = NULL;

  assert(mux != NULL);
  images = mux->images;  // First image.
  if (images == NULL || images->img == NULL ||
      images->img->data.bytes == NULL) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  // If VP8X chunk(s) is(are) already present, remove them (and later add new
  // VP8X chunk with updated flags).
  err = MuxDeleteAllNamedData(mux, kChunks[IDX_VP8X].tag);
  if (err != WEBP_MUX_OK && err != WEBP_MUX_NOT_FOUND) return err;

  // Set flags.
  if (mux->iccp != NULL && mux->iccp->data.bytes != NULL) {
    flags |= ICCP_FLAG;
  }
  if (mux->exif != NULL && mux->exif->data.bytes != NULL) {
    flags |= EXIF_FLAG;
  }
  if (mux->xmp != NULL && mux->xmp->data.bytes != NULL) {
    flags |= XMP_FLAG;
  }
  if (images->header != NULL) {
    if (images->header->tag == kChunks[IDX_ANMF].tag) {
      // This is an image with animation.
      flags |= ANIMATION_FLAG;
    }
  }
  if (MuxImageCount(images, WEBP_CHUNK_ALPHA) > 0) {
    flags |= ALPHA_FLAG;  // Some images have an alpha channel.
  }

  err = GetAdjustedCanvasSize(mux, &width, &height);
  if (err != WEBP_MUX_OK) return err;

  if (width <= 0 || height <= 0) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }
  if (width > MAX_CANVAS_SIZE || height > MAX_CANVAS_SIZE) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  if (mux->canvas_width != 0 || mux->canvas_height != 0) {
    if (width > mux->canvas_width || height > mux->canvas_height) {
      return WEBP_MUX_INVALID_ARGUMENT;
    }
    width = mux->canvas_width;
    height = mux->canvas_height;
  }

  if (flags == 0 && mux->unknown == NULL) {
    // For simple file format, VP8X chunk should not be added.
    return WEBP_MUX_OK;
  }

  if (MuxHasAlpha(images)) {
    // This means some frames explicitly/implicitly contain alpha.
    // Note: This 'flags' update must NOT be done for a lossless image
    // without a VP8X chunk!
    flags |= ALPHA_FLAG;
  }

  PutLE32(data + 0, flags);   // VP8X chunk flags.
  PutLE24(data + 4, width - 1);   // canvas width.
  PutLE24(data + 7, height - 1);  // canvas height.

  return MuxSet(mux, kChunks[IDX_VP8X].tag, &vp8x, 1);
}

// Cleans up 'mux' by removing any unnecessary chunks.
static WebPMuxError MuxCleanup(WebPMux* const mux) {
  int num_frames;
  int num_anim_chunks;

  // If we have an image with a single frame, and its rectangle
  // covers the whole canvas, convert it to a non-animated image
  // (to avoid writing ANMF chunk unnecessarily).
  WebPMuxError err = WebPMuxNumChunks(mux, kChunks[IDX_ANMF].id, &num_frames);
  if (err != WEBP_MUX_OK) return err;
  if (num_frames == 1) {
    WebPMuxImage* frame = NULL;
    err = MuxImageGetNth((const WebPMuxImage**)&mux->images, 1, &frame);
    if (err != WEBP_MUX_OK) return err;
    // We know that one frame does exist.
    assert(frame != NULL);
    if (frame->header != NULL &&
        ((mux->canvas_width == 0 && mux->canvas_height == 0) ||
         (frame->width == mux->canvas_width &&
          frame->height == mux->canvas_height))) {
      assert(frame->header->tag == kChunks[IDX_ANMF].tag);
      ChunkDelete(frame->header);  // Removes ANMF chunk.
      frame->header = NULL;
      num_frames = 0;
    }
  }
  // Remove ANIM chunk if this is a non-animated image.
  err = WebPMuxNumChunks(mux, kChunks[IDX_ANIM].id, &num_anim_chunks);
  if (err != WEBP_MUX_OK) return err;
  if (num_anim_chunks >= 1 && num_frames == 0) {
    err = MuxDeleteAllNamedData(mux, kChunks[IDX_ANIM].tag);
    if (err != WEBP_MUX_OK) return err;
  }
  return WEBP_MUX_OK;
}

// Total size of a list of images.
static size_t ImageListDiskSize(const WebPMuxImage* wpi_list) {
  size_t size = 0;
  while (wpi_list != NULL) {
    size += MuxImageDiskSize(wpi_list);
    wpi_list = wpi_list->next;
  }
  return size;
}

// Write out the given list of images into 'dst'.
static uint8_t* ImageListEmit(const WebPMuxImage* wpi_list, uint8_t* dst) {
  while (wpi_list != NULL) {
    dst = MuxImageEmit(wpi_list, dst);
    wpi_list = wpi_list->next;
  }
  return dst;
}

WebPMuxError WebPMuxAssemble(WebPMux* mux, WebPData* assembled_data) {
  size_t size = 0;
  uint8_t* data = NULL;
  uint8_t* dst = NULL;
  WebPMuxError err;

  if (assembled_data == NULL) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }
  // Clean up returned data, in case something goes wrong.
  memset(assembled_data, 0, sizeof(*assembled_data));

  if (mux == NULL) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  // Finalize mux.
  err = MuxCleanup(mux);
  if (err != WEBP_MUX_OK) return err;
  err = CreateVP8XChunk(mux);
  if (err != WEBP_MUX_OK) return err;

  // Allocate data.
  size = ChunkListDiskSize(mux->vp8x) + ChunkListDiskSize(mux->iccp)
       + ChunkListDiskSize(mux->anim) + ImageListDiskSize(mux->images)
       + ChunkListDiskSize(mux->exif) + ChunkListDiskSize(mux->xmp)
       + ChunkListDiskSize(mux->unknown) + RIFF_HEADER_SIZE;

  data = (uint8_t*)WebPSafeMalloc(1ULL, size);
  if (data == NULL) return WEBP_MUX_MEMORY_ERROR;

  // Emit header & chunks.
  dst = MuxEmitRiffHeader(data, size);
  dst = ChunkListEmit(mux->vp8x, dst);
  dst = ChunkListEmit(mux->iccp, dst);
  dst = ChunkListEmit(mux->anim, dst);
  dst = ImageListEmit(mux->images, dst);
  dst = ChunkListEmit(mux->exif, dst);
  dst = ChunkListEmit(mux->xmp, dst);
  dst = ChunkListEmit(mux->unknown, dst);
  assert(dst == data + size);

  // Validate mux.
  err = MuxValidate(mux);
  if (err != WEBP_MUX_OK) {
    WebPSafeFree(data);
    data = NULL;
    size = 0;
  }

  // Finalize data.
  assembled_data->bytes = data;
  assembled_data->size = size;

  return err;
}

//------------------------------------------------------------------------------

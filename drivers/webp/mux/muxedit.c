// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Set and delete APIs for mux.
//
// Authors: Urvang (urvang@google.com)
//          Vikas (vikasa@google.com)

#include <assert.h>
#include "./muxi.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------
// Life of a mux object.

static void MuxInit(WebPMux* const mux) {
  if (mux == NULL) return;
  memset(mux, 0, sizeof(*mux));
}

WebPMux* WebPNewInternal(int version) {
  if (WEBP_ABI_IS_INCOMPATIBLE(version, WEBP_MUX_ABI_VERSION)) {
    return NULL;
  } else {
    WebPMux* const mux = (WebPMux*)malloc(sizeof(WebPMux));
    // If mux is NULL MuxInit is a noop.
    MuxInit(mux);
    return mux;
  }
}

static void DeleteAllChunks(WebPChunk** const chunk_list) {
  while (*chunk_list) {
    *chunk_list = ChunkDelete(*chunk_list);
  }
}

static void MuxRelease(WebPMux* const mux) {
  if (mux == NULL) return;
  MuxImageDeleteAll(&mux->images_);
  DeleteAllChunks(&mux->vp8x_);
  DeleteAllChunks(&mux->iccp_);
  DeleteAllChunks(&mux->loop_);
  DeleteAllChunks(&mux->meta_);
  DeleteAllChunks(&mux->unknown_);
}

void WebPMuxDelete(WebPMux* mux) {
  // If mux is NULL MuxRelease is a noop.
  MuxRelease(mux);
  free(mux);
}

//------------------------------------------------------------------------------
// Helper method(s).

// Handy MACRO, makes MuxSet() very symmetric to MuxGet().
#define SWITCH_ID_LIST(INDEX, LIST)                                            \
  if (idx == (INDEX)) {                                                        \
    err = ChunkAssignData(&chunk, data, copy_data, kChunks[(INDEX)].tag);      \
    if (err == WEBP_MUX_OK) {                                                  \
      err = ChunkSetNth(&chunk, (LIST), nth);                                  \
    }                                                                          \
    return err;                                                                \
  }

static WebPMuxError MuxSet(WebPMux* const mux, CHUNK_INDEX idx, uint32_t nth,
                           const WebPData* const data, int copy_data) {
  WebPChunk chunk;
  WebPMuxError err = WEBP_MUX_NOT_FOUND;
  assert(mux != NULL);
  assert(!IsWPI(kChunks[idx].id));

  ChunkInit(&chunk);
  SWITCH_ID_LIST(IDX_VP8X, &mux->vp8x_);
  SWITCH_ID_LIST(IDX_ICCP, &mux->iccp_);
  SWITCH_ID_LIST(IDX_LOOP, &mux->loop_);
  SWITCH_ID_LIST(IDX_META, &mux->meta_);
  if (idx == IDX_UNKNOWN && data->size_ > TAG_SIZE) {
    // For raw-data unknown chunk, the first four bytes should be the tag to be
    // used for the chunk.
    const WebPData tmp = { data->bytes_ + TAG_SIZE, data->size_ - TAG_SIZE };
    err = ChunkAssignData(&chunk, &tmp, copy_data, GetLE32(data->bytes_ + 0));
    if (err == WEBP_MUX_OK)
      err = ChunkSetNth(&chunk, &mux->unknown_, nth);
  }
  return err;
}
#undef SWITCH_ID_LIST

static WebPMuxError MuxAddChunk(WebPMux* const mux, uint32_t nth, uint32_t tag,
                                const uint8_t* data, size_t size,
                                int copy_data) {
  const CHUNK_INDEX idx = ChunkGetIndexFromTag(tag);
  const WebPData chunk_data = { data, size };
  assert(mux != NULL);
  assert(size <= MAX_CHUNK_PAYLOAD);
  assert(idx != IDX_NIL);
  return MuxSet(mux, idx, nth, &chunk_data, copy_data);
}

// Create data for frame/tile given image data, offsets and duration.
static WebPMuxError CreateFrameTileData(const WebPData* const image,
                                        int x_offset, int y_offset,
                                        int duration, int is_lossless,
                                        int is_frame,
                                        WebPData* const frame_tile) {
  int width;
  int height;
  uint8_t* frame_tile_bytes;
  const size_t frame_tile_size = kChunks[is_frame ? IDX_FRAME : IDX_TILE].size;

  const int ok = is_lossless ?
      VP8LGetInfo(image->bytes_, image->size_, &width, &height, NULL) :
      VP8GetInfo(image->bytes_, image->size_, image->size_, &width, &height);
  if (!ok) return WEBP_MUX_INVALID_ARGUMENT;

  assert(width > 0 && height > 0 && duration > 0);
  // Note: assertion on upper bounds is done in PutLE24().

  frame_tile_bytes = (uint8_t*)malloc(frame_tile_size);
  if (frame_tile_bytes == NULL) return WEBP_MUX_MEMORY_ERROR;

  PutLE24(frame_tile_bytes + 0, x_offset / 2);
  PutLE24(frame_tile_bytes + 3, y_offset / 2);

  if (is_frame) {
    PutLE24(frame_tile_bytes + 6, width - 1);
    PutLE24(frame_tile_bytes + 9, height - 1);
    PutLE24(frame_tile_bytes + 12, duration - 1);
  }

  frame_tile->bytes_ = frame_tile_bytes;
  frame_tile->size_ = frame_tile_size;
  return WEBP_MUX_OK;
}

// Outputs image data given a bitstream. The bitstream can either be a
// single-image WebP file or raw VP8/VP8L data.
// Also outputs 'is_lossless' to be true if the given bitstream is lossless.
static WebPMuxError GetImageData(const WebPData* const bitstream,
                                 WebPData* const image, WebPData* const alpha,
                                 int* const is_lossless) {
  WebPDataInit(alpha);  // Default: no alpha.
  if (bitstream->size_ < TAG_SIZE ||
      memcmp(bitstream->bytes_, "RIFF", TAG_SIZE)) {
    // It is NOT webp file data. Return input data as is.
    *image = *bitstream;
  } else {
    // It is webp file data. Extract image data from it.
    const WebPMuxImage* wpi;
    WebPMux* const mux = WebPMuxCreate(bitstream, 0);
    if (mux == NULL) return WEBP_MUX_BAD_DATA;
    wpi = mux->images_;
    assert(wpi != NULL && wpi->img_ != NULL);
    *image = wpi->img_->data_;
    if (wpi->alpha_ != NULL) {
      *alpha = wpi->alpha_->data_;
    }
    WebPMuxDelete(mux);
  }
  *is_lossless = VP8LCheckSignature(image->bytes_, image->size_);
  return WEBP_MUX_OK;
}

static WebPMuxError DeleteChunks(WebPChunk** chunk_list, uint32_t tag) {
  WebPMuxError err = WEBP_MUX_NOT_FOUND;
  assert(chunk_list);
  while (*chunk_list) {
    WebPChunk* const chunk = *chunk_list;
    if (chunk->tag_ == tag) {
      *chunk_list = ChunkDelete(chunk);
      err = WEBP_MUX_OK;
    } else {
      chunk_list = &chunk->next_;
    }
  }
  return err;
}

static WebPMuxError MuxDeleteAllNamedData(WebPMux* const mux, CHUNK_INDEX idx) {
  const WebPChunkId id = kChunks[idx].id;
  WebPChunk** chunk_list;

  if (mux == NULL) return WEBP_MUX_INVALID_ARGUMENT;
  if (IsWPI(id)) return WEBP_MUX_INVALID_ARGUMENT;

  chunk_list = MuxGetChunkListFromId(mux, id);
  if (chunk_list == NULL) return WEBP_MUX_INVALID_ARGUMENT;

  return DeleteChunks(chunk_list, kChunks[idx].tag);
}

static WebPMuxError DeleteLoopCount(WebPMux* const mux) {
  return MuxDeleteAllNamedData(mux, IDX_LOOP);
}

//------------------------------------------------------------------------------
// Set API(s).

WebPMuxError WebPMuxSetImage(WebPMux* mux,
                             const WebPData* bitstream, int copy_data) {
  WebPMuxError err;
  WebPChunk chunk;
  WebPMuxImage wpi;
  WebPData image;
  WebPData alpha;
  int is_lossless;
  int image_tag;

  if (mux == NULL || bitstream == NULL || bitstream->bytes_ == NULL ||
      bitstream->size_ > MAX_CHUNK_PAYLOAD) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  // If given data is for a whole webp file,
  // extract only the VP8/VP8L data from it.
  err = GetImageData(bitstream, &image, &alpha, &is_lossless);
  if (err != WEBP_MUX_OK) return err;
  image_tag = is_lossless ? kChunks[IDX_VP8L].tag : kChunks[IDX_VP8].tag;

  // Delete the existing images.
  MuxImageDeleteAll(&mux->images_);

  MuxImageInit(&wpi);

  if (alpha.bytes_ != NULL) {  // Add alpha chunk.
    ChunkInit(&chunk);
    err = ChunkAssignData(&chunk, &alpha, copy_data, kChunks[IDX_ALPHA].tag);
    if (err != WEBP_MUX_OK) goto Err;
    err = ChunkSetNth(&chunk, &wpi.alpha_, 1);
    if (err != WEBP_MUX_OK) goto Err;
  }

  // Add image chunk.
  ChunkInit(&chunk);
  err = ChunkAssignData(&chunk, &image, copy_data, image_tag);
  if (err != WEBP_MUX_OK) goto Err;
  err = ChunkSetNth(&chunk, &wpi.img_, 1);
  if (err != WEBP_MUX_OK) goto Err;

  // Add this image to mux.
  err = MuxImagePush(&wpi, &mux->images_);
  if (err != WEBP_MUX_OK) goto Err;

  // All OK.
  return WEBP_MUX_OK;

 Err:
  // Something bad happened.
  ChunkRelease(&chunk);
  MuxImageRelease(&wpi);
  return err;
}

WebPMuxError WebPMuxSetMetadata(WebPMux* mux, const WebPData* metadata,
                                int copy_data) {
  WebPMuxError err;

  if (mux == NULL || metadata == NULL || metadata->bytes_ == NULL ||
      metadata->size_ > MAX_CHUNK_PAYLOAD) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  // Delete the existing metadata chunk(s).
  err = WebPMuxDeleteMetadata(mux);
  if (err != WEBP_MUX_OK && err != WEBP_MUX_NOT_FOUND) return err;

  // Add the given metadata chunk.
  return MuxSet(mux, IDX_META, 1, metadata, copy_data);
}

WebPMuxError WebPMuxSetColorProfile(WebPMux* mux, const WebPData* color_profile,
                                    int copy_data) {
  WebPMuxError err;

  if (mux == NULL || color_profile == NULL || color_profile->bytes_ == NULL ||
      color_profile->size_ > MAX_CHUNK_PAYLOAD) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  // Delete the existing ICCP chunk(s).
  err = WebPMuxDeleteColorProfile(mux);
  if (err != WEBP_MUX_OK && err != WEBP_MUX_NOT_FOUND) return err;

  // Add the given ICCP chunk.
  return MuxSet(mux, IDX_ICCP, 1, color_profile, copy_data);
}

WebPMuxError WebPMuxSetLoopCount(WebPMux* mux, int loop_count) {
  WebPMuxError err;
  uint8_t* data = NULL;

  if (mux == NULL) return WEBP_MUX_INVALID_ARGUMENT;
  if (loop_count >= MAX_LOOP_COUNT) return WEBP_MUX_INVALID_ARGUMENT;

  // Delete the existing LOOP chunk(s).
  err = DeleteLoopCount(mux);
  if (err != WEBP_MUX_OK && err != WEBP_MUX_NOT_FOUND) return err;

  // Add the given loop count.
  data = (uint8_t*)malloc(kChunks[IDX_LOOP].size);
  if (data == NULL) return WEBP_MUX_MEMORY_ERROR;

  PutLE16(data, loop_count);
  err = MuxAddChunk(mux, 1, kChunks[IDX_LOOP].tag, data,
                    kChunks[IDX_LOOP].size, 1);
  free(data);
  return err;
}

static WebPMuxError MuxPushFrameTileInternal(
    WebPMux* const mux, const WebPData* const bitstream, int x_offset,
    int y_offset, int duration, int copy_data, uint32_t tag) {
  WebPChunk chunk;
  WebPData image;
  WebPData alpha;
  WebPMuxImage wpi;
  WebPMuxError err;
  WebPData frame_tile;
  const int is_frame = (tag == kChunks[IDX_FRAME].tag) ? 1 : 0;
  int is_lossless;
  int image_tag;

  // Sanity checks.
  if (mux == NULL || bitstream == NULL || bitstream->bytes_ == NULL ||
      bitstream->size_ > MAX_CHUNK_PAYLOAD) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }
  if (x_offset < 0 || x_offset >= MAX_POSITION_OFFSET ||
      y_offset < 0 || y_offset >= MAX_POSITION_OFFSET ||
      duration <= 0 || duration > MAX_DURATION) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  // Snap offsets to even positions.
  x_offset &= ~1;
  y_offset &= ~1;

  // If given data is for a whole webp file,
  // extract only the VP8/VP8L data from it.
  err = GetImageData(bitstream, &image, &alpha, &is_lossless);
  if (err != WEBP_MUX_OK) return err;
  image_tag = is_lossless ? kChunks[IDX_VP8L].tag : kChunks[IDX_VP8].tag;

  WebPDataInit(&frame_tile);
  ChunkInit(&chunk);
  MuxImageInit(&wpi);

  if (alpha.bytes_ != NULL) {
    // Add alpha chunk.
    err = ChunkAssignData(&chunk, &alpha, copy_data, kChunks[IDX_ALPHA].tag);
    if (err != WEBP_MUX_OK) goto Err;
    err = ChunkSetNth(&chunk, &wpi.alpha_, 1);
    if (err != WEBP_MUX_OK) goto Err;
    ChunkInit(&chunk);  // chunk owned by wpi.alpha_ now.
  }

  // Add image chunk.
  err = ChunkAssignData(&chunk, &image, copy_data, image_tag);
  if (err != WEBP_MUX_OK) goto Err;
  err = ChunkSetNth(&chunk, &wpi.img_, 1);
  if (err != WEBP_MUX_OK) goto Err;
  ChunkInit(&chunk);  // chunk owned by wpi.img_ now.

  // Create frame/tile data.
  err = CreateFrameTileData(&image, x_offset, y_offset, duration, is_lossless,
                            is_frame, &frame_tile);
  if (err != WEBP_MUX_OK) goto Err;

  // Add frame/tile chunk (with copy_data = 1).
  err = ChunkAssignData(&chunk, &frame_tile, 1, tag);
  if (err != WEBP_MUX_OK) goto Err;
  WebPDataClear(&frame_tile);
  err = ChunkSetNth(&chunk, &wpi.header_, 1);
  if (err != WEBP_MUX_OK) goto Err;
  ChunkInit(&chunk);  // chunk owned by wpi.header_ now.

  // Add this WebPMuxImage to mux.
  err = MuxImagePush(&wpi, &mux->images_);
  if (err != WEBP_MUX_OK) goto Err;

  // All is well.
  return WEBP_MUX_OK;

 Err:  // Something bad happened.
  WebPDataClear(&frame_tile);
  ChunkRelease(&chunk);
  MuxImageRelease(&wpi);
  return err;
}

WebPMuxError WebPMuxPushFrame(WebPMux* mux, const WebPData* bitstream,
                              int x_offset, int y_offset,
                              int duration, int copy_data) {
  return MuxPushFrameTileInternal(mux, bitstream, x_offset, y_offset,
                                  duration, copy_data, kChunks[IDX_FRAME].tag);
}

WebPMuxError WebPMuxPushTile(WebPMux* mux, const WebPData* bitstream,
                             int x_offset, int y_offset,
                             int copy_data) {
  return MuxPushFrameTileInternal(mux, bitstream, x_offset, y_offset,
                                  1 /* unused duration */, copy_data,
                                  kChunks[IDX_TILE].tag);
}

//------------------------------------------------------------------------------
// Delete API(s).

WebPMuxError WebPMuxDeleteImage(WebPMux* mux) {
  WebPMuxError err;

  if (mux == NULL) return WEBP_MUX_INVALID_ARGUMENT;

  err = MuxValidateForImage(mux);
  if (err != WEBP_MUX_OK) return err;

  // All well, delete image.
  MuxImageDeleteAll(&mux->images_);
  return WEBP_MUX_OK;
}

WebPMuxError WebPMuxDeleteMetadata(WebPMux* mux) {
  return MuxDeleteAllNamedData(mux, IDX_META);
}

WebPMuxError WebPMuxDeleteColorProfile(WebPMux* mux) {
  return MuxDeleteAllNamedData(mux, IDX_ICCP);
}

static WebPMuxError DeleteFrameTileInternal(WebPMux* const mux, uint32_t nth,
                                            CHUNK_INDEX idx) {
  const WebPChunkId id = kChunks[idx].id;
  if (mux == NULL) return WEBP_MUX_INVALID_ARGUMENT;

  assert(idx == IDX_FRAME || idx == IDX_TILE);
  return MuxImageDeleteNth(&mux->images_, nth, id);
}

WebPMuxError WebPMuxDeleteFrame(WebPMux* mux, uint32_t nth) {
  return DeleteFrameTileInternal(mux, nth, IDX_FRAME);
}

WebPMuxError WebPMuxDeleteTile(WebPMux* mux, uint32_t nth) {
  return DeleteFrameTileInternal(mux, nth, IDX_TILE);
}

//------------------------------------------------------------------------------
// Assembly of the WebP RIFF file.

static WebPMuxError GetFrameTileInfo(const WebPChunk* const frame_tile_chunk,
                                     int* const x_offset, int* const y_offset,
                                     int* const duration) {
  const uint32_t tag = frame_tile_chunk->tag_;
  const int is_frame = (tag == kChunks[IDX_FRAME].tag);
  const WebPData* const data = &frame_tile_chunk->data_;
  const size_t expected_data_size =
      is_frame ? FRAME_CHUNK_SIZE : TILE_CHUNK_SIZE;
  assert(frame_tile_chunk != NULL);
  assert(tag == kChunks[IDX_FRAME].tag || tag ==  kChunks[IDX_TILE].tag);
  if (data->size_ != expected_data_size) return WEBP_MUX_INVALID_ARGUMENT;

  *x_offset = 2 * GetLE24(data->bytes_ + 0);
  *y_offset = 2 * GetLE24(data->bytes_ + 3);
  if (is_frame) *duration = 1 + GetLE24(data->bytes_ + 12);
  return WEBP_MUX_OK;
}

WebPMuxError MuxGetImageWidthHeight(const WebPChunk* const image_chunk,
                                    int* const width, int* const height) {
  const uint32_t tag = image_chunk->tag_;
  const WebPData* const data = &image_chunk->data_;
  int w, h;
  int ok;
  assert(image_chunk != NULL);
  assert(tag == kChunks[IDX_VP8].tag || tag ==  kChunks[IDX_VP8L].tag);
  ok = (tag == kChunks[IDX_VP8].tag) ?
      VP8GetInfo(data->bytes_, data->size_, data->size_, &w, &h) :
      VP8LGetInfo(data->bytes_, data->size_, &w, &h, NULL);
  if (ok) {
    *width = w;
    *height = h;
    return WEBP_MUX_OK;
  } else {
    return WEBP_MUX_BAD_DATA;
  }
}

static WebPMuxError GetImageInfo(const WebPMuxImage* const wpi,
                                 int* const x_offset, int* const y_offset,
                                 int* const duration,
                                 int* const width, int* const height) {
  const WebPChunk* const image_chunk = wpi->img_;
  const WebPChunk* const frame_tile_chunk = wpi->header_;

  // Get offsets and duration from FRM/TILE chunk.
  const WebPMuxError err =
      GetFrameTileInfo(frame_tile_chunk, x_offset, y_offset, duration);
  if (err != WEBP_MUX_OK) return err;

  // Get width and height from VP8/VP8L chunk.
  return MuxGetImageWidthHeight(image_chunk, width, height);
}

static WebPMuxError GetImageCanvasWidthHeight(
    const WebPMux* const mux, uint32_t flags,
    int* const width, int* const height) {
  WebPMuxImage* wpi = NULL;
  assert(mux != NULL);
  assert(width != NULL && height != NULL);

  wpi = mux->images_;
  assert(wpi != NULL);
  assert(wpi->img_ != NULL);

  if (wpi->next_) {
    int max_x = 0;
    int max_y = 0;
    int64_t image_area = 0;
    // Aggregate the bounding box for animation frames & tiled images.
    for (; wpi != NULL; wpi = wpi->next_) {
      int x_offset, y_offset, duration, w, h;
      const WebPMuxError err = GetImageInfo(wpi, &x_offset, &y_offset,
                                            &duration, &w, &h);
      const int max_x_pos = x_offset + w;
      const int max_y_pos = y_offset + h;
      if (err != WEBP_MUX_OK) return err;
      assert(x_offset < MAX_POSITION_OFFSET);
      assert(y_offset < MAX_POSITION_OFFSET);

      if (max_x_pos > max_x) max_x = max_x_pos;
      if (max_y_pos > max_y) max_y = max_y_pos;
      image_area += w * h;
    }
    *width = max_x;
    *height = max_y;
    // Crude check to validate that there are no image overlaps/holes for tile
    // images. Check that the aggregated image area for individual tiles exactly
    // matches the image area of the constructed canvas. However, the area-match
    // is necessary but not sufficient condition.
    if ((flags & TILE_FLAG) && (image_area != (max_x * max_y))) {
      *width = 0;
      *height = 0;
      return WEBP_MUX_INVALID_ARGUMENT;
    }
  } else {
    // For a single image, extract the width & height from VP8/VP8L image-data.
    int w, h;
    const WebPChunk* const image_chunk = wpi->img_;
    const WebPMuxError err = MuxGetImageWidthHeight(image_chunk, &w, &h);
    if (err != WEBP_MUX_OK) return err;
    *width = w;
    *height = h;
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
  const size_t data_size = VP8X_CHUNK_SIZE;
  const WebPMuxImage* images = NULL;

  assert(mux != NULL);
  images = mux->images_;  // First image.
  if (images == NULL || images->img_ == NULL ||
      images->img_->data_.bytes_ == NULL) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  // If VP8X chunk(s) is(are) already present, remove them (and later add new
  // VP8X chunk with updated flags).
  err = MuxDeleteAllNamedData(mux, IDX_VP8X);
  if (err != WEBP_MUX_OK && err != WEBP_MUX_NOT_FOUND) return err;

  // Set flags.
  if (mux->iccp_ != NULL && mux->iccp_->data_.bytes_ != NULL) {
    flags |= ICCP_FLAG;
  }

  if (mux->meta_ != NULL && mux->meta_->data_.bytes_ != NULL) {
    flags |= META_FLAG;
  }

  if (images->header_ != NULL) {
    if (images->header_->tag_ == kChunks[IDX_TILE].tag) {
      // This is a tiled image.
      flags |= TILE_FLAG;
    } else if (images->header_->tag_ == kChunks[IDX_FRAME].tag) {
      // This is an image with animation.
      flags |= ANIMATION_FLAG;
    }
  }

  if (MuxImageCount(images, WEBP_CHUNK_ALPHA) > 0) {
    flags |= ALPHA_FLAG;  // Some images have an alpha channel.
  }

  if (flags == 0) {
    // For Simple Image, VP8X chunk should not be added.
    return WEBP_MUX_OK;
  }

  err = GetImageCanvasWidthHeight(mux, flags, &width, &height);
  if (err != WEBP_MUX_OK) return err;

  if (width <= 0 || height <= 0) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }
  if (width > MAX_CANVAS_SIZE || height > MAX_CANVAS_SIZE) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  if (MuxHasLosslessImages(images)) {
    // We have a file with a VP8X chunk having some lossless images.
    // As lossless images implicitly contain alpha, force ALPHA_FLAG to be true.
    // Note: This 'flags' update must NOT be done for a lossless image
    // without a VP8X chunk!
    flags |= ALPHA_FLAG;
  }

  PutLE32(data + 0, flags);   // VP8X chunk flags.
  PutLE24(data + 4, width - 1);   // canvas width.
  PutLE24(data + 7, height - 1);  // canvas height.

  err = MuxAddChunk(mux, 1, kChunks[IDX_VP8X].tag, data, data_size, 1);
  return err;
}

WebPMuxError WebPMuxAssemble(WebPMux* mux, WebPData* assembled_data) {
  size_t size = 0;
  uint8_t* data = NULL;
  uint8_t* dst = NULL;
  int num_frames;
  int num_loop_chunks;
  WebPMuxError err;

  if (mux == NULL || assembled_data == NULL) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  // Remove LOOP chunk if unnecessary.
  err = WebPMuxNumChunks(mux, kChunks[IDX_LOOP].id, &num_loop_chunks);
  if (err != WEBP_MUX_OK) return err;
  if (num_loop_chunks >= 1) {
    err = WebPMuxNumChunks(mux, kChunks[IDX_FRAME].id, &num_frames);
    if (err != WEBP_MUX_OK) return err;
    if (num_frames == 0) {
      err = DeleteLoopCount(mux);
      if (err != WEBP_MUX_OK) return err;
    }
  }

  // Create VP8X chunk.
  err = CreateVP8XChunk(mux);
  if (err != WEBP_MUX_OK) return err;

  // Allocate data.
  size = ChunksListDiskSize(mux->vp8x_) + ChunksListDiskSize(mux->iccp_)
       + ChunksListDiskSize(mux->loop_) + MuxImageListDiskSize(mux->images_)
       + ChunksListDiskSize(mux->meta_) + ChunksListDiskSize(mux->unknown_)
       + RIFF_HEADER_SIZE;

  data = (uint8_t*)malloc(size);
  if (data == NULL) return WEBP_MUX_MEMORY_ERROR;

  // Emit header & chunks.
  dst = MuxEmitRiffHeader(data, size);
  dst = ChunkListEmit(mux->vp8x_, dst);
  dst = ChunkListEmit(mux->iccp_, dst);
  dst = ChunkListEmit(mux->loop_, dst);
  dst = MuxImageListEmit(mux->images_, dst);
  dst = ChunkListEmit(mux->meta_, dst);
  dst = ChunkListEmit(mux->unknown_, dst);
  assert(dst == data + size);

  // Validate mux.
  err = MuxValidate(mux);
  if (err != WEBP_MUX_OK) {
    free(data);
    data = NULL;
    size = 0;
  }

  // Finalize.
  assembled_data->bytes_ = data;
  assembled_data->size_ = size;

  return err;
}

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

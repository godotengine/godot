// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Read APIs for mux.
//
// Authors: Urvang (urvang@google.com)
//          Vikas (vikasa@google.com)

#include <assert.h>
#include "./muxi.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------
// Helper method(s).

// Handy MACRO.
#define SWITCH_ID_LIST(INDEX, LIST)                                           \
  if (idx == (INDEX)) {                                                       \
    const WebPChunk* const chunk = ChunkSearchList((LIST), nth,               \
                                                   kChunks[(INDEX)].tag);     \
    if (chunk) {                                                              \
      *data = chunk->data_;                                                   \
      return WEBP_MUX_OK;                                                     \
    } else {                                                                  \
      return WEBP_MUX_NOT_FOUND;                                              \
    }                                                                         \
  }

static WebPMuxError MuxGet(const WebPMux* const mux, CHUNK_INDEX idx,
                           uint32_t nth, WebPData* const data) {
  assert(mux != NULL);
  assert(!IsWPI(kChunks[idx].id));
  WebPDataInit(data);

  SWITCH_ID_LIST(IDX_VP8X, mux->vp8x_);
  SWITCH_ID_LIST(IDX_ICCP, mux->iccp_);
  SWITCH_ID_LIST(IDX_LOOP, mux->loop_);
  SWITCH_ID_LIST(IDX_META, mux->meta_);
  SWITCH_ID_LIST(IDX_UNKNOWN, mux->unknown_);
  return WEBP_MUX_NOT_FOUND;
}
#undef SWITCH_ID_LIST

// Fill the chunk with the given data (includes chunk header bytes), after some
// verifications.
static WebPMuxError ChunkVerifyAndAssignData(WebPChunk* chunk,
                                             const uint8_t* data,
                                             size_t data_size, size_t riff_size,
                                             int copy_data) {
  uint32_t chunk_size;
  WebPData chunk_data;

  // Sanity checks.
  if (data_size < TAG_SIZE) return WEBP_MUX_NOT_ENOUGH_DATA;
  chunk_size = GetLE32(data + TAG_SIZE);

  {
    const size_t chunk_disk_size = SizeWithPadding(chunk_size);
    if (chunk_disk_size > riff_size) return WEBP_MUX_BAD_DATA;
    if (chunk_disk_size > data_size) return WEBP_MUX_NOT_ENOUGH_DATA;
  }

  // Data assignment.
  chunk_data.bytes_ = data + CHUNK_HEADER_SIZE;
  chunk_data.size_ = chunk_size;
  return ChunkAssignData(chunk, &chunk_data, copy_data, GetLE32(data + 0));
}

//------------------------------------------------------------------------------
// Create a mux object from WebP-RIFF data.

WebPMux* WebPMuxCreateInternal(const WebPData* bitstream, int copy_data,
                               int version) {
  size_t riff_size;
  uint32_t tag;
  const uint8_t* end;
  WebPMux* mux = NULL;
  WebPMuxImage* wpi = NULL;
  const uint8_t* data;
  size_t size;
  WebPChunk chunk;
  ChunkInit(&chunk);

  // Sanity checks.
  if (WEBP_ABI_IS_INCOMPATIBLE(version, WEBP_MUX_ABI_VERSION)) {
    return NULL;  // version mismatch
  }
  if (bitstream == NULL) return NULL;

  data = bitstream->bytes_;
  size = bitstream->size_;

  if (data == NULL) return NULL;
  if (size < RIFF_HEADER_SIZE) return NULL;
  if (GetLE32(data + 0) != MKFOURCC('R', 'I', 'F', 'F') ||
      GetLE32(data + CHUNK_HEADER_SIZE) != MKFOURCC('W', 'E', 'B', 'P')) {
    return NULL;
  }

  mux = WebPMuxNew();
  if (mux == NULL) return NULL;

  if (size < RIFF_HEADER_SIZE + TAG_SIZE) goto Err;

  tag = GetLE32(data + RIFF_HEADER_SIZE);
  if (tag != kChunks[IDX_VP8].tag &&
      tag != kChunks[IDX_VP8L].tag &&
      tag != kChunks[IDX_VP8X].tag) {
    goto Err;  // First chunk should be VP8, VP8L or VP8X.
  }

  riff_size = SizeWithPadding(GetLE32(data + TAG_SIZE));
  if (riff_size > MAX_CHUNK_PAYLOAD || riff_size > size) {
    goto Err;
  } else {
    if (riff_size < size) {  // Redundant data after last chunk.
      size = riff_size;  // To make sure we don't read any data beyond mux_size.
    }
  }

  end = data + size;
  data += RIFF_HEADER_SIZE;
  size -= RIFF_HEADER_SIZE;

  wpi = (WebPMuxImage*)malloc(sizeof(*wpi));
  if (wpi == NULL) goto Err;
  MuxImageInit(wpi);

  // Loop over chunks.
  while (data != end) {
    WebPChunkId id;
    WebPMuxError err;

    err = ChunkVerifyAndAssignData(&chunk, data, size, riff_size, copy_data);
    if (err != WEBP_MUX_OK) goto Err;

    id = ChunkGetIdFromTag(chunk.tag_);

    if (IsWPI(id)) {  // An image chunk (frame/tile/alpha/vp8).
      WebPChunk** wpi_chunk_ptr =
          MuxImageGetListFromId(wpi, id);  // Image chunk to set.
      assert(wpi_chunk_ptr != NULL);
      if (*wpi_chunk_ptr != NULL) goto Err;  // Consecutive alpha chunks or
                                             // consecutive frame/tile chunks.
      if (ChunkSetNth(&chunk, wpi_chunk_ptr, 1) != WEBP_MUX_OK) goto Err;
      if (id == WEBP_CHUNK_IMAGE) {
        wpi->is_partial_ = 0;  // wpi is completely filled.
        // Add this to mux->images_ list.
        if (MuxImagePush(wpi, &mux->images_) != WEBP_MUX_OK) goto Err;
        MuxImageInit(wpi);  // Reset for reading next image.
      } else {
        wpi->is_partial_ = 1;  // wpi is only partially filled.
      }
    } else {  // A non-image chunk.
      WebPChunk** chunk_list;
      if (wpi->is_partial_) goto Err;  // Encountered a non-image chunk before
                                       // getting all chunks of an image.
      chunk_list = MuxGetChunkListFromId(mux, id);  // List to add this chunk.
      if (chunk_list == NULL) chunk_list = &mux->unknown_;
      if (ChunkSetNth(&chunk, chunk_list, 0) != WEBP_MUX_OK) goto Err;
    }
    {
      const size_t data_size = ChunkDiskSize(&chunk);
      data += data_size;
      size -= data_size;
    }
    ChunkInit(&chunk);
  }

  // Validate mux if complete.
  if (MuxValidate(mux) != WEBP_MUX_OK) goto Err;

  MuxImageDelete(wpi);
  return mux;  // All OK;

 Err:  // Something bad happened.
  ChunkRelease(&chunk);
  MuxImageDelete(wpi);
  WebPMuxDelete(mux);
  return NULL;
}

//------------------------------------------------------------------------------
// Get API(s).

WebPMuxError WebPMuxGetFeatures(const WebPMux* mux, uint32_t* flags) {
  WebPData data;
  WebPMuxError err;

  if (mux == NULL || flags == NULL) return WEBP_MUX_INVALID_ARGUMENT;
  *flags = 0;

  // Check if VP8X chunk is present.
  err = MuxGet(mux, IDX_VP8X, 1, &data);
  if (err == WEBP_MUX_NOT_FOUND) {
    // Check if VP8/VP8L chunk is present.
    err = WebPMuxGetImage(mux, &data);
    WebPDataClear(&data);
    return err;
  } else if (err != WEBP_MUX_OK) {
    return err;
  }

  if (data.size_ < CHUNK_SIZE_BYTES) return WEBP_MUX_BAD_DATA;

  // All OK. Fill up flags.
  *flags = GetLE32(data.bytes_);
  return WEBP_MUX_OK;
}

static uint8_t* EmitVP8XChunk(uint8_t* const dst, int width,
                              int height, uint32_t flags) {
  const size_t vp8x_size = CHUNK_HEADER_SIZE + VP8X_CHUNK_SIZE;
  assert(width >= 1 && height >= 1);
  assert(width <= MAX_CANVAS_SIZE && height <= MAX_CANVAS_SIZE);
  assert(width * (uint64_t)height < MAX_IMAGE_AREA);
  PutLE32(dst, MKFOURCC('V', 'P', '8', 'X'));
  PutLE32(dst + TAG_SIZE, VP8X_CHUNK_SIZE);
  PutLE32(dst + CHUNK_HEADER_SIZE, flags);
  PutLE24(dst + CHUNK_HEADER_SIZE + 4, width - 1);
  PutLE24(dst + CHUNK_HEADER_SIZE + 7, height - 1);
  return dst + vp8x_size;
}

// Assemble a single image WebP bitstream from 'wpi'.
static WebPMuxError SynthesizeBitstream(WebPMuxImage* const wpi,
                                        WebPData* const bitstream) {
  uint8_t* dst;

  // Allocate data.
  const int need_vp8x = (wpi->alpha_ != NULL);
  const size_t vp8x_size = need_vp8x ? CHUNK_HEADER_SIZE + VP8X_CHUNK_SIZE : 0;
  const size_t alpha_size = need_vp8x ? ChunkDiskSize(wpi->alpha_) : 0;
  // Note: No need to output FRM/TILE chunk for a single image.
  const size_t size = RIFF_HEADER_SIZE + vp8x_size + alpha_size +
                      ChunkDiskSize(wpi->img_);
  uint8_t* const data = (uint8_t*)malloc(size);
  if (data == NULL) return WEBP_MUX_MEMORY_ERROR;

  // Main RIFF header.
  dst = MuxEmitRiffHeader(data, size);

  if (need_vp8x) {
    int w, h;
    WebPMuxError err;
    assert(wpi->img_ != NULL);
    err = MuxGetImageWidthHeight(wpi->img_, &w, &h);
    if (err != WEBP_MUX_OK) {
      free(data);
      return err;
    }
    dst = EmitVP8XChunk(dst, w, h, ALPHA_FLAG);  // VP8X.
    dst = ChunkListEmit(wpi->alpha_, dst);       // ALPH.
  }

  // Bitstream.
  dst = ChunkListEmit(wpi->img_, dst);
  assert(dst == data + size);

  // Output.
  bitstream->bytes_ = data;
  bitstream->size_ = size;
  return WEBP_MUX_OK;
}

WebPMuxError WebPMuxGetImage(const WebPMux* mux, WebPData* bitstream) {
  WebPMuxError err;
  WebPMuxImage* wpi = NULL;

  if (mux == NULL || bitstream == NULL) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  err = MuxValidateForImage(mux);
  if (err != WEBP_MUX_OK) return err;

  // All well. Get the image.
  err = MuxImageGetNth((const WebPMuxImage**)&mux->images_, 1, WEBP_CHUNK_IMAGE,
                       &wpi);
  assert(err == WEBP_MUX_OK);  // Already tested above.

  return SynthesizeBitstream(wpi, bitstream);
}

WebPMuxError WebPMuxGetMetadata(const WebPMux* mux, WebPData* metadata) {
  if (mux == NULL || metadata == NULL) return WEBP_MUX_INVALID_ARGUMENT;
  return MuxGet(mux, IDX_META, 1, metadata);
}

WebPMuxError WebPMuxGetColorProfile(const WebPMux* mux,
                                    WebPData* color_profile) {
  if (mux == NULL || color_profile == NULL) return WEBP_MUX_INVALID_ARGUMENT;
  return MuxGet(mux, IDX_ICCP, 1, color_profile);
}

WebPMuxError WebPMuxGetLoopCount(const WebPMux* mux, int* loop_count) {
  WebPData image;
  WebPMuxError err;

  if (mux == NULL || loop_count == NULL) return WEBP_MUX_INVALID_ARGUMENT;

  err = MuxGet(mux, IDX_LOOP, 1, &image);
  if (err != WEBP_MUX_OK) return err;
  if (image.size_ < kChunks[WEBP_CHUNK_LOOP].size) return WEBP_MUX_BAD_DATA;
  *loop_count = GetLE16(image.bytes_);

  return WEBP_MUX_OK;
}

static WebPMuxError MuxGetFrameTileInternal(
    const WebPMux* const mux, uint32_t nth, WebPData* const bitstream,
    int* const x_offset, int* const y_offset, int* const duration,
    uint32_t tag) {
  const WebPData* frame_tile_data;
  WebPMuxError err;
  WebPMuxImage* wpi;

  const int is_frame = (tag == kChunks[WEBP_CHUNK_FRAME].tag) ? 1 : 0;
  const CHUNK_INDEX idx = is_frame ? IDX_FRAME : IDX_TILE;
  const WebPChunkId id = kChunks[idx].id;

  if (mux == NULL || bitstream == NULL ||
      x_offset == NULL || y_offset == NULL || (is_frame && duration == NULL)) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  // Get the nth WebPMuxImage.
  err = MuxImageGetNth((const WebPMuxImage**)&mux->images_, nth, id, &wpi);
  if (err != WEBP_MUX_OK) return err;

  // Get frame chunk.
  assert(wpi->header_ != NULL);  // As MuxImageGetNth() already checked header_.
  frame_tile_data = &wpi->header_->data_;

  if (frame_tile_data->size_ < kChunks[idx].size) return WEBP_MUX_BAD_DATA;
  *x_offset = 2 * GetLE24(frame_tile_data->bytes_ + 0);
  *y_offset = 2 * GetLE24(frame_tile_data->bytes_ + 3);
  if (is_frame) *duration = 1 + GetLE24(frame_tile_data->bytes_ + 12);

  return SynthesizeBitstream(wpi, bitstream);
}

WebPMuxError WebPMuxGetFrame(const WebPMux* mux, uint32_t nth,
                             WebPData* bitstream,
                             int* x_offset, int* y_offset, int* duration) {
  return MuxGetFrameTileInternal(mux, nth, bitstream, x_offset, y_offset,
                                 duration, kChunks[IDX_FRAME].tag);
}

WebPMuxError WebPMuxGetTile(const WebPMux* mux, uint32_t nth,
                            WebPData* bitstream,
                            int* x_offset, int* y_offset) {
  return MuxGetFrameTileInternal(mux, nth, bitstream, x_offset, y_offset, NULL,
                                 kChunks[IDX_TILE].tag);
}

// Get chunk index from chunk id. Returns IDX_NIL if not found.
static CHUNK_INDEX ChunkGetIndexFromId(WebPChunkId id) {
  int i;
  for (i = 0; kChunks[i].id != WEBP_CHUNK_NIL; ++i) {
    if (id == kChunks[i].id) return i;
  }
  return IDX_NIL;
}

// Count number of chunks matching 'tag' in the 'chunk_list'.
// If tag == NIL_TAG, any tag will be matched.
static int CountChunks(const WebPChunk* const chunk_list, uint32_t tag) {
  int count = 0;
  const WebPChunk* current;
  for (current = chunk_list; current != NULL; current = current->next_) {
    if (tag == NIL_TAG || current->tag_ == tag) {
      count++;  // Count chunks whose tags match.
    }
  }
  return count;
}

WebPMuxError WebPMuxNumChunks(const WebPMux* mux,
                              WebPChunkId id, int* num_elements) {
  if (mux == NULL || num_elements == NULL) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  if (IsWPI(id)) {
    *num_elements = MuxImageCount(mux->images_, id);
  } else {
    WebPChunk* const* chunk_list = MuxGetChunkListFromId(mux, id);
    if (chunk_list == NULL) {
      *num_elements = 0;
    } else {
      const CHUNK_INDEX idx = ChunkGetIndexFromId(id);
      *num_elements = CountChunks(*chunk_list, kChunks[idx].tag);
    }
  }

  return WEBP_MUX_OK;
}

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

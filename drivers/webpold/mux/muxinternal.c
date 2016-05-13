// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Internal objects and utils for mux.
//
// Authors: Urvang (urvang@google.com)
//          Vikas (vikasa@google.com)

#include <assert.h>
#include "./muxi.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#define UNDEFINED_CHUNK_SIZE (-1)

const ChunkInfo kChunks[] = {
  { MKFOURCC('V', 'P', '8', 'X'),  WEBP_CHUNK_VP8X,    VP8X_CHUNK_SIZE },
  { MKFOURCC('I', 'C', 'C', 'P'),  WEBP_CHUNK_ICCP,    UNDEFINED_CHUNK_SIZE },
  { MKFOURCC('L', 'O', 'O', 'P'),  WEBP_CHUNK_LOOP,    LOOP_CHUNK_SIZE },
  { MKFOURCC('F', 'R', 'M', ' '),  WEBP_CHUNK_FRAME,   FRAME_CHUNK_SIZE },
  { MKFOURCC('T', 'I', 'L', 'E'),  WEBP_CHUNK_TILE,    TILE_CHUNK_SIZE },
  { MKFOURCC('A', 'L', 'P', 'H'),  WEBP_CHUNK_ALPHA,   UNDEFINED_CHUNK_SIZE },
  { MKFOURCC('V', 'P', '8', ' '),  WEBP_CHUNK_IMAGE,   UNDEFINED_CHUNK_SIZE },
  { MKFOURCC('V', 'P', '8', 'L'),  WEBP_CHUNK_IMAGE,   UNDEFINED_CHUNK_SIZE },
  { MKFOURCC('M', 'E', 'T', 'A'),  WEBP_CHUNK_META,    UNDEFINED_CHUNK_SIZE },
  { MKFOURCC('U', 'N', 'K', 'N'),  WEBP_CHUNK_UNKNOWN, UNDEFINED_CHUNK_SIZE },

  { NIL_TAG,                    WEBP_CHUNK_NIL,     UNDEFINED_CHUNK_SIZE }
};

//------------------------------------------------------------------------------
// Life of a chunk object.

void ChunkInit(WebPChunk* const chunk) {
  assert(chunk);
  memset(chunk, 0, sizeof(*chunk));
  chunk->tag_ = NIL_TAG;
}

WebPChunk* ChunkRelease(WebPChunk* const chunk) {
  WebPChunk* next;
  if (chunk == NULL) return NULL;
  if (chunk->owner_) {
    WebPDataClear(&chunk->data_);
  }
  next = chunk->next_;
  ChunkInit(chunk);
  return next;
}

//------------------------------------------------------------------------------
// Chunk misc methods.

CHUNK_INDEX ChunkGetIndexFromTag(uint32_t tag) {
  int i;
  for (i = 0; kChunks[i].tag != NIL_TAG; ++i) {
    if (tag == kChunks[i].tag) return i;
  }
  return IDX_NIL;
}

WebPChunkId ChunkGetIdFromTag(uint32_t tag) {
  int i;
  for (i = 0; kChunks[i].tag != NIL_TAG; ++i) {
    if (tag == kChunks[i].tag) return kChunks[i].id;
  }
  return WEBP_CHUNK_NIL;
}

//------------------------------------------------------------------------------
// Chunk search methods.

// Returns next chunk in the chunk list with the given tag.
static WebPChunk* ChunkSearchNextInList(WebPChunk* chunk, uint32_t tag) {
  while (chunk && chunk->tag_ != tag) {
    chunk = chunk->next_;
  }
  return chunk;
}

WebPChunk* ChunkSearchList(WebPChunk* first, uint32_t nth, uint32_t tag) {
  uint32_t iter = nth;
  first = ChunkSearchNextInList(first, tag);
  if (!first) return NULL;

  while (--iter != 0) {
    WebPChunk* next_chunk = ChunkSearchNextInList(first->next_, tag);
    if (next_chunk == NULL) break;
    first = next_chunk;
  }
  return ((nth > 0) && (iter > 0)) ? NULL : first;
}

// Outputs a pointer to 'prev_chunk->next_',
//   where 'prev_chunk' is the pointer to the chunk at position (nth - 1).
// Returns 1 if nth chunk was found, 0 otherwise.
static int ChunkSearchListToSet(WebPChunk** chunk_list, uint32_t nth,
                                WebPChunk*** const location) {
  uint32_t count = 0;
  assert(chunk_list);
  *location = chunk_list;

  while (*chunk_list) {
    WebPChunk* const cur_chunk = *chunk_list;
    ++count;
    if (count == nth) return 1;  // Found.
    chunk_list = &cur_chunk->next_;
    *location = chunk_list;
  }

  // *chunk_list is ok to be NULL if adding at last location.
  return (nth == 0 || (count == nth - 1)) ? 1 : 0;
}

//------------------------------------------------------------------------------
// Chunk writer methods.

WebPMuxError ChunkAssignData(WebPChunk* chunk, const WebPData* const data,
                             int copy_data, uint32_t tag) {
  // For internally allocated chunks, always copy data & make it owner of data.
  if (tag == kChunks[IDX_VP8X].tag || tag == kChunks[IDX_LOOP].tag) {
    copy_data = 1;
  }

  ChunkRelease(chunk);

  if (data != NULL) {
    if (copy_data) {
      // Copy data.
      chunk->data_.bytes_ = (uint8_t*)malloc(data->size_);
      if (chunk->data_.bytes_ == NULL) return WEBP_MUX_MEMORY_ERROR;
      memcpy((uint8_t*)chunk->data_.bytes_, data->bytes_, data->size_);
      chunk->data_.size_ = data->size_;

      // Chunk is owner of data.
      chunk->owner_ = 1;
    } else {
      // Don't copy data.
      chunk->data_ = *data;
    }
  }

  chunk->tag_ = tag;

  return WEBP_MUX_OK;
}

WebPMuxError ChunkSetNth(const WebPChunk* chunk, WebPChunk** chunk_list,
                         uint32_t nth) {
  WebPChunk* new_chunk;

  if (!ChunkSearchListToSet(chunk_list, nth, &chunk_list)) {
    return WEBP_MUX_NOT_FOUND;
  }

  new_chunk = (WebPChunk*)malloc(sizeof(*new_chunk));
  if (new_chunk == NULL) return WEBP_MUX_MEMORY_ERROR;
  *new_chunk = *chunk;
  new_chunk->next_ = *chunk_list;
  *chunk_list = new_chunk;
  return WEBP_MUX_OK;
}

//------------------------------------------------------------------------------
// Chunk deletion method(s).

WebPChunk* ChunkDelete(WebPChunk* const chunk) {
  WebPChunk* const next = ChunkRelease(chunk);
  free(chunk);
  return next;
}

//------------------------------------------------------------------------------
// Chunk serialization methods.

size_t ChunksListDiskSize(const WebPChunk* chunk_list) {
  size_t size = 0;
  while (chunk_list) {
    size += ChunkDiskSize(chunk_list);
    chunk_list = chunk_list->next_;
  }
  return size;
}

static uint8_t* ChunkEmit(const WebPChunk* const chunk, uint8_t* dst) {
  const size_t chunk_size = chunk->data_.size_;
  assert(chunk);
  assert(chunk->tag_ != NIL_TAG);
  PutLE32(dst + 0, chunk->tag_);
  PutLE32(dst + TAG_SIZE, (uint32_t)chunk_size);
  assert(chunk_size == (uint32_t)chunk_size);
  memcpy(dst + CHUNK_HEADER_SIZE, chunk->data_.bytes_, chunk_size);
  if (chunk_size & 1)
    dst[CHUNK_HEADER_SIZE + chunk_size] = 0;  // Add padding.
  return dst + ChunkDiskSize(chunk);
}

uint8_t* ChunkListEmit(const WebPChunk* chunk_list, uint8_t* dst) {
  while (chunk_list) {
    dst = ChunkEmit(chunk_list, dst);
    chunk_list = chunk_list->next_;
  }
  return dst;
}

//------------------------------------------------------------------------------
// Manipulation of a WebPData object.

void WebPDataInit(WebPData* webp_data) {
  if (webp_data != NULL) {
    memset(webp_data, 0, sizeof(*webp_data));
  }
}

void WebPDataClear(WebPData* webp_data) {
  if (webp_data != NULL) {
    free((void*)webp_data->bytes_);
    WebPDataInit(webp_data);
  }
}

int WebPDataCopy(const WebPData* src, WebPData* dst) {
  if (src == NULL || dst == NULL) return 0;

  WebPDataInit(dst);
  if (src->bytes_ != NULL && src->size_ != 0) {
    dst->bytes_ = (uint8_t*)malloc(src->size_);
    if (dst->bytes_ == NULL) return 0;
    memcpy((void*)dst->bytes_, src->bytes_, src->size_);
    dst->size_ = src->size_;
  }
  return 1;
}

//------------------------------------------------------------------------------
// Life of a MuxImage object.

void MuxImageInit(WebPMuxImage* const wpi) {
  assert(wpi);
  memset(wpi, 0, sizeof(*wpi));
}

WebPMuxImage* MuxImageRelease(WebPMuxImage* const wpi) {
  WebPMuxImage* next;
  if (wpi == NULL) return NULL;
  ChunkDelete(wpi->header_);
  ChunkDelete(wpi->alpha_);
  ChunkDelete(wpi->img_);

  next = wpi->next_;
  MuxImageInit(wpi);
  return next;
}

//------------------------------------------------------------------------------
// MuxImage search methods.

int MuxImageCount(const WebPMuxImage* wpi_list, WebPChunkId id) {
  int count = 0;
  const WebPMuxImage* current;
  for (current = wpi_list; current != NULL; current = current->next_) {
    const WebPChunk* const wpi_chunk = *MuxImageGetListFromId(current, id);
    if (wpi_chunk != NULL) {
      const WebPChunkId wpi_chunk_id = ChunkGetIdFromTag(wpi_chunk->tag_);
      if (wpi_chunk_id == id) ++count;
    }
  }
  return count;
}

// Outputs a pointer to 'prev_wpi->next_',
//   where 'prev_wpi' is the pointer to the image at position (nth - 1).
// Returns 1 if nth image with given id was found, 0 otherwise.
static int SearchImageToGetOrDelete(WebPMuxImage** wpi_list, uint32_t nth,
                                    WebPChunkId id,
                                    WebPMuxImage*** const location) {
  uint32_t count = 0;
  assert(wpi_list);
  *location = wpi_list;

  // Search makes sense only for the following.
  assert(id == WEBP_CHUNK_FRAME || id == WEBP_CHUNK_TILE ||
         id == WEBP_CHUNK_IMAGE);
  assert(id != WEBP_CHUNK_IMAGE || nth == 1);

  if (nth == 0) {
    nth = MuxImageCount(*wpi_list, id);
    if (nth == 0) return 0;  // Not found.
  }

  while (*wpi_list) {
    WebPMuxImage* const cur_wpi = *wpi_list;
    const WebPChunk* const wpi_chunk = *MuxImageGetListFromId(cur_wpi, id);
    if (wpi_chunk != NULL) {
      const WebPChunkId wpi_chunk_id = ChunkGetIdFromTag(wpi_chunk->tag_);
      if (wpi_chunk_id == id) {
        ++count;
        if (count == nth) return 1;  // Found.
      }
    }
    wpi_list = &cur_wpi->next_;
    *location = wpi_list;
  }
  return 0;  // Not found.
}

//------------------------------------------------------------------------------
// MuxImage writer methods.

WebPMuxError MuxImagePush(const WebPMuxImage* wpi, WebPMuxImage** wpi_list) {
  WebPMuxImage* new_wpi;

  while (*wpi_list != NULL) {
    WebPMuxImage* const cur_wpi = *wpi_list;
    if (cur_wpi->next_ == NULL) break;
    wpi_list = &cur_wpi->next_;
  }

  new_wpi = (WebPMuxImage*)malloc(sizeof(*new_wpi));
  if (new_wpi == NULL) return WEBP_MUX_MEMORY_ERROR;
  *new_wpi = *wpi;
  new_wpi->next_ = NULL;

  if (*wpi_list != NULL) {
    (*wpi_list)->next_ = new_wpi;
  } else {
    *wpi_list = new_wpi;
  }
  return WEBP_MUX_OK;
}

//------------------------------------------------------------------------------
// MuxImage deletion methods.

WebPMuxImage* MuxImageDelete(WebPMuxImage* const wpi) {
  // Delete the components of wpi. If wpi is NULL this is a noop.
  WebPMuxImage* const next = MuxImageRelease(wpi);
  free(wpi);
  return next;
}

void MuxImageDeleteAll(WebPMuxImage** const wpi_list) {
  while (*wpi_list) {
    *wpi_list = MuxImageDelete(*wpi_list);
  }
}

WebPMuxError MuxImageDeleteNth(WebPMuxImage** wpi_list, uint32_t nth,
                               WebPChunkId id) {
  assert(wpi_list);
  if (!SearchImageToGetOrDelete(wpi_list, nth, id, &wpi_list)) {
    return WEBP_MUX_NOT_FOUND;
  }
  *wpi_list = MuxImageDelete(*wpi_list);
  return WEBP_MUX_OK;
}

//------------------------------------------------------------------------------
// MuxImage reader methods.

WebPMuxError MuxImageGetNth(const WebPMuxImage** wpi_list, uint32_t nth,
                            WebPChunkId id, WebPMuxImage** wpi) {
  assert(wpi_list);
  assert(wpi);
  if (!SearchImageToGetOrDelete((WebPMuxImage**)wpi_list, nth, id,
                                (WebPMuxImage***)&wpi_list)) {
    return WEBP_MUX_NOT_FOUND;
  }
  *wpi = (WebPMuxImage*)*wpi_list;
  return WEBP_MUX_OK;
}

//------------------------------------------------------------------------------
// MuxImage serialization methods.

// Size of an image.
size_t MuxImageDiskSize(const WebPMuxImage* const wpi) {
  size_t size = 0;
  if (wpi->header_ != NULL) size += ChunkDiskSize(wpi->header_);
  if (wpi->alpha_ != NULL) size += ChunkDiskSize(wpi->alpha_);
  if (wpi->img_ != NULL) size += ChunkDiskSize(wpi->img_);
  return size;
}

size_t MuxImageListDiskSize(const WebPMuxImage* wpi_list) {
  size_t size = 0;
  while (wpi_list) {
    size += MuxImageDiskSize(wpi_list);
    wpi_list = wpi_list->next_;
  }
  return size;
}

uint8_t* MuxImageEmit(const WebPMuxImage* const wpi, uint8_t* dst) {
  // Ordering of chunks to be emitted is strictly as follows:
  // 1. Frame/Tile chunk (if present).
  // 2. Alpha chunk (if present).
  // 3. VP8/VP8L chunk.
  assert(wpi);
  if (wpi->header_ != NULL) dst = ChunkEmit(wpi->header_, dst);
  if (wpi->alpha_ != NULL) dst = ChunkEmit(wpi->alpha_, dst);
  if (wpi->img_ != NULL) dst = ChunkEmit(wpi->img_, dst);
  return dst;
}

uint8_t* MuxImageListEmit(const WebPMuxImage* wpi_list, uint8_t* dst) {
  while (wpi_list) {
    dst = MuxImageEmit(wpi_list, dst);
    wpi_list = wpi_list->next_;
  }
  return dst;
}

//------------------------------------------------------------------------------
// Helper methods for mux.

int MuxHasLosslessImages(const WebPMuxImage* images) {
  while (images != NULL) {
    assert(images->img_ != NULL);
    if (images->img_->tag_ == kChunks[IDX_VP8L].tag) {
      return 1;
    }
    images = images->next_;
  }
  return 0;
}

uint8_t* MuxEmitRiffHeader(uint8_t* const data, size_t size) {
  PutLE32(data + 0, MKFOURCC('R', 'I', 'F', 'F'));
  PutLE32(data + TAG_SIZE, (uint32_t)size - CHUNK_HEADER_SIZE);
  assert(size == (uint32_t)size);
  PutLE32(data + TAG_SIZE + CHUNK_SIZE_BYTES, MKFOURCC('W', 'E', 'B', 'P'));
  return data + RIFF_HEADER_SIZE;
}

WebPChunk** MuxGetChunkListFromId(const WebPMux* mux, WebPChunkId id) {
  assert(mux != NULL);
  switch(id) {
    case WEBP_CHUNK_VP8X:    return (WebPChunk**)&mux->vp8x_;
    case WEBP_CHUNK_ICCP:    return (WebPChunk**)&mux->iccp_;
    case WEBP_CHUNK_LOOP:    return (WebPChunk**)&mux->loop_;
    case WEBP_CHUNK_META:    return (WebPChunk**)&mux->meta_;
    case WEBP_CHUNK_UNKNOWN: return (WebPChunk**)&mux->unknown_;
    default: return NULL;
  }
}

WebPMuxError MuxValidateForImage(const WebPMux* const mux) {
  const int num_images = MuxImageCount(mux->images_, WEBP_CHUNK_IMAGE);
  const int num_frames = MuxImageCount(mux->images_, WEBP_CHUNK_FRAME);
  const int num_tiles  = MuxImageCount(mux->images_, WEBP_CHUNK_TILE);

  if (num_images == 0) {
    // No images in mux.
    return WEBP_MUX_NOT_FOUND;
  } else if (num_images == 1 && num_frames == 0 && num_tiles == 0) {
    // Valid case (single image).
    return WEBP_MUX_OK;
  } else {
    // Frame/Tile case OR an invalid mux.
    return WEBP_MUX_INVALID_ARGUMENT;
  }
}

static int IsNotCompatible(int feature, int num_items) {
  return (feature != 0) != (num_items > 0);
}

#define NO_FLAG 0

// Test basic constraints:
// retrieval, maximum number of chunks by index (use -1 to skip)
// and feature incompatibility (use NO_FLAG to skip).
// On success returns WEBP_MUX_OK and stores the chunk count in *num.
static WebPMuxError ValidateChunk(const WebPMux* const mux, CHUNK_INDEX idx,
                                  WebPFeatureFlags feature,
                                  WebPFeatureFlags vp8x_flags,
                                  int max, int* num) {
  const WebPMuxError err =
      WebPMuxNumChunks(mux, kChunks[idx].id, num);
  if (err != WEBP_MUX_OK) return err;
  if (max > -1 && *num > max) return WEBP_MUX_INVALID_ARGUMENT;
  if (feature != NO_FLAG && IsNotCompatible(vp8x_flags & feature, *num)) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }
  return WEBP_MUX_OK;
}

WebPMuxError MuxValidate(const WebPMux* const mux) {
  int num_iccp;
  int num_meta;
  int num_loop_chunks;
  int num_frames;
  int num_tiles;
  int num_vp8x;
  int num_images;
  int num_alpha;
  uint32_t flags;
  WebPMuxError err;

  // Verify mux is not NULL.
  if (mux == NULL) return WEBP_MUX_INVALID_ARGUMENT;

  // Verify mux has at least one image.
  if (mux->images_ == NULL) return WEBP_MUX_INVALID_ARGUMENT;

  err = WebPMuxGetFeatures(mux, &flags);
  if (err != WEBP_MUX_OK) return err;

  // At most one color profile chunk.
  err = ValidateChunk(mux, IDX_ICCP, ICCP_FLAG, flags, 1, &num_iccp);
  if (err != WEBP_MUX_OK) return err;

  // At most one XMP metadata.
  err = ValidateChunk(mux, IDX_META, META_FLAG, flags, 1, &num_meta);
  if (err != WEBP_MUX_OK) return err;

  // Animation: ANIMATION_FLAG, loop chunk and frame chunk(s) are consistent.
  // At most one loop chunk.
  err = ValidateChunk(mux, IDX_LOOP, NO_FLAG, flags, 1, &num_loop_chunks);
  if (err != WEBP_MUX_OK) return err;
  err = ValidateChunk(mux, IDX_FRAME, NO_FLAG, flags, -1, &num_frames);
  if (err != WEBP_MUX_OK) return err;

  {
    const int has_animation = !!(flags & ANIMATION_FLAG);
    if (has_animation && (num_loop_chunks == 0 || num_frames == 0)) {
      return WEBP_MUX_INVALID_ARGUMENT;
    }
    if (!has_animation && (num_loop_chunks == 1 || num_frames > 0)) {
      return WEBP_MUX_INVALID_ARGUMENT;
    }
  }

  // Tiling: TILE_FLAG and tile chunk(s) are consistent.
  err = ValidateChunk(mux, IDX_TILE, TILE_FLAG, flags, -1, &num_tiles);
  if (err != WEBP_MUX_OK) return err;

  // Verify either VP8X chunk is present OR there is only one elem in
  // mux->images_.
  err = ValidateChunk(mux, IDX_VP8X, NO_FLAG, flags, 1, &num_vp8x);
  if (err != WEBP_MUX_OK) return err;
  err = ValidateChunk(mux, IDX_VP8, NO_FLAG, flags, -1, &num_images);
  if (err != WEBP_MUX_OK) return err;
  if (num_vp8x == 0 && num_images != 1) return WEBP_MUX_INVALID_ARGUMENT;

  // ALPHA_FLAG & alpha chunk(s) are consistent.
  if (num_vp8x > 0 && MuxHasLosslessImages(mux->images_)) {
    // Special case: we have a VP8X chunk as well as some lossless images.
    if (!(flags & ALPHA_FLAG)) return WEBP_MUX_INVALID_ARGUMENT;
  } else {
    err = ValidateChunk(mux, IDX_ALPHA, ALPHA_FLAG, flags, -1, &num_alpha);
    if (err != WEBP_MUX_OK) return err;
  }

  // num_tiles & num_images are consistent.
  if (num_tiles > 0 && num_images != num_tiles) {
    return WEBP_MUX_INVALID_ARGUMENT;
  }

  return WEBP_MUX_OK;
}

#undef NO_FLAG

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

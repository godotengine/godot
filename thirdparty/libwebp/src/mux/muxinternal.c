// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Internal objects and utils for mux.
//
// Authors: Urvang (urvang@google.com)
//          Vikas (vikasa@google.com)

#include <assert.h>
#include "src/mux/muxi.h"
#include "src/utils/utils.h"

#define UNDEFINED_CHUNK_SIZE ((uint32_t)(-1))

const ChunkInfo kChunks[] = {
  { MKFOURCC('V', 'P', '8', 'X'),  WEBP_CHUNK_VP8X,    VP8X_CHUNK_SIZE },
  { MKFOURCC('I', 'C', 'C', 'P'),  WEBP_CHUNK_ICCP,    UNDEFINED_CHUNK_SIZE },
  { MKFOURCC('A', 'N', 'I', 'M'),  WEBP_CHUNK_ANIM,    ANIM_CHUNK_SIZE },
  { MKFOURCC('A', 'N', 'M', 'F'),  WEBP_CHUNK_ANMF,    ANMF_CHUNK_SIZE },
  { MKFOURCC('A', 'L', 'P', 'H'),  WEBP_CHUNK_ALPHA,   UNDEFINED_CHUNK_SIZE },
  { MKFOURCC('V', 'P', '8', ' '),  WEBP_CHUNK_IMAGE,   UNDEFINED_CHUNK_SIZE },
  { MKFOURCC('V', 'P', '8', 'L'),  WEBP_CHUNK_IMAGE,   UNDEFINED_CHUNK_SIZE },
  { MKFOURCC('E', 'X', 'I', 'F'),  WEBP_CHUNK_EXIF,    UNDEFINED_CHUNK_SIZE },
  { MKFOURCC('X', 'M', 'P', ' '),  WEBP_CHUNK_XMP,     UNDEFINED_CHUNK_SIZE },
  { NIL_TAG,                       WEBP_CHUNK_UNKNOWN, UNDEFINED_CHUNK_SIZE },

  { NIL_TAG,                       WEBP_CHUNK_NIL,     UNDEFINED_CHUNK_SIZE }
};

//------------------------------------------------------------------------------

int WebPGetMuxVersion(void) {
  return (MUX_MAJ_VERSION << 16) | (MUX_MIN_VERSION << 8) | MUX_REV_VERSION;
}

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
    if (tag == kChunks[i].tag) return (CHUNK_INDEX)i;
  }
  return IDX_UNKNOWN;
}

WebPChunkId ChunkGetIdFromTag(uint32_t tag) {
  int i;
  for (i = 0; kChunks[i].tag != NIL_TAG; ++i) {
    if (tag == kChunks[i].tag) return kChunks[i].id;
  }
  return WEBP_CHUNK_UNKNOWN;
}

uint32_t ChunkGetTagFromFourCC(const char fourcc[4]) {
  return MKFOURCC(fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
}

CHUNK_INDEX ChunkGetIndexFromFourCC(const char fourcc[4]) {
  const uint32_t tag = ChunkGetTagFromFourCC(fourcc);
  return ChunkGetIndexFromTag(tag);
}

//------------------------------------------------------------------------------
// Chunk search methods.

// Returns next chunk in the chunk list with the given tag.
static WebPChunk* ChunkSearchNextInList(WebPChunk* chunk, uint32_t tag) {
  while (chunk != NULL && chunk->tag_ != tag) {
    chunk = chunk->next_;
  }
  return chunk;
}

WebPChunk* ChunkSearchList(WebPChunk* first, uint32_t nth, uint32_t tag) {
  uint32_t iter = nth;
  first = ChunkSearchNextInList(first, tag);
  if (first == NULL) return NULL;

  while (--iter != 0) {
    WebPChunk* next_chunk = ChunkSearchNextInList(first->next_, tag);
    if (next_chunk == NULL) break;
    first = next_chunk;
  }
  return ((nth > 0) && (iter > 0)) ? NULL : first;
}

// Outputs a pointer to 'prev_chunk->next_',
//   where 'prev_chunk' is the pointer to the chunk at position (nth - 1).
// Returns true if nth chunk was found.
static int ChunkSearchListToSet(WebPChunk** chunk_list, uint32_t nth,
                                WebPChunk*** const location) {
  uint32_t count = 0;
  assert(chunk_list != NULL);
  *location = chunk_list;

  while (*chunk_list != NULL) {
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
  if (tag == kChunks[IDX_VP8X].tag || tag == kChunks[IDX_ANIM].tag) {
    copy_data = 1;
  }

  ChunkRelease(chunk);

  if (data != NULL) {
    if (copy_data) {        // Copy data.
      if (!WebPDataCopy(data, &chunk->data_)) return WEBP_MUX_MEMORY_ERROR;
      chunk->owner_ = 1;    // Chunk is owner of data.
    } else {                // Don't copy data.
      chunk->data_ = *data;
    }
  }
  chunk->tag_ = tag;
  return WEBP_MUX_OK;
}

WebPMuxError ChunkSetNth(WebPChunk* chunk, WebPChunk** chunk_list,
                         uint32_t nth) {
  WebPChunk* new_chunk;

  if (!ChunkSearchListToSet(chunk_list, nth, &chunk_list)) {
    return WEBP_MUX_NOT_FOUND;
  }

  new_chunk = (WebPChunk*)WebPSafeMalloc(1ULL, sizeof(*new_chunk));
  if (new_chunk == NULL) return WEBP_MUX_MEMORY_ERROR;
  *new_chunk = *chunk;
  chunk->owner_ = 0;
  new_chunk->next_ = *chunk_list;
  *chunk_list = new_chunk;
  return WEBP_MUX_OK;
}

//------------------------------------------------------------------------------
// Chunk deletion method(s).

WebPChunk* ChunkDelete(WebPChunk* const chunk) {
  WebPChunk* const next = ChunkRelease(chunk);
  WebPSafeFree(chunk);
  return next;
}

void ChunkListDelete(WebPChunk** const chunk_list) {
  while (*chunk_list != NULL) {
    *chunk_list = ChunkDelete(*chunk_list);
  }
}

//------------------------------------------------------------------------------
// Chunk serialization methods.

static uint8_t* ChunkEmit(const WebPChunk* const chunk, uint8_t* dst) {
  const size_t chunk_size = chunk->data_.size;
  assert(chunk);
  assert(chunk->tag_ != NIL_TAG);
  PutLE32(dst + 0, chunk->tag_);
  PutLE32(dst + TAG_SIZE, (uint32_t)chunk_size);
  assert(chunk_size == (uint32_t)chunk_size);
  memcpy(dst + CHUNK_HEADER_SIZE, chunk->data_.bytes, chunk_size);
  if (chunk_size & 1)
    dst[CHUNK_HEADER_SIZE + chunk_size] = 0;  // Add padding.
  return dst + ChunkDiskSize(chunk);
}

uint8_t* ChunkListEmit(const WebPChunk* chunk_list, uint8_t* dst) {
  while (chunk_list != NULL) {
    dst = ChunkEmit(chunk_list, dst);
    chunk_list = chunk_list->next_;
  }
  return dst;
}

size_t ChunkListDiskSize(const WebPChunk* chunk_list) {
  size_t size = 0;
  while (chunk_list != NULL) {
    size += ChunkDiskSize(chunk_list);
    chunk_list = chunk_list->next_;
  }
  return size;
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
  ChunkListDelete(&wpi->unknown_);

  next = wpi->next_;
  MuxImageInit(wpi);
  return next;
}

//------------------------------------------------------------------------------
// MuxImage search methods.

// Get a reference to appropriate chunk list within an image given chunk tag.
static WebPChunk** GetChunkListFromId(const WebPMuxImage* const wpi,
                                      WebPChunkId id) {
  assert(wpi != NULL);
  switch (id) {
    case WEBP_CHUNK_ANMF:  return (WebPChunk**)&wpi->header_;
    case WEBP_CHUNK_ALPHA: return (WebPChunk**)&wpi->alpha_;
    case WEBP_CHUNK_IMAGE: return (WebPChunk**)&wpi->img_;
    default: return NULL;
  }
}

int MuxImageCount(const WebPMuxImage* wpi_list, WebPChunkId id) {
  int count = 0;
  const WebPMuxImage* current;
  for (current = wpi_list; current != NULL; current = current->next_) {
    if (id == WEBP_CHUNK_NIL) {
      ++count;  // Special case: count all images.
    } else {
      const WebPChunk* const wpi_chunk = *GetChunkListFromId(current, id);
      if (wpi_chunk != NULL) {
        const WebPChunkId wpi_chunk_id = ChunkGetIdFromTag(wpi_chunk->tag_);
        if (wpi_chunk_id == id) ++count;  // Count images with a matching 'id'.
      }
    }
  }
  return count;
}

// Outputs a pointer to 'prev_wpi->next_',
//   where 'prev_wpi' is the pointer to the image at position (nth - 1).
// Returns true if nth image was found.
static int SearchImageToGetOrDelete(WebPMuxImage** wpi_list, uint32_t nth,
                                    WebPMuxImage*** const location) {
  uint32_t count = 0;
  assert(wpi_list);
  *location = wpi_list;

  if (nth == 0) {
    nth = MuxImageCount(*wpi_list, WEBP_CHUNK_NIL);
    if (nth == 0) return 0;  // Not found.
  }

  while (*wpi_list != NULL) {
    WebPMuxImage* const cur_wpi = *wpi_list;
    ++count;
    if (count == nth) return 1;  // Found.
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

  new_wpi = (WebPMuxImage*)WebPSafeMalloc(1ULL, sizeof(*new_wpi));
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
  WebPSafeFree(wpi);
  return next;
}

WebPMuxError MuxImageDeleteNth(WebPMuxImage** wpi_list, uint32_t nth) {
  assert(wpi_list);
  if (!SearchImageToGetOrDelete(wpi_list, nth, &wpi_list)) {
    return WEBP_MUX_NOT_FOUND;
  }
  *wpi_list = MuxImageDelete(*wpi_list);
  return WEBP_MUX_OK;
}

//------------------------------------------------------------------------------
// MuxImage reader methods.

WebPMuxError MuxImageGetNth(const WebPMuxImage** wpi_list, uint32_t nth,
                            WebPMuxImage** wpi) {
  assert(wpi_list);
  assert(wpi);
  if (!SearchImageToGetOrDelete((WebPMuxImage**)wpi_list, nth,
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
  if (wpi->unknown_ != NULL) size += ChunkListDiskSize(wpi->unknown_);
  return size;
}

// Special case as ANMF chunk encapsulates other image chunks.
static uint8_t* ChunkEmitSpecial(const WebPChunk* const header,
                                 size_t total_size, uint8_t* dst) {
  const size_t header_size = header->data_.size;
  const size_t offset_to_next = total_size - CHUNK_HEADER_SIZE;
  assert(header->tag_ == kChunks[IDX_ANMF].tag);
  PutLE32(dst + 0, header->tag_);
  PutLE32(dst + TAG_SIZE, (uint32_t)offset_to_next);
  assert(header_size == (uint32_t)header_size);
  memcpy(dst + CHUNK_HEADER_SIZE, header->data_.bytes, header_size);
  if (header_size & 1) {
    dst[CHUNK_HEADER_SIZE + header_size] = 0;  // Add padding.
  }
  return dst + ChunkDiskSize(header);
}

uint8_t* MuxImageEmit(const WebPMuxImage* const wpi, uint8_t* dst) {
  // Ordering of chunks to be emitted is strictly as follows:
  // 1. ANMF chunk (if present).
  // 2. ALPH chunk (if present).
  // 3. VP8/VP8L chunk.
  assert(wpi);
  if (wpi->header_ != NULL) {
    dst = ChunkEmitSpecial(wpi->header_, MuxImageDiskSize(wpi), dst);
  }
  if (wpi->alpha_ != NULL) dst = ChunkEmit(wpi->alpha_, dst);
  if (wpi->img_ != NULL) dst = ChunkEmit(wpi->img_, dst);
  if (wpi->unknown_ != NULL) dst = ChunkListEmit(wpi->unknown_, dst);
  return dst;
}

//------------------------------------------------------------------------------
// Helper methods for mux.

int MuxHasAlpha(const WebPMuxImage* images) {
  while (images != NULL) {
    if (images->has_alpha_) return 1;
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
  switch (id) {
    case WEBP_CHUNK_VP8X:    return (WebPChunk**)&mux->vp8x_;
    case WEBP_CHUNK_ICCP:    return (WebPChunk**)&mux->iccp_;
    case WEBP_CHUNK_ANIM:    return (WebPChunk**)&mux->anim_;
    case WEBP_CHUNK_EXIF:    return (WebPChunk**)&mux->exif_;
    case WEBP_CHUNK_XMP:     return (WebPChunk**)&mux->xmp_;
    default:                 return (WebPChunk**)&mux->unknown_;
  }
}

static int IsNotCompatible(int feature, int num_items) {
  return (feature != 0) != (num_items > 0);
}

#define NO_FLAG ((WebPFeatureFlags)0)

// Test basic constraints:
// retrieval, maximum number of chunks by index (use -1 to skip)
// and feature incompatibility (use NO_FLAG to skip).
// On success returns WEBP_MUX_OK and stores the chunk count in *num.
static WebPMuxError ValidateChunk(const WebPMux* const mux, CHUNK_INDEX idx,
                                  WebPFeatureFlags feature,
                                  uint32_t vp8x_flags,
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
  int num_exif;
  int num_xmp;
  int num_anim;
  int num_frames;
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

  // At most one EXIF metadata.
  err = ValidateChunk(mux, IDX_EXIF, EXIF_FLAG, flags, 1, &num_exif);
  if (err != WEBP_MUX_OK) return err;

  // At most one XMP metadata.
  err = ValidateChunk(mux, IDX_XMP, XMP_FLAG, flags, 1, &num_xmp);
  if (err != WEBP_MUX_OK) return err;

  // Animation: ANIMATION_FLAG, ANIM chunk and ANMF chunk(s) are consistent.
  // At most one ANIM chunk.
  err = ValidateChunk(mux, IDX_ANIM, NO_FLAG, flags, 1, &num_anim);
  if (err != WEBP_MUX_OK) return err;
  err = ValidateChunk(mux, IDX_ANMF, NO_FLAG, flags, -1, &num_frames);
  if (err != WEBP_MUX_OK) return err;

  {
    const int has_animation = !!(flags & ANIMATION_FLAG);
    if (has_animation && (num_anim == 0 || num_frames == 0)) {
      return WEBP_MUX_INVALID_ARGUMENT;
    }
    if (!has_animation && (num_anim == 1 || num_frames > 0)) {
      return WEBP_MUX_INVALID_ARGUMENT;
    }
    if (!has_animation) {
      const WebPMuxImage* images = mux->images_;
      // There can be only one image.
      if (images == NULL || images->next_ != NULL) {
        return WEBP_MUX_INVALID_ARGUMENT;
      }
      // Size must match.
      if (mux->canvas_width_ > 0) {
        if (images->width_ != mux->canvas_width_ ||
            images->height_ != mux->canvas_height_) {
          return WEBP_MUX_INVALID_ARGUMENT;
        }
      }
    }
  }

  // Verify either VP8X chunk is present OR there is only one elem in
  // mux->images_.
  err = ValidateChunk(mux, IDX_VP8X, NO_FLAG, flags, 1, &num_vp8x);
  if (err != WEBP_MUX_OK) return err;
  err = ValidateChunk(mux, IDX_VP8, NO_FLAG, flags, -1, &num_images);
  if (err != WEBP_MUX_OK) return err;
  if (num_vp8x == 0 && num_images != 1) return WEBP_MUX_INVALID_ARGUMENT;

  // ALPHA_FLAG & alpha chunk(s) are consistent.
  // Note: ALPHA_FLAG can be set when there is actually no Alpha data present.
  if (MuxHasAlpha(mux->images_)) {
    if (num_vp8x > 0) {
      // VP8X chunk is present, so it should contain ALPHA_FLAG.
      if (!(flags & ALPHA_FLAG)) return WEBP_MUX_INVALID_ARGUMENT;
    } else {
      // VP8X chunk is not present, so ALPH chunks should NOT be present either.
      err = WebPMuxNumChunks(mux, WEBP_CHUNK_ALPHA, &num_alpha);
      if (err != WEBP_MUX_OK) return err;
      if (num_alpha > 0) return WEBP_MUX_INVALID_ARGUMENT;
    }
  }

  return WEBP_MUX_OK;
}

#undef NO_FLAG

//------------------------------------------------------------------------------


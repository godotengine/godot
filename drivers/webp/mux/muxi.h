// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Internal header for mux library.
//
// Author: Urvang (urvang@google.com)

#ifndef WEBP_MUX_MUXI_H_
#define WEBP_MUX_MUXI_H_

#include <stdlib.h>
#include "../dec/vp8i.h"
#include "../dec/vp8li.h"
#include "../webp/format_constants.h"
#include "../webp/mux.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------
// Defines and constants.

// Chunk object.
typedef struct WebPChunk WebPChunk;
struct WebPChunk {
  uint32_t        tag_;
  int             owner_;  // True if *data_ memory is owned internally.
                           // VP8X, Loop, and other internally created chunks
                           // like frame/tile are always owned.
  WebPData        data_;
  WebPChunk*      next_;
};

// MuxImage object. Store a full webp image (including frame/tile chunk, alpha
// chunk and VP8/VP8L chunk),
typedef struct WebPMuxImage WebPMuxImage;
struct WebPMuxImage {
  WebPChunk*  header_;      // Corresponds to WEBP_CHUNK_FRAME/WEBP_CHUNK_TILE.
  WebPChunk*  alpha_;       // Corresponds to WEBP_CHUNK_ALPHA.
  WebPChunk*  img_;         // Corresponds to WEBP_CHUNK_IMAGE.
  int         is_partial_;  // True if only some of the chunks are filled.
  WebPMuxImage* next_;
};

// Main mux object. Stores data chunks.
struct WebPMux {
  WebPMuxImage*   images_;
  WebPChunk*      iccp_;
  WebPChunk*      meta_;
  WebPChunk*      loop_;
  WebPChunk*      vp8x_;

  WebPChunk*  unknown_;
};

// CHUNK_INDEX enum: used for indexing within 'kChunks' (defined below) only.
// Note: the reason for having two enums ('WebPChunkId' and 'CHUNK_INDEX') is to
// allow two different chunks to have the same id (e.g. WebPChunkId
// 'WEBP_CHUNK_IMAGE' can correspond to CHUNK_INDEX 'IDX_VP8' or 'IDX_VP8L').
typedef enum {
  IDX_VP8X = 0,
  IDX_ICCP,
  IDX_LOOP,
  IDX_FRAME,
  IDX_TILE,
  IDX_ALPHA,
  IDX_VP8,
  IDX_VP8L,
  IDX_META,
  IDX_UNKNOWN,

  IDX_NIL,
  IDX_LAST_CHUNK
} CHUNK_INDEX;

#define NIL_TAG 0x00000000u  // To signal void chunk.

#define MKFOURCC(a, b, c, d) ((uint32_t)(a) | (b) << 8 | (c) << 16 | (d) << 24)

typedef struct {
  uint32_t      tag;
  WebPChunkId   id;
  uint32_t      size;
} ChunkInfo;

extern const ChunkInfo kChunks[IDX_LAST_CHUNK];

//------------------------------------------------------------------------------
// Helper functions.

// Read 16, 24 or 32 bits stored in little-endian order.
static WEBP_INLINE int GetLE16(const uint8_t* const data) {
  return (int)(data[0] << 0) | (data[1] << 8);
}

static WEBP_INLINE int GetLE24(const uint8_t* const data) {
  return GetLE16(data) | (data[2] << 16);
}

static WEBP_INLINE uint32_t GetLE32(const uint8_t* const data) {
  return (uint32_t)GetLE16(data) | (GetLE16(data + 2) << 16);
}

// Store 16, 24 or 32 bits in little-endian order.
static WEBP_INLINE void PutLE16(uint8_t* const data, int val) {
  assert(val < (1 << 16));
  data[0] = (val >> 0);
  data[1] = (val >> 8);
}

static WEBP_INLINE void PutLE24(uint8_t* const data, int val) {
  assert(val < (1 << 24));
  PutLE16(data, val & 0xffff);
  data[2] = (val >> 16);
}

static WEBP_INLINE void PutLE32(uint8_t* const data, uint32_t val) {
  PutLE16(data, (int)(val & 0xffff));
  PutLE16(data + 2, (int)(val >> 16));
}

static WEBP_INLINE size_t SizeWithPadding(size_t chunk_size) {
  return CHUNK_HEADER_SIZE + ((chunk_size + 1) & ~1U);
}

//------------------------------------------------------------------------------
// Chunk object management.

// Initialize.
void ChunkInit(WebPChunk* const chunk);

// Get chunk index from chunk tag. Returns IDX_NIL if not found.
CHUNK_INDEX ChunkGetIndexFromTag(uint32_t tag);

// Get chunk id from chunk tag. Returns WEBP_CHUNK_NIL if not found.
WebPChunkId ChunkGetIdFromTag(uint32_t tag);

// Search for nth chunk with given 'tag' in the chunk list.
// nth = 0 means "last of the list".
WebPChunk* ChunkSearchList(WebPChunk* first, uint32_t nth, uint32_t tag);

// Fill the chunk with the given data.
WebPMuxError ChunkAssignData(WebPChunk* chunk, const WebPData* const data,
                             int copy_data, uint32_t tag);

// Sets 'chunk' at nth position in the 'chunk_list'.
// nth = 0 has the special meaning "last of the list".
WebPMuxError ChunkSetNth(const WebPChunk* chunk, WebPChunk** chunk_list,
                         uint32_t nth);

// Releases chunk and returns chunk->next_.
WebPChunk* ChunkRelease(WebPChunk* const chunk);

// Deletes given chunk & returns chunk->next_.
WebPChunk* ChunkDelete(WebPChunk* const chunk);

// Size of a chunk including header and padding.
static WEBP_INLINE size_t ChunkDiskSize(const WebPChunk* chunk) {
  const size_t data_size = chunk->data_.size_;
  assert(data_size < MAX_CHUNK_PAYLOAD);
  return SizeWithPadding(data_size);
}

// Total size of a list of chunks.
size_t ChunksListDiskSize(const WebPChunk* chunk_list);

// Write out the given list of chunks into 'dst'.
uint8_t* ChunkListEmit(const WebPChunk* chunk_list, uint8_t* dst);

// Get the width & height of image stored in 'image_chunk'.
WebPMuxError MuxGetImageWidthHeight(const WebPChunk* const image_chunk,
                                    int* const width, int* const height);

//------------------------------------------------------------------------------
// MuxImage object management.

// Initialize.
void MuxImageInit(WebPMuxImage* const wpi);

// Releases image 'wpi' and returns wpi->next.
WebPMuxImage* MuxImageRelease(WebPMuxImage* const wpi);

// Delete image 'wpi' and return the next image in the list or NULL.
// 'wpi' can be NULL.
WebPMuxImage* MuxImageDelete(WebPMuxImage* const wpi);

// Delete all images in 'wpi_list'.
void MuxImageDeleteAll(WebPMuxImage** const wpi_list);

// Count number of images matching the given tag id in the 'wpi_list'.
int MuxImageCount(const WebPMuxImage* wpi_list, WebPChunkId id);

// Check if given ID corresponds to an image related chunk.
static WEBP_INLINE int IsWPI(WebPChunkId id) {
  switch (id) {
    case WEBP_CHUNK_FRAME:
    case WEBP_CHUNK_TILE:
    case WEBP_CHUNK_ALPHA:
    case WEBP_CHUNK_IMAGE:  return 1;
    default:        return 0;
  }
}

// Get a reference to appropriate chunk list within an image given chunk tag.
static WEBP_INLINE WebPChunk** MuxImageGetListFromId(
    const WebPMuxImage* const wpi, WebPChunkId id) {
  assert(wpi != NULL);
  switch (id) {
    case WEBP_CHUNK_FRAME:
    case WEBP_CHUNK_TILE:  return (WebPChunk**)&wpi->header_;
    case WEBP_CHUNK_ALPHA: return (WebPChunk**)&wpi->alpha_;
    case WEBP_CHUNK_IMAGE: return (WebPChunk**)&wpi->img_;
    default: return NULL;
  }
}

// Pushes 'wpi' at the end of 'wpi_list'.
WebPMuxError MuxImagePush(const WebPMuxImage* wpi, WebPMuxImage** wpi_list);

// Delete nth image in the image list with given tag id.
WebPMuxError MuxImageDeleteNth(WebPMuxImage** wpi_list, uint32_t nth,
                               WebPChunkId id);

// Get nth image in the image list with given tag id.
WebPMuxError MuxImageGetNth(const WebPMuxImage** wpi_list, uint32_t nth,
                            WebPChunkId id, WebPMuxImage** wpi);

// Total size of the given image.
size_t MuxImageDiskSize(const WebPMuxImage* const wpi);

// Total size of a list of images.
size_t MuxImageListDiskSize(const WebPMuxImage* wpi_list);

// Write out the given image into 'dst'.
uint8_t* MuxImageEmit(const WebPMuxImage* const wpi, uint8_t* dst);

// Write out the given list of images into 'dst'.
uint8_t* MuxImageListEmit(const WebPMuxImage* wpi_list, uint8_t* dst);

//------------------------------------------------------------------------------
// Helper methods for mux.

// Checks if the given image list contains at least one lossless image.
int MuxHasLosslessImages(const WebPMuxImage* images);

// Write out RIFF header into 'data', given total data size 'size'.
uint8_t* MuxEmitRiffHeader(uint8_t* const data, size_t size);

// Returns the list where chunk with given ID is to be inserted in mux.
// Return value is NULL if this chunk should be inserted in mux->images_ list
// or if 'id' is not known.
WebPChunk** MuxGetChunkListFromId(const WebPMux* mux, WebPChunkId id);

// Validates that the given mux has a single image.
WebPMuxError MuxValidateForImage(const WebPMux* const mux);

// Validates the given mux object.
WebPMuxError MuxValidate(const WebPMux* const mux);

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

#endif  /* WEBP_MUX_MUXI_H_ */

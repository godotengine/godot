// Copyright 2012 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
//  WebP container demux.
//

#include "../webp/mux.h"

#include <stdlib.h>
#include <string.h>

#include "../webp/decode.h"  // WebPGetInfo
#include "../webp/format_constants.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#define MKFOURCC(a, b, c, d) ((uint32_t)(a) | (b) << 8 | (c) << 16 | (d) << 24)

typedef struct {
  size_t start_;        // start location of the data
  size_t end_;          // end location
  size_t riff_end_;     // riff chunk end location, can be > end_.
  size_t buf_size_;     // size of the buffer
  const uint8_t* buf_;
} MemBuffer;

typedef struct {
  size_t offset_;
  size_t size_;
} ChunkData;

typedef struct Frame {
  int x_offset_, y_offset_;
  int width_, height_;
  int duration_;
  int is_tile_;    // this is an image fragment from a 'TILE'.
  int frame_num_;  // the referent frame number for use in assembling tiles.
  int complete_;   // img_components_ contains a full image.
  ChunkData img_components_[2];  // 0=VP8{,L} 1=ALPH
  struct Frame* next_;
} Frame;

typedef struct Chunk {
  ChunkData data_;
  struct Chunk* next_;
} Chunk;

struct WebPDemuxer {
  MemBuffer mem_;
  WebPDemuxState state_;
  int is_ext_format_;
  uint32_t feature_flags_;
  int canvas_width_, canvas_height_;
  int loop_count_;
  int num_frames_;
  Frame* frames_;
  Chunk* chunks_;  // non-image chunks
};

typedef enum {
  PARSE_OK,
  PARSE_NEED_MORE_DATA,
  PARSE_ERROR
} ParseStatus;

typedef struct ChunkParser {
  uint8_t id[4];
  ParseStatus (*parse)(WebPDemuxer* const dmux);
  int (*valid)(const WebPDemuxer* const dmux);
} ChunkParser;

static ParseStatus ParseSingleImage(WebPDemuxer* const dmux);
static ParseStatus ParseVP8X(WebPDemuxer* const dmux);
static int IsValidSimpleFormat(const WebPDemuxer* const dmux);
static int IsValidExtendedFormat(const WebPDemuxer* const dmux);

static const ChunkParser kMasterChunks[] = {
  { { 'V', 'P', '8', ' ' }, ParseSingleImage, IsValidSimpleFormat },
  { { 'V', 'P', '8', 'L' }, ParseSingleImage, IsValidSimpleFormat },
  { { 'V', 'P', '8', 'X' }, ParseVP8X,        IsValidExtendedFormat },
  { { '0', '0', '0', '0' }, NULL,             NULL },
};

// -----------------------------------------------------------------------------
// MemBuffer

static int RemapMemBuffer(MemBuffer* const mem,
                          const uint8_t* data, size_t size) {
  if (size < mem->buf_size_) return 0;  // can't remap to a shorter buffer!

  mem->buf_ = data;
  mem->end_ = mem->buf_size_ = size;
  return 1;
}

static int InitMemBuffer(MemBuffer* const mem,
                         const uint8_t* data, size_t size) {
  memset(mem, 0, sizeof(*mem));
  return RemapMemBuffer(mem, data, size);
}

// Return the remaining data size available in 'mem'.
static WEBP_INLINE size_t MemDataSize(const MemBuffer* const mem) {
  return (mem->end_ - mem->start_);
}

// Return true if 'size' exceeds the end of the RIFF chunk.
static WEBP_INLINE int SizeIsInvalid(const MemBuffer* const mem, size_t size) {
  return (size > mem->riff_end_ - mem->start_);
}

static WEBP_INLINE void Skip(MemBuffer* const mem, size_t size) {
  mem->start_ += size;
}

static WEBP_INLINE void Rewind(MemBuffer* const mem, size_t size) {
  mem->start_ -= size;
}

static WEBP_INLINE const uint8_t* GetBuffer(MemBuffer* const mem) {
  return mem->buf_ + mem->start_;
}

static WEBP_INLINE uint8_t GetByte(MemBuffer* const mem) {
  const uint8_t byte = mem->buf_[mem->start_];
  Skip(mem, 1);
  return byte;
}

// Read 16, 24 or 32 bits stored in little-endian order.
static WEBP_INLINE int ReadLE16s(const uint8_t* const data) {
  return (int)(data[0] << 0) | (data[1] << 8);
}

static WEBP_INLINE int ReadLE24s(const uint8_t* const data) {
  return ReadLE16s(data) | (data[2] << 16);
}

static WEBP_INLINE uint32_t ReadLE32(const uint8_t* const data) {
  return (uint32_t)ReadLE24s(data) | (data[3] << 24);
}

// In addition to reading, skip the read bytes.
static WEBP_INLINE int GetLE16s(MemBuffer* const mem) {
  const uint8_t* const data = mem->buf_ + mem->start_;
  const int val = ReadLE16s(data);
  Skip(mem, 2);
  return val;
}

static WEBP_INLINE int GetLE24s(MemBuffer* const mem) {
  const uint8_t* const data = mem->buf_ + mem->start_;
  const int val = ReadLE24s(data);
  Skip(mem, 3);
  return val;
}

static WEBP_INLINE uint32_t GetLE32(MemBuffer* const mem) {
  const uint8_t* const data = mem->buf_ + mem->start_;
  const uint32_t val = ReadLE32(data);
  Skip(mem, 4);
  return val;
}

// -----------------------------------------------------------------------------
// Secondary chunk parsing

static void AddChunk(WebPDemuxer* const dmux, Chunk* const chunk) {
  Chunk** c = &dmux->chunks_;
  while (*c != NULL) c = &(*c)->next_;
  *c = chunk;
  chunk->next_ = NULL;
}

// Add a frame to the end of the list, ensuring the last frame is complete.
// Returns true on success, false otherwise.
static int AddFrame(WebPDemuxer* const dmux, Frame* const frame) {
  const Frame* last_frame = NULL;
  Frame** f = &dmux->frames_;
  while (*f != NULL) {
    last_frame = *f;
    f = &(*f)->next_;
  }
  if (last_frame != NULL && !last_frame->complete_) return 0;
  *f = frame;
  frame->next_ = NULL;
  return 1;
}

// Store image bearing chunks to 'frame'.
static ParseStatus StoreFrame(int frame_num, MemBuffer* const mem,
                              Frame* const frame) {
  int alpha_chunks = 0;
  int image_chunks = 0;
  int done = (MemDataSize(mem) < CHUNK_HEADER_SIZE);
  ParseStatus status = PARSE_OK;

  if (done) return PARSE_NEED_MORE_DATA;

  do {
    const size_t chunk_start_offset = mem->start_;
    const uint32_t fourcc = GetLE32(mem);
    const uint32_t payload_size = GetLE32(mem);
    const uint32_t payload_size_padded = payload_size + (payload_size & 1);
    const size_t payload_available = (payload_size_padded > MemDataSize(mem))
                                   ? MemDataSize(mem) : payload_size_padded;
    const size_t chunk_size = CHUNK_HEADER_SIZE + payload_available;

    if (payload_size > MAX_CHUNK_PAYLOAD) return PARSE_ERROR;
    if (SizeIsInvalid(mem, payload_size_padded)) return PARSE_ERROR;
    if (payload_size_padded > MemDataSize(mem)) status = PARSE_NEED_MORE_DATA;

    switch (fourcc) {
      case MKFOURCC('A', 'L', 'P', 'H'):
        if (alpha_chunks == 0) {
          ++alpha_chunks;
          frame->img_components_[1].offset_ = chunk_start_offset;
          frame->img_components_[1].size_ = chunk_size;
          frame->frame_num_ = frame_num;
          Skip(mem, payload_available);
        } else {
          goto Done;
        }
        break;
      case MKFOURCC('V', 'P', '8', ' '):
      case MKFOURCC('V', 'P', '8', 'L'):
        if (image_chunks == 0) {
          int width = 0, height = 0;
          ++image_chunks;
          frame->img_components_[0].offset_ = chunk_start_offset;
          frame->img_components_[0].size_ = chunk_size;
          // Extract the width and height from the bitstream, tolerating
          // failures when the data is incomplete.
          if (!WebPGetInfo(mem->buf_ + frame->img_components_[0].offset_,
                           frame->img_components_[0].size_, &width, &height) &&
              status != PARSE_NEED_MORE_DATA) {
            return PARSE_ERROR;
          }

          frame->width_ = width;
          frame->height_ = height;
          frame->frame_num_ = frame_num;
          frame->complete_ = (status == PARSE_OK);
          Skip(mem, payload_available);
        } else {
          goto Done;
        }
        break;
 Done:
      default:
        // Restore fourcc/size when moving up one level in parsing.
        Rewind(mem, CHUNK_HEADER_SIZE);
        done = 1;
        break;
    }

    if (mem->start_ == mem->riff_end_) {
      done = 1;
    } else if (MemDataSize(mem) < CHUNK_HEADER_SIZE) {
      status = PARSE_NEED_MORE_DATA;
    }
  } while (!done && status == PARSE_OK);

  return status;
}

// Creates a new Frame if 'actual_size' is within bounds and 'mem' contains
// enough data ('min_size') to parse the payload.
// Returns PARSE_OK on success with *frame pointing to the new Frame.
// Returns PARSE_NEED_MORE_DATA with insufficient data, PARSE_ERROR otherwise.
static ParseStatus NewFrame(const MemBuffer* const mem,
                            uint32_t min_size, uint32_t expected_size,
                            uint32_t actual_size, Frame** frame) {
  if (SizeIsInvalid(mem, min_size)) return PARSE_ERROR;
  if (actual_size < expected_size) return PARSE_ERROR;
  if (MemDataSize(mem) < min_size)  return PARSE_NEED_MORE_DATA;

  *frame = (Frame*)calloc(1, sizeof(**frame));
  return (*frame == NULL) ? PARSE_ERROR : PARSE_OK;
}

// Parse a 'FRM ' chunk and any image bearing chunks that immediately follow.
// 'frame_chunk_size' is the previously validated, padded chunk size.
static ParseStatus ParseFrame(
    WebPDemuxer* const dmux, uint32_t frame_chunk_size) {
  const int has_frames = !!(dmux->feature_flags_ & ANIMATION_FLAG);
  const uint32_t min_size = frame_chunk_size + CHUNK_HEADER_SIZE;
  int added_frame = 0;
  MemBuffer* const mem = &dmux->mem_;
  Frame* frame;
  ParseStatus status =
      NewFrame(mem, min_size, FRAME_CHUNK_SIZE, frame_chunk_size, &frame);
  if (status != PARSE_OK) return status;

  frame->x_offset_ = 2 * GetLE24s(mem);
  frame->y_offset_ = 2 * GetLE24s(mem);
  frame->width_    = 1 + GetLE24s(mem);
  frame->height_   = 1 + GetLE24s(mem);
  frame->duration_ = 1 + GetLE24s(mem);
  Skip(mem, frame_chunk_size - FRAME_CHUNK_SIZE);  // skip any trailing data.
  if (frame->width_ * (uint64_t)frame->height_ >= MAX_IMAGE_AREA) {
    return PARSE_ERROR;
  }

  // Store a (potentially partial) frame only if the animation flag is set
  // and there is some data in 'frame'.
  status = StoreFrame(dmux->num_frames_ + 1, mem, frame);
  if (status != PARSE_ERROR && has_frames && frame->frame_num_ > 0) {
    added_frame = AddFrame(dmux, frame);
    if (added_frame) {
      ++dmux->num_frames_;
    } else {
      status = PARSE_ERROR;
    }
  }

  if (!added_frame) free(frame);
  return status;
}

// Parse a 'TILE' chunk and any image bearing chunks that immediately follow.
// 'tile_chunk_size' is the previously validated, padded chunk size.
static ParseStatus ParseTile(WebPDemuxer* const dmux,
                             uint32_t tile_chunk_size) {
  const int has_tiles = !!(dmux->feature_flags_ & TILE_FLAG);
  const uint32_t min_size = tile_chunk_size + CHUNK_HEADER_SIZE;
  int added_tile = 0;
  MemBuffer* const mem = &dmux->mem_;
  Frame* frame;
  ParseStatus status =
      NewFrame(mem, min_size, TILE_CHUNK_SIZE, tile_chunk_size, &frame);
  if (status != PARSE_OK) return status;

  frame->is_tile_  = 1;
  frame->x_offset_ = 2 * GetLE24s(mem);
  frame->y_offset_ = 2 * GetLE24s(mem);
  Skip(mem, tile_chunk_size - TILE_CHUNK_SIZE);  // skip any trailing data.

  // Store a (potentially partial) tile only if the tile flag is set
  // and the tile contains some data.
  status = StoreFrame(dmux->num_frames_, mem, frame);
  if (status != PARSE_ERROR && has_tiles && frame->frame_num_ > 0) {
    // Note num_frames_ is incremented only when all tiles have been consumed.
    added_tile = AddFrame(dmux, frame);
    if (!added_tile) status = PARSE_ERROR;
  }

  if (!added_tile) free(frame);
  return status;
}

// General chunk storage starting with the header at 'start_offset' allowing
// the user to request the payload via a fourcc string. 'size' includes the
// header and the unpadded payload size.
// Returns true on success, false otherwise.
static int StoreChunk(WebPDemuxer* const dmux,
                      size_t start_offset, uint32_t size) {
  Chunk* const chunk = (Chunk*)calloc(1, sizeof(*chunk));
  if (chunk == NULL) return 0;

  chunk->data_.offset_ = start_offset;
  chunk->data_.size_ = size;
  AddChunk(dmux, chunk);
  return 1;
}

// -----------------------------------------------------------------------------
// Primary chunk parsing

static int ReadHeader(MemBuffer* const mem) {
  const size_t min_size = RIFF_HEADER_SIZE + CHUNK_HEADER_SIZE;
  uint32_t riff_size;

  // Basic file level validation.
  if (MemDataSize(mem) < min_size) return 0;
  if (memcmp(GetBuffer(mem), "RIFF", CHUNK_SIZE_BYTES) ||
      memcmp(GetBuffer(mem) + CHUNK_HEADER_SIZE, "WEBP", CHUNK_SIZE_BYTES)) {
    return 0;
  }

  riff_size = ReadLE32(GetBuffer(mem) + TAG_SIZE);
  if (riff_size < CHUNK_HEADER_SIZE) return 0;
  if (riff_size > MAX_CHUNK_PAYLOAD) return 0;

  // There's no point in reading past the end of the RIFF chunk
  mem->riff_end_ = riff_size + CHUNK_HEADER_SIZE;
  if (mem->buf_size_ > mem->riff_end_) {
    mem->buf_size_ = mem->end_ = mem->riff_end_;
  }

  Skip(mem, RIFF_HEADER_SIZE);
  return 1;
}

static ParseStatus ParseSingleImage(WebPDemuxer* const dmux) {
  const size_t min_size = CHUNK_HEADER_SIZE;
  MemBuffer* const mem = &dmux->mem_;
  Frame* frame;
  ParseStatus status;

  if (dmux->frames_ != NULL) return PARSE_ERROR;
  if (SizeIsInvalid(mem, min_size)) return PARSE_ERROR;
  if (MemDataSize(mem) < min_size) return PARSE_NEED_MORE_DATA;

  frame = (Frame*)calloc(1, sizeof(*frame));
  if (frame == NULL) return PARSE_ERROR;

  status = StoreFrame(1, &dmux->mem_, frame);
  if (status != PARSE_ERROR) {
    const int has_alpha = !!(dmux->feature_flags_ & ALPHA_FLAG);
    // Clear any alpha when the alpha flag is missing.
    if (!has_alpha && frame->img_components_[1].size_ > 0) {
      frame->img_components_[1].offset_ = 0;
      frame->img_components_[1].size_ = 0;
    }

    // Use the frame width/height as the canvas values for non-vp8x files.
    if (!dmux->is_ext_format_ && frame->width_ > 0 && frame->height_ > 0) {
      dmux->state_ = WEBP_DEMUX_PARSED_HEADER;
      dmux->canvas_width_ = frame->width_;
      dmux->canvas_height_ = frame->height_;
    }
    AddFrame(dmux, frame);
    dmux->num_frames_ = 1;
  } else {
    free(frame);
  }

  return status;
}

static ParseStatus ParseVP8X(WebPDemuxer* const dmux) {
  MemBuffer* const mem = &dmux->mem_;
  int loop_chunks = 0;
  uint32_t vp8x_size;
  ParseStatus status = PARSE_OK;

  if (MemDataSize(mem) < CHUNK_HEADER_SIZE) return PARSE_NEED_MORE_DATA;

  dmux->is_ext_format_ = 1;
  Skip(mem, TAG_SIZE);  // VP8X
  vp8x_size = GetLE32(mem);
  if (vp8x_size > MAX_CHUNK_PAYLOAD) return PARSE_ERROR;
  if (vp8x_size < VP8X_CHUNK_SIZE) return PARSE_ERROR;
  vp8x_size += vp8x_size & 1;
  if (SizeIsInvalid(mem, vp8x_size)) return PARSE_ERROR;
  if (MemDataSize(mem) < vp8x_size) return PARSE_NEED_MORE_DATA;

  dmux->feature_flags_ = GetByte(mem);
  Skip(mem, 3);  // Reserved.
  dmux->canvas_width_  = 1 + GetLE24s(mem);
  dmux->canvas_height_ = 1 + GetLE24s(mem);
  if (dmux->canvas_width_ * (uint64_t)dmux->canvas_height_ >= MAX_IMAGE_AREA) {
    return PARSE_ERROR;  // image final dimension is too large
  }
  Skip(mem, vp8x_size - VP8X_CHUNK_SIZE);  // skip any trailing data.
  dmux->state_ = WEBP_DEMUX_PARSED_HEADER;

  if (SizeIsInvalid(mem, CHUNK_HEADER_SIZE)) return PARSE_ERROR;
  if (MemDataSize(mem) < CHUNK_HEADER_SIZE) return PARSE_NEED_MORE_DATA;

  do {
    int store_chunk = 1;
    const size_t chunk_start_offset = mem->start_;
    const uint32_t fourcc = GetLE32(mem);
    const uint32_t chunk_size = GetLE32(mem);
    const uint32_t chunk_size_padded = chunk_size + (chunk_size & 1);

    if (chunk_size > MAX_CHUNK_PAYLOAD) return PARSE_ERROR;
    if (SizeIsInvalid(mem, chunk_size_padded)) return PARSE_ERROR;

    switch (fourcc) {
      case MKFOURCC('V', 'P', '8', 'X'): {
        return PARSE_ERROR;
      }
      case MKFOURCC('A', 'L', 'P', 'H'):
      case MKFOURCC('V', 'P', '8', ' '):
      case MKFOURCC('V', 'P', '8', 'L'): {
        Rewind(mem, CHUNK_HEADER_SIZE);
        status = ParseSingleImage(dmux);
        break;
      }
      case MKFOURCC('L', 'O', 'O', 'P'): {
        if (chunk_size_padded < LOOP_CHUNK_SIZE) return PARSE_ERROR;

        if (MemDataSize(mem) < chunk_size_padded) {
          status = PARSE_NEED_MORE_DATA;
        } else if (loop_chunks == 0) {
          ++loop_chunks;
          dmux->loop_count_ = GetLE16s(mem);
          Skip(mem, chunk_size_padded - LOOP_CHUNK_SIZE);
        } else {
          store_chunk = 0;
          goto Skip;
        }
        break;
      }
      case MKFOURCC('F', 'R', 'M', ' '): {
        status = ParseFrame(dmux, chunk_size_padded);
        break;
      }
      case MKFOURCC('T', 'I', 'L', 'E'): {
        if (dmux->num_frames_ == 0) dmux->num_frames_ = 1;
        status = ParseTile(dmux, chunk_size_padded);
        break;
      }
      case MKFOURCC('I', 'C', 'C', 'P'): {
        store_chunk = !!(dmux->feature_flags_ & ICCP_FLAG);
        goto Skip;
      }
      case MKFOURCC('M', 'E', 'T', 'A'): {
        store_chunk = !!(dmux->feature_flags_ & META_FLAG);
        goto Skip;
      }
 Skip:
      default: {
        if (chunk_size_padded <= MemDataSize(mem)) {
          if (store_chunk) {
            // Store only the chunk header and unpadded size as only the payload
            // will be returned to the user.
            if (!StoreChunk(dmux, chunk_start_offset,
                            CHUNK_HEADER_SIZE + chunk_size)) {
              return PARSE_ERROR;
            }
          }
          Skip(mem, chunk_size_padded);
        } else {
          status = PARSE_NEED_MORE_DATA;
        }
      }
    }

    if (mem->start_ == mem->riff_end_) {
      break;
    } else if (MemDataSize(mem) < CHUNK_HEADER_SIZE) {
      status = PARSE_NEED_MORE_DATA;
    }
  } while (status == PARSE_OK);

  return status;
}

// -----------------------------------------------------------------------------
// Format validation

static int IsValidSimpleFormat(const WebPDemuxer* const dmux) {
  const Frame* const frame = dmux->frames_;
  if (dmux->state_ == WEBP_DEMUX_PARSING_HEADER) return 1;

  if (dmux->canvas_width_ <= 0 || dmux->canvas_height_ <= 0) return 0;
  if (dmux->state_ == WEBP_DEMUX_DONE && frame == NULL) return 0;

  if (frame->width_ <= 0 || frame->height_ <= 0) return 0;
  return 1;
}

static int IsValidExtendedFormat(const WebPDemuxer* const dmux) {
  const int has_tiles = !!(dmux->feature_flags_ & TILE_FLAG);
  const int has_frames = !!(dmux->feature_flags_ & ANIMATION_FLAG);
  const Frame* f;

  if (dmux->state_ == WEBP_DEMUX_PARSING_HEADER) return 1;

  if (dmux->canvas_width_ <= 0 || dmux->canvas_height_ <= 0) return 0;
  if (dmux->loop_count_ < 0) return 0;
  if (dmux->state_ == WEBP_DEMUX_DONE && dmux->frames_ == NULL) return 0;

  for (f = dmux->frames_; f != NULL; f = f->next_) {
    const int cur_frame_set = f->frame_num_;
    int frame_count = 0, tile_count = 0;

    // Check frame properties and if the image is composed of tiles that each
    // fragment came from a 'TILE'.
    for (; f != NULL && f->frame_num_ == cur_frame_set; f = f->next_) {
      const ChunkData* const image = f->img_components_;
      const ChunkData* const alpha = f->img_components_ + 1;

      if (!has_tiles && f->is_tile_) return 0;
      if (!has_frames && f->frame_num_ > 1) return 0;
      if (f->x_offset_ < 0 || f->y_offset_ < 0) return 0;
      if (f->complete_) {
        if (alpha->size_ == 0 && image->size_ == 0) return 0;
        // Ensure alpha precedes image bitstream.
        if (alpha->size_ > 0 && alpha->offset_ > image->offset_) {
          return 0;
        }

        if (f->width_ <= 0 || f->height_ <= 0) return 0;
      } else {
        // Ensure alpha precedes image bitstream.
        if (alpha->size_ > 0 && image->size_ > 0 &&
            alpha->offset_ > image->offset_) {
          return 0;
        }
        // There shouldn't be any frames after an incomplete one.
        if (f->next_ != NULL) return 0;
      }

      tile_count += f->is_tile_;
      ++frame_count;
    }
    if (!has_tiles && frame_count > 1) return 0;
    if (tile_count > 0 && frame_count != tile_count) return 0;
    if (f == NULL) break;
  }
  return 1;
}

// -----------------------------------------------------------------------------
// WebPDemuxer object

static void InitDemux(WebPDemuxer* const dmux, const MemBuffer* const mem) {
  dmux->state_ = WEBP_DEMUX_PARSING_HEADER;
  dmux->loop_count_ = 1;
  dmux->canvas_width_ = -1;
  dmux->canvas_height_ = -1;
  dmux->mem_ = *mem;
}

WebPDemuxer* WebPDemuxInternal(const WebPData* data, int allow_partial,
                               WebPDemuxState* state, int version) {
  const ChunkParser* parser;
  int partial;
  ParseStatus status = PARSE_ERROR;
  MemBuffer mem;
  WebPDemuxer* dmux;

  if (WEBP_ABI_IS_INCOMPATIBLE(version, WEBP_DEMUX_ABI_VERSION)) return NULL;
  if (data == NULL || data->bytes_ == NULL || data->size_ == 0) return NULL;

  if (!InitMemBuffer(&mem, data->bytes_, data->size_)) return NULL;
  if (!ReadHeader(&mem)) return NULL;

  partial = (mem.buf_size_ < mem.riff_end_);
  if (!allow_partial && partial) return NULL;

  dmux = (WebPDemuxer*)calloc(1, sizeof(*dmux));
  if (dmux == NULL) return NULL;
  InitDemux(dmux, &mem);

  for (parser = kMasterChunks; parser->parse != NULL; ++parser) {
    if (!memcmp(parser->id, GetBuffer(&dmux->mem_), TAG_SIZE)) {
      status = parser->parse(dmux);
      if (status == PARSE_OK) dmux->state_ = WEBP_DEMUX_DONE;
      if (status != PARSE_ERROR && !parser->valid(dmux)) status = PARSE_ERROR;
      break;
    }
  }
  if (state) *state = dmux->state_;

  if (status == PARSE_ERROR) {
    WebPDemuxDelete(dmux);
    return NULL;
  }
  return dmux;
}

void WebPDemuxDelete(WebPDemuxer* dmux) {
  Chunk* c;
  Frame* f;
  if (dmux == NULL) return;

  for (f = dmux->frames_; f != NULL;) {
    Frame* const cur_frame = f;
    f = f->next_;
    free(cur_frame);
  }
  for (c = dmux->chunks_; c != NULL;) {
    Chunk* const cur_chunk = c;
    c = c->next_;
    free(cur_chunk);
  }
  free(dmux);
}

// -----------------------------------------------------------------------------

uint32_t WebPDemuxGetI(const WebPDemuxer* dmux, WebPFormatFeature feature) {
  if (dmux == NULL) return 0;

  switch (feature) {
    case WEBP_FF_FORMAT_FLAGS:  return dmux->feature_flags_;
    case WEBP_FF_CANVAS_WIDTH:  return (uint32_t)dmux->canvas_width_;
    case WEBP_FF_CANVAS_HEIGHT: return (uint32_t)dmux->canvas_height_;
    case WEBP_FF_LOOP_COUNT:    return (uint32_t)dmux->loop_count_;
  }
  return 0;
}

// -----------------------------------------------------------------------------
// Frame iteration

// Find the first 'frame_num' frame. There may be multiple in a tiled frame.
static const Frame* GetFrame(const WebPDemuxer* const dmux, int frame_num) {
  const Frame* f;
  for (f = dmux->frames_; f != NULL; f = f->next_) {
    if (frame_num == f->frame_num_) break;
  }
  return f;
}

// Returns tile 'tile_num' and the total count.
static const Frame* GetTile(
    const Frame* const frame_set, int tile_num, int* const count) {
  const int this_frame = frame_set->frame_num_;
  const Frame* f = frame_set;
  const Frame* tile = NULL;
  int total;

  for (total = 0; f != NULL && f->frame_num_ == this_frame; f = f->next_) {
    if (++total == tile_num) tile = f;
  }
  *count = total;
  return tile;
}

static const uint8_t* GetFramePayload(const uint8_t* const mem_buf,
                                      const Frame* const frame,
                                      size_t* const data_size) {
  *data_size = 0;
  if (frame != NULL) {
    const ChunkData* const image = frame->img_components_;
    const ChunkData* const alpha = frame->img_components_ + 1;
    size_t start_offset = image->offset_;
    *data_size = image->size_;

    // if alpha exists it precedes image, update the size allowing for
    // intervening chunks.
    if (alpha->size_ > 0) {
      const size_t inter_size = (image->offset_ > 0)
                              ? image->offset_ - (alpha->offset_ + alpha->size_)
                              : 0;
      start_offset = alpha->offset_;
      *data_size  += alpha->size_ + inter_size;
    }
    return mem_buf + start_offset;
  }
  return NULL;
}

// Create a whole 'frame' from VP8 (+ alpha) or lossless.
static int SynthesizeFrame(const WebPDemuxer* const dmux,
                           const Frame* const first_frame,
                           int tile_num, WebPIterator* const iter) {
  const uint8_t* const mem_buf = dmux->mem_.buf_;
  int num_tiles;
  size_t payload_size = 0;
  const Frame* const tile = GetTile(first_frame, tile_num, &num_tiles);
  const uint8_t* const payload = GetFramePayload(mem_buf, tile, &payload_size);
  if (payload == NULL) return 0;

  iter->frame_num_   = first_frame->frame_num_;
  iter->num_frames_  = dmux->num_frames_;
  iter->tile_num_    = tile_num;
  iter->num_tiles_   = num_tiles;
  iter->x_offset_    = tile->x_offset_;
  iter->y_offset_    = tile->y_offset_;
  iter->width_       = tile->width_;
  iter->height_      = tile->height_;
  iter->duration_    = tile->duration_;
  iter->complete_    = tile->complete_;
  iter->tile_.bytes_ = payload;
  iter->tile_.size_  = payload_size;
  // TODO(jzern): adjust offsets for 'TILE's embedded in 'FRM 's
  return 1;
}

static int SetFrame(int frame_num, WebPIterator* const iter) {
  const Frame* frame;
  const WebPDemuxer* const dmux = (WebPDemuxer*)iter->private_;
  if (dmux == NULL || frame_num < 0) return 0;
  if (frame_num > dmux->num_frames_) return 0;
  if (frame_num == 0) frame_num = dmux->num_frames_;

  frame = GetFrame(dmux, frame_num);
  return SynthesizeFrame(dmux, frame, 1, iter);
}

int WebPDemuxGetFrame(const WebPDemuxer* dmux, int frame, WebPIterator* iter) {
  if (iter == NULL) return 0;

  memset(iter, 0, sizeof(*iter));
  iter->private_ = (void*)dmux;
  return SetFrame(frame, iter);
}

int WebPDemuxNextFrame(WebPIterator* iter) {
  if (iter == NULL) return 0;
  return SetFrame(iter->frame_num_ + 1, iter);
}

int WebPDemuxPrevFrame(WebPIterator* iter) {
  if (iter == NULL) return 0;
  if (iter->frame_num_ <= 1) return 0;
  return SetFrame(iter->frame_num_ - 1, iter);
}

int WebPDemuxSelectTile(WebPIterator* iter, int tile) {
  if (iter != NULL && iter->private_ != NULL && tile > 0) {
    const WebPDemuxer* const dmux = (WebPDemuxer*)iter->private_;
    const Frame* const frame = GetFrame(dmux, iter->frame_num_);
    if (frame == NULL) return 0;

    return SynthesizeFrame(dmux, frame, tile, iter);
  }
  return 0;
}

void WebPDemuxReleaseIterator(WebPIterator* iter) {
  (void)iter;
}

// -----------------------------------------------------------------------------
// Chunk iteration

static int ChunkCount(const WebPDemuxer* const dmux, const char fourcc[4]) {
  const uint8_t* const mem_buf = dmux->mem_.buf_;
  const Chunk* c;
  int count = 0;
  for (c = dmux->chunks_; c != NULL; c = c->next_) {
    const uint8_t* const header = mem_buf + c->data_.offset_;
    if (!memcmp(header, fourcc, TAG_SIZE)) ++count;
  }
  return count;
}

static const Chunk* GetChunk(const WebPDemuxer* const dmux,
                             const char fourcc[4], int chunk_num) {
  const uint8_t* const mem_buf = dmux->mem_.buf_;
  const Chunk* c;
  int count = 0;
  for (c = dmux->chunks_; c != NULL; c = c->next_) {
    const uint8_t* const header = mem_buf + c->data_.offset_;
    if (!memcmp(header, fourcc, TAG_SIZE)) ++count;
    if (count == chunk_num) break;
  }
  return c;
}

static int SetChunk(const char fourcc[4], int chunk_num,
                    WebPChunkIterator* const iter) {
  const WebPDemuxer* const dmux = (WebPDemuxer*)iter->private_;
  int count;

  if (dmux == NULL || fourcc == NULL || chunk_num < 0) return 0;
  count = ChunkCount(dmux, fourcc);
  if (count == 0) return 0;
  if (chunk_num == 0) chunk_num = count;

  if (chunk_num <= count) {
    const uint8_t* const mem_buf = dmux->mem_.buf_;
    const Chunk* const chunk = GetChunk(dmux, fourcc, chunk_num);
    iter->chunk_.bytes_ = mem_buf + chunk->data_.offset_ + CHUNK_HEADER_SIZE;
    iter->chunk_.size_  = chunk->data_.size_ - CHUNK_HEADER_SIZE;
    iter->num_chunks_   = count;
    iter->chunk_num_    = chunk_num;
    return 1;
  }
  return 0;
}

int WebPDemuxGetChunk(const WebPDemuxer* dmux,
                      const char fourcc[4], int chunk_num,
                      WebPChunkIterator* iter) {
  if (iter == NULL) return 0;

  memset(iter, 0, sizeof(*iter));
  iter->private_ = (void*)dmux;
  return SetChunk(fourcc, chunk_num, iter);
}

int WebPDemuxNextChunk(WebPChunkIterator* iter) {
  if (iter != NULL) {
    const char* const fourcc =
        (const char*)iter->chunk_.bytes_ - CHUNK_HEADER_SIZE;
    return SetChunk(fourcc, iter->chunk_num_ + 1, iter);
  }
  return 0;
}

int WebPDemuxPrevChunk(WebPChunkIterator* iter) {
  if (iter != NULL && iter->chunk_num_ > 1) {
    const char* const fourcc =
        (const char*)iter->chunk_.bytes_ - CHUNK_HEADER_SIZE;
    return SetChunk(fourcc, iter->chunk_num_ - 1, iter);
  }
  return 0;
}

void WebPDemuxReleaseChunkIterator(WebPChunkIterator* iter) {
  (void)iter;
}

#if defined(__cplusplus) || defined(c_plusplus)
}  // extern "C"
#endif

// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Incremental decoding
//
// Author: somnath@google.com (Somnath Banerjee)

#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include "./webpi.h"
#include "./vp8i.h"
#include "../utils/utils.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

// In append mode, buffer allocations increase as multiples of this value.
// Needs to be a power of 2.
#define CHUNK_SIZE 4096
#define MAX_MB_SIZE 4096

//------------------------------------------------------------------------------
// Data structures for memory and states

// Decoding states. State normally flows like HEADER->PARTS0->DATA->DONE.
// If there is any error the decoder goes into state ERROR.
typedef enum {
  STATE_PRE_VP8,  // All data before that of the first VP8 chunk.
  STATE_VP8_FRAME_HEADER,  // For VP8 Frame header (within VP8 chunk).
  STATE_VP8_PARTS0,
  STATE_VP8_DATA,
  STATE_VP8L_HEADER,
  STATE_VP8L_DATA,
  STATE_DONE,
  STATE_ERROR
} DecState;

// Operating state for the MemBuffer
typedef enum {
  MEM_MODE_NONE = 0,
  MEM_MODE_APPEND,
  MEM_MODE_MAP
} MemBufferMode;

// storage for partition #0 and partial data (in a rolling fashion)
typedef struct {
  MemBufferMode mode_;  // Operation mode
  size_t start_;        // start location of the data to be decoded
  size_t end_;          // end location
  size_t buf_size_;     // size of the allocated buffer
  uint8_t* buf_;        // We don't own this buffer in case WebPIUpdate()

  size_t part0_size_;         // size of partition #0
  const uint8_t* part0_buf_;  // buffer to store partition #0
} MemBuffer;

struct WebPIDecoder {
  DecState state_;         // current decoding state
  WebPDecParams params_;   // Params to store output info
  int is_lossless_;        // for down-casting 'dec_'.
  void* dec_;              // either a VP8Decoder or a VP8LDecoder instance
  VP8Io io_;

  MemBuffer mem_;          // input memory buffer.
  WebPDecBuffer output_;   // output buffer (when no external one is supplied)
  size_t chunk_size_;      // Compressed VP8/VP8L size extracted from Header.
};

// MB context to restore in case VP8DecodeMB() fails
typedef struct {
  VP8MB left_;
  VP8MB info_;
  uint8_t intra_t_[4];
  uint8_t intra_l_[4];
  VP8BitReader br_;
  VP8BitReader token_br_;
} MBContext;

//------------------------------------------------------------------------------
// MemBuffer: incoming data handling

static void RemapBitReader(VP8BitReader* const br, ptrdiff_t offset) {
  if (br->buf_ != NULL) {
    br->buf_ += offset;
    br->buf_end_ += offset;
  }
}

static WEBP_INLINE size_t MemDataSize(const MemBuffer* mem) {
  return (mem->end_ - mem->start_);
}

static void DoRemap(WebPIDecoder* const idec, ptrdiff_t offset) {
  MemBuffer* const mem = &idec->mem_;
  const uint8_t* const new_base = mem->buf_ + mem->start_;
  // note: for VP8, setting up idec->io_ is only really needed at the beginning
  // of the decoding, till partition #0 is complete.
  idec->io_.data = new_base;
  idec->io_.data_size = MemDataSize(mem);

  if (idec->dec_ != NULL) {
    if (!idec->is_lossless_) {
      VP8Decoder* const dec = (VP8Decoder*)idec->dec_;
      const int last_part = dec->num_parts_ - 1;
      if (offset != 0) {
        int p;
        for (p = 0; p <= last_part; ++p) {
          RemapBitReader(dec->parts_ + p, offset);
        }
        // Remap partition #0 data pointer to new offset, but only in MAP
        // mode (in APPEND mode, partition #0 is copied into a fixed memory).
        if (mem->mode_ == MEM_MODE_MAP) {
          RemapBitReader(&dec->br_, offset);
        }
      }
      assert(last_part >= 0);
      dec->parts_[last_part].buf_end_ = mem->buf_ + mem->end_;
    } else {    // Resize lossless bitreader
      VP8LDecoder* const dec = (VP8LDecoder*)idec->dec_;
      VP8LBitReaderSetBuffer(&dec->br_, new_base, MemDataSize(mem));
    }
  }
}

// Appends data to the end of MemBuffer->buf_. It expands the allocated memory
// size if required and also updates VP8BitReader's if new memory is allocated.
static int AppendToMemBuffer(WebPIDecoder* const idec,
                             const uint8_t* const data, size_t data_size) {
  MemBuffer* const mem = &idec->mem_;
  const uint8_t* const old_base = mem->buf_ + mem->start_;
  assert(mem->mode_ == MEM_MODE_APPEND);
  if (data_size > MAX_CHUNK_PAYLOAD) {
    // security safeguard: trying to allocate more than what the format
    // allows for a chunk should be considered a smoke smell.
    return 0;
  }

  if (mem->end_ + data_size > mem->buf_size_) {  // Need some free memory
    const size_t current_size = MemDataSize(mem);
    const uint64_t new_size = (uint64_t)current_size + data_size;
    const uint64_t extra_size = (new_size + CHUNK_SIZE - 1) & ~(CHUNK_SIZE - 1);
    uint8_t* const new_buf =
        (uint8_t*)WebPSafeMalloc(extra_size, sizeof(*new_buf));
    if (new_buf == NULL) return 0;
    memcpy(new_buf, old_base, current_size);
    free(mem->buf_);
    mem->buf_ = new_buf;
    mem->buf_size_ = (size_t)extra_size;
    mem->start_ = 0;
    mem->end_ = current_size;
  }

  memcpy(mem->buf_ + mem->end_, data, data_size);
  mem->end_ += data_size;
  assert(mem->end_ <= mem->buf_size_);

  DoRemap(idec, mem->buf_ + mem->start_ - old_base);
  return 1;
}

static int RemapMemBuffer(WebPIDecoder* const idec,
                          const uint8_t* const data, size_t data_size) {
  MemBuffer* const mem = &idec->mem_;
  const uint8_t* const old_base = mem->buf_ + mem->start_;
  assert(mem->mode_ == MEM_MODE_MAP);

  if (data_size < mem->buf_size_) return 0;  // can't remap to a shorter buffer!

  mem->buf_ = (uint8_t*)data;
  mem->end_ = mem->buf_size_ = data_size;

  DoRemap(idec, mem->buf_ + mem->start_ - old_base);
  return 1;
}

static void InitMemBuffer(MemBuffer* const mem) {
  mem->mode_       = MEM_MODE_NONE;
  mem->buf_        = NULL;
  mem->buf_size_   = 0;
  mem->part0_buf_  = NULL;
  mem->part0_size_ = 0;
}

static void ClearMemBuffer(MemBuffer* const mem) {
  assert(mem);
  if (mem->mode_ == MEM_MODE_APPEND) {
    free(mem->buf_);
    free((void*)mem->part0_buf_);
  }
}

static int CheckMemBufferMode(MemBuffer* const mem, MemBufferMode expected) {
  if (mem->mode_ == MEM_MODE_NONE) {
    mem->mode_ = expected;    // switch to the expected mode
  } else if (mem->mode_ != expected) {
    return 0;         // we mixed the modes => error
  }
  assert(mem->mode_ == expected);   // mode is ok
  return 1;
}

//------------------------------------------------------------------------------
// Macroblock-decoding contexts

static void SaveContext(const VP8Decoder* dec, const VP8BitReader* token_br,
                        MBContext* const context) {
  const VP8BitReader* const br = &dec->br_;
  const VP8MB* const left = dec->mb_info_ - 1;
  const VP8MB* const info = dec->mb_info_ + dec->mb_x_;

  context->left_ = *left;
  context->info_ = *info;
  context->br_ = *br;
  context->token_br_ = *token_br;
  memcpy(context->intra_t_, dec->intra_t_ + 4 * dec->mb_x_, 4);
  memcpy(context->intra_l_, dec->intra_l_, 4);
}

static void RestoreContext(const MBContext* context, VP8Decoder* const dec,
                           VP8BitReader* const token_br) {
  VP8BitReader* const br = &dec->br_;
  VP8MB* const left = dec->mb_info_ - 1;
  VP8MB* const info = dec->mb_info_ + dec->mb_x_;

  *left = context->left_;
  *info = context->info_;
  *br = context->br_;
  *token_br = context->token_br_;
  memcpy(dec->intra_t_ + 4 * dec->mb_x_, context->intra_t_, 4);
  memcpy(dec->intra_l_, context->intra_l_, 4);
}

//------------------------------------------------------------------------------

static VP8StatusCode IDecError(WebPIDecoder* const idec, VP8StatusCode error) {
  if (idec->state_ == STATE_VP8_DATA) {
    VP8Io* const io = &idec->io_;
    if (io->teardown) {
      io->teardown(io);
    }
  }
  idec->state_ = STATE_ERROR;
  return error;
}

static void ChangeState(WebPIDecoder* const idec, DecState new_state,
                        size_t consumed_bytes) {
  MemBuffer* const mem = &idec->mem_;
  idec->state_ = new_state;
  mem->start_ += consumed_bytes;
  assert(mem->start_ <= mem->end_);
  idec->io_.data = mem->buf_ + mem->start_;
  idec->io_.data_size = MemDataSize(mem);
}

// Headers
static VP8StatusCode DecodeWebPHeaders(WebPIDecoder* const idec) {
  MemBuffer* const mem = &idec->mem_;
  const uint8_t* data = mem->buf_ + mem->start_;
  size_t curr_size = MemDataSize(mem);
  VP8StatusCode status;
  WebPHeaderStructure headers;

  headers.data = data;
  headers.data_size = curr_size;
  status = WebPParseHeaders(&headers);
  if (status == VP8_STATUS_NOT_ENOUGH_DATA) {
    return VP8_STATUS_SUSPENDED;  // We haven't found a VP8 chunk yet.
  } else if (status != VP8_STATUS_OK) {
    return IDecError(idec, status);
  }

  idec->chunk_size_ = headers.compressed_size;
  idec->is_lossless_ = headers.is_lossless;
  if (!idec->is_lossless_) {
    VP8Decoder* const dec = VP8New();
    if (dec == NULL) {
      return VP8_STATUS_OUT_OF_MEMORY;
    }
    idec->dec_ = dec;
#ifdef WEBP_USE_THREAD
    dec->use_threads_ = (idec->params_.options != NULL) &&
                        (idec->params_.options->use_threads > 0);
#else
    dec->use_threads_ = 0;
#endif
    dec->alpha_data_ = headers.alpha_data;
    dec->alpha_data_size_ = headers.alpha_data_size;
    ChangeState(idec, STATE_VP8_FRAME_HEADER, headers.offset);
  } else {
    VP8LDecoder* const dec = VP8LNew();
    if (dec == NULL) {
      return VP8_STATUS_OUT_OF_MEMORY;
    }
    idec->dec_ = dec;
    ChangeState(idec, STATE_VP8L_HEADER, headers.offset);
  }
  return VP8_STATUS_OK;
}

static VP8StatusCode DecodeVP8FrameHeader(WebPIDecoder* const idec) {
  const uint8_t* data = idec->mem_.buf_ + idec->mem_.start_;
  const size_t curr_size = MemDataSize(&idec->mem_);
  uint32_t bits;

  if (curr_size < VP8_FRAME_HEADER_SIZE) {
    // Not enough data bytes to extract VP8 Frame Header.
    return VP8_STATUS_SUSPENDED;
  }
  if (!VP8GetInfo(data, curr_size, idec->chunk_size_, NULL, NULL)) {
    return IDecError(idec, VP8_STATUS_BITSTREAM_ERROR);
  }

  bits = data[0] | (data[1] << 8) | (data[2] << 16);
  idec->mem_.part0_size_ = (bits >> 5) + VP8_FRAME_HEADER_SIZE;

  idec->io_.data = data;
  idec->io_.data_size = curr_size;
  idec->state_ = STATE_VP8_PARTS0;
  return VP8_STATUS_OK;
}

// Partition #0
static int CopyParts0Data(WebPIDecoder* const idec) {
  VP8Decoder* const dec = (VP8Decoder*)idec->dec_;
  VP8BitReader* const br = &dec->br_;
  const size_t psize = br->buf_end_ - br->buf_;
  MemBuffer* const mem = &idec->mem_;
  assert(!idec->is_lossless_);
  assert(mem->part0_buf_ == NULL);
  assert(psize > 0);
  assert(psize <= mem->part0_size_);  // Format limit: no need for runtime check
  if (mem->mode_ == MEM_MODE_APPEND) {
    // We copy and grab ownership of the partition #0 data.
    uint8_t* const part0_buf = (uint8_t*)malloc(psize);
    if (part0_buf == NULL) {
      return 0;
    }
    memcpy(part0_buf, br->buf_, psize);
    mem->part0_buf_ = part0_buf;
    br->buf_ = part0_buf;
    br->buf_end_ = part0_buf + psize;
  } else {
    // Else: just keep pointers to the partition #0's data in dec_->br_.
  }
  mem->start_ += psize;
  return 1;
}

static VP8StatusCode DecodePartition0(WebPIDecoder* const idec) {
  VP8Decoder* const dec = (VP8Decoder*)idec->dec_;
  VP8Io* const io = &idec->io_;
  const WebPDecParams* const params = &idec->params_;
  WebPDecBuffer* const output = params->output;

  // Wait till we have enough data for the whole partition #0
  if (MemDataSize(&idec->mem_) < idec->mem_.part0_size_) {
    return VP8_STATUS_SUSPENDED;
  }

  if (!VP8GetHeaders(dec, io)) {
    const VP8StatusCode status = dec->status_;
    if (status == VP8_STATUS_SUSPENDED ||
        status == VP8_STATUS_NOT_ENOUGH_DATA) {
      // treating NOT_ENOUGH_DATA as SUSPENDED state
      return VP8_STATUS_SUSPENDED;
    }
    return IDecError(idec, status);
  }

  // Allocate/Verify output buffer now
  dec->status_ = WebPAllocateDecBuffer(io->width, io->height, params->options,
                                       output);
  if (dec->status_ != VP8_STATUS_OK) {
    return IDecError(idec, dec->status_);
  }

  if (!CopyParts0Data(idec)) {
    return IDecError(idec, VP8_STATUS_OUT_OF_MEMORY);
  }

  // Finish setting up the decoding parameters. Will call io->setup().
  if (VP8EnterCritical(dec, io) != VP8_STATUS_OK) {
    return IDecError(idec, dec->status_);
  }

  // Note: past this point, teardown() must always be called
  // in case of error.
  idec->state_ = STATE_VP8_DATA;
  // Allocate memory and prepare everything.
  if (!VP8InitFrame(dec, io)) {
    return IDecError(idec, dec->status_);
  }
  return VP8_STATUS_OK;
}

// Remaining partitions
static VP8StatusCode DecodeRemaining(WebPIDecoder* const idec) {
  VP8Decoder* const dec = (VP8Decoder*)idec->dec_;
  VP8Io* const io = &idec->io_;

  assert(dec->ready_);

  for (; dec->mb_y_ < dec->mb_h_; ++dec->mb_y_) {
    VP8BitReader* token_br = &dec->parts_[dec->mb_y_ & (dec->num_parts_ - 1)];
    if (dec->mb_x_ == 0) {
      VP8InitScanline(dec);
    }
    for (; dec->mb_x_ < dec->mb_w_;  dec->mb_x_++) {
      MBContext context;
      SaveContext(dec, token_br, &context);

      if (!VP8DecodeMB(dec, token_br)) {
        RestoreContext(&context, dec, token_br);
        // We shouldn't fail when MAX_MB data was available
        if (dec->num_parts_ == 1 && MemDataSize(&idec->mem_) > MAX_MB_SIZE) {
          return IDecError(idec, VP8_STATUS_BITSTREAM_ERROR);
        }
        return VP8_STATUS_SUSPENDED;
      }
      VP8ReconstructBlock(dec);
      // Store data and save block's filtering params
      VP8StoreBlock(dec);

      // Release buffer only if there is only one partition
      if (dec->num_parts_ == 1) {
        idec->mem_.start_ = token_br->buf_ - idec->mem_.buf_;
        assert(idec->mem_.start_ <= idec->mem_.end_);
      }
    }
    if (!VP8ProcessRow(dec, io)) {
      return IDecError(idec, VP8_STATUS_USER_ABORT);
    }
    dec->mb_x_ = 0;
  }
  // Synchronize the thread and check for errors.
  if (!VP8ExitCritical(dec, io)) {
    return IDecError(idec, VP8_STATUS_USER_ABORT);
  }
  dec->ready_ = 0;
  idec->state_ = STATE_DONE;

  return VP8_STATUS_OK;
}

static int ErrorStatusLossless(WebPIDecoder* const idec, VP8StatusCode status) {
  if (status == VP8_STATUS_SUSPENDED || status == VP8_STATUS_NOT_ENOUGH_DATA) {
    return VP8_STATUS_SUSPENDED;
  }
  return IDecError(idec, status);
}

static VP8StatusCode DecodeVP8LHeader(WebPIDecoder* const idec) {
  VP8Io* const io = &idec->io_;
  VP8LDecoder* const dec = (VP8LDecoder*)idec->dec_;
  const WebPDecParams* const params = &idec->params_;
  WebPDecBuffer* const output = params->output;
  size_t curr_size = MemDataSize(&idec->mem_);
  assert(idec->is_lossless_);

  // Wait until there's enough data for decoding header.
  if (curr_size < (idec->chunk_size_ >> 3)) {
    return VP8_STATUS_SUSPENDED;
  }
  if (!VP8LDecodeHeader(dec, io)) {
    return ErrorStatusLossless(idec, dec->status_);
  }
  // Allocate/verify output buffer now.
  dec->status_ = WebPAllocateDecBuffer(io->width, io->height, params->options,
                                       output);
  if (dec->status_ != VP8_STATUS_OK) {
    return IDecError(idec, dec->status_);
  }

  idec->state_ = STATE_VP8L_DATA;
  return VP8_STATUS_OK;
}

static VP8StatusCode DecodeVP8LData(WebPIDecoder* const idec) {
  VP8LDecoder* const dec = (VP8LDecoder*)idec->dec_;
  const size_t curr_size = MemDataSize(&idec->mem_);
  assert(idec->is_lossless_);

  // At present Lossless decoder can't decode image incrementally. So wait till
  // all the image data is aggregated before image can be decoded.
  if (curr_size < idec->chunk_size_) {
    return VP8_STATUS_SUSPENDED;
  }

  if (!VP8LDecodeImage(dec)) {
    return ErrorStatusLossless(idec, dec->status_);
  }

  idec->state_ = STATE_DONE;

  return VP8_STATUS_OK;
}

  // Main decoding loop
static VP8StatusCode IDecode(WebPIDecoder* idec) {
  VP8StatusCode status = VP8_STATUS_SUSPENDED;

  if (idec->state_ == STATE_PRE_VP8) {
    status = DecodeWebPHeaders(idec);
  } else {
    if (idec->dec_ == NULL) {
      return VP8_STATUS_SUSPENDED;    // can't continue if we have no decoder.
    }
  }
  if (idec->state_ == STATE_VP8_FRAME_HEADER) {
    status = DecodeVP8FrameHeader(idec);
  }
  if (idec->state_ == STATE_VP8_PARTS0) {
    status = DecodePartition0(idec);
  }
  if (idec->state_ == STATE_VP8_DATA) {
    status = DecodeRemaining(idec);
  }
  if (idec->state_ == STATE_VP8L_HEADER) {
    status = DecodeVP8LHeader(idec);
  }
  if (idec->state_ == STATE_VP8L_DATA) {
    status = DecodeVP8LData(idec);
  }
  return status;
}

//------------------------------------------------------------------------------
// Public functions

WebPIDecoder* WebPINewDecoder(WebPDecBuffer* output_buffer) {
  WebPIDecoder* idec = (WebPIDecoder*)calloc(1, sizeof(*idec));
  if (idec == NULL) {
    return NULL;
  }

  idec->state_ = STATE_PRE_VP8;
  idec->chunk_size_ = 0;

  InitMemBuffer(&idec->mem_);
  WebPInitDecBuffer(&idec->output_);
  VP8InitIo(&idec->io_);

  WebPResetDecParams(&idec->params_);
  idec->params_.output = output_buffer ? output_buffer : &idec->output_;
  WebPInitCustomIo(&idec->params_, &idec->io_);  // Plug the I/O functions.

  return idec;
}

WebPIDecoder* WebPIDecode(const uint8_t* data, size_t data_size,
                          WebPDecoderConfig* config) {
  WebPIDecoder* idec;

  // Parse the bitstream's features, if requested:
  if (data != NULL && data_size > 0 && config != NULL) {
    if (WebPGetFeatures(data, data_size, &config->input) != VP8_STATUS_OK) {
      return NULL;
    }
  }
  // Create an instance of the incremental decoder
  idec = WebPINewDecoder(config ? &config->output : NULL);
  if (idec == NULL) {
    return NULL;
  }
  // Finish initialization
  if (config != NULL) {
    idec->params_.options = &config->options;
  }
  return idec;
}

void WebPIDelete(WebPIDecoder* idec) {
  if (idec == NULL) return;
  if (idec->dec_ != NULL) {
    if (!idec->is_lossless_) {
      VP8Delete(idec->dec_);
    } else {
      VP8LDelete(idec->dec_);
    }
  }
  ClearMemBuffer(&idec->mem_);
  WebPFreeDecBuffer(&idec->output_);
  free(idec);
}

//------------------------------------------------------------------------------
// Wrapper toward WebPINewDecoder

WebPIDecoder* WebPINewRGB(WEBP_CSP_MODE mode, uint8_t* output_buffer,
                          size_t output_buffer_size, int output_stride) {
  WebPIDecoder* idec;
  if (mode >= MODE_YUV) return NULL;
  idec = WebPINewDecoder(NULL);
  if (idec == NULL) return NULL;
  idec->output_.colorspace = mode;
  idec->output_.is_external_memory = 1;
  idec->output_.u.RGBA.rgba = output_buffer;
  idec->output_.u.RGBA.stride = output_stride;
  idec->output_.u.RGBA.size = output_buffer_size;
  return idec;
}

WebPIDecoder* WebPINewYUVA(uint8_t* luma, size_t luma_size, int luma_stride,
                           uint8_t* u, size_t u_size, int u_stride,
                           uint8_t* v, size_t v_size, int v_stride,
                           uint8_t* a, size_t a_size, int a_stride) {
  WebPIDecoder* const idec = WebPINewDecoder(NULL);
  if (idec == NULL) return NULL;
  idec->output_.colorspace = (a == NULL) ? MODE_YUV : MODE_YUVA;
  idec->output_.is_external_memory = 1;
  idec->output_.u.YUVA.y = luma;
  idec->output_.u.YUVA.y_stride = luma_stride;
  idec->output_.u.YUVA.y_size = luma_size;
  idec->output_.u.YUVA.u = u;
  idec->output_.u.YUVA.u_stride = u_stride;
  idec->output_.u.YUVA.u_size = u_size;
  idec->output_.u.YUVA.v = v;
  idec->output_.u.YUVA.v_stride = v_stride;
  idec->output_.u.YUVA.v_size = v_size;
  idec->output_.u.YUVA.a = a;
  idec->output_.u.YUVA.a_stride = a_stride;
  idec->output_.u.YUVA.a_size = a_size;
  return idec;
}

WebPIDecoder* WebPINewYUV(uint8_t* luma, size_t luma_size, int luma_stride,
                          uint8_t* u, size_t u_size, int u_stride,
                          uint8_t* v, size_t v_size, int v_stride) {
  return WebPINewYUVA(luma, luma_size, luma_stride,
                      u, u_size, u_stride,
                      v, v_size, v_stride,
                      NULL, 0, 0);
}

//------------------------------------------------------------------------------

static VP8StatusCode IDecCheckStatus(const WebPIDecoder* const idec) {
  assert(idec);
  if (idec->state_ == STATE_ERROR) {
    return VP8_STATUS_BITSTREAM_ERROR;
  }
  if (idec->state_ == STATE_DONE) {
    return VP8_STATUS_OK;
  }
  return VP8_STATUS_SUSPENDED;
}

VP8StatusCode WebPIAppend(WebPIDecoder* idec,
                          const uint8_t* data, size_t data_size) {
  VP8StatusCode status;
  if (idec == NULL || data == NULL) {
    return VP8_STATUS_INVALID_PARAM;
  }
  status = IDecCheckStatus(idec);
  if (status != VP8_STATUS_SUSPENDED) {
    return status;
  }
  // Check mixed calls between RemapMemBuffer and AppendToMemBuffer.
  if (!CheckMemBufferMode(&idec->mem_, MEM_MODE_APPEND)) {
    return VP8_STATUS_INVALID_PARAM;
  }
  // Append data to memory buffer
  if (!AppendToMemBuffer(idec, data, data_size)) {
    return VP8_STATUS_OUT_OF_MEMORY;
  }
  return IDecode(idec);
}

VP8StatusCode WebPIUpdate(WebPIDecoder* idec,
                          const uint8_t* data, size_t data_size) {
  VP8StatusCode status;
  if (idec == NULL || data == NULL) {
    return VP8_STATUS_INVALID_PARAM;
  }
  status = IDecCheckStatus(idec);
  if (status != VP8_STATUS_SUSPENDED) {
    return status;
  }
  // Check mixed calls between RemapMemBuffer and AppendToMemBuffer.
  if (!CheckMemBufferMode(&idec->mem_, MEM_MODE_MAP)) {
    return VP8_STATUS_INVALID_PARAM;
  }
  // Make the memory buffer point to the new buffer
  if (!RemapMemBuffer(idec, data, data_size)) {
    return VP8_STATUS_INVALID_PARAM;
  }
  return IDecode(idec);
}

//------------------------------------------------------------------------------

static const WebPDecBuffer* GetOutputBuffer(const WebPIDecoder* const idec) {
  if (idec == NULL || idec->dec_ == NULL) {
    return NULL;
  }
  if (idec->state_ <= STATE_VP8_PARTS0) {
    return NULL;
  }
  return idec->params_.output;
}

const WebPDecBuffer* WebPIDecodedArea(const WebPIDecoder* idec,
                                      int* left, int* top,
                                      int* width, int* height) {
  const WebPDecBuffer* const src = GetOutputBuffer(idec);
  if (left != NULL) *left = 0;
  if (top != NULL) *top = 0;
  // TODO(skal): later include handling of rotations.
  if (src) {
    if (width != NULL) *width = src->width;
    if (height != NULL) *height = idec->params_.last_y;
  } else {
    if (width != NULL) *width = 0;
    if (height != NULL) *height = 0;
  }
  return src;
}

uint8_t* WebPIDecGetRGB(const WebPIDecoder* idec, int* last_y,
                        int* width, int* height, int* stride) {
  const WebPDecBuffer* const src = GetOutputBuffer(idec);
  if (src == NULL) return NULL;
  if (src->colorspace >= MODE_YUV) {
    return NULL;
  }

  if (last_y != NULL) *last_y = idec->params_.last_y;
  if (width != NULL) *width = src->width;
  if (height != NULL) *height = src->height;
  if (stride != NULL) *stride = src->u.RGBA.stride;

  return src->u.RGBA.rgba;
}

uint8_t* WebPIDecGetYUVA(const WebPIDecoder* idec, int* last_y,
                         uint8_t** u, uint8_t** v, uint8_t** a,
                         int* width, int* height,
                         int* stride, int* uv_stride, int* a_stride) {
  const WebPDecBuffer* const src = GetOutputBuffer(idec);
  if (src == NULL) return NULL;
  if (src->colorspace < MODE_YUV) {
    return NULL;
  }

  if (last_y != NULL) *last_y = idec->params_.last_y;
  if (u != NULL) *u = src->u.YUVA.u;
  if (v != NULL) *v = src->u.YUVA.v;
  if (a != NULL) *a = src->u.YUVA.a;
  if (width != NULL) *width = src->width;
  if (height != NULL) *height = src->height;
  if (stride != NULL) *stride = src->u.YUVA.y_stride;
  if (uv_stride != NULL) *uv_stride = src->u.YUVA.u_stride;
  if (a_stride != NULL) *a_stride = src->u.YUVA.a_stride;

  return src->u.YUVA.y;
}

int WebPISetIOHooks(WebPIDecoder* const idec,
                    VP8IoPutHook put,
                    VP8IoSetupHook setup,
                    VP8IoTeardownHook teardown,
                    void* user_data) {
  if (idec == NULL || idec->state_ > STATE_PRE_VP8) {
    return 0;
  }

  idec->io_.put = put;
  idec->io_.setup = setup;
  idec->io_.teardown = teardown;
  idec->io_.opaque = user_data;

  return 1;
}

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

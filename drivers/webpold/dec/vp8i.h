// Copyright 2010 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// VP8 decoder: internal header.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef WEBP_DEC_VP8I_H_
#define WEBP_DEC_VP8I_H_

#include <string.h>     // for memcpy()
#include "./vp8li.h"
#include "../utils/bit_reader.h"
#include "../utils/thread.h"
#include "../dsp/dsp.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------
// Various defines and enums

// version numbers
#define DEC_MAJ_VERSION 0
#define DEC_MIN_VERSION 2
#define DEC_REV_VERSION 0

#define ONLY_KEYFRAME_CODE      // to remove any code related to P-Frames

// intra prediction modes
enum { B_DC_PRED = 0,   // 4x4 modes
       B_TM_PRED,
       B_VE_PRED,
       B_HE_PRED,
       B_RD_PRED,
       B_VR_PRED,
       B_LD_PRED,
       B_VL_PRED,
       B_HD_PRED,
       B_HU_PRED,
       NUM_BMODES = B_HU_PRED + 1 - B_DC_PRED,  // = 10

       // Luma16 or UV modes
       DC_PRED = B_DC_PRED, V_PRED = B_VE_PRED,
       H_PRED = B_HE_PRED, TM_PRED = B_TM_PRED,
       B_PRED = NUM_BMODES,   // refined I4x4 mode

       // special modes
       B_DC_PRED_NOTOP = 4,
       B_DC_PRED_NOLEFT = 5,
       B_DC_PRED_NOTOPLEFT = 6,
       NUM_B_DC_MODES = 7 };

enum { MB_FEATURE_TREE_PROBS = 3,
       NUM_MB_SEGMENTS = 4,
       NUM_REF_LF_DELTAS = 4,
       NUM_MODE_LF_DELTAS = 4,    // I4x4, ZERO, *, SPLIT
       MAX_NUM_PARTITIONS = 8,
       // Probabilities
       NUM_TYPES = 4,
       NUM_BANDS = 8,
       NUM_CTX = 3,
       NUM_PROBAS = 11,
       NUM_MV_PROBAS = 19 };

// YUV-cache parameters.
// Constraints are: We need to store one 16x16 block of luma samples (y),
// and two 8x8 chroma blocks (u/v). These are better be 16-bytes aligned,
// in order to be SIMD-friendly. We also need to store the top, left and
// top-left samples (from previously decoded blocks), along with four
// extra top-right samples for luma (intra4x4 prediction only).
// One possible layout is, using 32 * (17 + 9) bytes:
//
//   .+------   <- only 1 pixel high
//   .|yyyyt.
//   .|yyyyt.
//   .|yyyyt.
//   .|yyyy..
//   .+--.+--   <- only 1 pixel high
//   .|uu.|vv
//   .|uu.|vv
//
// Every character is a 4x4 block, with legend:
//  '.' = unused
//  'y' = y-samples   'u' = u-samples     'v' = u-samples
//  '|' = left sample,   '-' = top sample,    '+' = top-left sample
//  't' = extra top-right sample for 4x4 modes
// With this layout, BPS (=Bytes Per Scan-line) is one cacheline size.
#define BPS       32    // this is the common stride used by yuv[]
#define YUV_SIZE (BPS * 17 + BPS * 9)
#define Y_SIZE   (BPS * 17)
#define Y_OFF    (BPS * 1 + 8)
#define U_OFF    (Y_OFF + BPS * 16 + BPS)
#define V_OFF    (U_OFF + 16)

//------------------------------------------------------------------------------
// Headers

typedef struct {
  uint8_t key_frame_;
  uint8_t profile_;
  uint8_t show_;
  uint32_t partition_length_;
} VP8FrameHeader;

typedef struct {
  uint16_t width_;
  uint16_t height_;
  uint8_t xscale_;
  uint8_t yscale_;
  uint8_t colorspace_;   // 0 = YCbCr
  uint8_t clamp_type_;
} VP8PictureHeader;

// segment features
typedef struct {
  int use_segment_;
  int update_map_;        // whether to update the segment map or not
  int absolute_delta_;    // absolute or delta values for quantizer and filter
  int8_t quantizer_[NUM_MB_SEGMENTS];        // quantization changes
  int8_t filter_strength_[NUM_MB_SEGMENTS];  // filter strength for segments
} VP8SegmentHeader;

// Struct collecting all frame-persistent probabilities.
typedef struct {
  uint8_t segments_[MB_FEATURE_TREE_PROBS];
  // Type: 0:Intra16-AC  1:Intra16-DC   2:Chroma   3:Intra4
  uint8_t coeffs_[NUM_TYPES][NUM_BANDS][NUM_CTX][NUM_PROBAS];
#ifndef ONLY_KEYFRAME_CODE
  uint8_t ymode_[4], uvmode_[3];
  uint8_t mv_[2][NUM_MV_PROBAS];
#endif
} VP8Proba;

// Filter parameters
typedef struct {
  int simple_;                  // 0=complex, 1=simple
  int level_;                   // [0..63]
  int sharpness_;               // [0..7]
  int use_lf_delta_;
  int ref_lf_delta_[NUM_REF_LF_DELTAS];
  int mode_lf_delta_[NUM_MODE_LF_DELTAS];
} VP8FilterHeader;

//------------------------------------------------------------------------------
// Informations about the macroblocks.

typedef struct {  // filter specs
  unsigned int f_level_:6;      // filter strength: 0..63
  unsigned int f_ilevel_:6;     // inner limit: 1..63
  unsigned int f_inner_:1;      // do inner filtering?
} VP8FInfo;

typedef struct {  // used for syntax-parsing
  unsigned int nz_;          // non-zero AC/DC coeffs
  unsigned int dc_nz_:1;     // non-zero DC coeffs
  unsigned int skip_:1;      // block type
} VP8MB;

// Dequantization matrices
typedef int quant_t[2];      // [DC / AC].  Can be 'uint16_t[2]' too (~slower).
typedef struct {
  quant_t y1_mat_, y2_mat_, uv_mat_;
} VP8QuantMatrix;

// Persistent information needed by the parallel processing
typedef struct {
  int id_;            // cache row to process (in [0..2])
  int mb_y_;          // macroblock position of the row
  int filter_row_;    // true if row-filtering is needed
  VP8FInfo* f_info_;  // filter strengths
  VP8Io io_;          // copy of the VP8Io to pass to put()
} VP8ThreadContext;

//------------------------------------------------------------------------------
// VP8Decoder: the main opaque structure handed over to user

struct VP8Decoder {
  VP8StatusCode status_;
  int ready_;     // true if ready to decode a picture with VP8Decode()
  const char* error_msg_;  // set when status_ is not OK.

  // Main data source
  VP8BitReader br_;

  // headers
  VP8FrameHeader   frm_hdr_;
  VP8PictureHeader pic_hdr_;
  VP8FilterHeader  filter_hdr_;
  VP8SegmentHeader segment_hdr_;

  // Worker
  WebPWorker worker_;
  int use_threads_;    // use multi-thread
  int cache_id_;       // current cache row
  int num_caches_;     // number of cached rows of 16 pixels (1, 2 or 3)
  VP8ThreadContext thread_ctx_;  // Thread context

  // dimension, in macroblock units.
  int mb_w_, mb_h_;

  // Macroblock to process/filter, depending on cropping and filter_type.
  int tl_mb_x_, tl_mb_y_;  // top-left MB that must be in-loop filtered
  int br_mb_x_, br_mb_y_;  // last bottom-right MB that must be decoded

  // number of partitions.
  int num_parts_;
  // per-partition boolean decoders.
  VP8BitReader parts_[MAX_NUM_PARTITIONS];

  // buffer refresh flags
  //   bit 0: refresh Gold, bit 1: refresh Alt
  //   bit 2-3: copy to Gold, bit 4-5: copy to Alt
  //   bit 6: Gold sign bias, bit 7: Alt sign bias
  //   bit 8: refresh last frame
  uint32_t buffer_flags_;

  // dequantization (one set of DC/AC dequant factor per segment)
  VP8QuantMatrix dqm_[NUM_MB_SEGMENTS];

  // probabilities
  VP8Proba proba_;
  int use_skip_proba_;
  uint8_t skip_p_;
#ifndef ONLY_KEYFRAME_CODE
  uint8_t intra_p_, last_p_, golden_p_;
  VP8Proba proba_saved_;
  int update_proba_;
#endif

  // Boundary data cache and persistent buffers.
  uint8_t* intra_t_;     // top intra modes values: 4 * mb_w_
  uint8_t  intra_l_[4];  // left intra modes values
  uint8_t* y_t_;         // top luma samples: 16 * mb_w_
  uint8_t* u_t_, *v_t_;  // top u/v samples: 8 * mb_w_ each

  VP8MB* mb_info_;       // contextual macroblock info (mb_w_ + 1)
  VP8FInfo* f_info_;     // filter strength info
  uint8_t* yuv_b_;       // main block for Y/U/V (size = YUV_SIZE)
  int16_t* coeffs_;      // 384 coeffs = (16+8+8) * 4*4

  uint8_t* cache_y_;     // macroblock row for storing unfiltered samples
  uint8_t* cache_u_;
  uint8_t* cache_v_;
  int cache_y_stride_;
  int cache_uv_stride_;

  // main memory chunk for the above data. Persistent.
  void* mem_;
  size_t mem_size_;

  // Per macroblock non-persistent infos.
  int mb_x_, mb_y_;       // current position, in macroblock units
  uint8_t is_i4x4_;       // true if intra4x4
  uint8_t imodes_[16];    // one 16x16 mode (#0) or sixteen 4x4 modes
  uint8_t uvmode_;        // chroma prediction mode
  uint8_t segment_;       // block's segment

  // bit-wise info about the content of each sub-4x4 blocks: there are 16 bits
  // for luma (bits #0->#15), then 4 bits for chroma-u (#16->#19) and 4 bits for
  // chroma-v (#20->#23), each corresponding to one 4x4 block in decoding order.
  // If the bit is set, the 4x4 block contains some non-zero coefficients.
  uint32_t non_zero_;
  uint32_t non_zero_ac_;

  // Filtering side-info
  int filter_type_;                         // 0=off, 1=simple, 2=complex
  int filter_row_;                          // per-row flag
  uint8_t filter_levels_[NUM_MB_SEGMENTS];  // precalculated per-segment

  // extensions
  const uint8_t* alpha_data_;   // compressed alpha data (if present)
  size_t alpha_data_size_;
  uint8_t* alpha_plane_;        // output. Persistent, contains the whole data.

  int layer_colorspace_;
  const uint8_t* layer_data_;   // compressed layer data (if present)
  size_t layer_data_size_;
};

//------------------------------------------------------------------------------
// internal functions. Not public.

// in vp8.c
int VP8SetError(VP8Decoder* const dec,
                VP8StatusCode error, const char* const msg);

// in tree.c
void VP8ResetProba(VP8Proba* const proba);
void VP8ParseProba(VP8BitReader* const br, VP8Decoder* const dec);
void VP8ParseIntraMode(VP8BitReader* const br,  VP8Decoder* const dec);

// in quant.c
void VP8ParseQuant(VP8Decoder* const dec);

// in frame.c
int VP8InitFrame(VP8Decoder* const dec, VP8Io* io);
// Predict a block and add residual
void VP8ReconstructBlock(VP8Decoder* const dec);
// Call io->setup() and finish setting up scan parameters.
// After this call returns, one must always call VP8ExitCritical() with the
// same parameters. Both functions should be used in pair. Returns VP8_STATUS_OK
// if ok, otherwise sets and returns the error status on *dec.
VP8StatusCode VP8EnterCritical(VP8Decoder* const dec, VP8Io* const io);
// Must always be called in pair with VP8EnterCritical().
// Returns false in case of error.
int VP8ExitCritical(VP8Decoder* const dec, VP8Io* const io);
// Process the last decoded row (filtering + output)
int VP8ProcessRow(VP8Decoder* const dec, VP8Io* const io);
// Store a block, along with filtering params
void VP8StoreBlock(VP8Decoder* const dec);
// To be called at the start of a new scanline, to initialize predictors.
void VP8InitScanline(VP8Decoder* const dec);
// Decode one macroblock. Returns false if there is not enough data.
int VP8DecodeMB(VP8Decoder* const dec, VP8BitReader* const token_br);

// in alpha.c
const uint8_t* VP8DecompressAlphaRows(VP8Decoder* const dec,
                                      int row, int num_rows);

// in layer.c
int VP8DecodeLayer(VP8Decoder* const dec);

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

#endif  /* WEBP_DEC_VP8I_H_ */

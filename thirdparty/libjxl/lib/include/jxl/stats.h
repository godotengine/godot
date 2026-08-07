/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_encoder
 * @{
 * @file stats.h
 * @brief API to collect various statistics from JXL encoder.
 */

#ifndef JXL_STATS_H_
#define JXL_STATS_H_

#include <jxl/jxl_export.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque structure that holds the encoder statistics.
 *
 * Allocated and initialized with @ref JxlEncoderStatsCreate().
 * Cleaned up and deallocated with @ref JxlEncoderStatsDestroy().
 */
typedef struct JxlEncoderStatsStruct JxlEncoderStats;

/**
 * Creates an instance of JxlEncoderStats and initializes it.
 *
 * @return pointer to initialized @ref JxlEncoderStats instance
 */
JXL_EXPORT JxlEncoderStats* JxlEncoderStatsCreate(void);

/**
 * Deinitializes and frees JxlEncoderStats instance.
 *
 * @param stats instance to be cleaned up and deallocated. No-op if stats is
 * null pointer.
 */
JXL_EXPORT void JxlEncoderStatsDestroy(JxlEncoderStats* stats);

/** Data type for querying @ref JxlEncoderStats object
 */
typedef enum {
  JXL_ENC_STAT_HEADER_BITS,
  JXL_ENC_STAT_TOC_BITS,
  JXL_ENC_STAT_DICTIONARY_BITS,
  JXL_ENC_STAT_SPLINES_BITS,
  JXL_ENC_STAT_NOISE_BITS,
  JXL_ENC_STAT_QUANT_BITS,
  JXL_ENC_STAT_MODULAR_TREE_BITS,
  JXL_ENC_STAT_MODULAR_GLOBAL_BITS,
  JXL_ENC_STAT_DC_BITS,
  JXL_ENC_STAT_MODULAR_DC_GROUP_BITS,
  JXL_ENC_STAT_CONTROL_FIELDS_BITS,
  JXL_ENC_STAT_COEF_ORDER_BITS,
  JXL_ENC_STAT_AC_HISTOGRAM_BITS,
  JXL_ENC_STAT_AC_BITS,
  JXL_ENC_STAT_MODULAR_AC_GROUP_BITS,
  JXL_ENC_STAT_NUM_SMALL_BLOCKS,
  JXL_ENC_STAT_NUM_DCT4X8_BLOCKS,
  JXL_ENC_STAT_NUM_AFV_BLOCKS,
  JXL_ENC_STAT_NUM_DCT8_BLOCKS,
  JXL_ENC_STAT_NUM_DCT8X32_BLOCKS,
  JXL_ENC_STAT_NUM_DCT16_BLOCKS,
  JXL_ENC_STAT_NUM_DCT16X32_BLOCKS,
  JXL_ENC_STAT_NUM_DCT32_BLOCKS,
  JXL_ENC_STAT_NUM_DCT32X64_BLOCKS,
  JXL_ENC_STAT_NUM_DCT64_BLOCKS,
  JXL_ENC_STAT_NUM_BUTTERAUGLI_ITERS,
  JXL_ENC_NUM_STATS,
} JxlEncoderStatsKey;

/** Returns the value of the statistics corresponding the given key.
 *
 * @param stats object that was passed to the encoder with a
 *   @ref JxlEncoderCollectStats function
 * @param key the particular statistics to query
 *
 * @return the value of the statistics
 */
JXL_EXPORT size_t JxlEncoderStatsGet(const JxlEncoderStats* stats,
                                     JxlEncoderStatsKey key);

/** Updates the values of the given stats object with that of an other.
 *
 * @param stats object whose values will be updated (usually added together)
 * @param other stats object whose values will be merged with stats
 */
JXL_EXPORT void JxlEncoderStatsMerge(JxlEncoderStats* stats,
                                     const JxlEncoderStats* other);

#ifdef __cplusplus
}
#endif

#endif /* JXL_STATS_H_ */

/** @}*/

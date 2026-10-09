/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_metadata
 * @{
 * @file gain_map.h
 * @brief Utility functions to manipulate jhgm (gain map) boxes.
 */

#ifndef JXL_GAIN_MAP_H_
#define JXL_GAIN_MAP_H_

#include <jxl/color_encoding.h>
#include <jxl/jxl_export.h>
#include <jxl/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Gain map bundle
 *
 * This structure is used to serialize gain map data to and from an input
 * buffer. It holds pointers to sections within the buffer, and different parts
 * of the gain map data such as metadata, ICC profile data, and the gain map
 * itself.
 *
 * The pointers in this structure do not take ownership of the memory they point
 * to. Instead, they reference specific locations within the provided buffer. It
 * is the caller's responsibility to ensure that the buffer remains valid and is
 * not deallocated as long as these pointers are in use. The structure should be
 * considered as providing a view into the buffer, not as an owner of the data.
 */
typedef struct {
  /** Version number of the gain map bundle. */
  uint8_t jhgm_version;
  /** Size of the gain map metadata in bytes. */
  uint16_t gain_map_metadata_size;
  /** Pointer to the gain map metadata, which is a binary
   * blob following ISO 21496-1. This pointer references data within the input
   * buffer. */
  const uint8_t* gain_map_metadata;
  /** Indicates whether a color encoding is present. */
  JXL_BOOL has_color_encoding;
  /** If has_color_encoding is true, this field contains the
   *        uncompressed color encoding data. */
  JxlColorEncoding color_encoding;
  /** Size of the alternative ICC profile in bytes (compressed
   * size). */
  uint32_t alt_icc_size;
  /** Pointer to the compressed ICC profile. This pointer references
   * data within the input buffer. */
  const uint8_t* alt_icc;
  /** Size of the gain map in bytes. */
  uint32_t gain_map_size;
  /** Pointer to the gain map data, which is a JPEG XL naked
   * codestream. This pointer references data within the input buffer.*/
  const uint8_t* gain_map;
} JxlGainMapBundle;

/**
 * Calculates the total size required to serialize the gain map bundle into a
 * binary buffer. This function accounts for all the necessary space to
 * serialize fields such as gain map metadata, color encoding, compressed ICC
 * profile data, and the gain map itself.
 *
 * @param[in] map_bundle Pointer to the JxlGainMapBundle containing all
 * necessary data to compute the size.
 * @param[out] bundle_size The size in bytes required to serialize the bundle.
 * @return Whether setting the size was successful.
 */
JXL_EXPORT JXL_BOOL JxlGainMapGetBundleSize(const JxlGainMapBundle* map_bundle,
                                            size_t* bundle_size);

/**
 * Serializes the gain map bundle into a preallocated buffer. The function
 * ensures that all parts of the bundle such as metadata, color encoding,
 * compressed ICC profile, and the gain map are correctly encoded into the
 * buffer. First call `JxlGainMapGetBundleSize` to get the size needed for
 * the buffer.
 *
 * @param[in] map_bundle Pointer to the `JxlGainMapBundle` to serialize.
 * @param[out] output_buffer Pointer to the buffer where the serialized data
 * will be written.
 * @param[in] output_buffer_size The size of the output buffer in bytes. Must be
 * large enough to hold the entire serialized data.
 * @param[out] bytes_written The number of bytes written to the output buffer.
 * @return Whether writing the bundle was successful.
 */
JXL_EXPORT JXL_BOOL JxlGainMapWriteBundle(const JxlGainMapBundle* map_bundle,
                                          uint8_t* output_buffer,
                                          size_t output_buffer_size,
                                          size_t* bytes_written);

/**
 * Deserializes a gain map bundle from a provided buffer and populates a
 * `JxlGainMapBundle` structure with the data extracted. This function assumes
 * the buffer contains a valid serialized gain map bundle. After successful
 * execution, the `JxlGainMapBundle` structure will reference three different
 * sections within the buffer:
 *  - gain_map_metadata
 *  - alt_icc
 *  - gain_map
 * These sections will be accompanied by their respective sizes. Users must
 * ensure that the buffer remains valid as long as these pointers are in use.
 * @param[in,out] map_bundle Pointer to a preallocated `JxlGainMapBundle` where
 * the deserialized data will be stored.
 * @param[in] input_buffer Pointer to the buffer containing the serialized gain
 * map bundle data.
 * @param[in] input_buffer_size The size of the input buffer in bytes.
 * @param[out] bytes_read The number of bytes read from the input buffer.
 * @return Whether reading the bundle was successful.
 */
JXL_EXPORT JXL_BOOL JxlGainMapReadBundle(JxlGainMapBundle* map_bundle,
                                         const uint8_t* input_buffer,
                                         size_t input_buffer_size,
                                         size_t* bytes_read);

#ifdef __cplusplus
}
#endif

#endif /* JXL_GAIN_MAP_H_ */

/** @} */

/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_color
 * @{
 * @file cms_interface.h
 * @brief Interface to allow the injection of different color management systems
 * (CMSes, also called color management modules, or CMMs) in JPEG XL.
 *
 * A CMS is needed by the JPEG XL encoder and decoder to perform colorspace
 * conversions. This defines an interface that can be implemented for different
 * CMSes and then passed to the library.
 */

#ifndef JXL_CMS_INTERFACE_H_
#define JXL_CMS_INTERFACE_H_

#include <jxl/color_encoding.h>
#include <jxl/types.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Parses an ICC profile and populates @p c and @p cmyk with the data.
 *
 * @param user_data @ref JxlCmsInterface::set_fields_data passed as-is.
 * @param icc_data the ICC data to parse.
 * @param icc_size how many bytes of icc_data are valid.
 * @param c a @ref JxlColorEncoding to populate if applicable.
 * @param cmyk a boolean to set to whether the colorspace is a CMYK colorspace.
 * @return Whether the relevant fields in @p c were successfully populated.
 */
typedef JXL_BOOL (*jpegxl_cms_set_fields_from_icc_func)(void* user_data,
                                                        const uint8_t* icc_data,
                                                        size_t icc_size,
                                                        JxlColorEncoding* c,
                                                        JXL_BOOL* cmyk);

/** Represents an input or output colorspace to a color transform, as a
 * serialized ICC profile. */
typedef struct {
  /** The serialized ICC profile. This is guaranteed to be present and valid. */
  struct {
    const uint8_t* data;
    size_t size;
  } icc;

  /** Structured representation of the colorspace, if applicable. If all fields
   * are different from their "unknown" value, then this is equivalent to the
   * ICC representation of the colorspace. If some are "unknown", those that are
   * not are still valid and can still be used on their own if they are useful.
   */
  JxlColorEncoding color_encoding;

  /** Number of components per pixel. This can be deduced from the other
   * representations of the colorspace but is provided for convenience and
   * validation. */
  size_t num_channels;
} JxlColorProfile;

/** Allocates and returns the data needed for @p num_threads parallel transforms
 * from the @p input colorspace to @p output, with up to @p pixels_per_thread
 * pixels to transform per call to @ref JxlCmsInterface::run. @p init_data comes
 * directly from the @ref JxlCmsInterface instance. Since @c run only receives
 * the data returned by @c init, a reference to @p init_data should be kept
 * there if access to it is desired in @c run. Likewise for @ref
 * JxlCmsInterface::destroy.
 *
 * The ICC data in @p input and @p output is guaranteed to outlive the @c init /
 * @c run / @c destroy cycle.
 *
 * @param init_data @ref JxlCmsInterface::init_data passed as-is.
 * @param num_threads the maximum number of threads from which
 *        @ref JxlCmsInterface::run will be called.
 * @param pixels_per_thread the maximum number of pixels that each call to
 *        @ref JxlCmsInterface::run will have to transform.
 * @param input_profile the input colorspace for the transform.
 * @param output_profile the colorspace to which @ref JxlCmsInterface::run
 * should convert the input data.
 * @param intensity_target for colorspaces where luminance is relative
 *        (essentially: not PQ), indicates the luminance at which (1, 1, 1) will
 *        be displayed. This is useful for conversions between PQ and a relative
 *        luminance colorspace, in either direction: @p intensity_target cd/m²
 *        in PQ should map to and from (1, 1, 1) in the relative one.\n
 *        It is also used for conversions to and from HLG, as it is
 *        scene-referred while other colorspaces are assumed to be
 *        display-referred. That is, conversions from HLG should apply the OOTF
 *        for a peak display luminance of @p intensity_target, and conversions
 *        to HLG should undo it. The OOTF is a gamma function applied to the
 *        luminance channel (https://www.itu.int/rec/R-REC-BT.2100-2-201807-I
 *        page 7), with the gamma value computed as
 *        <tt>1.2 * 1.111^log2(intensity_target / 1000)</tt> (footnote 2 page 8
 *        of the same document).
 * @return The data needed for the transform, or @c NULL in case of failure.
 *         This will be passed to the other functions as @c user_data.
 */
typedef void* (*jpegxl_cms_init_func)(void* init_data, size_t num_threads,
                                      size_t pixels_per_thread,
                                      const JxlColorProfile* input_profile,
                                      const JxlColorProfile* output_profile,
                                      float intensity_target);

/** Returns a buffer that can be used by callers of the interface to store the
 * input of the conversion or read its result, if they pass it as the input or
 * output of the @c run function.
 * @param user_data the data returned by @c init.
 * @param thread the index of the thread for which to return a buffer.
 * @return A buffer that can be used by the caller for passing to @c run.
 */
typedef float* (*jpegxl_cms_get_buffer_func)(void* user_data, size_t thread);

/** Executes one transform and returns true on success or false on error. It
 * must be possible to call this from different threads with different values
 * for @p thread, all between 0 (inclusive) and the value of @p num_threads
 * passed to @c init (exclusive). It is allowed to implement this by locking
 * such that the transforms are essentially performed sequentially, if such a
 * performance profile is acceptable. @p user_data is the data returned by
 * @c init.
 * The buffers each contain @p num_pixels × @c num_channels interleaved floating
 * point (0..1) samples where @c num_channels is the number of color channels of
 * their respective color profiles. It is guaranteed that the only case in which
 * they might overlap is if the output has fewer channels than the input, in
 * which case the pointers may be identical.
 * For CMYK data, 0 represents the maximum amount of ink while 1 represents no
 * ink.
 * @param user_data the data returned by @c init.
 * @param thread the index of the thread from which the function is being
 *        called.
 * @param input_buffer the buffer containing the pixel data to be transformed.
 * @param output_buffer the buffer receiving the transformed pixel data.
 * @param num_pixels the number of pixels to transform from @p input to
 * @p output.
 * @return ::JXL_TRUE on success, ::JXL_FALSE on failure.
 */
typedef JXL_BOOL (*jpegxl_cms_run_func)(void* user_data, size_t thread,
                                        const float* input_buffer,
                                        float* output_buffer,
                                        size_t num_pixels);

/** Performs the necessary clean-up and frees the memory allocated for user
 * data.
 */
typedef void (*jpegxl_cms_destroy_func)(void*);

/**
 * Interface for performing colorspace transforms. The @c init function can be
 * called several times to instantiate several transforms, including before
 * other transforms have been destroyed.
 *
 * The call sequence for a given colorspace transform could look like the
 * following:
 * @dot
 * digraph calls {
 *   newrank = true
 *   node [shape = box, fontname = monospace]
 *   init [label = "user_data <- init(\l\
 *     init_data = data,\l\
 *     num_threads = 3,\l\
 *     pixels_per_thread = 20,\l\
 *     input = (sRGB, 3 channels),\l\
 *     output = (Display-P3, 3 channels),\l\
 *     intensity_target = 255\l\
 *   )\l"]
 *   subgraph cluster_0 {
 *   color = lightgrey
 *   label = "thread 1"
 *   labeljust = "c"
 *   run_1_1 [label = "run(\l\
 *     user_data,\l\
 *     thread = 1,\l\
 *     input = in[0],\l\
 *     output = out[0],\l\
 *     num_pixels = 20\l\
 *   )\l"]
 *   run_1_2 [label = "run(\l\
 *     user_data,\l\
 *     thread = 1,\l\
 *     input = in[3],\l\
 *     output = out[3],\l\
 *     num_pixels = 20\l\
 *   )\l"]
 *   }
 *   subgraph cluster_1 {
 *   color = lightgrey
 *   label = "thread 2"
 *   labeljust = "l"
 *   run_2_1 [label = "run(\l\
 *     user_data,\l\
 *     thread = 2,\l\
 *     input = in[1],\l\
 *     output = out[1],\l\
 *     num_pixels = 20\l\
 *   )\l"]
 *   run_2_2 [label = "run(\l\
 *     user_data,\l\
 *     thread = 2,\l\
 *     input = in[4],\l\
 *     output = out[4],\l\
 *     num_pixels = 13\l\
 *   )\l"]
 *   }
 *   subgraph cluster_3 {
 *   color = lightgrey
 *   label = "thread 3"
 *   labeljust = "c"
 *   run_3_1 [label = "run(\l\
 *     user_data,\l\
 *     thread = 3,\l\
 *     input = in[2],\l\
 *     output = out[2],\l\
 *     num_pixels = 20\l\
 *   )\l"]
 *   }
 *   init -> {run_1_1; run_2_1; run_3_1; rank = same}
 *   run_1_1 -> run_1_2
 *   run_2_1 -> run_2_2
 *   {run_1_2; run_2_2, run_3_1} -> "destroy(user_data)"
 * }
 * @enddot
 */
typedef struct {
  /** CMS-specific data that will be passed to @ref set_fields_from_icc. */
  void* set_fields_data;
  /** Populates a @ref JxlColorEncoding from an ICC profile. */
  jpegxl_cms_set_fields_from_icc_func set_fields_from_icc;

  /** CMS-specific data that will be passed to @ref init. */
  void* init_data;
  /** Prepares a colorspace transform as described in the documentation of @ref
   * jpegxl_cms_init_func. */
  jpegxl_cms_init_func init;
  /** Returns a buffer that can be used as input to @c run. */
  jpegxl_cms_get_buffer_func get_src_buf;
  /** Returns a buffer that can be used as output from @c run. */
  jpegxl_cms_get_buffer_func get_dst_buf;
  /** Executes the transform on a batch of pixels, per @ref jpegxl_cms_run_func.
   */
  jpegxl_cms_run_func run;
  /** Cleans up the transform. */
  jpegxl_cms_destroy_func destroy;
} JxlCmsInterface;

#ifdef __cplusplus
}
#endif

#endif /* JXL_CMS_INTERFACE_H_ */

/** @} */

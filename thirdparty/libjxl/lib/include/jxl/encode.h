/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_encoder
 * @{
 * @file encode.h
 * @brief Encoding API for JPEG XL.
 */

#ifndef JXL_ENCODE_H_
#define JXL_ENCODE_H_

#include <jxl/cms_interface.h>
#include <jxl/codestream_header.h>
#include <jxl/color_encoding.h>
#include <jxl/jxl_export.h>
#include <jxl/memory_manager.h>
#include <jxl/parallel_runner.h>
#include <jxl/stats.h>
#include <jxl/types.h>
#include <jxl/version.h>  // TODO(eustas): remove before v1.0
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Encoder library version.
 *
 * @return the encoder library version as an integer:
 * MAJOR_VERSION * 1000000 + MINOR_VERSION * 1000 + PATCH_VERSION. For example,
 * version 1.2.3 would return 1002003.
 */
JXL_EXPORT uint32_t JxlEncoderVersion(void);

/**
 * Opaque structure that holds the JPEG XL encoder.
 *
 * Allocated and initialized with @ref JxlEncoderCreate().
 * Cleaned up and deallocated with @ref JxlEncoderDestroy().
 */
typedef struct JxlEncoderStruct JxlEncoder;

/**
 * Settings and metadata for a single image frame. This includes encoder options
 * for a frame such as compression quality and speed.
 *
 * Allocated and initialized with @ref JxlEncoderFrameSettingsCreate().
 * Cleaned up and deallocated when the encoder is destroyed with
 * @ref JxlEncoderDestroy().
 */
typedef struct JxlEncoderFrameSettingsStruct JxlEncoderFrameSettings;

/**
 * Return value for multiple encoder functions.
 */
typedef enum {
  /** Function call finished successfully, or encoding is finished and there is
   * nothing more to be done.
   */
  JXL_ENC_SUCCESS = 0,

  /** An error occurred, for example out of memory.
   */
  JXL_ENC_ERROR = 1,

  /** The encoder needs more output buffer to continue encoding.
   */
  JXL_ENC_NEED_MORE_OUTPUT = 2,

} JxlEncoderStatus;

/**
 * Error conditions:
 * API usage errors have the 0x80 bit set to 1
 * Other errors have the 0x80 bit set to 0
 */
typedef enum {
  /** No error
   */
  JXL_ENC_ERR_OK = 0,

  /** Generic encoder error due to unspecified cause
   */
  JXL_ENC_ERR_GENERIC = 1,

  /** Out of memory
   *  TODO(jon): actually catch this and return this error
   */
  JXL_ENC_ERR_OOM = 2,

  /** JPEG bitstream reconstruction data could not be
   *  represented (e.g. too much tail data)
   */
  JXL_ENC_ERR_JBRD = 3,

  /** Input is invalid (e.g. corrupt JPEG file or ICC profile)
   */
  JXL_ENC_ERR_BAD_INPUT = 4,

  /** The encoder doesn't (yet) support this. Either no version of libjxl
   * supports this, and the API is used incorrectly, or the libjxl version
   * should have been checked before trying to do this.
   */
  JXL_ENC_ERR_NOT_SUPPORTED = 0x80,

  /** The encoder API is used in an incorrect way.
   *  In this case, a debug build of libjxl should output a specific error
   * message. (if not, please open an issue about it)
   */
  JXL_ENC_ERR_API_USAGE = 0x81,

} JxlEncoderError;

/**
 * Id of encoder options for a frame. This includes options such as setting
 * encoding effort/speed or overriding the use of certain coding tools, for this
 * frame. This does not include non-frame related encoder options such as for
 * boxes.
 */
typedef enum {
  /** Sets encoder effort/speed level without affecting decoding speed. Valid
   * values are, from faster to slower speed: 1:lightning 2:thunder 3:falcon
   * 4:cheetah 5:hare 6:wombat 7:squirrel 8:kitten 9:tortoise 10:glacier.
   * Default: squirrel (7).
   */
  JXL_ENC_FRAME_SETTING_EFFORT = 0,

  /** Sets the decoding speed tier for the provided options. Minimum is 0
   * (slowest to decode, best quality/density), and maximum is 4 (fastest to
   * decode, at the cost of some quality/density). Default is 0.
   */
  JXL_ENC_FRAME_SETTING_DECODING_SPEED = 1,

  /** Sets resampling option. If enabled, the image is downsampled before
   * compression, and upsampled to original size in the decoder. Integer option,
   * use -1 for the default behavior (resampling only applied for low quality),
   * 1 for no downsampling (1x1), 2 for 2x2 downsampling, 4 for 4x4
   * downsampling, 8 for 8x8 downsampling.
   */
  JXL_ENC_FRAME_SETTING_RESAMPLING = 2,

  /** Similar to ::JXL_ENC_FRAME_SETTING_RESAMPLING, but for extra channels.
   * Integer option, use -1 for the default behavior (depends on encoder
   * implementation), 1 for no downsampling (1x1), 2 for 2x2 downsampling, 4 for
   * 4x4 downsampling, 8 for 8x8 downsampling.
   */
  JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING = 3,

  /** Indicates the frame added with @ref JxlEncoderAddImageFrame is already
   * downsampled by the downsampling factor set with @ref
   * JXL_ENC_FRAME_SETTING_RESAMPLING. The input frame must then be given in the
   * downsampled resolution, not the full image resolution. The downsampled
   * resolution is given by ceil(xsize / resampling), ceil(ysize / resampling)
   * with xsize and ysize the dimensions given in the basic info, and resampling
   * the factor set with ::JXL_ENC_FRAME_SETTING_RESAMPLING.
   * Use 0 to disable, 1 to enable. Default value is 0.
   */
  JXL_ENC_FRAME_SETTING_ALREADY_DOWNSAMPLED = 4,

  /** Adds noise to the image emulating photographic film noise, the higher the
   * given number, the grainier the image will be. As an example, a value of 100
   * gives low noise whereas a value of 3200 gives a lot of noise. The default
   * value is 0.
   */
  JXL_ENC_FRAME_SETTING_PHOTON_NOISE = 5,

  /** Enables adaptive noise generation. This setting is not recommended for
   * use, please use ::JXL_ENC_FRAME_SETTING_PHOTON_NOISE instead. Use -1 for
   * the default (encoder chooses), 0 to disable, 1 to enable.
   */
  JXL_ENC_FRAME_SETTING_NOISE = 6,

  /** Enables or disables dots generation. Use -1 for the default (encoder
   * chooses), 0 to disable, 1 to enable.
   */
  JXL_ENC_FRAME_SETTING_DOTS = 7,

  /** Enables or disables patches generation. Use -1 for the default (encoder
   * chooses), 0 to disable, 1 to enable.
   */
  JXL_ENC_FRAME_SETTING_PATCHES = 8,

  /** Edge preserving filter level, -1 to 3. Use -1 for the default (encoder
   * chooses), 0 to 3 to set a strength.
   */
  JXL_ENC_FRAME_SETTING_EPF = 9,

  /** Enables or disables the gaborish filter. Use -1 for the default (encoder
   * chooses), 0 to disable, 1 to enable.
   */
  JXL_ENC_FRAME_SETTING_GABORISH = 10,

  /** Enables modular encoding. Use -1 for default (encoder
   * chooses), 0 to enforce VarDCT mode (e.g. for photographic images), 1 to
   * enforce modular mode (e.g. for lossless images).
   */
  JXL_ENC_FRAME_SETTING_MODULAR = 11,

  /** Enables or disables preserving color of invisible pixels. Use -1 for the
   * default (1 if lossless, 0 if lossy), 0 to disable, 1 to enable.
   */
  JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE = 12,

  /** Determines the order in which 256x256 regions are stored in the codestream
   * for progressive rendering. Use -1 for the encoder
   * default, 0 for scanline order, 1 for center-first order.
   */
  JXL_ENC_FRAME_SETTING_GROUP_ORDER = 13,

  /** Determines the horizontal position of center for the center-first group
   * order. Use -1 to automatically use the middle of the image, 0..xsize to
   * specifically set it.
   */
  JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_X = 14,

  /** Determines the center for the center-first group order. Use -1 to
   * automatically use the middle of the image, 0..ysize to specifically set it.
   */
  JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_Y = 15,

  /** Enables or disables progressive encoding for modular mode. Use -1 for the
   * encoder default, 0 to disable, 1 to enable.
   */
  JXL_ENC_FRAME_SETTING_RESPONSIVE = 16,

  /** Set the progressive mode for the AC coefficients of VarDCT, using spectral
   * progression from the DCT coefficients. Use -1 for the encoder default, 0 to
   * disable, 1 to enable.
   */
  JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC = 17,

  /** Set the progressive mode for the AC coefficients of VarDCT, using
   * quantization of the least significant bits. Use -1 for the encoder default,
   * 0 to disable, 1 to enable.
   */
  JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC = 18,

  /** Set the progressive mode using lower-resolution DC images for VarDCT. Use
   * -1 for the encoder default, 0 to disable, 1 to have an extra 64x64 lower
   * resolution pass, 2 to have a 512x512 and 64x64 lower resolution pass.
   */
  JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC = 19,

  /** Use Global channel palette if the amount of colors is smaller than this
   * percentage of range. Use 0-100 to set an explicit percentage, -1 to use the
   * encoder default. Used for modular encoding.
   */
  JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GLOBAL_PERCENT = 20,

  /** Use Local (per-group) channel palette if the amount of colors is smaller
   * than this percentage of range. Use 0-100 to set an explicit percentage, -1
   * to use the encoder default. Used for modular encoding.
   */
  JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GROUP_PERCENT = 21,

  /** Use color palette if amount of colors is smaller than or equal to this
   * amount, or -1 to use the encoder default. Used for modular encoding.
   */
  JXL_ENC_FRAME_SETTING_PALETTE_COLORS = 22,

  /** Enables or disables delta palette. Use -1 for the default (encoder
   * chooses), 0 to disable, 1 to enable. Used in modular mode.
   */
  JXL_ENC_FRAME_SETTING_LOSSY_PALETTE = 23,

  /** Color transform for internal encoding: -1 = default, 0=XYB, 1=none (RGB),
   * 2=YCbCr. The XYB setting performs the forward XYB transform. None and
   * YCbCr both perform no transform, but YCbCr is used to indicate that the
   * encoded data losslessly represents YCbCr values.
   */
  JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM = 24,

  /** Reversible color transform for modular encoding: -1=default, 0-41=RCT
   * index, e.g. index 0 = none, index 6 = YCoCg.
   * If this option is set to a non-default value, the RCT will be globally
   * applied to the whole frame.
   * The default behavior is to try several RCTs locally per modular group,
   * depending on the speed and distance setting.
   */
  JXL_ENC_FRAME_SETTING_MODULAR_COLOR_SPACE = 25,

  /** Group size for modular encoding: -1=default, 0=128, 1=256, 2=512, 3=1024.
   */
  JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE = 26,

  /** Predictor for modular encoding. -1 = default, 0=zero, 1=left, 2=top,
   * 3=avg0, 4=select, 5=gradient, 6=weighted, 7=topright, 8=topleft,
   * 9=leftleft, 10=avg1, 11=avg2, 12=avg3, 13=toptop predictive average 14=mix
   * 5 and 6, 15=mix everything.
   */
  JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR = 27,

  /** Fraction of pixels used to learn MA trees as a percentage. -1 = default,
   * 0 = no MA and fast decode, 50 = default value, 100 = all, values above
   * 100 are also permitted. Higher values use more encoder memory.
   */
  JXL_ENC_FRAME_SETTING_MODULAR_MA_TREE_LEARNING_PERCENT = 28,

  /** Number of extra (previous-channel) MA tree properties to use. -1 =
   * default, 0-11 = valid values. Recommended values are in the range 0 to 3,
   * or 0 to amount of channels minus 1 (including all extra channels, and
   * excluding color channels when using VarDCT mode). Higher value gives slower
   * encoding and slower decoding.
   */
  JXL_ENC_FRAME_SETTING_MODULAR_NB_PREV_CHANNELS = 29,

  /** Enable or disable CFL (chroma-from-luma) for lossless JPEG recompression.
   * -1 = default, 0 = disable CFL, 1 = enable CFL.
   */
  JXL_ENC_FRAME_SETTING_JPEG_RECON_CFL = 30,

  /** Prepare the frame for indexing in the frame index box.
   * 0 = ignore this frame (same as not setting a value),
   * 1 = index this frame within the Frame Index Box.
   * If any frames are indexed, the first frame needs to
   * be indexed, too. If the first frame is not indexed, and
   * a later frame is attempted to be indexed, ::JXL_ENC_ERROR will occur.
   * If non-keyframes, i.e., frames with cropping, blending or patches are
   * attempted to be indexed, ::JXL_ENC_ERROR will occur.
   */
  JXL_ENC_FRAME_INDEX_BOX = 31,

  /** Sets brotli encode effort for use in JPEG recompression and
   * compressed metadata boxes (brob). Can be -1 (default) or 0 (fastest) to 11
   * (slowest). Default is based on the general encode effort in case of JPEG
   * recompression, and 4 for brob boxes.
   */
  JXL_ENC_FRAME_SETTING_BROTLI_EFFORT = 32,

  /** Enables or disables brotli compression of metadata boxes derived from
   * a JPEG frame when using @ref JxlEncoderAddJPEGFrame. This has no effect on
   * boxes added using @ref JxlEncoderAddBox. -1 = default, 0 = disable
   * compression, 1 = enable compression.
   */
  JXL_ENC_FRAME_SETTING_JPEG_COMPRESS_BOXES = 33,

  /** Control what kind of buffering is used, when using chunked image frames.
   * -1 = default (let the encoder decide)
   * 0 = buffers everything, basically the same as non-streamed code path
   (mainly for testing)
   * 1 = buffers everything for images that are smaller than 2048 x 2048, and
   *     uses streaming input and output for larger images
   * 2 = uses streaming input and output for all images that are larger than
   *     one group, i.e. 256 x 256 pixels by default
   * 3 = currently same as 2
   *
   * When using streaming input and output the encoder minimizes memory usage at
   * the cost of compression density. Also note that images produced with
   * streaming mode might not be progressively decodable.
   */
  JXL_ENC_FRAME_SETTING_BUFFERING = 34,

  /** Keep or discard Exif metadata boxes derived from a JPEG frame when using
   * @ref JxlEncoderAddJPEGFrame. This has no effect on boxes added using
   * @ref JxlEncoderAddBox. When @ref JxlEncoderStoreJPEGMetadata is set to 1,
   * this option cannot be set to 0. Even when Exif metadata is discarded, the
   * orientation will still be applied. 0 = discard Exif metadata, 1 = keep Exif
   * metadata (default).
   */
  JXL_ENC_FRAME_SETTING_JPEG_KEEP_EXIF = 35,

  /** Keep or discard XMP metadata boxes derived from a JPEG frame when using
   * @ref JxlEncoderAddJPEGFrame. This has no effect on boxes added using
   * @ref JxlEncoderAddBox. When @ref JxlEncoderStoreJPEGMetadata is set to 1,
   * this option cannot be set to 0. 0 = discard XMP metadata, 1 = keep XMP
   * metadata (default).
   */
  JXL_ENC_FRAME_SETTING_JPEG_KEEP_XMP = 36,

  /** Keep or discard JUMBF metadata boxes derived from a JPEG frame when using
   * @ref JxlEncoderAddJPEGFrame. This has no effect on boxes added using
   * @ref JxlEncoderAddBox. 0 = discard JUMBF metadata, 1 = keep JUMBF metadata
   * (default).
   */
  JXL_ENC_FRAME_SETTING_JPEG_KEEP_JUMBF = 37,

  /** If this mode is disabled, the encoder will not make any image quality
   * decisions that are computed based on the full image, but stored only once
   * (e.g. the X quant multiplier in the frame header). Used mainly for testing
   * equivalence of streaming and non-streaming code.
   * 0 = disabled, 1 = enabled (default)
   */
  JXL_ENC_FRAME_SETTING_USE_FULL_IMAGE_HEURISTICS = 38,

  /** Disable perceptual optimizations. 0 = optimizations enabled (default), 1 =
   * optimizations disabled.
   */
  JXL_ENC_FRAME_SETTING_DISABLE_PERCEPTUAL_HEURISTICS = 39,

  /** Enum value not to be used as an option. This value is added to force the
   * C compiler to have the enum to take a known size.
   */
  JXL_ENC_FRAME_SETTING_FILL_ENUM = 65535,

} JxlEncoderFrameSettingId;

/**
 * Creates an instance of @ref JxlEncoder and initializes it.
 *
 * @p memory_manager will be used for all the library dynamic allocations made
 * from this instance. The parameter may be NULL, in which case the default
 * allocator will be used. See jpegxl/memory_manager.h for details.
 *
 * @param memory_manager custom allocator function. It may be NULL. The memory
 *        manager will be copied internally.
 * @return @c NULL if the instance can not be allocated or initialized
 * @return pointer to initialized @ref JxlEncoder otherwise
 */
JXL_EXPORT JxlEncoder* JxlEncoderCreate(const JxlMemoryManager* memory_manager);

/**
 * Re-initializes a @ref JxlEncoder instance, so it can be re-used for encoding
 * another image. All state and settings are reset as if the object was
 * newly created with @ref JxlEncoderCreate, but the memory manager is kept.
 *
 * @param enc instance to be re-initialized.
 */
JXL_EXPORT void JxlEncoderReset(JxlEncoder* enc);

/**
 * Deinitializes and frees a @ref JxlEncoder instance.
 *
 * @param enc instance to be cleaned up and deallocated.
 */
JXL_EXPORT void JxlEncoderDestroy(JxlEncoder* enc);

/**
 * Sets the color management system (CMS) that will be used for color conversion
 * (if applicable) during encoding. May only be set before starting encoding. If
 * left unset, the default CMS implementation will be used.
 *
 * @param enc encoder object.
 * @param cms structure representing a CMS implementation. See @ref
 * JxlCmsInterface for more details.
 */
JXL_EXPORT void JxlEncoderSetCms(JxlEncoder* enc, JxlCmsInterface cms);

/**
 * Set the parallel runner for multithreading. May only be set before starting
 * encoding.
 *
 * @param enc encoder object.
 * @param parallel_runner function pointer to runner for multithreading. It may
 *        be NULL to use the default, single-threaded, runner. A multithreaded
 *        runner should be set to reach fast performance.
 * @param parallel_runner_opaque opaque pointer for parallel_runner.
 * @return ::JXL_ENC_SUCCESS if the runner was set, ::JXL_ENC_ERROR
 * otherwise (the previous runner remains set).
 */
JXL_EXPORT JxlEncoderStatus
JxlEncoderSetParallelRunner(JxlEncoder* enc, JxlParallelRunner parallel_runner,
                            void* parallel_runner_opaque);

/**
 * Get the (last) error code in case ::JXL_ENC_ERROR was returned.
 *
 * @param enc encoder object.
 * @return the @ref JxlEncoderError that caused the (last) ::JXL_ENC_ERROR to
 * be returned.
 */
JXL_EXPORT JxlEncoderError JxlEncoderGetError(JxlEncoder* enc);

/**
 * Encodes a JPEG XL file using the available bytes. @p *avail_out indicates how
 * many output bytes are available, and @p *next_out points to the input bytes.
 * *avail_out will be decremented by the amount of bytes that have been
 * processed by the encoder and *next_out will be incremented by the same
 * amount, so *next_out will now point at the amount of *avail_out unprocessed
 * bytes.
 *
 * The returned status indicates whether the encoder needs more output bytes.
 * When the return value is not ::JXL_ENC_ERROR or ::JXL_ENC_SUCCESS, the
 * encoding requires more @ref JxlEncoderProcessOutput calls to continue.
 *
 * The caller must guarantee that *avail_out >= 32 when calling
 * @ref JxlEncoderProcessOutput; otherwise, ::JXL_ENC_NEED_MORE_OUTPUT will
 * be returned. It is guaranteed that, if *avail_out >= 32, at least one byte of
 * output will be written.
 *
 * This encodes the frames and/or boxes added so far. If the last frame or last
 * box has been added, @ref JxlEncoderCloseInput, @ref JxlEncoderCloseFrames
 * and/or @ref JxlEncoderCloseBoxes must be called before the next
 * @ref JxlEncoderProcessOutput call, or the codestream won't be encoded
 * correctly.
 *
 * @param enc encoder object.
 * @param next_out pointer to next bytes to write to.
 * @param avail_out amount of bytes available starting from *next_out.
 * @return ::JXL_ENC_SUCCESS when encoding finished and all events handled.
 * @return ::JXL_ENC_ERROR when encoding failed, e.g. invalid input.
 * @return ::JXL_ENC_NEED_MORE_OUTPUT more output buffer is necessary.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderProcessOutput(JxlEncoder* enc,
                                                    uint8_t** next_out,
                                                    size_t* avail_out);

/**
 * Sets the frame information for this frame to the encoder. This includes
 * animation information such as frame duration to store in the frame header.
 * The frame header fields represent the frame as passed to the encoder, but not
 * necessarily the exact values as they will be encoded file format: the encoder
 * could change crop and blending options of a frame for more efficient encoding
 * or introduce additional internal frames. Animation duration and time code
 * information is not altered since those are immutable metadata of the frame.
 *
 * It is not required to use this function, however if have_animation is set
 * to true in the basic info, then this function should be used to set the
 * time duration of this individual frame. By default individual frames have a
 * time duration of 0, making them form a composite still. See @ref
 * JxlFrameHeader for more information.
 *
 * This information is stored in the @ref JxlEncoderFrameSettings and so is used
 * for any frame encoded with these @ref JxlEncoderFrameSettings. It is ok to
 * change between @ref JxlEncoderAddImageFrame calls, each added image frame
 * will have the frame header that was set in the options at the time of calling
 * @ref JxlEncoderAddImageFrame.
 *
 * The is_last and name_length fields of the @ref JxlFrameHeader are ignored,
 * use
 * @ref JxlEncoderCloseFrames to indicate last frame, and @ref
 * JxlEncoderSetFrameName to indicate the name and its length instead.
 * Calling this function will clear any name that was previously set with @ref
 * JxlEncoderSetFrameName.
 *
 * @param frame_settings set of options and metadata for this frame. Also
 * includes reference to the encoder object.
 * @param frame_header frame header data to set. Object owned by the caller and
 * does not need to be kept in memory, its information is copied internally.
 * @return ::JXL_ENC_SUCCESS on success, ::JXL_ENC_ERROR on error
 */
JXL_EXPORT JxlEncoderStatus
JxlEncoderSetFrameHeader(JxlEncoderFrameSettings* frame_settings,
                         const JxlFrameHeader* frame_header);

/**
 * Sets blend info of an extra channel. The blend info of extra channels is set
 * separately from that of the color channels, the color channels are set with
 * @ref JxlEncoderSetFrameHeader.
 *
 * @param frame_settings set of options and metadata for this frame. Also
 * includes reference to the encoder object.
 * @param index index of the extra channel to use.
 * @param blend_info blend info to set for the extra channel
 * @return ::JXL_ENC_SUCCESS on success, ::JXL_ENC_ERROR on error
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetExtraChannelBlendInfo(
    JxlEncoderFrameSettings* frame_settings, size_t index,
    const JxlBlendInfo* blend_info);

/**
 * Sets the name of the animation frame. This function is optional, frames are
 * not required to have a name. This setting is a part of the frame header, and
 * the same principles as for @ref JxlEncoderSetFrameHeader apply. The
 * name_length field of @ref JxlFrameHeader is ignored by the encoder, this
 * function determines the name length instead as the length in bytes of the C
 * string.
 *
 * The maximum possible name length is 1071 bytes (excluding terminating null
 * character).
 *
 * Calling @ref JxlEncoderSetFrameHeader clears any name that was
 * previously set.
 *
 * @param frame_settings set of options and metadata for this frame. Also
 * includes reference to the encoder object.
 * @param frame_name name of the next frame to be encoded, as a UTF-8 encoded C
 * string (zero terminated). Owned by the caller, and copied internally.
 * @return ::JXL_ENC_SUCCESS on success, ::JXL_ENC_ERROR on error
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetFrameName(
    JxlEncoderFrameSettings* frame_settings, const char* frame_name);

/**
 * Sets the bit depth of the input buffer.
 *
 * For float pixel formats, only the default @ref
 JXL_BIT_DEPTH_FROM_PIXEL_FORMAT
 * setting is allowed, while for unsigned pixel formats,
 * ::JXL_BIT_DEPTH_FROM_CODESTREAM setting is also allowed. See the comment
 on
 * @ref JxlEncoderAddImageFrame for the effects of the bit depth setting.

 * @param frame_settings set of options and metadata for this frame. Also
 * includes reference to the encoder object.
 * @param bit_depth the bit depth setting of the pixel input
 * @return ::JXL_ENC_SUCCESS on success, ::JXL_ENC_ERROR on error
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetFrameBitDepth(
    JxlEncoderFrameSettings* frame_settings, const JxlBitDepth* bit_depth);

/**
 * Sets the buffer to read JPEG encoded bytes from for the next frame to encode.
 *
 * If @ref JxlEncoderSetBasicInfo has not yet been called, calling
 * @ref JxlEncoderAddJPEGFrame will implicitly call it with the parameters of
 * the added JPEG frame.
 *
 * If @ref JxlEncoderSetColorEncoding or @ref JxlEncoderSetICCProfile has not
 * yet been called, calling @ref JxlEncoderAddJPEGFrame will implicitly call it
 * with the parameters of the added JPEG frame.
 *
 * If the encoder is set to store JPEG reconstruction metadata using @ref
 * JxlEncoderStoreJPEGMetadata and a single JPEG frame is added, it will be
 * possible to losslessly reconstruct the JPEG codestream.
 *
 * If this is the last frame, @ref JxlEncoderCloseInput or @ref
 * JxlEncoderCloseFrames must be called before the next
 * @ref JxlEncoderProcessOutput call.
 *
 * Note, this can only be used to add JPEG frames for lossless compression. To
 * encode with lossy compression, the JPEG must be decoded manually and a pixel
 * buffer added using JxlEncoderAddImageFrame.
 *
 * @param frame_settings set of options and metadata for this frame. Also
 * includes reference to the encoder object.
 * @param buffer bytes to read JPEG from. Owned by the caller and its contents
 * are copied internally.
 * @param size size of buffer in bytes.
 * @return ::JXL_ENC_SUCCESS on success, ::JXL_ENC_ERROR on error
 */
JXL_EXPORT JxlEncoderStatus
JxlEncoderAddJPEGFrame(const JxlEncoderFrameSettings* frame_settings,
                       const uint8_t* buffer, size_t size);

/**
 * Sets the buffer to read pixels from for the next image to encode. Must call
 * @ref JxlEncoderSetBasicInfo before @ref JxlEncoderAddImageFrame.
 *
 * Currently only some data types for pixel formats are supported:
 * - ::JXL_TYPE_UINT8, with range 0..255
 * - ::JXL_TYPE_UINT16, with range 0..65535
 * - ::JXL_TYPE_FLOAT16, with nominal range 0..1
 * - ::JXL_TYPE_FLOAT, with nominal range 0..1
 *
 * Note: the sample data type in pixel_format is allowed to be different from
 * what is described in the @ref JxlBasicInfo. The type in pixel_format,
 * together with an optional @ref JxlBitDepth parameter set by @ref
 * JxlEncoderSetFrameBitDepth describes the format of the uncompressed pixel
 * buffer. The bits_per_sample and exponent_bits_per_sample in the @ref
 * JxlBasicInfo describes what will actually be encoded in the JPEG XL
 * codestream. For example, to encode a 12-bit image, you would set
 * bits_per_sample to 12, while the input frame buffer can be in the following
 * formats:
 *  - if pixel format is in ::JXL_TYPE_UINT16 with default bit depth setting
 *    (i.e. ::JXL_BIT_DEPTH_FROM_PIXEL_FORMAT), input sample values are
 * rescaled to 16-bit, i.e. multiplied by 65535/4095;
 *  - if pixel format is in ::JXL_TYPE_UINT16 with @ref
 * JXL_BIT_DEPTH_FROM_CODESTREAM bit depth setting, input sample values are
 * provided unscaled;
 *  - if pixel format is in ::JXL_TYPE_FLOAT, input sample values are
 * rescaled to 0..1, i.e.  multiplied by 1.f/4095.f. While it is allowed, it is
 * obviously not recommended to use a pixel_format with lower precision than
 * what is specified in the @ref JxlBasicInfo.
 *
 * We support interleaved channels as described by the @ref JxlPixelFormat
 * "JxlPixelFormat":
 * - single-channel data, e.g. grayscale
 * - single-channel + alpha
 * - trichromatic, e.g. RGB
 * - trichromatic + alpha
 *
 * Extra channels not handled here need to be set by @ref
 * JxlEncoderSetExtraChannelBuffer.
 * If the image has alpha, and alpha is not passed here, it will implicitly be
 * set to all-opaque (an alpha value of 1.0 everywhere).
 *
 * The pixels are assumed to be encoded in the original profile that is set with
 * @ref JxlEncoderSetColorEncoding or @ref JxlEncoderSetICCProfile. If none of
 * these functions were used, the pixels are assumed to be nonlinear sRGB for
 * integer data types (::JXL_TYPE_UINT8, ::JXL_TYPE_UINT16), and linear
 * sRGB for floating point data types (::JXL_TYPE_FLOAT16, @ref
 * JXL_TYPE_FLOAT).
 *
 * Sample values in floating-point pixel formats are allowed to be outside the
 * nominal range, e.g. to represent out-of-sRGB-gamut colors in the
 * uses_original_profile=false case. They are however not allowed to be NaN or
 * +-infinity.
 *
 * If this is the last frame, @ref JxlEncoderCloseInput or @ref
 * JxlEncoderCloseFrames must be called before the next
 * @ref JxlEncoderProcessOutput call.
 *
 * @param frame_settings set of options and metadata for this frame. Also
 * includes reference to the encoder object.
 * @param pixel_format format for pixels. Object owned by the caller and its
 * contents are copied internally.
 * @param buffer buffer type to input the pixel data from. Owned by the caller
 * and its contents are copied internally.
 * @param size size of buffer in bytes. This size should match what is implied
 * by the frame dimensions and the pixel format.
 * @return ::JXL_ENC_SUCCESS on success, ::JXL_ENC_ERROR on error
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderAddImageFrame(
    const JxlEncoderFrameSettings* frame_settings,
    const JxlPixelFormat* pixel_format, const void* buffer, size_t size);

/**
 * The @ref JxlEncoderOutputProcessor structure provides an interface for the
 * encoder's output processing. Users of the library, who want to do streaming
 * encoding, should implement the required callbacks for buffering, writing,
 * seeking (if supported), and setting a finalized position during the encoding
 * process.
 *
 * At a high level, the processor can be in one of two states:
 * - With an active buffer: This indicates that a buffer has been acquired using
 *   `get_buffer` and encoded data can be written to it.
 * - Without an active buffer: In this state, no data can be written. A new
 * buffer must be acquired after releasing any previously active buffer.
 *
 * The library will not acquire more than one buffer at a given time.
 *
 * The state of the processor includes `position` and `finalized position`,
 * which have the following meaning.
 *
 * - position: Represents the current position, in bytes, within the output
 * stream where the encoded data will be written next. This position moves
 * forward with each `release_buffer` call as data is written, and can also be
 * adjusted through the optional seek callback, if provided. At this position
 * the next write will occur.
 *
 * - finalized position:  A position in the output stream that ensures all bytes
 * before this point are finalized and won't be changed by later writes.
 *
 * All fields but `seek` are required, `seek` is optional and can be NULL.
 */
struct JxlEncoderOutputProcessor {
  /**
   * Required.
   * An opaque pointer that the client can use to store custom data.
   * This data will be passed to the associated callback functions.
   */
  void* opaque;

  /**
   * Required.
   * Acquires a buffer at the current position into which the library will write
   * the output data.
   *
   * If the `size` argument points to 0 and the returned value is NULL, this
   * will be interpreted as asking the output writing to stop. In such a case,
   * the library will return an error. The client is expected to set the size of
   * the returned buffer based on the suggested `size` when this function is
   * called.
   *
   * @param opaque user supplied parameters to the callback
   * @param size points to a suggested buffer size when called; must be set to
   * the size of the returned buffer once the function returns.
   * @return a pointer to the acquired buffer or NULL to indicate a stop
   * condition.
   */
  void* (*get_buffer)(void* opaque, size_t* size);

  /**
   * Required.
   * Notifies the user of library that the current buffer's data has been
   * written and can be released. This function should advance the current
   * position of the buffer by `written_bytes` number of bytes.
   *
   * @param opaque user supplied parameters to the callback
   * @param written_bytes the number of bytes written to the buffer.
   */
  void (*release_buffer)(void* opaque, size_t written_bytes);

  /**
   * Optional, can be NULL
   * Seeks to a specific position in the output. This function is optional and
   * can be set to NULL if the output doesn't support seeking. Can only be done
   * when there is no buffer. Cannot be used to seek before the finalized
   * position.
   *
   * @param opaque user supplied parameters to the callback
   * @param position the position to seek to, in bytes.
   */
  void (*seek)(void* opaque, uint64_t position);

  /**
   * Required.
   * Sets a finalized position on the output data, at a specific position.
   * Seeking will never request a position before the finalized position.
   *
   * Will only be called if there is no active buffer.
   *
   * @param opaque user supplied parameters to the callback
   * @param finalized_position the position, in bytes, where the finalized
   * position should be set.
   */
  void (*set_finalized_position)(void* opaque, uint64_t finalized_position);
};

/**
 * Sets the output processor for the encoder. This processor determines how the
 * encoder will handle buffering, writing, seeking (if supported), and
 * setting a finalized position during the encoding process.
 *
 * This should not be used when using @ref JxlEncoderProcessOutput.
 *
 * @param enc encoder object.
 * @param output_processor the struct containing the callbacks for managing
 * output.
 * @return ::JXL_ENC_SUCCESS on success, ::JXL_ENC_ERROR on error.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetOutputProcessor(
    JxlEncoder* enc, struct JxlEncoderOutputProcessor output_processor);

/**
 * Flushes any buffered input in the encoder, ensuring that all available input
 * data has been processed and written to the output.
 *
 * This function can only be used after @ref JxlEncoderSetOutputProcessor.
 * Before making the last call to @ref JxlEncoderFlushInput, users should call
 * @ref JxlEncoderCloseInput to signal the end of input data.
 *
 * This should not be used when using @ref JxlEncoderProcessOutput.
 *
 * @param enc encoder object.
 * @return ::JXL_ENC_SUCCESS on success, ::JXL_ENC_ERROR on error.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderFlushInput(JxlEncoder* enc);

/**
 * This struct provides callback functions to pass pixel data in a streaming
 * manner instead of requiring the entire frame data in memory at once.
 */
struct JxlChunkedFrameInputSource {
  /**
   * A pointer to any user-defined data or state. This can be used to pass
   * information to the callback functions.
   */
  void* opaque;

  /**
   * Get the pixel format that color channel data will be provided in.
   * When called, `pixel_format` points to a suggested pixel format; if
   * color channel data can be given in this pixel format, processing might
   * be more efficient.
   *
   * This function will be called exactly once, before any call to
   * get_color_channel_at.
   *
   * @param opaque user supplied parameters to the callback
   * @param pixel_format format for pixels
   */
  void (*get_color_channels_pixel_format)(void* opaque,
                                          JxlPixelFormat* pixel_format);

  /**
   * Callback to retrieve a rectangle of color channel data at a specific
   * location. It is guaranteed that xpos and ypos are multiples of 8. xsize,
   * ysize will be multiples of 8, unless the resulting rectangle would be out
   * of image bounds. Moreover, xsize and ysize will be at most 2048. The
   * returned data will be assumed to be in the format returned by the
   * (preceding) call to get_color_channels_pixel_format, except the `align`
   * parameter of the pixel format will be ignored. Instead, the `i`-th row will
   * be assumed to start at position `return_value + i * *row_offset`, with the
   * value of `*row_offset` decided by the callee.
   *
   * Note that multiple calls to `get_color_channel_data_at` may happen before a
   * call to `release_buffer`.
   *
   * @param opaque user supplied parameters to the callback
   * @param xpos horizontal position for the data.
   * @param ypos vertical position for the data.
   * @param xsize horizontal size of the requested rectangle of data.
   * @param ysize vertical size of the requested rectangle of data.
   * @param row_offset pointer to a the byte offset between consecutive rows of
   * the retrieved pixel data.
   * @return pointer to the retrieved pixel data.
   */
  const void* (*get_color_channel_data_at)(void* opaque, size_t xpos,
                                           size_t ypos, size_t xsize,
                                           size_t ysize, size_t* row_offset);

  /**
   * Get the pixel format that extra channel data will be provided in.
   * When called, `pixel_format` points to a suggested pixel format; if
   * extra channel data can be given in this pixel format, processing might
   * be more efficient.
   *
   * This function will be called exactly once per index, before any call to
   * get_extra_channel_data_at with that given index.
   *
   * @param opaque user supplied parameters to the callback
   * @param ec_index zero-indexed index of the extra channel
   * @param pixel_format format for extra channel data
   */
  void (*get_extra_channel_pixel_format)(void* opaque, size_t ec_index,
                                         JxlPixelFormat* pixel_format);

  /**
   * Callback to retrieve a rectangle of extra channel `ec_index` data at a
   * specific location. It is guaranteed that xpos and ypos are multiples of
   * 8. xsize, ysize will be multiples of 8, unless the resulting rectangle
   * would be out of image bounds. Moreover, xsize and ysize will be at most
   * 2048. The returned data will be assumed to be in the format returned by the
   * (preceding) call to get_extra_channels_pixel_format_at with the
   * corresponding extra channel index `ec_index`, except the `align` parameter
   * of the pixel format will be ignored. Instead, the `i`-th row will be
   * assumed to start at position `return_value + i * *row_offset`, with the
   * value of `*row_offset` decided by the callee.
   *
   * Note that multiple calls to `get_extra_channel_data_at` may happen before a
   * call to `release_buffer`.
   *
   * @param opaque user supplied parameters to the callback
   * @param xpos horizontal position for the data.
   * @param ypos vertical position for the data.
   * @param xsize horizontal size of the requested rectangle of data.
   * @param ysize vertical size of the requested rectangle of data.
   * @param row_offset pointer to a the byte offset between consecutive rows of
   * the retrieved pixel data.
   * @return pointer to the retrieved pixel data.
   */
  const void* (*get_extra_channel_data_at)(void* opaque, size_t ec_index,
                                           size_t xpos, size_t ypos,
                                           size_t xsize, size_t ysize,
                                           size_t* row_offset);

  /**
   * Releases the buffer `buf` (obtained through a call to
   * `get_color_channel_data_at` or `get_extra_channel_data_at`). This function
   * will be called exactly once per call to `get_color_channel_data_at` or
   * `get_extra_channel_data_at`.
   *
   * @param opaque user supplied parameters to the callback
   * @param buf pointer returned by `get_color_channel_data_at` or
   * `get_extra_channel_data_at`
   */
  void (*release_buffer)(void* opaque, const void* buf);
};

/**
 * @brief Adds a frame to the encoder using a chunked input source.
 *
 * This function gives a way to encode a frame by providing pixel data in a
 * chunked or streaming manner, which can be especially useful when dealing with
 * large images that may not fit entirely in memory or when trying to optimize
 * memory usage. The input data is provided through callbacks defined in the
 * @ref JxlChunkedFrameInputSource struct. Once the frame data has been
 * completely retrieved, this function will flush the input and close it if it
 * is the last frame.
 *
 * @param frame_settings set of options and metadata for this frame. Also
 * includes reference to the encoder object.
 * @param is_last_frame indicates if this is the last frame.
 * @param chunked_frame_input struct providing callback methods for retrieving
 * pixel data in chunks.
 *
 * @return Returns a status indicating the success or failure of adding the
 * frame.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderAddChunkedFrame(
    const JxlEncoderFrameSettings* frame_settings, JXL_BOOL is_last_frame,
    struct JxlChunkedFrameInputSource chunked_frame_input);

/**
 * Sets the buffer to read pixels from for an extra channel at a given index.
 * The index must be smaller than the num_extra_channels in the associated
 * @ref JxlBasicInfo. Must call @ref JxlEncoderSetExtraChannelInfo before @ref
 * JxlEncoderSetExtraChannelBuffer.
 *
 * TODO(firsching): mention what data types in pixel formats are supported.
 *
 * It is required to call this function for every extra channel, except for the
 * alpha channel if that was already set through @ref JxlEncoderAddImageFrame.
 *
 * @param frame_settings set of options and metadata for this frame. Also
 * includes reference to the encoder object.
 * @param pixel_format format for pixels. Object owned by the caller and its
 * contents are copied internally. The num_channels value is ignored, since the
 * number of channels for an extra channel is always assumed to be one.
 * @param buffer buffer type to input the pixel data from. Owned by the caller
 * and its contents are copied internally.
 * @param size size of buffer in bytes. This size should match what is implied
 * by the frame dimensions and the pixel format.
 * @param index index of the extra channel to use.
 * @return ::JXL_ENC_SUCCESS on success, ::JXL_ENC_ERROR on error
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetExtraChannelBuffer(
    const JxlEncoderFrameSettings* frame_settings,
    const JxlPixelFormat* pixel_format, const void* buffer, size_t size,
    uint32_t index);

/** Adds a metadata box to the file format. @ref JxlEncoderProcessOutput must be
 * used to effectively write the box to the output. @ref JxlEncoderUseBoxes must
 * be enabled before using this function.
 *
 * Boxes allow inserting application-specific data and metadata (Exif, XML/XMP,
 * JUMBF and user defined boxes).
 *
 * The box format follows ISO BMFF and shares features and box types with other
 * image and video formats, including the Exif, XML and JUMBF boxes. The box
 * format for JPEG XL is specified in ISO/IEC 18181-2.
 *
 * Boxes in general don't contain other boxes inside, except a JUMBF superbox.
 * Boxes follow each other sequentially and are byte-aligned. If the container
 * format is used, the JXL stream consists of concatenated boxes.
 * It is also possible to use a direct codestream without boxes, but in that
 * case metadata cannot be added.
 *
 * Each box generally has the following byte structure in the file:
 * - 4 bytes: box size including box header (Big endian. If set to 0, an
 *   8-byte 64-bit size follows instead).
 * - 4 bytes: type, e.g. "JXL " for the signature box, "jxlc" for a codestream
 *   box.
 * - N bytes: box contents.
 *
 * Only the box contents are provided to the contents argument of this function,
 * the encoder encodes the size header itself. Most boxes are written
 * automatically by the encoder as needed ("JXL ", "ftyp", "jxll", "jxlc",
 * "jxlp", "jxli", "jbrd"), and this function only needs to be called to add
 * optional metadata when encoding from pixels (using @ref
 * JxlEncoderAddImageFrame). When recompressing JPEG files (using @ref
 * JxlEncoderAddJPEGFrame), if the input JPEG contains EXIF, XMP or JUMBF
 * metadata, the corresponding boxes are already added automatically.
 *
 * Box types are given by 4 characters. The following boxes can be added with
 * this function:
 * - "Exif": a box with EXIF metadata, can be added by libjxl users, or is
 *   automatically added when needed for JPEG reconstruction. The contents of
 *   this box must be prepended by a 4-byte tiff header offset, which may
 *   be 4 zero bytes in case the tiff header follows immediately.
 *   The EXIF metadata must be in sync with what is encoded in the JPEG XL
 *   codestream, specifically the image orientation. While this is not
 *   recommended in practice, in case of conflicting metadata, the JPEG XL
 *   codestream takes precedence.
 * - "xml ": a box with XML data, in particular XMP metadata, can be added by
 *   libjxl users, or is automatically added when needed for JPEG reconstruction
 * - "jumb": a JUMBF superbox, which can contain boxes with different types of
 *   metadata inside. This box type can be added by the encoder transparently,
 *   and other libraries to create and handle JUMBF content exist.
 * - Application-specific boxes. Their typename should not begin with "jxl" or
 *   "JXL" or conflict with other existing typenames, and they should be
 *   registered with MP4RA (mp4ra.org).
 *
 * These boxes can be stored uncompressed or Brotli-compressed (using a "brob"
 * box), depending on the compress_box parameter.
 *
 * @param enc encoder object.
 * @param type the box type, e.g. "Exif" for EXIF metadata, "xml " for XMP or
 * IPTC metadata, "jumb" for JUMBF metadata.
 * @param contents the full contents of the box, for example EXIF
 * data. ISO BMFF box header must not be included, only the contents. Owned by
 * the caller and its contents are copied internally.
 * @param size size of the box contents.
 * @param compress_box Whether to compress this box as a "brob" box. Requires
 * Brotli support.
 * @return ::JXL_ENC_SUCCESS on success, ::JXL_ENC_ERROR on error, such as
 * when using this function without @ref JxlEncoderUseContainer, or adding a box
 * type that would result in an invalid file format.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderAddBox(JxlEncoder* enc,
                                             const JxlBoxType type,
                                             const uint8_t* contents,
                                             size_t size,
                                             JXL_BOOL compress_box);

/**
 * Indicates the intention to add metadata boxes. This allows @ref
 * JxlEncoderAddBox to be used. When using this function, then it is required
 * to use @ref JxlEncoderCloseBoxes at the end.
 *
 * By default the encoder assumes no metadata boxes will be added.
 *
 * This setting can only be set at the beginning, before encoding starts.
 *
 * @param enc encoder object.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderUseBoxes(JxlEncoder* enc);

/**
 * Declares that no further boxes will be added with @ref JxlEncoderAddBox.
 * This function must be called after the last box is added so the encoder knows
 * the stream will be finished. It is not necessary to use this function if
 * @ref JxlEncoderUseBoxes is not used. Further frames may still be added.
 *
 * Must be called between @ref JxlEncoderAddBox of the last box
 * and the next call to @ref JxlEncoderProcessOutput, or @ref
 * JxlEncoderProcessOutput won't output the last box correctly.
 *
 * NOTE: if you don't need to close frames and boxes at separate times, you can
 * use @ref JxlEncoderCloseInput instead to close both at once.
 *
 * @param enc encoder object.
 */
JXL_EXPORT void JxlEncoderCloseBoxes(JxlEncoder* enc);

/**
 * Declares that no frames will be added and @ref JxlEncoderAddImageFrame and
 * @ref JxlEncoderAddJPEGFrame won't be called anymore. Further metadata boxes
 * may still be added. This function or @ref JxlEncoderCloseInput must be called
 * after adding the last frame and the next call to
 * @ref JxlEncoderProcessOutput, or the frame won't be properly marked as last.
 *
 * NOTE: if you don't need to close frames and boxes at separate times, you can
 * use @ref JxlEncoderCloseInput instead to close both at once.
 *
 * @param enc encoder object.
 */
JXL_EXPORT void JxlEncoderCloseFrames(JxlEncoder* enc);

/**
 * Closes any input to the encoder, equivalent to calling @ref
 * JxlEncoderCloseFrames as well as calling @ref JxlEncoderCloseBoxes if needed.
 * No further input of any kind may be given to the encoder, but further @ref
 * JxlEncoderProcessOutput calls should be done to create the final output.
 *
 * The requirements of both @ref JxlEncoderCloseFrames and @ref
 * JxlEncoderCloseBoxes apply to this function. Either this function or the
 * other two must be called after the final frame and/or box, and the next
 * @ref JxlEncoderProcessOutput call, or the codestream won't be encoded
 * correctly.
 *
 * @param enc encoder object.
 */
JXL_EXPORT void JxlEncoderCloseInput(JxlEncoder* enc);

/**
 * Sets the original color encoding of the image encoded by this encoder. This
 * is an alternative to @ref JxlEncoderSetICCProfile and only one of these two
 * must be used. This one sets the color encoding as a @ref JxlColorEncoding,
 * while the other sets it as ICC binary data. Must be called after @ref
 * JxlEncoderSetBasicInfo.
 *
 * @param enc encoder object.
 * @param color color encoding. Object owned by the caller and its contents are
 * copied internally.
 * @return ::JXL_ENC_SUCCESS if the operation was successful, @ref
 * JXL_ENC_ERROR otherwise
 */
JXL_EXPORT JxlEncoderStatus
JxlEncoderSetColorEncoding(JxlEncoder* enc, const JxlColorEncoding* color);

/**
 * Sets the original color encoding of the image encoded by this encoder as an
 * ICC color profile. This is an alternative to @ref JxlEncoderSetColorEncoding
 * and only one of these two must be used. This one sets the color encoding as
 * ICC binary data, while the other defines it as a @ref JxlColorEncoding. Must
 * be called after @ref JxlEncoderSetBasicInfo.
 *
 * @param enc encoder object.
 * @param icc_profile bytes of the original ICC profile
 * @param size size of the icc_profile buffer in bytes
 * @return ::JXL_ENC_SUCCESS if the operation was successful, @ref
 * JXL_ENC_ERROR otherwise
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetICCProfile(JxlEncoder* enc,
                                                    const uint8_t* icc_profile,
                                                    size_t size);

/**
 * Initializes a @ref JxlBasicInfo struct to default values.
 * For forwards-compatibility, this function has to be called before values
 * are assigned to the struct fields.
 * The default values correspond to an 8-bit RGB image, no alpha or any
 * other extra channels.
 *
 * @param info global image metadata. Object owned by the caller.
 */
JXL_EXPORT void JxlEncoderInitBasicInfo(JxlBasicInfo* info);

/**
 * Initializes a @ref JxlFrameHeader struct to default values.
 * For forwards-compatibility, this function has to be called before values
 * are assigned to the struct fields.
 * The default values correspond to a frame with no animation duration and the
 * 'replace' blend mode. After using this function, For animation duration must
 * be set, for composite still blend settings must be set.
 *
 * @param frame_header frame metadata. Object owned by the caller.
 */
JXL_EXPORT void JxlEncoderInitFrameHeader(JxlFrameHeader* frame_header);

/**
 * Initializes a @ref JxlBlendInfo struct to default values.
 * For forwards-compatibility, this function has to be called before values
 * are assigned to the struct fields.
 *
 * @param blend_info blending info. Object owned by the caller.
 */
JXL_EXPORT void JxlEncoderInitBlendInfo(JxlBlendInfo* blend_info);

/**
 * Sets the global metadata of the image encoded by this encoder.
 *
 * If the @ref JxlBasicInfo contains information of extra channels beyond an
 * alpha channel, then @ref JxlEncoderSetExtraChannelInfo must be called between
 * @ref JxlEncoderSetBasicInfo and @ref JxlEncoderAddImageFrame. In order to
 * indicate extra channels, the value of `info.num_extra_channels` should be set
 * to the number of extra channels, also counting the alpha channel if present.
 *
 * @param enc encoder object.
 * @param info global image metadata. Object owned by the caller and its
 * contents are copied internally.
 * @return ::JXL_ENC_SUCCESS if the operation was successful,
 * ::JXL_ENC_ERROR otherwise
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetBasicInfo(JxlEncoder* enc,
                                                   const JxlBasicInfo* info);

/**
 * Sets the upsampling method the decoder will use in case there are frames
 * with ::JXL_ENC_FRAME_SETTING_RESAMPLING set. This is useful in combination
 * with the ::JXL_ENC_FRAME_SETTING_ALREADY_DOWNSAMPLED option, to control
 * the type of upsampling that will be used.
 *
 * @param enc encoder object.
 * @param factor upsampling factor to configure (1, 2, 4 or 8; for 1 this
 * function has no effect at all)
 * @param mode upsampling mode to use for this upsampling:
 * -1: default (good for photographic images, no signaling overhead)
 * 0: nearest neighbor (good for pixel art)
 * 1: 'pixel dots' (same as NN for 2x, diamond-shaped 'pixel dots' for 4x/8x)
 * @return ::JXL_ENC_SUCCESS if the operation was successful,
 * ::JXL_ENC_ERROR otherwise
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetUpsamplingMode(JxlEncoder* enc,
                                                        int64_t factor,
                                                        int64_t mode);

/**
 * Initializes a @ref JxlExtraChannelInfo struct to default values.
 * For forwards-compatibility, this function has to be called before values
 * are assigned to the struct fields.
 * The default values correspond to an 8-bit channel of the provided type.
 *
 * @param type type of the extra channel.
 * @param info global extra channel metadata. Object owned by the caller and its
 * contents are copied internally.
 */
JXL_EXPORT void JxlEncoderInitExtraChannelInfo(JxlExtraChannelType type,
                                               JxlExtraChannelInfo* info);

/**
 * Sets information for the extra channel at the given index. The index
 * must be smaller than num_extra_channels in the associated @ref JxlBasicInfo.
 *
 * @param enc encoder object
 * @param index index of the extra channel to set.
 * @param info global extra channel metadata. Object owned by the caller and its
 * contents are copied internally.
 * @return ::JXL_ENC_SUCCESS on success, ::JXL_ENC_ERROR on error
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetExtraChannelInfo(
    JxlEncoder* enc, size_t index, const JxlExtraChannelInfo* info);

/**
 * Sets the name for the extra channel at the given index in UTF-8. The index
 * must be smaller than the num_extra_channels in the associated @ref
 * JxlBasicInfo.
 *
 * TODO(lode): remove size parameter for consistency with
 * @ref JxlEncoderSetFrameName
 *
 * @param enc encoder object
 * @param index index of the extra channel to set.
 * @param name buffer with the name of the extra channel.
 * @param size size of the name buffer in bytes, not counting the terminating
 * character.
 * @return JXL_ENC_SUCCESS on success, JXL_ENC_ERROR on error
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetExtraChannelName(JxlEncoder* enc,
                                                          size_t index,
                                                          const char* name,
                                                          size_t size);

/**
 * Sets a frame-specific option of integer type to the encoder options.
 * The @ref JxlEncoderFrameSettingId argument determines which option is set.
 *
 * @param frame_settings set of options and metadata for this frame. Also
 * includes reference to the encoder object.
 * @param option ID of the option to set.
 * @param value Integer value to set for this option.
 * @return ::JXL_ENC_SUCCESS if the operation was successful, @ref
 * JXL_ENC_ERROR in case of an error, such as invalid or unknown option id, or
 * invalid integer value for the given option. If an error is returned, the
 * state of the
 * @ref JxlEncoderFrameSettings object is still valid and is the same as before
 * this function was called.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderFrameSettingsSetOption(
    JxlEncoderFrameSettings* frame_settings, JxlEncoderFrameSettingId option,
    int64_t value);

/**
 * Sets a frame-specific option of float type to the encoder options.
 * The @ref JxlEncoderFrameSettingId argument determines which option is set.
 *
 * @param frame_settings set of options and metadata for this frame. Also
 * includes reference to the encoder object.
 * @param option ID of the option to set.
 * @param value Float value to set for this option.
 * @return ::JXL_ENC_SUCCESS if the operation was successful, @ref
 * JXL_ENC_ERROR in case of an error, such as invalid or unknown option id, or
 * invalid integer value for the given option. If an error is returned, the
 * state of the
 * @ref JxlEncoderFrameSettings object is still valid and is the same as before
 * this function was called.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderFrameSettingsSetFloatOption(
    JxlEncoderFrameSettings* frame_settings, JxlEncoderFrameSettingId option,
    float value);

/** Forces the encoder to use the box-based container format (BMFF) even
 * when not necessary.
 *
 * When using @ref JxlEncoderUseBoxes, @ref JxlEncoderStoreJPEGMetadata or @ref
 * JxlEncoderSetCodestreamLevel with level 10, the encoder will automatically
 * also use the container format, it is not necessary to use
 * @ref JxlEncoderUseContainer for those use cases.
 *
 * By default this setting is disabled.
 *
 * This setting can only be set at the beginning, before encoding starts.
 *
 * @param enc encoder object.
 * @param use_container true if the encoder should always output the JPEG XL
 * container format, false to only output it when necessary.
 * @return JXL_ENC_SUCCESS if the operation was successful, JXL_ENC_ERROR
 * otherwise.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderUseContainer(JxlEncoder* enc,
                                                   JXL_BOOL use_container);

/**
 * Configure the encoder to store JPEG reconstruction metadata in the JPEG XL
 * container.
 *
 * If this is set to true and a single JPEG frame is added, it will be
 * possible to losslessly reconstruct the JPEG codestream.
 *
 * This setting can only be set at the beginning, before encoding starts.
 *
 * @param enc encoder object.
 * @param store_jpeg_metadata true if the encoder should store JPEG metadata.
 * @return ::JXL_ENC_SUCCESS if the operation was successful, @ref
 * JXL_ENC_ERROR otherwise.
 */
JXL_EXPORT JxlEncoderStatus
JxlEncoderStoreJPEGMetadata(JxlEncoder* enc, JXL_BOOL store_jpeg_metadata);

/** Sets the feature level of the JPEG XL codestream. Valid values are 5 and
 * 10, or -1 (to choose automatically). Using the minimum required level, or
 * level 5 in most cases, is recommended for compatibility with all decoders.
 *
 * Level 5: for end-user image delivery, this level is the most widely
 * supported level by image decoders and the recommended level to use unless a
 * level 10 feature is absolutely necessary. Supports a maximum resolution
 * 268435456 pixels total with a maximum width or height of 262144 pixels,
 * maximum 16-bit color channel depth, maximum 120 frames per second for
 * animation, maximum ICC color profile size of 4 MiB, it allows all color
 * models and extra channel types except CMYK and the JXL_CHANNEL_BLACK
 * extra channel, and a maximum of 4 extra channels in addition to the 3 color
 * channels. It also sets boundaries to certain internally used coding tools.
 *
 * Level 10: this level removes or increases the bounds of most of the level
 * 5 limitations, allows CMYK color and up to 32 bits per color channel, but
 * may be less widely supported.
 *
 * The default value is -1. This means the encoder will automatically choose
 * between level 5 and level 10 based on what information is inside the @ref
 * JxlBasicInfo structure. Do note that some level 10 features, particularly
 * those used by animated JPEG XL codestreams, might require level 10, even
 * though the @ref JxlBasicInfo only suggests level 5. In this case, the level
 * must be explicitly set to 10, otherwise the encoder will return an error.
 * The encoder will restrict internal encoding choices to those compatible with
 * the level setting.
 *
 * This setting can only be set at the beginning, before encoding starts.
 *
 * @param enc encoder object.
 * @param level the level value to set, must be -1, 5, or 10.
 * @return ::JXL_ENC_SUCCESS if the operation was successful, @ref
 * JXL_ENC_ERROR otherwise.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetCodestreamLevel(JxlEncoder* enc,
                                                         int level);

/** Returns the codestream level required to support the currently configured
 * settings and basic info. This function can only be used at the beginning,
 * before encoding starts, but after setting basic info.
 *
 * This does not support per-frame settings, only global configuration, such as
 * the image dimensions, that are known at the time of writing the header of
 * the JPEG XL file.
 *
 * If this returns 5, nothing needs to be done and the codestream can be
 * compatible with any decoder. If this returns 10, @ref
 * JxlEncoderSetCodestreamLevel has to be used to set the codestream level to
 * 10, or the encoder can be configured differently to allow using the more
 * compatible level 5.
 *
 * @param enc encoder object.
 * @return -1 if no level can support the configuration (e.g. image dimensions
 * larger than even level 10 supports), 5 if level 5 is supported, 10 if setting
 * the codestream level to 10 is required.
 *
 */
JXL_EXPORT int JxlEncoderGetRequiredCodestreamLevel(const JxlEncoder* enc);

/**
 * Enables lossless encoding.
 *
 * This is not an option like the others on itself, but rather while enabled it
 * overrides a set of existing options (such as distance, modular mode and
 * color transform) that enables bit-for-bit lossless encoding.
 *
 * When disabled, those options are not overridden, but since those options
 * could still have been manually set to a combination that operates losslessly,
 * using this function with lossless set to ::JXL_FALSE does not
 * guarantee lossy encoding, though the default set of options is lossy.
 *
 * @param frame_settings set of options and metadata for this frame. Also
 * includes reference to the encoder object.
 * @param lossless whether to override options for lossless mode
 * @return ::JXL_ENC_SUCCESS if the operation was successful, @ref
 * JXL_ENC_ERROR otherwise.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetFrameLossless(
    JxlEncoderFrameSettings* frame_settings, JXL_BOOL lossless);

/**
 * Sets the distance level for lossy compression: target max butteraugli
 * distance, lower = higher quality. Range: 0 .. 25.
 * 0.0 = mathematically lossless (however, use @ref JxlEncoderSetFrameLossless
 * instead to use true lossless, as setting distance to 0 alone is not the only
 * requirement). 1.0 = visually lossless. Recommended range: 0.5 .. 3.0. Default
 * value: 1.0.
 *
 * @param frame_settings set of options and metadata for this frame. Also
 * includes reference to the encoder object.
 * @param distance the distance value to set.
 * @return ::JXL_ENC_SUCCESS if the operation was successful, @ref
 * JXL_ENC_ERROR otherwise.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetFrameDistance(
    JxlEncoderFrameSettings* frame_settings, float distance);

/**
 * Sets the distance level for lossy compression of extra channels.
 * The distance is as in @ref JxlEncoderSetFrameDistance (lower = higher
 * quality). If not set, or if set to the special value -1, the distance that
 * was set with
 * @ref JxlEncoderSetFrameDistance will be used.
 *
 * @param frame_settings set of options and metadata for this frame. Also
 * includes reference to the encoder object.
 * @param index index of the extra channel to set a distance value for.
 * @param distance the distance value to set.
 * @return ::JXL_ENC_SUCCESS if the operation was successful, @ref
 * JXL_ENC_ERROR otherwise.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetExtraChannelDistance(
    JxlEncoderFrameSettings* frame_settings, size_t index, float distance);

/**
 * Maps JPEG-style quality factor to distance.
 *
 * This function takes in input a JPEG-style quality factor `quality` and
 * produces as output a `distance` value suitable to be used with @ref
 * JxlEncoderSetFrameDistance and @ref JxlEncoderSetExtraChannelDistance.
 *
 * The `distance` value influences the level of compression, with lower values
 * indicating higher quality:
 * - 0.0 implies lossless compression (however, note that calling @ref
 * JxlEncoderSetFrameLossless is required).
 * - 1.0 represents a visually lossy compression, which is also the default
 * setting.
 *
 * The `quality` parameter, ranging up to 100, is inversely related to
 * 'distance':
 * - A `quality` of 100.0 maps to a `distance` of 0.0 (lossless).
 * - A `quality` of 90.0 corresponds to a `distance` of 1.0.
 *
 * Recommended Range:
 * - `distance`: 0.5 to 3.0.
 * - corresponding `quality`: approximately 96 to 68.
 *
 * Allowed Range:
 * - `distance`: 0.0 to 25.0.
 * - corresponding `quality`: 100.0 to 0.0.
 *
 * Note: the `quality` parameter has no consistent psychovisual meaning
 * across different codecs and libraries. Using the mapping defined by @ref
 * JxlEncoderDistanceFromQuality will result in a visual quality roughly
 * equivalent to what would be obtained with `libjpeg-turbo` with the same
 * `quality` parameter, but that is by no means guaranteed; do not assume that
 * the same quality value will result in similar file sizes and image quality
 * across different codecs.
 */
JXL_EXPORT float JxlEncoderDistanceFromQuality(float quality);

/**
 * Create a new set of encoder options, with all values initially copied from
 * the @p source options, or set to default if @p source is NULL.
 *
 * The returned pointer is an opaque struct tied to the encoder and it will be
 * deallocated by the encoder when @ref JxlEncoderDestroy() is called. For
 * functions taking both a @ref JxlEncoder and a @ref JxlEncoderFrameSettings,
 * only @ref JxlEncoderFrameSettings created with this function for the same
 * encoder instance can be used.
 *
 * @param enc encoder object.
 * @param source source options to copy initial values from, or NULL to get
 * defaults initialized to defaults.
 * @return the opaque struct pointer identifying a new set of encoder options.
 */
JXL_EXPORT JxlEncoderFrameSettings* JxlEncoderFrameSettingsCreate(
    JxlEncoder* enc, const JxlEncoderFrameSettings* source);

/**
 * Sets a color encoding to be sRGB.
 *
 * @param color_encoding color encoding instance.
 * @param is_gray whether the color encoding should be gray scale or color.
 */
JXL_EXPORT void JxlColorEncodingSetToSRGB(JxlColorEncoding* color_encoding,
                                          JXL_BOOL is_gray);

/**
 * Sets a color encoding to be linear sRGB.
 *
 * @param color_encoding color encoding instance.
 * @param is_gray whether the color encoding should be gray scale or color.
 */
JXL_EXPORT void JxlColorEncodingSetToLinearSRGB(
    JxlColorEncoding* color_encoding, JXL_BOOL is_gray);

/**
 * Enables usage of expert options.
 *
 * At the moment, the only expert option is setting an effort value of 11,
 * which gives the best compression for pixel-lossless modes but is very slow.
 *
 * @param enc encoder object.
 */
JXL_EXPORT void JxlEncoderAllowExpertOptions(JxlEncoder* enc);

/**
 * Function type for @ref JxlEncoderSetDebugImageCallback.
 *
 * The callback may be called simultaneously by different threads when using a
 * threaded parallel runner, on different debug images.
 *
 * @param opaque optional user data, as given to @ref
 *   JxlEncoderSetDebugImageCallback.
 * @param label label of debug image, can be used in filenames
 * @param xsize width of debug image
 * @param ysize height of debug image
 * @param color color encoding of debug image
 * @param pixels pixel data of debug image as big-endian 16-bit unsigned
 *   samples. The memory is not owned by the user, and is only valid during the
 *   time the callback is running.
 */
typedef void (*JxlDebugImageCallback)(void* opaque, const char* label,
                                      size_t xsize, size_t ysize,
                                      const JxlColorEncoding* color,
                                      const uint16_t* pixels);

/**
 * Sets the given debug image callback that will be used by the encoder to
 * output various debug images during encoding.
 *
 * This only has any effect if the encoder was compiled with the appropriate
 * debug build flags.
 *
 * @param frame_settings set of options and metadata for this frame. Also
 * includes reference to the encoder object.
 * @param callback used to return the debug image
 * @param opaque user supplied parameter to the image callback
 */
JXL_EXPORT void JxlEncoderSetDebugImageCallback(
    JxlEncoderFrameSettings* frame_settings, JxlDebugImageCallback callback,
    void* opaque);

/**
 * Sets the given stats object for gathering various statistics during encoding.
 *
 * This only has any effect if the encoder was compiled with the appropriate
 * debug build flags.
 *
 * @param frame_settings set of options and metadata for this frame. Also
 * includes reference to the encoder object.
 * @param stats object that can be used to query the gathered stats (created
 *   by @ref JxlEncoderStatsCreate)
 */
JXL_EXPORT void JxlEncoderCollectStats(JxlEncoderFrameSettings* frame_settings,
                                       JxlEncoderStats* stats);

#ifdef __cplusplus
}
#endif

#endif /* JXL_ENCODE_H_ */

/** @}*/

/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_decoder
 * @{
 * @file decode.h
 * @brief Decoding API for JPEG XL.
 */

#ifndef JXL_DECODE_H_
#define JXL_DECODE_H_

#include <jxl/cms_interface.h>
#include <jxl/codestream_header.h>
#include <jxl/color_encoding.h>
#include <jxl/jxl_export.h>
#include <jxl/memory_manager.h>
#include <jxl/parallel_runner.h>
#include <jxl/types.h>
#include <jxl/version.h>  // TODO(eustas): remove before v1.0
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Decoder library version.
 *
 * @return the decoder library version as an integer:
 * MAJOR_VERSION * 1000000 + MINOR_VERSION * 1000 + PATCH_VERSION. For example,
 * version 1.2.3 would return 1002003.
 */
JXL_EXPORT uint32_t JxlDecoderVersion(void);

/** The result of @ref JxlSignatureCheck.
 */
typedef enum {
  /** Not enough bytes were passed to determine if a valid signature was found.
   */
  JXL_SIG_NOT_ENOUGH_BYTES = 0,

  /** No valid JPEG XL header was found. */
  JXL_SIG_INVALID = 1,

  /** A valid JPEG XL codestream signature was found, that is a JPEG XL image
   * without container.
   */
  JXL_SIG_CODESTREAM = 2,

  /** A valid container signature was found, that is a JPEG XL image embedded
   * in a box format container.
   */
  JXL_SIG_CONTAINER = 3,
} JxlSignature;

/**
 * JPEG XL signature identification.
 *
 * Checks if the passed buffer contains a valid JPEG XL signature. The passed @p
 * buf of size
 * @p size doesn't need to be a full image, only the beginning of the file.
 *
 * @return a flag indicating if a JPEG XL signature was found and what type.
 *  - ::JXL_SIG_NOT_ENOUGH_BYTES if not enough bytes were passed to
 *    determine if a valid signature is there.
 *  - ::JXL_SIG_INVALID if no valid signature found for JPEG XL decoding.
 *  - ::JXL_SIG_CODESTREAM if a valid JPEG XL codestream signature was
 *    found.
 *  - ::JXL_SIG_CONTAINER if a valid JPEG XL container signature was found.
 */
JXL_EXPORT JxlSignature JxlSignatureCheck(const uint8_t* buf, size_t len);

/**
 * Opaque structure that holds the JPEG XL decoder.
 *
 * Allocated and initialized with @ref JxlDecoderCreate().
 * Cleaned up and deallocated with @ref JxlDecoderDestroy().
 */
typedef struct JxlDecoderStruct JxlDecoder;

/**
 * Creates an instance of @ref JxlDecoder and initializes it.
 *
 * @p memory_manager will be used for all the library dynamic allocations made
 * from this instance. The parameter may be NULL, in which case the default
 * allocator will be used. See jxl/memory_manager.h for details.
 *
 * @param memory_manager custom allocator function. It may be NULL. The memory
 *        manager will be copied internally.
 * @return @c NULL if the instance can not be allocated or initialized
 * @return pointer to initialized @ref JxlDecoder otherwise
 */
JXL_EXPORT JxlDecoder* JxlDecoderCreate(const JxlMemoryManager* memory_manager);

/**
 * Re-initializes a @ref JxlDecoder instance, so it can be re-used for decoding
 * another image. All state and settings are reset as if the object was
 * newly created with @ref JxlDecoderCreate, but the memory manager is kept.
 *
 * @param dec instance to be re-initialized.
 */
JXL_EXPORT void JxlDecoderReset(JxlDecoder* dec);

/**
 * Deinitializes and frees @ref JxlDecoder instance.
 *
 * @param dec instance to be cleaned up and deallocated.
 */
JXL_EXPORT void JxlDecoderDestroy(JxlDecoder* dec);

/**
 * Return value for @ref JxlDecoderProcessInput.
 * The values from ::JXL_DEC_BASIC_INFO onwards are optional informative
 * events that can be subscribed to, they are never returned if they
 * have not been registered with @ref JxlDecoderSubscribeEvents.
 */
typedef enum {
  /** Function call finished successfully, or decoding is finished and there is
   * nothing more to be done.
   *
   * Note that @ref JxlDecoderProcessInput will return ::JXL_DEC_SUCCESS if
   * all events that were registered with @ref JxlDecoderSubscribeEvents were
   * processed, even before the end of the JPEG XL codestream.
   *
   * In this case, the return value @ref JxlDecoderReleaseInput will be the same
   * as it was at the last signaled event. E.g. if ::JXL_DEC_FULL_IMAGE was
   * subscribed to, then all bytes from the end of the JPEG XL codestream
   * (including possible boxes needed for jpeg reconstruction) will be returned
   * as unprocessed.
   */
  JXL_DEC_SUCCESS = 0,

  /** An error occurred, for example invalid input file or out of memory.
   * TODO(lode): add function to get error information from decoder.
   */
  JXL_DEC_ERROR = 1,

  /** The decoder needs more input bytes to continue. Before the next @ref
   * JxlDecoderProcessInput call, more input data must be set, by calling @ref
   * JxlDecoderReleaseInput (if input was set previously) and then calling @ref
   * JxlDecoderSetInput. @ref JxlDecoderReleaseInput returns how many bytes
   * are not yet processed, before a next call to @ref JxlDecoderProcessInput
   * all unprocessed bytes must be provided again (the address need not match,
   * but the contents must), and more bytes must be concatenated after the
   * unprocessed bytes.
   * In most cases, @ref JxlDecoderReleaseInput will return no unprocessed bytes
   * at this event, the only exceptions are if the previously set input ended
   * within (a) the raw codestream signature, (b) the signature box, (c) a box
   * header, or (d) the first 4 bytes of a `brob`, `ftyp`, or `jxlp` box. In any
   * of these cases the number of unprocessed bytes is less than 20.
   */
  JXL_DEC_NEED_MORE_INPUT = 2,

  /** The decoder is able to decode a preview image and requests setting a
   * preview output buffer using @ref JxlDecoderSetPreviewOutBuffer. This occurs
   * if ::JXL_DEC_PREVIEW_IMAGE is requested and it is possible to decode a
   * preview image from the codestream and the preview out buffer was not yet
   * set. There is maximum one preview image in a codestream.
   * In this case, @ref JxlDecoderReleaseInput will return all bytes from the
   * end of the frame header (including ToC) of the preview frame as
   * unprocessed.
   */
  JXL_DEC_NEED_PREVIEW_OUT_BUFFER = 3,

  /** The decoder requests an output buffer to store the full resolution image,
   * which can be set with @ref JxlDecoderSetImageOutBuffer or with @ref
   * JxlDecoderSetImageOutCallback. This event re-occurs for new frames if
   * there are multiple animation frames and requires setting an output again.
   * In this case, @ref JxlDecoderReleaseInput will return all bytes from the
   * end of the frame header (including ToC) as unprocessed.
   */
  JXL_DEC_NEED_IMAGE_OUT_BUFFER = 5,

  /** The JPEG reconstruction buffer is too small for reconstructed JPEG
   * codestream to fit. @ref JxlDecoderSetJPEGBuffer must be called again to
   * make room for remaining bytes. This event may occur multiple times
   * after ::JXL_DEC_JPEG_RECONSTRUCTION.
   */
  JXL_DEC_JPEG_NEED_MORE_OUTPUT = 6,

  /** The box contents output buffer is too small. @ref JxlDecoderSetBoxBuffer
   * must be called again to make room for remaining bytes. This event may occur
   * multiple times after ::JXL_DEC_BOX.
   */
  JXL_DEC_BOX_NEED_MORE_OUTPUT = 7,

  /** Informative event by @ref JxlDecoderProcessInput
   * "JxlDecoderProcessInput": Basic information such as image dimensions and
   * extra channels. This event occurs max once per image.
   * In this case, @ref JxlDecoderReleaseInput will return all bytes from the
   * end of the basic info as unprocessed (including the last byte of basic info
   * if it did not end on a byte boundary).
   */
  JXL_DEC_BASIC_INFO = 0x40,

  /** Informative event by @ref JxlDecoderProcessInput
   * "JxlDecoderProcessInput": Color encoding or ICC profile from the
   * codestream header. This event occurs max once per image and always later
   * than ::JXL_DEC_BASIC_INFO and earlier than any pixel data.
   * In this case, @ref JxlDecoderReleaseInput will return all bytes from the
   * end of the image header (which is the start of the first frame) as
   * unprocessed.
   */
  JXL_DEC_COLOR_ENCODING = 0x100,

  /** Informative event by @ref JxlDecoderProcessInput
   * "JxlDecoderProcessInput": Preview image, a small frame, decoded. This
   * event can only happen if the image has a preview frame encoded. This event
   * occurs max once for the codestream and always later than @ref
   * JXL_DEC_COLOR_ENCODING and before ::JXL_DEC_FRAME.
   * In this case, @ref JxlDecoderReleaseInput will return all bytes from the
   * end of the preview frame as unprocessed.
   */
  JXL_DEC_PREVIEW_IMAGE = 0x200,

  /** Informative event by @ref JxlDecoderProcessInput
   * "JxlDecoderProcessInput": Beginning of a frame. @ref
   * JxlDecoderGetFrameHeader can be used at this point. A note on frames:
   * a JPEG XL image can have internal frames that are not intended to be
   * displayed (e.g. used for compositing a final frame), but this only returns
   * displayed frames, unless @ref JxlDecoderSetCoalescing was set to @ref
   * JXL_FALSE "JXL_FALSE": in that case, the individual layers are returned,
   * without blending. Note that even when coalescing is disabled, only frames
   * of type kRegularFrame are returned; frames of type kReferenceOnly
   * and kLfFrame are always for internal purposes only and cannot be accessed.
   * A displayed frame either has an animation duration or is the only or last
   * frame in the image. This event occurs max once per displayed frame, always
   * later than ::JXL_DEC_COLOR_ENCODING, and always earlier than any pixel
   * data. While JPEG XL supports encoding a single frame as the composition of
   * multiple internal sub-frames also called frames, this event is not
   * indicated for the internal frames. In this case, @ref
   * JxlDecoderReleaseInput will return all bytes from the end of the frame
   * header (including ToC) as unprocessed.
   */
  JXL_DEC_FRAME = 0x400,

  /** Informative event by @ref JxlDecoderProcessInput
   * "JxlDecoderProcessInput": full frame (or layer, in case coalescing is
   * disabled) is decoded. @ref JxlDecoderSetImageOutBuffer must be used after
   * getting the basic image information to be able to get the image pixels, if
   * not this return status only indicates we're past this point in the
   * codestream. This event occurs max once per frame.
   * In this case, @ref JxlDecoderReleaseInput will return all bytes from the
   * end of the frame (or if ::JXL_DEC_JPEG_RECONSTRUCTION is subscribed to,
   * from the end of the last box that is needed for jpeg reconstruction) as
   * unprocessed.
   */
  JXL_DEC_FULL_IMAGE = 0x1000,

  /** Informative event by @ref JxlDecoderProcessInput
   * "JxlDecoderProcessInput": JPEG reconstruction data decoded. @ref
   * JxlDecoderSetJPEGBuffer may be used to set a JPEG reconstruction buffer
   * after getting the JPEG reconstruction data. If a JPEG reconstruction buffer
   * is set a byte stream identical to the JPEG codestream used to encode the
   * image will be written to the JPEG reconstruction buffer instead of pixels
   * to the image out buffer. This event occurs max once per image and always
   * before ::JXL_DEC_FULL_IMAGE.
   * In this case, @ref JxlDecoderReleaseInput will return all bytes from the
   * end of the `jbrd` box as unprocessed.
   */
  JXL_DEC_JPEG_RECONSTRUCTION = 0x2000,

  /** Informative event by @ref JxlDecoderProcessInput
   * "JxlDecoderProcessInput": The header of a box of the container format
   * (BMFF) is decoded. The following API functions related to boxes can be used
   * after this event:
   *  - @ref JxlDecoderSetBoxBuffer and @ref JxlDecoderReleaseBoxBuffer
   *    "JxlDecoderReleaseBoxBuffer": set and release a buffer to get the box
   *    data.
   *  - @ref JxlDecoderGetBoxType get the 4-character box typename.
   *  - @ref JxlDecoderGetBoxSizeRaw get the size of the box as it appears in
   *    the container file, not decompressed.
   *  - @ref JxlDecoderSetDecompressBoxes to configure whether to get the box
   *    data decompressed, or possibly compressed.
   *
   * Boxes can be compressed. This is so when their box type is
   * "brob". In that case, they have an underlying decompressed box
   * type and decompressed data. @ref JxlDecoderSetDecompressBoxes allows
   * configuring which data to get. Decompressing requires
   * Brotli. @ref JxlDecoderGetBoxType has a flag to get the compressed box
   * type, which can be "brob", or the decompressed box type. If a box
   * is not compressed (its compressed type is not "brob"), then
   * the output decompressed box type and data is independent of what
   * setting is configured.
   *
   * The buffer set with @ref JxlDecoderSetBoxBuffer must be set again for each
   * next box to be obtained, or can be left unset to skip outputting this box.
   * The output buffer contains the full box data when the
   * ::JXL_DEC_BOX_COMPLETE (if subscribed to) or subsequent ::JXL_DEC_SUCCESS
   * or ::JXL_DEC_BOX event occurs. ::JXL_DEC_BOX occurs for all boxes,
   * including non-metadata boxes such as the signature box or codestream boxes.
   * To check whether the box is a metadata type for respectively EXIF, XMP or
   * JUMBF, use @ref JxlDecoderGetBoxType and check for types "Exif", "xml " and
   * "jumb" respectively.
   *
   * In this case, @ref JxlDecoderReleaseInput will return all bytes from the
   * start of the box header as unprocessed.
   */
  JXL_DEC_BOX = 0x4000,

  /** Informative event by @ref JxlDecoderProcessInput
   * "JxlDecoderProcessInput": a progressive step in decoding the frame is
   * reached. When calling @ref JxlDecoderFlushImage at this point, the flushed
   * image will correspond exactly to this point in decoding, and not yet
   * contain partial results (such as partially more fine detail) of a next
   * step. By default, this event will trigger maximum once per frame, when a
   * 8x8th resolution (DC) image is ready (the image data is still returned at
   * full resolution, giving upscaled DC). Use @ref
   * JxlDecoderSetProgressiveDetail to configure more fine-grainedness. The
   * event is not guaranteed to trigger, not all images have progressive steps
   * or DC encoded.
   * In this case, @ref JxlDecoderReleaseInput will return all bytes from the
   * end of the section that was needed to produce this progressive event as
   * unprocessed.
   */
  JXL_DEC_FRAME_PROGRESSION = 0x8000,

  /** The box being decoded is now complete. This is only emitted if a buffer
   * was set for the box.
   */
  JXL_DEC_BOX_COMPLETE = 0x10000,
} JxlDecoderStatus;

/** Types of progressive detail.
 * Setting a progressive detail with value N implies all progressive details
 * with smaller or equal value. Currently only the following level of
 * progressive detail is implemented:
 *  - @ref kDC (which implies kFrames)
 *  - @ref kLastPasses (which implies @ref kDC and @ref kFrames)
 *  - @ref kPasses (which implies @ref kLastPasses, kDC and @ref kFrames)
 */
typedef enum {
  /**
   * after completed kRegularFrames
   */
  kFrames = 0,
  /**
   * after completed DC (1:8)
   */
  kDC = 1,
  /**
   * after completed AC passes that are the last pass for their resolution
   * target.
   */
  kLastPasses = 2,
  /**
   * after completed AC passes that are not the last pass for their resolution
   * target.
   */
  kPasses = 3,
  /**
   * during DC frame when lower resolution are completed (1:32, 1:16)
   */
  kDCProgressive = 4,
  /**
   * after completed groups
   */
  kDCGroups = 5,
  /**
   * after completed groups
   */
  kGroups = 6,
} JxlProgressiveDetail;

/** Rewinds decoder to the beginning. The same input must be given again from
 * the beginning of the file and the decoder will emit events from the beginning
 * again. When rewinding (as opposed to @ref JxlDecoderReset), the decoder can
 * keep state about the image, which it can use to skip to a requested frame
 * more efficiently with @ref JxlDecoderSkipFrames. Settings such as parallel
 * runner or subscribed events are kept. After rewind, @ref
 * JxlDecoderSubscribeEvents can be used again, and it is feasible to leave out
 * events that were already handled before, such as ::JXL_DEC_BASIC_INFO
 * and ::JXL_DEC_COLOR_ENCODING, since they will provide the same information
 * as before.
 * The difference to @ref JxlDecoderReset is that some state is kept, namely
 * settings set by a call to
 *  - @ref JxlDecoderSetCoalescing,
 *  - @ref JxlDecoderSetDesiredIntensityTarget,
 *  - @ref JxlDecoderSetDecompressBoxes,
 *  - @ref JxlDecoderSetKeepOrientation,
 *  - @ref JxlDecoderSetUnpremultiplyAlpha,
 *  - @ref JxlDecoderSetParallelRunner,
 *  - @ref JxlDecoderSetRenderSpotcolors, and
 *  - @ref JxlDecoderSubscribeEvents.
 *
 * @param dec decoder object
 */
JXL_EXPORT void JxlDecoderRewind(JxlDecoder* dec);

/** Makes the decoder skip the next `amount` frames. It still needs to process
 * the input, but will not output the frame events. It can be more efficient
 * when skipping frames, and even more so when using this after @ref
 * JxlDecoderRewind. If the decoder is already processing a frame (could
 * have emitted ::JXL_DEC_FRAME but not yet ::JXL_DEC_FULL_IMAGE), it
 * starts skipping from the next frame. If the amount is larger than the amount
 * of frames remaining in the image, all remaining frames are skipped. Calling
 * this function multiple times adds the amount to skip to the already existing
 * amount.
 *
 * A frame here is defined as a frame that without skipping emits events such
 * as ::JXL_DEC_FRAME and ::JXL_DEC_FULL_IMAGE, frames that are internal
 * to the file format but are not rendered as part of an animation, or are not
 * the final still frame of a still image, are not counted.
 *
 * @param dec decoder object
 * @param amount the amount of frames to skip
 */
JXL_EXPORT void JxlDecoderSkipFrames(JxlDecoder* dec, size_t amount);

/**
 * Skips processing the current frame. Can be called after frame processing
 * already started, signaled by a ::JXL_DEC_NEED_IMAGE_OUT_BUFFER event,
 * but before the corresponding ::JXL_DEC_FULL_IMAGE event. The next signaled
 * event will be another ::JXL_DEC_FRAME, or ::JXL_DEC_SUCCESS if there
 * are no more frames. If pixel data is required from the already processed part
 * of the frame, @ref JxlDecoderFlushImage must be called before this.
 *
 * @param dec decoder object
 * @return ::JXL_DEC_SUCCESS if there is a frame to skip, and @ref
 *     JXL_DEC_ERROR if the function was not called during frame processing.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderSkipCurrentFrame(JxlDecoder* dec);

/**
 * Set the parallel runner for multithreading. May only be set before starting
 * decoding.
 *
 * @param dec decoder object
 * @param parallel_runner function pointer to runner for multithreading. It may
 *     be NULL to use the default, single-threaded, runner. A multithreaded
 *     runner should be set to reach fast performance.
 * @param parallel_runner_opaque opaque pointer for parallel_runner.
 * @return ::JXL_DEC_SUCCESS if the runner was set, ::JXL_DEC_ERROR
 *     otherwise (the previous runner remains set).
 */
JXL_EXPORT JxlDecoderStatus
JxlDecoderSetParallelRunner(JxlDecoder* dec, JxlParallelRunner parallel_runner,
                            void* parallel_runner_opaque);

/**
 * Returns a hint indicating how many more bytes the decoder is expected to
 * need to make @ref JxlDecoderGetBasicInfo available after the next @ref
 * JxlDecoderProcessInput call. This is a suggested large enough value for
 * the amount of bytes to provide in the next @ref JxlDecoderSetInput call, but
 * it is not guaranteed to be an upper bound nor a lower bound. This number does
 * not include bytes that have already been released from the input. Can be used
 * before the first @ref JxlDecoderProcessInput call, and is correct the first
 * time in most cases. If not, @ref JxlDecoderSizeHintBasicInfo can be called
 * again to get an updated hint.
 *
 * @param dec decoder object
 * @return the size hint in bytes if the basic info is not yet fully decoded.
 * @return 0 when the basic info is already available.
 */
JXL_EXPORT size_t JxlDecoderSizeHintBasicInfo(const JxlDecoder* dec);

/** Select for which informative events, i.e. ::JXL_DEC_BASIC_INFO, etc., the
 * decoder should return with a status. It is not required to subscribe to any
 * events, data can still be requested from the decoder as soon as it available.
 * By default, the decoder is subscribed to no events (events_wanted == 0), and
 * the decoder will then only return when it cannot continue because it needs
 * more input data or more output buffer. This function may only be be called
 * before using @ref JxlDecoderProcessInput.
 *
 * @param dec decoder object
 * @param events_wanted bitfield of desired events.
 * @return ::JXL_DEC_SUCCESS if no error, ::JXL_DEC_ERROR otherwise.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderSubscribeEvents(JxlDecoder* dec,
                                                      int events_wanted);

/** Enables or disables preserving of as-in-bitstream pixeldata
 * orientation. Some images are encoded with an Orientation tag
 * indicating that the decoder must perform a rotation and/or
 * mirroring to the encoded image data.
 *
 *  - If skip_reorientation is ::JXL_FALSE (the default): the decoder
 *    will apply the transformation from the orientation setting, hence
 *    rendering the image according to its specified intent. When
 *    producing a @ref JxlBasicInfo, the decoder will always set the
 *    orientation field to JXL_ORIENT_IDENTITY (matching the returned
 *    pixel data) and also align xsize and ysize so that they correspond
 *    to the width and the height of the returned pixel data.
 *  - If skip_reorientation is ::JXL_TRUE "JXL_TRUE": the decoder will skip
 *    applying the transformation from the orientation setting, returning
 *    the image in the as-in-bitstream pixeldata orientation.
 *    This may be faster to decode since the decoder doesn't have to apply the
 *    transformation, but can cause wrong display of the image if the
 *    orientation tag is not correctly taken into account by the user.
 *
 * By default, this option is disabled, and the returned pixel data is
 * re-oriented according to the image's Orientation setting.
 *
 * This function must be called at the beginning, before decoding is performed.
 *
 * @see JxlBasicInfo for the orientation field, and @ref JxlOrientation for the
 * possible values.
 *
 * @param dec decoder object
 * @param skip_reorientation JXL_TRUE to enable, JXL_FALSE to disable.
 * @return ::JXL_DEC_SUCCESS if no error, ::JXL_DEC_ERROR otherwise.
 */
JXL_EXPORT JxlDecoderStatus
JxlDecoderSetKeepOrientation(JxlDecoder* dec, JXL_BOOL skip_reorientation);

/**
 * Enables or disables preserving of associated alpha channels. If
 * unpremul_alpha is set to ::JXL_FALSE then for associated alpha channel,
 * the pixel data is returned with premultiplied colors. If it is set to @ref
 * JXL_TRUE, The colors will be unpremultiplied based on the alpha channel. This
 * function has no effect if the image does not have an associated alpha
 * channel.
 *
 * By default, this option is disabled, and the returned pixel data "as is".
 *
 * This function must be called at the beginning, before decoding is performed.
 *
 * @param dec decoder object
 * @param unpremul_alpha JXL_TRUE to enable, JXL_FALSE to disable.
 * @return ::JXL_DEC_SUCCESS if no error, ::JXL_DEC_ERROR otherwise.
 */
JXL_EXPORT JxlDecoderStatus
JxlDecoderSetUnpremultiplyAlpha(JxlDecoder* dec, JXL_BOOL unpremul_alpha);

/** Enables or disables rendering spot colors. By default, spot colors
 * are rendered, which is OK for viewing the decoded image. If render_spotcolors
 * is ::JXL_FALSE, then spot colors are not rendered, and have to be
 * retrieved separately using @ref JxlDecoderSetExtraChannelBuffer. This is
 * useful for e.g. printing applications.
 *
 * @param dec decoder object
 * @param render_spotcolors JXL_TRUE to enable (default), JXL_FALSE to disable.
 * @return ::JXL_DEC_SUCCESS if no error, ::JXL_DEC_ERROR otherwise.
 */
JXL_EXPORT JxlDecoderStatus
JxlDecoderSetRenderSpotcolors(JxlDecoder* dec, JXL_BOOL render_spotcolors);

/** Enables or disables coalescing of zero-duration frames. By default, frames
 * are returned with coalescing enabled, i.e. all frames have the image
 * dimensions, and are blended if needed. When coalescing is disabled, frames
 * can have arbitrary dimensions, a non-zero crop offset, and blending is not
 * performed. For display, coalescing is recommended. For loading a multi-layer
 * still image as separate layers (as opposed to the merged image), coalescing
 * has to be disabled.
 *
 * @param dec decoder object
 * @param coalescing JXL_TRUE to enable coalescing (default), JXL_FALSE to
 *     disable it.
 * @return ::JXL_DEC_SUCCESS if no error, ::JXL_DEC_ERROR otherwise.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderSetCoalescing(JxlDecoder* dec,
                                                    JXL_BOOL coalescing);

/**
 * Decodes JPEG XL file using the available bytes. Requires input has been
 * set with @ref JxlDecoderSetInput. After @ref JxlDecoderProcessInput, input
 * can optionally be released with @ref JxlDecoderReleaseInput and then set
 * again to next bytes in the stream. @ref JxlDecoderReleaseInput returns how
 * many bytes are not yet processed, before a next call to @ref
 * JxlDecoderProcessInput all unprocessed bytes must be provided again (the
 * address need not match, but the contents must), and more bytes may be
 * concatenated after the unprocessed bytes.
 *
 * The returned status indicates whether the decoder needs more input bytes, or
 * more output buffer for a certain type of output data. No matter what the
 * returned status is (other than ::JXL_DEC_ERROR), new information, such
 * as @ref JxlDecoderGetBasicInfo, may have become available after this call.
 * When the return value is not ::JXL_DEC_ERROR or ::JXL_DEC_SUCCESS, the
 * decoding requires more @ref JxlDecoderProcessInput calls to continue.
 *
 * @param dec decoder object
 * @return ::JXL_DEC_SUCCESS when decoding finished and all events handled.
 *     If you still have more unprocessed input data anyway, then you can still
 *     continue by using @ref JxlDecoderSetInput and calling @ref
 *     JxlDecoderProcessInput again, similar to handling @ref
 *     JXL_DEC_NEED_MORE_INPUT. ::JXL_DEC_SUCCESS can occur instead of @ref
 *     JXL_DEC_NEED_MORE_INPUT when, for example, the input data ended right at
 *     the boundary of a box of the container format, all essential codestream
 *     boxes were already decoded, but extra metadata boxes are still present in
 *     the next data. @ref JxlDecoderProcessInput cannot return success if all
 *     codestream boxes have not been seen yet.
 * @return ::JXL_DEC_ERROR when decoding failed, e.g. invalid codestream.
 *     TODO(lode): document the input data mechanism
 * @return ::JXL_DEC_NEED_MORE_INPUT when more input data is necessary.
 * @return ::JXL_DEC_BASIC_INFO when basic info such as image dimensions is
 *     available and this informative event is subscribed to.
 * @return ::JXL_DEC_COLOR_ENCODING when color profile information is
 *     available and this informative event is subscribed to.
 * @return ::JXL_DEC_PREVIEW_IMAGE when preview pixel information is
 *     available and output in the preview buffer.
 * @return ::JXL_DEC_FULL_IMAGE when all pixel information at highest detail
 *     is available and has been output in the pixel buffer.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderProcessInput(JxlDecoder* dec);

/**
 * Sets input data for @ref JxlDecoderProcessInput. The data is owned by the
 * caller and may be used by the decoder until @ref JxlDecoderReleaseInput is
 * called or the decoder is destroyed or reset so must be kept alive until then.
 * Cannot be called if @ref JxlDecoderSetInput was already called and @ref
 * JxlDecoderReleaseInput was not yet called, and cannot be called after @ref
 * JxlDecoderCloseInput indicating the end of input was called.
 *
 * @param dec decoder object
 * @param data pointer to next bytes to read from
 * @param size amount of bytes available starting from data
 * @return ::JXL_DEC_ERROR if input was already set without releasing or @ref
 *     JxlDecoderCloseInput was already called, ::JXL_DEC_SUCCESS otherwise.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderSetInput(JxlDecoder* dec,
                                               const uint8_t* data,
                                               size_t size);

/**
 * Releases input which was provided with @ref JxlDecoderSetInput. Between @ref
 * JxlDecoderProcessInput and @ref JxlDecoderReleaseInput, the user may not
 * alter the data in the buffer. Calling @ref JxlDecoderReleaseInput is required
 * whenever any input is already set and new input needs to be added with @ref
 * JxlDecoderSetInput, but is not required before @ref JxlDecoderDestroy or @ref
 * JxlDecoderReset. Calling @ref JxlDecoderReleaseInput when no input is set is
 * not an error and returns `0`.
 *
 * @param dec decoder object
 * @return The amount of bytes the decoder has not yet processed that are still
 *     remaining in the data set by @ref JxlDecoderSetInput, or `0` if no input
 * is set or @ref JxlDecoderReleaseInput was already called. For a next call to
 * @ref JxlDecoderProcessInput, the buffer must start with these unprocessed
 * bytes. From this value it is possible to infer the position of certain JPEG
 * XL codestream elements (e.g. end of headers, frame start/end). See the
 * documentation of individual values of @ref JxlDecoderStatus for more
 * information.
 */
JXL_EXPORT size_t JxlDecoderReleaseInput(JxlDecoder* dec);

/**
 * Marks the input as finished, indicates that no more @ref JxlDecoderSetInput
 * will be called. This function allows the decoder to determine correctly if it
 * should return success, need more input or error in certain cases. For
 * backwards compatibility with a previous version of the API, using this
 * function is optional when not using the ::JXL_DEC_BOX event (the decoder
 * is able to determine the end of the image frames without marking the end),
 * but using this function is required when using ::JXL_DEC_BOX for getting
 * metadata box contents. This function does not replace @ref
 * JxlDecoderReleaseInput, that function should still be called if its return
 * value is needed.
 *
 * @ref JxlDecoderCloseInput should be called as soon as all known input bytes
 * are set (e.g. at the beginning when not streaming but setting all input
 * at once), before the final @ref JxlDecoderProcessInput calls.
 *
 * @param dec decoder object
 */
JXL_EXPORT void JxlDecoderCloseInput(JxlDecoder* dec);

/**
 * Outputs the basic image information, such as image dimensions, bit depth and
 * all other JxlBasicInfo fields, if available.
 *
 * @param dec decoder object
 * @param info struct to copy the information into, or NULL to only check
 *     whether the information is available through the return value.
 * @return ::JXL_DEC_SUCCESS if the value is available, @ref
 *     JXL_DEC_NEED_MORE_INPUT if not yet available, ::JXL_DEC_ERROR
 *     in case of other error conditions.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderGetBasicInfo(const JxlDecoder* dec,
                                                   JxlBasicInfo* info);

/**
 * Outputs information for extra channel at the given index. The index must be
 * smaller than num_extra_channels in the associated @ref JxlBasicInfo.
 *
 * @param dec decoder object
 * @param index index of the extra channel to query.
 * @param info struct to copy the information into, or NULL to only check
 *     whether the information is available through the return value.
 * @return ::JXL_DEC_SUCCESS if the value is available, @ref
 *     JXL_DEC_NEED_MORE_INPUT if not yet available, ::JXL_DEC_ERROR
 *     in case of other error conditions.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderGetExtraChannelInfo(
    const JxlDecoder* dec, size_t index, JxlExtraChannelInfo* info);

/**
 * Outputs name for extra channel at the given index in UTF-8. The index must be
 * smaller than `num_extra_channels` in the associated @ref JxlBasicInfo. The
 * buffer for name must have at least `name_length + 1` bytes allocated, gotten
 * from the associated @ref JxlExtraChannelInfo.
 *
 * @param dec decoder object
 * @param index index of the extra channel to query.
 * @param name buffer to copy the name into
 * @param size size of the name buffer in bytes
 * @return ::JXL_DEC_SUCCESS if the value is available, @ref
 *     JXL_DEC_NEED_MORE_INPUT if not yet available, ::JXL_DEC_ERROR
 *     in case of other error conditions.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderGetExtraChannelName(const JxlDecoder* dec,
                                                          size_t index,
                                                          char* name,
                                                          size_t size);

/** Defines which color profile to get: the profile from the codestream
 * metadata header, which represents the color profile of the original image,
 * or the color profile from the pixel data produced by the decoder. Both are
 * the same if the JxlBasicInfo has uses_original_profile set.
 */
typedef enum {
  /** Get the color profile of the original image from the metadata.
   */
  JXL_COLOR_PROFILE_TARGET_ORIGINAL = 0,

  /** Get the color profile of the pixel data the decoder outputs. */
  JXL_COLOR_PROFILE_TARGET_DATA = 1,
} JxlColorProfileTarget;

/**
 * Outputs the color profile as JPEG XL encoded structured data, if available.
 * This is an alternative to an ICC Profile, which can represent a more limited
 * amount of color spaces, but represents them exactly through enum values.
 *
 * It is often possible to use @ref JxlDecoderGetColorAsICCProfile as an
 * alternative anyway. The following scenarios are possible:
 *  - The JPEG XL image has an attached ICC Profile, in that case, the encoded
 *    structured data is not available and this function will return an error
 *    status. @ref JxlDecoderGetColorAsICCProfile should be called instead.
 *  - The JPEG XL image has an encoded structured color profile, and it
 *    represents an RGB or grayscale color space. This function will return it.
 *    You can still use @ref JxlDecoderGetColorAsICCProfile as well as an
 *    alternative if desired, though depending on which RGB color space is
 *    represented, the ICC profile may be a close approximation. It is also not
 *    always feasible to deduce from an ICC profile which named color space it
 *    exactly represents, if any, as it can represent any arbitrary space.
 *    HDR color spaces such as those using PQ and HLG are also potentially
 *    problematic, in that: while ICC profiles can encode a transfer function
 *    that happens to approximate those of PQ and HLG (HLG for only one given
 *    system gamma at a time, and necessitating a 3D LUT if gamma is to be
 *    different from `1`), they cannot (before ICCv4.4) semantically signal that
 *    this is the color space that they represent. Therefore, they will
 *    typically not actually be interpreted as representing an HDR color space.
 *    This is especially detrimental to PQ which will then be interpreted as if
 *    the maximum signal value represented SDR white instead of 10000 cd/m^2,
 *    meaning that the image will be displayed two orders of magnitude (5-7 EV)
 *    too dim.
 *  - The JPEG XL image has an encoded structured color profile, and it
 *    indicates an unknown or xyb color space. In that case, @ref
 *    JxlDecoderGetColorAsICCProfile is not available.
 *
 * When rendering an image on a system where ICC-based color management is used,
 * @ref JxlDecoderGetColorAsICCProfile should generally be used first as it will
 * return a ready-to-use profile (with the aforementioned caveat about HDR).
 * When knowledge about the nominal color space is desired if available, @ref
 * JxlDecoderGetColorAsEncodedProfile should be used first.
 *
 * @param dec decoder object
 * @param target whether to get the original color profile from the metadata
 *     or the color profile of the decoded pixels.
 * @param color_encoding struct to copy the information into, or NULL to only
 *     check whether the information is available through the return value.
 * @return ::JXL_DEC_SUCCESS if the data is available and returned, @ref
 *     JXL_DEC_NEED_MORE_INPUT if not yet available, ::JXL_DEC_ERROR in
 *     case the encoded structured color profile does not exist in the
 *     codestream.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderGetColorAsEncodedProfile(
    const JxlDecoder* dec, JxlColorProfileTarget target,
    JxlColorEncoding* color_encoding);

/**
 * Outputs the size in bytes of the ICC profile returned by @ref
 * JxlDecoderGetColorAsICCProfile, if available, or indicates there is none
 * available. In most cases, the image will have an ICC profile available, but
 * if it does not, @ref JxlDecoderGetColorAsEncodedProfile must be used instead.
 *
 * @see JxlDecoderGetColorAsEncodedProfile for more information. The ICC
 * profile is either the exact ICC profile attached to the codestream metadata,
 * or a close approximation generated from JPEG XL encoded structured data,
 * depending of what is encoded in the codestream.
 *
 * @param dec decoder object
 * @param target whether to get the original color profile from the metadata
 *     or the color profile of the decoded pixels.
 * @param size variable to output the size into, or NULL to only check the
 *     return status.
 * @return ::JXL_DEC_SUCCESS if the ICC profile is available, @ref
 *     JXL_DEC_NEED_MORE_INPUT if the decoder has not yet received enough
 *     input data to determine whether an ICC profile is available or what its
 *     size is, ::JXL_DEC_ERROR in case the ICC profile is not available and
 *     cannot be generated.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderGetICCProfileSize(
    const JxlDecoder* dec, JxlColorProfileTarget target, size_t* size);

/**
 * Outputs ICC profile if available. The profile is only available if @ref
 * JxlDecoderGetICCProfileSize returns success. The output buffer must have
 * at least as many bytes as given by @ref JxlDecoderGetICCProfileSize.
 *
 * @param dec decoder object
 * @param target whether to get the original color profile from the metadata
 *     or the color profile of the decoded pixels.
 * @param icc_profile buffer to copy the ICC profile into
 * @param size size of the icc_profile buffer in bytes
 * @return ::JXL_DEC_SUCCESS if the profile was successfully returned,
 *     ::JXL_DEC_NEED_MORE_INPUT if not yet available, @ref
 *     JXL_DEC_ERROR if the profile doesn't exist or the output size is not
 *     large enough.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderGetColorAsICCProfile(
    const JxlDecoder* dec, JxlColorProfileTarget target, uint8_t* icc_profile,
    size_t size);

/** Sets the desired output color profile of the decoded image by calling
 * @ref JxlDecoderSetOutputColorProfile, passing on @c color_encoding and
 * setting @c icc_data to NULL. See @ref JxlDecoderSetOutputColorProfile for
 * details.
 *
 * @param dec decoder object
 * @param color_encoding the default color encoding to set
 * @return ::JXL_DEC_SUCCESS if the preference was set successfully, @ref
 *     JXL_DEC_ERROR otherwise.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderSetPreferredColorProfile(
    JxlDecoder* dec, const JxlColorEncoding* color_encoding);

/** Requests that the decoder perform tone mapping to the peak display luminance
 * passed as @c desired_intensity_target, if appropriate.
 * @note This is provided for convenience and the exact tone mapping that is
 * performed is not meant to be considered authoritative in any way. It may
 * change from version to version.
 * @param dec decoder object
 * @param desired_intensity_target the intended target peak luminance
 * @return ::JXL_DEC_SUCCESS if the preference was set successfully, @ref
 * JXL_DEC_ERROR otherwise.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderSetDesiredIntensityTarget(
    JxlDecoder* dec, float desired_intensity_target);

/**
 * Sets the desired output color profile of the decoded image either from a
 * color encoding or an ICC profile. Valid calls of this function have either @c
 * color_encoding or @c icc_data set to NULL and @c icc_size must be `0` if and
 * only if @c icc_data is NULL.
 *
 * Depending on whether a color management system (CMS) has been set the
 * behavior is as follows:
 *
 * If a color management system (CMS) has been set with @ref JxlDecoderSetCms,
 * and the CMS supports output to the desired color encoding or ICC profile,
 * then it will provide the output in that color encoding or ICC profile. If the
 * desired color encoding or the ICC is not supported, then an error will be
 * returned.
 *
 * If no CMS has been set with @ref JxlDecoderSetCms, there are two cases:
 *
 * (1) Calling this function with a color encoding will convert XYB images to
 * the desired color encoding. In this case, if the requested color encoding has
 * a narrower gamut, or the white points differ, then the resulting image can
 * have significant color distortion. Non-XYB images will not be converted to
 * the desired color space.
 *
 * (2) Calling this function with an ICC profile will result in an error.
 *
 * If called with an ICC profile (after a call to @ref JxlDecoderSetCms), the
 * ICC profile has to be a valid RGB or grayscale color profile.
 *
 * Can only be set after the ::JXL_DEC_COLOR_ENCODING event occurred and
 * before any other event occurred, and should be used before getting
 * ::JXL_COLOR_PROFILE_TARGET_DATA.
 *
 * This function must not be called before @ref JxlDecoderSetCms.
 *
 * @param dec decoder object
 * @param color_encoding the output color encoding
 * @param icc_data bytes of the icc profile
 * @param icc_size size of the icc profile in bytes
 * @return ::JXL_DEC_SUCCESS if the color profile was set successfully, @ref
 *     JXL_DEC_ERROR otherwise.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderSetOutputColorProfile(
    JxlDecoder* dec, const JxlColorEncoding* color_encoding,
    const uint8_t* icc_data, size_t icc_size);

/**
 * Sets the color management system (CMS) that will be used for color
 * conversion (if applicable) during decoding. May only be set before starting
 * decoding and must not be called after @ref JxlDecoderSetOutputColorProfile.
 *
 * See @ref JxlDecoderSetOutputColorProfile for how color conversions are done
 * depending on whether or not a CMS has been set with @ref JxlDecoderSetCms.
 *
 * @param dec decoder object.
 * @param cms structure representing a CMS implementation. See @ref
 * JxlCmsInterface for more details.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderSetCms(JxlDecoder* dec,
                                             JxlCmsInterface cms);
// TODO(firsching): add a function JxlDecoderSetDefaultCms() for setting a
// default in case libjxl is build with a CMS.

/**
 * Returns the minimum size in bytes of the preview image output pixel buffer
 * for the given format. This is the buffer for @ref
 * JxlDecoderSetPreviewOutBuffer. Requires the preview header information is
 * available in the decoder.
 *
 * @param dec decoder object
 * @param format format of pixels
 * @param size output value, buffer size in bytes
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR on error, such as
 *     information not available yet.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderPreviewOutBufferSize(
    const JxlDecoder* dec, const JxlPixelFormat* format, size_t* size);

/**
 * Sets the buffer to write the low-resolution preview image
 * to. The size of the buffer must be at least as large as given by @ref
 * JxlDecoderPreviewOutBufferSize. The buffer follows the format described
 * by @ref JxlPixelFormat. The preview image dimensions are given by the
 * @ref JxlPreviewHeader. The buffer is owned by the caller.
 *
 * @param dec decoder object
 * @param format format of pixels. Object owned by user and its contents are
 *     copied internally.
 * @param buffer buffer type to output the pixel data to
 * @param size size of buffer in bytes
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR on error, such as
 *     size too small.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderSetPreviewOutBuffer(
    JxlDecoder* dec, const JxlPixelFormat* format, void* buffer, size_t size);

/**
 * Outputs the information from the frame, such as duration when have_animation.
 * This function can be called when ::JXL_DEC_FRAME occurred for the current
 * frame, even when have_animation in the JxlBasicInfo is JXL_FALSE.
 *
 * @param dec decoder object
 * @param header struct to copy the information into, or NULL to only check
 *     whether the information is available through the return value.
 * @return ::JXL_DEC_SUCCESS if the value is available, @ref
 *     JXL_DEC_NEED_MORE_INPUT if not yet available, ::JXL_DEC_ERROR in
 *     case of other error conditions.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderGetFrameHeader(const JxlDecoder* dec,
                                                     JxlFrameHeader* header);

/**
 * Outputs name for the current frame. The buffer for name must have at least
 * `name_length + 1` bytes allocated, gotten from the associated JxlFrameHeader.
 *
 * @param dec decoder object
 * @param name buffer to copy the name into
 * @param size size of the name buffer in bytes, including zero termination
 *    character, so this must be at least @ref JxlFrameHeader.name_length + 1.
 * @return ::JXL_DEC_SUCCESS if the value is available, @ref
 *     JXL_DEC_NEED_MORE_INPUT if not yet available, ::JXL_DEC_ERROR in
 *     case of other error conditions.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderGetFrameName(const JxlDecoder* dec,
                                                   char* name, size_t size);

/**
 * Outputs the blend information for the current frame for a specific extra
 * channel. This function can be called once the ::JXL_DEC_FRAME event occurred
 * for the current frame, even if the `have_animation` field in the @ref
 * JxlBasicInfo is @ref JXL_FALSE. This information is only useful if coalescing
 * is disabled; otherwise the decoder will have performed blending already.
 *
 * @param dec decoder object
 * @param index the index of the extra channel
 * @param blend_info struct to copy the information into
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR on error
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderGetExtraChannelBlendInfo(
    const JxlDecoder* dec, size_t index, JxlBlendInfo* blend_info);

/**
 * Returns the minimum size in bytes of the image output pixel buffer for the
 * given format. This is the buffer for @ref JxlDecoderSetImageOutBuffer.
 * Requires that the basic image information is available in the decoder in the
 * case of coalescing enabled (default). In case coalescing is disabled, this
 * can only be called after the ::JXL_DEC_FRAME event occurs. In that case,
 * it will return the size required to store the possibly cropped frame (which
 * can be larger or smaller than the image dimensions).
 *
 * @param dec decoder object
 * @param format format of the pixels.
 * @param size output value, buffer size in bytes
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR on error, such as
 *     information not available yet.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderImageOutBufferSize(
    const JxlDecoder* dec, const JxlPixelFormat* format, size_t* size);

/**
 * Sets the buffer to write the full resolution image to. This can be set when
 * the ::JXL_DEC_FRAME event occurs, must be set when the @ref
 * JXL_DEC_NEED_IMAGE_OUT_BUFFER event occurs, and applies only for the
 * current frame. The size of the buffer must be at least as large as given
 * by @ref JxlDecoderImageOutBufferSize. The buffer follows the format described
 * by @ref JxlPixelFormat. The buffer is owned by the caller.
 *
 * @param dec decoder object
 * @param format format of the pixels. Object owned by user and its contents
 *     are copied internally.
 * @param buffer buffer type to output the pixel data to
 * @param size size of buffer in bytes
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR on error, such as
 *     size too small.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderSetImageOutBuffer(
    JxlDecoder* dec, const JxlPixelFormat* format, void* buffer, size_t size);

/**
 * Function type for @ref JxlDecoderSetImageOutCallback.
 *
 * The callback may be called simultaneously by different threads when using a
 * threaded parallel runner, on different pixels.
 *
 * @param opaque optional user data, as given to @ref
 *     JxlDecoderSetImageOutCallback.
 * @param x horizontal position of leftmost pixel of the pixel data.
 * @param y vertical position of the pixel data.
 * @param num_pixels amount of pixels included in the pixel data, horizontally.
 *     This is not the same as xsize of the full image, it may be smaller.
 * @param pixels pixel data as a horizontal stripe, in the format passed to @ref
 *     JxlDecoderSetImageOutCallback. The memory is not owned by the user, and
 *     is only valid during the time the callback is running.
 */
typedef void (*JxlImageOutCallback)(void* opaque, size_t x, size_t y,
                                    size_t num_pixels, const void* pixels);

/**
 * Initialization callback for @ref JxlDecoderSetMultithreadedImageOutCallback.
 *
 * @param init_opaque optional user data, as given to @ref
 *     JxlDecoderSetMultithreadedImageOutCallback.
 * @param num_threads maximum number of threads that will call the @c run
 *     callback concurrently.
 * @param num_pixels_per_thread maximum number of pixels that will be passed in
 *     one call to @c run.
 * @return a pointer to data that will be passed to the @c run callback, or
 *     @c NULL if initialization failed.
 */
typedef void* (*JxlImageOutInitCallback)(void* init_opaque, size_t num_threads,
                                         size_t num_pixels_per_thread);

/**
 * Worker callback for @ref JxlDecoderSetMultithreadedImageOutCallback.
 *
 * @param run_opaque user data returned by the @c init callback.
 * @param thread_id number in `[0, num_threads)` identifying the thread of the
 *     current invocation of the callback.
 * @param x horizontal position of the first (leftmost) pixel of the pixel data.
 * @param y vertical position of the pixel data.
 * @param num_pixels number of pixels in the pixel data. May be less than the
 *     full @c xsize of the image, and will be at most equal to the @c
 *     num_pixels_per_thread that was passed to @c init.
 * @param pixels pixel data as a horizontal stripe, in the format passed to @ref
 *     JxlDecoderSetMultithreadedImageOutCallback. The data pointed to
 *     remains owned by the caller and is only guaranteed to outlive the current
 *     callback invocation.
 */
typedef void (*JxlImageOutRunCallback)(void* run_opaque, size_t thread_id,
                                       size_t x, size_t y, size_t num_pixels,
                                       const void* pixels);

/**
 * Destruction callback for @ref JxlDecoderSetMultithreadedImageOutCallback,
 * called after all invocations of the @c run callback to perform any
 * appropriate clean-up of the @c run_opaque data returned by @c init.
 *
 * @param run_opaque user data returned by the @c init callback.
 */
typedef void (*JxlImageOutDestroyCallback)(void* run_opaque);

/**
 * Sets pixel output callback. This is an alternative to @ref
 * JxlDecoderSetImageOutBuffer. This can be set when the ::JXL_DEC_FRAME
 * event occurs, must be set when the ::JXL_DEC_NEED_IMAGE_OUT_BUFFER event
 * occurs, and applies only for the current frame. Only one of @ref
 * JxlDecoderSetImageOutBuffer or @ref JxlDecoderSetImageOutCallback may be used
 * for the same frame, not both at the same time.
 *
 * The callback will be called multiple times, to receive the image
 * data in small chunks. The callback receives a horizontal stripe of pixel
 * data, `1` pixel high, xsize pixels wide, called a scanline. The xsize here is
 * not the same as the full image width, the scanline may be a partial section,
 * and xsize may differ between calls. The user can then process and/or copy the
 * partial scanline to an image buffer. The callback may be called
 * simultaneously by different threads when using a threaded parallel runner, on
 * different pixels.
 *
 * If @ref JxlDecoderFlushImage is not used, then each pixel will be visited
 * exactly once by the different callback calls, during processing with one or
 * more @ref JxlDecoderProcessInput calls. These pixels are decoded to full
 * detail, they are not part of a lower resolution or lower quality progressive
 * pass, but the final pass.
 *
 * If @ref JxlDecoderFlushImage is used, then in addition each pixel will be
 * visited zero or one times during the blocking @ref JxlDecoderFlushImage call.
 * Pixels visited as a result of @ref JxlDecoderFlushImage may represent a lower
 * resolution or lower quality intermediate progressive pass of the image. Any
 * visited pixel will be of a quality at least as good or better than previous
 * visits of this pixel. A pixel may be visited zero times if it cannot be
 * decoded yet or if it was already decoded to full precision (this behavior is
 * not guaranteed).
 *
 * @param dec decoder object
 * @param format format of the pixels. Object owned by user; its contents are
 *     copied internally.
 * @param callback the callback function receiving partial scanlines of pixel
 *     data.
 * @param opaque optional user data, which will be passed on to the callback,
 *     may be NULL.
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR on error, such
 *     as @ref JxlDecoderSetImageOutBuffer already set.
 */
JXL_EXPORT JxlDecoderStatus
JxlDecoderSetImageOutCallback(JxlDecoder* dec, const JxlPixelFormat* format,
                              JxlImageOutCallback callback, void* opaque);

/** Similar to @ref JxlDecoderSetImageOutCallback except that the callback is
 * allowed an initialization phase during which it is informed of how many
 * threads will call it concurrently, and those calls are further informed of
 * which thread they are occurring in.
 *
 * @param dec decoder object
 * @param format format of the pixels. Object owned by user; its contents are
 *     copied internally.
 * @param init_callback initialization callback.
 * @param run_callback the callback function receiving partial scanlines of
 *     pixel data.
 * @param destroy_callback clean-up callback invoked after all calls to @c
 *     run_callback. May be NULL if no clean-up is necessary.
 * @param init_opaque optional user data passed to @c init_callback, may be NULL
 *     (unlike the return value from @c init_callback which may only be NULL if
 *     initialization failed).
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR on error, such
 *     as @ref JxlDecoderSetImageOutBuffer having already been called.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderSetMultithreadedImageOutCallback(
    JxlDecoder* dec, const JxlPixelFormat* format,
    JxlImageOutInitCallback init_callback, JxlImageOutRunCallback run_callback,
    JxlImageOutDestroyCallback destroy_callback, void* init_opaque);

/**
 * Returns the minimum size in bytes of an extra channel pixel buffer for the
 * given format. This is the buffer for @ref JxlDecoderSetExtraChannelBuffer.
 * Requires the basic image information is available in the decoder.
 *
 * @param dec decoder object
 * @param format format of the pixels. The num_channels value is ignored and is
 *     always treated to be `1`.
 * @param size output value, buffer size in bytes
 * @param index which extra channel to get, matching the index used in @ref
 *     JxlDecoderGetExtraChannelInfo. Must be smaller than num_extra_channels in
 *     the associated @ref JxlBasicInfo.
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR on error, such as
 *     information not available yet or invalid index.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderExtraChannelBufferSize(
    const JxlDecoder* dec, const JxlPixelFormat* format, size_t* size,
    uint32_t index);

/**
 * Sets the buffer to write an extra channel to. This can be set when
 * the ::JXL_DEC_FRAME or ::JXL_DEC_NEED_IMAGE_OUT_BUFFER event occurs,
 * and applies only for the current frame. The size of the buffer must be at
 * least as large as given by @ref JxlDecoderExtraChannelBufferSize. The buffer
 * follows the format described by @ref JxlPixelFormat, but where num_channels
 * is `1`. The buffer is owned by the caller. The amount of extra channels is
 * given by the num_extra_channels field in the associated @ref JxlBasicInfo,
 * and the information of individual extra channels can be queried with @ref
 * JxlDecoderGetExtraChannelInfo. To get multiple extra channels, this function
 * must be called multiple times, once for each wanted index. Not all images
 * have extra channels. The alpha channel is an extra channel and can be gotten
 * as part of the color channels when using an RGBA pixel buffer with @ref
 * JxlDecoderSetImageOutBuffer, but additionally also can be gotten
 * separately as extra channel. The color channels themselves cannot be gotten
 * this way.
 *
 *
 * @param dec decoder object
 * @param format format of the pixels. Object owned by user and its contents
 *     are copied internally. The num_channels value is ignored and is always
 *     treated to be `1`.
 * @param buffer buffer type to output the pixel data to
 * @param size size of buffer in bytes
 * @param index which extra channel to get, matching the index used in @ref
 *     JxlDecoderGetExtraChannelInfo. Must be smaller than num_extra_channels in
 *     the associated @ref JxlBasicInfo.
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR on error, such as
 *     size too small or invalid index.
 */
JXL_EXPORT JxlDecoderStatus
JxlDecoderSetExtraChannelBuffer(JxlDecoder* dec, const JxlPixelFormat* format,
                                void* buffer, size_t size, uint32_t index);

/**
 * Sets output buffer for reconstructed JPEG codestream.
 *
 * The data is owned by the caller and may be used by the decoder until @ref
 * JxlDecoderReleaseJPEGBuffer is called or the decoder is destroyed or
 * reset so must be kept alive until then.
 *
 * If a JPEG buffer was set before and released with @ref
 * JxlDecoderReleaseJPEGBuffer, bytes that the decoder has already output
 * should not be included, only the remaining bytes output must be set.
 *
 * @param dec decoder object
 * @param data pointer to next bytes to write to
 * @param size amount of bytes available starting from data
 * @return ::JXL_DEC_ERROR if output buffer was already set and @ref
 *     JxlDecoderReleaseJPEGBuffer was not called on it, ::JXL_DEC_SUCCESS
 *     otherwise
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderSetJPEGBuffer(JxlDecoder* dec,
                                                    uint8_t* data, size_t size);

/**
 * Releases buffer which was provided with @ref JxlDecoderSetJPEGBuffer.
 *
 * Calling @ref JxlDecoderReleaseJPEGBuffer is required whenever
 * a buffer is already set and a new buffer needs to be added with @ref
 * JxlDecoderSetJPEGBuffer, but is not required before @ref
 * JxlDecoderDestroy or @ref JxlDecoderReset.
 *
 * Calling @ref JxlDecoderReleaseJPEGBuffer when no buffer is set is
 * not an error and returns `0`.
 *
 * @param dec decoder object
 * @return the amount of bytes the decoder has not yet written to of the data
 *     set by @ref JxlDecoderSetJPEGBuffer, or `0` if no buffer is set or @ref
 *     JxlDecoderReleaseJPEGBuffer was already called.
 */
JXL_EXPORT size_t JxlDecoderReleaseJPEGBuffer(JxlDecoder* dec);

/**
 * Sets output buffer for box output codestream.
 *
 * The data is owned by the caller and may be used by the decoder until @ref
 * JxlDecoderReleaseBoxBuffer is called or the decoder is destroyed or
 * reset so must be kept alive until then.
 *
 * If for the current box a box buffer was set before and released with @ref
 * JxlDecoderReleaseBoxBuffer, bytes that the decoder has already output
 * should not be included, only the remaining bytes output must be set.
 *
 * The @ref JxlDecoderReleaseBoxBuffer must be used at the next ::JXL_DEC_BOX
 * event or final ::JXL_DEC_SUCCESS event to compute the size of the output
 * box bytes.
 *
 * @param dec decoder object
 * @param data pointer to next bytes to write to
 * @param size amount of bytes available starting from data
 * @return ::JXL_DEC_ERROR if output buffer was already set and @ref
 *     JxlDecoderReleaseBoxBuffer was not called on it, ::JXL_DEC_SUCCESS
 *     otherwise
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderSetBoxBuffer(JxlDecoder* dec,
                                                   uint8_t* data, size_t size);

/**
 * Releases buffer which was provided with @ref JxlDecoderSetBoxBuffer.
 *
 * Calling @ref JxlDecoderReleaseBoxBuffer is required whenever
 * a buffer is already set and a new buffer needs to be added with @ref
 * JxlDecoderSetBoxBuffer, but is not required before @ref
 * JxlDecoderDestroy or @ref JxlDecoderReset.
 *
 * Calling @ref JxlDecoderReleaseBoxBuffer when no buffer is set is
 * not an error and returns `0`.
 *
 * @param dec decoder object
 * @return the amount of bytes the decoder has not yet written to of the data
 *     set by @ref JxlDecoderSetBoxBuffer, or `0` if no buffer is set or @ref
 *     JxlDecoderReleaseBoxBuffer was already called.
 */
JXL_EXPORT size_t JxlDecoderReleaseBoxBuffer(JxlDecoder* dec);

/**
 * Configures whether to get boxes in raw mode or in decompressed mode. In raw
 * mode, boxes are output as their bytes appear in the container file, which may
 * be decompressed, or compressed if their type is "brob". In decompressed mode,
 * "brob" boxes are decompressed with Brotli before outputting them. The size of
 * the decompressed stream is not known before the decompression has already
 * finished.
 *
 * The default mode is raw. This setting can only be changed before decoding, or
 * directly after a ::JXL_DEC_BOX event, and is remembered until the decoder
 * is reset or destroyed.
 *
 * Enabling decompressed mode requires Brotli support from the library.
 *
 * @param dec decoder object
 * @param decompress ::JXL_TRUE to transparently decompress, ::JXL_FALSE
 * to get boxes in raw mode.
 * @return ::JXL_DEC_ERROR if decompressed mode is set and Brotli is not
 *     available, ::JXL_DEC_SUCCESS otherwise.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderSetDecompressBoxes(JxlDecoder* dec,
                                                         JXL_BOOL decompress);

/**
 * Outputs the type of the current box, after a ::JXL_DEC_BOX event occurred,
 * as `4` characters without null termination character. In case of a compressed
 * "brob" box, this will return "brob" if the decompressed argument is
 * JXL_FALSE, or the underlying box type if the decompressed argument is
 * JXL_TRUE.
 *
 * The following box types are currently described in ISO/IEC 18181-2:
 *  - "Exif": a box with EXIF metadata.  Starts with a 4-byte tiff header offset
 *    (big-endian uint32) that indicates the start of the actual EXIF data
 *    (which starts with a tiff header). Usually the offset will be zero and the
 *    EXIF data starts immediately after the offset field. The Exif orientation
 *    should be ignored by applications; the JPEG XL codestream orientation
 *    takes precedence and libjxl will by default apply the correct orientation
 *    automatically (see @ref JxlDecoderSetKeepOrientation).
 *  - "xml ": a box with XML data, in particular XMP metadata.
 *  - "jumb": a JUMBF superbox (JPEG Universal Metadata Box Format, ISO/IEC
 *    19566-5).
 *  - "JXL ": mandatory signature box, must come first, `12` bytes long
 * including the box header
 *  - "ftyp": a second mandatory signature box, must come second, `20` bytes
 * long including the box header
 *  - "jxll": a JXL level box. This indicates if the codestream is level `5` or
 *    level `10` compatible. If not present, it is level `5`. Level `10` allows
 * more features such as very high image resolution and bit-depths above `16`
 * bits per channel. Added automatically by the encoder when
 *    @ref JxlEncoderSetCodestreamLevel is used
 *  - "jxlc": a box with the image codestream, in case the codestream is not
 *    split across multiple boxes. The codestream contains the JPEG XL image
 *    itself, including the basic info such as image dimensions, ICC color
 *    profile, and all the pixel data of all the image frames.
 *  - "jxlp": a codestream box in case it is split across multiple boxes.
 *    The contents are the same as in case of a jxlc box, when concatenated.
 *  - "brob": a Brotli-compressed box, which otherwise represents an existing
 *    type of box such as Exif or "xml ". When @ref JxlDecoderSetDecompressBoxes
 *    is set to JXL_TRUE, these boxes will be transparently decompressed by the
 *    decoder.
 *  - "jxli": frame index box, can list the keyframes in case of a JPEG XL
 *    animation allowing the decoder to jump to individual frames more
 *    efficiently.
 *  - "jbrd": JPEG reconstruction box, contains the information required to
 *    byte-for-byte losslessly reconstruct a JPEG-1 image. The JPEG DCT
 *    coefficients (pixel content) themselves as well as the ICC profile are
 *    encoded in the JXL codestream (jxlc or jxlp) itself. EXIF, XMP and JUMBF
 *    metadata is encoded in the corresponding boxes. The jbrd box itself
 *    contains information such as the remaining app markers of the JPEG-1 file
 *    and everything else required to fit the information together into the
 *    exact original JPEG file.
 *
 * Other application-specific boxes can exist. Their typename should not begin
 * with "jxl" or "JXL" or conflict with other existing typenames.
 *
 * The signature, jxl* and jbrd boxes are processed by the decoder and would
 * typically be ignored by applications. The typical way to use this function is
 * to check if an encountered box contains metadata that the application is
 * interested in (e.g. EXIF or XMP metadata), in order to conditionally set a
 * box buffer.
 *
 * @param dec decoder object
 * @param type buffer to copy the type into
 * @param decompressed which box type to get: JXL_FALSE to get the raw box type,
 *     which can be "brob", JXL_TRUE, get the underlying box type.
 * @return ::JXL_DEC_SUCCESS if the value is available, ::JXL_DEC_ERROR if
 *     not, for example the JPEG XL file does not use the container format.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderGetBoxType(JxlDecoder* dec,
                                                 JxlBoxType type,
                                                 JXL_BOOL decompressed);

/**
 * Returns the size of a box as it appears in the container file, after the @ref
 * JXL_DEC_BOX event. This includes all the box headers.
 *
 * @param dec decoder object
 * @param size raw size of the box in bytes
 * @return ::JXL_DEC_ERROR if no box size is available, ::JXL_DEC_SUCCESS
 *     otherwise.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderGetBoxSizeRaw(const JxlDecoder* dec,
                                                    uint64_t* size);

/**
 * Returns the size of the contents of a box, after the @ref
 * JXL_DEC_BOX event. This does not include any of the headers of the box. For
 * compressed "brob" boxes, this is the size of the compressed content. Even
 * when @ref JxlDecoderSetDecompressBoxes is enabled, the return value of
 * function does not change, and the decompressed size is not known before it
 * has already been decompressed and output.
 *
 * @param dec decoder object
 * @param size size of the payload of the box in bytes
 * @return @ref JXL_DEC_ERROR if no box size is available, @ref JXL_DEC_SUCCESS
 *     otherwise.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderGetBoxSizeContents(const JxlDecoder* dec,
                                                         uint64_t* size);

/**
 * Configures at which progressive steps in frame decoding these @ref
 * JXL_DEC_FRAME_PROGRESSION event occurs. The default value for the level
 * of detail if this function is never called is `kDC`.
 *
 * @param dec decoder object
 * @param detail at which level of detail to trigger @ref
 *     JXL_DEC_FRAME_PROGRESSION
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR on error, such as
 *     an invalid value for the progressive detail.
 */
JXL_EXPORT JxlDecoderStatus
JxlDecoderSetProgressiveDetail(JxlDecoder* dec, JxlProgressiveDetail detail);

/**
 * Returns the intended downsampling ratio for the progressive frame produced
 * by @ref JxlDecoderFlushImage after the latest ::JXL_DEC_FRAME_PROGRESSION
 * event.
 *
 * @param dec decoder object
 * @return The intended downsampling ratio, can be `1`, `2`, `4` or `8`.
 */
JXL_EXPORT size_t JxlDecoderGetIntendedDownsamplingRatio(JxlDecoder* dec);

/**
 * Outputs progressive step towards the decoded image so far when only partial
 * input was received. If the flush was successful, the buffer set with @ref
 * JxlDecoderSetImageOutBuffer will contain partial image data.
 *
 * Can be called when @ref JxlDecoderProcessInput returns @ref
 * JXL_DEC_NEED_MORE_INPUT, after the ::JXL_DEC_FRAME event already occurred
 * and before the ::JXL_DEC_FULL_IMAGE event occurred for a frame.
 *
 * @param dec decoder object
 * @return ::JXL_DEC_SUCCESS if image data was flushed to the output buffer,
 *     or ::JXL_DEC_ERROR when no flush was done, e.g. if not enough image
 *     data was available yet even for flush, or no output buffer was set yet.
 *     This error is not fatal, it only indicates no flushed image is available
 *     right now. Regular decoding can still be performed.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderFlushImage(JxlDecoder* dec);

/**
 * Sets the bit depth of the output buffer or callback.
 *
 * Can be called after @ref JxlDecoderSetImageOutBuffer or @ref
 * JxlDecoderSetImageOutCallback. For float pixel data types, only the default
 * ::JXL_BIT_DEPTH_FROM_PIXEL_FORMAT setting is supported.
 *
 * @param dec decoder object
 * @param bit_depth the bit depth setting of the pixel output
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR on error, such as
 *     incompatible custom bit depth and pixel data type.
 */
JXL_EXPORT JxlDecoderStatus
JxlDecoderSetImageOutBitDepth(JxlDecoder* dec, const JxlBitDepth* bit_depth);

#ifdef __cplusplus
}
#endif

#endif /* JXL_DECODE_H_ */

/** @}*/

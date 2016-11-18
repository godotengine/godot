/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Copyright (c) 2008-2012 Gregory Maxwell
   Written by Jean-Marc Valin and Gregory Maxwell */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
  @file opus_custom.h
  @brief Opus-Custom reference implementation API
 */

#ifndef OPUS_CUSTOM_H
#define OPUS_CUSTOM_H

#include "opus_defines.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef CUSTOM_MODES
# define OPUS_CUSTOM_EXPORT OPUS_EXPORT
# define OPUS_CUSTOM_EXPORT_STATIC OPUS_EXPORT
#else
# define OPUS_CUSTOM_EXPORT
# ifdef OPUS_BUILD
#  define OPUS_CUSTOM_EXPORT_STATIC static OPUS_INLINE
# else
#  define OPUS_CUSTOM_EXPORT_STATIC
# endif
#endif

/** @defgroup opus_custom Opus Custom
  * @{
  *  Opus Custom is an optional part of the Opus specification and
  * reference implementation which uses a distinct API from the regular
  * API and supports frame sizes that are not normally supported.\ Use
  * of Opus Custom is discouraged for all but very special applications
  * for which a frame size different from 2.5, 5, 10, or 20 ms is needed
  * (for either complexity or latency reasons) and where interoperability
  * is less important.
  *
  * In addition to the interoperability limitations the use of Opus custom
  * disables a substantial chunk of the codec and generally lowers the
  * quality available at a given bitrate. Normally when an application needs
  * a different frame size from the codec it should buffer to match the
  * sizes but this adds a small amount of delay which may be important
  * in some very low latency applications. Some transports (especially
  * constant rate RF transports) may also work best with frames of
  * particular durations.
  *
  * Libopus only supports custom modes if they are enabled at compile time.
  *
  * The Opus Custom API is similar to the regular API but the
  * @ref opus_encoder_create and @ref opus_decoder_create calls take
  * an additional mode parameter which is a structure produced by
  * a call to @ref opus_custom_mode_create. Both the encoder and decoder
  * must create a mode using the same sample rate (fs) and frame size
  * (frame size) so these parameters must either be signaled out of band
  * or fixed in a particular implementation.
  *
  * Similar to regular Opus the custom modes support on the fly frame size
  * switching, but the sizes available depend on the particular frame size in
  * use. For some initial frame sizes on a single on the fly size is available.
  */

/** Contains the state of an encoder. One encoder state is needed
    for each stream. It is initialized once at the beginning of the
    stream. Do *not* re-initialize the state for every frame.
   @brief Encoder state
 */
typedef struct OpusCustomEncoder OpusCustomEncoder;

/** State of the decoder. One decoder state is needed for each stream.
    It is initialized once at the beginning of the stream. Do *not*
    re-initialize the state for every frame.
   @brief Decoder state
 */
typedef struct OpusCustomDecoder OpusCustomDecoder;

/** The mode contains all the information necessary to create an
    encoder. Both the encoder and decoder need to be initialized
    with exactly the same mode, otherwise the output will be
    corrupted.
   @brief Mode configuration
 */
typedef struct OpusCustomMode OpusCustomMode;

/** Creates a new mode struct. This will be passed to an encoder or
  * decoder. The mode MUST NOT BE DESTROYED until the encoders and
  * decoders that use it are destroyed as well.
  * @param [in] Fs <tt>int</tt>: Sampling rate (8000 to 96000 Hz)
  * @param [in] frame_size <tt>int</tt>: Number of samples (per channel) to encode in each
  *        packet (64 - 1024, prime factorization must contain zero or more 2s, 3s, or 5s and no other primes)
  * @param [out] error <tt>int*</tt>: Returned error code (if NULL, no error will be returned)
  * @return A newly created mode
  */
OPUS_CUSTOM_EXPORT OPUS_WARN_UNUSED_RESULT OpusCustomMode *opus_custom_mode_create(opus_int32 Fs, int frame_size, int *error);

/** Destroys a mode struct. Only call this after all encoders and
  * decoders using this mode are destroyed as well.
  * @param [in] mode <tt>OpusCustomMode*</tt>: Mode to be freed.
  */
OPUS_CUSTOM_EXPORT void opus_custom_mode_destroy(OpusCustomMode *mode);


#if !defined(OPUS_BUILD) || defined(CELT_ENCODER_C)

/* Encoder */
/** Gets the size of an OpusCustomEncoder structure.
  * @param [in] mode <tt>OpusCustomMode *</tt>: Mode configuration
  * @param [in] channels <tt>int</tt>: Number of channels
  * @returns size
  */
OPUS_CUSTOM_EXPORT_STATIC OPUS_WARN_UNUSED_RESULT int opus_custom_encoder_get_size(
    const OpusCustomMode *mode,
    int channels
) OPUS_ARG_NONNULL(1);

# ifdef CUSTOM_MODES
/** Initializes a previously allocated encoder state
  * The memory pointed to by st must be the size returned by opus_custom_encoder_get_size.
  * This is intended for applications which use their own allocator instead of malloc.
  * @see opus_custom_encoder_create(),opus_custom_encoder_get_size()
  * To reset a previously initialized state use the OPUS_RESET_STATE CTL.
  * @param [in] st <tt>OpusCustomEncoder*</tt>: Encoder state
  * @param [in] mode <tt>OpusCustomMode *</tt>: Contains all the information about the characteristics of
  *  the stream (must be the same characteristics as used for the
  *  decoder)
  * @param [in] channels <tt>int</tt>: Number of channels
  * @return OPUS_OK Success or @ref opus_errorcodes
  */
OPUS_CUSTOM_EXPORT int opus_custom_encoder_init(
    OpusCustomEncoder *st,
    const OpusCustomMode *mode,
    int channels
) OPUS_ARG_NONNULL(1) OPUS_ARG_NONNULL(2);
# endif
#endif


/** Creates a new encoder state. Each stream needs its own encoder
  * state (can't be shared across simultaneous streams).
  * @param [in] mode <tt>OpusCustomMode*</tt>: Contains all the information about the characteristics of
  *  the stream (must be the same characteristics as used for the
  *  decoder)
  * @param [in] channels <tt>int</tt>: Number of channels
  * @param [out] error <tt>int*</tt>: Returns an error code
  * @return Newly created encoder state.
*/
OPUS_CUSTOM_EXPORT OPUS_WARN_UNUSED_RESULT OpusCustomEncoder *opus_custom_encoder_create(
    const OpusCustomMode *mode,
    int channels,
    int *error
) OPUS_ARG_NONNULL(1);


/** Destroys a an encoder state.
  * @param[in] st <tt>OpusCustomEncoder*</tt>: State to be freed.
  */
OPUS_CUSTOM_EXPORT void opus_custom_encoder_destroy(OpusCustomEncoder *st);

/** Encodes a frame of audio.
  * @param [in] st <tt>OpusCustomEncoder*</tt>: Encoder state
  * @param [in] pcm <tt>float*</tt>: PCM audio in float format, with a normal range of +/-1.0.
  *          Samples with a range beyond +/-1.0 are supported but will
  *          be clipped by decoders using the integer API and should
  *          only be used if it is known that the far end supports
  *          extended dynamic range. There must be exactly
  *          frame_size samples per channel.
  * @param [in] frame_size <tt>int</tt>: Number of samples per frame of input signal
  * @param [out] compressed <tt>char *</tt>: The compressed data is written here. This may not alias pcm and must be at least maxCompressedBytes long.
  * @param [in] maxCompressedBytes <tt>int</tt>: Maximum number of bytes to use for compressing the frame
  *          (can change from one frame to another)
  * @return Number of bytes written to "compressed".
  *       If negative, an error has occurred (see error codes). It is IMPORTANT that
  *       the length returned be somehow transmitted to the decoder. Otherwise, no
  *       decoding is possible.
  */
OPUS_CUSTOM_EXPORT OPUS_WARN_UNUSED_RESULT int opus_custom_encode_float(
    OpusCustomEncoder *st,
    const float *pcm,
    int frame_size,
    unsigned char *compressed,
    int maxCompressedBytes
) OPUS_ARG_NONNULL(1) OPUS_ARG_NONNULL(2) OPUS_ARG_NONNULL(4);

/** Encodes a frame of audio.
  * @param [in] st <tt>OpusCustomEncoder*</tt>: Encoder state
  * @param [in] pcm <tt>opus_int16*</tt>: PCM audio in signed 16-bit format (native endian).
  *          There must be exactly frame_size samples per channel.
  * @param [in] frame_size <tt>int</tt>: Number of samples per frame of input signal
  * @param [out] compressed <tt>char *</tt>: The compressed data is written here. This may not alias pcm and must be at least maxCompressedBytes long.
  * @param [in] maxCompressedBytes <tt>int</tt>: Maximum number of bytes to use for compressing the frame
  *          (can change from one frame to another)
  * @return Number of bytes written to "compressed".
  *       If negative, an error has occurred (see error codes). It is IMPORTANT that
  *       the length returned be somehow transmitted to the decoder. Otherwise, no
  *       decoding is possible.
 */
OPUS_CUSTOM_EXPORT OPUS_WARN_UNUSED_RESULT int opus_custom_encode(
    OpusCustomEncoder *st,
    const opus_int16 *pcm,
    int frame_size,
    unsigned char *compressed,
    int maxCompressedBytes
) OPUS_ARG_NONNULL(1) OPUS_ARG_NONNULL(2) OPUS_ARG_NONNULL(4);

/** Perform a CTL function on an Opus custom encoder.
  *
  * Generally the request and subsequent arguments are generated
  * by a convenience macro.
  * @see opus_encoderctls
  */
OPUS_CUSTOM_EXPORT int opus_custom_encoder_ctl(OpusCustomEncoder * OPUS_RESTRICT st, int request, ...) OPUS_ARG_NONNULL(1);


#if !defined(OPUS_BUILD) || defined(CELT_DECODER_C)
/* Decoder */

/** Gets the size of an OpusCustomDecoder structure.
  * @param [in] mode <tt>OpusCustomMode *</tt>: Mode configuration
  * @param [in] channels <tt>int</tt>: Number of channels
  * @returns size
  */
OPUS_CUSTOM_EXPORT_STATIC OPUS_WARN_UNUSED_RESULT int opus_custom_decoder_get_size(
    const OpusCustomMode *mode,
    int channels
) OPUS_ARG_NONNULL(1);

/** Initializes a previously allocated decoder state
  * The memory pointed to by st must be the size returned by opus_custom_decoder_get_size.
  * This is intended for applications which use their own allocator instead of malloc.
  * @see opus_custom_decoder_create(),opus_custom_decoder_get_size()
  * To reset a previously initialized state use the OPUS_RESET_STATE CTL.
  * @param [in] st <tt>OpusCustomDecoder*</tt>: Decoder state
  * @param [in] mode <tt>OpusCustomMode *</tt>: Contains all the information about the characteristics of
  *  the stream (must be the same characteristics as used for the
  *  encoder)
  * @param [in] channels <tt>int</tt>: Number of channels
  * @return OPUS_OK Success or @ref opus_errorcodes
  */
OPUS_CUSTOM_EXPORT_STATIC int opus_custom_decoder_init(
    OpusCustomDecoder *st,
    const OpusCustomMode *mode,
    int channels
) OPUS_ARG_NONNULL(1) OPUS_ARG_NONNULL(2);

#endif


/** Creates a new decoder state. Each stream needs its own decoder state (can't
  * be shared across simultaneous streams).
  * @param [in] mode <tt>OpusCustomMode</tt>: Contains all the information about the characteristics of the
  *          stream (must be the same characteristics as used for the encoder)
  * @param [in] channels <tt>int</tt>: Number of channels
  * @param [out] error <tt>int*</tt>: Returns an error code
  * @return Newly created decoder state.
  */
OPUS_CUSTOM_EXPORT OPUS_WARN_UNUSED_RESULT OpusCustomDecoder *opus_custom_decoder_create(
    const OpusCustomMode *mode,
    int channels,
    int *error
) OPUS_ARG_NONNULL(1);

/** Destroys a an decoder state.
  * @param[in] st <tt>OpusCustomDecoder*</tt>: State to be freed.
  */
OPUS_CUSTOM_EXPORT void opus_custom_decoder_destroy(OpusCustomDecoder *st);

/** Decode an opus custom frame with floating point output
  * @param [in] st <tt>OpusCustomDecoder*</tt>: Decoder state
  * @param [in] data <tt>char*</tt>: Input payload. Use a NULL pointer to indicate packet loss
  * @param [in] len <tt>int</tt>: Number of bytes in payload
  * @param [out] pcm <tt>float*</tt>: Output signal (interleaved if 2 channels). length
  *  is frame_size*channels*sizeof(float)
  * @param [in] frame_size Number of samples per channel of available space in *pcm.
  * @returns Number of decoded samples or @ref opus_errorcodes
  */
OPUS_CUSTOM_EXPORT OPUS_WARN_UNUSED_RESULT int opus_custom_decode_float(
    OpusCustomDecoder *st,
    const unsigned char *data,
    int len,
    float *pcm,
    int frame_size
) OPUS_ARG_NONNULL(1) OPUS_ARG_NONNULL(4);

/** Decode an opus custom frame
  * @param [in] st <tt>OpusCustomDecoder*</tt>: Decoder state
  * @param [in] data <tt>char*</tt>: Input payload. Use a NULL pointer to indicate packet loss
  * @param [in] len <tt>int</tt>: Number of bytes in payload
  * @param [out] pcm <tt>opus_int16*</tt>: Output signal (interleaved if 2 channels). length
  *  is frame_size*channels*sizeof(opus_int16)
  * @param [in] frame_size Number of samples per channel of available space in *pcm.
  * @returns Number of decoded samples or @ref opus_errorcodes
  */
OPUS_CUSTOM_EXPORT OPUS_WARN_UNUSED_RESULT int opus_custom_decode(
    OpusCustomDecoder *st,
    const unsigned char *data,
    int len,
    opus_int16 *pcm,
    int frame_size
) OPUS_ARG_NONNULL(1) OPUS_ARG_NONNULL(4);

/** Perform a CTL function on an Opus custom decoder.
  *
  * Generally the request and subsequent arguments are generated
  * by a convenience macro.
  * @see opus_genericctls
  */
OPUS_CUSTOM_EXPORT int opus_custom_decoder_ctl(OpusCustomDecoder * OPUS_RESTRICT st, int request, ...) OPUS_ARG_NONNULL(1);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* OPUS_CUSTOM_H */

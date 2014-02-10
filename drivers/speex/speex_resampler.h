/* Copyright (C) 2007 Jean-Marc Valin
      
   File: speex_resampler.h
   Resampling code
      
   The design goals of this code are:
      - Very fast algorithm
      - Low memory requirement
      - Good *perceptual* quality (and not best SNR)

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

   1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
   IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
   OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef SPEEX_RESAMPLER_H
#define SPEEX_RESAMPLER_H

#ifdef OUTSIDE_SPEEX

/********* WARNING: MENTAL SANITY ENDS HERE *************/

/* If the resampler is defined outside of Speex, we change the symbol names so that 
   there won't be any clash if linking with Speex later on. */

/* #define RANDOM_PREFIX your software name here */
#ifndef RANDOM_PREFIX
#error "Please define RANDOM_PREFIX (above) to something specific to your project to prevent symbol name clashes"
#endif

#define CAT_PREFIX2(a,b) a ## b
#define CAT_PREFIX(a,b) CAT_PREFIX2(a, b)
      
#define speex_resampler_init CAT_PREFIX(RANDOM_PREFIX,_resampler_init)
#define speex_resampler_init_frac CAT_PREFIX(RANDOM_PREFIX,_resampler_init_frac)
#define speex_resampler_destroy CAT_PREFIX(RANDOM_PREFIX,_resampler_destroy)
#define speex_resampler_process_float CAT_PREFIX(RANDOM_PREFIX,_resampler_process_float)
#define speex_resampler_process_int CAT_PREFIX(RANDOM_PREFIX,_resampler_process_int)
#define speex_resampler_process_interleaved_float CAT_PREFIX(RANDOM_PREFIX,_resampler_process_interleaved_float)
#define speex_resampler_process_interleaved_int CAT_PREFIX(RANDOM_PREFIX,_resampler_process_interleaved_int)
#define speex_resampler_set_rate CAT_PREFIX(RANDOM_PREFIX,_resampler_set_rate)
#define speex_resampler_get_rate CAT_PREFIX(RANDOM_PREFIX,_resampler_get_rate)
#define speex_resampler_set_rate_frac CAT_PREFIX(RANDOM_PREFIX,_resampler_set_rate_frac)
#define speex_resampler_get_ratio CAT_PREFIX(RANDOM_PREFIX,_resampler_get_ratio)
#define speex_resampler_set_quality CAT_PREFIX(RANDOM_PREFIX,_resampler_set_quality)
#define speex_resampler_get_quality CAT_PREFIX(RANDOM_PREFIX,_resampler_get_quality)
#define speex_resampler_set_input_stride CAT_PREFIX(RANDOM_PREFIX,_resampler_set_input_stride)
#define speex_resampler_get_input_stride CAT_PREFIX(RANDOM_PREFIX,_resampler_get_input_stride)
#define speex_resampler_set_output_stride CAT_PREFIX(RANDOM_PREFIX,_resampler_set_output_stride)
#define speex_resampler_get_output_stride CAT_PREFIX(RANDOM_PREFIX,_resampler_get_output_stride)
#define speex_resampler_get_input_latency CAT_PREFIX(RANDOM_PREFIX,_resampler_get_input_latency)
#define speex_resampler_get_output_latency CAT_PREFIX(RANDOM_PREFIX,_resampler_get_output_latency)
#define speex_resampler_skip_zeros CAT_PREFIX(RANDOM_PREFIX,_resampler_skip_zeros)
#define speex_resampler_reset_mem CAT_PREFIX(RANDOM_PREFIX,_resampler_reset_mem)
#define speex_resampler_strerror CAT_PREFIX(RANDOM_PREFIX,_resampler_strerror)

#define spx_int16_t short
#define spx_int32_t int
#define spx_uint16_t unsigned short
#define spx_uint32_t unsigned int
      
#else /* OUTSIDE_SPEEX */

#include "speex/speex_types.h"

#endif /* OUTSIDE_SPEEX */

#ifdef __cplusplus
extern "C" {
#endif

#define SPEEX_RESAMPLER_QUALITY_MAX 10
#define SPEEX_RESAMPLER_QUALITY_MIN 0
#define SPEEX_RESAMPLER_QUALITY_DEFAULT 4
#define SPEEX_RESAMPLER_QUALITY_VOIP 3
#define SPEEX_RESAMPLER_QUALITY_DESKTOP 5

enum {
   RESAMPLER_ERR_SUCCESS         = 0,
   RESAMPLER_ERR_ALLOC_FAILED    = 1,
   RESAMPLER_ERR_BAD_STATE       = 2,
   RESAMPLER_ERR_INVALID_ARG     = 3,
   RESAMPLER_ERR_PTR_OVERLAP     = 4,
   
   RESAMPLER_ERR_MAX_ERROR
};

struct SpeexResamplerState_;
typedef struct SpeexResamplerState_ SpeexResamplerState;

/** Create a new resampler with integer input and output rates.
 * @param nb_channels Number of channels to be processed
 * @param in_rate Input sampling rate (integer number of Hz).
 * @param out_rate Output sampling rate (integer number of Hz).
 * @param quality Resampling quality between 0 and 10, where 0 has poor quality
 * and 10 has very high quality.
 * @return Newly created resampler state
 * @retval NULL Error: not enough memory
 */
SpeexResamplerState *speex_resampler_init(spx_uint32_t nb_channels, 
                                          spx_uint32_t in_rate, 
                                          spx_uint32_t out_rate, 
                                          int quality,
                                          int *err);

/** Create a new resampler with fractional input/output rates. The sampling 
 * rate ratio is an arbitrary rational number with both the numerator and 
 * denominator being 32-bit integers.
 * @param nb_channels Number of channels to be processed
 * @param ratio_num Numerator of the sampling rate ratio
 * @param ratio_den Denominator of the sampling rate ratio
 * @param in_rate Input sampling rate rounded to the nearest integer (in Hz).
 * @param out_rate Output sampling rate rounded to the nearest integer (in Hz).
 * @param quality Resampling quality between 0 and 10, where 0 has poor quality
 * and 10 has very high quality.
 * @return Newly created resampler state
 * @retval NULL Error: not enough memory
 */
SpeexResamplerState *speex_resampler_init_frac(spx_uint32_t nb_channels, 
                                               spx_uint32_t ratio_num, 
                                               spx_uint32_t ratio_den, 
                                               spx_uint32_t in_rate, 
                                               spx_uint32_t out_rate, 
                                               int quality,
                                               int *err);

/** Destroy a resampler state.
 * @param st Resampler state
 */
void speex_resampler_destroy(SpeexResamplerState *st);

/** Resample a float array. The input and output buffers must *not* overlap.
 * @param st Resampler state
 * @param channel_index Index of the channel to process for the multi-channel 
 * base (0 otherwise)
 * @param in Input buffer
 * @param in_len Number of input samples in the input buffer. Returns the 
 * number of samples processed
 * @param out Output buffer
 * @param out_len Size of the output buffer. Returns the number of samples written
 */
int speex_resampler_process_float(SpeexResamplerState *st, 
                                   spx_uint32_t channel_index, 
                                   const float *in, 
                                   spx_uint32_t *in_len, 
                                   float *out, 
                                   spx_uint32_t *out_len);

/** Resample an int array. The input and output buffers must *not* overlap.
 * @param st Resampler state
 * @param channel_index Index of the channel to process for the multi-channel 
 * base (0 otherwise)
 * @param in Input buffer
 * @param in_len Number of input samples in the input buffer. Returns the number
 * of samples processed
 * @param out Output buffer
 * @param out_len Size of the output buffer. Returns the number of samples written
 */
int speex_resampler_process_int(SpeexResamplerState *st, 
                                 spx_uint32_t channel_index, 
                                 const spx_int16_t *in, 
                                 spx_uint32_t *in_len, 
                                 spx_int16_t *out, 
                                 spx_uint32_t *out_len);

/** Resample an interleaved float array. The input and output buffers must *not* overlap.
 * @param st Resampler state
 * @param in Input buffer
 * @param in_len Number of input samples in the input buffer. Returns the number
 * of samples processed. This is all per-channel.
 * @param out Output buffer
 * @param out_len Size of the output buffer. Returns the number of samples written.
 * This is all per-channel.
 */
int speex_resampler_process_interleaved_float(SpeexResamplerState *st, 
                                               const float *in, 
                                               spx_uint32_t *in_len, 
                                               float *out, 
                                               spx_uint32_t *out_len);

/** Resample an interleaved int array. The input and output buffers must *not* overlap.
 * @param st Resampler state
 * @param in Input buffer
 * @param in_len Number of input samples in the input buffer. Returns the number
 * of samples processed. This is all per-channel.
 * @param out Output buffer
 * @param out_len Size of the output buffer. Returns the number of samples written.
 * This is all per-channel.
 */
int speex_resampler_process_interleaved_int(SpeexResamplerState *st, 
                                             const spx_int16_t *in, 
                                             spx_uint32_t *in_len, 
                                             spx_int16_t *out, 
                                             spx_uint32_t *out_len);

/** Set (change) the input/output sampling rates (integer value).
 * @param st Resampler state
 * @param in_rate Input sampling rate (integer number of Hz).
 * @param out_rate Output sampling rate (integer number of Hz).
 */
int speex_resampler_set_rate(SpeexResamplerState *st, 
                              spx_uint32_t in_rate, 
                              spx_uint32_t out_rate);

/** Get the current input/output sampling rates (integer value).
 * @param st Resampler state
 * @param in_rate Input sampling rate (integer number of Hz) copied.
 * @param out_rate Output sampling rate (integer number of Hz) copied.
 */
void speex_resampler_get_rate(SpeexResamplerState *st, 
                              spx_uint32_t *in_rate, 
                              spx_uint32_t *out_rate);

/** Set (change) the input/output sampling rates and resampling ratio 
 * (fractional values in Hz supported).
 * @param st Resampler state
 * @param ratio_num Numerator of the sampling rate ratio
 * @param ratio_den Denominator of the sampling rate ratio
 * @param in_rate Input sampling rate rounded to the nearest integer (in Hz).
 * @param out_rate Output sampling rate rounded to the nearest integer (in Hz).
 */
int speex_resampler_set_rate_frac(SpeexResamplerState *st, 
                                   spx_uint32_t ratio_num, 
                                   spx_uint32_t ratio_den, 
                                   spx_uint32_t in_rate, 
                                   spx_uint32_t out_rate);

/** Get the current resampling ratio. This will be reduced to the least
 * common denominator.
 * @param st Resampler state
 * @param ratio_num Numerator of the sampling rate ratio copied
 * @param ratio_den Denominator of the sampling rate ratio copied
 */
void speex_resampler_get_ratio(SpeexResamplerState *st, 
                               spx_uint32_t *ratio_num, 
                               spx_uint32_t *ratio_den);

/** Set (change) the conversion quality.
 * @param st Resampler state
 * @param quality Resampling quality between 0 and 10, where 0 has poor 
 * quality and 10 has very high quality.
 */
int speex_resampler_set_quality(SpeexResamplerState *st, 
                                 int quality);

/** Get the conversion quality.
 * @param st Resampler state
 * @param quality Resampling quality between 0 and 10, where 0 has poor 
 * quality and 10 has very high quality.
 */
void speex_resampler_get_quality(SpeexResamplerState *st, 
                                 int *quality);

/** Set (change) the input stride.
 * @param st Resampler state
 * @param stride Input stride
 */
void speex_resampler_set_input_stride(SpeexResamplerState *st, 
                                      spx_uint32_t stride);

/** Get the input stride.
 * @param st Resampler state
 * @param stride Input stride copied
 */
void speex_resampler_get_input_stride(SpeexResamplerState *st, 
                                      spx_uint32_t *stride);

/** Set (change) the output stride.
 * @param st Resampler state
 * @param stride Output stride
 */
void speex_resampler_set_output_stride(SpeexResamplerState *st, 
                                      spx_uint32_t stride);

/** Get the output stride.
 * @param st Resampler state copied
 * @param stride Output stride
 */
void speex_resampler_get_output_stride(SpeexResamplerState *st, 
                                      spx_uint32_t *stride);

/** Get the latency in input samples introduced by the resampler.
 * @param st Resampler state
 */
int speex_resampler_get_input_latency(SpeexResamplerState *st);

/** Get the latency in output samples introduced by the resampler.
 * @param st Resampler state
 */
int speex_resampler_get_output_latency(SpeexResamplerState *st);

/** Make sure that the first samples to go out of the resamplers don't have 
 * leading zeros. This is only useful before starting to use a newly created 
 * resampler. It is recommended to use that when resampling an audio file, as
 * it will generate a file with the same length. For real-time processing,
 * it is probably easier not to use this call (so that the output duration
 * is the same for the first frame).
 * @param st Resampler state
 */
int speex_resampler_skip_zeros(SpeexResamplerState *st);

/** Reset a resampler so a new (unrelated) stream can be processed.
 * @param st Resampler state
 */
int speex_resampler_reset_mem(SpeexResamplerState *st);

/** Returns the English meaning for an error code
 * @param err Error code
 * @return English string
 */
const char *speex_resampler_strerror(int err);

#ifdef __cplusplus
}
#endif

#endif

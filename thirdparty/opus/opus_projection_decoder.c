/* Copyright (c) 2017 Google Inc.
   Written by Andrew Allen */
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "mathops.h"
#include "os_support.h"
#include "opus_private.h"
#include "opus_defines.h"
#include "opus_projection.h"
#include "opus_multistream.h"
#include "mapping_matrix.h"
#include "stack_alloc.h"

struct OpusProjectionDecoder
{
  opus_int32 demixing_matrix_size_in_bytes;
  /* Encoder states go here */
};

#if !defined(DISABLE_FLOAT_API)
static void opus_projection_copy_channel_out_float(
  void *dst,
  int dst_stride,
  int dst_channel,
  const opus_val16 *src,
  int src_stride,
  int frame_size,
  void *user_data)
{
  float *float_dst;
  const MappingMatrix *matrix;
  float_dst = (float *)dst;
  matrix = (const MappingMatrix *)user_data;

  if (dst_channel == 0)
    OPUS_CLEAR(float_dst, frame_size * dst_stride);

  if (src != NULL)
    mapping_matrix_multiply_channel_out_float(matrix, src, dst_channel,
      src_stride, float_dst, dst_stride, frame_size);
}
#endif

static void opus_projection_copy_channel_out_short(
  void *dst,
  int dst_stride,
  int dst_channel,
  const opus_val16 *src,
  int src_stride,
  int frame_size,
  void *user_data)
{
  opus_int16 *short_dst;
  const MappingMatrix *matrix;
  short_dst = (opus_int16 *)dst;
  matrix = (const MappingMatrix *)user_data;
  if (dst_channel == 0)
    OPUS_CLEAR(short_dst, frame_size * dst_stride);

  if (src != NULL)
    mapping_matrix_multiply_channel_out_short(matrix, src, dst_channel,
      src_stride, short_dst, dst_stride, frame_size);
}

static MappingMatrix *get_dec_demixing_matrix(OpusProjectionDecoder *st)
{
  /* void* cast avoids clang -Wcast-align warning */
  return (MappingMatrix*)(void*)((char*)st +
    align(sizeof(OpusProjectionDecoder)));
}

static OpusMSDecoder *get_multistream_decoder(OpusProjectionDecoder *st)
{
  /* void* cast avoids clang -Wcast-align warning */
  return (OpusMSDecoder*)(void*)((char*)st +
    align(sizeof(OpusProjectionDecoder) +
    st->demixing_matrix_size_in_bytes));
}

opus_int32 opus_projection_decoder_get_size(int channels, int streams,
                                            int coupled_streams)
{
  opus_int32 matrix_size;
  opus_int32 decoder_size;

  matrix_size =
    mapping_matrix_get_size(streams + coupled_streams, channels);
  if (!matrix_size)
    return 0;

  decoder_size = opus_multistream_decoder_get_size(streams, coupled_streams);
  if (!decoder_size)
    return 0;

  return align(sizeof(OpusProjectionDecoder)) + matrix_size + decoder_size;
}

int opus_projection_decoder_init(OpusProjectionDecoder *st, opus_int32 Fs,
  int channels, int streams, int coupled_streams,
  unsigned char *demixing_matrix, opus_int32 demixing_matrix_size)
{
  int nb_input_streams;
  opus_int32 expected_matrix_size;
  int i, ret;
  unsigned char mapping[255];
  VARDECL(opus_int16, buf);
  ALLOC_STACK;

  /* Verify supplied matrix size. */
  nb_input_streams = streams + coupled_streams;
  expected_matrix_size = nb_input_streams * channels * sizeof(opus_int16);
  if (expected_matrix_size != demixing_matrix_size)
  {
    RESTORE_STACK;
    return OPUS_BAD_ARG;
  }

  /* Convert demixing matrix input into internal format. */
  ALLOC(buf, nb_input_streams * channels, opus_int16);
  for (i = 0; i < nb_input_streams * channels; i++)
  {
    int s = demixing_matrix[2*i + 1] << 8 | demixing_matrix[2*i];
    s = ((s & 0xFFFF) ^ 0x8000) - 0x8000;
    buf[i] = (opus_int16)s;
  }

  /* Assign demixing matrix. */
  st->demixing_matrix_size_in_bytes =
    mapping_matrix_get_size(channels, nb_input_streams);
  if (!st->demixing_matrix_size_in_bytes)
  {
    RESTORE_STACK;
    return OPUS_BAD_ARG;
  }

  mapping_matrix_init(get_dec_demixing_matrix(st), channels, nb_input_streams, 0,
    buf, demixing_matrix_size);

  /* Set trivial mapping so each input channel pairs with a matrix column. */
  for (i = 0; i < channels; i++)
    mapping[i] = i;

  ret = opus_multistream_decoder_init(
    get_multistream_decoder(st), Fs, channels, streams, coupled_streams, mapping);
  RESTORE_STACK;
  return ret;
}

OpusProjectionDecoder *opus_projection_decoder_create(
  opus_int32 Fs, int channels, int streams, int coupled_streams,
  unsigned char *demixing_matrix, opus_int32 demixing_matrix_size, int *error)
{
  int size;
  int ret;
  OpusProjectionDecoder *st;

  /* Allocate space for the projection decoder. */
  size = opus_projection_decoder_get_size(channels, streams, coupled_streams);
  if (!size) {
    if (error)
      *error = OPUS_ALLOC_FAIL;
    return NULL;
  }
  st = (OpusProjectionDecoder *)opus_alloc(size);
  if (!st)
  {
    if (error)
      *error = OPUS_ALLOC_FAIL;
    return NULL;
  }

  /* Initialize projection decoder with provided settings. */
  ret = opus_projection_decoder_init(st, Fs, channels, streams, coupled_streams,
                                     demixing_matrix, demixing_matrix_size);
  if (ret != OPUS_OK)
  {
    opus_free(st);
    st = NULL;
  }
  if (error)
    *error = ret;
  return st;
}

#ifdef FIXED_POINT
int opus_projection_decode(OpusProjectionDecoder *st, const unsigned char *data,
                           opus_int32 len, opus_int16 *pcm, int frame_size,
                           int decode_fec)
{
  return opus_multistream_decode_native(get_multistream_decoder(st), data, len,
    pcm, opus_projection_copy_channel_out_short, frame_size, decode_fec, 0,
    get_dec_demixing_matrix(st));
}
#else
int opus_projection_decode(OpusProjectionDecoder *st, const unsigned char *data,
                           opus_int32 len, opus_int16 *pcm, int frame_size,
                           int decode_fec)
{
  return opus_multistream_decode_native(get_multistream_decoder(st), data, len,
    pcm, opus_projection_copy_channel_out_short, frame_size, decode_fec, 1,
    get_dec_demixing_matrix(st));
}
#endif

#ifndef DISABLE_FLOAT_API
int opus_projection_decode_float(OpusProjectionDecoder *st, const unsigned char *data,
                                 opus_int32 len, float *pcm, int frame_size, int decode_fec)
{
  return opus_multistream_decode_native(get_multistream_decoder(st), data, len,
    pcm, opus_projection_copy_channel_out_float, frame_size, decode_fec, 0,
    get_dec_demixing_matrix(st));
}
#endif

int opus_projection_decoder_ctl(OpusProjectionDecoder *st, int request, ...)
{
  va_list ap;
  int ret = OPUS_OK;

  va_start(ap, request);
  ret = opus_multistream_decoder_ctl_va_list(get_multistream_decoder(st),
    request, ap);
  va_end(ap);
  return ret;
}

void opus_projection_decoder_destroy(OpusProjectionDecoder *st)
{
  opus_free(st);
}


/*
  Copyright (c) 2005-2009, The Musepack Development Team
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided
  with the distribution.

  * Neither the name of the The Musepack Development Team nor the
  names of its contributors may be used to endorse or promote
  products derived from this software without specific prior
  written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/// \file mpcdec.h
/// Top level include file for libmpcdec.
#ifndef _MPCDEC_H_
#define _MPCDEC_H_
#ifdef WIN32
#pragma once
#endif

#include "reader.h"
#include "streaminfo.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MPC_FRAME_LENGTH          = (36 * 32),              ///< Samples per mpc frame
    MPC_DECODER_BUFFER_LENGTH = (MPC_FRAME_LENGTH * 4), ///< Required buffer size for decoder
    MPC_DECODER_SYNTH_DELAY   = 481
};

typedef struct mpc_decoder_t mpc_decoder;
typedef struct mpc_demux_t mpc_demux;

typedef struct mpc_bits_reader_t {
	unsigned char * buff; /// pointer on current byte
	unsigned int count; /// unread bits in current byte
} mpc_bits_reader;

typedef struct mpc_frame_info_t {
	mpc_uint32_t samples;		/// number of samples in the frame (counting once for multiple channels)
	mpc_int32_t bits;			/// number of bits consumed by this frame (-1) if end of stream
	MPC_SAMPLE_FORMAT * buffer;	/// frame samples buffer (size = samples * channels * sizeof(MPC_SAMPLE_FORMAT))
	mpc_bool_t is_key_frame; 	/// 1 if this frame is a key frame (first in block) 0 else. Set by the demuxer.
} mpc_frame_info;

typedef struct mpc_chap_info_t {
	mpc_uint64_t sample;	/// sample where the chapter starts
	mpc_uint16_t gain;		/// replaygain chapter value
	mpc_uint16_t peak;		/// peak chapter loudness level
	mpc_uint_t tag_size;	/// size of the tag element (0 if no tag is present for this chapter)
	char * tag;				/// pointer to an APEv2 tag without the preamble
} mpc_chap_info;

/// Initializes mpc decoder with the supplied stream info parameters.
/// \param si streaminfo structure indicating format of source stream
/// \return pointer on the initialized decoder structure if successful, 0 if not
MPC_API mpc_decoder * mpc_decoder_init(mpc_streaminfo *si);

/// Releases input mpc decoder
MPC_API void mpc_decoder_exit(mpc_decoder *p_dec);

/**
 * Sets decoder sample scaling factor.  All decoded samples will be multiplied
 * by this factor. Useful for applying replay gain.
 * @param scale_factor multiplicative scaling factor
 */
MPC_API void mpc_decoder_scale_output(mpc_decoder *p_dec, double scale_factor);

MPC_API void mpc_decoder_decode_frame(mpc_decoder * d, mpc_bits_reader * r, mpc_frame_info * i);

// This is the gain reference used in old replaygain
#define MPC_OLD_GAIN_REF 64.82

/**
 * init demuxer
 * @param p_reader initialized mpc_reader pointer
 * @return an initialized mpc_demux pointer
 */
MPC_API mpc_demux * mpc_demux_init(mpc_reader * p_reader);
/// free demuxer
MPC_API void mpc_demux_exit(mpc_demux * d);
/**
 * Calls mpc_decoder_scale_output to set the scaling factor according to the
 * replay gain stream information and the supplied ouput level
 * @param d pointer to a musepack demuxer
 * @param level the desired ouput level (in db). Must be MPC_OLD_GAIN_REF (64.82 db) if you want to get the old replaygain behavior
 * @param use_gain set it to MPC_TRUE if you want to set the scaling factor according to the stream gain
 * @param use_title MPC_TRUE : uses the title gain, MPC_FALSE : uses the album gain
 * @param clip_prevention MPC_TRUE : uses cliping prevention
 */
MPC_API void mpc_set_replay_level(mpc_demux * d, float level, mpc_bool_t use_gain,
                          mpc_bool_t use_title, mpc_bool_t clip_prevention);
/// decode frame
MPC_API mpc_status mpc_demux_decode(mpc_demux * d, mpc_frame_info * i);
/// get streaminfo
MPC_API void mpc_demux_get_info(mpc_demux * d, mpc_streaminfo * i);
/// seeks to a given sample
MPC_API mpc_status mpc_demux_seek_sample(mpc_demux * d, mpc_uint64_t destsample);
/// seeks to a given second
MPC_API mpc_status mpc_demux_seek_second(mpc_demux * d, double seconds);

/// \return the current position in the stream (in bits) from the beginning of the file
MPC_API mpc_seek_t mpc_demux_pos(mpc_demux * d);

/// chapters : only for sv8 streams
/**
 * Gets the number of chapters in the stream
 * @param d pointer to a musepack demuxer
 * @return the number of chapters found in the stream
 */
MPC_API mpc_int_t mpc_demux_chap_nb(mpc_demux * d);
/**
 * Gets datas associated to a given chapter
 * The chapter tag is an APEv2 tag without the preamble
 * @param d pointer to a musepack demuxer
 * @param chap_nb chapter number you want datas (from 0 to mpc_demux_chap_nb(d) - 1)
 * @return the chapter information structure
 */
MPC_API mpc_chap_info const * mpc_demux_chap(mpc_demux * d, int chap_nb);

#ifdef __cplusplus
}
#endif
#endif

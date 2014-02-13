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
/// \file streaminfo.h
#ifndef _MPCDEC_STREAMINFO_H_
#define _MPCDEC_STREAMINFO_H_
#ifdef WIN32
#pragma once
#endif

#include "mpc_types.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef mpc_int32_t mpc_streaminfo_off_t;

/// \brief mpc stream properties structure
///
/// Structure containing all the properties of an mpc stream.  Populated
/// by the streaminfo_read function.
typedef struct mpc_streaminfo {
    /// @name Core mpc stream properties
    //@{
    mpc_uint32_t         sample_freq;        ///< Sample frequency of stream
    mpc_uint32_t         channels;           ///< Number of channels in stream
    mpc_uint32_t         stream_version;     ///< Streamversion of stream
    mpc_uint32_t         bitrate;            ///< Bitrate of stream file (in bps)
    double               average_bitrate;    ///< Average bitrate of stream (in bits/sec)
    mpc_uint32_t         max_band;           ///< Maximum band-index used in stream (0...31)
    mpc_uint32_t         ms;                 ///< Mid/side stereo (0: off, 1: on)
	mpc_uint32_t         fast_seek;          ///< True if stream supports fast-seeking (sv7)
	mpc_uint32_t         block_pwr;          ///< Number of frames in a block = 2^block_pwr (sv8)
    //@}

    /// @name Replaygain properties
    //@{
    mpc_uint16_t         gain_title;         ///< Replaygain title value
    mpc_uint16_t         gain_album;         ///< Replaygain album value
    mpc_uint16_t         peak_album;         ///< Peak album loudness level
    mpc_uint16_t         peak_title;         ///< Peak title loudness level
    //@}

    /// @name True gapless properties
    //@{
    mpc_uint32_t         is_true_gapless;    ///< True gapless? (0: no, 1: yes)
	mpc_uint64_t         samples;            ///< Number of samples in the stream
	mpc_uint64_t         beg_silence;        ///< Number of samples that must not be played at the beginning of the stream
    //@}

	/// @name Encoder informations
    //@{
    mpc_uint32_t         encoder_version;    ///< Version of encoder used
    char                 encoder[256];       ///< Encoder name
	mpc_bool_t           pns;                ///< pns used
	float                profile;            ///< Quality profile of stream
	const char*          profile_name;       ///< Name of profile used by stream
	//@}


	mpc_streaminfo_off_t header_position;    ///< Byte offset of position of header in stream
    mpc_streaminfo_off_t tag_offset;         ///< Offset to file tags
    mpc_streaminfo_off_t total_file_length;  ///< Total length of underlying file
} mpc_streaminfo;

/// Gets length of stream si, in seconds.
/// \return length of stream in seconds
MPC_API double mpc_streaminfo_get_length(mpc_streaminfo *si);

/// Returns length of stream si, in samples.
/// \return length of stream in samples
MPC_API mpc_int64_t mpc_streaminfo_get_length_samples(mpc_streaminfo *si);

#ifdef __cplusplus
}
#endif
#endif

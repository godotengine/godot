/**************************************************************************/
/*  movie_writer_ogv.h                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "servers/audio/audio_server.h"
#include "servers/movie_writer/movie_writer.h"

#include <theora/theoraenc.h>
#include <vorbis/codec.h>
#include <vorbis/vorbisenc.h>

class MovieWriterOGV : public MovieWriter {
	GDCLASS(MovieWriterOGV, MovieWriter)

	uint32_t mix_rate = 48000;
	AudioServer::SpeakerMode speaker_mode = AudioServer::SPEAKER_MODE_STEREO;
	String base_path;
	uint32_t frame_count = 0;
	uint32_t fps = 0;
	uint32_t audio_ch = 0;
	uint32_t audio_frames = 0;

	Ref<FileAccess> f;

	// Vorbis quality -0.1 to 1 (-0.1 yields smallest files but lowest fidelity; 1 yields highest fidelity but large files. '0.2' is a reasonable default).
	float audio_quality = 0.5;

	// Bitrate target for Theora video.
	int video_bitrate = 0;

	// Theora quality selector from 0 to 1.0 (0 yields smallest files but lowest video quality. 1.0 yields highest fidelity but large files).
	float video_quality = 0.75;

	// Video stream keyframe frequency (one every N frames).
	ogg_uint32_t keyframe_frequency = 64;

	// Sets the encoder speed level. Higher speed levels favor quicker encoding over better quality per bit. Depending on the encoding
	// mode, and the internal algorithms used, quality may actually improve with higher speeds, but in this case bitrate will also
	// likely increase. The maximum value, and the meaning of each value, are implementation-specific and may change depending on the
	// current encoding mode.
	int speed = 4;

	// Take physical pages, weld into a logical stream of packets.
	ogg_stream_state to;

	// Take physical pages, weld into a logical stream of packets.
	ogg_stream_state vo;

	// Theora encoding context.
	th_enc_ctx *td;

	// Theora bitstream information.
	th_info ti;

	// Theora comment information.
	th_comment tc;

	// Vorbis bitstream information.
	vorbis_info vi;

	// Vorbis comment information.
	vorbis_comment vc;

	// Central working state for the packet->PCM decoder.
	vorbis_dsp_state vd;

	// Local working space for packet->PCM decode.
	vorbis_block vb;

	// Video buffer.
	uint8_t *y, *u, *v;
	th_ycbcr_buffer ycbcr;

	bool audio_flag = false;
	bool video_flag = false;
	ogg_page audio_page;
	ogg_page video_page;
	ogg_page backup_page;
	unsigned int backup_page_size = 0;
	unsigned char *backup_page_data = nullptr;

	void write_to_file(bool p_finish = false);
	void push_audio(const int32_t *p_audio_data);
	void push_video(const Ref<Image> &p_image);
	void pull_audio(bool p_last = false);
	void pull_video(bool p_last = false);
	void save_page(ogg_page page);
	void restore_page(ogg_page *page);

	inline int ilog(unsigned _v) {
		int ret;
		for (ret = 0; _v; ret++) {
			_v >>= 1;
		}
		return ret;
	}

protected:
	virtual uint32_t get_audio_mix_rate() const override;
	virtual AudioServer::SpeakerMode get_audio_speaker_mode() const override;
	virtual void get_supported_extensions(List<String> *r_extensions) const override;

	virtual Error write_begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path) override;
	virtual Error write_frame(const Ref<Image> &p_image, const int32_t *p_audio_data) override;
	virtual void write_end() override;

	virtual bool handles_file(const String &p_path) const override;

public:
	MovieWriterOGV();
};

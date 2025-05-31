/**************************************************************************/
/*  movie_writer_ogv.cpp                                                  */
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

#include "movie_writer_ogv.h"
#include "core/config/project_settings.h"
#include "thirdparty/misc/rgb2yuv.h"

int MovieWriterOGV::encode_audio(const int32_t *p_audio_data) {
	ogg_packet op;
	if (ogg_stream_eos(&vo)) {
		return 0;
	}

	if (p_audio_data == nullptr) {
		// End of file. This can be done implicitly, but it's
		// easier to see here in an obvious fashion. Tell the
		// library we're at end of stream so that it can handle the
		// last frame and mark end of stream in the output properly.
		vorbis_analysis_wrote(&vd, 0);
	} else {
		// Read and process more audio.
		float **vorbis_buffer = vorbis_analysis_buffer(&vd, audio_frames);

		// Deinterleave samples.
		uint32_t count = 0;
		for (uint32_t i = 0; i < audio_frames; i++) {
			for (uint32_t j = 0; j < audio_ch; j++) {
				vorbis_buffer[j][i] = p_audio_data[count] / 2147483647.f;
				count++;
			}
		}

		vorbis_analysis_wrote(&vd, audio_frames);
	}

	while (vorbis_analysis_blockout(&vd, &vb) > 0) {
		// Analysis, assume we want to use bitrate management.
		vorbis_analysis(&vb, NULL);
		vorbis_bitrate_addblock(&vb);

		// Weld packets into the bitstream.
		while (vorbis_bitrate_flushpacket(&vd, &op) > 0) {
			ogg_stream_packetin(&vo, &op);
		}
	}

	if (ogg_stream_pageout(&vo, &audio_page) > 0) {
		return 1;
	}

	return 0;
}

int MovieWriterOGV::encode_video(const Ref<Image> &p_image) {
	ogg_packet op;
	if (ogg_stream_eos(&to)) {
		return 0;
	}

	if (p_image != nullptr) {
		PackedByteArray data = p_image->get_data();
		rgb2yuv420(y, u, v, data.ptrw(), p_image->get_width(), p_image->get_height());

		// We submit the buffer using the size of the picture region.
		// libtheora will pad the picture region out to the full frame size for us,
		// whether we pass in a full frame or not.
		ycbcr[0].width = p_image->get_width();
		ycbcr[0].height = p_image->get_height();
		ycbcr[0].stride = p_image->get_width();
		ycbcr[0].data = y;
		ycbcr[1].width = p_image->get_width() / 2;
		ycbcr[1].height = p_image->get_height() / 2;
		ycbcr[1].stride = p_image->get_width() / 2;
		ycbcr[1].data = u;
		ycbcr[2].width = p_image->get_width() / 2;
		ycbcr[2].height = p_image->get_height() / 2;
		ycbcr[2].stride = p_image->get_width() / 2;
		ycbcr[2].data = v;
		th_encode_ycbcr_in(td, ycbcr);
	}

	int ret = 0;
	do {
		ret = th_encode_packetout(td, p_image == nullptr, &op);
		if (ret > 0) {
			ogg_stream_packetin(&to, &op);
		}
	} while (ret > 0);

	if (ogg_stream_pageout(&to, &video_page) > 0) {
		return 1;
	}

	return 0;
}

uint32_t MovieWriterOGV::get_audio_mix_rate() const {
	return mix_rate;
}

AudioServer::SpeakerMode MovieWriterOGV::get_audio_speaker_mode() const {
	return speaker_mode;
}

bool MovieWriterOGV::handles_file(const String &p_path) const {
	return p_path.get_extension().to_lower() == "ogv";
}

void MovieWriterOGV::get_supported_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("ogv");
}

Error MovieWriterOGV::write_begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path) {
	base_path = p_base_path.get_basename();
	if (base_path.is_relative_path()) {
		base_path = "res://" + base_path;
	}
	base_path += ".ogv";

	f = FileAccess::open(base_path, FileAccess::WRITE_READ);
	ERR_FAIL_COND_V(f.is_null(), ERR_CANT_OPEN);

	fps = p_fps;
	speed = 4;

	audio_ch = 2;
	switch (speaker_mode) {
		case AudioServer::SPEAKER_MODE_STEREO:
			audio_ch = 2;
			break;
		case AudioServer::SPEAKER_SURROUND_31:
			audio_ch = 4;
			break;
		case AudioServer::SPEAKER_SURROUND_51:
			audio_ch = 6;
			break;
		case AudioServer::SPEAKER_SURROUND_71:
			audio_ch = 8;
			break;
	}
	audio_frames = mix_rate / fps;

	// Set up Ogg output streams.
	srand(time(NULL));
	ogg_stream_init(&to, rand()); // Video.
	ogg_stream_init(&vo, rand()); // Audio.

	// Initialize Vorbis audio encoding.
	vorbis_info_init(&vi);
	int ret = vorbis_encode_init_vbr(&vi, audio_ch, mix_rate, audio_quality);
	ERR_FAIL_COND_V_MSG(ret, ERR_UNAVAILABLE, "The Ogg Vorbis encoder couldn't set up a mode according to the requested quality or bitrate.");

	vorbis_comment_init(&vc);
	vorbis_analysis_init(&vd, &vi);
	vorbis_block_init(&vd, &vb);

	// Set up Theora encoder.
	// Theora has a divisible-by-16 restriction for the encoded frame size
	// scale the picture size up to the nearest /16 and calculate offsets.
	int pic_w = p_movie_size.width;
	int pic_h = p_movie_size.height;
	int frame_w = (pic_w + 15) & ~0xF;
	int frame_h = (pic_h + 15) & ~0xF;
	// Force the offsets to be even so that chroma samples line up like we expect.
	int pic_x = (frame_w - pic_w) / 2 & ~1;
	int pic_y = (frame_h - pic_h) / 2 & ~1;

	y = (uint8_t *)memalloc(pic_w * pic_h);
	u = (uint8_t *)memalloc(pic_w * pic_h / 4);
	v = (uint8_t *)memalloc(pic_w * pic_h / 4);

	th_info_init(&ti);
	ti.frame_width = frame_w;
	ti.frame_height = frame_h;
	ti.pic_width = pic_w;
	ti.pic_height = pic_h;
	ti.pic_x = pic_x;
	ti.pic_y = pic_y;
	ti.fps_numerator = fps;
	ti.fps_denominator = 1;
	ti.aspect_numerator = 1;
	ti.aspect_denominator = 1;
	ti.colorspace = TH_CS_UNSPECIFIED;
	// Account for the Ogg page overhead.
	// This is 1 byte per 255 for lacing values, plus 26 bytes per 4096 bytes for
	// the page header, plus approximately 1/2 byte per packet (not accounted for here).
	ti.target_bitrate = (int)(64870 * (ogg_int64_t)video_bitrate >> 16);
	ti.quality = video_quality * 63;
	ti.pixel_fmt = TH_PF_420;
	td = th_encode_alloc(&ti);
	th_info_clear(&ti);
	ERR_FAIL_COND_V_MSG(td == NULL, ERR_UNCONFIGURED, "Couldn't create a Theora encoder instance. Check that the video parameters are valid.");

	// Setting just the granule shift only allows power-of-two keyframe spacing.
	// Set the actual requested spacing.
	ret = th_encode_ctl(td, TH_ENCCTL_SET_KEYFRAME_FREQUENCY_FORCE, &keyframe_frequency, sizeof(keyframe_frequency - 1));
	if (ret < 0) {
		ERR_PRINT("Couldn't set keyframe interval.");
	}

	// Speed should also be set after the current encoder mode is established,
	// since the available speed levels may change depending on the encoder mode.
	if (speed >= 0) {
		int speed_max;
		int ret;
		ret = th_encode_ctl(td, TH_ENCCTL_GET_SPLEVEL_MAX, &speed_max, sizeof(speed_max));
		if (ret < 0) {
			WARN_PRINT("Couldn't determine maximum speed level.");
			speed_max = 0;
		}
		ret = th_encode_ctl(td, TH_ENCCTL_SET_SPLEVEL, &speed, sizeof(speed));
		if (ret < 0) {
			if (ret < 0) {
				WARN_PRINT(vformat("Couldn't set speed level to %d of %d.", speed, speed_max));
			}
			if (speed > speed_max) {
				WARN_PRINT(vformat("Setting speed level to %d instead.", speed_max));
			}
			ret = th_encode_ctl(td, TH_ENCCTL_SET_SPLEVEL, &speed_max, sizeof(speed_max));
			if (ret < 0) {
				WARN_PRINT(vformat("Couldn't set speed level to %d of %d.", speed_max, speed_max));
			}
		}
	}

	// Write the bitstream header packets with proper page interleave.
	th_comment_init(&tc);
	// The first packet will get its own page automatically.
	ogg_packet op;
	if (th_encode_flushheader(td, &tc, &op) <= 0) {
		ERR_FAIL_V_MSG(ERR_UNCONFIGURED, "Internal Theora library error.");
	}

	ogg_stream_packetin(&to, &op);
	if (ogg_stream_pageout(&to, &video_page) != 1) {
		ERR_FAIL_V_MSG(ERR_UNCONFIGURED, "Internal Ogg library error.");
	}
	f->store_buffer(video_page.header, video_page.header_len);
	f->store_buffer(video_page.body, video_page.body_len);

	// Create the remaining Theora headers.
	while (true) {
		ret = th_encode_flushheader(td, &tc, &op);
		if (ret < 0) {
			ERR_FAIL_V_MSG(ERR_UNCONFIGURED, "Internal Theora library error.");
		} else if (ret == 0) {
			break;
		}
		ogg_stream_packetin(&to, &op);
	}

	// Vorbis streams start with 3 standard header packets.
	ogg_packet id;
	ogg_packet comment;
	ogg_packet code;
	if (vorbis_analysis_headerout(&vd, &vc, &id, &comment, &code) < 0) {
		ERR_FAIL_V_MSG(ERR_UNCONFIGURED, "Internal Vorbis library error.");
	}

	// ID header is automatically placed in its own page.
	ogg_stream_packetin(&vo, &id);
	if (ogg_stream_pageout(&vo, &audio_page) != 1) {
		ERR_FAIL_V_MSG(ERR_UNCONFIGURED, "Internal Ogg library error.");
	}
	f->store_buffer(audio_page.header, audio_page.header_len);
	f->store_buffer(audio_page.body, audio_page.body_len);

	// Append remaining Vorbis header packets.
	ogg_stream_packetin(&vo, &comment);
	ogg_stream_packetin(&vo, &code);

	// Flush the rest of our headers. This ensures the actual data in each stream will start on a new page, as per spec.
	while (true) {
		ret = ogg_stream_flush(&to, &video_page);
		if (ret < 0) {
			ERR_FAIL_V_MSG(ERR_UNCONFIGURED, "Internal Ogg library error.");
		} else if (ret == 0) {
			break;
		}
		f->store_buffer(video_page.header, video_page.header_len);
		f->store_buffer(video_page.body, video_page.body_len);
	}

	while (true) {
		ret = ogg_stream_flush(&vo, &audio_page);
		if (ret < 0) {
			ERR_FAIL_V_MSG(ERR_UNCONFIGURED, "Internal Ogg library error.");
		} else if (ret == 0) {
			break;
		}
		f->store_buffer(audio_page.header, audio_page.header_len);
		f->store_buffer(audio_page.body, audio_page.body_len);
	}

	return OK;
}

Error MovieWriterOGV::write_frame(const Ref<Image> &p_image, const int32_t *p_audio_data) {
	ERR_FAIL_COND_V(!f.is_valid() || td == NULL, ERR_UNCONFIGURED);

	int audio_or_video = -1;

	// Is there an audio page flushed? If not, fetch one if possible.
	int audioflag = encode_audio(p_audio_data);

	// Is there a video page flushed? If not, fetch one if possible.
	int videoflag = encode_video(p_image);

	// No pages of either? Must be end of stream.
	if (!audioflag && !videoflag) {
		return OK;
	}

	// Which is earlier: the end of the audio page or the end of the video page?
	// Flush the one that is earlier to the stream.
	double audiotime = audioflag ? vorbis_granule_time(&vd, ogg_page_granulepos(&audio_page)) : -1;
	double videotime = videoflag ? th_granule_time(td, ogg_page_granulepos(&video_page)) : -1;
	if (!audioflag) {
		audio_or_video = 1;
	} else if (!videoflag) {
		audio_or_video = 0;
	} else {
		if (audiotime < videotime) {
			audio_or_video = 0;
		} else {
			audio_or_video = 1;
		}
	}

	if (audio_or_video == 1) {
		// Flush a video page.
		f->store_buffer(video_page.header, video_page.header_len);
		f->store_buffer(video_page.body, video_page.body_len);
	} else {
		// Flush an audio page.
		f->store_buffer(audio_page.header, audio_page.header_len);
		f->store_buffer(audio_page.body, audio_page.body_len);
	}

	frame_count++;

	return OK;
}

void MovieWriterOGV::write_end() {
	write_frame(nullptr, nullptr);

	th_encode_free(td);

	ogg_stream_clear(&vo);
	vorbis_block_clear(&vb);
	vorbis_dsp_clear(&vd);
	vorbis_comment_clear(&vc);
	vorbis_info_clear(&vi);

	ogg_stream_clear(&to);
	th_comment_clear(&tc);

	memfree(y);
	memfree(u);
	memfree(v);

	if (f.is_valid()) {
		f.unref();
	}
}

MovieWriterOGV::MovieWriterOGV() {
	mix_rate = GLOBAL_GET("editor/movie_writer/mix_rate");
	speaker_mode = AudioServer::SpeakerMode(int(GLOBAL_GET("editor/movie_writer/speaker_mode")));
	video_quality = GLOBAL_GET("editor/movie_writer/video_quality");
	audio_quality = GLOBAL_GET("editor/movie_writer/audio_quality");
}

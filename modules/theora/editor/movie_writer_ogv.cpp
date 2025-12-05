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
#include "core/io/file_access.h"
#include "rgb2yuv.h"

void MovieWriterOGV::push_audio(const int32_t *p_audio_data) {
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

void MovieWriterOGV::pull_audio(bool p_last) {
	ogg_packet op;

	while (vorbis_analysis_blockout(&vd, &vb) > 0) {
		// Analysis, assume we want to use bitrate management.
		vorbis_analysis(&vb, nullptr);
		vorbis_bitrate_addblock(&vb);

		// Weld packets into the bitstream.
		while (vorbis_bitrate_flushpacket(&vd, &op) > 0) {
			ogg_stream_packetin(&vo, &op);
		}
	}

	if (p_last) {
		vorbis_analysis_wrote(&vd, 0);
		pull_audio();
	}
}

void MovieWriterOGV::push_video(const Ref<Image> &p_image) {
	PackedByteArray data = p_image->get_data();
	if (p_image->get_format() == Image::FORMAT_RGBA8) {
		rgba2yuv420(y, u, v, data.ptrw(), p_image->get_width(), p_image->get_height());
	} else {
		rgb2yuv420(y, u, v, data.ptrw(), p_image->get_width(), p_image->get_height());
	}
	th_encode_ycbcr_in(td, ycbcr);
}

void MovieWriterOGV::pull_video(bool p_last) {
	ogg_packet op;

	int ret = 0;
	do {
		ret = th_encode_packetout(td, p_last, &op);
		if (ret > 0) {
			ogg_stream_packetin(&to, &op);
		}
	} while (ret > 0);
}

uint32_t MovieWriterOGV::get_audio_mix_rate() const {
	return mix_rate;
}

AudioServer::SpeakerMode MovieWriterOGV::get_audio_speaker_mode() const {
	return speaker_mode;
}

bool MovieWriterOGV::handles_file(const String &p_path) const {
	return p_path.has_extension("ogv");
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
	srand(time(nullptr));
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
	pic_w = p_movie_size.width;
	pic_h = p_movie_size.height;
	int frame_w = (pic_w + 15) & ~0xF;
	int frame_h = (pic_h + 15) & ~0xF;
	// Force the offsets to be even so that chroma samples line up like we expect.
	int pic_x = (frame_w - pic_w) / 2 & ~1;
	int pic_y = (frame_h - pic_h) / 2 & ~1;
	// Chroma size
	int c_w = (pic_w + 1) / 2;
	int c_h = (pic_h + 1) / 2;

	y = (uint8_t *)memalloc(pic_w * pic_h);
	u = (uint8_t *)memalloc(c_w * c_h);
	v = (uint8_t *)memalloc(c_w * c_h);

	// We submit the buffer using the size of the picture region.
	// libtheora will pad the picture region out to the full frame size for us,
	// whether we pass in a full frame or not.
	ycbcr[0].width = pic_w;
	ycbcr[0].height = pic_h;
	ycbcr[0].stride = pic_w;
	ycbcr[0].data = y;
	ycbcr[1].width = c_w;
	ycbcr[1].height = c_h;
	ycbcr[1].stride = c_w;
	ycbcr[1].data = u;
	ycbcr[2].width = c_w;
	ycbcr[2].height = c_h;
	ycbcr[2].stride = c_w;
	ycbcr[2].data = v;

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
	ERR_FAIL_NULL_V_MSG(td, ERR_UNCONFIGURED, "Couldn't create a Theora encoder instance. Check that the video parameters are valid.");

	// Setting just the granule shift only allows power-of-two keyframe spacing.
	// Set the actual requested spacing.
	ret = th_encode_ctl(td, TH_ENCCTL_SET_KEYFRAME_FREQUENCY_FORCE, &keyframe_frequency, sizeof(keyframe_frequency));
	if (ret < 0) {
		ERR_PRINT("Couldn't set keyframe interval.");
	}

	// Speed should also be set after the current encoder mode is established,
	// since the available speed levels may change depending on the encoder mode.
	if (speed >= 0) {
		int speed_max;
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

// The order of the operations has been chosen so we're one frame behind writing to the stream so we can put the eos
// mark in the last frame.
// Flushing streams to the file every X frames is done to improve audio/video page interleaving thus avoiding large runs
// of video or audio pages.
Error MovieWriterOGV::write_frame(const Ref<Image> &p_image, const int32_t *p_audio_data) {
	ERR_FAIL_COND_V(f.is_null() || td == nullptr, ERR_UNCONFIGURED);
	ERR_FAIL_COND_V_MSG(p_image->get_width() != pic_w || p_image->get_height() != pic_h, ERR_INVALID_PARAMETER, vformat("Video capture size has changed to: %dx%d", p_image->get_width(), p_image->get_height()));

	frame_count++;

	pull_audio();
	pull_video();

	if ((frame_count % 8) == 0) {
		write_to_file();
	}

	push_audio(p_audio_data);
	push_video(p_image);

	return OK;
}

void MovieWriterOGV::save_page(ogg_page page) {
	unsigned int page_size = page.header_len + page.body_len;
	if (page_size > backup_page_size) {
		backup_page_data = (unsigned char *)memrealloc(backup_page_data, page_size);
		backup_page_size = page_size;
	}
	backup_page.header = backup_page_data;
	backup_page.header_len = page.header_len;
	backup_page.body = backup_page_data + page.header_len;
	backup_page.body_len = page.body_len;
	memcpy(backup_page.header, page.header, page.header_len);
	memcpy(backup_page.body, page.body, page.body_len);
}

void MovieWriterOGV::restore_page(ogg_page *page) {
	page->header = backup_page.header;
	page->header_len = backup_page.header_len;
	page->body = backup_page.body;
	page->body_len = backup_page.body_len;
}

// The added complexity here is because we have to ensure pages are written in ascending timestamp order.
// libOgg doesn't allow checking the next page granulepos without requesting the page, and once requested it can't be
// returned, thus, we need to save it so that it doesn't get erased by the next `ogg_stream_packetin` call.
void MovieWriterOGV::write_to_file(bool p_finish) {
	if (audio_flag) {
		restore_page(&audio_page);
	} else {
		audio_flag = ogg_stream_flush(&vo, &audio_page);
	}
	if (video_flag) {
		restore_page(&video_page);
	} else {
		video_flag = ogg_stream_flush(&to, &video_page);
	}

	bool finishing = p_finish && (audio_flag || video_flag);
	while (finishing || (audio_flag && video_flag)) {
		double audiotime = vorbis_granule_time(&vd, ogg_page_granulepos(&audio_page));
		double videotime = th_granule_time(td, ogg_page_granulepos(&video_page));
		bool video_first = audiotime >= videotime;

		if (video_flag && video_first) {
			// Flush a video page.
			f->store_buffer(video_page.header, video_page.header_len);
			f->store_buffer(video_page.body, video_page.body_len);
			video_flag = ogg_stream_flush(&to, &video_page) > 0;
		} else {
			// Flush an audio page.
			f->store_buffer(audio_page.header, audio_page.header_len);
			f->store_buffer(audio_page.body, audio_page.body_len);
			audio_flag = ogg_stream_flush(&vo, &audio_page) > 0;
		}
		finishing = p_finish && (audio_flag || video_flag);
	}

	if (video_flag) {
		save_page(video_page);
	} else if (audio_flag) {
		save_page(audio_page);
	}
}

void MovieWriterOGV::write_end() {
	pull_audio(true);
	pull_video(true);
	write_to_file(true);

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

	if (backup_page_data != nullptr) {
		memfree(backup_page_data);
	}

	if (f.is_valid()) {
		f.unref();
	}
}

MovieWriterOGV::MovieWriterOGV() {
	mix_rate = GLOBAL_GET("editor/movie_writer/mix_rate");
	speaker_mode = AudioServer::SpeakerMode(int(GLOBAL_GET("editor/movie_writer/speaker_mode")));
	video_quality = GLOBAL_GET("editor/movie_writer/video_quality");
	audio_quality = GLOBAL_GET("editor/movie_writer/ogv/audio_quality");
	speed = GLOBAL_GET("editor/movie_writer/ogv/encoding_speed");
	keyframe_frequency = GLOBAL_GET("editor/movie_writer/ogv/keyframe_interval");
}

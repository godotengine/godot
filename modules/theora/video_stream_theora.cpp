/**************************************************************************/
/*  video_stream_theora.cpp                                               */
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

#include "video_stream_theora.h"

#include "core/config/project_settings.h"
#include "core/io/image.h"
#include "scene/resources/image_texture.h"

#include "thirdparty/misc/yuv2rgb.h"

int VideoStreamPlaybackTheora::buffer_data() {
	char *buffer = ogg_sync_buffer(&oy, 4096);

	uint64_t bytes = file->get_buffer((uint8_t *)buffer, 4096);
	ogg_sync_wrote(&oy, bytes);
	return bytes;
}

int VideoStreamPlaybackTheora::queue_page(ogg_page *page) {
	ogg_stream_pagein(&to, page);
	if (to.e_o_s) {
		theora_eos = true;
	}
	if (has_audio) {
		ogg_stream_pagein(&vo, page);
		if (vo.e_o_s) {
			vorbis_eos = true;
		}
	}
	return 0;
}

int VideoStreamPlaybackTheora::read_page(ogg_page *page) {
	int ret = 0;

	while (ret <= 0) {
		ret = ogg_sync_pageout(&oy, page);
		if (ret <= 0) {
			int bytes = buffer_data();
			if (bytes == 0) {
				return 0;
			}
		}
	}

	return ret;
}

double VideoStreamPlaybackTheora::get_page_time(ogg_page *page) {
	uint64_t granulepos = ogg_page_granulepos(page);
	int page_serialno = ogg_page_serialno(page);
	double page_time = -1;

	if (page_serialno == to.serialno) {
		page_time = th_granule_time(td, granulepos);
	}
	if (has_audio && page_serialno == vo.serialno) {
		page_time = vorbis_granule_time(&vd, granulepos);
	}

	return page_time;
}

// Read one buffer worth of pages and feed them to the streams.
int VideoStreamPlaybackTheora::feed_pages() {
	int pages = 0;
	ogg_page og;

	while (pages == 0) {
		while (ogg_sync_pageout(&oy, &og) > 0) {
			queue_page(&og);
			pages++;
		}
		if (pages == 0) {
			int bytes = buffer_data();
			if (bytes == 0) {
				break;
			}
		}
	}

	return pages;
}

// Seek the video and audio streams simultaneously to find the granulepos where we should start decoding.
// It will return the position where we should start reading pages, and the video and audio granulepos.
int64_t VideoStreamPlaybackTheora::seek_streams(double p_time, int64_t &cur_video_granulepos, int64_t &cur_audio_granulepos) {
	// Backtracking less than this is probably a waste of time.
	const int64_t min_seek = 512 * 1024;
	int64_t target_video_granulepos;
	int64_t target_audio_granulepos;
	double target_time = 0;
	int64_t seek_pos;

	// Make a guess where we should start reading in the file, and scan from there.
	// We base the guess on the mean bitrate of the streams. It would be theoretically faster to use the bisect method but
	// in practice there's a lot of linear scanning to do to find the right pages.
	// We want to catch the previous keyframe to the seek time. Since we only know the max GOP, we use that.
	if (p_time == -1) { // This is a special case to find the last packets and calculate the video length.
		seek_pos = MAX(stream_data_size - min_seek, stream_data_offset);
		target_video_granulepos = INT64_MAX;
		target_audio_granulepos = INT64_MAX;
	} else {
		int64_t video_frame = (int64_t)(p_time / frame_duration);
		target_video_granulepos = MAX(1LL, video_frame - (1LL << ti.keyframe_granule_shift)) << ti.keyframe_granule_shift;
		target_audio_granulepos = 0;
		seek_pos = MAX(((target_video_granulepos >> ti.keyframe_granule_shift) - 1) * frame_duration * stream_data_size / stream_length, stream_data_offset);
		target_time = th_granule_time(td, target_video_granulepos);
		if (has_audio) {
			target_audio_granulepos = video_frame * frame_duration * vi.rate;
			target_time = MIN(target_time, vorbis_granule_time(&vd, target_audio_granulepos));
		}
	}

	int64_t video_seek_pos = seek_pos;
	int64_t audio_seek_pos = seek_pos;
	double backtrack_time = 0;
	bool video_catch = false;
	bool audio_catch = false;
	int64_t last_video_granule_seek_pos = seek_pos;
	int64_t last_audio_granule_seek_pos = seek_pos;

	cur_video_granulepos = -1;
	cur_audio_granulepos = -1;

	while (!video_catch || (has_audio && !audio_catch)) { // Backtracking loop
		if (seek_pos < stream_data_offset) {
			seek_pos = stream_data_offset;
		}
		file->seek(seek_pos);
		ogg_sync_reset(&oy);

		backtrack_time = 0;
		last_video_granule_seek_pos = seek_pos;
		last_audio_granule_seek_pos = seek_pos;
		while (!video_catch || (has_audio && !audio_catch)) { // Page scanning loop
			ogg_page page;
			uint64_t last_seek_pos = file->get_position() - oy.fill + oy.returned;
			int ret = read_page(&page);
			if (ret <= 0) { // End of file.
				if (seek_pos < stream_data_offset) { // We've already searched the whole file
					return -1;
				}
				seek_pos -= min_seek;
				break;
			}
			int64_t cur_granulepos = ogg_page_granulepos(&page);
			if (cur_granulepos >= 0) {
				int page_serialno = ogg_page_serialno(&page);
				if (!video_catch && page_serialno == to.serialno) {
					if (cur_granulepos >= target_video_granulepos) {
						video_catch = true;
						if (cur_video_granulepos < 0) {
							// Adding 1s helps catching the start of the page and avoids backtrack_time = 0.
							backtrack_time = MAX(backtrack_time, 1 + th_granule_time(td, cur_granulepos) - target_time);
						}
					} else {
						video_seek_pos = last_video_granule_seek_pos;
						cur_video_granulepos = cur_granulepos;
					}
					last_video_granule_seek_pos = last_seek_pos;
				}
				if ((has_audio && !audio_catch) && page_serialno == vo.serialno) {
					if (cur_granulepos >= target_audio_granulepos) {
						audio_catch = true;
						if (cur_audio_granulepos < 0) {
							// Adding 1s helps catching the start of the page and avoids backtrack_time = 0.
							backtrack_time = MAX(backtrack_time, 1 + vorbis_granule_time(&vd, cur_granulepos) - target_time);
						}
					} else {
						audio_seek_pos = last_audio_granule_seek_pos;
						cur_audio_granulepos = cur_granulepos;
					}
					last_audio_granule_seek_pos = last_seek_pos;
				}
			}
		}
		if (backtrack_time > 0) {
			if (seek_pos <= stream_data_offset) {
				break;
			}
			int64_t delta_seek = MAX(backtrack_time * stream_data_size / stream_length, min_seek);
			seek_pos -= delta_seek;
		}
		video_catch = cur_video_granulepos != -1;
		audio_catch = cur_audio_granulepos != -1;
	}

	if (cur_video_granulepos < (1LL << ti.keyframe_granule_shift)) {
		video_seek_pos = stream_data_offset;
		cur_video_granulepos = 1LL << ti.keyframe_granule_shift;
	}
	if (has_audio) {
		if (cur_audio_granulepos == -1) {
			audio_seek_pos = stream_data_offset;
			cur_audio_granulepos = 0;
		}
		seek_pos = MIN(video_seek_pos, audio_seek_pos);
	} else {
		seek_pos = video_seek_pos;
	}

	return seek_pos;
}

void VideoStreamPlaybackTheora::video_write(th_ycbcr_buffer yuv) {
	uint8_t *w = frame_data.ptrw();
	char *dst = (char *)w;
	uint32_t y_offset = region.position.y * yuv[0].stride + region.position.x;
	uint32_t uv_offset = 0;

	if (px_fmt == TH_PF_444) {
		uv_offset += region.position.y * yuv[1].stride + region.position.x;
		yuv444_2_rgb8888((uint8_t *)dst, (uint8_t *)yuv[0].data + y_offset, (uint8_t *)yuv[1].data + uv_offset, (uint8_t *)yuv[2].data + uv_offset, region.size.x, region.size.y, yuv[0].stride, yuv[1].stride, region.size.x << 2);
	} else if (px_fmt == TH_PF_422) {
		uv_offset += region.position.y * yuv[1].stride + region.position.x / 2;
		yuv422_2_rgb8888((uint8_t *)dst, (uint8_t *)yuv[0].data + y_offset, (uint8_t *)yuv[1].data + uv_offset, (uint8_t *)yuv[2].data + uv_offset, region.size.x, region.size.y, yuv[0].stride, yuv[1].stride, region.size.x << 2);
	} else if (px_fmt == TH_PF_420) {
		uv_offset += region.position.y * yuv[1].stride / 2 + region.position.x / 2;
		yuv420_2_rgb8888((uint8_t *)dst, (uint8_t *)yuv[0].data + y_offset, (uint8_t *)yuv[1].data + uv_offset, (uint8_t *)yuv[2].data + uv_offset, region.size.x, region.size.y, yuv[0].stride, yuv[1].stride, region.size.x << 2);
	}

	Ref<Image> img;
	img.instantiate(region.size.x, region.size.y, false, Image::FORMAT_RGBA8, frame_data); //zero copy image creation

	texture->update(img); // Zero-copy send to rendering server.
}

void VideoStreamPlaybackTheora::clear() {
	if (!file.is_null()) {
		file.unref();
	}
	if (has_audio) {
		vorbis_block_clear(&vb);
		vorbis_dsp_clear(&vd);
		vorbis_comment_clear(&vc);
		vorbis_info_clear(&vi);
		ogg_stream_clear(&vo);
		if (audio_buffer_size) {
			memdelete_arr(audio_buffer);
		}
	}
	if (has_video) {
		th_decode_free(td);
		th_comment_clear(&tc);
		th_info_clear(&ti);
		ogg_stream_clear(&to);
		ogg_sync_clear(&oy);
	}

	audio_buffer = nullptr;
	playing = false;
	has_video = false;
	has_audio = false;
	theora_eos = false;
	vorbis_eos = false;
}

void VideoStreamPlaybackTheora::find_streams(th_setup_info *&ts) {
	ogg_stream_state test;
	ogg_packet op;
	ogg_page og;
	int stateflag = 0;
	int audio_track_skip = audio_track;

	/* Only interested in Vorbis/Theora streams */
	while (!stateflag) {
		int ret = buffer_data();
		if (!ret) {
			break;
		}
		while (ogg_sync_pageout(&oy, &og) > 0) {
			/* is this a mandated initial header? If not, stop parsing */
			if (!ogg_page_bos(&og)) {
				/* don't leak the page; get it into the appropriate stream */
				queue_page(&og);
				stateflag = 1;
				break;
			}

			ogg_stream_init(&test, ogg_page_serialno(&og));
			ogg_stream_pagein(&test, &og);
			ogg_stream_packetout(&test, &op);

			/* identify the codec: try theora */
			if (!has_video && th_decode_headerin(&ti, &tc, &ts, &op) >= 0) {
				/* it is theora */
				memcpy(&to, &test, sizeof(test));
				has_video = true;
			} else if (!has_audio && vorbis_synthesis_headerin(&vi, &vc, &op) >= 0) {
				/* it is vorbis */
				if (audio_track_skip) {
					vorbis_info_clear(&vi);
					vorbis_comment_clear(&vc);
					ogg_stream_clear(&test);
					vorbis_info_init(&vi);
					vorbis_comment_init(&vc);
					audio_track_skip--;
				} else {
					memcpy(&vo, &test, sizeof(test));
					has_audio = true;
				}
			} else {
				/* whatever it is, we don't care about it */
				ogg_stream_clear(&test);
			}
		}
	}
}

void VideoStreamPlaybackTheora::read_headers(th_setup_info *&ts) {
	ogg_packet op;
	int theora_header_packets = 1;
	int vorbis_header_packets = 1;

	/* we're expecting more header packets. */
	while (theora_header_packets < 3 || (has_audio && vorbis_header_packets < 3)) {
		/* look for further theora headers */
		// The API says there can be more than three but only three are mandatory.
		while (theora_header_packets < 3 && ogg_stream_packetout(&to, &op) > 0) {
			if (th_decode_headerin(&ti, &tc, &ts, &op) > 0) {
				theora_header_packets++;
			}
		}

		/* look for more vorbis header packets */
		while (has_audio && vorbis_header_packets < 3 && ogg_stream_packetout(&vo, &op) > 0) {
			if (!vorbis_synthesis_headerin(&vi, &vc, &op)) {
				vorbis_header_packets++;
			}
		}

		/* The header pages/packets will arrive before anything else we care about, or the stream is not obeying spec */
		if (theora_header_packets < 3 || (has_audio && vorbis_header_packets < 3)) {
			ogg_page page;
			if (read_page(&page)) {
				queue_page(&page);
			} else {
				fprintf(stderr, "End of file while searching for codec headers.\n");
				break;
			}
		}
	}

	has_video = theora_header_packets == 3;
	has_audio = vorbis_header_packets == 3;
}

void VideoStreamPlaybackTheora::set_file(const String &p_file) {
	ERR_FAIL_COND(playing);
	th_setup_info *ts = nullptr;

	clear();

	file = FileAccess::open(p_file, FileAccess::READ);
	ERR_FAIL_COND_MSG(file.is_null(), "Cannot open file '" + p_file + "'.");

	file_name = p_file;

	ogg_sync_init(&oy);

	/* init supporting Vorbis structures needed in header parsing */
	vorbis_info_init(&vi);
	vorbis_comment_init(&vc);

	/* init supporting Theora structures needed in header parsing */
	th_comment_init(&tc);
	th_info_init(&ti);

	/* Zero stream state structs so they can be checked later. */
	memset(&to, 0, sizeof(to));
	memset(&vo, 0, sizeof(vo));

	/* Ogg file open; parse the headers */
	find_streams(ts);
	read_headers(ts);

	if (!has_audio) {
		vorbis_comment_clear(&vc);
		vorbis_info_clear(&vi);
		if (!ogg_stream_check(&vo)) {
			ogg_stream_clear(&vo);
		}
	}

	// One video stream is mandatory.
	if (!has_video) {
		th_setup_free(ts);
		th_comment_clear(&tc);
		th_info_clear(&ti);
		if (!ogg_stream_check(&to)) {
			ogg_stream_clear(&to);
		}
		file.unref();
		return;
	}

	/* And now we have it all. Initialize decoders. */
	td = th_decode_alloc(&ti, ts);
	th_setup_free(ts);
	px_fmt = ti.pixel_fmt;
	switch (ti.pixel_fmt) {
		case TH_PF_420:
		case TH_PF_422:
		case TH_PF_444:
			break;
		default:
			WARN_PRINT(" video\n  (UNKNOWN Chroma sampling!)\n");
			break;
	}
	th_decode_ctl(td, TH_DECCTL_GET_PPLEVEL_MAX, &pp_level_max, sizeof(pp_level_max));
	pp_level = 0;
	th_decode_ctl(td, TH_DECCTL_SET_PPLEVEL, &pp_level, sizeof(pp_level));
	pp_inc = 0;

	size.x = ti.frame_width;
	size.y = ti.frame_height;
	region.position.x = ti.pic_x;
	region.position.y = ti.pic_y;
	region.size.x = ti.pic_width;
	region.size.y = ti.pic_height;

	Ref<Image> img = Image::create_empty(region.size.x, region.size.y, false, Image::FORMAT_RGBA8);
	texture->set_image(img);
	frame_data.resize(region.size.x * region.size.y * 4);

	frame_duration = (double)ti.fps_denominator / ti.fps_numerator;

	if (has_audio) {
		vorbis_synthesis_init(&vd, &vi);
		vorbis_block_init(&vd, &vb);
		audio_buffer_size = MIN(vi.channels, 8) * 1024;
		audio_buffer = memnew_arr(float, audio_buffer_size);
	}

	stream_data_offset = file->get_position() - oy.fill + oy.returned;
	stream_data_size = file->get_length() - stream_data_offset;

	// Sync to last page to find video length.
	int64_t seek_pos = MAX(stream_data_offset, (int64_t)file->get_length() - 64 * 1024);
	int64_t video_granulepos = INT64_MAX;
	int64_t audio_granulepos = INT64_MAX;
	file->seek(seek_pos);
	seek_pos = seek_streams(-1, video_granulepos, audio_granulepos);
	file->seek(seek_pos);
	ogg_sync_reset(&oy);

	stream_length = 0;
	ogg_page page;
	while (read_page(&page) > 0) {
		// Use MAX because, even though pages are ordered, page time can be -1
		// for pages without full frames. Streams could be truncated too.
		stream_length = MAX(stream_length, get_page_time(&page));
	}

	seek(0);
}

double VideoStreamPlaybackTheora::get_time() const {
	// FIXME: AudioServer output latency was fixed in af9bb0e, previously it used to
	// systematically return 0. Now that it gives a proper latency, it broke this
	// code where the delay compensation likely never really worked.
	return time - /* AudioServer::get_singleton()->get_output_latency() - */ delay_compensation;
}

Ref<Texture2D> VideoStreamPlaybackTheora::get_texture() const {
	return texture;
}

void VideoStreamPlaybackTheora::update(double p_delta) {
	if (file.is_null()) {
		return;
	}

	if (!playing || paused) {
		return;
	}

	time += p_delta;

	double comp_time = get_time();
	bool audio_ready = false;

	// Read data until we fill the audio buffer and get a new video frame.
	while ((!audio_ready && !audio_done) || (!video_ready && !video_done)) {
		ogg_packet op;

		while (!audio_ready && !audio_done) {
			// Send remaining frames
			if (!send_audio()) {
				audio_ready = true;
				break;
			}

			float **pcm;
			int ret = vorbis_synthesis_pcmout(&vd, &pcm);
			if (ret > 0) {
				int frames_read = 0;
				while (frames_read < ret) {
					int m = MIN(audio_buffer_size / vi.channels, ret - frames_read);
					int count = 0;
					for (int j = 0; j < m; j++) {
						for (int i = 0; i < vi.channels; i++) {
							audio_buffer[count++] = pcm[i][frames_read + j];
						}
					}
					frames_read += m;
					audio_ptr_end = m;
					if (!send_audio()) {
						audio_ready = true;
						break;
					}
				}
				vorbis_synthesis_read(&vd, frames_read);
			} else {
				/* no pending audio; is there a pending packet to decode? */
				if (ogg_stream_packetout(&vo, &op) > 0) {
					if (vorbis_synthesis(&vb, &op) == 0) { /* test for success! */
						vorbis_synthesis_blockin(&vd, &vb);
					}
				} else { /* we need more data; break out to suck in another page */
					audio_done = vorbis_eos;
					break;
				}
			}
		}

		while (!video_ready && !video_done) {
			if (ogg_stream_packetout(&to, &op) > 0) {
				if (op.granulepos >= 0) {
					th_decode_ctl(td, TH_DECCTL_SET_GRANPOS, &op.granulepos, sizeof(op.granulepos));
				}
				int64_t videobuf_granulepos;
				int ret = th_decode_packetin(td, &op, &videobuf_granulepos);
				if (ret == 0 || ret == TH_DUPFRAME) {
					next_frame_time = th_granule_time(td, videobuf_granulepos);
					if (next_frame_time > comp_time) {
						dup_frame = (ret == TH_DUPFRAME);
						video_ready = true;
					} else {
						/*If we are too slow, reduce the pp level.*/
						pp_inc = pp_level > 0 ? -1 : 0;
					}
				}
			} else { /* we need more data; break out to suck in another page */
				video_done = theora_eos;
				break;
			}
		}

		if (!video_ready || !audio_ready) {
			int ret = feed_pages();
			if (ret == 0) {
				vorbis_eos = true;
				theora_eos = true;
				break;
			}
		}

		double tdiff = next_frame_time - comp_time;
		/*If we have lots of extra time, increase the post-processing level.*/
		if (tdiff > ti.fps_denominator * 0.25 / ti.fps_numerator) {
			pp_inc = pp_level < pp_level_max ? 1 : 0;
		} else if (tdiff < ti.fps_denominator * 0.05 / ti.fps_numerator) {
			pp_inc = pp_level > 0 ? -1 : 0;
		}
	}

	if (!video_ready && video_done && audio_done) {
		stop();
		return;
	}

	// Wait for the last frame to end before rendering the next one.
	if (video_ready && comp_time >= current_frame_time) {
		if (!dup_frame) {
			th_ycbcr_buffer yuv;
			th_decode_ycbcr_out(td, yuv);
			video_write(yuv);
		}
		dup_frame = false;
		video_ready = false;
		current_frame_time = next_frame_time;
	}
}

void VideoStreamPlaybackTheora::play() {
	if (playing) {
		return;
	}

	playing = true;
	delay_compensation = GLOBAL_GET("audio/video/video_delay_compensation_ms");
	delay_compensation /= 1000.0;
}

void VideoStreamPlaybackTheora::stop() {
	playing = false;
	seek(0);
}

bool VideoStreamPlaybackTheora::is_playing() const {
	return playing;
}

void VideoStreamPlaybackTheora::set_paused(bool p_paused) {
	paused = p_paused;
}

bool VideoStreamPlaybackTheora::is_paused() const {
	return paused;
}

double VideoStreamPlaybackTheora::get_length() const {
	return stream_length;
}

double VideoStreamPlaybackTheora::get_playback_position() const {
	return time;
}

void VideoStreamPlaybackTheora::seek(double p_time) {
	ERR_FAIL_COND(p_time < 0.0);

	if (file.is_null()) {
		return;
	}
	if (p_time >= stream_length) {
		return;
	}

	video_ready = false;
	next_frame_time = 0;
	current_frame_time = -1;
	dup_frame = false;
	video_done = false;
	audio_done = !has_audio;
	theora_eos = false;
	vorbis_eos = false;
	audio_ptr_start = 0;
	audio_ptr_end = 0;

	ogg_stream_reset(&to);
	if (has_audio) {
		ogg_stream_reset(&vo);
		vorbis_synthesis_restart(&vd);
	}

	int64_t seek_pos;
	int64_t video_granulepos;
	int64_t audio_granulepos;
	// Find the granules we need so we can start playing at the seek time.
	seek_pos = seek_streams(p_time, video_granulepos, audio_granulepos);
	if (seek_pos < 0) {
		return;
	}
	file->seek(seek_pos);
	ogg_sync_reset(&oy);

	time = p_time;

	double last_audio_time = 0;
	double last_video_time = 0;
	bool first_frame_decoded = false;
	bool start_audio = (audio_granulepos == 0);
	bool start_video = (video_granulepos == (1LL << ti.keyframe_granule_shift));
	bool keyframe_found = false;
	uint64_t current_frame = 0;

	// Read from the streams skipping pages until we reach the granules we want. We won't skip pages from both video and
	// audio streams, only one of them, until decoding of both starts.
	// video_granulepos and audio_granulepos are guaranteed to be found by checking the granulepos in the packets, no
	// need to keep track of packets with granulepos == -1 until decoding starts.
	while ((has_audio && last_audio_time < p_time) || (last_video_time <= p_time)) {
		ogg_packet op;
		if (feed_pages() == 0) {
			break;
		}
		while (has_audio && last_audio_time < p_time && ogg_stream_packetout(&vo, &op) > 0) {
			if (start_audio) {
				if (vorbis_synthesis(&vb, &op) == 0) { /* test for success! */
					vorbis_synthesis_blockin(&vd, &vb);
					float **pcm;
					int samples_left = ceil((p_time - last_audio_time) * vi.rate);
					int samples_read = vorbis_synthesis_pcmout(&vd, &pcm);
					int samples_consumed = MIN(samples_left, samples_read);
					vorbis_synthesis_read(&vd, samples_consumed);
					last_audio_time += (double)samples_consumed / vi.rate;
				}
			} else if (op.granulepos >= audio_granulepos) {
				last_audio_time = vorbis_granule_time(&vd, op.granulepos);
				// Start tracking audio now. This won't produce any samples but will update the decoder state.
				if (vorbis_synthesis_trackonly(&vb, &op) == 0) {
					vorbis_synthesis_blockin(&vd, &vb);
				}
				start_audio = true;
			}
		}
		while (last_video_time <= p_time && ogg_stream_packetout(&to, &op) > 0) {
			if (!start_video && (op.granulepos >= video_granulepos || video_granulepos == (1LL << ti.keyframe_granule_shift))) {
				if (op.granulepos > 0) {
					current_frame = th_granule_frame(td, op.granulepos);
				}
				start_video = true;
			}
			// Don't start decoding until a keyframe is found, but count frames.
			if (start_video) {
				if (!keyframe_found && th_packet_iskeyframe(&op)) {
					keyframe_found = true;
					int64_t cur_granulepos = (current_frame + 1) << ti.keyframe_granule_shift;
					th_decode_ctl(td, TH_DECCTL_SET_GRANPOS, &cur_granulepos, sizeof(cur_granulepos));
				}
				if (keyframe_found) {
					int64_t videobuf_granulepos;
					if (op.granulepos >= 0) {
						th_decode_ctl(td, TH_DECCTL_SET_GRANPOS, &op.granulepos, sizeof(op.granulepos));
					}
					int ret = th_decode_packetin(td, &op, &videobuf_granulepos);
					if (ret == 0 || ret == TH_DUPFRAME) {
						last_video_time = th_granule_time(td, videobuf_granulepos);
						first_frame_decoded = true;
					}
				} else {
					current_frame++;
				}
			}
		}
	}

	if (first_frame_decoded) {
		if (is_playing()) {
			// Draw the current frame.
			th_ycbcr_buffer yuv;
			th_decode_ycbcr_out(td, yuv);
			video_write(yuv);
			current_frame_time = last_video_time;
		} else {
			next_frame_time = current_frame_time;
			video_ready = true;
		}
	}
}

int VideoStreamPlaybackTheora::get_channels() const {
	return vi.channels;
}

void VideoStreamPlaybackTheora::set_audio_track(int p_idx) {
	audio_track = p_idx;
}

int VideoStreamPlaybackTheora::get_mix_rate() const {
	return vi.rate;
}

VideoStreamPlaybackTheora::VideoStreamPlaybackTheora() {
	texture.instantiate();
}

VideoStreamPlaybackTheora::~VideoStreamPlaybackTheora() {
	clear();
}

void VideoStreamTheora::_bind_methods() {}

Ref<Resource> ResourceFormatLoaderTheora::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	if (f.is_null()) {
		if (r_error) {
			*r_error = ERR_CANT_OPEN;
		}
		return Ref<Resource>();
	}

	VideoStreamTheora *stream = memnew(VideoStreamTheora);
	stream->set_file(p_path);

	Ref<VideoStreamTheora> ogv_stream = Ref<VideoStreamTheora>(stream);

	if (r_error) {
		*r_error = OK;
	}

	return ogv_stream;
}

void ResourceFormatLoaderTheora::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("ogv");
}

bool ResourceFormatLoaderTheora::handles_type(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "VideoStream");
}

String ResourceFormatLoaderTheora::get_resource_type(const String &p_path) const {
	if (p_path.has_extension("ogv")) {
		return "VideoStreamTheora";
	}
	return "";
}

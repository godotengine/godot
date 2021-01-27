/*************************************************************************/
/*  video_stream_theora.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "video_stream_theora.h"

#include "core/os/os.h"
#include "core/project_settings.h"

#include "thirdparty/misc/yuv2rgb.h"

int VideoStreamPlaybackTheora::buffer_data() {

	char *buffer = ogg_sync_buffer(&oy, 4096);

#ifdef THEORA_USE_THREAD_STREAMING

	int read;

	do {
		thread_sem.post();
		read = MIN(ring_buffer.data_left(), 4096);
		if (read) {
			ring_buffer.read((uint8_t *)buffer, read);
			ogg_sync_wrote(&oy, read);
		} else {
			OS::get_singleton()->delay_usec(100);
		}

	} while (read == 0);

	return read;

#else

	int bytes = file->get_buffer((uint8_t *)buffer, 4096);
	ogg_sync_wrote(&oy, bytes);
	return (bytes);

#endif
}

int VideoStreamPlaybackTheora::queue_page(ogg_page *page) {
	if (theora_p) {
		ogg_stream_pagein(&to, page);
		if (to.e_o_s)
			theora_eos = true;
	}
	if (vorbis_p) {
		ogg_stream_pagein(&vo, page);
		if (vo.e_o_s)
			vorbis_eos = true;
	}
	return 0;
}

void VideoStreamPlaybackTheora::video_write(void) {
	th_ycbcr_buffer yuv;
	th_decode_ycbcr_out(td, yuv);

	int pitch = 4;
	frame_data.resize(size.x * size.y * pitch);
	{
		PoolVector<uint8_t>::Write w = frame_data.write();
		char *dst = (char *)w.ptr();

		//uv_offset=(ti.pic_x/2)+(yuv[1].stride)*(ti.pic_y/2);

		if (px_fmt == TH_PF_444) {

			yuv444_2_rgb8888((uint8_t *)dst, (uint8_t *)yuv[0].data, (uint8_t *)yuv[1].data, (uint8_t *)yuv[2].data, size.x, size.y, yuv[0].stride, yuv[1].stride, size.x << 2);

		} else if (px_fmt == TH_PF_422) {

			yuv422_2_rgb8888((uint8_t *)dst, (uint8_t *)yuv[0].data, (uint8_t *)yuv[1].data, (uint8_t *)yuv[2].data, size.x, size.y, yuv[0].stride, yuv[1].stride, size.x << 2);

		} else if (px_fmt == TH_PF_420) {

			yuv420_2_rgb8888((uint8_t *)dst, (uint8_t *)yuv[0].data, (uint8_t *)yuv[1].data, (uint8_t *)yuv[2].data, size.x, size.y, yuv[0].stride, yuv[1].stride, size.x << 2);
		};

		format = Image::FORMAT_RGBA8;
	}

	Ref<Image> img = memnew(Image(size.x, size.y, 0, Image::FORMAT_RGBA8, frame_data)); //zero copy image creation

	texture->set_data(img); //zero copy send to visual server

	frames_pending = 1;
}

void VideoStreamPlaybackTheora::clear() {

	if (!file)
		return;

	if (vorbis_p) {
		ogg_stream_clear(&vo);
		if (vorbis_p >= 3) {
			vorbis_block_clear(&vb);
			vorbis_dsp_clear(&vd);
		};
		vorbis_comment_clear(&vc);
		vorbis_info_clear(&vi);
		vorbis_p = 0;
	}
	if (theora_p) {
		ogg_stream_clear(&to);
		th_decode_free(td);
		th_comment_clear(&tc);
		th_info_clear(&ti);
		theora_p = 0;
	}
	ogg_sync_clear(&oy);

#ifdef THEORA_USE_THREAD_STREAMING
	thread_exit = true;
	thread_sem.post(); //just in case
	thread.wait_to_finish();
	ring_buffer.clear();
#endif

	theora_p = 0;
	vorbis_p = 0;
	videobuf_ready = 0;
	frames_pending = 0;
	videobuf_time = 0;
	theora_eos = false;
	vorbis_eos = false;

	if (file) {
		memdelete(file);
	}
	file = NULL;
	playing = false;
};

void VideoStreamPlaybackTheora::set_file(const String &p_file) {

	ERR_FAIL_COND(playing);
	ogg_packet op;
	th_setup_info *ts = NULL;

	file_name = p_file;
	if (file) {
		memdelete(file);
	}
	file = FileAccess::open(p_file, FileAccess::READ);
	ERR_FAIL_COND_MSG(!file, "Cannot open file '" + p_file + "'.");

#ifdef THEORA_USE_THREAD_STREAMING
	thread_exit = false;
	thread_eof = false;
	//pre-fill buffer
	int to_read = ring_buffer.space_left();
	int read = file->get_buffer(read_buffer.ptr(), to_read);
	ring_buffer.write(read_buffer.ptr(), read);

	thread.start(_streaming_thread, this);

#endif

	ogg_sync_init(&oy);

	/* init supporting Vorbis structures needed in header parsing */
	vorbis_info_init(&vi);
	vorbis_comment_init(&vc);

	/* init supporting Theora structures needed in header parsing */
	th_comment_init(&tc);
	th_info_init(&ti);

	theora_eos = false;
	vorbis_eos = false;

	/* Ogg file open; parse the headers */
	/* Only interested in Vorbis/Theora streams */
	int stateflag = 0;

	int audio_track_skip = audio_track;

	while (!stateflag) {
		int ret = buffer_data();
		if (ret == 0) break;
		while (ogg_sync_pageout(&oy, &og) > 0) {
			ogg_stream_state test;

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
			if (!theora_p && th_decode_headerin(&ti, &tc, &ts, &op) >= 0) {
				/* it is theora */
				copymem(&to, &test, sizeof(test));
				theora_p = 1;
			} else if (!vorbis_p && vorbis_synthesis_headerin(&vi, &vc, &op) >= 0) {

				/* it is vorbis */
				if (audio_track_skip) {
					vorbis_info_clear(&vi);
					vorbis_comment_clear(&vc);
					ogg_stream_clear(&test);
					vorbis_info_init(&vi);
					vorbis_comment_init(&vc);

					audio_track_skip--;
				} else {
					copymem(&vo, &test, sizeof(test));
					vorbis_p = 1;
				}
			} else {
				/* whatever it is, we don't care about it */
				ogg_stream_clear(&test);
			}
		}
		/* fall through to non-bos page parsing */
	}

	/* we're expecting more header packets. */
	while ((theora_p && theora_p < 3) || (vorbis_p && vorbis_p < 3)) {
		int ret;

		/* look for further theora headers */
		while (theora_p && (theora_p < 3) && (ret = ogg_stream_packetout(&to, &op))) {
			if (ret < 0) {
				fprintf(stderr, "Error parsing Theora stream headers; corrupt stream?\n");
				clear();
				return;
			}
			if (!th_decode_headerin(&ti, &tc, &ts, &op)) {
				fprintf(stderr, "Error parsing Theora stream headers; corrupt stream?\n");
				clear();
				return;
			}
			theora_p++;
		}

		/* look for more vorbis header packets */
		while (vorbis_p && (vorbis_p < 3) && (ret = ogg_stream_packetout(&vo, &op))) {
			if (ret < 0) {
				fprintf(stderr, "Error parsing Vorbis stream headers; corrupt stream?\n");
				clear();
				return;
			}
			ret = vorbis_synthesis_headerin(&vi, &vc, &op);
			if (ret) {
				fprintf(stderr, "Error parsing Vorbis stream headers; corrupt stream?\n");
				clear();
				return;
			}
			vorbis_p++;
			if (vorbis_p == 3) break;
		}

		/* The header pages/packets will arrive before anything else we
		care about, or the stream is not obeying spec */

		if (ogg_sync_pageout(&oy, &og) > 0) {
			queue_page(&og); /* demux into the appropriate stream */
		} else {
			int ret2 = buffer_data(); /* someone needs more data */
			if (ret2 == 0) {
				fprintf(stderr, "End of file while searching for codec headers.\n");
				clear();
				return;
			}
		}
	}

	/* and now we have it all.  initialize decoders */
	if (theora_p) {
		td = th_decode_alloc(&ti, ts);
		px_fmt = ti.pixel_fmt;
		switch (ti.pixel_fmt) {
			case TH_PF_420:
				//printf(" 4:2:0 video\n");
				break;
			case TH_PF_422:
				//printf(" 4:2:2 video\n");
				break;
			case TH_PF_444:
				//printf(" 4:4:4 video\n");
				break;
			case TH_PF_RSVD:
			default:
				printf(" video\n  (UNKNOWN Chroma sampling!)\n");
				break;
		}
		th_decode_ctl(td, TH_DECCTL_GET_PPLEVEL_MAX, &pp_level_max,
				sizeof(pp_level_max));
		pp_level = 0;
		th_decode_ctl(td, TH_DECCTL_SET_PPLEVEL, &pp_level, sizeof(pp_level));
		pp_inc = 0;

		int w;
		int h;
		w = ((ti.pic_x + ti.frame_width + 1) & ~1) - (ti.pic_x & ~1);
		h = ((ti.pic_y + ti.frame_height + 1) & ~1) - (ti.pic_y & ~1);
		size.x = w;
		size.y = h;

		texture->create(w, h, Image::FORMAT_RGBA8, Texture::FLAG_FILTER | Texture::FLAG_VIDEO_SURFACE);

	} else {
		/* tear down the partial theora setup */
		th_info_clear(&ti);
		th_comment_clear(&tc);
	}

	th_setup_free(ts);

	if (vorbis_p) {
		vorbis_synthesis_init(&vd, &vi);
		vorbis_block_init(&vd, &vb);
		//_setup(vi.channels, vi.rate);
	} else {
		/* tear down the partial vorbis setup */
		vorbis_info_clear(&vi);
		vorbis_comment_clear(&vc);
	}

	playing = false;
	buffering = true;
	time = 0;
	audio_frames_wrote = 0;
};

float VideoStreamPlaybackTheora::get_time() const {
	// FIXME: AudioServer output latency was fixed in af9bb0e, previously it used to
	// systematically return 0. Now that it gives a proper latency, it broke this
	// code where the delay compensation likely never really worked.
	return time - /* AudioServer::get_singleton()->get_output_latency() - */ delay_compensation;
};

Ref<Texture> VideoStreamPlaybackTheora::get_texture() const {

	return texture;
}

void VideoStreamPlaybackTheora::update(float p_delta) {

	if (!file)
		return;

	if (!playing || paused) {
		//printf("not playing\n");
		return;
	};

#ifdef THEORA_USE_THREAD_STREAMING
	thread_sem.post();
#endif

	time += p_delta;

	if (videobuf_time > get_time()) {
		return; //no new frames need to be produced
	}

	bool frame_done = false;
	bool audio_done = !vorbis_p;

	while (!frame_done || (!audio_done && !vorbis_eos)) {
		//a frame needs to be produced

		ogg_packet op;
		bool no_theora = false;
		bool buffer_full = false;

		while (vorbis_p && !audio_done && !buffer_full) {
			int ret;
			float **pcm;

			/* if there's pending, decoded audio, grab it */
			ret = vorbis_synthesis_pcmout(&vd, &pcm);
			if (ret > 0) {

				const int AUXBUF_LEN = 4096;
				int to_read = ret;
				float aux_buffer[AUXBUF_LEN];

				while (to_read) {

					int m = MIN(AUXBUF_LEN / vi.channels, to_read);

					int count = 0;

					for (int j = 0; j < m; j++) {
						for (int i = 0; i < vi.channels; i++) {
							aux_buffer[count++] = pcm[i][j];
						}
					}

					if (mix_callback) {
						int mixed = mix_callback(mix_udata, aux_buffer, m);
						to_read -= mixed;
						if (mixed != m) { //could mix no more
							buffer_full = true;
							break;
						}
					} else {
						to_read -= m; //just pretend we sent the audio
					}
				}

				vorbis_synthesis_read(&vd, ret - to_read);

				audio_frames_wrote += ret - to_read;

			} else {

				/* no pending audio; is there a pending packet to decode? */
				if (ogg_stream_packetout(&vo, &op) > 0) {
					if (vorbis_synthesis(&vb, &op) == 0) { /* test for success! */
						vorbis_synthesis_blockin(&vd, &vb);
					}
				} else { /* we need more data; break out to suck in another page */
					break;
				};
			}

			audio_done = videobuf_time < (audio_frames_wrote / float(vi.rate));

			if (buffer_full)
				break;
		}

		while (theora_p && !frame_done) {
			/* theora is one in, one out... */
			if (ogg_stream_packetout(&to, &op) > 0) {

				if (false && pp_inc) {
					pp_level += pp_inc;
					th_decode_ctl(td, TH_DECCTL_SET_PPLEVEL, &pp_level,
							sizeof(pp_level));
					pp_inc = 0;
				}
				/*HACK: This should be set after a seek or a gap, but we might not have
				a granulepos for the first packet (we only have them for the last
				packet on a page), so we just set it as often as we get it.
				To do this right, we should back-track from the last packet on the
				page and compute the correct granulepos for the first packet after
				a seek or a gap.*/
				if (op.granulepos >= 0) {
					th_decode_ctl(td, TH_DECCTL_SET_GRANPOS, &op.granulepos,
							sizeof(op.granulepos));
				}
				ogg_int64_t videobuf_granulepos;
				if (th_decode_packetin(td, &op, &videobuf_granulepos) == 0) {
					videobuf_time = th_granule_time(td, videobuf_granulepos);

					//printf("frame time %f, play time %f, ready %i\n", (float)videobuf_time, get_time(), videobuf_ready);

					/* is it already too old to be useful?  This is only actually
					 useful cosmetically after a SIGSTOP.  Note that we have to
					 decode the frame even if we don't show it (for now) due to
					 keyframing.  Soon enough libtheora will be able to deal
					 with non-keyframe seeks.  */

					if (videobuf_time >= get_time()) {
						frame_done = true;
					} else {
						/*If we are too slow, reduce the pp level.*/
						pp_inc = pp_level > 0 ? -1 : 0;
					}
				}

			} else {
				no_theora = true;
				break;
			}
		}

#ifdef THEORA_USE_THREAD_STREAMING
		if (file && thread_eof && no_theora && theora_eos && ring_buffer.data_left() == 0) {
#else
		if (file && /*!videobuf_ready && */ no_theora && theora_eos) {
#endif
			//printf("video done, stopping\n");
			stop();
			return;
		};

		if (!frame_done || !audio_done) {
			//what's the point of waiting for audio to grab a page?

			buffer_data();
			while (ogg_sync_pageout(&oy, &og) > 0) {
				queue_page(&og);
			}
		}

		/* If playback has begun, top audio buffer off immediately. */
		//if(stateflag) audio_write_nonblocking();

		/* are we at or past time for this video frame? */
		if (videobuf_ready && videobuf_time <= get_time()) {

			//video_write();
			//videobuf_ready=0;
		} else {
			//printf("frame at %f not ready (time %f), ready %i\n", (float)videobuf_time, get_time(), videobuf_ready);
		}

		float tdiff = videobuf_time - get_time();
		/*If we have lots of extra time, increase the post-processing level.*/
		if (tdiff > ti.fps_denominator * 0.25 / ti.fps_numerator) {
			pp_inc = pp_level < pp_level_max ? 1 : 0;
		} else if (tdiff < ti.fps_denominator * 0.05 / ti.fps_numerator) {
			pp_inc = pp_level > 0 ? -1 : 0;
		}
	}

	video_write();
};

void VideoStreamPlaybackTheora::play() {

	if (!playing)
		time = 0;
	else {
		stop();
	}

	playing = true;
	delay_compensation = ProjectSettings::get_singleton()->get("audio/video_delay_compensation_ms");
	delay_compensation /= 1000.0;
};

void VideoStreamPlaybackTheora::stop() {

	if (playing) {

		clear();
		set_file(file_name); //reset
	}
	playing = false;
	time = 0;
};

bool VideoStreamPlaybackTheora::is_playing() const {

	return playing;
};

void VideoStreamPlaybackTheora::set_paused(bool p_paused) {

	paused = p_paused;
};

bool VideoStreamPlaybackTheora::is_paused() const {

	return paused;
};

void VideoStreamPlaybackTheora::set_loop(bool p_enable){

};

bool VideoStreamPlaybackTheora::has_loop() const {

	return false;
};

float VideoStreamPlaybackTheora::get_length() const {

	return 0;
};

String VideoStreamPlaybackTheora::get_stream_name() const {

	return "";
};

int VideoStreamPlaybackTheora::get_loop_count() const {

	return 0;
};

float VideoStreamPlaybackTheora::get_playback_position() const {

	return get_time();
};

void VideoStreamPlaybackTheora::seek(float p_time){

	// no
};

void VideoStreamPlaybackTheora::set_mix_callback(AudioMixCallback p_callback, void *p_userdata) {

	mix_callback = p_callback;
	mix_udata = p_userdata;
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

#ifdef THEORA_USE_THREAD_STREAMING

void VideoStreamPlaybackTheora::_streaming_thread(void *ud) {

	VideoStreamPlaybackTheora *vs = (VideoStreamPlaybackTheora *)ud;

	while (!vs->thread_exit) {

		//just fill back the buffer
		if (!vs->thread_eof) {

			int to_read = vs->ring_buffer.space_left();
			if (to_read) {
				int read = vs->file->get_buffer(vs->read_buffer.ptr(), to_read);
				vs->ring_buffer.write(vs->read_buffer.ptr(), read);
				vs->thread_eof = vs->file->eof_reached();
			}
		}

		vs->thread_sem.wait();
	}
}

#endif

VideoStreamPlaybackTheora::VideoStreamPlaybackTheora() {

	file = NULL;
	theora_p = 0;
	vorbis_p = 0;
	videobuf_ready = 0;
	playing = false;
	frames_pending = 0;
	videobuf_time = 0;
	paused = false;

	buffering = false;
	texture = Ref<ImageTexture>(memnew(ImageTexture));
	mix_callback = NULL;
	mix_udata = NULL;
	audio_track = 0;
	delay_compensation = 0;
	audio_frames_wrote = 0;

#ifdef THEORA_USE_THREAD_STREAMING
	int rb_power = nearest_shift(RB_SIZE_KB * 1024);
	ring_buffer.resize(rb_power);
	read_buffer.resize(RB_SIZE_KB * 1024);
	thread_exit = false;
	thread_eof = false;

#endif
};

VideoStreamPlaybackTheora::~VideoStreamPlaybackTheora() {

	clear();

	if (file)
		memdelete(file);
};

void VideoStreamTheora::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_file", "file"), &VideoStreamTheora::set_file);
	ClassDB::bind_method(D_METHOD("get_file"), &VideoStreamTheora::get_file);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "file", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "set_file", "get_file");
}

////////////

RES ResourceFormatLoaderTheora::load(const String &p_path, const String &p_original_path, Error *r_error) {

	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	if (!f) {
		if (r_error) {
			*r_error = ERR_CANT_OPEN;
		}
		return RES();
	}

	VideoStreamTheora *stream = memnew(VideoStreamTheora);
	stream->set_file(p_path);

	Ref<VideoStreamTheora> ogv_stream = Ref<VideoStreamTheora>(stream);

	if (r_error) {
		*r_error = OK;
	}

	f->close();
	memdelete(f);
	return ogv_stream;
}

void ResourceFormatLoaderTheora::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("ogv");
}

bool ResourceFormatLoaderTheora::handles_type(const String &p_type) const {

	return ClassDB::is_parent_class(p_type, "VideoStream");
}

String ResourceFormatLoaderTheora::get_resource_type(const String &p_path) const {

	String el = p_path.get_extension().to_lower();
	if (el == "ogv")
		return "VideoStreamTheora";
	return "";
}

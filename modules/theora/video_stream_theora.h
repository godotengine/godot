/*************************************************************************/
/*  video_stream_theora.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef VIDEO_STREAM_THEORA_H
#define VIDEO_STREAM_THEORA_H

#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"
#include "core/templates/ring_buffer.h"
#include "core/templates/safe_refcount.h"
#include "scene/resources/video_stream.h"
#include "servers/audio_server.h"

#include <theora/theoradec.h>
#include <vorbis/codec.h>

//#define THEORA_USE_THREAD_STREAMING

class VideoStreamPlaybackTheora : public VideoStreamPlayback {
	GDCLASS(VideoStreamPlaybackTheora, VideoStreamPlayback);

	enum {
		MAX_FRAMES = 4,
	};

	//Image frames[MAX_FRAMES];
	Image::Format format = Image::Format::FORMAT_L8;
	Vector<uint8_t> frame_data;
	int frames_pending = 0;
	FileAccess *file = nullptr;
	String file_name;
	int audio_frames_wrote = 0;
	Point2i size;

	int buffer_data();
	int queue_page(ogg_page *page);
	void video_write();
	float get_time() const;

	bool theora_eos = false;
	bool vorbis_eos = false;

	ogg_sync_state oy;
	ogg_page og;
	ogg_stream_state vo;
	ogg_stream_state to;
	th_info ti;
	th_comment tc;
	th_dec_ctx *td = nullptr;
	vorbis_info vi;
	vorbis_dsp_state vd;
	vorbis_block vb;
	vorbis_comment vc;
	th_pixel_fmt px_fmt;
	double videobuf_time = 0;
	int pp_inc = 0;

	int theora_p = 0;
	int vorbis_p = 0;
	int pp_level_max = 0;
	int pp_level = 0;
	int videobuf_ready = 0;

	bool playing = false;
	bool buffering = false;

	double last_update_time = 0;
	double time = 0;
	double delay_compensation = 0;

	Ref<ImageTexture> texture;

	AudioMixCallback mix_callback;
	void *mix_udata = nullptr;
	bool paused = false;

#ifdef THEORA_USE_THREAD_STREAMING

	enum {
		RB_SIZE_KB = 1024
	};

	RingBuffer<uint8_t> ring_buffer;
	Vector<uint8_t> read_buffer;
	bool thread_eof = false;
	Semaphore *thread_sem;
	Thread thread;
	SafeFlag thread_exit;

	static void _streaming_thread(void *ud);

#endif

	int audio_track = 0;

protected:
	void clear();

public:
	virtual void play() override;
	virtual void stop() override;
	virtual bool is_playing() const override;

	virtual void set_paused(bool p_paused) override;
	virtual bool is_paused() const override;

	virtual void set_loop(bool p_enable) override;
	virtual bool has_loop() const override;

	virtual float get_length() const override;

	virtual String get_stream_name() const;

	virtual int get_loop_count() const;

	virtual float get_playback_position() const override;
	virtual void seek(float p_time) override;

	void set_file(const String &p_file);

	virtual Ref<Texture2D> get_texture() const override;
	virtual void update(float p_delta) override;

	virtual void set_mix_callback(AudioMixCallback p_callback, void *p_userdata) override;
	virtual int get_channels() const override;
	virtual int get_mix_rate() const override;

	virtual void set_audio_track(int p_idx) override;

	VideoStreamPlaybackTheora();
	~VideoStreamPlaybackTheora();
};

class VideoStreamTheora : public VideoStream {
	GDCLASS(VideoStreamTheora, VideoStream);

	String file;
	int audio_track;

protected:
	static void _bind_methods();

public:
	Ref<VideoStreamPlayback> instance_playback() override {
		Ref<VideoStreamPlaybackTheora> pb = memnew(VideoStreamPlaybackTheora);
		pb->set_audio_track(audio_track);
		pb->set_file(file);
		return pb;
	}

	void set_file(const String &p_file) { file = p_file; }
	String get_file() { return file; }
	void set_audio_track(int p_track) override { audio_track = p_track; }

	VideoStreamTheora() { audio_track = 0; }
};

class ResourceFormatLoaderTheora : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

#endif

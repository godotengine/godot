/**************************************************************************/
/*  video_stream_wmf.h                                                    */
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

#include "core/io/resource_loader.h"
#include "core/os/mutex.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/video_stream.h"

#include <deque>

class SampleGrabberCallback;
class AudioSampleGrabberCallback;
struct IMFMediaSession;
struct IMFMediaSource;
struct IMFTopology;
struct IMFPresentationClock;

struct FrameData {
	int64_t sample_time = 0;
	Vector<uint8_t> data;
};

struct AudioData {
	int64_t sample_time = 0;
	Vector<float> data;
};

class VideoStreamPlaybackWMF : public VideoStreamPlayback {
	GDCLASS(VideoStreamPlaybackWMF, VideoStreamPlayback);

	IMFMediaSession *media_session;
	IMFMediaSource *media_source;
	IMFTopology *topology;
	IMFPresentationClock *presentation_clock;
	AudioSampleGrabberCallback *audio_sample_grabber_callback;

	Vector<FrameData> cache_frames;
	int read_frame_idx = 0;
	int write_frame_idx = 0;

	Vector<AudioData> audio_cache_frames;
	int audio_read_frame_idx = 0;
	int audio_write_frame_idx = 0;

	Vector<uint8_t> frame_data;
	Ref<ImageTexture> texture;
	Mutex mtx;
	Mutex audio_mtx;

	bool is_video_playing = false;
	bool is_video_paused = false;
	bool is_video_seekable = false;

	double time = 0.0;
	double next_frame_time = 0.0;
	double current_frame_time = -1.0;
	bool frame_ready = false;

	int id = 0;

	AudioMixCallback mix_callback = nullptr;
	void *mix_udata = nullptr;

	// Audio properties
	int audio_channels = 0;
	int audio_sample_rate = 0;
	Vector<float> audio_buffer;
	int audio_buffer_pos = 0;

	void shutdown_stream();

public:
	struct StreamInfo {
		Point2i size;
		float fps = 0.0f;
		float duration = 0.0f;
	};
	StreamInfo stream_info;

	virtual void play() override;
	virtual void stop() override;
	virtual bool is_playing() const override;

	virtual void set_paused(bool p_paused) override;
	virtual bool is_paused() const override;

	virtual double get_length() const override;
	virtual String get_stream_name() const;

	virtual int get_loop_count() const;

	virtual double get_playback_position() const override;
	virtual void seek(double p_time) override;

	void set_file(const String &p_file);

	virtual Ref<Texture2D> get_texture() const override;
	virtual void update(double p_delta) override;

	virtual void set_mix_callback(AudioMixCallback p_callback, void *p_userdata) override;
	virtual int get_channels() const override;
	virtual int get_mix_rate() const override;

	virtual void set_audio_track(int p_idx) override;

	FrameData *get_next_writable_frame();
	void write_frame_done();
	void present();

	int64_t next_sample_time();

	// Audio methods
	bool send_audio();
	void add_audio_data(int64_t sample_time, const Vector<float> &audio_data);
	AudioData *get_next_writable_audio_frame();
	void write_audio_frame_done();
	void set_audio_format(int sample_rate, int channels);

	VideoStreamPlaybackWMF();
	~VideoStreamPlaybackWMF();
};

class VideoStreamWMF : public VideoStream {
	GDCLASS(VideoStreamWMF, VideoStream);

	String file;
	int audio_track = 0;

protected:
	static void _bind_methods();

public:
	Ref<VideoStreamPlayback> instantiate_playback() override {
		Ref<VideoStreamPlaybackWMF> pb = memnew(VideoStreamPlaybackWMF);
		pb->set_audio_track(audio_track);
		pb->set_file(file);
		return pb;
	}

	void set_file(const String &p_file) { file = p_file; }
	String get_file() const { return file; }

	void set_audio_track(int p_track) override { audio_track = p_track; }

	VideoStreamWMF() { audio_track = 0; }
};

class ResourceFormatLoaderWMF : public ResourceFormatLoader {
public:
	virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE) override;
	virtual void get_recognized_extensions(List<String> *p_extensions) const override;
	virtual bool handles_type(const String &p_type) const override;
	virtual String get_resource_type(const String &p_path) const override;
};

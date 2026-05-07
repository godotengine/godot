/**************************************************************************/
/*  video_stream_webm.h                                                   */
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
#include "scene/resources/image_texture.h"
#include "scene/resources/video_stream.h"

class WebMFrame;
class WebMDemuxer;
class VPXDecoder;
class OpusVorbisDecoder;

class VideoStreamPlaybackWebm : public VideoStreamPlayback {
	GDCLASS(VideoStreamPlaybackWebm, VideoStreamPlayback);

	String file_name;
	int audio_track = 0;

	WebMDemuxer *webm = nullptr;
	VPXDecoder *video = nullptr;
	OpusVorbisDecoder *audio = nullptr;

	WebMFrame **video_frames = nullptr;
	WebMFrame *audio_frame = nullptr;
	int video_frames_pos = 0;
	int video_frames_capacity = 0;

	int num_decoded_samples = 0;
	int samples_offset = -1;

	bool playing = false;
	bool paused = false;
	double delay_compensation = 0.0;
	double time = 0.0;
	double video_frame_delay = 0.0;
	double video_pos = 0.0;

	Vector<uint8_t> frame_data;
	Ref<ImageTexture> texture;

	TightLocalVector<float> pcm;

public:
	VideoStreamPlaybackWebm();
	~VideoStreamPlaybackWebm();

	bool set_file(const String &p_file);

	virtual void stop() override;
	virtual void play() override;

	virtual bool is_playing() const override;

	virtual void set_paused(bool p_paused) override;
	virtual bool is_paused() const override;

	virtual double get_length() const override;

	virtual double get_playback_position() const override;
	virtual void seek(double p_time) override;

	virtual void set_audio_track(int p_idx) override;

	virtual Ref<Texture2D> get_texture() const override;
	virtual void update(double p_delta) override;

	virtual int get_channels() const override;
	virtual int get_mix_rate() const override;

private:
	inline bool has_enough_video_frames() const;
	bool should_process(const WebMFrame &video_frame);
	void delete_pointers();
};

/**/

class VideoStreamWebm : public VideoStream {
	GDCLASS(VideoStreamWebm, VideoStream);

protected:
	static void _bind_methods();

public:
	virtual Ref<VideoStreamPlayback> instantiate_playback() override {
		Ref<VideoStreamPlaybackWebm> pb = memnew(VideoStreamPlaybackWebm);
		pb->set_audio_track(audio_track);
		pb->set_file(file);
		return pb;
	}

	virtual void set_audio_track(int p_track) override { audio_track = p_track; }

	VideoStreamWebm() { audio_track = 0; }
};

class ResourceFormatLoaderWebm : public ResourceFormatLoader {
public:
	virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE) override;
	virtual void get_recognized_extensions(List<String> *p_extensions) const override;
	virtual bool handles_type(const String &p_type) const override;
	virtual String get_resource_type(const String &p_path) const override;
};

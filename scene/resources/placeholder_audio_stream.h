/**************************************************************************/
/*  placeholder_audio_stream.h                                            */
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

#include "servers/audio/audio_stream.h"

class PlaceholderAudioStream;

class PlaceholderAudioStreamPlayback : public AudioStreamPlayback {
	GDCLASS(PlaceholderAudioStreamPlayback, AudioStreamPlayback);

	int64_t offset = 0;
	int8_t sign = 1;
	bool active = false;
	Ref<PlaceholderAudioStream> base;

	friend class PlaceholderAudioStream;

	bool _is_sample = false;
	Ref<AudioSamplePlayback> sample_playback;

	virtual int _mix_internal(AudioFrame *p_buffer, int p_frames);

public:
	virtual void start(double p_from_pos = 0.0) override;
	virtual void stop() override;
	virtual bool is_playing() const override;

	virtual int get_loop_count() const override;

	virtual double get_playback_position() const override;
	virtual void seek(double p_time) override;

	virtual void tag_used_streams() override;

	virtual void set_is_sample(bool p_is_sample) override;
	virtual bool get_is_sample() const override;
	virtual Ref<AudioSamplePlayback> get_sample_playback() const override;
	virtual void set_sample_playback(const Ref<AudioSamplePlayback> &p_playback) override;

	virtual int mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) override;
};

///////////

class PlaceholderAudioStream : public AudioStream {
	GDCLASS(PlaceholderAudioStream, AudioStream)

public:
	enum LoopMode {
		LOOP_DISABLED,
		LOOP_FORWARD,
		LOOP_PINGPONG,
		LOOP_BACKWARD
	};

private:
	double length = 0.0;
	LoopMode loop_mode = LOOP_DISABLED;
	int64_t loop_begin = 0;
	int64_t loop_end = 0;
	Dictionary tags;

	friend class PlaceholderAudioStreamPlayback;

protected:
	static void _bind_methods();

public:
	void set_length(double p_length);
	double get_length() const override;

	void set_loop_mode(LoopMode p_loop_mode);
	LoopMode get_loop_mode() const;

	void set_loop_begin(int p_frame);
	int get_loop_begin() const;

	void set_loop_end(int p_frame);
	int get_loop_end() const;

	void set_tags(const Dictionary &p_tags);
	Dictionary get_tags() const override;

	virtual Ref<AudioStreamPlayback> instantiate_playback() override;

	PlaceholderAudioStream();
	~PlaceholderAudioStream();
};

VARIANT_ENUM_CAST(PlaceholderAudioStream::LoopMode);

/**************************************************************************/
/*  audio_stream_ogg_opus.h                                               */
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

#include <include/opusfile.h>

class AudioStreamOggOpus;

class AudioStreamPlaybackOggOpus : public AudioStreamPlaybackResampled {
	GDCLASS(AudioStreamPlaybackOggOpus, AudioStreamPlaybackResampled);

	OggOpusFile *opus_file;
	uint32_t frames_mixed;
	bool active;
	int loops;

	friend class AudioStreamOggOpus;

	Ref<AudioStreamOggOpus> opus_stream;

protected:
	virtual int _mix_internal(AudioFrame *p_buffer, int p_frames) override;
	virtual float get_stream_sampling_rate() override;

public:
	virtual void start(double p_from_pos = 0.0) override;
	virtual void stop() override;
	virtual bool is_playing() const override;

	virtual int get_loop_count() const override; //times it looped

	virtual double get_playback_position() const override;
	virtual void seek(double p_time) override;

	AudioStreamPlaybackOggOpus() {}
	~AudioStreamPlaybackOggOpus();
};

class AudioStreamOggOpus : public AudioStream {
	GDCLASS(AudioStreamOggOpus, AudioStream);
	OBJ_SAVE_TYPE(AudioStream); // Saves derived classes with common type so they can be interchanged.
	RES_BASE_EXTENSION("oggopusstr");

	friend class AudioStreamPlaybackOggOpus;

	void *data;
	uint32_t data_len;
	float length;
	bool loop;
	float loop_offset;
	void clear_data();

	double bpm = 0;
	int beat_count = 0;
	int bar_beats = 4;

protected:
	static void _bind_methods();

public:
	static Ref<AudioStreamOggOpus> load_from_file(const String &p_path);
	static Ref<AudioStreamOggOpus> load_from_buffer(const Vector<uint8_t> &p_stream_data);

	void set_loop(bool p_enable);
	virtual bool has_loop() const override;

	void set_loop_offset(float p_seconds);
	float get_loop_offset() const;

	void set_bpm(double p_bpm);
	virtual double get_bpm() const override;

	void set_beat_count(int p_beat_count);
	virtual int get_beat_count() const override;

	void set_bar_beats(int p_bar_beats);
	virtual int get_bar_beats() const override;

	virtual Ref<AudioStreamPlayback> instantiate_playback() override;
	virtual String get_stream_name() const override;

	void set_data(const Vector<uint8_t> &p_data);
	Vector<uint8_t> get_data() const;

	virtual double get_length() const override;

	virtual bool is_monophonic() const override;

	virtual bool can_be_sampled() const override {
		return true;
	}
	virtual Ref<AudioSample> generate_sample() const override;

	AudioStreamOggOpus();
	virtual ~AudioStreamOggOpus();
};

/**************************************************************************/
/*  audio_stream_qoa.h                                                    */
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

#ifndef AUDIO_STREAM_QOA_H
#define AUDIO_STREAM_QOA_H

#include "core/io/resource_loader.h"
#include "servers/audio/audio_stream.h"

#include "thirdparty/misc/qoa.h"

class AudioStreamQOA;

class AudioStreamPlaybackQOA : public AudioStreamPlaybackResampled {
	GDCLASS(AudioStreamPlaybackQOA, AudioStreamPlaybackResampled);

	qoa_desc *qoad = nullptr;
	uint32_t data_offset = 0;
	uint32_t frames_mixed = 0;
	uint32_t frame_data_len = 0;
	int16_t *decoded = nullptr;
	uint32_t decoded_len = 0;
	uint32_t decoded_offset = 0;

	bool active = false;
	int increment = 1;

	friend class AudioStreamQOA;

	Ref<AudioStreamQOA> qoa_stream;

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

	virtual void tag_used_streams() override;

	AudioStreamPlaybackQOA() {}
	~AudioStreamPlaybackQOA();
};

class AudioStreamQOA : public AudioStream {
	GDCLASS(AudioStreamQOA, AudioStream);
	OBJ_SAVE_TYPE(AudioStream) //children are all saved as AudioStream, so they can be exchanged
	RES_BASE_EXTENSION("qoastr");

public:
	// Keep the ResourceImporterQOA `edit/loop_mode` enum hint in sync with these options.
	enum LoopMode {
		LOOP_DISABLED,
		LOOP_FORWARD,
		LOOP_PINGPONG,
		LOOP_BACKWARD,
	};

private:
	friend class AudioStreamPlaybackQOA;

	PackedByteArray data;
	uint32_t data_len = 0;

	LoopMode loop_mode = LOOP_DISABLED;
	bool stereo = false;
	float length = 0.0;
	int loop_begin = 0;
	int loop_end = -1;
	int mix_rate = 44100;
	void clear_data();

protected:
	static void _bind_methods();

public:
	void set_loop_mode(LoopMode p_loop_mode);
	LoopMode get_loop_mode() const;

	void set_loop_begin(int p_frame);
	int get_loop_begin() const;

	void set_loop_end(int p_frame);
	int get_loop_end() const;

	void set_mix_rate(int p_hz);
	int get_mix_rate() const;

	void set_stereo(bool p_stereo);
	bool is_stereo() const;

	virtual double get_length() const override;

	virtual bool is_monophonic() const override;

	void set_data(const Vector<uint8_t> &p_data);
	Vector<uint8_t> get_data() const;

	virtual Ref<AudioStreamPlayback> instantiate_playback() override;
	virtual String get_stream_name() const override;

	AudioStreamQOA();
	virtual ~AudioStreamQOA();
};

VARIANT_ENUM_CAST(AudioStreamQOA::LoopMode)

#endif // AUDIO_STREAM_QOA_H

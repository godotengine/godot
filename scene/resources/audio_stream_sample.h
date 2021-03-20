/*************************************************************************/
/*  audio_stream_sample.h                                                */
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

#ifndef AUDIO_STREAM_SAMPLE_H
#define AUDIO_STREAM_SAMPLE_H

#include "servers/audio/audio_stream.h"

class AudioStreamSample;

class AudioStreamPlaybackSample : public AudioStreamPlayback {
	GDCLASS(AudioStreamPlaybackSample, AudioStreamPlayback);
	enum {
		MIX_FRAC_BITS = 13,
		MIX_FRAC_LEN = (1 << MIX_FRAC_BITS),
		MIX_FRAC_MASK = MIX_FRAC_LEN - 1,
	};

	struct IMA_ADPCM_State {
		int16_t step_index = 0;
		int32_t predictor = 0;
		/* values at loop point */
		int16_t loop_step_index = 0;
		int32_t loop_predictor = 0;
		int32_t last_nibble = 0;
		int32_t loop_pos = 0;
		int32_t window_ofs = 0;
	} ima_adpcm[2];

	int64_t offset = 0;
	int sign = 1;
	bool active = false;
	friend class AudioStreamSample;
	Ref<AudioStreamSample> base;

	template <class Depth, bool is_stereo, bool is_ima_adpcm>
	void do_resample(const Depth *p_src, AudioFrame *p_dst, int64_t &offset, int32_t &increment, uint32_t amount, IMA_ADPCM_State *ima_adpcm);

public:
	virtual void start(float p_from_pos = 0.0) override;
	virtual void stop() override;
	virtual bool is_playing() const override;

	virtual int get_loop_count() const override; //times it looped

	virtual float get_playback_position() const override;
	virtual void seek(float p_time) override;

	virtual void mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) override;

	AudioStreamPlaybackSample();
};

class AudioStreamSample : public AudioStream {
	GDCLASS(AudioStreamSample, AudioStream);
	RES_BASE_EXTENSION("sample")

public:
	enum Format {
		FORMAT_8_BITS,
		FORMAT_16_BITS,
		FORMAT_IMA_ADPCM
	};

	enum LoopMode {
		LOOP_DISABLED,
		LOOP_FORWARD,
		LOOP_PING_PONG,
		LOOP_BACKWARD
	};

private:
	friend class AudioStreamPlaybackSample;

	enum {
		DATA_PAD = 16 //padding for interpolation
	};

	Format format = FORMAT_8_BITS;
	LoopMode loop_mode = LOOP_DISABLED;
	bool stereo = false;
	int loop_begin = 0;
	int loop_end = 0;
	int mix_rate = 44100;
	void *data = nullptr;
	uint32_t data_bytes = 0;

protected:
	static void _bind_methods();

public:
	void set_format(Format p_format);
	Format get_format() const;

	void set_loop_mode(LoopMode p_loop_mode);
	LoopMode get_loop_mode() const;

	void set_loop_begin(int p_frame);
	int get_loop_begin() const;

	void set_loop_end(int p_frame);
	int get_loop_end() const;

	void set_mix_rate(int p_hz);
	int get_mix_rate() const;

	void set_stereo(bool p_enable);
	bool is_stereo() const;

	virtual float get_length() const override; //if supported, otherwise return 0

	void set_data(const Vector<uint8_t> &p_data);
	Vector<uint8_t> get_data() const;

	Error save_to_wav(const String &p_path);

	virtual Ref<AudioStreamPlayback> instance_playback() override;
	virtual String get_stream_name() const override;

	AudioStreamSample();
	~AudioStreamSample();
};

VARIANT_ENUM_CAST(AudioStreamSample::Format)
VARIANT_ENUM_CAST(AudioStreamSample::LoopMode)

#endif // AUDIO_STREAM_SAMPLE_H

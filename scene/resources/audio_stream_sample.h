/**************************************************************************/
/*  audio_stream_sample.h                                                 */
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
		int16_t step_index;
		int32_t predictor;
		/* values at loop point */
		int16_t loop_step_index;
		int32_t loop_predictor;
		int32_t last_nibble;
		int32_t loop_pos;
		int32_t window_ofs;
	} ima_adpcm[2];

	int64_t offset;
	int sign;
	bool active;
	friend class AudioStreamSample;
	Ref<AudioStreamSample> base;

	template <class Depth, bool is_stereo, bool is_ima_adpcm>
	void do_resample(const Depth *p_src, AudioFrame *p_dst, int64_t &offset, int32_t &increment, uint32_t amount, IMA_ADPCM_State *ima_adpcm);

public:
	virtual void start(float p_from_pos = 0.0);
	virtual void stop();
	virtual bool is_playing() const;

	virtual int get_loop_count() const; //times it looped

	virtual float get_playback_position() const;
	virtual void seek(float p_time);

	virtual void mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames);

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

	// Keep the ResourceImporterWAV `edit/loop_mode` enum hint in sync with these options.
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

	Format format;
	LoopMode loop_mode;
	bool stereo;
	int loop_begin;
	int loop_end;
	int mix_rate;
	void *data;
	uint32_t data_bytes;

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

	virtual float get_length() const; //if supported, otherwise return 0

	void set_data(const PoolVector<uint8_t> &p_data);
	PoolVector<uint8_t> get_data() const;

	Error save_to_wav(const String &p_path);

	virtual Ref<AudioStreamPlayback> instance_playback();
	virtual String get_stream_name() const;

	AudioStreamSample();
	~AudioStreamSample();
};

VARIANT_ENUM_CAST(AudioStreamSample::Format)
VARIANT_ENUM_CAST(AudioStreamSample::LoopMode)

#endif // AUDIO_STREAM_SAMPLE_H

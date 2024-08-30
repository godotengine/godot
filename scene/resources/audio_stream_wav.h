/**************************************************************************/
/*  audio_stream_wav.h                                                    */
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

#ifndef AUDIO_STREAM_WAV_H
#define AUDIO_STREAM_WAV_H

#define QOA_IMPLEMENTATION
#define QOA_NO_STDIO

#include "servers/audio/audio_stream.h"
#include "thirdparty/misc/qoa.h"

class AudioStreamWAV;

class AudioStreamPlaybackWAV : public AudioStreamPlayback {
	GDCLASS(AudioStreamPlaybackWAV, AudioStreamPlayback);
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

	struct QOA_State {
		qoa_desc desc = {};
		uint32_t data_ofs = 0;
		uint32_t frame_len = 0;
		int16_t *dec = nullptr;
		uint32_t dec_len = 0;
		int64_t cache_pos = -1;
		int16_t cache[2] = { 0, 0 };
		int16_t cache_r[2] = { 0, 0 };
	} qoa;

	int64_t offset = 0;
	int sign = 1;
	bool active = false;
	friend class AudioStreamWAV;
	Ref<AudioStreamWAV> base;

	template <typename Depth, bool is_stereo, bool is_ima_adpcm, bool is_qoa>
	void do_resample(const Depth *p_src, AudioFrame *p_dst, int64_t &p_offset, int32_t &p_increment, uint32_t p_amount, IMA_ADPCM_State *p_ima_adpcm, QOA_State *p_qoa);

	bool _is_sample = false;
	Ref<AudioSamplePlayback> sample_playback;

public:
	virtual void start(double p_from_pos = 0.0) override;
	virtual void stop() override;
	virtual bool is_playing() const override;

	virtual int get_loop_count() const override; //times it looped

	virtual double get_playback_position() const override;
	virtual void seek(double p_time) override;

	virtual int mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) override;

	virtual void tag_used_streams() override;

	virtual void set_is_sample(bool p_is_sample) override;
	virtual bool get_is_sample() const override;
	virtual Ref<AudioSamplePlayback> get_sample_playback() const override;
	virtual void set_sample_playback(const Ref<AudioSamplePlayback> &p_playback) override;

	AudioStreamPlaybackWAV();
	~AudioStreamPlaybackWAV();
};

class AudioStreamWAV : public AudioStream {
	GDCLASS(AudioStreamWAV, AudioStream);
	RES_BASE_EXTENSION("sample")

public:
	enum Format {
		FORMAT_8_BITS,
		FORMAT_16_BITS,
		FORMAT_IMA_ADPCM,
		FORMAT_QOA,
	};

	// Keep the ResourceImporterWAV `edit/loop_mode` enum hint in sync with these options.
	enum LoopMode {
		LOOP_DISABLED,
		LOOP_FORWARD,
		LOOP_PINGPONG,
		LOOP_BACKWARD
	};

private:
	friend class AudioStreamPlaybackWAV;

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

	virtual double get_length() const override; //if supported, otherwise return 0

	virtual bool is_monophonic() const override;

	void set_data(const Vector<uint8_t> &p_data);
	Vector<uint8_t> get_data() const;

	Error save_to_wav(const String &p_path);

	virtual Ref<AudioStreamPlayback> instantiate_playback() override;
	virtual String get_stream_name() const override;

	virtual bool can_be_sampled() const override {
		return true;
	}
	virtual Ref<AudioSample> generate_sample() const override;

	AudioStreamWAV();
	~AudioStreamWAV();
};

VARIANT_ENUM_CAST(AudioStreamWAV::Format)
VARIANT_ENUM_CAST(AudioStreamWAV::LoopMode)

#endif // AUDIO_STREAM_WAV_H

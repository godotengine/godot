/*************************************************************************/
/*  audio_mixer_sw.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef AUDIO_MIXER_SW_H
#define AUDIO_MIXER_SW_H

#include "servers/audio/audio_filter_sw.h"
#include "servers/audio/reverb_sw.h"
#include "servers/audio/sample_manager_sw.h"
#include "servers/audio_server.h"

class AudioMixerSW : public AudioMixer {
public:
	enum InterpolationType {

		INTERPOLATION_RAW,
		INTERPOLATION_LINEAR,
		INTERPOLATION_CUBIC
	};

	enum MixChannels {

		MIX_STEREO = 2,
		MIX_QUAD = 4
	};

	typedef void (*MixStepCallback)(void *);

private:
	SampleManagerSW *sample_manager;

	enum {

		MAX_CHANNELS = 64,
		// fixed point defs

		MIX_FRAC_BITS = 13,
		MIX_FRAC_LEN = (1 << MIX_FRAC_BITS),
		MIX_FRAC_MASK = MIX_FRAC_LEN - 1,
		MIX_VOL_FRAC_BITS = 12,
		MIX_VOLRAMP_FRAC_BITS = 16,
		MIX_VOLRAMP_FRAC_LEN = (1 << MIX_VOLRAMP_FRAC_BITS),
		MIX_VOLRAMP_FRAC_MASK = MIX_VOLRAMP_FRAC_LEN - 1,
		MIX_FILTER_FRAC_BITS = 16,
		MIX_FILTER_RAMP_FRAC_BITS = 8,
		MIX_VOL_MOVE_TO_24 = 4
	};

	struct Channel {

		RID sample;
		struct Mix {
			int64_t offset;
			int32_t increment;

			int32_t vol[4];
			int32_t reverb_vol[4];
			int32_t chorus_vol[4];

			int32_t old_vol[4];
			int32_t old_reverb_vol[4];
			int32_t old_chorus_vol[4];

			struct Filter { //history (stereo)
				float ha[2], hb[2];
			} filter_l, filter_r;

			struct IMA_ADPCM_State {

				int16_t step_index;
				int32_t predictor;
				/* values at loop point */
				int16_t loop_step_index;
				int32_t loop_predictor;
				int32_t last_nibble;
				int32_t loop_pos;
				int32_t window_ofs;
				const uint8_t *ptr;
			} ima_adpcm[2];

		} mix;

		float vol;
		float pan;
		float depth;
		float height;

		float chorus_send;
		ReverbRoomType reverb_room;
		float reverb_send;
		int speed;
		int check;
		bool positional;

		bool had_prev_reverb;
		bool had_prev_chorus;
		bool had_prev_vol;

		struct Filter {

			bool dirty;

			FilterType type;
			float cutoff;
			float resonance;
			float gain;

			struct Coefs {

				float a1, a2, b0, b1, b2; // fixed point coefficients
			} coefs, old_coefs;

		} filter;

		bool first_mix;
		bool active;
		Channel() {
			active = false;
			check = -1;
			first_mix = false;
			filter.dirty = true;
			filter.type = FILTER_NONE;
			filter.cutoff = 8000;
			filter.resonance = 0;
			filter.gain = 0;
		}
	};

	Channel channels[MAX_CHANNELS];

	uint32_t mix_rate;
	bool fx_enabled;
	InterpolationType interpolation_type;

	int mix_chunk_bits;
	int mix_chunk_size;
	int mix_chunk_mask;

	int32_t *mix_buffer;
	int32_t *zero_buffer; // fx feed when no input was mixed

	struct ResamplerState {

		uint32_t amount;
		int32_t increment;

		int32_t pos;

		int32_t vol[4];
		int32_t reverb_vol[4];
		int32_t chorus_vol[4];

		int32_t vol_inc[4];
		int32_t reverb_vol_inc[4];
		int32_t chorus_vol_inc[4];

		Channel::Mix::Filter *filter_l;
		Channel::Mix::Filter *filter_r;
		Channel::Filter::Coefs coefs;
		Channel::Filter::Coefs coefs_inc;

		Channel::Mix::IMA_ADPCM_State *ima_adpcm;

		int32_t *reverb_buffer;
	};

	template <class Depth, bool is_stereo, bool use_filter, bool is_ima_adpcm, bool use_fx, InterpolationType type, MixChannels>
	_FORCE_INLINE_ void do_resample(const Depth *p_src, int32_t *p_dst, ResamplerState *p_state);

	MixChannels mix_channels;

	void mix_channel(Channel &p_channel);
	int mix_chunk_left;
	void mix_chunk();

	float channel_nrg;
	int channel_id_count;
	bool inside_mix;
	MixStepCallback step_callback;
	void *step_udata;
	_FORCE_INLINE_ int _get_channel(ChannelID p_channel) const;

	int max_reverbs;
	struct ReverbState {

		bool used_in_chunk;
		bool enabled;
		ReverbSW *reverb;
		int frames_idle;
		int32_t *buffer; //reverb is sent here
		ReverbState() {
			enabled = false;
			frames_idle = 0;
			used_in_chunk = false;
		}
	};

	ReverbState *reverb_state;

public:
	virtual ChannelID channel_alloc(RID p_sample);

	virtual void channel_set_volume(ChannelID p_channel, float p_gain);
	virtual void channel_set_pan(ChannelID p_channel, float p_pan, float p_depth = 0, float height = 0); //pan and depth go from -1 to 1
	virtual void channel_set_filter(ChannelID p_channel, FilterType p_type, float p_cutoff, float p_resonance, float p_gain = 1.0);
	virtual void channel_set_chorus(ChannelID p_channel, float p_chorus);
	virtual void channel_set_reverb(ChannelID p_channel, ReverbRoomType p_room_type, float p_reverb);
	virtual void channel_set_mix_rate(ChannelID p_channel, int p_mix_rate);
	virtual void channel_set_positional(ChannelID p_channel, bool p_positional);

	virtual float channel_get_volume(ChannelID p_channel) const;
	virtual float channel_get_pan(ChannelID p_channel) const; //pan and depth go from -1 to 1
	virtual float channel_get_pan_depth(ChannelID p_channel) const; //pan and depth go from -1 to 1
	virtual float channel_get_pan_height(ChannelID p_channel) const; //pan and depth go from -1 to 1
	virtual FilterType channel_get_filter_type(ChannelID p_channel) const;
	virtual float channel_get_filter_cutoff(ChannelID p_channel) const;
	virtual float channel_get_filter_resonance(ChannelID p_channel) const;
	virtual float channel_get_filter_gain(ChannelID p_channel) const;

	virtual float channel_get_chorus(ChannelID p_channel) const;
	virtual ReverbRoomType channel_get_reverb_type(ChannelID p_channel) const;
	virtual float channel_get_reverb(ChannelID p_channel) const;

	virtual int channel_get_mix_rate(ChannelID p_channel) const;
	virtual bool channel_is_positional(ChannelID p_channel) const;

	virtual bool channel_is_valid(ChannelID p_channel) const;

	virtual void channel_free(ChannelID p_channel);

	int mix(int32_t *p_buffer, int p_frames); //return amount of mixsteps
	uint64_t get_step_usecs() const;

	virtual void set_mixer_volume(float p_volume);

	AudioMixerSW(SampleManagerSW *p_sample_manager, int p_desired_latency_ms, int p_mix_rate, MixChannels p_mix_channels, bool p_use_fx = true, InterpolationType p_interp = INTERPOLATION_LINEAR, MixStepCallback p_step_callback = NULL, void *p_callback_udata = NULL);
	~AudioMixerSW();
};

#endif // AUDIO_MIXER_SW_H

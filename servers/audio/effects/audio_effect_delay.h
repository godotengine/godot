#ifndef AUDIOEFFECTECHO_H
#define AUDIOEFFECTECHO_H

#include "servers/audio/audio_effect.h"

class AudioEffectDelay;

class AudioEffectDelayInstance : public AudioEffectInstance {
	GDCLASS(AudioEffectDelayInstance,AudioEffectInstance)
friend class AudioEffectDelay;
	Ref<AudioEffectDelay> base;

	Vector<AudioFrame> ring_buffer;

	unsigned int ring_buffer_pos;
	unsigned int ring_buffer_mask;

	/* feedback buffer */
	Vector<AudioFrame> feedback_buffer;

	unsigned int feedback_buffer_pos;

	AudioFrame h;
	void _process_chunk(const AudioFrame *p_src_frames,AudioFrame *p_dst_frames,int p_frame_count);

public:

	virtual void process(const AudioFrame *p_src_frames,AudioFrame *p_dst_frames,int p_frame_count);

};


class AudioEffectDelay : public AudioEffect {
	GDCLASS(AudioEffectDelay,AudioEffect)

friend class AudioEffectDelayInstance;
	enum {

		MAX_DELAY_MS=3000,
		MAX_TAPS=2
	};

	float dry;

	bool tap_1_active;
	float tap_1_delay_ms;
	float tap_1_level;
	float tap_1_pan;

	bool tap_2_active;
	float tap_2_delay_ms;
	float tap_2_level;
	float tap_2_pan;

	bool feedback_active;
	float feedback_delay_ms;
	float feedback_level;
	float feedback_lowpass;



protected:

	static void _bind_methods();
public:

	void set_dry(float p_dry);
	float get_dry();

	void set_tap1_active(bool p_active);
	bool is_tap1_active() const;

	void set_tap1_delay_ms(float p_delay_ms);
	float get_tap1_delay_ms() const;

	void set_tap1_level_db(float p_level_db);
	float get_tap1_level_db() const;

	void set_tap1_pan(float p_pan);
	float get_tap1_pan() const;

	void set_tap2_active(bool p_active);
	bool is_tap2_active() const;

	void set_tap2_delay_ms(float p_delay_ms);
	float get_tap2_delay_ms() const;

	void set_tap2_level_db(float p_level_db);
	float get_tap2_level_db() const;

	void set_tap2_pan(float p_pan);
	float get_tap2_pan() const;

	void set_feedback_active(bool p_active);
	bool is_feedback_active() const;

	void set_feedback_delay_ms(float p_delay_ms);
	float get_feedback_delay_ms() const;

	void set_feedback_level_db(float p_level_db);
	float get_feedback_level_db() const;

	void set_feedback_lowpass(float p_lowpass);
	float get_feedback_lowpass() const;

	Ref<AudioEffectInstance> instance();

	AudioEffectDelay();
};


#endif // AUDIOEFFECTECHO_H

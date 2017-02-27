#ifndef AUDIOEFFECTSTEREOENHANCE_H
#define AUDIOEFFECTSTEREOENHANCE_H


#include "servers/audio/audio_effect.h"

class AudioEffectStereoEnhance;

class AudioEffectStereoEnhanceInstance : public AudioEffectInstance {
	GDCLASS(AudioEffectStereoEnhanceInstance,AudioEffectInstance)
friend class AudioEffectStereoEnhance;
	Ref<AudioEffectStereoEnhance> base;

	enum {

		MAX_DELAY_MS=50
	};

	float *delay_ringbuff;
	unsigned int ringbuff_pos;
	unsigned int ringbuff_mask;


public:

	virtual void process(const AudioFrame *p_src_frames,AudioFrame *p_dst_frames,int p_frame_count);

	~AudioEffectStereoEnhanceInstance();
};


class AudioEffectStereoEnhance : public AudioEffect {
	GDCLASS(AudioEffectStereoEnhance,AudioEffect)

friend class AudioEffectStereoEnhanceInstance;
	float volume_db;

	float pan_pullout;
	float time_pullout;
	float surround;

protected:

	static void _bind_methods();
public:


	Ref<AudioEffectInstance> instance();

	void set_pan_pullout(float p_amount);
	float get_pan_pullout() const;

	void set_time_pullout(float p_amount);
	float get_time_pullout() const;

	void set_surround(float p_amount);
	float get_surround() const;

	AudioEffectStereoEnhance();
};

#endif // AUDIOEFFECTSTEREOENHANCE_H

#ifndef AUDIO_EFFECT_PHASER_H
#define AUDIO_EFFECT_PHASER_H



#include "servers/audio/audio_effect.h"

class AudioEffectPhaser;

class AudioEffectPhaserInstance : public AudioEffectInstance {
	GDCLASS(AudioEffectPhaserInstance,AudioEffectInstance)
friend class AudioEffectPhaser;
	Ref<AudioEffectPhaser> base;

	float phase;
	AudioFrame h;

	class AllpassDelay{
		float a, h;
	public:

		_ALWAYS_INLINE_ void delay( float d ) {
			a = (1.f - d) / (1.f + d);
		}

		_ALWAYS_INLINE_ float update( float s ){
			float y = s * -a + h;
			h = y * a + s;
			return y;
		}

		AllpassDelay() { a =0; h = 0;}

	};

	AllpassDelay allpass[2][6];
public:

	virtual void process(const AudioFrame *p_src_frames,AudioFrame *p_dst_frames,int p_frame_count);

};


class AudioEffectPhaser : public AudioEffect {
	GDCLASS(AudioEffectPhaser,AudioEffect)

friend class AudioEffectPhaserInstance;
	float range_min;
	float range_max;
	float rate;
	float feedback;
	float depth;

protected:

	static void _bind_methods();
public:


	Ref<AudioEffectInstance> instance();

	void set_range_min_hz(float p_hz);
	float get_range_min_hz() const;

	void set_range_max_hz(float p_hz);
	float get_range_max_hz() const;

	void set_rate_hz(float p_hz);
	float get_rate_hz() const;

	void set_feedback(float p_fbk);
	float get_feedback() const;

	void set_depth(float p_depth);
	float get_depth() const;

	AudioEffectPhaser();
};


#endif // AUDIO_EFFECT_PHASER_H

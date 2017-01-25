#ifndef AUDIOEFFECTAMPLIFY_H
#define AUDIOEFFECTAMPLIFY_H

#include "servers/audio/audio_effect.h"

class AudioEffectAmplify;

class AudioEffectAmplifyInstance : public AudioEffectInstance {
	GDCLASS(AudioEffectAmplifyInstance,AudioEffectInstance)
friend class AudioEffectAmplify;
	Ref<AudioEffectAmplify> base;

	float mix_volume_db;
public:

	virtual void process(const AudioFrame *p_src_frames,AudioFrame *p_dst_frames,int p_frame_count);

};


class AudioEffectAmplify : public AudioEffect {
	GDCLASS(AudioEffectAmplify,AudioEffect)

friend class AudioEffectAmplifyInstance;
	float volume_db;

protected:

	static void _bind_methods();
public:


	Ref<AudioEffectInstance> instance();
	void set_volume_db(float p_volume);
	float get_volume_db() const;

	AudioEffectAmplify();
};

#endif // AUDIOEFFECTAMPLIFY_H

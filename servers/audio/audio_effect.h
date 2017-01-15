#ifndef AUDIOEFFECT_H
#define AUDIOEFFECT_H

#include "audio_frame.h"
#include "resource.h"


class AudioEffectInstance : public Reference {
	GDCLASS(AudioEffectInstance,Reference)

public:

	virtual void process(AudioFrame *p_frames,int p_frame_count)=0;

};


class AudioEffect : public Resource {
	GDCLASS(AudioEffect,Resource)
public:

	virtual Ref<AudioEffectInstance> instance()=0;
	AudioEffect();
};

#endif // AUDIOEFFECT_H

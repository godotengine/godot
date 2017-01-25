#ifndef AUDIO_EFFECT_PITCH_SHIFT_H
#define AUDIO_EFFECT_PITCH_SHIFT_H


#include "servers/audio/audio_effect.h"

class SMBPitchShift {

	enum {
		MAX_FRAME_LENGTH=8192
	};

	float gInFIFO[MAX_FRAME_LENGTH];
	float gOutFIFO[MAX_FRAME_LENGTH];
	float gFFTworksp[2*MAX_FRAME_LENGTH];
	float gLastPhase[MAX_FRAME_LENGTH/2+1];
	float gSumPhase[MAX_FRAME_LENGTH/2+1];
	float gOutputAccum[2*MAX_FRAME_LENGTH];
	float gAnaFreq[MAX_FRAME_LENGTH];
	float gAnaMagn[MAX_FRAME_LENGTH];
	float gSynFreq[MAX_FRAME_LENGTH];
	float gSynMagn[MAX_FRAME_LENGTH];
	long gRover;

	void smbFft(float *fftBuffer, long fftFrameSize, long sign);
public:
	void PitchShift(float pitchShift, long numSampsToProcess, long fftFrameSize, long osamp, float sampleRate, float *indata, float *outdata, int stride);

	SMBPitchShift() {
		gRover=0;
		memset(gInFIFO, 0, MAX_FRAME_LENGTH*sizeof(float));
		memset(gOutFIFO, 0, MAX_FRAME_LENGTH*sizeof(float));
		memset(gFFTworksp, 0, 2*MAX_FRAME_LENGTH*sizeof(float));
		memset(gLastPhase, 0, (MAX_FRAME_LENGTH/2+1)*sizeof(float));
		memset(gSumPhase, 0, (MAX_FRAME_LENGTH/2+1)*sizeof(float));
		memset(gOutputAccum, 0, 2*MAX_FRAME_LENGTH*sizeof(float));
		memset(gAnaFreq, 0, MAX_FRAME_LENGTH*sizeof(float));
		memset(gAnaMagn, 0, MAX_FRAME_LENGTH*sizeof(float));
	}


};


class AudioEffectPitchShift;

class AudioEffectPitchShiftInstance : public AudioEffectInstance {
	GDCLASS(AudioEffectPitchShiftInstance,AudioEffectInstance)
friend class AudioEffectPitchShift;
	Ref<AudioEffectPitchShift> base;

	SMBPitchShift shift_l;
	SMBPitchShift shift_r;


public:

	virtual void process(const AudioFrame *p_src_frames,AudioFrame *p_dst_frames,int p_frame_count);

};


class AudioEffectPitchShift : public AudioEffect {
	GDCLASS(AudioEffectPitchShift,AudioEffect)

friend class AudioEffectPitchShiftInstance;

	float pitch_scale;
	int window_size;
	float wet;
	float dry;
	bool filter;

protected:

	static void _bind_methods();
public:


	Ref<AudioEffectInstance> instance();

	void set_pitch_scale(float p_adjust);
	float get_pitch_scale() const;

	AudioEffectPitchShift();
};


#endif // AUDIO_EFFECT_PITCH_SHIFT_H

#ifndef AUDIOEFFECTFILTER_H
#define AUDIOEFFECTFILTER_H

#include "servers/audio/audio_effect.h"
#include "servers/audio/audio_filter_sw.h"

class AudioEffectFilter;

class AudioEffectFilterInstance : public AudioEffectInstance {
	GDCLASS(AudioEffectFilterInstance,AudioEffectInstance)
friend class AudioEffectFilter;

	Ref<AudioEffectFilter> base;

	AudioFilterSW filter;
	AudioFilterSW::Processor filter_process[2][4];

	template<int S>
	void _process_filter(const AudioFrame *p_src_frames,AudioFrame *p_dst_frames,int p_frame_count);
public:

	virtual void process(const AudioFrame *p_src_frames,AudioFrame *p_dst_frames,int p_frame_count);

	AudioEffectFilterInstance();
};


class AudioEffectFilter : public AudioEffect {
	GDCLASS(AudioEffectFilter,AudioEffect)
public:

	enum FilterDB {
	     FILTER_6DB,
	     FILTER_12DB,
	     FILTER_18DB,
	     FILTER_24DB,
	};
	friend class AudioEffectFilterInstance;

	AudioFilterSW::Mode mode;
	float cutoff;
	float resonance;
	float gain;
	FilterDB db;


protected:


	static void _bind_methods();
public:

	void set_cutoff(float p_freq);
	float get_cutoff() const;

	void set_resonance(float p_amount);
	float get_resonance() const;

	void set_gain(float p_amount);
	float get_gain() const;

	void set_db(FilterDB p_db);
	FilterDB get_db() const;

	Ref<AudioEffectInstance> instance();

	AudioEffectFilter(AudioFilterSW::Mode p_mode=AudioFilterSW::LOWPASS);
};

VARIANT_ENUM_CAST(AudioEffectFilter::FilterDB)

class AudioEffectLowPass : public AudioEffectFilter {
	GDCLASS(AudioEffectLowPass,AudioEffectFilter)
public:

	AudioEffectLowPass() : AudioEffectFilter(AudioFilterSW::LOWPASS) {}
};

class AudioEffectHighPass : public AudioEffectFilter {
	GDCLASS(AudioEffectHighPass,AudioEffectFilter)
public:

	AudioEffectHighPass() : AudioEffectFilter(AudioFilterSW::HIGHPASS) {}
};

class AudioEffectBandPass : public AudioEffectFilter {
	GDCLASS(AudioEffectBandPass,AudioEffectFilter)
public:

	AudioEffectBandPass() : AudioEffectFilter(AudioFilterSW::BANDPASS) {}
};

class AudioEffectNotchPass : public AudioEffectFilter {
	GDCLASS(AudioEffectNotchPass,AudioEffectFilter)
public:

	AudioEffectNotchPass() : AudioEffectFilter(AudioFilterSW::NOTCH) {}
};

class AudioEffectBandLimit : public AudioEffectFilter {
	GDCLASS(AudioEffectBandLimit,AudioEffectFilter)
public:

	AudioEffectBandLimit() : AudioEffectFilter(AudioFilterSW::BANDLIMIT) {}
};


class AudioEffectLowShelf : public AudioEffectFilter {
	GDCLASS(AudioEffectLowShelf,AudioEffectFilter)
public:

	AudioEffectLowShelf() : AudioEffectFilter(AudioFilterSW::LOWSHELF) {}
};


class AudioEffectHighShelf : public AudioEffectFilter {
	GDCLASS(AudioEffectHighShelf,AudioEffectFilter)
public:

	AudioEffectHighShelf() : AudioEffectFilter(AudioFilterSW::HIGHSHELF) {}
};



#endif // AUDIOEFFECTFILTER_H

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

class AudioEffectLowPassFilter : public AudioEffectFilter {
	GDCLASS(AudioEffectLowPassFilter,AudioEffectFilter)
public:

	AudioEffectLowPassFilter() : AudioEffectFilter(AudioFilterSW::LOWPASS) {}
};

class AudioEffectHighPassFilter : public AudioEffectFilter {
	GDCLASS(AudioEffectHighPassFilter,AudioEffectFilter)
public:

	AudioEffectHighPassFilter() : AudioEffectFilter(AudioFilterSW::HIGHPASS) {}
};

class AudioEffectBandPassFilter : public AudioEffectFilter {
	GDCLASS(AudioEffectBandPassFilter,AudioEffectFilter)
public:

	AudioEffectBandPassFilter() : AudioEffectFilter(AudioFilterSW::BANDPASS) {}
};

class AudioEffectNotchFilter : public AudioEffectFilter {
	GDCLASS(AudioEffectNotchFilter,AudioEffectFilter)
public:

	AudioEffectNotchFilter() : AudioEffectFilter(AudioFilterSW::NOTCH) {}
};

class AudioEffectBandLimitFilter : public AudioEffectFilter {
	GDCLASS(AudioEffectBandLimitFilter,AudioEffectFilter)
public:

	AudioEffectBandLimitFilter() : AudioEffectFilter(AudioFilterSW::BANDLIMIT) {}
};


class AudioEffectLowShelfFilter : public AudioEffectFilter {
	GDCLASS(AudioEffectLowShelfFilter,AudioEffectFilter)
public:

	AudioEffectLowShelfFilter() : AudioEffectFilter(AudioFilterSW::LOWSHELF) {}
};


class AudioEffectHighShelfFilter : public AudioEffectFilter {
	GDCLASS(AudioEffectHighShelfFilter,AudioEffectFilter)
public:

	AudioEffectHighShelfFilter() : AudioEffectFilter(AudioFilterSW::HIGHSHELF) {}
};



#endif // AUDIOEFFECTFILTER_H

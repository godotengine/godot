#include "audio_effect_filter.h"
#include "servers/audio_server.h"

template<int S>
void AudioEffectFilterInstance::_process_filter(const AudioFrame *p_src_frames,AudioFrame *p_dst_frames,int p_frame_count) {

	for(int i=0;i<p_frame_count;i++) {
		float f = p_src_frames[i].l;
		filter_process[0][0].process_one(f);
		if (S>1)
			filter_process[0][1].process_one(f);
		if (S>2)
			filter_process[0][2].process_one(f);
		if (S>3)
			filter_process[0][3].process_one(f);

		p_dst_frames[i].l=f;
	}

	for(int i=0;i<p_frame_count;i++) {
		float f = p_src_frames[i].r;
		filter_process[1][0].process_one(f);
		if (S>1)
			filter_process[1][1].process_one(f);
		if (S>2)
			filter_process[1][2].process_one(f);
		if (S>3)
			filter_process[1][3].process_one(f);

		p_dst_frames[i].r=f;
	}

}

void AudioEffectFilterInstance::process(const AudioFrame *p_src_frames,AudioFrame *p_dst_frames,int p_frame_count) {

	filter.set_cutoff(base->cutoff);
	filter.set_gain(base->gain);
	filter.set_resonance(base->resonance);
	filter.set_mode(base->mode);
	int stages = int(base->db)+1;
	filter.set_stages(stages);
	filter.set_sampling_rate(AudioServer::get_singleton()->get_mix_rate());

	for(int i=0;i<2;i++) {
		for(int j=0;j<4;j++) {
			filter_process[i][j].update_coeffs();
		}
	}


	if (stages==1) {
		_process_filter<1>(p_src_frames,p_dst_frames,p_frame_count);
	} else if (stages==2) {
		_process_filter<2>(p_src_frames,p_dst_frames,p_frame_count);
	} else if (stages==3) {
		_process_filter<3>(p_src_frames,p_dst_frames,p_frame_count);
	} else if (stages==4) {
		_process_filter<4>(p_src_frames,p_dst_frames,p_frame_count);
	}

}


AudioEffectFilterInstance::AudioEffectFilterInstance() {

	for(int i=0;i<2;i++) {
		for(int j=0;j<4;j++) {
			filter_process[i][j].set_filter(&filter);
		}
	}

}


Ref<AudioEffectInstance> AudioEffectFilter::instance() {
	Ref<AudioEffectFilterInstance> ins;
	ins.instance();
	ins->base=Ref<AudioEffectFilter>(this);

	return ins;
}

void AudioEffectFilter::set_cutoff(float p_freq) {

	cutoff=p_freq;
}

float AudioEffectFilter::get_cutoff() const{

	return cutoff;
}

void AudioEffectFilter::set_resonance(float p_amount){

	resonance=p_amount;
}
float AudioEffectFilter::get_resonance() const{

	return resonance;
}

void AudioEffectFilter::set_gain(float p_amount){

	gain=p_amount;
}
float AudioEffectFilter::get_gain() const {

	return gain;
}



void AudioEffectFilter::set_db(FilterDB p_db) {
	db=p_db;
}

AudioEffectFilter::FilterDB AudioEffectFilter::get_db() const {

	return db;
}

void AudioEffectFilter::_bind_methods() {

	ClassDB::bind_method(_MD("set_cutoff","freq"),&AudioEffectFilter::set_cutoff);
	ClassDB::bind_method(_MD("get_cutoff"),&AudioEffectFilter::get_cutoff);

	ClassDB::bind_method(_MD("set_resonance","amount"),&AudioEffectFilter::set_resonance);
	ClassDB::bind_method(_MD("get_resonance"),&AudioEffectFilter::get_resonance);

	ClassDB::bind_method(_MD("set_gain","amount"),&AudioEffectFilter::set_gain);
	ClassDB::bind_method(_MD("get_gain"),&AudioEffectFilter::get_gain);

	ClassDB::bind_method(_MD("set_db","amount"),&AudioEffectFilter::set_db);
	ClassDB::bind_method(_MD("get_db"),&AudioEffectFilter::get_db);

	ADD_PROPERTY(PropertyInfo(Variant::REAL,"cutoff_hz",PROPERTY_HINT_RANGE,"1,40000,0.1"),_SCS("set_cutoff"),_SCS("get_cutoff"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"resonance",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_resonance"),_SCS("get_resonance"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"gain",PROPERTY_HINT_RANGE,"0,4,0.01"),_SCS("set_gain"),_SCS("get_gain"));
	ADD_PROPERTY(PropertyInfo(Variant::INT,"dB",PROPERTY_HINT_ENUM,"6db,12db,18db,24db"),_SCS("set_db"),_SCS("get_db"));
}

AudioEffectFilter::AudioEffectFilter(AudioFilterSW::Mode p_mode)
{

	mode=p_mode;
	cutoff=2000;
	resonance=0.5;
	gain=1.0;
	db=FILTER_6DB;
}

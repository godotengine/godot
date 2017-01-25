#include "audio_effect_reverb.h"
#include "servers/audio_server.h"
void AudioEffectReverbInstance::process(const AudioFrame *p_src_frames,AudioFrame *p_dst_frames,int p_frame_count) {

	for(int i=0;i<2;i++) {
		Reverb &r=reverb[i];

		r.set_predelay( base->predelay);
		r.set_predelay_feedback( base->predelay_fb );
		r.set_highpass( base->hpf );
		r.set_room_size( base->room_size );
		r.set_damp( base->damping );
		r.set_extra_spread( base->spread );
		r.set_wet( base->wet );
		r.set_dry( base->dry );
	}

	int todo = p_frame_count;
	int offset=0;

	while(todo) {

		int to_mix = MIN(todo,Reverb::INPUT_BUFFER_MAX_SIZE);

		for(int j=0;j<to_mix;j++) {
			tmp_src[j]=p_src_frames[offset+j].l;
		}

		reverb[0].process(tmp_src,tmp_dst,to_mix);

		for(int j=0;j<to_mix;j++) {
			p_dst_frames[offset+j].l=tmp_dst[j];
			tmp_src[j]=p_src_frames[offset+j].r;
		}

		reverb[1].process(tmp_src,tmp_dst,to_mix);

		for(int j=0;j<to_mix;j++) {
			p_dst_frames[offset+j].r=tmp_dst[j];
		}

		offset+=to_mix;
		todo-=to_mix;
	}
}

AudioEffectReverbInstance::AudioEffectReverbInstance() {

	reverb[0].set_mix_rate( AudioServer::get_singleton()->get_mix_rate() );
	reverb[0].set_extra_spread_base(0);
	reverb[1].set_mix_rate( AudioServer::get_singleton()->get_mix_rate() );
	reverb[1].set_extra_spread_base(0.000521); //for stereo effect

}

Ref<AudioEffectInstance> AudioEffectReverb::instance() {
	Ref<AudioEffectReverbInstance> ins;
	ins.instance();
	ins->base=Ref<AudioEffectReverb>(this);
	return ins;
}

void AudioEffectReverb::set_predelay_msec(float p_msec) {

	predelay=p_msec;
}

void AudioEffectReverb::set_predelay_feedback(float p_feedback){

	predelay_fb=p_feedback;
}
void AudioEffectReverb::set_room_size(float p_size){

	room_size=p_size;
}
void AudioEffectReverb::set_damping(float p_damping){

	damping=p_damping;
}
void AudioEffectReverb::set_spread(float p_spread){

	spread=p_spread;
}

void AudioEffectReverb::set_dry(float p_dry){

	dry=p_dry;
}
void AudioEffectReverb::set_wet(float p_wet){

	wet=p_wet;
}
void AudioEffectReverb::set_hpf(float p_hpf) {

	hpf=p_hpf;
}

float AudioEffectReverb::get_predelay_msec() const {

	return predelay;
}
float AudioEffectReverb::get_predelay_feedback() const {

	return predelay_fb;
}
float AudioEffectReverb::get_room_size() const {

	return room_size;
}
float AudioEffectReverb::get_damping() const {

	return damping;
}
float AudioEffectReverb::get_spread() const {

	return spread;
}
float AudioEffectReverb::get_dry() const {

	return dry;
}
float AudioEffectReverb::get_wet() const {

	return wet;
}
float AudioEffectReverb::get_hpf() const {

	return hpf;
}


void AudioEffectReverb::_bind_methods() {


	ClassDB::bind_method(_MD("set_predelay_msec","msec"),&AudioEffectReverb::set_predelay_msec);
	ClassDB::bind_method(_MD("get_predelay_msec"),&AudioEffectReverb::get_predelay_msec);

	ClassDB::bind_method(_MD("set_predelay_feedback","feedback"),&AudioEffectReverb::set_predelay_feedback);
	ClassDB::bind_method(_MD("get_predelay_feedback"),&AudioEffectReverb::get_predelay_feedback);

	ClassDB::bind_method(_MD("set_room_size","size"),&AudioEffectReverb::set_room_size);
	ClassDB::bind_method(_MD("get_room_size"),&AudioEffectReverb::get_room_size);

	ClassDB::bind_method(_MD("set_damping","amount"),&AudioEffectReverb::set_damping);
	ClassDB::bind_method(_MD("get_damping"),&AudioEffectReverb::get_damping);

	ClassDB::bind_method(_MD("set_spread","amount"),&AudioEffectReverb::set_spread);
	ClassDB::bind_method(_MD("get_spread"),&AudioEffectReverb::get_spread);

	ClassDB::bind_method(_MD("set_dry","amount"),&AudioEffectReverb::set_dry);
	ClassDB::bind_method(_MD("get_dry"),&AudioEffectReverb::get_dry);

	ClassDB::bind_method(_MD("set_wet","amount"),&AudioEffectReverb::set_wet);
	ClassDB::bind_method(_MD("get_wet"),&AudioEffectReverb::get_wet);

	ClassDB::bind_method(_MD("set_hpf","amount"),&AudioEffectReverb::set_hpf);
	ClassDB::bind_method(_MD("get_hpf"),&AudioEffectReverb::get_hpf);


	ADD_GROUP("Predelay","predelay_");
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"predelay_msec",PROPERTY_HINT_RANGE,"20,500,1"),_SCS("set_predelay_msec"),_SCS("get_predelay_msec"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"predelay_feedback",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_predelay_msec"),_SCS("get_predelay_msec"));
	ADD_GROUP("","");
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"room_size",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_room_size"),_SCS("get_room_size"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"damping",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_damping"),_SCS("get_damping"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"spread",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_spread"),_SCS("get_spread"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"hipass",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_hpf"),_SCS("get_hpf"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"dry",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_dry"),_SCS("get_dry"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"wet",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_wet"),_SCS("get_wet"));
}

AudioEffectReverb::AudioEffectReverb() {
	predelay=150;
	predelay_fb=0.4;
	hpf=0;
	room_size=0.8;
	damping=0.5;
	spread=1.0;
	dry=1.0;
	wet=0.5;

}

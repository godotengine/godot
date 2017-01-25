#include "audio_effect_phaser.h"
#include "servers/audio_server.h"
#include "math_funcs.h"

void AudioEffectPhaserInstance::process(const AudioFrame *p_src_frames,AudioFrame *p_dst_frames,int p_frame_count) {

	float sampling_rate = AudioServer::get_singleton()->get_mix_rate();

	float dmin = base->range_min / (sampling_rate/2.0);
	float dmax = base->range_max / (sampling_rate/2.0);

	float increment = 2.f * Math_PI * (base->rate / sampling_rate);

	for(int i=0;i<p_frame_count;i++) {

		phase += increment;

		while ( phase >= Math_PI * 2.f ) {
			phase -= Math_PI * 2.f;
		}

		float d  = dmin + (dmax-dmin) * ((sin( phase ) + 1.f)/2.f);


		//update filter coeffs
		for( int j=0; j<6; j++ ) {
			allpass[0][j].delay( d );
			allpass[1][j].delay( d );
		}

		//calculate output
		float y = 	allpass[0][0].update(
					allpass[0][1].update(
					allpass[0][2].update(
					allpass[0][3].update(
					allpass[0][4].update(
					allpass[0][5].update( p_src_frames[i].l + h.l * base->feedback ))))));
		h.l=y;

		p_dst_frames[i].l = p_src_frames[i].l + y * base->depth;

		y = 	allpass[1][0].update(
					allpass[1][1].update(
					allpass[1][2].update(
					allpass[1][3].update(
					allpass[1][4].update(
					allpass[1][5].update( p_src_frames[i].r + h.r * base->feedback ))))));
		h.r=y;

		p_dst_frames[i].r = p_src_frames[i].r + y * base->depth;


	}

}


Ref<AudioEffectInstance> AudioEffectPhaser::instance() {
	Ref<AudioEffectPhaserInstance> ins;
	ins.instance();
	ins->base=Ref<AudioEffectPhaser>(this);
	ins->phase=0;
	ins->h=AudioFrame(0,0);

	return ins;
}


void AudioEffectPhaser::set_range_min_hz(float p_hz) {

	range_min=p_hz;
}

float AudioEffectPhaser::get_range_min_hz() const{

	return range_min;
}

void AudioEffectPhaser::set_range_max_hz(float p_hz){

	range_max=p_hz;
}
float AudioEffectPhaser::get_range_max_hz() const{

	return range_max;
}

void AudioEffectPhaser::set_rate_hz(float p_hz){

	rate=p_hz;
}
float AudioEffectPhaser::get_rate_hz() const{

	return rate;
}

void AudioEffectPhaser::set_feedback(float p_fbk){

	feedback=p_fbk;
}
float AudioEffectPhaser::get_feedback() const{

	return feedback;
}

void AudioEffectPhaser::set_depth(float p_depth) {

	depth=p_depth;
}

float AudioEffectPhaser::get_depth() const {

	return depth;
}

void AudioEffectPhaser::_bind_methods() {

	ClassDB::bind_method(_MD("set_range_min_hz","hz"),&AudioEffectPhaser::set_range_min_hz);
	ClassDB::bind_method(_MD("get_range_min_hz"),&AudioEffectPhaser::get_range_min_hz);

	ClassDB::bind_method(_MD("set_range_max_hz","hz"),&AudioEffectPhaser::set_range_max_hz);
	ClassDB::bind_method(_MD("get_range_max_hz"),&AudioEffectPhaser::get_range_max_hz);

	ClassDB::bind_method(_MD("set_rate_hz","hz"),&AudioEffectPhaser::set_rate_hz);
	ClassDB::bind_method(_MD("get_rate_hz"),&AudioEffectPhaser::get_rate_hz);

	ClassDB::bind_method(_MD("set_feedback","fbk"),&AudioEffectPhaser::set_feedback);
	ClassDB::bind_method(_MD("get_feedback"),&AudioEffectPhaser::get_feedback);

	ClassDB::bind_method(_MD("set_depth","depth"),&AudioEffectPhaser::set_depth);
	ClassDB::bind_method(_MD("get_depth"),&AudioEffectPhaser::get_depth);

	ADD_PROPERTY(PropertyInfo(Variant::REAL,"range_min_hz",PROPERTY_HINT_RANGE,"10,10000"),_SCS("set_range_min_hz"),_SCS("get_range_min_hz"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"range_max_hz",PROPERTY_HINT_RANGE,"10,10000"),_SCS("set_range_max_hz"),_SCS("get_range_max_hz"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"rate_hz",PROPERTY_HINT_RANGE,"0.01,20"),_SCS("set_rate_hz"),_SCS("get_rate_hz"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"feedback",PROPERTY_HINT_RANGE,"0.1,0.9,0.1"),_SCS("set_feedback"),_SCS("get_feedback"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"depth",PROPERTY_HINT_RANGE,"0.1,4,0.1"),_SCS("set_depth"),_SCS("get_depth"));

}

AudioEffectPhaser::AudioEffectPhaser()
{
	range_min=440;
	range_max=1600;
	rate=0.5;
	feedback=0.7;
	depth=1;
}

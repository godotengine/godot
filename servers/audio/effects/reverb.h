//
// C++ Interface: reverb
//
// Description:
//
//
// Author: Juan Linietsky <reduzio@gmail.com>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef REVERB_H
#define REVERB_H

#include "typedefs.h"
#include "os/memory.h"
#include "audio_frame.h"

class Reverb {
public:
	enum {
		INPUT_BUFFER_MAX_SIZE=1024,

	};
private:
	enum {

		MAX_COMBS=8,
		MAX_ALLPASS=4,
		MAX_ECHO_MS=500

	};



	static const float comb_tunings[MAX_COMBS];
	static const float allpass_tunings[MAX_ALLPASS];

	struct Comb {

		int size;
		float *buffer;
		float feedback;
		float damp; //lowpass
		float damp_h; //history
		int pos;
		int extra_spread_frames;

		Comb() { size=0; buffer=0; feedback=0; damp_h=0; pos=0; }
	};

	struct AllPass {

		int size;
		float *buffer;
		int pos;
		int extra_spread_frames;
		AllPass() { size=0; buffer=0; pos=0; }
	};

	Comb comb[MAX_COMBS];
	AllPass allpass[MAX_ALLPASS];
	float *input_buffer;
	float *echo_buffer;
	int echo_buffer_size;
	int echo_buffer_pos;

	float hpf_h1,hpf_h2;


	struct Parameters {

		float room_size;
		float damp;
		float wet;
		float dry;
		float mix_rate;
		float extra_spread_base;
		float extra_spread;
		float predelay;
		float predelay_fb;
		float hpf;
	} params;

	void configure_buffers();
	void update_parameters();
	void clear_buffers();
public:

	void set_room_size(float p_size);
	void set_damp(float p_damp);
	void set_wet(float p_wet);
	void set_dry(float p_dry);
	void set_predelay(float p_predelay); // in ms
	void set_predelay_feedback(float p_predelay_fb); // in ms
	void set_highpass(float p_frq);
	void set_mix_rate(float p_mix_rate);
	void set_extra_spread(float p_spread);
	void set_extra_spread_base(float p_sec);

	void process(float *p_src,float *p_dst,int p_frames);

	Reverb();

	~Reverb();

};



#endif

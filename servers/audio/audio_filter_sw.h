/*************************************************************************/
/*  audio_filter_sw.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifndef AUDIO_FILTER_SW_H
#define AUDIO_FILTER_SW_H

#include "math_funcs.h"

class AudioFilterSW {
public:
	struct Coeffs {

		float a1, a2;
		float b0, b1, b2;

		//bool operator==(const Coeffs &p_rv) { return (FLOATS_EQ(a1,p_rv.a1) && FLOATS_EQ(a2,p_rv.a2) && FLOATS_EQ(b1,p_rv.b1) && FLOATS_EQ(b2,p_rv.b2) && FLOATS_EQ(b0,p_rv.b0) ); }
		Coeffs() { a1 = a2 = b0 = b1 = b2 = 0.0; }
	};

	enum Mode {
		BANDPASS,
		HIGHPASS,
		LOWPASS,
		NOTCH,
		PEAK,
		BANDLIMIT,
		LOWSHELF,
		HIGHSHELF

	};

	class Processor { // simple filter processor

		AudioFilterSW *filter;
		Coeffs coeffs;
		float ha1, ha2, hb1, hb2; //history
	public:
		void set_filter(AudioFilterSW *p_filter);
		void process(float *p_samples, int p_amount, int p_stride = 1);
		void update_coeffs();
		_ALWAYS_INLINE_ void process_one(float &p_sample);

		Processor();
	};

private:
	float cutoff;
	float resonance;
	float gain;
	float sampling_rate;
	int stages;
	Mode mode;

public:
	float get_response(float p_freq, Coeffs *p_coeffs);

	void set_mode(Mode p_mode);
	void set_cutoff(float p_cutoff);
	void set_resonance(float p_resonance);
	void set_gain(float p_gain);
	void set_sampling_rate(float p_srate);
	void set_stages(int p_stages); //adjust for multiple stages

	void prepare_coefficients(Coeffs *p_coeffs);

	AudioFilterSW();
};

/* inline methods */

void AudioFilterSW::Processor::process_one(float &p_val) {

	float pre = p_val;
	p_val = (p_val * coeffs.b0 + hb1 * coeffs.b1 + hb2 * coeffs.b2 + ha1 * coeffs.a1 + ha2 * coeffs.a2);
	ha2 = ha1;
	hb2 = hb1;
	hb1 = pre;
	ha1 = p_val;
}

#endif // AUDIO_FILTER_SW_H

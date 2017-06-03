/*************************************************************************/
/*  audio_filter_sw.cpp                                                  */
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
#include "audio_filter_sw.h"

void AudioFilterSW::set_mode(Mode p_mode) {

	mode = p_mode;
}
void AudioFilterSW::set_cutoff(float p_cutoff) {

	cutoff = p_cutoff;
}
void AudioFilterSW::set_resonance(float p_resonance) {

	resonance = p_resonance;
}

void AudioFilterSW::set_gain(float p_gain) {

	gain = p_gain;
}

void AudioFilterSW::set_sampling_rate(float p_srate) {

	sampling_rate = p_srate;
}

void AudioFilterSW::prepare_coefficients(Coeffs *p_coeffs) {

	int sr_limit = (sampling_rate / 2) + 512;

	double final_cutoff = (cutoff > sr_limit) ? sr_limit : cutoff;
	if (final_cutoff < 1) //avoid crapness
		final_cutoff = 1; //don't allow less than this

	double omega = 2.0 * Math_PI * final_cutoff / sampling_rate;

	double sin_v = Math::sin(omega);
	double cos_v = Math::cos(omega);

	double Q = resonance;
	if (Q <= 0.0) {
		Q = 0.0001;
	}

	if (mode == BANDPASS)
		Q *= 2.0;
	else if (mode == PEAK)
		Q *= 3.0;

	double tmpgain = gain;

	if (tmpgain < 0.001)
		tmpgain = 0.001;

	if (stages > 1) {

		Q = (Q > 1.0 ? Math::pow(Q, 1.0 / stages) : Q);
		tmpgain = Math::pow(tmpgain, 1.0 / (stages + 1));
	}
	double alpha = sin_v / (2 * Q);

	double a0 = 1.0 + alpha;

	switch (mode) {

		case LOWPASS: {

			p_coeffs->b0 = (1.0 - cos_v) / 2.0;
			p_coeffs->b1 = 1.0 - cos_v;
			p_coeffs->b2 = (1.0 - cos_v) / 2.0;
			p_coeffs->a1 = -2.0 * cos_v;
			p_coeffs->a2 = 1.0 - alpha;
		} break;

		case HIGHPASS: {

			p_coeffs->b0 = (1.0 + cos_v) / 2.0;
			p_coeffs->b1 = -(1.0 + cos_v);
			p_coeffs->b2 = (1.0 + cos_v) / 2.0;
			p_coeffs->a1 = -2.0 * cos_v;
			p_coeffs->a2 = 1.0 - alpha;
		} break;

		case BANDPASS: {

			p_coeffs->b0 = alpha * sqrt(Q + 1);
			p_coeffs->b1 = 0.0;
			p_coeffs->b2 = -alpha * sqrt(Q + 1);
			p_coeffs->a1 = -2.0 * cos_v;
			p_coeffs->a2 = 1.0 - alpha;
		} break;

		case NOTCH: {

			p_coeffs->b0 = 1.0;
			p_coeffs->b1 = -2.0 * cos_v;
			p_coeffs->b2 = 1.0;
			p_coeffs->a1 = -2.0 * cos_v;
			p_coeffs->a2 = 1.0 - alpha;
		} break;
		case PEAK: {
			p_coeffs->b0 = (1.0 + alpha * tmpgain);
			p_coeffs->b1 = (-2.0 * cos_v);
			p_coeffs->b2 = (1.0 - alpha * tmpgain);
			p_coeffs->a1 = -2 * cos_v;
			p_coeffs->a2 = (1 - alpha / tmpgain);
		} break;
		case BANDLIMIT: {
			//this one is extra tricky
			double hicutoff = resonance;
			double centercutoff = (cutoff + resonance) / 2.0;
			double bandwidth = (Math::log(centercutoff) - Math::log(hicutoff)) / Math::log((double)2);
			omega = 2.0 * Math_PI * centercutoff / sampling_rate;
			alpha = Math::sin(omega) * Math::sinh(Math::log((double)2) / 2 * bandwidth * omega / Math::sin(omega));
			a0 = 1 + alpha;

			p_coeffs->b0 = alpha;
			p_coeffs->b1 = 0;
			p_coeffs->b2 = -alpha;
			p_coeffs->a1 = -2 * Math::cos(omega);
			p_coeffs->a2 = 1 - alpha;

		} break;
		case LOWSHELF: {

			double tmpq = Math::sqrt(Q);
			if (tmpq <= 0)
				tmpq = 0.001;
			alpha = sin_v / (2 * tmpq);
			double beta = Math::sqrt(tmpgain) / tmpq;

			a0 = (tmpgain + 1.0) + (tmpgain - 1.0) * cos_v + beta * sin_v;
			p_coeffs->b0 = tmpgain * ((tmpgain + 1.0) - (tmpgain - 1.0) * cos_v + beta * sin_v);
			p_coeffs->b1 = 2.0 * tmpgain * ((tmpgain - 1.0) - (tmpgain + 1.0) * cos_v);
			p_coeffs->b2 = tmpgain * ((tmpgain + 1.0) - (tmpgain - 1.0) * cos_v - beta * sin_v);
			p_coeffs->a1 = -2.0 * ((tmpgain - 1.0) + (tmpgain + 1.0) * cos_v);
			p_coeffs->a2 = ((tmpgain + 1.0) + (tmpgain - 1.0) * cos_v - beta * sin_v);

		} break;
		case HIGHSHELF: {
			double tmpq = Math::sqrt(Q);
			if (tmpq <= 0)
				tmpq = 0.001;
			alpha = sin_v / (2 * tmpq);
			double beta = Math::sqrt(tmpgain) / tmpq;

			a0 = (tmpgain + 1.0) - (tmpgain - 1.0) * cos_v + beta * sin_v;
			p_coeffs->b0 = tmpgain * ((tmpgain + 1.0) + (tmpgain - 1.0) * cos_v + beta * sin_v);
			p_coeffs->b1 = -2.0 * tmpgain * ((tmpgain - 1.0) + (tmpgain + 1.0) * cos_v);
			p_coeffs->b2 = tmpgain * ((tmpgain + 1.0) + (tmpgain - 1.0) * cos_v - beta * sin_v);
			p_coeffs->a1 = 2.0 * ((tmpgain - 1.0) - (tmpgain + 1.0) * cos_v);
			p_coeffs->a2 = ((tmpgain + 1.0) - (tmpgain - 1.0) * cos_v - beta * sin_v);

		} break;
	};

	p_coeffs->b0 /= a0;
	p_coeffs->b1 /= a0;
	p_coeffs->b2 /= a0;
	p_coeffs->a1 /= 0.0 - a0;
	p_coeffs->a2 /= 0.0 - a0;

	//undenormalise
	/*    p_coeffs->b0=undenormalise(p_coeffs->b0);
    p_coeffs->b1=undenormalise(p_coeffs->b1);
    p_coeffs->b2=undenormalise(p_coeffs->b2);
    p_coeffs->a1=undenormalise(p_coeffs->a1);
    p_coeffs->a2=undenormalise(p_coeffs->a2);*/
}

void AudioFilterSW::set_stages(int p_stages) { //adjust for multiple stages

	stages = p_stages;
}

/* Fouriertransform kernel to obtain response */

float AudioFilterSW::get_response(float p_freq, Coeffs *p_coeffs) {

	float freq = p_freq / sampling_rate * Math_PI * 2.0f;

	float cx = p_coeffs->b0, cy = 0.0;

	cx += cos(freq) * p_coeffs->b1;
	cy -= sin(freq) * p_coeffs->b1;
	cx += cos(2 * freq) * p_coeffs->b2;
	cy -= sin(2 * freq) * p_coeffs->b2;

	float H = cx * cx + cy * cy;
	cx = 1.0;
	cy = 0.0;

	cx -= cos(freq) * p_coeffs->a1;
	cy += sin(freq) * p_coeffs->a1;
	cx -= cos(2 * freq) * p_coeffs->a2;
	cy += sin(2 * freq) * p_coeffs->a2;

	H = H / (cx * cx + cy * cy);
	return H;
}

AudioFilterSW::AudioFilterSW() {

	sampling_rate = 44100;
	resonance = 0.5;
	cutoff = 5000;
	gain = 1.0;
	mode = LOWPASS;
	stages = 1;
}

AudioFilterSW::Processor::Processor() {

	set_filter(NULL);
}

void AudioFilterSW::Processor::set_filter(AudioFilterSW *p_filter) {

	ha1 = ha2 = hb1 = hb2 = 0;
	filter = p_filter;
}

void AudioFilterSW::Processor::update_coeffs() {

	if (!filter)
		return;

	filter->prepare_coefficients(&coeffs);
}

void AudioFilterSW::Processor::process(float *p_samples, int p_amount, int p_stride) {

	if (!filter)
		return;

	for (int i = 0; i < p_amount; i++) {

		process_one(*p_samples);
		p_samples += p_stride;
	}
}

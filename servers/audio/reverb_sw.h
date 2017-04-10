/*************************************************************************/
/*  reverb_sw.h                                                          */
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
#ifndef REVERB_SW_H
#define REVERB_SW_H

#include "os/memory.h"
#include "typedefs.h"

class ReverbParamsSW;

class ReverbSW {
public:
	enum ReverbMode {
		REVERB_MODE_ROOM,
		REVERB_MODE_STUDIO_SMALL,
		REVERB_MODE_STUDIO_MEDIUM,
		REVERB_MODE_STUDIO_LARGE,
		REVERB_MODE_HALL,
		REVERB_MODE_SPACE_ECHO,
		REVERB_MODE_ECHO,
		REVERB_MODE_DELAY,
		REVERB_MODE_HALF_ECHO
	};

private:
	struct State {
		int lwl;
		int lwr;
		int rwl;
		int rwr;
		unsigned int Offset;
		void reset() {
			lwl = 0;
			lwr = 0;
			rwl = 0;
			rwr = 0;
			Offset = 0;
		}
		State() { reset(); }
	} state;

	ReverbParamsSW *current_params;

	int *reverb_buffer;
	unsigned int reverb_buffer_size;
	ReverbMode mode;
	int mix_rate;

	void adjust_current_params();

public:
	void set_mode(ReverbMode p_mode);
	bool process(int *p_input, int *p_output, int p_frames, int p_stereo_stride = 1); // return tru if audio was created
	void set_mix_rate(int p_mix_rate);

	ReverbSW();
	~ReverbSW();
};

#endif

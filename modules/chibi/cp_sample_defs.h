/*************************************************************************/
/*  cp_sample_defs.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef CP_SAMPLE_DEFS_H
#define CP_SAMPLE_DEFS_H

#include "cp_config.h"

enum CPSample_Loop_Type {
	
	CP_LOOP_NONE,
	CP_LOOP_FORWARD,
	CP_LOOP_BIDI
};

//#define INVALID_SAMPLE_ID -1

#define CP_MIXING_FRAC_BITS_MACRO 13
#define CP_MIXING_FRAC_BITS_TEXT "13"
// 1<<9 - 1
#define CP_MIXING_FRAC_BITS_MASK_TEXT "8191"

enum CPMixConstants {
	CP_MIXING_FRAC_BITS=CP_MIXING_FRAC_BITS_MACRO,
	CP_MIXING_FRAC_LENGTH=(1<<CP_MIXING_FRAC_BITS),
	CP_MIXING_FRAC_MASK=CP_MIXING_FRAC_LENGTH-1,
	CP_MIXING_VOL_FRAC_BITS=8,
	CP_MIXING_FREQ_FRAC_BITS=8
};

enum CPFilterConstants {
	CP_FILTER_SHIFT=16,
	CP_FILTER_LENGTH=(1<<CP_FILTER_SHIFT)
};


enum CPInterpolationType {
	CP_INTERPOLATION_RAW,
	CP_INTERPOLATION_LINEAR,
	CP_INTERPOLATION_CUBIC
};
	
enum CPPanConstants {
	
	CP_PAN_BITS=8, // 0 .. 256
	CP_PAN_LEFT=0,
	CP_PAN_RIGHT=((1<<CP_PAN_BITS)-1), // 255
	CP_PAN_CENTER=CP_PAN_RIGHT/2, // 128
	CP_PAN_SURROUND=512
};

enum CPMixerVolConstants {
	CP_VOL_MAX=512,
	CP_VOL_RAMP_BITS=9,
	CP_VOL_SHIFT=2
			
	
};	

enum CPStereoCannels {
	CP_CHAN_LEFT,
	CP_CHAN_RIGHT
};

#define CP_FIRST_SAMPLE_DECLICK_THRESHOLD 1000
#define CP_FIRST_SAMPLE_RAMP_LEN 32

typedef signed char CPFrame8;
typedef signed short CPFrame16;


#endif

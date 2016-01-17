/*************************************************************************/
/*  pc_joystick_map.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef PC_JOYSTICK_MAP_H
#define PC_JOYSTICK_MAP_H

#include "input_event.h"

static const int _pc_joystick_button_remap[JOY_BUTTON_MAX]={

	JOY_SELECT,
	JOY_L3,
	JOY_R3,
	JOY_START,

	JOY_DPAD_UP,
	JOY_DPAD_RIGHT,
	JOY_DPAD_DOWN,
	JOY_DPAD_LEFT,

	JOY_L2,
	JOY_R2,
	JOY_L,
	JOY_R,

	JOY_SNES_X,
	JOY_SNES_A,
	JOY_SNES_B,
	JOY_SNES_Y,

	// JOY_HOME = 16
};


static int _pc_joystick_get_native_button(int p_pc_button) {

	if (p_pc_button<0 || p_pc_button>=JOY_BUTTON_MAX)
		return p_pc_button;
	return _pc_joystick_button_remap[p_pc_button];
}

static const int _pc_joystick_axis_remap[JOY_AXIS_MAX]={
	JOY_ANALOG_0_X,
	JOY_ANALOG_0_Y,
	JOY_ANALOG_1_X,
	JOY_ANALOG_1_Y,
	JOY_ANALOG_2_X,
	JOY_ANALOG_2_Y,
	JOY_AXIS_6,
	JOY_AXIS_7
};


static int _pc_joystick_get_native_axis(int p_pc_axis) {

	if (p_pc_axis<0 || p_pc_axis>=JOY_BUTTON_MAX)
		return p_pc_axis;
	return _pc_joystick_axis_remap[p_pc_axis];
}

#endif // PC_JOYSTICK_MAP_H

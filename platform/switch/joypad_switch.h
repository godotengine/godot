/**************************************************************************/
/*  joypad_switch.h                                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef JOYPAD_SWITCH_H
#define JOYPAD_SWITCH_H

#include "switch_wrapper.h"

#include <core/input/input.h>
#include <core/input/input_enums.h>

#include <array>
#include <vector>

typedef std::vector<std::pair<uint, JoyButton>> PadMappingSwitch;

struct PadStateSwitch : public PadState {
	bool initialized = false;
	int id = 0;
	PadMappingSwitch mapping = {};
};

class JoypadSwitch {
private:
	std::array<PadStateSwitch, 8> _pads; //switch support up to 8 controllers
	Input *_input = nullptr;

protected:
public:
	PadStateSwitch &get_pad(int i = 0) { return _pads[i]; }

	//when only both joy-con are use as a single controller (general case)
	static const PadMappingSwitch switch_joy_dual_button_map;
	//when only right joy-con is use as a controller horizontally
	static const PadMappingSwitch switch_joy_right_button_map;
	//when only left joy-con is use as a controller horizontally
	static const PadMappingSwitch switch_joy_left_button_map;

	void initialize(Input *input);
	void discover_pad(PadStateSwitch &pad);
	void dispatch(PadStateSwitch &pad);

	void process();

	JoypadSwitch();
	virtual ~JoypadSwitch() = default;
};

#endif //JOYPAD_SWITCH_H
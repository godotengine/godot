/**************************************************************************/
/*  input_event_midi.hpp                                                  */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/input_event.hpp>
#include <godot_cpp/classes/ref.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class InputEventMIDI : public InputEvent {
	GDEXTENSION_CLASS(InputEventMIDI, InputEvent)

public:
	void set_channel(int32_t p_channel);
	int32_t get_channel() const;
	void set_message(MIDIMessage p_message);
	MIDIMessage get_message() const;
	void set_pitch(int32_t p_pitch);
	int32_t get_pitch() const;
	void set_velocity(int32_t p_velocity);
	int32_t get_velocity() const;
	void set_instrument(int32_t p_instrument);
	int32_t get_instrument() const;
	void set_pressure(int32_t p_pressure);
	int32_t get_pressure() const;
	void set_controller_number(int32_t p_controller_number);
	int32_t get_controller_number() const;
	void set_controller_value(int32_t p_controller_value);
	int32_t get_controller_value() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		InputEvent::register_virtuals<T, B>();
	}

public:
};

} // namespace godot


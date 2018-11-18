/*************************************************************************/
/*  midi_driver.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "midi_driver.h"

#include "core/os/os.h"
#include "main/input_default.h"

MIDIDriver *MIDIDriver::singleton = NULL;
MIDIDriver *MIDIDriver::get_singleton() {

	return singleton;
}

void MIDIDriver::set_singleton() {

	singleton = this;
}

void MIDIDriver::receive_input_packet(uint64_t timestamp, uint8_t *data, uint32_t length) {

	Ref<InputEventMIDI> event;
	event.instance();

	if (length >= 1) {
		event->set_channel(data[0] & 0xF);
		event->set_message(data[0] >> 4);
	}

	switch (event->get_message()) {
		case MIDI_MESSAGE_AFTERTOUCH:
			if (length >= 3) {
				event->set_pitch(data[1]);
				event->set_pressure(data[2]);
			}
			break;

		case MIDI_MESSAGE_CONTROL_CHANGE:
			if (length >= 3) {
				event->set_controller_number(data[1]);
				event->set_controller_value(data[2]);
			}
			break;

		case MIDI_MESSAGE_NOTE_ON:
		case MIDI_MESSAGE_NOTE_OFF:
		case MIDI_MESSAGE_PITCH_BEND:
			if (length >= 3) {
				event->set_pitch(data[1]);
				event->set_velocity(data[2]);
			}
			break;

		case MIDI_MESSAGE_PROGRAM_CHANGE:
			if (length >= 2) {
				event->set_instrument(data[1]);
			}
			break;

		case MIDI_MESSAGE_CHANNEL_PRESSURE:
			if (length >= 2) {
				event->set_pressure(data[1]);
			}
			break;
	}

	InputDefault *id = Object::cast_to<InputDefault>(Input::get_singleton());
	id->parse_input_event(event);
}

PoolStringArray MIDIDriver::get_connected_inputs() {

	PoolStringArray list;
	return list;
}

MIDIDriver::MIDIDriver() {

	set_singleton();
}

/**************************************************************************/
/*  midi_driver.cpp                                                       */
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

#include "midi_driver.h"

#include "core/input/input.h"

uint8_t MIDIDriver::last_received_message = 0x00;
MIDIDriver *MIDIDriver::singleton = nullptr;
MIDIDriver *MIDIDriver::get_singleton() {
	return singleton;
}

void MIDIDriver::set_singleton() {
	singleton = this;
}

void MIDIDriver::receive_input_packet(int device_index, uint64_t timestamp, uint8_t *data, uint32_t length) {
	Ref<InputEventMIDI> event;
	event.instantiate();
	event->set_device(device_index);
	uint32_t param_position = 1;

	if (length >= 1) {
		if (data[0] >= 0xF0) {
			// channel does not apply to system common messages
			event->set_channel(0);
			event->set_message(MIDIMessage(data[0]));
			last_received_message = data[0];
		} else if ((data[0] & 0x80) == 0x00) {
			// running status
			event->set_channel(last_received_message & 0xF);
			event->set_message(MIDIMessage(last_received_message >> 4));
			param_position = 0;
		} else {
			event->set_channel(data[0] & 0xF);
			event->set_message(MIDIMessage(data[0] >> 4));
			param_position = 1;
			last_received_message = data[0];
		}
	}

	switch (event->get_message()) {
		case MIDIMessage::AFTERTOUCH:
			if (length >= 2 + param_position) {
				event->set_pitch(data[param_position]);
				event->set_pressure(data[param_position + 1]);
			}
			break;

		case MIDIMessage::CONTROL_CHANGE:
			if (length >= 2 + param_position) {
				event->set_controller_number(data[param_position]);
				event->set_controller_value(data[param_position + 1]);
			}
			break;

		case MIDIMessage::NOTE_ON:
		case MIDIMessage::NOTE_OFF:
			if (length >= 2 + param_position) {
				event->set_pitch(data[param_position]);
				event->set_velocity(data[param_position + 1]);
			}
			break;

		case MIDIMessage::PITCH_BEND:
			if (length >= 2 + param_position) {
				event->set_pitch((data[param_position + 1] << 7) | data[param_position]);
			}
			break;

		case MIDIMessage::PROGRAM_CHANGE:
			if (length >= 1 + param_position) {
				event->set_instrument(data[param_position]);
			}
			break;

		case MIDIMessage::CHANNEL_PRESSURE:
			if (length >= 1 + param_position) {
				event->set_pressure(data[param_position]);
			}
			break;
		default:
			break;
	}

	Input *id = Input::get_singleton();
	id->parse_input_event(event);
}

PackedStringArray MIDIDriver::get_connected_inputs() {
	PackedStringArray list;
	return list;
}

MIDIDriver::MIDIDriver() {
	set_singleton();
}

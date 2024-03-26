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

void MIDIDriver::receive_input_packet(int p_device_index, uint64_t p_timestamp, uint8_t *p_data, uint32_t p_length) {
	// p_data may contain multiple messages, eg multiple note on using MIDI 'running status'.
	uint32_t index = 0; // index will be incremented as each status or data byte is read from p_data.

	// Each time through loop will consume a MIDI message worth of bytes from p_data and create one InputEventMIDI.
	while (index < p_length) {
		Ref<InputEventMIDI> event;
		event.instantiate();
		event->set_device(p_device_index);

		if (p_length - index >= 1) {
			if (p_data[index] >= 0xF0) {
				// Channel does not apply to system common messages.
				event->set_channel(0);
				event->set_message(MIDIMessage(p_data[index]));
				last_received_message = p_data[index];
				index++;
			} else if ((p_data[index] & 0x80) == 0x00) {
				// Running status, index is not changed because no new MIDI status byte, just additional data bytes.
				event->set_channel(last_received_message & 0xF);
				event->set_message(MIDIMessage(last_received_message >> 4));
			} else {
				event->set_channel(p_data[index] & 0xF);
				event->set_message(MIDIMessage(p_data[index] >> 4));
				last_received_message = p_data[index];
				index++;
			}
		}

		switch (event->get_message()) {
			case MIDIMessage::AFTERTOUCH:
				if (p_length >= 2 + index) {
					event->set_pitch(p_data[index++]);
					event->set_pressure(p_data[index++]);
				}
				break;

			case MIDIMessage::CONTROL_CHANGE:
				if (p_length >= 2 + index) {
					event->set_controller_number(p_data[index++]);
					event->set_controller_value(p_data[index++]);
				}
				break;

			case MIDIMessage::NOTE_ON:
			case MIDIMessage::NOTE_OFF:
				if (p_length >= 2 + index) {
					event->set_pitch(p_data[index++]);
					event->set_velocity(p_data[index++]);
				}
				break;

			case MIDIMessage::PITCH_BEND:
				if (p_length >= 2 + index) {
					event->set_pitch((p_data[index + 1] << 7) | p_data[index]);
					index += 2;
				}
				break;

			case MIDIMessage::PROGRAM_CHANGE:
				if (p_length >= 1 + index) {
					event->set_instrument(p_data[index++]);
				}
				break;

			case MIDIMessage::CHANNEL_PRESSURE:
				if (p_length >= 1 + index) {
					event->set_pressure(p_data[index++]);
				}
				break;

			default:
				// Anything else means unsupported MIDI or invalid packet,
				// cannot further parse p_data, so exit.
				return;
		}

		Input::get_singleton()->parse_input_event(event);
	}
}

PackedStringArray MIDIDriver::get_connected_inputs() {
	PackedStringArray list;
	return list;
}

MIDIDriver::MIDIDriver() {
	set_singleton();
}

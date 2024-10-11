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

MIDIDriver::MIDIDriver() {
	singleton = this;
}

MIDIDriver::MessageCategory MIDIDriver::Parser::category(uint8_t p_midi_fragment) {
	if (p_midi_fragment >= 0xf8) {
		return MessageCategory::RealTime;
	} else if (p_midi_fragment >= 0xf0) {
		// System Exclusive begin/end are specified as System Common Category
		// messages, but we separate them here and give them their own categories
		// as their behavior is significantly different.
		if (p_midi_fragment == 0xf0) {
			return MessageCategory::SysExBegin;
		} else if (p_midi_fragment == 0xf7) {
			return MessageCategory::SysExEnd;
		}
		return MessageCategory::SystemCommon;
	} else if (p_midi_fragment >= 0x80) {
		return MessageCategory::Voice;
	}
	return MessageCategory::Data;
}

MIDIMessage MIDIDriver::Parser::status_to_msg_enum(uint8_t p_status_byte) {
	if (p_status_byte & 0x80) {
		if (p_status_byte < 0xf0) {
			return MIDIMessage(p_status_byte >> 4);
		} else {
			return MIDIMessage(p_status_byte);
		}
	}
	return MIDIMessage::NONE;
}

size_t MIDIDriver::Parser::expected_data(uint8_t p_status_byte) {
	return expected_data(status_to_msg_enum(p_status_byte));
}

size_t MIDIDriver::Parser::expected_data(MIDIMessage p_msg_type) {
	switch (p_msg_type) {
		case MIDIMessage::NOTE_OFF:
		case MIDIMessage::NOTE_ON:
		case MIDIMessage::AFTERTOUCH:
		case MIDIMessage::CONTROL_CHANGE:
		case MIDIMessage::PITCH_BEND:
		case MIDIMessage::SONG_POSITION_POINTER:
			return 2;
		case MIDIMessage::PROGRAM_CHANGE:
		case MIDIMessage::CHANNEL_PRESSURE:
		case MIDIMessage::QUARTER_FRAME:
		case MIDIMessage::SONG_SELECT:
			return 1;
		default:
			return 0;
	}
}

uint8_t MIDIDriver::Parser::channel(uint8_t p_status_byte) {
	if (category(p_status_byte) == MessageCategory::Voice) {
		return p_status_byte & 0x0f;
	}
	return 0;
}

void MIDIDriver::send_event(int p_device_index, uint8_t p_status,
		const uint8_t *p_data, size_t p_data_len) {
	const MIDIMessage msg = Parser::status_to_msg_enum(p_status);
	ERR_FAIL_COND(p_data_len < Parser::expected_data(msg));

	Ref<InputEventMIDI> event;
	event.instantiate();
	event->set_device(p_device_index);
	event->set_channel(Parser::channel(p_status));
	event->set_message(msg);
	switch (msg) {
		case MIDIMessage::NOTE_OFF:
		case MIDIMessage::NOTE_ON:
			event->set_pitch(p_data[0]);
			event->set_velocity(p_data[1]);
			break;
		case MIDIMessage::AFTERTOUCH:
			event->set_pitch(p_data[0]);
			event->set_pressure(p_data[1]);
			break;
		case MIDIMessage::CONTROL_CHANGE:
			event->set_controller_number(p_data[0]);
			event->set_controller_value(p_data[1]);
			break;
		case MIDIMessage::PROGRAM_CHANGE:
			event->set_instrument(p_data[0]);
			break;
		case MIDIMessage::CHANNEL_PRESSURE:
			event->set_pressure(p_data[0]);
			break;
		case MIDIMessage::PITCH_BEND:
			event->set_pitch((p_data[1] << 7) | p_data[0]);
			break;
		// QUARTER_FRAME, SONG_POSITION_POINTER, and SONG_SELECT not yet implemented.
		default:
			break;
	}
	Input::get_singleton()->parse_input_event(event);
}

void MIDIDriver::Parser::parse_fragment(uint8_t p_fragment) {
	switch (category(p_fragment)) {
		case MessageCategory::RealTime:
			// Real-Time messages are single byte messages that can
			// occur at any point and do not interrupt other messages.
			// We pass them straight through.
			MIDIDriver::send_event(device_index, p_fragment);
			break;

		case MessageCategory::SysExBegin:
			status_byte = p_fragment;
			skipping_sys_ex = true;
			break;

		case MessageCategory::SysExEnd:
			status_byte = 0;
			skipping_sys_ex = false;
			break;

		case MessageCategory::Voice:
		case MessageCategory::SystemCommon:
			skipping_sys_ex = false; // If we were in SysEx, assume it was aborted.
			received_data_len = 0;
			status_byte = 0;
			ERR_FAIL_COND(expected_data(p_fragment) > DATA_BUFFER_SIZE);
			if (expected_data(p_fragment) == 0) {
				// No data bytes needed, post it now.
				MIDIDriver::send_event(device_index, p_fragment);
			} else {
				status_byte = p_fragment;
			}
			break;

		case MessageCategory::Data:
			// We don't currently process SysEx messages, so ignore their data.
			if (!skipping_sys_ex) {
				const size_t expected = expected_data(status_byte);
				if (received_data_len < expected) {
					data_buffer[received_data_len] = p_fragment;
					received_data_len++;
					if (received_data_len == expected) {
						MIDIDriver::send_event(device_index, status_byte,
								data_buffer, expected);
						received_data_len = 0;
						// Voice messages can use 'running status', sending further
						// messages without resending their status byte.
						// For other messages types we clear the cached status byte.
						if (category(status_byte) != MessageCategory::Voice) {
							status_byte = 0;
						}
					}
				}
			}
			break;
	}
}

PackedStringArray MIDIDriver::get_connected_inputs() const {
	return connected_input_names;
}

PackedStringArray MIDIDriver::get_connected_outputs() const {
	return connected_output_names;
}

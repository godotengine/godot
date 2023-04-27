/**************************************************************************/
/*  midi_driver_alsamidi.cpp                                              */
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

#ifdef ALSAMIDI_ENABLED

#include "midi_driver_alsamidi.h"

#include "core/os/os.h"
#include "core/string/print_string.h"

#include <errno.h>

MIDIDriverALSAMidi::MessageCategory MIDIDriverALSAMidi::msg_category(uint8_t msg_part) {
	if (msg_part >= 0xf8) {
		return MessageCategory::RealTime;
	} else if (msg_part >= 0xf0) {
		// System Exclusive begin/end are specified as System Common Category messages,
		// but we separate them here and give them their own categories as their
		// behavior is significantly different.
		if (msg_part == 0xf0) {
			return MessageCategory::SysExBegin;
		} else if (msg_part == 0xf7) {
			return MessageCategory::SysExEnd;
		}
		return MessageCategory::SystemCommon;
	} else if (msg_part >= 0x80) {
		return MessageCategory::Voice;
	}
	return MessageCategory::Data;
}

size_t MIDIDriverALSAMidi::msg_expected_data(uint8_t status_byte) {
	if (msg_category(status_byte) == MessageCategory::Voice) {
		// Voice messages have a channel number in the status byte, mask it out.
		status_byte &= 0xf0;
	}

	switch (status_byte) {
		case 0x80: // Note Off
		case 0x90: // Note On
		case 0xA0: // Polyphonic Key Pressure (Aftertouch)
		case 0xB0: // Control Change (CC)
		case 0xE0: // Pitch Bend Change
		case 0xF2: // Song Position Pointer
			return 2;

		case 0xC0: // Program Change
		case 0xD0: // Channel Pressure (Aftertouch)
		case 0xF1: // MIDI Time Code Quarter Frame
		case 0xF3: // Song Select
			return 1;
	}

	return 0;
}

void MIDIDriverALSAMidi::InputConnection::parse_byte(uint8_t byte, MIDIDriverALSAMidi &driver,
		uint64_t timestamp) {
	switch (msg_category(byte)) {
		case MessageCategory::RealTime:
			// Real-Time messages are single byte messages that can
			// occur at any point.
			// We pass them straight through.
			driver.receive_input_packet(timestamp, &byte, 1);
			break;

		case MessageCategory::Data:
			// We don't currently forward System Exclusive messages so skip their data.
			// Collect any expected data for other message types.
			if (!skipping_sys_ex && expected_data > received_data) {
				buffer[received_data + 1] = byte;
				received_data++;

				// Forward a complete message and reset relevant state.
				if (received_data == expected_data) {
					driver.receive_input_packet(timestamp, buffer, received_data + 1);
					received_data = 0;

					if (msg_category(buffer[0]) != MessageCategory::Voice) {
						// Voice Category messages can be sent with "running status".
						// This means they don't resend the status byte until it changes.
						// For other categories, we reset expected data, to require a new status byte.
						expected_data = 0;
					}
				}
			}
			break;

		case MessageCategory::SysExBegin:
			buffer[0] = byte;
			skipping_sys_ex = true;
			break;

		case MessageCategory::SysExEnd:
			expected_data = 0;
			skipping_sys_ex = false;
			break;

		case MessageCategory::Voice:
		case MessageCategory::SystemCommon:
			buffer[0] = byte;
			received_data = 0;
			expected_data = msg_expected_data(byte);
			skipping_sys_ex = false;
			if (expected_data == 0) {
				driver.receive_input_packet(timestamp, &byte, 1);
			}
			break;
	}
}

int MIDIDriverALSAMidi::InputConnection::read_in(MIDIDriverALSAMidi &driver, uint64_t timestamp) {
	int ret;
	do {
		uint8_t byte = 0;
		ret = snd_rawmidi_read(rawmidi_ptr, &byte, 1);

		if (ret < 0) {
			if (ret != -EAGAIN) {
				ERR_PRINT("snd_rawmidi_read error: " + String(snd_strerror(ret)));
			}
		} else {
			parse_byte(byte, driver, timestamp);
		}
	} while (ret > 0);

	return ret;
}

void MIDIDriverALSAMidi::thread_func(void *p_udata) {
	MIDIDriverALSAMidi *md = static_cast<MIDIDriverALSAMidi *>(p_udata);
	uint64_t timestamp = 0;

	while (!md->exit_thread.is_set()) {
		md->lock();

		InputConnection *connections = md->connected_inputs.ptrw();
		size_t connection_count = md->connected_inputs.size();

		for (size_t i = 0; i < connection_count; i++) {
			connections[i].read_in(*md, timestamp);
		}

		md->unlock();

		OS::get_singleton()->delay_usec(1000);
	}
}

Error MIDIDriverALSAMidi::open() {
	void **hints;

	if (snd_device_name_hint(-1, "rawmidi", &hints) < 0) {
		return ERR_CANT_OPEN;
	}

	int i = 0;
	for (void **n = hints; *n != nullptr; n++) {
		char *name = snd_device_name_get_hint(*n, "NAME");

		if (name != nullptr) {
			snd_rawmidi_t *midi_in;
			int ret = snd_rawmidi_open(&midi_in, nullptr, name, SND_RAWMIDI_NONBLOCK);
			if (ret >= 0) {
				connected_inputs.insert(i++, InputConnection(midi_in));
			}
		}

		if (name != nullptr) {
			free(name);
		}
	}
	snd_device_name_free_hint(hints);

	exit_thread.clear();
	thread.start(MIDIDriverALSAMidi::thread_func, this);

	return OK;
}

void MIDIDriverALSAMidi::close() {
	exit_thread.set();
	if (thread.is_started()) {
		thread.wait_to_finish();
	}

	for (int i = 0; i < connected_inputs.size(); i++) {
		snd_rawmidi_t *midi_in = connected_inputs[i].rawmidi_ptr;
		snd_rawmidi_close(midi_in);
	}
	connected_inputs.clear();
}

void MIDIDriverALSAMidi::lock() const {
	mutex.lock();
}

void MIDIDriverALSAMidi::unlock() const {
	mutex.unlock();
}

PackedStringArray MIDIDriverALSAMidi::get_connected_inputs() {
	PackedStringArray list;

	lock();
	for (int i = 0; i < connected_inputs.size(); i++) {
		snd_rawmidi_t *midi_in = connected_inputs[i].rawmidi_ptr;
		snd_rawmidi_info_t *info;

		snd_rawmidi_info_malloc(&info);
		snd_rawmidi_info(midi_in, info);
		list.push_back(snd_rawmidi_info_get_name(info));
		snd_rawmidi_info_free(info);
	}
	unlock();

	return list;
}

MIDIDriverALSAMidi::MIDIDriverALSAMidi() {
	exit_thread.clear();
}

MIDIDriverALSAMidi::~MIDIDriverALSAMidi() {
	close();
}

#endif // ALSAMIDI_ENABLED

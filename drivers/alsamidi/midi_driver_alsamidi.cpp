/*************************************************************************/
/*  midi_driver_alsamidi.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifdef ALSAMIDI_ENABLED

#include "midi_driver_alsamidi.h"

#include "core/os/os.h"
#include "core/print_string.h"

#include <errno.h>

static int get_message_size(uint8_t message) {
	switch (message & 0xF0) {
		case 0x80: // note off
		case 0x90: // note on
		case 0xA0: // aftertouch
		case 0xB0: // continuous controller
		case 0xE0: // pitch bend
		case 0xF2: // song position pointer
			return 3;

		case 0xC0: // patch change
		case 0xD0: // channel pressure
		case 0xF1: // time code quarter frame
		case 0xF3: // song select
			return 2;

		case 0xF0: // SysEx start
		case 0xF4: // reserved
		case 0xF5: // reserved
		case 0xF6: // tune request
		case 0xF7: // SysEx end
		case 0xF8: // timing clock
		case 0xF9: // reserved
		case 0xFA: // start
		case 0xFB: // continue
		case 0xFC: // stop
		case 0xFD: // reserved
		case 0xFE: // active sensing
		case 0xFF: // reset
			return 1;
	}

	return 256;
}

void MIDIDriverALSAMidi::thread_func(void *p_udata) {
	MIDIDriverALSAMidi *md = (MIDIDriverALSAMidi *)p_udata;
	uint64_t timestamp = 0;
	uint8_t buffer[256];
	int expected_size = 255;
	int bytes = 0;

	while (!md->exit_thread) {
		int ret;

		md->lock();

		for (int i = 0; i < md->connected_inputs.size(); i++) {
			snd_rawmidi_t *midi_in = md->connected_inputs[i];
			do {
				uint8_t byte = 0;
				ret = snd_rawmidi_read(midi_in, &byte, 1);
				if (ret < 0) {
					if (ret != -EAGAIN) {
						ERR_PRINT("snd_rawmidi_read error: " + String(snd_strerror(ret)));
					}
				} else {
					if (byte & 0x80) {
						// Flush previous packet if there is any
						if (bytes) {
							md->receive_input_packet(timestamp, buffer, bytes);
							bytes = 0;
						}
						expected_size = get_message_size(byte);
						// After a SysEx start, all bytes are data until a SysEx end, so
						// we're going to end the command at the SES, and let the common
						// driver ignore the following data bytes.
					}

					if (bytes < 256) {
						buffer[bytes++] = byte;
						// If we know the size of the current packet receive it if it reached the expected size
						if (bytes >= expected_size) {
							md->receive_input_packet(timestamp, buffer, bytes);
							bytes = 0;
						}
					}
				}
			} while (ret > 0);
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
				connected_inputs.insert(i++, midi_in);
			}
		}

		if (name != nullptr) {
			free(name);
		}
	}
	snd_device_name_free_hint(hints);

	exit_thread = false;
	thread = Thread::create(MIDIDriverALSAMidi::thread_func, this);

	return OK;
}

void MIDIDriverALSAMidi::close() {
	if (thread) {
		exit_thread = true;
		Thread::wait_to_finish(thread);

		memdelete(thread);
		thread = nullptr;
	}

	for (int i = 0; i < connected_inputs.size(); i++) {
		snd_rawmidi_t *midi_in = connected_inputs[i];
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
		snd_rawmidi_t *midi_in = connected_inputs[i];
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
	thread = nullptr;

	exit_thread = false;
}

MIDIDriverALSAMidi::~MIDIDriverALSAMidi() {
	close();
}

#endif // ALSAMIDI_ENABLED

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

#include <cerrno>

MIDIDriverALSAMidi::InputConnection::InputConnection(int p_device_index,
		snd_rawmidi_t *p_rawmidi) :
		parser(p_device_index), rawmidi_ptr(p_rawmidi) {}

void MIDIDriverALSAMidi::InputConnection::read() {
	int read_count;
	do {
		uint8_t buffer[32];
		read_count = snd_rawmidi_read(rawmidi_ptr, buffer, sizeof(buffer));

		if (read_count < 0) {
			if (read_count != -EAGAIN) {
				ERR_PRINT("snd_rawmidi_read error: " + String(snd_strerror(read_count)));
			}
		} else {
			for (int i = 0; i < read_count; i++) {
				parser.parse_fragment(buffer[i]);
			}
		}
	} while (read_count > 0);
}

void MIDIDriverALSAMidi::thread_func(void *p_udata) {
	MIDIDriverALSAMidi *md = static_cast<MIDIDriverALSAMidi *>(p_udata);

	while (!md->exit_thread.is_set()) {
		md->lock();
		for (InputConnection &conn : md->connected_inputs) {
			conn.read();
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

	lock();
	int device_index = 0;
	for (void **h = hints; *h != nullptr; h++) {
		char *name = snd_device_name_get_hint(*h, "NAME");

		if (name != nullptr) {
			snd_rawmidi_t *midi_in;
			snd_rawmidi_t *midi_out;
			int ret = snd_rawmidi_open(&midi_in, &midi_out, name, SND_RAWMIDI_NONBLOCK);
			if (ret >= 0) {
				// Get display name.
				if (midi_in != nullptr) {
					snd_rawmidi_info_t *info;
					snd_rawmidi_info_malloc(&info);
					snd_rawmidi_info(midi_in, info);
					connected_input_names.push_back(snd_rawmidi_info_get_name(info));
					snd_rawmidi_info_free(info);
					connected_inputs.push_back(InputConnection(device_index, midi_in));
					// Only increment device_index for successfully connected devices.
					device_index++;
				}
				if (midi_out != nullptr) {
					snd_rawmidi_info_t *info;
					snd_rawmidi_info_malloc(&info);
					snd_rawmidi_info(midi_out, info);
					connected_output_names.push_back(snd_rawmidi_info_get_name(info));
					connected_outputs.push_back(midi_out);
					snd_rawmidi_info_free(info);
				}
			}
		}

		if (name != nullptr) {
			free(name);
		}
	}
	snd_device_name_free_hint(hints);
	unlock();

	exit_thread.clear();
	thread.start(MIDIDriverALSAMidi::thread_func, this);

	return OK;
}

void MIDIDriverALSAMidi::close() {
	exit_thread.set();
	if (thread.is_started()) {
		thread.wait_to_finish();
	}

	for (const InputConnection &conn : connected_inputs) {
		snd_rawmidi_close(conn.rawmidi_ptr);
	}

	connected_inputs.clear();
	connected_input_names.clear();
}

Error MIDIDriverALSAMidi::send(Ref<InputEventMIDI> p_event) {
	ERR_FAIL_COND_V(p_event.is_null(), ERR_INVALID_PARAMETER);
	int device_id = p_event->get_device();
	ERR_FAIL_INDEX_V(device_id, connected_outputs.size(), ERR_PARAMETER_RANGE_ERROR);
	PackedByteArray packet = p_event->get_midi_bytes();
	snd_rawmidi_write(connected_outputs[device_id], packet.ptrw(), packet.size());
	return OK;
}

void MIDIDriverALSAMidi::lock() const {
	mutex.lock();
}

void MIDIDriverALSAMidi::unlock() const {
	mutex.unlock();
}

MIDIDriverALSAMidi::MIDIDriverALSAMidi() {
	exit_thread.clear();
}

MIDIDriverALSAMidi::~MIDIDriverALSAMidi() {
	close();
}

#endif // ALSAMIDI_ENABLED

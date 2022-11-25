/*************************************************************************/
/*  midi_driver_alsamidi.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef MIDI_DRIVER_ALSAMIDI_H
#define MIDI_DRIVER_ALSAMIDI_H

#ifdef ALSAMIDI_ENABLED

#include "core/os/midi_driver.h"
#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "core/string/ustring.h"
#include "core/templates/safe_refcount.h"
#include "core/templates/vector.h"

#include "../alsa/asound-so_wrap.h"
#include <stdio.h>

#define ALSA_MAX_MIDI_EVENT_SIZE 16 * 1024

class MIDIDriverALSAMidi : public MIDIDriver {
	Thread thread;
	Mutex mutex;

	snd_seq_t *seq_handle = nullptr;
	void read(MIDIDriverALSAMidi &driver);

	class ConnectedDevices {
	public:
		ConnectedDevices() = default;
		ConnectedDevices(const char *p_name, int p_client, int p_port) :
				name(p_name), client(p_client), port(p_port) {}

		String name;
		int client;
		int port;
	};

	Vector<ConnectedDevices> connected_devices;
	int get_devices();
	int get_connected_devices();

	SafeFlag exit_thread;

	static void thread_func(void *p_udata);

	int numPfds;
	struct pollfd *pfd = nullptr;

	class Decoder {
	public:
		Decoder() = default;
		void init() { snd_midi_event_new(ALSA_MAX_MIDI_EVENT_SIZE, &midiev); };
		void reset() { snd_midi_event_init(midiev); };
		void free() {
			if (midiev) {
				snd_midi_event_free(midiev);
				midiev = nullptr;
			}
		};
		snd_midi_event_t *get() { return midiev; };

	private:
		snd_midi_event_t *midiev = nullptr;
	};

	Decoder decoder;

	void lock() const;
	void unlock() const;

public:
	virtual Error open();
	virtual void close();

	virtual PackedStringArray get_connected_inputs();

	MIDIDriverALSAMidi();
	virtual ~MIDIDriverALSAMidi();
};

#endif // ALSAMIDI_ENABLED

#endif // MIDI_DRIVER_ALSAMIDI_H

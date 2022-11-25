/*************************************************************************/
/*  midi_driver_alsamidi.cpp                                             */
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

#ifdef ALSAMIDI_ENABLED

#include "midi_driver_alsamidi.h"

#include "core/os/os.h"
#include "core/string/print_string.h"

#include <errno.h>

void MIDIDriverALSAMidi::read(MIDIDriverALSAMidi &driver) {
	snd_seq_event_t *ev;
	unsigned char buf[ALSA_MAX_MIDI_EVENT_SIZE];
	do {
		if (snd_seq_event_input(driver.seq_handle, &ev) >= 0) {
			const int numBytes = snd_midi_event_decode(driver.decoder.get(), buf, sizeof(buf), ev);
			driver.decoder.reset();
			if (numBytes > 0) {
				driver.receive_input_packet(0, buf, snd_seq_event_length(ev));
			}
			snd_seq_free_event(ev);
		}
	} while (snd_seq_event_input_pending(driver.seq_handle, 0) > 0);
}

void MIDIDriverALSAMidi::thread_func(void *p_udata) {
	MIDIDriverALSAMidi *md = static_cast<MIDIDriverALSAMidi *>(p_udata);

	while (!md->exit_thread.is_set()) {
		if (poll(md->pfd, md->numPfds, 100) > 0) {
			md->read(*md);
		}
	}
}

Error MIDIDriverALSAMidi::open() {
	lock();
	if (!seq_handle) {
		if (int ret = snd_seq_open(&seq_handle, "default", SND_SEQ_OPEN_INPUT, SND_SEQ_NONBLOCK) < 0) {
			ERR_PRINT("snd_seq_open failed: " + String(snd_strerror(ret)));
			unlock();
			return ERR_CANT_OPEN;
		}
		snd_seq_set_client_name(seq_handle, "Godot");

		// setting port type as SND_SEQ_PORT_TYPE_MIDI_GENERIC allows other sofware to directly connect to Godot without routing app
		if (int ret = snd_seq_create_simple_port(seq_handle, "Godot Input", SND_SEQ_PORT_CAP_WRITE | SND_SEQ_PORT_CAP_SUBS_WRITE, SND_SEQ_PORT_TYPE_MIDI_GENERIC) < 0) {
			ERR_PRINT("snd_seq_create_simple_port failed: " + String(snd_strerror(ret)));
			unlock();
			return ERR_CANT_OPEN;
		}
		// we need to init the decoder after ALSA has openned
		decoder.init();
		// set up poll descriptors
		numPfds = snd_seq_poll_descriptors_count(seq_handle, POLLIN);
		pfd = (struct pollfd *)malloc((numPfds) * sizeof(struct pollfd));
		snd_seq_poll_descriptors(seq_handle, pfd, numPfds, POLLIN);
	}

	if (get_devices() > 0) {
		for (int i = 0; i < connected_devices.size(); i++) {
			// connect all the found input clients to our ALSA sequencer handle
			//printf("connect to: %s, %d, %d\n", connected_devices[i].name.utf8().ptr(), connected_devices[i].client, connected_devices[i].port);
			snd_seq_connect_from(seq_handle, 0, connected_devices[i].client, connected_devices[i].port);
		}
	}
	if (!thread.is_started()) {
		thread.start(MIDIDriverALSAMidi::thread_func, this);
	}
	unlock();
	return OK;
}

int MIDIDriverALSAMidi::get_devices() {
	if (!seq_handle) {
		ERR_PRINT("ALSA not initialized");
		return -1;
	}
	// find MIDI clients (devices)
	connected_devices.clear();
	snd_seq_client_info_t *cinfo;
	snd_seq_client_info_alloca(&cinfo);
	snd_seq_port_info_t *pinfo;
	snd_seq_port_info_alloca(&pinfo);
	snd_seq_client_info_set_client(cinfo, -1);
	while (snd_seq_query_next_client(seq_handle, cinfo) >= 0) {
		bool found_valid_port = false;
		snd_seq_port_info_set_client(pinfo, snd_seq_client_info_get_client(cinfo));
		snd_seq_port_info_set_port(pinfo, -1);
		// loop over all found devices, connect if they are only hardware at this point
		while (!found_valid_port && snd_seq_query_next_port(seq_handle, pinfo) >= 0 && snd_seq_client_info_get_card(cinfo) >= 0) {
			found_valid_port = true;
			const char *name = snd_seq_client_info_get_name(cinfo);
			int client = snd_seq_client_info_get_client(cinfo);
			int port = snd_seq_port_info_get_port(pinfo);
			connected_devices.push_back(ConnectedDevices(name, client, port));
		}
	}
	return connected_devices.size();
}

void MIDIDriverALSAMidi::lock() const {
	mutex.lock();
}

void MIDIDriverALSAMidi::unlock() const {
	mutex.unlock();
}

int MIDIDriverALSAMidi::get_connected_devices() {
	if (seq_handle) {
		connected_devices.clear();
		snd_seq_client_info_t *cinfo;
		snd_seq_port_info_t *pinfo;

		snd_seq_client_info_alloca(&cinfo);
		snd_seq_port_info_alloca(&pinfo);
		/*
		 * set the client id of Godot from the seq_handle,
		 * this stays the same for each seq_handle init
		 * but we check again, incase it has changed
		 */
		snd_seq_client_info_set_client(cinfo, snd_seq_client_id(seq_handle));
		snd_seq_port_info_set_client(pinfo, snd_seq_client_info_get_client(cinfo));
		//we only have one input port
		snd_seq_port_info_set_port(pinfo, 0);

		snd_seq_query_subscribe_t *subs;
		snd_seq_query_subscribe_alloca(&subs);
		snd_seq_query_subscribe_set_root(subs, snd_seq_port_info_get_addr(pinfo));

		snd_seq_query_subscribe_set_type(subs, SND_SEQ_QUERY_SUBS_WRITE);
		snd_seq_query_subscribe_set_index(subs, 0);

		while (snd_seq_query_port_subscribers(seq_handle, subs) >= 0) {
			const snd_seq_addr_t *addr;
			addr = snd_seq_query_subscribe_get_addr(subs);
			/*
			 * apparently we can't just use this:
			 * snd_seq_client_info_set_client();
			 * as that will only populate the client id & port number if
			 * we don't move to next client via snd_seq_query_next_client()
			 */
			snd_seq_get_any_client_info(seq_handle, addr->client, cinfo);
			const char *name = snd_seq_client_info_get_name(cinfo);
			connected_devices.push_back(ConnectedDevices(name, addr->client, addr->port));
			snd_seq_query_subscribe_set_index(subs, snd_seq_query_subscribe_get_index(subs) + 1);
		}
		return connected_devices.size();
	}
	return 0;
}

PackedStringArray MIDIDriverALSAMidi::get_connected_inputs() {
	PackedStringArray list;

	lock();
	get_connected_devices();

	for (int i = 0; i < connected_devices.size(); i++) {
		list.push_back(connected_devices[i].name);
	}

	unlock();
	return list;
}

void MIDIDriverALSAMidi::close() {
	lock();
	exit_thread.set();
	thread.wait_to_finish();

	decoder.free();
	if (pfd) {
		free(pfd);
		pfd = nullptr;
	}
	if (seq_handle) {
		snd_seq_close(seq_handle);
		seq_handle = nullptr;
	}
	connected_devices.clear();
	exit_thread.clear();
	unlock();
}

MIDIDriverALSAMidi::MIDIDriverALSAMidi() {
	exit_thread.clear();
}

MIDIDriverALSAMidi::~MIDIDriverALSAMidi() {
	close();
}

#endif // ALSAMIDI_ENABLED

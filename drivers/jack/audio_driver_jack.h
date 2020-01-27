/*************************************************************************/
/*  audio_driver_jack.h                                                  */
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

#ifndef AUDIO_DRIVER_JACK_H
#define AUDIO_DRIVER_JACK_H

#ifdef JACK_ENABLED

#include "core/os/mutex.h"
#include "servers/audio_server.h"

#include <jack/jack.h>

class AudioDriverJACK : public AudioDriver {

	Mutex *mutex;

	jack_client_t *client;
	Vector<jack_port_t *> ports;
	Vector<jack_port_t *> capture_ports;

	Vector<int32_t> samples_in;

	Error init_device();
	void finish_device();

	static int process_func(jack_nframes_t total_frames, void *p_udata);

	struct DeviceJACK {
		const char *name;
		SpeakerMode speaker_mode;
		int channels() const;
	};

	static const DeviceJACK devices[];
	static const unsigned num_devices;

	static const DeviceJACK capture_devices[];
	static const unsigned num_capture_devices;

	int device_index;
	int capture_device_index;

	bool active;
	bool capture_active;

	void connect_physical_ports();
	void connect_physical_capture_ports();

public:
	const char *get_name() const {
		return "JACK";
	}

	Error init();
	void start();
	int get_mix_rate() const;
	SpeakerMode get_speaker_mode() const;
	Array get_device_list();
	String get_device();
	void set_device(String device);
	void lock();
	void unlock();
	void finish();

	Error capture_start();
	Error capture_stop();
	void capture_set_device(const String &device);
	String capture_get_device();
	Array capture_get_device_list();

	AudioDriverJACK();
	~AudioDriverJACK();
};

#endif // AUDIO_DRIVER_JACK_H
#endif

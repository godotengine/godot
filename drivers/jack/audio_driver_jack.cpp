/*************************************************************************/
/*  audio_driver_jack.cpp                                                */
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

#ifdef JACK_ENABLED

// TODO: the capture devices

#include "audio_driver_jack.h"
#include "core/project_settings.h"
#include "core/version.h"

#include <stdio.h>
#include <string.h>

#define PORT_NAME_FORMAT "Out %u"

const AudioDriverJACK::DeviceJACK AudioDriverJACK::devices[] = {

	// default device outputs to sound card in stereo
	{ "Sound card playback", SPEAKER_MODE_STEREO },

	// others are just virtual software clients
	{ "Stereo playback", SPEAKER_MODE_STEREO },
	{ "Surround 3.1 playback", SPEAKER_SURROUND_31 },
	{ "Surround 5.1 playback", SPEAKER_SURROUND_51 },
	{ "Surround 7.1 playback", SPEAKER_SURROUND_71 },
};

const unsigned AudioDriverJACK::num_devices =
		sizeof(devices) / sizeof(devices[0]);

int AudioDriverJACK::DeviceJACK::channels() const {

	switch (speaker_mode) {
		default:
		case SPEAKER_MODE_STEREO: return 2;
		case SPEAKER_SURROUND_31: return 4;
		case SPEAKER_SURROUND_51: return 6;
		case SPEAKER_SURROUND_71: return 8;
	}
}

Error AudioDriverJACK::init_device() {

	String client_name = GLOBAL_GET("application/config/name");

	if (client_name.length() == 0)
		client_name = VERSION_NAME;

	client = jack_client_open(client_name.utf8(), JackNoStartServer, NULL);

	if (!client)
		ERR_FAIL_COND_V(!client, ERR_CANT_OPEN);

	jack_set_process_callback(client, &process_func, this);

	const DeviceJACK &jdev = devices[device_index];
	unsigned channels = jdev.channels();

#define CHECK_FAIL(m_cond, msg)                 \
	if (m_cond) {                               \
		fprintf(stderr, "JACK ERR: %s\n", msg); \
		finish_device();                        \
		ERR_FAIL_COND_V(m_cond, ERR_CANT_OPEN); \
	}

	for (unsigned ch = 0; ch < channels; ++ch) {
		char port_name[32];
		sprintf(port_name, PORT_NAME_FORMAT, ch + 1);

		jack_port_t *port = jack_port_register(
				client, port_name, JACK_DEFAULT_AUDIO_TYPE,
				JackPortIsOutput, 0);

		CHECK_FAIL(!port, "cannot register output port");

		ports.push_back(port);
	}

	jack_nframes_t buffer_size = jack_get_buffer_size(client);
	samples_in.resize(buffer_size * channels);

	return OK;
}

Error AudioDriverJACK::init() {

	Error err = init_device();
	if (err == OK)
		mutex = Mutex::create();

	return err;
}

void AudioDriverJACK::start() {

	active = true;

	if (!client)
		return;

	jack_activate(client);

	if (device_index == 0) {
		// if the sound card output was picked,
		// identify the physical client and connect
		// note: must always go after activate
		connect_physical_ports();
	}
}

int AudioDriverJACK::get_mix_rate() const {

	if (!client)
		return 0;

	return jack_get_sample_rate(client);
}

AudioDriver::SpeakerMode AudioDriverJACK::get_speaker_mode() const {

	const DeviceJACK &jdev = devices[device_index];
	return jdev.speaker_mode;
}

Array AudioDriverJACK::get_device_list() {

	Array names;

	for (unsigned i = 0; i < num_devices; ++i) {
		const DeviceJACK &jdev = devices[i];
		names.push_back(jdev.name);
	}

	return names;
}

String AudioDriverJACK::get_device() {

	const DeviceJACK &jdev = devices[device_index];
	return jdev.name;
}

void AudioDriverJACK::set_device(String device) {

	int new_index = -1;

	for (unsigned i = 0; i < num_devices && new_index == -1; ++i) {
		const DeviceJACK &jdev = devices[i];
		if (device == jdev.name)
			new_index = i;
	}

	// fallback to default
	if (new_index == -1)
		new_index = 0;

	// close current client, and reopen
	if (device_index != new_index) {
		finish_device();
		device_index = new_index;
		init_device();

		// if it was started before recreating, restart it
		if (active)
			start();
	}
}

void AudioDriverJACK::lock() {

	if (!mutex)
		return;
	mutex->lock();
}

void AudioDriverJACK::unlock() {

	if (!mutex)
		return;
	mutex->unlock();
}

void AudioDriverJACK::finish_device() {

	if (client) {
		jack_client_close(client);
		client = NULL;
	}

	ports.clear();
}

void AudioDriverJACK::finish() {

	finish_device();

	if (mutex) {
		memdelete(mutex);
		mutex = NULL;
	}
}

int AudioDriverJACK::process_func(jack_nframes_t total_frames, void *p_udata) {

	AudioDriverJACK *jd = (AudioDriverJACK *)p_udata;

	jack_port_t *const *ports = jd->ports.ptr();
	unsigned channels = jd->ports.size();

	Mutex *mutex = jd->mutex;

	// invoke try-lock; if it's a failure, fill channels with zero
	// never xrun the JACK server!
	if (mutex && mutex->try_lock() != OK) {
		for (unsigned ch = 0; ch < channels; ++ch) {
			float *ch_out = (float *)jack_port_get_buffer(ports[ch], total_frames);
			memset(ch_out, 0, total_frames * sizeof(float));
		}
		return 0;
	}

	int32_t *frames_in = jd->samples_in.ptrw();
	jack_nframes_t max_frames_in = jd->samples_in.size() / channels;

	jack_nframes_t frame_index = 0;

	while (frame_index < total_frames) {
		// the buffer size can change dynamically, and grow bigger than
		// our initial allocation. process the data in segments.

		jack_nframes_t current_frames = total_frames - frame_index;

		if (current_frames > max_frames_in)
			current_frames = max_frames_in;

		jd->audio_server_process(current_frames, frames_in);

		// deinterleave what we obtained, convert to +-1 floats
		for (unsigned ch = 0; ch < channels; ++ch) {
			float *ch_out = (float *)jack_port_get_buffer(ports[ch], total_frames) + frame_index;
			for (jack_nframes_t i = 0; i < current_frames; ++i) {
				int32_t sample = frames_in[ch + i * channels];
				ch_out[i] = (sample >> 16) * (1.0f / (1 << 16));
			}
		}

		frame_index += current_frames;
	}

	if (mutex)
		mutex->unlock();

	return 0;
}

void AudioDriverJACK::connect_physical_ports() {

	const DeviceJACK &jdev = devices[device_index];
	unsigned channels = jdev.channels();

	// we need the effective name of our client
	const char *src_client = jack_get_client_name(client);
	if (!src_client)
		return;

	// list physical output ports
	const char **ports = jack_get_ports(
			client, NULL, JACK_DEFAULT_AUDIO_TYPE,
			JackPortIsInput | JackPortIsPhysical);

	if (!ports)
		return;

	// first port belongs to our wanted client
	const char *first_port = ports[0];
	if (!first_port) {
		jack_free(ports);
		return;
	}

	// port name must be "client:port"
	const char *dst_client_start = first_port;
	const char *dst_client_end = strchr(dst_client_start, ':');
	size_t dst_client_length = dst_client_end - dst_client_start;
	if (!dst_client_end) {
		jack_free(ports);
		return;
	}

	// allocate some space to construct the source port name
	Vector<char> src_port_buffer;
	src_port_buffer.resize(strlen(src_client) + 32);

	// connect up to `channels` ports, as long as it belongs to the client
	for (unsigned ch = 0; ch < channels; ++ch) {
		const char *dst_port = ports[ch];
		if (!dst_port)
			break;

		if (strncmp(dst_port, dst_client_start, dst_client_length))
			break;

		char *src_port = src_port_buffer.ptrw();
		sprintf(src_port, "%s:" PORT_NAME_FORMAT, src_client, ch + 1);

		jack_connect(client, src_port, dst_port);
	}

	jack_free(ports);
}

AudioDriverJACK::AudioDriverJACK() :
		mutex(NULL),
		client(NULL),
		device_index(0),
		active(false) {
}

AudioDriverJACK::~AudioDriverJACK() {
}

#endif
